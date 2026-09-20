"""A small persistent memory of which models are fast, slow or dead.

Live measurement 2026-09-19 with tamfis-code's own prompt and 21 tools against NVIDIA
NIM: nemotron-3-super made its tool call in 1.0s, nemotron-3-ultra in 9.2s,
nemotron-3.5-lightning in 14s -- and kimi-k3 and glm-5.3, the two models first in the
configured order, never answered inside 60s. The first-byte timeout is 45s, so a run
that started on kimi-k3 spent 45s (or 90s, when the fallback then picked glm) waiting
on a model that was never going to answer, before reaching one that takes a second.

The health circuit that should have prevented that lives in process memory and lasts
30s, so every new run -- and nearly every run after a pause -- rediscovered the same
dead model. This module remembers across runs:

  * a model that timed out or stalled is skipped for a growing cool-off (5 min, then
    15, 45, 60), and comes back automatically -- one success clears it;
  * how long a model takes to its first useful output (an exponentially-weighted
    average), so a model 3x+ slower than an alternative is tried after it.

`rank` reorders a preference list by that memory. It never REMOVES a model (a penalised
one is only moved last), so the owner's preference order stays the default whenever
the preferred models are healthy and fast.

Stored as JSON in CONFIG_DIR, written atomically; every operation is best effort and
never raises -- bookkeeping must not be able to fail a task.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from . import config as _config

# How long a model that failed to answer is skipped, by consecutive-failure level.
PENALTY_SECONDS = (5 * 60, 15 * 60, 45 * 60, 60 * 60)
# Weight of the newest latency sample in the running average.
EWMA_ALPHA = 0.35
# A model measured slower than this to first useful output is "slow" outright...
SLOW_ABSOLUTE_SECONDS = 12.0
# ...and one this many times slower than the fastest measured healthy alternative
# (and not trivially quick) is demoted behind it.
SLOW_RELATIVE_FACTOR = 3.0
SLOW_RELATIVE_FLOOR_SECONDS = 3.0
# Measurements older than this are ignored: providers change.
STALE_AFTER_SECONDS = 24 * 3600
_MAX_MODELS = 200

_LOCK = threading.Lock()

# Off under pytest unless a test turns it on: the suite drives the real stream code
# with fake models, and must neither write the host's real memory of which models are
# slow nor have its model selection depend on it. TAMFIS_CODE_ROUTE_STATS=0 disables
# it everywhere.
ENABLE_UNDER_TEST = False


def enabled() -> bool:
    if os.environ.get("TAMFIS_CODE_ROUTE_STATS", "1").strip().lower() in {"0", "false", "off", "no"}:
        return False
    if os.environ.get("PYTEST_CURRENT_TEST") and not ENABLE_UNDER_TEST:
        return False
    return True


def _path() -> Path:
    return _config.CONFIG_DIR / "route_stats.json"


def _load() -> dict[str, dict[str, Any]]:
    if not enabled():
        return {}
    try:
        data = json.loads(_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    models = data.get("models") if isinstance(data, dict) else None
    return models if isinstance(models, dict) else {}


def _save(models: dict[str, dict[str, Any]]) -> None:
    if not enabled():
        return
    try:
        path = _path()
        path.parent.mkdir(parents=True, exist_ok=True)
        if len(models) > _MAX_MODELS:
            newest = sorted(models.items(), key=lambda item: item[1].get("updated", 0), reverse=True)
            models = dict(newest[:_MAX_MODELS])
        handle, temp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".route_stats.", suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump({"version": 1, "models": models}, stream, separators=(",", ":"))
            os.replace(temp_name, path)
        except BaseException:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise
    except Exception:
        return


def _now() -> float:
    return time.time()


def record_latency(model: str, seconds: float) -> None:
    """A successful round: `seconds` from sending the request to its first useful
    output (answer text or a tool call). Clears any penalty."""
    if not model or not isinstance(seconds, (int, float)) or seconds < 0:
        return
    try:
        with _LOCK:
            models = _load()
            entry = models.setdefault(model, {})
            previous = entry.get("ewma")
            fresh = (_now() - float(entry.get("updated", 0) or 0)) < STALE_AFTER_SECONDS
            entry["ewma"] = (
                round(EWMA_ALPHA * seconds + (1 - EWMA_ALPHA) * float(previous), 3)
                if isinstance(previous, (int, float)) and fresh else round(float(seconds), 3)
            )
            entry["samples"] = int(entry.get("samples", 0) or 0) + 1
            entry["penalty_level"] = 0
            entry["penalized_until"] = 0
            entry["updated"] = _now()
            _save(models)
    except Exception:
        return


def record_failure(model: str, kind: str = "timeout") -> None:
    """The model did not answer (timeout, stall, empty stream, 5xx). Skip it for a
    cool-off that grows with each consecutive failure."""
    if not model:
        return
    try:
        with _LOCK:
            models = _load()
            entry = models.setdefault(model, {})
            level = int(entry.get("penalty_level", 0) or 0)
            entry["penalized_until"] = _now() + PENALTY_SECONDS[min(level, len(PENALTY_SECONDS) - 1)]
            entry["penalty_level"] = min(level + 1, len(PENALTY_SECONDS))
            entry["last_failure"] = str(kind)[:24]
            entry["failures"] = int(entry.get("failures", 0) or 0) + 1
            entry["updated"] = _now()
            _save(models)
    except Exception:
        return


def clear(model: Optional[str] = None) -> None:
    """Forget one model (or everything)."""
    try:
        with _LOCK:
            models = _load()
            if model is None:
                models = {}
            else:
                models.pop(model, None)
            _save(models)
    except Exception:
        return


def is_penalized(model: str, *, now: Optional[float] = None) -> bool:
    entry = _load().get(model) or {}
    return float(entry.get("penalized_until", 0) or 0) > (now if now is not None else _now())


def expected_latency(model: str, *, now: Optional[float] = None) -> Optional[float]:
    entry = _load().get(model) or {}
    ewma = entry.get("ewma")
    if not isinstance(ewma, (int, float)):
        return None
    if (now if now is not None else _now()) - float(entry.get("updated", 0) or 0) > STALE_AFTER_SECONDS:
        return None
    return float(ewma)


def describe(models: Optional[Sequence[str]] = None, *, now: Optional[float] = None) -> list[dict[str, Any]]:
    """Rows for a human ("/routes"-style) view: one per remembered model."""
    stamp = now if now is not None else _now()
    stored = _load()
    names = list(models) if models is not None else sorted(stored)
    rows = []
    for name in names:
        entry = stored.get(name) or {}
        remaining = max(0.0, float(entry.get("penalized_until", 0) or 0) - stamp)
        rows.append({
            "model": name,
            "latency": expected_latency(name, now=stamp),
            "samples": int(entry.get("samples", 0) or 0),
            "failures": int(entry.get("failures", 0) or 0),
            "skipped_for": remaining,
        })
    return rows


def rank(models: Sequence[str], *, now: Optional[float] = None) -> list[str]:
    """`models` (in preference order) reordered by what has been measured.

    Healthy models that are unmeasured or fast keep their preference order and come
    first; measured-slow ones follow, fastest first; penalised ones come last, the
    one whose cool-off ends soonest first. Nothing is dropped."""
    ordered = [m for m in dict.fromkeys(models) if m]
    if len(ordered) < 2:
        return ordered
    stamp = now if now is not None else _now()
    stored = _load()
    info: dict[str, tuple[bool, Optional[float], float]] = {}
    for model in ordered:
        entry = stored.get(model) or {}
        until = float(entry.get("penalized_until", 0) or 0)
        info[model] = (until > stamp, expected_latency(model, now=stamp), until)
    fastest = min(
        (latency for penalized, latency, _ in info.values() if not penalized and latency is not None),
        default=None,
    )

    def bucket(model: str) -> int:
        penalized, latency, _ = info[model]
        if penalized:
            return 2
        if latency is None:
            return 0
        if latency > SLOW_ABSOLUTE_SECONDS:
            return 1
        if (
            fastest is not None and latency > SLOW_RELATIVE_FLOOR_SECONDS
            and latency > fastest * SLOW_RELATIVE_FACTOR
        ):
            return 1
        return 0

    def key(model: str) -> tuple:
        group = bucket(model)
        penalized, latency, until = info[model]
        position = ordered.index(model)
        if group == 0:
            return (0, position)
        if group == 1:
            return (1, latency if latency is not None else 1e9, position)
        return (2, until, position)

    return sorted(ordered, key=key)
