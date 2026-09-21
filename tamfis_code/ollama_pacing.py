"""Pace Ollama Cloud's weekly allowance instead of running it to zero.

Ollama Cloud's allowance is weekly and account-wide -- tamfis-code and TamfisGPT draw on the SAME
one. On 2026-09-21 it hit 100% ("resets in 25 minutes") after ~2,860 requests in a week, because
every route failure fell through to Ollama as the first fallback hop.

This module counts tamfis-code's own Ollama requests in hourly buckets and answers one question:
"has this machine spent its share for now?". When it has, ``ProviderManager.route_is_healthy``
reports Ollama unhealthy, so automatic routing uses NIM / free OpenRouter first and reaches Ollama
only when nothing else answers. An explicit choice (``--provider`` / ``/model``) is never blocked.

Ollama publishes a percentage, not a request quota, so the budget is an estimate:

    TAMFIS_CODE_OLLAMA_WEEKLY_BUDGET   requests per rolling 7 days (default 500; 0 disables)
    TAMFIS_CODE_OLLAMA_DAILY_PACE      multiple of budget/7 allowed in any 24h (default 1.5)

TamfisGPT keeps its own, larger share (TAMGPT_OLLAMA_WEEKLY_REQUEST_BUDGET); together they stay
under the ~2,860 requests that exhausted the account.

The check runs on the routing hot path, so it is a cached in-memory read (re-read from disk every
20 s); the file write happens once per real Ollama request.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Optional

from . import config as _config

_log = logging.getLogger("tamfis_code.ollama_pacing")
_CACHE_SECONDS = 20.0
_cache: tuple[float, int, int] = (0.0, 0, 0)
_paced_logged = False


def weekly_budget() -> int:
    try:
        return max(0, int(os.environ.get("TAMFIS_CODE_OLLAMA_WEEKLY_BUDGET", "500")))
    except ValueError:
        return 500


def daily_budget() -> int:
    weekly = weekly_budget()
    try:
        pace = max(1.0, float(os.environ.get("TAMFIS_CODE_OLLAMA_DAILY_PACE", "1.5")))
    except ValueError:
        pace = 1.5
    return int(weekly / 7.0 * pace) if weekly else 0


def _path():
    return _config.CONFIG_DIR / "ollama_usage.json"


def _bucket(ts: float) -> str:
    return time.strftime("%Y%m%d%H", time.gmtime(ts))


def _load() -> dict[str, int]:
    try:
        data = json.loads(_path().read_text(encoding="utf-8"))
        buckets = data.get("hours", {})
        return {str(k): int(v) for k, v in buckets.items()}
    except Exception:
        return {}


def record_request() -> None:
    """Count one real Ollama Cloud request (best effort; never raises)."""
    global _cache
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return  # test traffic must not eat the real allowance counter
    try:
        now = time.time()
        buckets = _load()
        key = _bucket(now)
        buckets[key] = buckets.get(key, 0) + 1
        keep = {_bucket(now - h * 3600) for h in range(170)}
        buckets = {k: v for k, v in buckets.items() if k in keep}
        path = _path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"hours": buckets}), encoding="utf-8")
        os.replace(tmp, path)
        _cache = (0.0, 0, 0)
    except Exception:
        pass


def usage() -> tuple[int, int]:
    """(requests in the last 24h, requests in the last 7 days)."""
    global _cache
    stamp, day, week = _cache
    if stamp and time.monotonic() - stamp < _CACHE_SECONDS:
        return day, week
    now = time.time()
    buckets = _load()
    counts = [buckets.get(_bucket(now - h * 3600), 0) for h in range(168)]
    day, week = sum(counts[:24]), sum(counts)
    _cache = (time.monotonic(), day, week)
    return day, week


def is_paced_out() -> bool:
    """True once this machine's weekly/daily Ollama budget is spent."""
    global _paced_logged
    weekly = weekly_budget()
    if weekly <= 0:
        return False
    day, week = usage()
    over = week >= weekly or day >= daily_budget()
    if over != _paced_logged:
        _paced_logged = over
        _log.warning(
            "ollama pacing: %s (last24h=%d/%d last7d=%d/%d)",
            "budget spent -- routing to NIM/OpenRouter first" if over else "back under budget",
            day, daily_budget(), week, weekly,
        )
    return over


# "resets in 25 minutes", "sessions resume in 2 hours", "resets in 3 days" ...
_RESET_RE = re.compile(
    r"(?:resets?|resumes?)\s+in\s+(?:about\s+|~)?(\d+(?:\.\d+)?)\s*"
    r"(seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h|days?|d)\b",
    re.IGNORECASE,
)
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_reset_delay(text: object) -> Optional[float]:
    """Seconds until the provider says the allowance resets, or None when it does not say."""
    match = _RESET_RE.search(str(text or ""))
    if not match:
        return None
    return float(match.group(1)) * _UNIT_SECONDS[match.group(2)[0].lower()]
