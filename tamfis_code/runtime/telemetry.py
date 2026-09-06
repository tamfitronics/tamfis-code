"""Bounded, local-only trace telemetry for execution economics.

Records intentionally contain operational identifiers and durations only:
never prompts, responses, source paths, tool arguments, credentials, or
provider error text.  Telemetry is best-effort and must not affect execution.
"""
from __future__ import annotations

import contextvars
import json
import os
import stat
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from secrets import token_hex
from typing import Any, Iterator, Mapping

from tamfis_code.config import CONFIG_DIR

TELEMETRY_PATH = CONFIG_DIR / "ai-inference-events.jsonl"
# Bounded local retention. Operators may size this for >=90 days of their CLI
# volume; unlike journald, the file remains a durable, structured data source.
MAX_TELEMETRY_BYTES = int(os.getenv("TAMFIS_CODE_TELEMETRY_MAX_BYTES", str(128 * 1024 * 1024)))
KEEP_TELEMETRY_BYTES = int(os.getenv("TAMFIS_CODE_TELEMETRY_KEEP_BYTES", str(96 * 1024 * 1024)))
_WRITE_LOCK = threading.Lock()
_TRACE_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "tamfis_trace_id", default=None,
)
_SPAN_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "tamfis_span_id", default=None,
)
_PROVIDER: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "tamfis_provider", default=None,
)
_USAGE: contextvars.ContextVar[dict[str, int] | None] = contextvars.ContextVar("tamfis_usage", default=None)


def current_trace_id() -> str | None:
    """Return the active trace ID, if this execution is being observed."""
    return _TRACE_ID.get()


@contextmanager
def provider_context(provider: str) -> Iterator[None]:
    """Associate lower-level compatible-client calls with their provider."""
    token = _PROVIDER.set(_safe_attribute(provider))
    try:
        yield
    finally:
        _PROVIDER.reset(token)


def current_provider() -> str | None:
    return _PROVIDER.get()


def record_usage(*, input_tokens: Any = None, output_tokens: Any = None,
                 reasoning_tokens: Any = None, cached_input_tokens: Any = None) -> None:
    """Attach provider-exposed counts to the active invocation span."""
    usage = _USAGE.get()
    if usage is None:
        return
    for key, value in (("input_tokens", input_tokens), ("output_tokens", output_tokens),
                       ("reasoning_tokens", reasoning_tokens), ("cached_input_tokens", cached_input_tokens)):
        if value is not None:
            try: usage[key] = int(value)
            except (TypeError, ValueError): pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_attribute(value: Any) -> str:
    """Return a bounded identifier-like value, never arbitrary payload text."""
    text = str(value or "")
    return "".join(
        character if character.isalnum() or character in "._:/-"
        else "_"
        for character in text
    )[:160]


def _safe_attributes(attributes: Mapping[str, Any]) -> dict[str, str]:
    # The names are deliberately closed. In particular, callers cannot
    # accidentally attach messages, arguments, paths, prompts, or output.
    allowed = {
        "mode", "provider", "model", "tool_name", "outcome",
        "error_class", "attempt", "operation",
    }
    return {
        key: _safe_attribute(value)
        for key, value in attributes.items()
        if key in allowed and value not in (None, "")
    }


def _rotate_if_needed() -> None:
    if not TELEMETRY_PATH.exists() or TELEMETRY_PATH.stat().st_size <= MAX_TELEMETRY_BYTES:
        return
    payload = TELEMETRY_PATH.read_bytes()[-KEEP_TELEMETRY_BYTES:]
    newline = payload.find(b"\n")
    if newline >= 0:
        payload = payload[newline + 1:]
    replacement = TELEMETRY_PATH.with_name(".trace-spans.rotate")
    try:
        with replacement.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(replacement, stat.S_IRUSR | stat.S_IWUSR)
        os.replace(replacement, TELEMETRY_PATH)
    finally:
        try:
            replacement.unlink()
        except FileNotFoundError:
            pass


def _append(record: Mapping[str, Any]) -> None:
    """Persist one completion record. Any storage failure is intentionally ignored."""
    try:
        with _WRITE_LOCK:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            _rotate_if_needed()
            line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            descriptor = os.open(
                TELEMETRY_PATH, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600,
            )
            try:
                os.write(descriptor, line.encode("utf-8"))
            finally:
                os.close(descriptor)
    except Exception:
        return


@contextmanager
def trace(name: str, **attributes: Any) -> Iterator[str]:
    """Create a root trace, nesting safely when an adapter already has one."""
    existing = _TRACE_ID.get()
    token = None
    if existing is None:
        existing = token_hex(16)
        token = _TRACE_ID.set(existing)
    try:
        with span(name, **attributes):
            yield existing
    finally:
        if token is not None:
            _TRACE_ID.reset(token)


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[str]:
    """Record one operation as a child of the current trace, fail-open."""
    trace_id = _TRACE_ID.get()
    if trace_id is None:
        # Direct library use still produces a coherent trace, without asking
        # every caller to know whether it is invoked from the CLI.
        with trace("cli.task", operation="library"):
            with span(name, **attributes) as span_id:
                yield span_id
        return

    span_id = token_hex(8)
    parent_span_id = _SPAN_ID.get()
    span_token = _SPAN_ID.set(span_id)
    usage: dict[str, int] = {}
    usage_token = _USAGE.set(usage)
    started_at = _now()
    started = time.monotonic()
    status = "ok"
    error_class = ""
    try:
        yield span_id
    except BaseException as exc:
        status = "cancelled" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "error"
        error_class = type(exc).__name__
        raise
    finally:
        _USAGE.reset(usage_token)
        _SPAN_ID.reset(span_token)
        attrs = _safe_attributes(attributes)
        if error_class:
            attrs["error_class"] = _safe_attribute(error_class)
        provider = attrs.get("provider") or current_provider()
        model = attrs.get("model")
        is_provider = name == "provider.invoke"
        ended_at = _now()
        _append({
            "event_type": "invocation" if is_provider else "operation",
            "trace_id": trace_id,
            "span_id": span_id,
            "parent_span_id": parent_span_id or "",
            "source_system": "tamfis-code",
            "tier": "tamfis-code",
            "component": "main_completion" if is_provider else _safe_attribute(name),
            "provider": provider,
            "canonical_model": model,
            "provider_native_model": model,
            "success": status == "ok",
            "error_class": attrs.get("error_class") or None,
            "streaming": attrs.get("operation") == "stream",
            "timestamp_start": started_at,
            "timestamp_first_token": None,
            "timestamp_end": ended_at,
            "latency_ms": int((time.monotonic() - started) * 1000),
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "reasoning_tokens": usage.get("reasoning_tokens"),
            "cached_input_tokens": usage.get("cached_input_tokens"),
            "is_retry": int(attrs.get("attempt") or "1") > 1,
            "is_fallback": False,
            "actual_provider_cost": None,
            "calculated_cost": None,
            "currency": "USD",
            "cost_confidence": "unknown",
            "attributes": attrs,
        })


def read_spans(limit: int = 100) -> list[dict[str, Any]]:
    """Read bounded recent telemetry for local, read-only diagnostics."""
    if limit <= 0 or not TELEMETRY_PATH.is_file():
        return []
    try:
        lines = TELEMETRY_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    result: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            result.append(record)
    return result
