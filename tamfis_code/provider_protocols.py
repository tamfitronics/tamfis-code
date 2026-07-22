"""Provider-specific stream normalization into canonical internal events."""
from __future__ import annotations

from typing import Any


class ProviderStreamError(RuntimeError):
    """Structured error reported after a provider stream has already opened."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        provider: str | None = None,
        model: str | None = None,
        retryable: bool = True,
        error_type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.provider = provider
        self.model = model
        self.retryable = retryable
        self.error_type = error_type


def _embedded_stream_error(chunk: Any) -> tuple[str, int | None, str | None] | None:
    """Extract an error embedded in an otherwise successful streaming response."""

    error = _get(chunk, "error")
    kind = str(_get(chunk, "event_type") or _get(chunk, "type") or "").strip().lower()

    payload = _get(chunk, "payload", {}) or {}
    candidate = error if error not in (None, "", {}, []) else (payload if kind in {"error", "stream_error"} and payload else chunk)
    candidate_type = str(_get(candidate, "type") or _get(candidate, "code") or kind or "").strip()
    message = _get(candidate, "message") or _get(candidate, "detail") or _get(candidate, "error")

    if isinstance(message, dict):
        candidate_type = str(_get(message, "type") or _get(message, "code") or candidate_type).strip()
        message = _get(message, "message") or _get(message, "detail") or str(message)

    status = _get(candidate, "status_code") or _get(candidate, "status")
    code = _get(candidate, "code")
    if status is None and isinstance(code, int):
        status = code
    try:
        status_code = int(status) if status is not None else None
    except (TypeError, ValueError):
        status_code = None

    error_kinds = {
        "error",
        "stream_error",
        "internal_server_error",
        "service_unavailable",
        "resource_exhausted",
        "overloaded_error",
    }
    has_error_shape = error not in (None, "", {}, []) or kind in error_kinds
    if not has_error_shape:
        return None

    rendered = str(message or candidate_type or "Provider stream failed").strip()
    return rendered, status_code, candidate_type or None

from .orchestrator.protocols import CanonicalEvent, EventType


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def normalize_stream_chunk(chunk: Any, *, provider: str | None = None, model: str | None = None) -> list[CanonicalEvent]:
    """Normalize OpenAI, Ollama-native, Anthropic-style and Tier IV chunks.

    Unknown shapes yield no events rather than leaking provider-specific data
    into the renderer. Structured tool calls are accepted only from recognized
    protocol fields, never from JSON-looking assistant prose.
    """
    events: list[CanonicalEvent] = []

    embedded_error = _embedded_stream_error(chunk)
    if embedded_error is not None:
        message, status_code, error_type = embedded_error
        lowered = f"{error_type or ''} {message}".lower()
        retryable = (
            status_code in {408, 409, 425, 429}
            or (status_code is not None and status_code >= 500)
            or any(marker in lowered for marker in (
                "resourceexhausted",
                "resource_exhausted",
                "total request limit reached",
                "worker local",
                "worker capacity",
                "temporarily unavailable",
                "service unavailable",
                "overloaded",
                "rate limit",
                "timeout",
            ))
        )
        raise ProviderStreamError(
            message,
            status_code=status_code,
            provider=provider,
            model=model,
            retryable=retryable,
            error_type=error_type,
        )

    # Tier IV canonical envelope or already-normalized dictionary.
    kind = _get(chunk, "event_type") or _get(chunk, "event") or _get(chunk, "type")
    payload = _get(chunk, "payload", {}) or {}
    if kind in {item.value for item in EventType}:
        if not payload and isinstance(chunk, dict):
            payload = {
                key: value for key, value in chunk.items()
                if key not in {"event_type", "event", "type", "provider", "model", "created_at"}
            }
        events.append(CanonicalEvent(EventType(kind), dict(payload), provider, model))
        return events

    # Ollama native /api/chat JSON line.
    message = _get(chunk, "message")
    if message is not None and _get(chunk, "choices") is None:
        content = _get(message, "content", "") or ""
        if content:
            events.append(CanonicalEvent(EventType.ASSISTANT_DELTA, {"content": content}, provider, model))
        if _get(chunk, "done", False):
            events.append(CanonicalEvent(EventType.DONE, {"reason": _get(chunk, "done_reason")}, provider, model))
        return events

    # Anthropic Messages streaming event shapes.
    if kind == "content_block_delta":
        delta = _get(chunk, "delta", {})
        text = _get(delta, "text", "") or ""
        if text:
            events.append(CanonicalEvent(EventType.ASSISTANT_DELTA, {"content": text}, provider, model))
        partial = _get(delta, "partial_json", "") or ""
        if partial:
            events.append(CanonicalEvent(EventType.TOOL_CALL_DELTA, {"arguments": partial}, provider, model))
        return events
    if kind == "message_stop":
        return [CanonicalEvent(EventType.DONE, {}, provider, model)]

    # OpenAI-compatible chat-completions chunk.
    choices = _get(chunk, "choices", []) or []
    if choices:
        choice = choices[0]
        delta = _get(choice, "delta", {})
        # `reasoning_content` (some providers use `reasoning`) is a real,
        # separate pre-answer stream some OpenAI-compatible reasoning models
        # emit when `reasoning_effort` is set -- confirmed live against
        # NVIDIA NIM's nemotron-3-super. Not a declared OpenAI SDK field, so
        # it only ever appears via the delta object's own extra attributes
        # (`_get` handles both dict and attribute access).
        reasoning = _get(delta, "reasoning_content", "") or _get(delta, "reasoning", "") or ""
        if reasoning:
            events.append(CanonicalEvent(EventType.REASONING_DELTA, {"content": reasoning}, provider, model))
        content = _get(delta, "content", "") or ""
        if content:
            events.append(CanonicalEvent(EventType.ASSISTANT_DELTA, {"content": content}, provider, model))
        for tool in _get(delta, "tool_calls", []) or []:
            fn = _get(tool, "function", {})
            events.append(CanonicalEvent(EventType.TOOL_CALL_DELTA, {
                "index": int(_get(tool, "index", 0) or 0),
                "id": _get(tool, "id", "") or "",
                "name": _get(fn, "name", "") or "",
                "arguments": _get(fn, "arguments", "") or "",
            }, provider, model))
        if _get(choice, "finish_reason"):
            events.append(CanonicalEvent(EventType.DONE, {"reason": _get(choice, "finish_reason")}, provider, model))
    return events
