"""Provider-specific stream normalization into canonical internal events."""
from __future__ import annotations

import json
import re
from typing import Any

_SINGLE_TOOL_CALL_ERROR_MARKERS = (
    "only supports single tool-calls",
    "only supports single tool calls",
    "single tool-call at once",
    "single tool call at once",
)


TOOL_CALL_PLACEHOLDER = "[tool call]"
_PLACEHOLDER_ECHO_RE = re.compile(r"^\s*\[?\s*tool[ _-]?calls?\s*\]?\s*$", re.IGNORECASE)


def is_tool_call_placeholder(text: object) -> bool:
    """True when ``text`` is only the request-side placeholder for an assistant turn that made tool calls
    without saying anything. Models see it in their history and echo it back as if it were a reply, which
    rendered as an empty "Assistant" panel reading "[tool call]" (owner report 2026-09-21)."""
    return isinstance(text, str) and bool(_PLACEHOLDER_ECHO_RE.match(text))


def normalize_tool_call(
    raw_name: Any,
    raw_arguments: Any = "",
    *,
    allowed_names: set[str] | None = None,
) -> tuple[str, str]:
    """Recover a registered tool from provider-injected channel markup.

    Some reasoning endpoints leak their internal message protocol into the
    function name, for example ``search_code<|Channel|>Commentary({...})``.
    Treating that string as a tool name reaches MCP as an unknown *dangerous*
    tool and can incorrectly open an approval prompt. Only names offered in
    this request are accepted; the surrounding channel text is discarded and
    an embedded JSON object is recovered when the arguments field is empty.
    """
    raw = str(raw_name or "").strip()
    if isinstance(raw_arguments, dict):
        arguments = json.dumps(raw_arguments, separators=(",", ":"))
    else:
        arguments = str(raw_arguments or "")
    names = {str(name).strip() for name in (allowed_names or set()) if str(name).strip()}
    if not raw:
        return "", arguments

    matched = ""
    lowered = raw.casefold()
    # Reasoning models sometimes narrate a correction in the header, e.g.
    # ``write_file? Actually read_file``. Prefer the LAST boundary-safe
    # registered name: it is the model's final protocol intent, while the
    # earlier name is usually discarded internal scratch text.
    matches: list[tuple[int, int, str]] = []
    for candidate in names:
        candidate_lower = candidate.casefold()
        search_from = 0
        while True:
            start = lowered.find(candidate_lower, search_from)
            if start < 0:
                break
            before = raw[start - 1] if start else ""
            end = start + len(candidate)
            after = raw[end:end + 1]
            if (not before or not (before.isalnum() or before == "_")) and (
                not after or not (after.isalnum() or after == "_")
            ):
                matches.append((start, len(candidate), candidate))
            search_from = start + 1
    if matches:
        matched = max(matches, key=lambda item: (item[0], item[1]))[2]

    if not matched:
        # Even without an allow-list, strip the well-known internal channel
        # suffix so it cannot become part of the executable tool identifier.
        # The runner still rejects the resulting name unless it was offered.
        marker_position = raw.find("<|")
        if allowed_names is None and marker_position > 0:
            prefix = raw[:marker_position].strip()
            if re.fullmatch(r"[A-Za-z_][\w.-]*", prefix):
                matched = prefix
        if not matched:
            return raw, arguments

    # A malformed provider may put the JSON call arguments in the name after
    # a channel/recipient suffix. Prefer the real arguments field when it is
    # already a JSON object; otherwise recover the balanced parenthesized
    # object from the contaminated name.
    if not arguments.strip() or arguments.strip() in {"{}", "null"}:
        marker = raw.find("(", raw.casefold().find(matched.casefold()) + len(matched))
        if marker >= 0:
            candidate_arguments = raw[marker + 1:].strip()
            if candidate_arguments.endswith(")"):
                candidate_arguments = candidate_arguments[:-1].rstrip()
            try:
                if isinstance(json.loads(candidate_arguments), dict):
                    arguments = candidate_arguments
            except (TypeError, ValueError):
                pass
    return matched, arguments


def provider_requires_single_tool_call(error: Any) -> bool:
    """Recognize the narrow provider validation error for tool batches."""
    text = str(error or "").lower()
    return any(marker in text for marker in _SINGLE_TOOL_CALL_ERROR_MARKERS)


def single_tool_call_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split assistant tool batches into provider-compatible one-call turns.

    Tamfis-Code may execute independent tools concurrently, but not every
    OpenAI-compatible model accepts the resulting multi-call assistant
    history. This request-only transformation preserves each call and its
    matching result without mutating the durable transcript.
    """
    normalized = system_messages_first(messages)
    result: list[dict[str, Any]] = []
    index = 0
    while index < len(normalized):
        message = normalized[index]
        calls = message.get("tool_calls") if isinstance(message, dict) else None
        if message.get("role") != "assistant" or not isinstance(calls, list) or len(calls) <= 1:
            result.append(message)
            index += 1
            continue

        results_by_id: dict[str, list[dict[str, Any]]] = {}
        cursor = index + 1
        while cursor < len(normalized) and normalized[cursor].get("role") == "tool":
            tool_message = normalized[cursor]
            results_by_id.setdefault(str(tool_message.get("tool_call_id") or ""), []).append(tool_message)
            cursor += 1
        matched: set[int] = set()
        for call in calls:
            if not isinstance(call, dict):
                continue
            call_id = str(call.get("id") or "")
            result.append({**message, "tool_calls": [call]})
            for tool_message in results_by_id.get(call_id, []):
                result.append(tool_message)
                matched.add(id(tool_message))
        for tool_message in normalized[index + 1:cursor]:
            if id(tool_message) not in matched:
                result.append(tool_message)
        index = cursor
    return result


def system_messages_first(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return provider-safe chat ordering without disturbing the transcript.

    Runtime repair, reconnect, and resume instructions are deliberately
    represented as trusted system messages, but they are often created after
    tool/assistant messages already exist -- a long session can easily
    accumulate several (the leading identity/format prompt, a scope rule, a
    resume instruction, and any number of NARRATED_TOOL_CORRECTION/
    CAPITULATION_CORRECTION/PORT_CONFLICT_CORRECTION-style mid-conversation
    nudges appended over the turn).

    FIX (2026-09-05, live-confirmed): a simple stable-partition-by-role
    (move every system message to the front, keep their relative order) was
    not enough -- a user still hit "System message must be at the
    beginning" on the very next provider in the fallback chain, meaning
    that backend's real constraint is stricter than "ordered first": having
    several distinct system messages, even correctly clustered at the very
    front, was rejected too. Collapsing them into exactly one combined
    system message satisfies both readings of the constraint. Content is
    joined with blank lines in original order, each coerced to plain text
    (a system message with list/multi-part content -- a different shape
    some backends also reject under an identically-worded error -- is
    flattened rather than merged as-is).
    """
    def _as_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    parts.append(str(block.get("text", block.get("content", ""))))
                else:
                    parts.append(str(block))
            return "\n".join(part for part in parts if part)
        return str(content) if content is not None else ""

    system_texts = [
        _as_text(message.get("content"))
        for message in messages
        if message.get("role") == "system"
    ]

    def _provider_safe_message(message: dict[str, Any]) -> dict[str, Any] | None:
        """Repair malformed historical assistant/tool messages at dispatch.

        An interrupted stream can checkpoint an assistant tool call before
        any visible text exists. OpenAI-compatible providers disagree about
        whether ``content: null`` is legal; TamfisGPT's local endpoint
        rejects both null and empty assistant content. Repair the request
        copy only, leaving the durable transcript unchanged.
        """
        if message.get("role") != "assistant":
            return message

        content = message.get("content")
        has_content = (
            bool(content.strip()) if isinstance(content, str)
            else bool(content)
        )
        if not has_content:
            if message.get("tool_calls"):
                message = {**message, "content": TOOL_CALL_PLACEHOLDER}
            else:
                return None

        if not message.get("tool_calls"):
            return message

        # An interrupted streamed tool call can be checkpointed after its
        # name/id arrive but before its JSON argument string closes. Sending
        # that truncated history verbatim makes every provider reject the
        # conversation before it can continue.
        repaired_calls: list[Any] = []
        changed = False
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                repaired_calls.append(call)
                continue
            function = call.get("function")
            if not isinstance(function, dict):
                repaired_calls.append(call)
                continue
            arguments = function.get("arguments", "")
            normalized_name, normalized_arguments = normalize_tool_call(
                function.get("name", ""), arguments,
            )
            if normalized_name != function.get("name", "") or normalized_arguments != arguments:
                function = {
                    **function,
                    "name": normalized_name,
                    "arguments": normalized_arguments,
                }
                arguments = normalized_arguments
                changed = True
            valid = False
            if isinstance(arguments, str):
                try:
                    valid = isinstance(json.loads(arguments), dict)
                except (TypeError, ValueError):
                    valid = False
                if valid:
                    repaired_calls.append({
                        **call,
                        "function": {**function, "name": normalized_name, "arguments": arguments},
                    } if changed or not isinstance(function.get("arguments"), str) else call)
                    continue
            elif isinstance(arguments, dict):
                arguments = json.dumps(arguments, separators=(",", ":"))
                valid = True
            if valid:
                repaired_calls.append({
                    **call,
                    "function": {**function, "arguments": arguments},
                })
                changed = True
                continue
            repaired_calls.append({
                **call,
                "function": {
                    **function,
                    "arguments": json.dumps({
                        "_tamfis_code_recovered": (
                            "malformed historical tool arguments omitted"
                        ),
                    }),
                },
            })
            changed = True
        return {**message, "tool_calls": repaired_calls} if changed else message

    # A provider failover can receive a checkpoint assembled between the
    # assistant tool-call message and its result.  It is also possible for
    # context compaction to retain the result while eliding the assistant
    # message.  OpenAI-compatible APIs reject such a request with errors like
    # ``tool call id ... not found in previous tool calls``.  Do not replay an
    # orphaned result: it is not executable conversation state and replaying
    # it cannot add evidence.  This is request-only sanitisation; the durable
    # transcript remains available for audit/resume.
    repaired_remainder = [
        repaired
        for message in messages
        if message.get("role") != "system"
        for repaired in [_provider_safe_message(message)]
        if repaired is not None
    ]
    remainder: list[dict[str, Any]] = []
    pending_tool_call_ids: set[str] = set()
    for message in repaired_remainder:
        role = message.get("role")
        if role == "assistant":
            calls = message.get("tool_calls") or []
            pending_tool_call_ids = {
                str(call.get("id"))
                for call in calls
                if isinstance(call, dict) and call.get("id")
            }
            remainder.append(message)
            continue
        if role == "tool":
            call_id = str(message.get("tool_call_id") or "")
            if call_id not in pending_tool_call_ids:
                # Never send a tool result without its matching assistant
                # call.  The model can issue a fresh call on the next round.
                continue
            pending_tool_call_ids.discard(call_id)
            remainder.append(message)
            continue
        pending_tool_call_ids.clear()
        remainder.append(message)
    combined_text = "\n\n".join(text for text in system_texts if text.strip())
    if not combined_text:
        return remainder
    return [{"role": "system", "content": combined_text}, *remainder]


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
            tool_name, tool_arguments = normalize_tool_call(
                _get(fn, "name", "") or "",
                _get(fn, "arguments", "") or "",
            )
            events.append(CanonicalEvent(EventType.TOOL_CALL_DELTA, {
                "index": int(_get(tool, "index", 0) or 0),
                "id": _get(tool, "id", "") or "",
                "name": tool_name,
                "arguments": tool_arguments,
            }, provider, model))
        if _get(choice, "finish_reason"):
            events.append(CanonicalEvent(EventType.DONE, {"reason": _get(choice, "finish_reason")}, provider, model))
    return events
