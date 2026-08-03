"""Stable, product-owned names for user-facing model and routing surfaces.

Backend providers and catalog model ids are deployment details.  They may be
used internally for routing, but must not become part of TamfisGPT Code's
public CLI contract (including debug and machine-readable output).
"""
from __future__ import annotations

import re
from typing import Any


PUBLIC_PROVIDER_NAME = "TamfisGPT"
PUBLIC_MODEL_AUTO = "TamfisGPT Auto"
PUBLIC_MODEL_CODE = "TamfisGPT Code"
PUBLIC_MODEL_FAST = "TamfisGPT Fast"
PUBLIC_MODEL_PRO = "TamfisGPT Pro"
PUBLIC_MODEL_VISION = "TamfisGPT Vision"

PUBLIC_MODEL_ALIASES = (
    PUBLIC_MODEL_AUTO,
    PUBLIC_MODEL_FAST,
    PUBLIC_MODEL_CODE,
    PUBLIC_MODEL_PRO,
    PUBLIC_MODEL_VISION,
)

_PUBLIC_MODEL_INPUTS = {
    "auto": PUBLIC_MODEL_AUTO,
    "fast": PUBLIC_MODEL_FAST,
    "code": PUBLIC_MODEL_CODE,
    "pro": PUBLIC_MODEL_PRO,
    "vision": PUBLIC_MODEL_VISION,
}

_PROVIDER_RE = re.compile(
    r"(?i)\b(?:ollama(?:[ _-]cloud|[ _-]gpu)?|hugging\s*face|hf|"
    r"openrouter|nvidia(?:\s*nim)?|nvidia_nim|tier[ _-]?iv|apiframe|"
    r"moonshot(?:\s*ai)?|anthropic|openai|google(?:\s*genai)?|x-ai|xai)\b"
)
_MODEL_HINT_RE = re.compile(
    r"(?i)(?:kimi|qwen|deepseek|nemotron|gemma|glm|minimax|gemini|llama|"
    r"mistral|mixtral|claude|gpt|coder|grok|"
    r":cloud|/.*(?:flash|pro|instruct|reasoning))"
)


def public_model_name(model: Any = None) -> str:
    """Return a stable TamfisGPT alias without revealing a catalog id."""
    value = str(model or "").strip()
    lowered = value.lower()
    if not value or lowered in {"auto", "default", "(provider default)"}:
        return PUBLIC_MODEL_AUTO
    parsed = parse_public_model_alias(value)
    if parsed:
        return parsed
    if lowered.startswith("tamfisgpt "):
        return PUBLIC_MODEL_CODE
    if any(token in lowered for token in ("vision", "omni", "gemini", "gemma")):
        return PUBLIC_MODEL_VISION
    if any(token in lowered for token in ("flash", "nano", "mini", ":free")):
        return PUBLIC_MODEL_FAST
    if any(token in lowered for token in ("pro", "ultra", "k3", "480b", "550b")):
        return PUBLIC_MODEL_PRO
    return PUBLIC_MODEL_CODE


def parse_public_model_alias(value: Any) -> str | None:
    """Return the canonical public alias when *value* names one."""
    normalized = re.sub(r"[ _-]+", " ", str(value or "").strip().lower())
    if normalized.startswith("tamfisgpt "):
        normalized = normalized.removeprefix("tamfisgpt ")
    return _PUBLIC_MODEL_INPUTS.get(normalized)


def resolve_public_model_alias(
    value: Any,
    *,
    models: Any = (),
    default_model: Any = None,
    free_model: Any = None,
) -> str | None:
    """Resolve a public alias to a private catalog id at the request edge.

    Raw ids remain accepted for backwards compatibility but are never
    advertised. ``None`` means automatic model selection.
    """
    alias = parse_public_model_alias(value)
    if alias is None:
        return str(value).strip() if value is not None else None
    if alias == PUBLIC_MODEL_AUTO:
        return None
    candidates = [free_model, *(models or ()), default_model]
    for candidate in candidates:
        if candidate and public_model_name(candidate) == alias:
            return str(candidate)
    return str(default_model or free_model or "").strip() or None


def public_route_name(provider: Any = None, model: Any = None) -> str:
    """Render a route exclusively in TamfisGPT product vocabulary."""
    if str(provider or "").strip().lower() == "auto" and not model:
        return PUBLIC_MODEL_AUTO
    return public_model_name(model)


def redact_routing_text(value: Any) -> str:
    """Redact backend names from internal status/error text before display.

    This is intentionally used only for runtime-owned metadata, never for
    assistant answers or tool/file output, where provider words may be the
    user's actual subject matter.
    """
    text = str(value or "")
    text = _PROVIDER_RE.sub(PUBLIC_PROVIDER_NAME, text)
    # Common raw catalog ids contain a namespace, a model-family hint, or a
    # cloud suffix.  Replace token-sized occurrences while retaining the rest
    # of an actionable error message.
    tokens = re.split(r"(\s+)", text)
    for index, token in enumerate(tokens):
        clean = token.strip("'\"`()[]{}.,;:")
        if clean and _MODEL_HINT_RE.search(clean) and not clean.startswith("TamfisGPT"):
            tokens[index] = token.replace(clean, public_model_name(clean))
    return "".join(tokens)


def sanitize_public_event(event: dict[str, Any]) -> dict[str, Any]:
    """Copy and brand routing metadata in a structured CLI event."""
    event_type = str(event.get("event_type") or event.get("event") or event.get("type") or "")
    result = dict(event)
    payload = dict(result.get("payload") or {})
    for container in (result, payload):
        raw_model = container.get("model")
        for key in ("provider", "requested_provider", "resolved_provider"):
            if key in container:
                container[key] = PUBLIC_PROVIDER_NAME
        if "model" in container:
            container["model"] = public_model_name(raw_model)
        if "models" in container:
            container["models"] = [public_model_name(item) for item in container.get("models") or []]
        for key in ("requested_model", "resolved_model", "selected_model"):
            if key in container:
                container[key] = public_model_name(container[key])
        for key in ("fallback_chain", "providers", "routes"):
            if key in container:
                container[key] = [PUBLIC_PROVIDER_NAME for _item in container.get(key) or []]
    if event_type.startswith("orchestrator_") or event_type in {
        "diagnostics", "task_diagnostics", "model_selected", "routing_started",
        "provider_request_started", "provider_unavailable", "model_unavailable",
        "ai_task_failed", "outcome",
    }:
        for container in (result, payload):
            for key in ("content", "message", "error", "reason", "selection_reason"):
                if key in container:
                    container[key] = redact_routing_text(container[key])
    result["payload"] = payload
    return result
