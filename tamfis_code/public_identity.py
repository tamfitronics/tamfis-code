"""Stable, product-owned names for user-facing model and routing surfaces.

Backend providers and catalog model ids are deployment details.  They may be
used internally for routing, but must not become part of TamfisGPT Code's
public CLI contract (including debug and machine-readable output).

Public model names are four capability tiers -- Smart, Pro, Ultra, Ultima --
that mirror the TamfisGPT subscription tiers, plus the "Auto" meta-selector.
Tier membership is derived from each model's live-verified capability/cost
attributes (`model_registry.py`'s `quality_tier`/`cost_tier`), never a
hand-set field on an individual model: see the "no human should be managing
individual models" principle this mirrors from the main TamfisGPT product's
tier-gating design. Ultima is the frontier+high-cost bracket ("the super
models") -- which subscription tiers can actually reach it is enforced
server-side (only entitled models come back from `/models`), so the CLI
never needs its own admin/plan flag to gate display.
"""
from __future__ import annotations

import re
from typing import Any

from .model_registry import MODELS as _MODEL_REGISTRY


PUBLIC_PROVIDER_NAME = "TamfisGPT"
PUBLIC_MODEL_AUTO = "TamfisGPT Auto"
PUBLIC_MODEL_SMART = "TamfisGPT Smart"
PUBLIC_MODEL_PRO = "TamfisGPT Pro"
PUBLIC_MODEL_ULTRA = "TamfisGPT Ultra"
PUBLIC_MODEL_ULTIMA = "TamfisGPT Ultima"

# FIX 2026-08-11: the TAMFIS provider's real per-tier catalog ids (see
# providers.py's ProviderConfig.models for ProviderType.TAMFIS) -- added
# so public_model_name() can match them directly instead of falling through
# to the coarse keyword heuristic at the bottom of that function, which
# had no rule that would ever produce PUBLIC_MODEL_SMART/PRO/ULTRA/ULTIMA
# for these exact strings.
_TAMFIS_ID_TO_TIER = {
    "tamfis-gpt-smart": PUBLIC_MODEL_SMART,
    "tamfis-gpt-pro": PUBLIC_MODEL_PRO,
    "tamfis-gpt-ultra": PUBLIC_MODEL_ULTRA,
    "tamfis-gpt-ultima": PUBLIC_MODEL_ULTIMA,
}

PUBLIC_MODEL_ALIASES = (
    PUBLIC_MODEL_AUTO,
    PUBLIC_MODEL_SMART,
    PUBLIC_MODEL_PRO,
    PUBLIC_MODEL_ULTRA,
    PUBLIC_MODEL_ULTIMA,
)

# Tiers a model can be gated behind, ordered lowest -> highest. Kept distinct
# from PUBLIC_MODEL_ALIASES (which also contains the non-tier "Auto" entry).
PUBLIC_MODEL_TIERS = (
    PUBLIC_MODEL_SMART,
    PUBLIC_MODEL_PRO,
    PUBLIC_MODEL_ULTRA,
    PUBLIC_MODEL_ULTIMA,
)

_PUBLIC_MODEL_INPUTS = {
    "auto": PUBLIC_MODEL_AUTO,
    "smart": PUBLIC_MODEL_SMART,
    "pro": PUBLIC_MODEL_PRO,
    "ultra": PUBLIC_MODEL_ULTRA,
    "ultima": PUBLIC_MODEL_ULTIMA,
    # Back-compat for the old five-alias scheme (Auto/Fast/Code/Pro/Vision):
    # callers/scripts/history that still type the old words land on the
    # closest new tier rather than erroring.
    "fast": PUBLIC_MODEL_SMART,
    "code": PUBLIC_MODEL_PRO,
    "vision": PUBLIC_MODEL_ULTRA,
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

# (quality_tier, cost_tier) -> public tier, straight from model_registry.py's
# capability/cost fields. This is the single source of truth for grouping;
# nothing here is a per-model hand-set field.
_QUALITY_COST_TO_TIER = {
    ("frontier", "high"): PUBLIC_MODEL_ULTIMA,
    ("frontier", "medium"): PUBLIC_MODEL_ULTRA,
    ("frontier", "low"): PUBLIC_MODEL_ULTRA,
    ("high", "high"): PUBLIC_MODEL_ULTRA,
    ("high", "medium"): PUBLIC_MODEL_PRO,
    ("high", "low"): PUBLIC_MODEL_PRO,
    ("balanced", "medium"): PUBLIC_MODEL_PRO,
    ("balanced", "low"): PUBLIC_MODEL_SMART,
    ("balanced", "high"): PUBLIC_MODEL_PRO,
}


def _tier_from_registry(catalog_id: str) -> str | None:
    record = _MODEL_REGISTRY.get(catalog_id)
    if record is None:
        # Registry keys are case-sensitive catalog ids (e.g. HF's
        # "moonshotai/Kimi-K2.6" vs NVIDIA's lowercase alias); fall back to a
        # case-insensitive lookup before giving up.
        lowered = catalog_id.lower()
        record = next(
            (item for key, item in _MODEL_REGISTRY.items() if key.lower() == lowered),
            None,
        )
    if record is None:
        return None
    return _QUALITY_COST_TO_TIER.get(
        (record.quality_tier, record.cost_tier), PUBLIC_MODEL_PRO,
    )


def public_model_name(model: Any = None) -> str:
    """Return a stable TamfisGPT tier name without revealing a catalog id."""
    value = str(model or "").strip()
    lowered = value.lower()
    if not value or lowered in {"auto", "default", "(provider default)"}:
        return PUBLIC_MODEL_AUTO
    parsed = parse_public_model_alias(value)
    if parsed:
        return parsed
    by_tamfis_id = _TAMFIS_ID_TO_TIER.get(lowered)
    if by_tamfis_id:
        return by_tamfis_id
    by_capability = _tier_from_registry(value)
    if by_capability:
        return by_capability
    if lowered.startswith("tamfisgpt "):
        return PUBLIC_MODEL_PRO
    # Unknown/raw id not in the local capability registry (e.g. a Remote
    # catalog id this client build doesn't know about yet): fall back to a
    # coarse keyword heuristic rather than silently defaulting everything
    # into one bucket.
    if any(token in lowered for token in ("480b", "550b", "ultra", ":thinking")):
        return PUBLIC_MODEL_ULTIMA
    if any(token in lowered for token in ("flash", "nano", "mini", ":free")):
        return PUBLIC_MODEL_SMART
    if any(token in lowered for token in ("pro", "k3", "reasoning", "vision", "omni", "gemini", "gemma")):
        return PUBLIC_MODEL_PRO
    return PUBLIC_MODEL_ULTRA


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
    # Model-hint pass runs FIRST, on whole tokens, before the provider-name
    # pass below. Raw catalog ids are frequently a single "namespace/model"
    # token (e.g. "nvidia/nemotron-3-super", "moonshotai/Kimi-K2.6"). Running
    # the provider pass first left a "TamfisGPT/nemotron-3-super"-shaped
    # token whose model half then survived: the old "not
    # clean.startswith('TamfisGPT')" guard -- meant to avoid re-redacting an
    # already-public token -- also skipped these partially-redacted
    # compound tokens, so the model name leaked. Redacting the whole token
    # to a single public alias here, before the provider substitution ever
    # runs, closes that gap.
    tokens = re.split(r"(\s+)", text)
    for index, token in enumerate(tokens):
        clean = token.strip("'\"`()[]{}.,;:")
        if clean and _MODEL_HINT_RE.search(clean) and not clean.startswith("TamfisGPT"):
            tokens[index] = token.replace(clean, public_model_name(clean))
    text = "".join(tokens)
    text = _PROVIDER_RE.sub(PUBLIC_PROVIDER_NAME, text)
    return text


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
