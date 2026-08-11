import json
from io import StringIO

from tamfis_code.public_identity import (
    PUBLIC_MODEL_AUTO,
    PUBLIC_MODEL_PRO,
    PUBLIC_MODEL_SMART,
    PUBLIC_MODEL_ULTRA,
    PUBLIC_MODEL_ULTIMA,
    parse_public_model_alias,
    public_model_name,
    resolve_public_model_alias,
    sanitize_public_event,
)
from tamfis_code.render import StructuredRenderer


PRIVATE_MARKERS = ("ollama", "hugging", "openrouter", "nvidia", "kimi", "qwen")


def test_public_aliases_parse_in_short_and_branded_forms():
    assert parse_public_model_alias("auto") == PUBLIC_MODEL_AUTO
    assert parse_public_model_alias("TamfisGPT-fast") == PUBLIC_MODEL_SMART  # legacy alias
    assert parse_public_model_alias("PRO") == PUBLIC_MODEL_PRO
    assert parse_public_model_alias("ultra") == PUBLIC_MODEL_ULTRA
    assert parse_public_model_alias("ultima") == PUBLIC_MODEL_ULTIMA


def test_tier_is_derived_from_capability_registry_not_hardcoded_per_model():
    # frontier/high ("the super models") -> Ultima; frontier/medium -> Ultra;
    # quality "high" cost "medium" -> Pro. None of this is a hand-set field
    # on the model -- see model_registry.py's quality_tier/cost_tier.
    assert public_model_name("Qwen/Qwen3-Coder-480B-A35B-Instruct") == PUBLIC_MODEL_ULTIMA
    assert public_model_name("qwen/qwen3-coder") == PUBLIC_MODEL_ULTRA
    assert public_model_name("google/gemini-2.5-flash") == PUBLIC_MODEL_PRO


def test_public_alias_resolves_to_private_catalog_id_only_at_request_edge():
    selected = resolve_public_model_alias(
        "TamfisGPT Ultra",
        models=("google/gemini-2.5-flash", "qwen/qwen3-coder"),
        default_model="google/gemini-2.5-flash",
        free_model="google/gemini-2.5-flash",
    )
    assert selected == "qwen/qwen3-coder"


def test_subscription_tier_ids_round_trip_without_collapsing_to_auto():
    expected = {
        "tamfis-gpt-smart": PUBLIC_MODEL_SMART,
        "tamfis-gpt-pro": PUBLIC_MODEL_PRO,
        "tamfis-gpt-ultra": PUBLIC_MODEL_ULTRA,
        "tamfis-gpt-ultima": PUBLIC_MODEL_ULTIMA,
    }
    for model_id, public_name in expected.items():
        assert public_model_name(model_id) == public_name
        assert resolve_public_model_alias(
            public_name,
            models=tuple(expected),
            default_model="tamfis-gpt-auto",
        ) == model_id


def test_structured_routing_event_contains_only_tamfisgpt_identity():
    stream = StringIO()
    renderer = StructuredRenderer(mode="jsonl", stream=stream)
    renderer.handle_event({
        "event_type": "model_selected",
        "payload": {
            "provider": "ollama_cloud",
            "model": "qwen/qwen3-coder",
            "fallback_chain": ["nvidia", "hf", "openrouter"],
            "selection_reason": "Ollama selected qwen/qwen3-coder via HF",
        },
    })
    output = stream.getvalue()
    parsed = json.loads(output)
    assert parsed["payload"]["provider"] == "TamfisGPT"
    assert parsed["payload"]["model"] == PUBLIC_MODEL_ULTRA
    assert parsed["payload"]["fallback_chain"] == ["TamfisGPT"] * 3
    assert not any(marker in output.lower() for marker in PRIVATE_MARKERS)


def test_top_level_legacy_event_shape_is_also_sanitized():
    event = sanitize_public_event({"type": "model_selected", "provider": "nvidia", "model": "kimi-k2"})
    assert event["provider"] == "TamfisGPT"
    assert event["model"] == PUBLIC_MODEL_ULTRA
