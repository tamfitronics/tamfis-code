import json
from io import StringIO

from tamfis_code.public_identity import (
    PUBLIC_MODEL_AUTO,
    PUBLIC_MODEL_CODE,
    PUBLIC_MODEL_FAST,
    PUBLIC_MODEL_PRO,
    parse_public_model_alias,
    resolve_public_model_alias,
    sanitize_public_event,
)
from tamfis_code.render import StructuredRenderer


PRIVATE_MARKERS = ("ollama", "hugging", "openrouter", "nvidia", "kimi", "qwen")


def test_public_aliases_parse_in_short_and_branded_forms():
    assert parse_public_model_alias("auto") == PUBLIC_MODEL_AUTO
    assert parse_public_model_alias("TamfisGPT-fast") == PUBLIC_MODEL_FAST
    assert parse_public_model_alias("PRO") == PUBLIC_MODEL_PRO


def test_public_alias_resolves_to_private_catalog_id_only_at_request_edge():
    selected = resolve_public_model_alias(
        "TamfisGPT Code",
        models=("vendor/flash-mini", "vendor/qwen-coder"),
        default_model="vendor/qwen-coder",
        free_model="vendor/flash-mini",
    )
    assert selected == "vendor/qwen-coder"


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
    assert parsed["payload"]["model"] == PUBLIC_MODEL_CODE
    assert parsed["payload"]["fallback_chain"] == ["TamfisGPT"] * 3
    assert not any(marker in output.lower() for marker in PRIVATE_MARKERS)


def test_top_level_legacy_event_shape_is_also_sanitized():
    event = sanitize_public_event({"type": "model_selected", "provider": "nvidia", "model": "kimi-k2"})
    assert event["provider"] == "TamfisGPT"
    assert event["model"] == PUBLIC_MODEL_CODE
