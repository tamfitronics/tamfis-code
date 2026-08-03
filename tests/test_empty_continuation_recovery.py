import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from tamfis_code.providers import ProviderType
from tamfis_code.runner_local import (
    EMPTY_CONTINUATION_INSTRUCTION,
    _empty_continuation_messages,
    _normalise_tool_result,
    _recover_empty_continuation,
    _same_provider_recovery_models,
    _update_unresolved_edit_paths,
)


def test_empty_continuation_retry_is_a_fresh_user_message():
    original = [{"role": "tool", "tool_call_id": "call-1", "content": "failed"}]

    first = _empty_continuation_messages(original, 1)
    second = _empty_continuation_messages(original, 2)

    assert first[:-1] == original
    assert first[-1]["role"] == "user"
    assert EMPTY_CONTINUATION_INSTRUCTION in first[-1]["content"]
    assert first[-1]["content"] != second[-1]["content"]
    assert "attempt 2" in second[-1]["content"]


def test_same_provider_recovery_uses_safe_alternate_without_k3(monkeypatch):
    monkeypatch.delenv("TAMFIS_CODE_OLLAMA_EXTRA_USAGE", raising=False)
    config = SimpleNamespace(
        default_model="gemma4:cloud",
        models=[
            "gemma4:cloud",
            "kimi-k3:cloud",
            "kimi-k2.7-code:cloud",
            "minimax-m2.7:cloud",
        ],
    )

    assert _same_provider_recovery_models(config, "kimi-k2.7-code:cloud") == [
        "gemma4:cloud",
        "minimax-m2.7:cloud",
    ]


def test_same_provider_recovery_can_use_k3_only_with_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("TAMFIS_CODE_OLLAMA_EXTRA_USAGE", "true")
    config = SimpleNamespace(
        default_model="gemma4:cloud",
        models=["gemma4:cloud", "kimi-k3:cloud", "kimi-k2.7-code:cloud"],
    )

    assert _same_provider_recovery_models(config, "kimi-k2.7-code:cloud") == [
        "gemma4:cloud",
        "kimi-k3:cloud",
    ]


def test_empty_kimi_continuation_recovers_on_same_ollama_provider(monkeypatch):
    monkeypatch.delenv("TAMFIS_CODE_OLLAMA_EXTRA_USAGE", raising=False)
    config = SimpleNamespace(
        default_model="gemma4:cloud",
        models=["gemma4:cloud", "kimi-k2.7-code:cloud"],
    )
    renderer = SimpleNamespace(events=[])
    renderer.handle_event = renderer.events.append
    stream_completion = AsyncMock(
        side_effect=[
            ("", [], "stop"),
            ("Recovered.", [], "stop"),
        ]
    )
    nonstream_completion = AsyncMock(return_value=("", []))

    with patch(
        "tamfis_code.runner_local._stream_one_completion", stream_completion
    ), patch(
        "tamfis_code.runner_local._nonstream_one_completion",
        nonstream_completion,
    ):
        result = asyncio.run(
            _recover_empty_continuation(
                SimpleNamespace(),
                requested_provider=ProviderType.OLLAMA_CLOUD,
                resolved_provider=ProviderType.OLLAMA_CLOUD,
                config=config,
                client=object(),
                model="kimi-k2.7-code:cloud",
                messages=[
                    {
                        "role": "tool",
                        "tool_call_id": "call-1",
                        "content": "read failed",
                    }
                ],
                tools=[],
                renderer=renderer,
                task_profile=SimpleNamespace(),
            )
        )

    content, calls, provider, recovered_config, _client, model = result
    assert content == "Recovered."
    assert calls == []
    assert provider == ProviderType.OLLAMA_CLOUD
    assert recovered_config is config
    assert model == "gemma4:cloud"
    assert stream_completion.await_args_list[-1].kwargs["model"] == "gemma4:cloud"


def test_glyph_prefixed_edit_error_is_not_treated_as_a_success():
    result = _normalise_tool_result(
        "edit_file",
        {"path": "/workspace/app.ts"},
        {
            "success": True,
            "result": (
                "❌ Error: old_string not found in '/workspace/app.ts' "
                "-- no changes made"
            ),
        },
        "/workspace",
    )

    assert result["success"] is False
    assert "old_string not found" in result["error"]


def test_successful_retry_clears_prior_stale_edit_failure(tmp_path):
    workspace = tmp_path / "workspace"
    target = workspace / "api" / "router.ts"
    target.parent.mkdir(parents=True)
    target.write_text("current source", encoding="utf-8")
    unresolved: set[str] = set()

    failed_key = _update_unresolved_edit_paths(
        unresolved,
        path="api/router.ts",
        workspace_root=str(workspace),
        failed=True,
    )
    assert unresolved == {str(target.resolve())}

    recovered_key = _update_unresolved_edit_paths(
        unresolved,
        path=str(target),
        workspace_root=str(workspace),
        failed=False,
    )
    assert recovered_key == failed_key
    assert unresolved == set()
