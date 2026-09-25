"""Pool-wide anchor regression tests (2026-09-25 owner report).

"One model failing after 90s is really a bug -- we have pools of over 75
models across different providers." Before this file's fix, two structural
gaps meant a single bad model could still end a task:

1. The persistent retry loop (`while True` after the candidate pass) was
   gated on `nim_anchor` -- NVIDIA being configured. Without NIM, ONE pass
   over the other providers ended the task even though Ollama Cloud /
   OpenRouter Free pools held dozens of healthy sibling models.
2. Even with NIM, every OTHER provider contributed exactly ONE model per
   pass (`_select_public_group_model` re-rolled a single pick), so a
   provider with six configured models got six identical chances, not six
   different models.

The contract pinned here, NIM absent:
- every configured provider is physically attempted, repeatedly, until the
  retry budget (TAMFIS_CODE_NIM_RETRY_SECONDS, default 600s) expires;
- sibling models within a provider actually rotate (the retry loop reaches
  the provider's NEXT model, not a re-roll of the same one);
- the task never reports failure while any candidate model is untried;
- with NIM present, the original NIM-anchored behaviour is unchanged
  (covered by tests/test_provider_failover.py NimAnchoredFailoverTests).

Also: MAX_TRUNCATION_CONTINUATIONS honours its env override so low-token
models can continue across as many rounds as the operator budgets.
"""
from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from tamfis_code import state as state_module
from tamfis_code.providers import ProviderManager, ProviderType
from tamfis_code.runner_local import (
    MAX_TRUNCATION_CONTINUATIONS,
    _anchor_provider_available,
    _anchor_retry_model,
    _nim_anchor_available,
    _nim_retry_model,
    _StreamedToolCall,
    run_local_agent_turn,
)


class _Config:
    def __init__(
        self, model: str, *, models: list[str] | None = None, name: str = "",
        base_url: str = "https://example.invalid/v1",
        api_key_env: str = "TAMFIS_CODE_TEST_KEY",
    ):
        self.name = name
        self.default_model = model
        self.models = list(models) if models is not None else [model]
        self.free_model = model
        self.tool_calling = True
        self.long_context = True
        self.context_window = 32768
        self.vision_supported = False
        self.vision_models = []
        self.base_url = base_url
        self.api_key_env = api_key_env


class _PoolManager(ProviderManager):
    """No NVIDIA client at all. Ollama Cloud and HF hold the routes.

    `failures` maps (provider, model) -> number of times that route raises
    before succeeding, so tests can assert that the loop ADVANCED to the
    sibling model instead of re-rolling the same dead one.
    """

    def __init__(self, failures: dict | None = None):
        self.clients = {
            ProviderType.OLLAMA_CLOUD: object(),
            ProviderType.HF: object(),
            ProviderType.OPENROUTER: object(),
        }
        self.PROVIDERS = {
            ProviderType.OLLAMA_CLOUD: _Config(
                "kimi-k2.7-code:cloud",
                models=["kimi-k2.7-code:cloud", "glm-5.3:cloud"],
                name="Ollama Cloud", base_url="https://ollama.com/v1",
            ),
            ProviderType.HF: _Config(
                "Qwen/Qwen3.6-35B-A3B",
                models=["Qwen/Qwen3.6-35B-A3B", "deepseek-ai/DeepSeek-V4.1-Flash"],
                name="Hugging Face",
            ),
            ProviderType.OPENROUTER: _Config(
                "z-ai/glm-4.5-air", name="OpenRouter",
            ),
        }
        self.failures = dict(failures or {})
        self.attempts: list[tuple[str, str]] = []
        self._evidence_done = False

    def _get_api_key(self, provider_type):
        # A constant, NOT an env lookup: conftest strips every
        # TAMFIS_CODE_* variable, so a key sourced from api_key_env would
        # vanish and fallback_candidates would silently drop the provider.
        return "test-key-for-anchor" if provider_type in self.clients else None

    def _check_ollama_available(self) -> bool:
        return True

    def route_is_healthy(self, provider, model="", *, ignore_pacing=False) -> bool:
        # Keep the test on selection/rotation semantics; cooldown behavior
        # has its own routing tests and otherwise masks which model was picked.
        return True

    def _stream_error_for(self, provider: ProviderType, model: str):
        # Call sites disagree on model spelling: some paths hand the raw
        # configured id ("...:cloud"), some the endpoint-normalised form.
        # A failure count is a property of the ROUTE, so look up both.
        for key in (
            (provider.value, model),
            (provider.value, model.removesuffix(":cloud")),
        ):
            remaining = self.failures.get(key, 0)
            if remaining > 0:
                self.failures[key] = remaining - 1
                self.failures[provider.value + "|any"] = self.failures.get(
                    provider.value + "|any", 0
                )
                error = Exception("HTTP 503 temporarily unavailable")
                error.same_route_reconnectable = False
                return error
        return None


def _wire_stream(manager: _PoolManager):
    """Route runner streaming through the per-(provider, model) failure map."""

    async def fake_stream(client, *, model, renderer=None, **kwargs):
        provider = next(
            (p for p, c in manager.clients.items() if c is client), None
        )
        manager.attempts.append(
            (provider.value if provider else "?", model)
        )
        error = (
            manager._stream_error_for(provider, model)
            if provider is not None
            else None
        )
        if error is not None:
            raise error
        # Audit-turn evidence contract (see test_provider_failover.py's
        # _wire_stream): a real read-only tool call before the final prose.
        if kwargs.get("tools") and not manager._evidence_done:
            manager._evidence_done = True
            return "", [
                _StreamedToolCall(
                    call_id="evidence_pool_anchor",
                    name="list_directory",
                    arguments=json.dumps({"path": "/tmp"}),
                )
            ], "tool_calls"
        return "the audit continues", [], "stop"

    return patch("tamfis_code.runner_local._stream_one_completion", side_effect=fake_stream)


class _StateHarness(unittest.TestCase):
    def setUp(self):
        self._originals = (
            state_module.CONFIG_DIR, state_module.STATE_PATH, state_module._LOCK_PATH,
        )
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        state_module.CONFIG_DIR = base / ".config"
        state_module.STATE_PATH = base / ".config" / "state.json"
        state_module._LOCK_PATH = base / ".config" / ".state.lock"
        state_module._STATE_CACHE = None
        from tamfis_code import providers as providers_module

        self.providers = providers_module
        providers_module._ROUTE_HEALTH.clear()
        delay = patch("tamfis_code.runner_local._nim_retry_delay", return_value=0.0)
        delay.start()
        self.addCleanup(delay.stop)
        title_stub = patch("tamfis_code.state.upgrade_session_title_with_ai", new=AsyncMock())
        title_stub.start()
        self.addCleanup(title_stub.stop)

    def tearDown(self):
        self.providers._ROUTE_HEALTH.clear()
        (state_module.CONFIG_DIR, state_module.STATE_PATH, state_module._LOCK_PATH) = (
            self._originals
        )
        state_module._STATE_CACHE = None
        self._tmp.cleanup()

    def _run(self, manager, session_id, *, provider=ProviderType.HF):
        """Start the turn on HF (the manager's first configurable provider).
        Ollama Cloud is deliberately NOT the turn's provider: the real
        manager pins its model via TAMFIS_CODE_OLLAMA_CODING_MODEL, so it
        cannot demonstrate sibling-model rotation in a fixture."""
        from io import StringIO

        from rich.console import Console
        from test_reasoning_plan import _RecordingRenderer

        renderer = _RecordingRenderer()
        state_module.save_session_state(session_id, workspace_root="/tmp")
        with _wire_stream(manager):
            outcome = asyncio.run(run_local_agent_turn(
                manager, provider, None,
                [{"role": "user", "content": "continue the audit"}],
                Console(file=StringIO(), no_color=True, width=200), renderer,
                workspace_root="/tmp", session_id=session_id, approval_policy="auto",
                interactive=False,
            ))
        return outcome, renderer

    def _failures(self, renderer):
        return [
            str(event.get("payload", {}).get("error", ""))
            for event in renderer.events
            if event.get("event_type") == "ai_task_failed"
        ]


class AnchorHelperTests(unittest.TestCase):
    def test_anchor_helpers_generalize_nim_helpers(self):
        config = SimpleNamespace(default_model="a", models=["a", "b", "c"])
        # _anchor_retry_model IS _nim_retry_model's engine: same rotation.
        self.assertEqual(
            [_anchor_retry_model(config, n) for n in (1, 2, 3, 4, 5)],
            ["a", "b", "c", "a", "b"],
        )
        self.assertEqual(_nim_retry_model(config, 2), "b")
        self.assertIsNone(_anchor_retry_model(SimpleNamespace(default_model="", models=[]), 1))

    def test_anchor_provider_available_mirrors_nim_check(self):
        class _Manager:
            PROVIDERS = {ProviderType.HF: object()}

            def get_client(self, provider):
                # Mirror the real manager: a client exists per configured
                # provider, None for an unconfigured one.
                return object() if provider in self.PROVIDERS else None

        manager = _Manager()
        self.assertTrue(_anchor_provider_available(manager, ProviderType.HF))
        self.assertFalse(_anchor_provider_available(manager, ProviderType.OPENROUTER))
        # The original NIM helper keeps its meaning through the shared engine.
        self.assertFalse(_nim_anchor_available(manager))  # NIM not configured

        class _NimManager(_Manager):
            PROVIDERS = {ProviderType.NVIDIA: object()}

        self.assertTrue(_nim_anchor_available(_NimManager()))


class PoolWideAnchorTests(_StateHarness):
    # This machine's real Ollama Cloud budget is often spent (the pacing gate
    # then excludes ollama_cloud from EVERY candidate list and the scenarios
    # below would silently exercise a different pool). The budget env override
    # keeps the fixture deterministic; pacing itself is covered by
    # tests/test_ollama_pacing.py.
    def setUp(self):
        super().setUp()
        pacing = patch.dict(
            "os.environ", {"TAMFIS_CODE_OLLAMA_WEEKLY_BUDGET": "0"}
        )
        pacing.start()
        self.addCleanup(pacing.stop)

    def test_without_nim_a_failing_model_rotates_to_the_next_sibling_model(self):
        """The 90s-stall incident: a provider's default model hangs. HF's
        default is seeded to fail through planning, the main stream, AND the
        first anchor pass; openrouter fails its pass too. The anchor's next
        pass must return to HF and rotation must advance past the failed
        qwen to its sibling deepseek -- not a re-roll of the dead model."""
        manager = _PoolManager(
            failures={
                # planning + main stream + first anchor pass on hf qwen.
                ("hf", "Qwen/Qwen3.6-35B-A3B"): 3,
                # planning hop + anchor pass on ollama's selected model
                # (selection is env-pinned, so whichever spelling arrives).
                ("ollama_cloud", "kimi-k2.7-code"): 2,
                ("ollama_cloud", "kimi-k2.7-code:cloud"): 2,
                # openrouter's own pass over the fallback sweep.
                ("openrouter", "z-ai/glm-4.5-air"): 2,
            }
        )
        outcome, renderer = self._run(manager, 81)

        self.assertEqual(self._failures(renderer), [])
        self.assertIn(("hf", "deepseek-ai/DeepSeek-V4.1-Flash"), manager.attempts)
        self.assertEqual(getattr(outcome, "status", ""), "completed")

    def test_without_nim_the_loop_keeps_rotating_across_the_pool(self):
        """Both HF models fail across planning/main/first-pass; only the
        anchor's later passes keep the task alive until deepseek's failure
        count runs out. Under the pre-fix code, NIM's absence meant the
        first exhausted pass ended the task."""
        manager = _PoolManager(
            failures={
                ("hf", "Qwen/Qwen3.6-35B-A3B"): 4,
                ("hf", "deepseek-ai/DeepSeek-V4.1-Flash"): 1,
                ("ollama_cloud", "kimi-k2.7-code"): 2,
                ("ollama_cloud", "kimi-k2.7-code:cloud"): 2,
                ("openrouter", "z-ai/glm-4.5-air"): 2,
            }
        )
        outcome, renderer = self._run(manager, 82)

        self.assertEqual(self._failures(renderer), [])
        providers_tried = {provider for provider, _model in manager.attempts}
        self.assertIn("openrouter", providers_tried)
        self.assertIn(("hf", "deepseek-ai/DeepSeek-V4.1-Flash"), manager.attempts)
        self.assertEqual(getattr(outcome, "status", ""), "completed")

    def test_exhausting_every_pool_model_reports_a_checkpoint_not_a_silent_stop(self):
        """Worst case: every model in the pool fails permanently. The task
        still ends the safe way -- an interrupted-with-checkpoint message --
        after having attempted the pool repeatedly, never a bare failure
        while untried candidates remain."""
        manager = _PoolManager(
            failures={
                ("hf", "Qwen/Qwen3.6-35B-A3B"): 10 ** 6,
                ("hf", "deepseek-ai/DeepSeek-V4.1-Flash"): 10 ** 6,
                # Ollama selection is env-pinned to its coding default
                # (glm-5.3 here); key both spellings regardless.
                ("ollama_cloud", "kimi-k2.7-code"): 10 ** 6,
                ("ollama_cloud", "kimi-k2.7-code:cloud"): 10 ** 6,
                ("ollama_cloud", "glm-5.3"): 10 ** 6,
                ("openrouter", "z-ai/glm-4.5-air"): 10 ** 6,
            }
        )
        with patch.dict("os.environ", {"TAMFIS_CODE_NIM_RETRY_SECONDS": "0.5"}):
            outcome, renderer = self._run(manager, 83)

        failures = self._failures(renderer)
        self.assertTrue(failures)
        self.assertIn("checkpointed", failures[0])
        # The loop really did keep going past a single pass.
        self.assertGreater(len(manager.attempts), 5)


class TruncationContinuationBudgetTests(unittest.TestCase):
    def test_env_override_raises_the_continuation_cap(self):
        with patch.dict("os.environ", {"TAMFIS_CODE_TRUNCATION_CONTINUATIONS": "18"}):
            import importlib
            from tamfis_code import runner_local

            importlib.reload(runner_local)
            try:
                self.assertEqual(runner_local.MAX_TRUNCATION_CONTINUATIONS, 18)
            finally:
                importlib.reload(runner_local)  # restore the module default

    def test_invalid_override_keeps_the_default(self):
        with patch.dict("os.environ", {"TAMFIS_CODE_TRUNCATION_CONTINUATIONS": "not-a-number"}):
            import importlib
            from tamfis_code import runner_local

            importlib.reload(runner_local)
            try:
                self.assertEqual(runner_local.MAX_TRUNCATION_CONTINUATIONS, 6)
            finally:
                importlib.reload(runner_local)

    def test_default_cap_unchanged_for_existing_suites(self):
        self.assertEqual(MAX_TRUNCATION_CONTINUATIONS, 6)


if __name__ == "__main__":
    unittest.main()
