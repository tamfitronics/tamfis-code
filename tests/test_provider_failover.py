"""A provider that is out of credits must never end the task.

Live-reported 2026-09-19: a 35-minute TamfisGPT-Ultra run ended with

    Error: TamfisGPT-Ultra streaming failed: Error code: 402 - {'error': 'You
    have depleted your monthly included credits. Purchase pre-paid credits to
    continue using Inference Providers.'} ... type `continue` to resume

...with NVIDIA NIM configured and unused.

Two independent gaps made that stop possible, and both are pinned here:

1. The 402 arrived as a wrapper that preserved only the SDK's str(), so
   `provider_error_status` read None and no retryable marker matched the
   credit wording -> the error was classified NON-retryable, so the runner's
   fallback gate (`can_fallback`) was False and no other route was ever tried.
2. Even when an error IS retryable, `fallback_candidates` skips routes whose
   health circuit is open (recently failed). If every alternative is cooling,
   the candidate list came back empty and the task stopped anyway -- even
   though a cooling route is a latency heuristic, not a dead provider.

Second live report (same day): an Ollama Cloud 429 -- "you have reached your
weekly usage limit" -- STOPPED the task the same way, with four other
configured providers unused. The 429 was already classified retryable and
candidates were returned, so this file also pins the END-TO-END contract: the
fallback loop must physically attempt every alternative (primary 429 ->
NVIDIA/HF/OpenRouter/Grok tried in order) and only then surface an
interrupted-with-checkpoint message, never a plain stop while a route remains.
"""
from __future__ import annotations

import asyncio
import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from tamfis_code import state as state_module
from tamfis_code.providers import ProviderManager, ProviderType
from tamfis_code.runner_local import (
    _fallback_candidates_for_turn,
    _StreamedToolCall,
    run_local_agent_turn,
)

LIVE_402 = (
    "Error code: 402 - {'error': 'You have depleted your monthly included "
    "credits. Purchase pre-paid credits to continue using Inference Providers.'}"
)
LIVE_OLLAMA_429 = (
    "Error code: 429 - {'error': {'message': 'you (tamfitron) have reached "
    "your weekly usage limit, upgrade for higher limits: "
    "https://ollama.com/upgrade or add usage credits: "
    "https://ollama.com/settings (ref: 90a421d1-9e37-467b-be2e-fc3a6f4cd9af)', "
    "'type': 'api_error', 'param': None, 'code': None}}"
)
LIVE_TAMFIS_403 = (
    "Error code: 403 - {'code': 'permission-denied', 'error': 'Your team "
    "c7f035e2-b400-443c-8604-d93b9eb639f2 has either used all available "
    "credits or reached its monthly spending limit. To continue making API "
    "requests, please purchase more credits or raise your spending limit.'}"
)


class ExhaustedProviderIsRetryableTests(unittest.TestCase):
    def test_the_live_wrapped_402_is_classified_retryable(self):
        """The exact live message, with the SDK attributes lost."""
        exc = Exception(LIVE_402)
        self.assertEqual(ProviderManager.provider_error_status(exc), 402)
        self.assertTrue(ProviderManager.is_retryable_provider_error(exc))

    def test_the_live_tamfis_team_credit_403_is_retryable_and_exhaustion(self):
        """The exact live 403: the subscription team ran out of credits. The
        error is account-level -- every OTHER configured (free) provider must
        be tried, never a stop."""
        exc = Exception(LIVE_TAMFIS_403)
        self.assertEqual(ProviderManager.provider_error_status(exc), 403)
        self.assertTrue(ProviderManager.is_retryable_provider_error(exc))
        from tamfis_code.state import route_error_is_exhaustion

        self.assertTrue(route_error_is_exhaustion(LIVE_TAMFIS_403))

    def test_the_live_ollama_weekly_limit_429_is_retryable_and_exhaustion(self):
        """The exact live 429: account-level, retryable, exhaustion-class."""
        exc = Exception(LIVE_OLLAMA_429)
        self.assertEqual(ProviderManager.provider_error_status(exc), 429)
        self.assertTrue(ProviderManager.is_retryable_provider_error(exc))
        from tamfis_code.state import route_error_is_exhaustion

        self.assertTrue(route_error_is_exhaustion(LIVE_OLLAMA_429))

    def test_credit_wording_alone_is_enough(self):
        for message in (
            "You have depleted your monthly included credits.",
            "Insufficient credits to continue.",
            "Out of credits for this account.",
            "Please purchase pre-paid credits.",
            "Billing issue on this account.",
            "you have reached your weekly usage limit",
        ):
            with self.subTest(message=message):
                self.assertTrue(
                    ProviderManager.is_retryable_provider_error(Exception(message))
                )

    def test_an_http_code_in_the_text_restores_the_routing_decision(self):
        self.assertEqual(
            ProviderManager.provider_error_status(Exception("HTTP 503 Service Unavailable")),
            503,
        )
        self.assertEqual(
            ProviderManager.provider_error_status(Exception("status code: 429")),
            429,
        )

    def test_a_genuinely_non_retryable_error_stays_non_retryable(self):
        """The failover path must not swallow every error: an unrelated,
        unclassifiable failure still stops (that is what `continue` is for)."""
        self.assertFalse(
            ProviderManager.is_retryable_provider_error(Exception("something odd happened"))
        )


class _Config:
    def __init__(
        self, model: str, *, tool_calling: bool = True, name: str = "",
        base_url: str = "https://example.invalid/v1",
        api_key_env: str = "TAMFIS_CODE_TEST_KEY",
    ):
        self.name = name
        self.default_model = model
        self.models = [model]
        self.free_model = model
        self.tool_calling = tool_calling
        self.long_context = True
        self.context_window = 32768
        self.vision_supported = False
        self.vision_models = []
        self.base_url = base_url
        self.api_key_env = api_key_env


class _ExhaustionChainManager(ProviderManager):
    """Ollama Cloud (429) plus four alternatives, all streaming-able.

    Every provider is key-valid (`_get_api_key` override) so the fallback
    chain sees all five, the way the operator's machine is configured. The
    stream raises each provider's mapped failure; the recorder asserts the
    loop PHYSICALLY attempted every alternative in order.
    """

    def __init__(self):
        self.clients = {
            ProviderType.TAMFIS: object(),
            ProviderType.OLLAMA_CLOUD: object(),
            ProviderType.NVIDIA: object(),
            ProviderType.HF: object(),
            ProviderType.OPENROUTER: object(),
            ProviderType.GROK: object(),
        }
        self.PROVIDERS = {
            ProviderType.TAMFIS: _Config(
                "tamfis-gpt-ultima", name="TamfisGPT",
                base_url="https://gateway.internal/v1",
            ),
            ProviderType.OLLAMA_CLOUD: _Config(
                "gpt-oss:120b", name="Ollama Cloud",
                base_url="https://ollama.com/v1",
            ),
            ProviderType.NVIDIA: _Config("moonshotai/kimi-k3", name="NVIDIA NIM"),
            ProviderType.HF: _Config("meta-llama/Llama-3.3-70B", name="Hugging Face"),
            ProviderType.OPENROUTER: _Config("z-ai/glm-4.5-air", name="OpenRouter"),
            ProviderType.GROK: _Config("grok-3-mini", name="Grok"),
        }
        self.attempts: list[str] = []
        self._evidence_routes: set[ProviderType] = set()
        self._primary_error = Exception(LIVE_OLLAMA_429)

    def _stream_error_for(self, provider: ProviderType) -> Exception | None:
        if provider in {ProviderType.OLLAMA_CLOUD, ProviderType.TAMFIS}:
            return self._primary_error
        if provider in self.fail_everywhere:
            return Exception(f"HTTP 503 on {provider.value}")
        return None

    def _get_api_key(self, provider_type):
        return "test-key-for-failover" if provider_type in self.clients else None

    def _check_ollama_available(self) -> bool:
        return True

    def _check_tier_iv_available(self) -> bool:
        return True

    fail_everywhere: set = set()


def _wire_stream(manager: _ExhaustionChainManager):
    """Route runner streaming through the manager's per-provider failure map."""
    async def fake_stream(client, *, model, renderer=None, **kwargs):
        provider = next(
            (p for p, c in manager.clients.items() if c is client), None
        )
        manager.attempts.append(provider.value if provider else model)
        error = manager._stream_error_for(provider)
        if error is not None:
            # 429/402-class failures move DIRECTLY to cross-provider fallback
            # (see _same_route_reconnectable); emulate that decisiveness for
            # the 503s here too, so the test measures the fallback chain and
            # not 50s of same-route backoff.
            error.same_route_reconnectable = False
            raise error
        # Successful audit turns must contain real tool evidence.  Returning
        # prose only here used to bypass the intended provider-failover
        # protocol and made the fixture incompatible with the completion gate.
        if kwargs.get("tools") and provider not in manager._evidence_routes:
            manager._evidence_routes.add(provider)
            return "", [
                _StreamedToolCall(
                    call_id=f"evidence_{len(manager.attempts)}",
                    name="list_directory",
                    arguments=json.dumps({"path": "/tmp"}),
                )
            ], "tool_calls"
        return "the audit continues", [], "stop"

    return patch("tamfis_code.runner_local._stream_one_completion", side_effect=fake_stream)


class OllamaWeeklyLimitMustNotStopTheTaskTests(unittest.TestCase):
    """The live 429 shape, end to end through the real runner loop."""

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

        providers_module._ROUTE_HEALTH.clear()
        # These tests pin the "every alternative is physically tried" contract
        # with NIM ALSO failing. With the NIM retry loop on (its default) that
        # state would keep going back to NIM for minutes -- covered by
        # NimAnchoredFailoverTests below -- so it is switched off here.
        patcher = patch.dict("os.environ", {"TAMFIS_CODE_NIM_RETRY_SECONDS": "0"})
        patcher.start()
        self.addCleanup(patcher.stop)
        title_stub = patch("tamfis_code.state.upgrade_session_title_with_ai", new=AsyncMock())
        title_stub.start()
        self.addCleanup(title_stub.stop)

    def tearDown(self):
        from tamfis_code import providers as providers_module

        providers_module._ROUTE_HEALTH.clear()
        (state_module.CONFIG_DIR, state_module.STATE_PATH, state_module._LOCK_PATH) = (
            self._originals
        )
        state_module._STATE_CACHE = None
        self._tmp.cleanup()

    def _console(self):
        from io import StringIO
        from rich.console import Console

        return Console(file=StringIO(), no_color=True, width=200)

    def _renderer(self):
        from test_reasoning_plan import _RecordingRenderer

        return _RecordingRenderer()

    def _failure_events(self, renderer) -> list[str]:
        return [
            str(event.get("payload", {}).get("error", ""))
            for event in renderer.events
            if event.get("event_type") == "ai_task_failed"
        ]

    def test_a_403_team_credit_exhaustion_exhausts_alternatives_before_stopping(self):
        """The LIVE 2026-09-19 TamfisGPT-Ultra 403: the team ran out of
        credits and the task stopped. The contract: the fallback loop must
        physically attempt every other configured provider (the free ones:
        NVIDIA NIM, HF, OpenRouter, Grok) before any stop, and record the
        exhaustion so /status and the footer can explain it."""
        manager = _ExhaustionChainManager()
        manager._primary_error = Exception(LIVE_TAMFIS_403)
        manager.fail_everywhere = {
            ProviderType.NVIDIA, ProviderType.HF,
            ProviderType.OPENROUTER, ProviderType.GROK,
        }
        renderer = self._renderer()
        state_module.save_session_state(43, workspace_root="/tmp")
        state_module.record_route_event(43, provider="tamfis", model="ultima")

        with _wire_stream(manager):
            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.TAMFIS, None,
                [{"role": "user", "content": "continue the audit"}],
                self._console(), renderer,
                workspace_root="/tmp", session_id=43, approval_policy="auto",
                interactive=False,
            ))

        self.assertIn("tamfis", manager.attempts)
        for name in ("nvidia", "hf", "openrouter", "grok"):
            self.assertIn(name, manager.attempts, f"{name} was never tried")
        diag = state_module.route_diagnostics(43)
        self.assertEqual(diag["last_exhaustion"]["provider"], "tamfis")
        self.assertIn("403", diag["last_exhaustion"]["reason"])

    def test_a_403_with_one_working_free_alternative_never_fails_the_task(self):
        manager = _ExhaustionChainManager()
        manager._primary_error = Exception(LIVE_TAMFIS_403)
        # NVIDIA NIM (the first free alternative) is healthy: the task MUST
        # continue there, no user intervention.
        manager.fail_everywhere = set()
        renderer = self._renderer()
        state_module.save_session_state(44, workspace_root="/tmp")
        state_module.record_route_event(44, provider="tamfis", model="ultima")

        with _wire_stream(manager):
            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.TAMFIS, None,
                [{"role": "user", "content": "continue the audit"}],
                self._console(), renderer,
                workspace_root="/tmp", session_id=44, approval_policy="auto",
                interactive=False,
            ))

        failures = self._failure_events(renderer)
        self.assertEqual(failures, [], f"task must not fail: {failures}")
        self.assertIn("nvidia", manager.attempts)
        diag = state_module.route_diagnostics(44)
        self.assertEqual(diag["current"]["provider"], "nvidia")

    def test_a_429_on_the_primary_exhausts_alternatives_before_stopping(self):
        manager = _ExhaustionChainManager()
        manager.fail_everywhere = {
            ProviderType.NVIDIA, ProviderType.HF,
            ProviderType.OPENROUTER, ProviderType.GROK,
        }
        renderer = self._renderer()
        state_module.save_session_state(41, workspace_root="/tmp")
        state_module.record_route_event(41, provider="ollama_cloud", model="gpt-oss:120b")

        with _wire_stream(manager):
            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.OLLAMA_CLOUD, None,
                [{"role": "user", "content": "continue the audit"}],
                self._console(), renderer,
                workspace_root="/tmp", session_id=41, approval_policy="auto",
                interactive=False,
            ))

        # Every configured alternative was PHYSICALLY attempted, in order --
        # not just listed. This is the contract the live 429 violated.
        self.assertIn("ollama_cloud", manager.attempts)
        for name in ("nvidia", "hf", "openrouter", "grok"):
            self.assertIn(name, manager.attempts, f"{name} was never tried")
        # Nothing succeeded, so the task reports failure -- but the recorded
        # exhaustion says WHY, and the message is checkpoint-continue, meaning
        # the conversation (and the other providers' next turn) survives.
        failures = self._failure_events(renderer)
        self.assertTrue(failures)
        self.assertIn("checkpointed", failures[0])
        diag = state_module.route_diagnostics(41)
        self.assertEqual(diag["last_exhaustion"]["provider"], "ollama_cloud")
        self.assertIn("429", diag["last_exhaustion"]["reason"])

    def test_a_429_with_one_working_alternative_never_fails_the_task(self):
        manager = _ExhaustionChainManager()
        # NVIDIA (the first alternative) works: the turn must continue there.
        manager.fail_everywhere = set()
        renderer = self._renderer()
        state_module.save_session_state(42, workspace_root="/tmp")
        state_module.record_route_event(42, provider="ollama_cloud", model="gpt-oss:120b")

        with _wire_stream(manager):
            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.OLLAMA_CLOUD, None,
                [{"role": "user", "content": "continue the audit"}],
                self._console(), renderer,
                workspace_root="/tmp", session_id=42, approval_policy="auto",
                interactive=False,
            ))

        failures = self._failure_events(renderer)
        self.assertEqual(failures, [], f"task must not fail: {failures}")
        self.assertIn("nvidia", manager.attempts)
        diag = state_module.route_diagnostics(42)
        self.assertEqual(diag["current"]["provider"], "nvidia")
        self.assertEqual(diag["last_exhaustion"]["provider"], "ollama_cloud")



class _NimFlakyManager(_ExhaustionChainManager):
    """NIM answers 503/429 a set number of times and then 200; every other
    provider's behaviour is configurable. Used to pin the NIM-anchored policy."""

    def __init__(self, *, nim_failures: int = 2, nim_error: str = "HTTP 503 Service Unavailable"):
        super().__init__()
        self.nim_failures = nim_failures
        self.nim_error = nim_error
        self.other_errors: dict = {}

    def _stream_error_for(self, provider):
        if provider == ProviderType.NVIDIA:
            if self.nim_failures > 0 or self.nim_failures < 0:
                if self.nim_failures > 0:
                    self.nim_failures -= 1
                return Exception(self.nim_error)
            return None
        return self.other_errors.get(provider)


class _NimPolicyHarness(unittest.TestCase):
    """Owner ruling 2026-09-19: NIM first; Ollama Cloud / HF / OpenRouter / Grok
    only while they have credit; when they are erroring (429, 503, 403, 402) go
    back to NIM and keep trying until a 200 arrives -- never hand the user a raw
    429 / 503."""

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
        # No real sleeping between NIM attempts.
        delay = patch("tamfis_code.runner_local._nim_retry_delay", return_value=0.0)
        delay.start()
        self.addCleanup(delay.stop)
        # The end-of-turn LLM title call would otherwise hit live NIM from a test.
        title_stub = patch("tamfis_code.state.upgrade_session_title_with_ai", new=AsyncMock())
        title_stub.start()
        self.addCleanup(title_stub.stop)

    def tearDown(self):
        self.providers._ROUTE_HEALTH.clear()
        (state_module.CONFIG_DIR, state_module.STATE_PATH, state_module._LOCK_PATH) = self._originals
        state_module._STATE_CACHE = None
        self._tmp.cleanup()

    def _run(self, manager, session_id, *, provider=ProviderType.NVIDIA, model=None):
        from io import StringIO

        from rich.console import Console
        from test_reasoning_plan import _RecordingRenderer

        renderer = _RecordingRenderer()
        state_module.save_session_state(session_id, workspace_root="/tmp")
        with _wire_stream(manager):
            outcome = asyncio.run(run_local_agent_turn(
                manager, provider, model,
                [{"role": "user", "content": "continue the audit"}],
                Console(file=StringIO(), no_color=True, width=200), renderer,
                workspace_root="/tmp", session_id=session_id, approval_policy="auto",
                interactive=False,
            ))
        return outcome, renderer

    def _park_as_out_of_credit(self, *providers):
        """Record the live credit walls so these providers are parked."""
        for provider in providers:
            self.providers.record_route_failure_for(
                provider, manager_default_model(provider), Exception(LIVE_TAMFIS_403),
            )

    def _failures(self, renderer):
        return [
            str(event.get("payload", {}).get("error", ""))
            for event in renderer.events if event.get("event_type") == "ai_task_failed"
        ]



class NimAnchoredFailoverTests(_NimPolicyHarness):
    """NIM first; other providers only with credit; back to NIM until a 200."""

    def test_a_503_storm_on_nim_returns_to_nim_until_a_200(self):
        manager = _NimFlakyManager(nim_failures=3)
        self._park_as_out_of_credit(ProviderType.HF, ProviderType.OPENROUTER, ProviderType.GROK,
                                    ProviderType.OLLAMA_CLOUD)
        outcome, renderer = self._run(manager, 61)
        self.assertEqual(self._failures(renderer), [])
        self.assertEqual(manager.attempts, ["nvidia"] * 5)  # 3 x 503, read evidence, then final answer

    def test_out_of_credit_providers_are_never_tried_even_when_nim_is_failing(self):
        manager = _NimFlakyManager(nim_failures=2)
        self._park_as_out_of_credit(ProviderType.HF, ProviderType.OPENROUTER, ProviderType.GROK,
                                    ProviderType.OLLAMA_CLOUD)
        self._run(manager, 62)
        for parked in ("hf", "openrouter", "grok", "ollama_cloud"):
            self.assertNotIn(parked, manager.attempts)

    def test_a_provider_with_credit_is_used_when_nim_fails(self):
        manager = _NimFlakyManager(nim_failures=1)
        outcome, renderer = self._run(manager, 63)
        self.assertEqual(self._failures(renderer), [])
        self.assertEqual(manager.attempts[0], "nvidia")
        self.assertNotEqual(manager.attempts[1], "nvidia")  # a funded alternative answered

    def test_alternatives_erroring_with_429_and_503_send_the_task_back_to_nim(self):
        manager = _NimFlakyManager(nim_failures=1)
        for provider in (ProviderType.HF, ProviderType.OPENROUTER, ProviderType.GROK,
                         ProviderType.OLLAMA_CLOUD, ProviderType.TAMFIS):
            manager.other_errors[provider] = Exception(
                "HTTP 429 Too Many Requests" if provider == ProviderType.HF else "HTTP 503 unavailable"
            )
        outcome, renderer = self._run(manager, 64)
        self.assertEqual(self._failures(renderer), [])
        self.assertEqual(manager.attempts[-1], "nvidia")  # the task ended on NIM's 200

    def test_the_user_never_sees_a_raw_429_or_503_when_the_budget_runs_out(self):
        manager = _NimFlakyManager(nim_failures=-1, nim_error=LIVE_OLLAMA_429)  # NIM never recovers
        self._park_as_out_of_credit(ProviderType.HF, ProviderType.OPENROUTER, ProviderType.GROK,
                                    ProviderType.OLLAMA_CLOUD)
        with patch.dict("os.environ", {"TAMFIS_CODE_NIM_RETRY_SECONDS": "0.3"}):
            with patch("tamfis_code.runner_local._nim_retry_delay", return_value=0.05):
                outcome, renderer = self._run(manager, 65)
        failures = self._failures(renderer)
        self.assertTrue(failures)
        text = failures[0]
        self.assertIn("checkpointed", text)
        self.assertIn("busy", text)
        for leak in ("429", "503", "Error code", "ollama", "upgrade", "tamfitron"):
            self.assertNotIn(leak, text)
        self.assertGreater(manager.attempts.count("nvidia"), 2)  # it really did keep trying

    def test_a_deterministic_nim_failure_does_not_loop_for_ten_minutes(self):
        manager = _NimFlakyManager(nim_failures=-1, nim_error="Error code: 401 - invalid api key")
        self._park_as_out_of_credit(ProviderType.HF, ProviderType.OPENROUTER, ProviderType.GROK,
                                    ProviderType.OLLAMA_CLOUD)
        outcome, renderer = self._run(manager, 66)
        self.assertTrue(self._failures(renderer))
        self.assertLessEqual(manager.attempts.count("nvidia"), 6)

    def test_the_retry_keeps_going_when_every_nim_model_is_inside_its_cooldown(self):
        """With an explicit tier requested, normal selection skips models whose
        health circuit is open -- during a NIM-wide 503 storm that is ALL of
        them. The loop must not give up there: it rotates through the NIM
        models regardless, because a cooldown is only a heuristic."""
        manager = _NimFlakyManager(nim_failures=3)
        self._park_as_out_of_credit(ProviderType.HF, ProviderType.OPENROUTER, ProviderType.GROK,
                                    ProviderType.OLLAMA_CLOUD)
        outcome, renderer = self._run(manager, 68, model="pro")
        self.assertEqual(self._failures(renderer), [])
        self.assertEqual(manager.attempts, ["nvidia"] * 5)

    def test_the_retry_loop_can_be_switched_off(self):
        manager = _NimFlakyManager(nim_failures=-1)
        self._park_as_out_of_credit(ProviderType.HF, ProviderType.OPENROUTER, ProviderType.GROK,
                                    ProviderType.OLLAMA_CLOUD)
        with patch.dict("os.environ", {"TAMFIS_CODE_NIM_RETRY_SECONDS": "0"}):
            outcome, renderer = self._run(manager, 67)
        # Opt-out means the pre-1.6.60 behaviour: no going back to NIM in a
        # loop (the legacy "retry cooling routes" path may still answer).
        self.assertLessEqual(manager.attempts.count("nvidia"), 2)


def manager_default_model(provider: ProviderType) -> str:
    return _ExhaustionChainManager().PROVIDERS[provider].default_model


class CreditExhaustionCooldownTests(unittest.TestCase):
    """An out-of-credit provider is parked for 15 minutes, not 30 seconds."""

    def setUp(self):
        from tamfis_code import providers as providers_module

        self.providers = providers_module
        providers_module._ROUTE_HEALTH.clear()
        self.addCleanup(providers_module._ROUTE_HEALTH.clear)

    def _cooldown(self, provider, error):
        import time

        self.providers.record_route_failure_for(provider, "m", Exception(error))
        state = self.providers._ROUTE_HEALTH[(provider.value, "m")]
        return state.circuit_open_until - time.monotonic()

    def test_credit_walls_park_a_non_nim_provider_for_fifteen_minutes(self):
        for error in (LIVE_TAMFIS_403, LIVE_OLLAMA_429, LIVE_402):
            with self.subTest(error=error[:40]):
                self.assertGreater(self._cooldown(ProviderType.HF, error), 800)

    def test_weekly_usage_limit_is_parked_until_after_the_reset_window(self):
        cooldown = self._cooldown(ProviderType.OLLAMA_CLOUD, LIVE_OLLAMA_429)
        self.assertGreater(cooldown, 5 * 60 * 60)

    def test_a_plain_rate_limit_or_503_stays_a_short_pause(self):
        self.assertLess(self._cooldown(ProviderType.HF, "HTTP 429 rate limit, retry in 5s"), 40)
        self.assertLess(self._cooldown(ProviderType.OPENROUTER, "HTTP 503 unavailable"), 40)

    def test_nim_is_never_parked_for_fifteen_minutes(self):
        self.assertLess(self._cooldown(ProviderType.NVIDIA, LIVE_OLLAMA_429), 40)


class PublicFailureDetailTests(unittest.TestCase):
    def _detail(self, text):
        from tamfis_code.runner_local import _public_failure_detail

        return _public_failure_detail(ProviderManager.__new__(ProviderManager), Exception(text))

    def test_rate_limits_and_capacity_errors_are_plain_language(self):
        for text in (LIVE_OLLAMA_429, "HTTP 503 Service Unavailable", "Error code: 502"):
            detail = self._detail(text)
            self.assertIn("busy", detail)
            for leak in ("429", "503", "502", "ollama", "Error code"):
                self.assertNotIn(leak, detail)

    def test_credit_walls_do_not_expose_the_vendor_or_the_team_id(self):
        detail = self._detail(LIVE_TAMFIS_403)
        self.assertIn("credit", detail)
        self.assertNotIn("c7f035e2", detail)
        self.assertNotIn("403", detail)

    def test_other_errors_keep_their_text_with_backend_names_redacted(self):
        detail = self._detail("nvidia returned a malformed tool call")
        self.assertNotIn("nvidia", detail.lower())
        self.assertIn("malformed tool call", detail)


class NimRetryModelRotationTests(unittest.TestCase):
    def test_rotation_starts_at_the_default_model_and_wraps(self):
        from tamfis_code.runner_local import _nim_retry_model

        config = SimpleNamespace(default_model="a", models=["a", "b", "c"])
        self.assertEqual([_nim_retry_model(config, n) for n in (1, 2, 3, 4, 5)], ["a", "b", "c", "a", "b"])

    def test_no_models_means_no_selection(self):
        from tamfis_code.runner_local import _nim_retry_model

        self.assertIsNone(_nim_retry_model(SimpleNamespace(default_model="", models=[]), 1))



class PlannerPathFailoverTests(_NimPolicyHarness):
    """Planning / reconnaissance calls follow the same policy as the answer
    stream. Before, a planning call that hit a dead route (403 credit wall,
    usage-limit 429) neither recorded the failure nor moved the turn: the main
    stream then opened on the SAME dead route and paid a second full timeout --
    the "nearly ten minutes before it stopped" shape."""

    def test_a_credit_wall_during_planning_moves_the_whole_turn_off_that_route(self):
        manager = _NimFlakyManager(nim_failures=0)
        manager._primary_error = Exception(LIVE_TAMFIS_403)
        # The subscription route is the turn's own route, and it is out of credit.
        manager.other_errors[ProviderType.TAMFIS] = manager._primary_error
        outcome, renderer = self._run(manager, 71, provider=ProviderType.TAMFIS)
        self.assertEqual(self._failures(renderer), [])
        # One planning attempt hit the dead route; the main stream never went back to it.
        self.assertEqual(manager.attempts.count("tamfis"), 1, manager.attempts)
        self.assertIn("nvidia", manager.attempts)
        # It is parked for 15 minutes, not merely paused.
        self.assertFalse(self.providers.ProviderManager.route_is_healthy(ProviderType.TAMFIS, "*"))
        # ...and /status + the footer still learn that the route ran out of credit.
        diag = state_module.route_diagnostics(71)
        self.assertEqual(diag["last_exhaustion"]["provider"], "tamfis")
        self.assertEqual(diag["current"]["provider"], "nvidia")

    def test_planning_diagnostics_never_show_a_raw_provider_error(self):
        manager = _NimFlakyManager(nim_failures=0)
        manager.other_errors[ProviderType.TAMFIS] = Exception(LIVE_TAMFIS_403)
        outcome, renderer = self._run(manager, 72, provider=ProviderType.TAMFIS)
        texts = [
            str(event.get("payload", {}).get("content", ""))
            for event in renderer.events if event.get("event_type") == "diagnostics"
        ]
        planning = [text for text in texts if text.startswith("Planning request failed")]
        self.assertTrue(planning, texts)
        for text in planning:
            for leak in ("403", "Error code", "c7f035e2", "permission-denied"):
                self.assertNotIn(leak, text)

    def test_the_mutation_check_the_turn_would_otherwise_return_to_the_dead_route(self):
        """Guards the test above: with the post-planning re-resolve disabled the
        main stream DOES go back to the dead route (2 attempts), so the
        assertion there is really measuring the fix."""
        manager = _NimFlakyManager(nim_failures=0)
        manager.other_errors[ProviderType.TAMFIS] = Exception(LIVE_TAMFIS_403)
        with patch("tamfis_code.runner_local._fresh_fallback_route", return_value=None):
            self._run(manager, 73, provider=ProviderType.TAMFIS)
        self.assertGreaterEqual(manager.attempts.count("tamfis"), 2, manager.attempts)



_RAW_BACKEND_NAMES = re.compile(
    r"(?i)(nvidia|\bnim\b|ollama|hugging\s*face|\bhf\b|openrouter|\bgrok\b|x-ai|kimi|glm|nemotron|"
    r"qwen|deepseek|moonshot|z-ai|tier[_ -]?iv|minimax|llama|mistral|gemma|gpt-oss|tamfis-gpt-)"
)


class NoRawBackendNamesReachTheUserTests(_NimPolicyHarness):
    """Owner ruling (repeated 2026-09-19): native provider and model names are
    deployment details and must never reach the public terminal. Drives the REAL
    StreamRenderer with debug output ON (every diagnostic prints) through the
    failover scenarios and scans everything it printed."""

    def _printed(self, manager, session_id, *, provider=ProviderType.NVIDIA, env=None):
        from io import StringIO

        from rich.console import Console

        from tamfis_code.render import StreamRenderer

        buffer = StringIO()
        console = Console(file=buffer, no_color=True, width=200, force_terminal=False)
        with patch.dict("os.environ", {"TAMFIS_CODE_DEBUG": "1", **(env or {})}):
            renderer = StreamRenderer(console)
            state_module.save_session_state(session_id, workspace_root="/tmp")
            with _wire_stream(manager):
                asyncio.run(run_local_agent_turn(
                    manager, provider, None,
                    [{"role": "user", "content": "continue the audit"}],
                    console, renderer, workspace_root="/tmp", session_id=session_id,
                    approval_policy="auto", interactive=False,
                ))
        return buffer.getvalue()

    def _assert_clean(self, printed, scenario):
        leaks = sorted({match.group(0) for match in _RAW_BACKEND_NAMES.finditer(printed)})
        self.assertEqual(leaks, [], f"{scenario}: raw backend names reached the terminal: {leaks}")
        self.assertGreater(len(printed.strip()), 0)

    def test_a_nim_503_storm_with_out_of_credit_alternatives(self):
        manager = _NimFlakyManager(nim_failures=3)
        self._park_as_out_of_credit(ProviderType.HF, ProviderType.OPENROUTER, ProviderType.GROK,
                                    ProviderType.OLLAMA_CLOUD)
        self._assert_clean(self._printed(manager, 81), "NIM 503 storm")

    def test_a_credit_wall_during_planning(self):
        manager = _NimFlakyManager(nim_failures=0)
        manager.other_errors[ProviderType.TAMFIS] = Exception(LIVE_TAMFIS_403)
        self._assert_clean(self._printed(manager, 82, provider=ProviderType.TAMFIS), "planner 403")

    def test_the_failure_after_the_retry_budget_is_spent(self):
        manager = _NimFlakyManager(nim_failures=-1, nim_error=LIVE_OLLAMA_429)
        self._park_as_out_of_credit(ProviderType.HF, ProviderType.OPENROUTER, ProviderType.GROK,
                                    ProviderType.OLLAMA_CLOUD)
        with patch("tamfis_code.runner_local._nim_retry_delay", return_value=0.05):
            printed = self._printed(
                manager, 83, env={"TAMFIS_CODE_NIM_RETRY_SECONDS": "0.3"},
            )
        self._assert_clean(printed, "budget spent")
        self.assertIn("checkpointed", printed)

    def test_the_scan_really_detects_a_leak(self):
        """Guards the scan itself: with the public-event sanitizer bypassed, the
        very same scenario DOES print raw names."""
        manager = _NimFlakyManager(nim_failures=1)
        with patch("tamfis_code.render.sanitize_public_event", side_effect=lambda event: event):
            printed = self._printed(manager, 84)
        self.assertTrue(_RAW_BACKEND_NAMES.search(printed), "scan is blind: no raw name found even unsanitized")
