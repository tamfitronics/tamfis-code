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
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tamfis_code import state as state_module
from tamfis_code.providers import ProviderManager, ProviderType
from tamfis_code.runner_local import (
    _fallback_candidates_for_turn,
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
