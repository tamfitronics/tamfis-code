"""Provider route churn: the stall that made a live turn take minutes.

Live-reported (2026-09-19): a single `tamfis-code ask` turn spent its entire
270s wall on two route switches ("Switching TamfisGPT route; task is still
running…") and never reached the model. Two mechanisms were missing:

* nothing bounded the wait for a provider's FIRST byte -- only the gap between
  subsequent chunks -- so a provider that accepted the connection and went
  quiet parked each attempt for the SDK's full 120s transport timeout;
* the direct-client call paths (planning, revision, recovery) never recorded a
  route failure, so the hung route was never demoted and the next attempt
  resolved straight back to it.

These tests pin the fixes and the per-turn route-recovery budget.
"""
from __future__ import annotations

import asyncio
import tempfile
import time
import unittest
from pathlib import Path

from tamfis_code import runner_local as rl
from tamfis_code import state as state_module
from tamfis_code.providers import ProviderType, ProviderManager


class _RecordingRenderer:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def handle_event(self, event: dict) -> None:
        self.events.append(event)

    def diagnostics(self) -> list[str]:
        return [
            str(event.get("payload", {}).get("content", ""))
            for event in self.events
            if event.get("event_type") == "diagnostics"
        ]


class _SilentClient:
    """Accepts the request and never answers -- the exact live symptom."""

    class chat:
        class completions:
            @staticmethod
            async def create(**kwargs):
                await asyncio.sleep(300)


class FirstByteWatchdogTests(unittest.TestCase):
    def setUp(self):
        self._original = rl.PROVIDER_FIRST_BYTE_TIMEOUT_SECONDS
        rl.PROVIDER_FIRST_BYTE_TIMEOUT_SECONDS = 0.2

    def tearDown(self):
        rl.PROVIDER_FIRST_BYTE_TIMEOUT_SECONDS = self._original

    def test_a_provider_that_never_answers_is_abandoned_quickly(self):
        started = time.monotonic()
        with self.assertRaises(TimeoutError) as ctx:
            asyncio.run(rl._stream_one_completion(
                _SilentClient(), model="x", messages=[{"role": "user", "content": "hi"}],
                tools=[], renderer=_RecordingRenderer(), provider=ProviderType.NVIDIA,
            ))
        elapsed = time.monotonic() - started
        self.assertIn("time-to-first-byte", str(ctx.exception))
        self.assertLess(elapsed, 3.0, "a silent provider must not hold the turn for the SDK's 120s")

    def test_the_watchdog_is_classified_retryable_so_fallback_kicks_in(self):
        """A hang must be treated as a route failure, not as task failure --
        otherwise the turn dies instead of trying another provider."""
        try:
            asyncio.run(rl._stream_one_completion(
                _SilentClient(), model="x", messages=[{"role": "user", "content": "hi"}],
                tools=[], renderer=_RecordingRenderer(), provider=ProviderType.NVIDIA,
            ))
        except Exception as exc:  # noqa: BLE001 - the type is the assertion
            self.assertTrue(ProviderManager.is_retryable_provider_error(exc))


class DirectRouteFailureRecordingTests(unittest.TestCase):
    def setUp(self):
        # Route health is deliberately process-global (that is what makes it
        # useful across a session), so each test starts from a clean slate.
        from tamfis_code import providers as providers_module

        self._providers = providers_module
        self._saved_health = dict(providers_module._ROUTE_HEALTH)
        providers_module._ROUTE_HEALTH.clear()

    def tearDown(self):
        self._providers._ROUTE_HEALTH.clear()
        self._providers._ROUTE_HEALTH.update(self._saved_health)

    def test_a_direct_client_failure_demotes_the_resolved_route(self):
        from tamfis_code.runtime.telemetry import provider_context

        manager = ProviderManager()
        # A route is healthy until something records a failure for it.
        before = manager.route_is_healthy(ProviderType.HF, "*")
        with provider_context(ProviderType.HF.value):
            rl._note_direct_route_failure(TimeoutError("no response"))
        self.assertTrue(before)
        self.assertFalse(
            manager.route_is_healthy(ProviderType.HF, "*"),
            "a hung direct-client route must open its circuit",
        )

    def test_an_explicit_provider_wins_over_the_telemetry_context(self):
        from tamfis_code.runtime.telemetry import provider_context

        manager = ProviderManager()
        with provider_context(ProviderType.HF.value):
            rl._note_direct_route_failure(TimeoutError("no response"), provider=ProviderType.GROK, model="grok-4")
        self.assertFalse(manager.route_is_healthy(ProviderType.GROK, "grok-4"))
        self.assertTrue(manager.route_is_healthy(ProviderType.HF, "*"))

    def test_an_unresolved_route_is_not_demoted(self):
        """With no telemetry context and no explicit provider there is no
        specific route to blame -- blaming one would punish a healthy route."""
        manager = ProviderManager()
        rl._note_direct_route_failure(TimeoutError("no response"))
        self.assertTrue(manager.route_is_healthy(ProviderType.GROK, "*"))
        self.assertTrue(manager.route_is_healthy(ProviderType.HF, "*"))

    def test_recording_never_raises_on_a_non_exception(self):
        rl._note_direct_route_failure(Exception("boom"), provider=ProviderType.NVIDIA, model="m")


class _FakePlanManager:
    """Minimal ProviderManager stand-in for the planning retry loop."""

    def __init__(self) -> None:
        from types import SimpleNamespace

        self.PROVIDERS = {
            ProviderType.GROK: SimpleNamespace(
                default_model="grok-4", models=("grok-4",),
                tool_calling=True, context_window=131072,
            ),
            ProviderType.HF: SimpleNamespace(
                default_model="hf-model", models=("hf-model",),
                tool_calling=True, context_window=131072,
            ),
        }

    @staticmethod
    def is_retryable_provider_error(exc: Exception) -> bool:
        return True

    @staticmethod
    def is_quota_or_rate_limit_error(exc: Exception) -> bool:
        return False

    @staticmethod
    def get_client(provider):
        return object()

    @staticmethod
    def fallback_candidates(provider, task_profile, **kwargs):
        return [ProviderType.GROK]


class RouteRecoveryBudgetTests(unittest.TestCase):
    def setUp(self):
        self._originals = (state_module.CONFIG_DIR, state_module.STATE_PATH, state_module._LOCK_PATH)
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        state_module.CONFIG_DIR = base / ".config"
        state_module.STATE_PATH = base / ".config" / "state.json"
        state_module._LOCK_PATH = base / ".config" / ".state.lock"
        state_module._STATE_CACHE = None
        self._original_budget = rl.PROVIDER_RECOVERY_BUDGET_SECONDS

    def tearDown(self):
        state_module.CONFIG_DIR, state_module.STATE_PATH, state_module._LOCK_PATH = self._originals
        state_module._STATE_CACHE = None
        rl.PROVIDER_RECOVERY_BUDGET_SECONDS = self._original_budget
        self._tmp.cleanup()

    def test_planning_stops_cycling_routes_once_the_budget_is_spent(self):
        state_module.save_session_state(1, workspace_root="/a")
        rl.PROVIDER_RECOVERY_BUDGET_SECONDS = 0.0  # budget already spent
        renderer = _RecordingRenderer()
        attempts: list[str] = []

        async def failing_stream(client, **kwargs):
            attempts.append(str(kwargs.get("model")))
            raise TimeoutError("no response")

        original = rl._stream_one_completion
        rl._stream_one_completion = failing_stream
        try:
            from tamfis_code.routing import classify_task

            profile = classify_task("Fix the login bug in auth.py")
            plan = asyncio.run(rl._attempt_reasoning_plan(
                object(), model="m", objective="Fix the login bug in auth.py",
                task_profile=profile, session_id=1, renderer=renderer,
                manager=_FakePlanManager(), provider=ProviderType.HF,
            ))
        finally:
            rl._stream_one_completion = original

        self.assertIsNone(plan, "planning must fall back to the existing plan")
        self.assertEqual(len(attempts), 1, "the budget must stop the route cycling after one attempt")
        diagnostics = " ".join(renderer.diagnostics())
        self.assertIn("gave up after", diagnostics)
        self.assertIn("hf", diagnostics.lower(), "the failure must name the route that was tried")


if __name__ == "__main__":
    unittest.main()
