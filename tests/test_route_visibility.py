"""Route failover and credit exhaustion must be visible in the UI.

Live-reported 2026-09-19: a 35-minute run died with

    TamfisGPT-Ultra streaming failed: Error code: 402 - {'error': 'You have
    depleted your monthly included credits...'}

while NVIDIA NIM sat configured and unused -- and nothing in `/status` or the
footer said the route was dead or that no failover had happened. That story
existed only as a debug diagnostic scrolling past.

These tests pin the persistence and the rendering: which route a session is
actually on, every failover it took, an exhausted route recorded with its
reason, and the two UI surfaces (the persistent footer, and the /status block)
showing them without needing --debug.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tamfis_code import state as state_module


class RouteEventFixture(unittest.TestCase):
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

    def tearDown(self):
        (
            state_module.CONFIG_DIR, state_module.STATE_PATH, state_module._LOCK_PATH,
        ) = self._originals
        state_module._STATE_CACHE = None
        self._tmp.cleanup()


class ExhaustionDetectionTests(unittest.TestCase):
    def test_the_live_402_credit_message_is_an_exhaustion(self):
        self.assertTrue(state_module.route_error_is_exhaustion(
            "Error code: 402 - {'error': 'You have depleted your monthly included "
            "credits. Purchase pre-paid credits to continue using Inference Providers.'}"
        ))

    def test_quota_and_rate_limit_wording_are_exhaustion(self):
        for message in (
            "429 Too Many Requests: rate limit exceeded",
            "monthly quota exhausted",
            "insufficient credits",
            "billing issue on this account",
        ):
            with self.subTest(message=message):
                self.assertTrue(state_module.route_error_is_exhaustion(message))

    def test_an_ordinary_error_is_not_an_exhaustion(self):
        self.assertFalse(state_module.route_error_is_exhaustion(
            "Unterminated string starting at: line 1 column 12"
        ))
        self.assertFalse(state_module.route_error_is_exhaustion(""))


class RouteDiagnosticsTests(RouteEventFixture):
    def test_a_failover_session_reports_the_live_route_and_the_history(self):
        state_module.save_session_state(1, workspace_root="/home")
        state_module.record_route_event(1, provider="tamfisgpt-ultra", model="ultra")
        state_module.record_route_error(
            1, provider="tamfisgpt-ultra",
            error="Error code: 402 - depleted your monthly included credits",
        )
        state_module.record_route_event(
            1, provider="nvidia", model="nim-1",
            previous_provider="tamfisgpt-ultra", reason="402 credits", kind="failover",
        )

        diag = state_module.route_diagnostics(1)
        self.assertEqual(diag["current"]["provider"], "nvidia")
        self.assertEqual(len(diag["failovers"]), 1)
        self.assertEqual(diag["failovers"][0]["from_provider"], "tamfisgpt-ultra")
        self.assertEqual(diag["last_exhaustion"]["provider"], "tamfisgpt-ultra")
        self.assertIn("402", diag["last_exhaustion"]["reason"])

    def test_the_footer_line_is_silent_until_something_actually_happened(self):
        state_module.save_session_state(1, workspace_root="/home")
        state_module.record_route_event(1, provider="nvidia", model="nim-1")
        # A healthy session adds nothing to a footer that already names the model.
        self.assertEqual(
            state_module.route_status_line(1, include_current=False), "",
        )
        # /status still wants the route itself.
        self.assertEqual(state_module.route_status_line(1), "nvidia/nim-1")

    def test_the_footer_line_reports_failover_and_exhaustion(self):
        state_module.save_session_state(1, workspace_root="/home")
        state_module.record_route_event(1, provider="nvidia", model="nim-1")
        state_module.record_route_error(1, provider="nvidia", error="Error code: 402 - out of credits")
        state_module.record_route_event(
            1, provider="openrouter", model="owl",
            previous_provider="nvidia", reason="402", kind="failover",
        )
        line = state_module.route_status_line(1, include_current=False)
        self.assertIn("nvidia → openrouter/owl", line)
        self.assertIn("1 failover", line)
        self.assertIn("exhausted", line)

    def test_a_session_with_no_route_record_is_empty(self):
        state_module.save_session_state(9, workspace_root="/home")
        self.assertEqual(state_module.route_status_line(9), "")
        self.assertEqual(state_module.route_diagnostics(9)["failovers"], [])

    def test_route_events_are_bounded(self):
        state_module.save_session_state(1, workspace_root="/home")
        for index in range(state_module.ROUTE_EVENT_LIMIT + 8):
            state_module.record_route_event(1, provider=f"route-{index}", model="m")
        self.assertLessEqual(
            len(state_module.get_session_state(1).route_events),
            state_module.ROUTE_EVENT_LIMIT,
        )

    def test_recording_never_raises_for_an_unknown_session(self):
        state_module.record_route_event(424242, provider="nvidia")
        state_module.record_route_error(424242, provider="nvidia", error="402")


class OrchestratorRecordsRouteChangesTests(RouteEventFixture):
    """record_route is the single authoritative route-change hook -- it must
    feed the same persisted history the UI reads."""

    def _orchestrator(self):
        from tamfis_code.orchestrator.engine import AgentOrchestrator

        orchestrator = object.__new__(AgentOrchestrator)
        orchestrator.session_id = 1
        orchestrator.run = SimpleNamespace(route={}, phase=None)
        return orchestrator

    def test_a_provider_switch_is_recorded_as_a_failover(self):
        state_module.save_session_state(1, workspace_root="/home")
        orchestrator = self._orchestrator()
        with patch.object(orchestrator, "transition", lambda *a, **k: None):
            orchestrator.record_route(provider="tamfisgpt-ultra", model="ultra", reason="initial")
            orchestrator.record_route(
                provider="nvidia", model="nim-1", reason="automatic provider fallback",
            )
        events = state_module.get_session_state(1).route_events
        self.assertEqual(events[-1]["kind"], "failover")
        self.assertEqual(events[-1]["from_provider"], "tamfisgpt-ultra")
        self.assertEqual(events[-1]["provider"], "nvidia")

    def test_a_staying_route_is_not_recorded_as_a_failover(self):
        state_module.save_session_state(1, workspace_root="/home")
        orchestrator = self._orchestrator()
        with patch.object(orchestrator, "transition", lambda *a, **k: None):
            orchestrator.record_route(provider="nvidia", model="nim-1", reason="initial")
            orchestrator.record_route(provider="nvidia", model="nim-1", reason="same route")
        kinds = [event["kind"] for event in state_module.get_session_state(1).route_events]
        self.assertNotIn("failover", kinds)


class FooterShowsRouteExceptionsTests(RouteEventFixture):
    def _bar(self, session_id, model="auto"):
        from tamfis_code.config import Config
        from tamfis_code.live_input import idle_bottom_toolbar

        return str(idle_bottom_toolbar(Config(), session_id, model=model))

    def test_a_healthy_session_has_no_failover_marker(self):
        state_module.save_session_state(1, workspace_root="/home")
        state_module.record_route_event(1, provider="nvidia", model="nim-1")
        self.assertNotIn("⟳", self._bar(1))

    def test_a_failover_session_shows_the_marker_and_the_reason(self):
        state_module.save_session_state(1, workspace_root="/home")
        state_module.record_route_event(1, provider="nvidia", model="nim-1")
        state_module.record_route_error(
            1, provider="nvidia", error="Error code: 402 - depleted your monthly included credits",
        )
        state_module.record_route_event(
            1, provider="openrouter", model="owl",
            previous_provider="nvidia", reason="402", kind="failover",
        )
        bar = self._bar(1)
        self.assertIn("⟳", bar)
        self.assertIn("failover", bar)
        self.assertIn("nvidia→openrouter", bar)
        self.assertIn("credits", bar)
        # The footer shares one terminal row with the title/mode/agents, so the
        # note is short by construction; the full wording stays in /status.
        note = state_module.route_status_compact(1)
        self.assertLessEqual(len(note), 56)
        self.assertIn("exhausted", state_module.route_status_line(1))

    def test_the_compact_note_is_empty_for_a_healthy_session(self):
        state_module.save_session_state(1, workspace_root="/home")
        state_module.record_route_event(1, provider="nvidia", model="nim-1")
        self.assertEqual(state_module.route_status_compact(1), "")


class StatusBlockTests(RouteEventFixture):
    def test_the_status_route_section_lists_failures_and_exhaustion(self):
        """/status builds this text from the same persisted record; a missing
        route line is exactly the live complaint."""
        state_module.save_session_state(1, workspace_root="/home")
        state_module.record_route_event(1, provider="tamfisgpt-ultra", model="ultra")
        state_module.record_route_error(
            1, provider="tamfisgpt-ultra", error="Error code: 402 - depleted",
        )
        state_module.record_route_event(
            1, provider="nvidia", model="nim-1",
            previous_provider="tamfisgpt-ultra", reason="402", kind="failover",
        )
        diag = state_module.route_diagnostics(1)
        rendered = (
            f"route={diag['current']['provider']}"
            f"  exhausted={diag['last_exhaustion']['provider']}"
            f"  failover={diag['failovers'][0]['from_provider']}"
            f" -> {diag['failovers'][0]['provider']}"
        )
        self.assertIn("route=nvidia", rendered)
        self.assertIn("exhausted=tamfisgpt-ultra", rendered)
        self.assertIn("failover=tamfisgpt-ultra -> nvidia", rendered)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
