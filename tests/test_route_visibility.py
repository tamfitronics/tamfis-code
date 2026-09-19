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
        # Route-health circuits live in providers as process-global state and are
        # opened by any failing route anywhere in the suite (a real behaviour: a
        # cooling route IS reported). Isolate it so these assertions are about
        # the session's own record, not about which tests ran first.
        from tamfis_code import providers as providers_module

        self._health_snapshot = dict(providers_module._ROUTE_HEALTH)
        providers_module._ROUTE_HEALTH.clear()

    def tearDown(self):
        from tamfis_code import providers as providers_module

        providers_module._ROUTE_HEALTH.clear()
        providers_module._ROUTE_HEALTH.update(self._health_snapshot)
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
        # /status still wants the route itself -- in product vocabulary.
        self.assertEqual(state_module.route_status_line(1), "TamfisGPT")

    def test_the_footer_line_reports_failover_and_exhaustion(self):
        state_module.save_session_state(1, workspace_root="/home")
        state_module.record_route_event(1, provider="nvidia", model="nim-1")
        state_module.record_route_error(1, provider="nvidia", error="Error code: 402 - out of credits")
        state_module.record_route_event(
            1, provider="openrouter", model="owl",
            previous_provider="nvidia", reason="402", kind="failover",
        )
        line = state_module.route_status_line(1, include_current=False)
        self.assertIn("1 failover", line)
        self.assertIn("exhausted", line)
        for backend in ("nvidia", "openrouter", "owl", "nim-1"):
            self.assertNotIn(backend, line.lower())

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


class RouteHistoryWithTimingsTests(RouteEventFixture):
    """/routes answers "which provider held this task, and for how long" -- the
    stored events alone only say what happened, not how long each route was
    actually carrying the work."""

    def test_each_route_reports_how_long_it_held_the_task(self):
        import time

        state_module.save_session_state(1, workspace_root="/home")
        state_module.record_route_event(1, provider="tamfisgpt-ultra", model="ultra")
        time.sleep(0.05)
        state_module.record_route_error(1, provider="tamfisgpt-ultra", error="402 credits")
        state_module.record_route_event(
            1, provider="nvidia", model="nim-1",
            previous_provider="tamfisgpt-ultra", reason="402", kind="failover",
        )
        time.sleep(0.05)

        history = state_module.route_history(1)
        self.assertEqual([event["kind"] for event in history],
                         ["select", "exhaustion", "failover"])
        # The first route held the task until the failover replaced it.
        self.assertGreaterEqual(history[0]["held_seconds"], 0.05)
        # The exhaustion note does not end a route's hold.
        self.assertIsNone(history[1]["held_seconds"])
        # The route still in effect is measured against now.
        self.assertGreaterEqual(history[2]["held_seconds"], 0.05)

    def test_hold_totals_aggregate_per_provider(self):
        state_module.save_session_state(1, workspace_root="/home")
        state_module.record_route_event(1, provider="nvidia", model="nim-1")
        state_module.record_route_event(
            1, provider="openrouter", model="owl",
            previous_provider="nvidia", reason="recovered", kind="recovery",
        )
        totals = state_module.route_hold_totals(1)
        providers = [provider for provider, _seconds, _count in totals]
        self.assertIn("nvidia", providers)
        self.assertIn("openrouter", providers)
        self.assertTrue(all(seconds >= 0 for _p, seconds, _c in totals))

    def test_the_routes_command_is_registered(self):
        from tamfis_code.interactive import SLASH_COMMANDS

        self.assertIn("/routes", [name for name, _description in SLASH_COMMANDS])


class ProviderLatencyPercentilesTests(unittest.TestCase):
    """Counters say a route is failing; only percentiles say it is slow."""

    def setUp(self):
        from tamfis_code import providers as providers_module

        self._providers = providers_module
        self._snapshot = {k: list(v) for k, v in providers_module._PROVIDER_LATENCY.items()}
        providers_module._PROVIDER_LATENCY.clear()

    def tearDown(self):
        self._providers._PROVIDER_LATENCY.clear()
        self._providers._PROVIDER_LATENCY.update(self._snapshot)

    def test_percentiles_reflect_the_samples(self):
        from tamfis_code.providers import ProviderType

        for value in (2.0, 4.0, 6.0, 8.0, 10.0):
            self._providers.record_provider_latency(ProviderType.NVIDIA, value)
        stats = self._providers.provider_latency_stats()["nvidia"]
        self.assertEqual(stats["samples"], 5)
        self.assertEqual(stats["p50"], 6.0)
        self.assertEqual(stats["max"], 10.0)

    def test_a_slow_outlier_shows_up_in_p95(self):
        from tamfis_code.providers import ProviderType

        for value in (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 120.0):
            self._providers.record_provider_latency(ProviderType.NVIDIA, value)
        stats = self._providers.provider_latency_stats()["nvidia"]
        self.assertEqual(stats["p50"], 1.0)
        self.assertGreaterEqual(stats["p95"], 1.0)
        self.assertEqual(stats["max"], 120.0)

    def test_samples_are_bounded(self):
        from tamfis_code.providers import ProviderType

        for index in range(self._providers._LATENCY_SAMPLES_PER_ROUTE + 25):
            self._providers.record_provider_latency(ProviderType.NVIDIA, float(index))
        self.assertEqual(
            len(self._providers._PROVIDER_LATENCY["nvidia"]),
            self._providers._LATENCY_SAMPLES_PER_ROUTE,
        )

    def test_recording_never_raises_on_junk(self):
        self._providers.record_provider_latency(None, 1.0)
        self._providers.record_provider_latency("nvidia", "not-a-number")
        self._providers.record_provider_latency("nvidia", -5.0)


class RouteReportJsonTests(RouteEventFixture):
    def test_the_json_report_carries_timeline_totals_and_latency(self):
        import json

        from tamfis_code import providers as providers_module
        from tamfis_code.providers import ProviderType

        state_module.save_session_state(1, workspace_root="/home")
        state_module.record_route_event(1, provider="tamfisgpt-ultra", model="ultra")
        state_module.record_route_error(1, provider="tamfisgpt-ultra", error="402 credits")
        state_module.record_route_event(
            1, provider="nvidia", model="nim-1",
            previous_provider="tamfisgpt-ultra", reason="402", kind="failover",
        )
        previous = {k: list(v) for k, v in providers_module._PROVIDER_LATENCY.items()}
        providers_module._PROVIDER_LATENCY.clear()
        try:
            providers_module.record_provider_latency(ProviderType.NVIDIA, 30.0)
            report = json.loads(state_module.route_report_json(1))
        finally:
            providers_module._PROVIDER_LATENCY.clear()
            providers_module._PROVIDER_LATENCY.update(previous)

        self.assertEqual(report["session_id"], 1)
        # Branded: first route in the timeline is "TamfisGPT", the next
        # distinct backend "TamfisGPT (alt 2)" -- no vendor name in the export.
        self.assertEqual(report["current"]["provider"], "TamfisGPT (alt 2)")
        self.assertEqual(len(report["events"]), 3)
        self.assertEqual(report["last_exhaustion"]["provider"], "TamfisGPT")
        held_providers = {entry["provider"] for entry in report["hold_totals"]}
        self.assertEqual(held_providers, {"TamfisGPT", "TamfisGPT (alt 2)"})
        self.assertEqual(report["latency"]["TamfisGPT (alt 2)"]["p50"], 30.0)
        blob = json.dumps(report)
        # Case-sensitive on purpose: "TamfisGPT-Ultra" is the PUBLIC tier name
        # the raw model id was mapped to, not a leak of the raw "tamfisgpt-ultra".
        for backend in ("nvidia", "nim-1", "tamfisgpt-ultra"):
            self.assertNotIn(backend, blob)
        # Every event carries the duration it held the task, for charting.
        self.assertIn("held_seconds", report["events"][0])

    def test_latency_lines_put_the_slowest_route_first(self):
        from tamfis_code import providers as providers_module
        from tamfis_code.providers import ProviderType

        previous = {k: list(v) for k, v in providers_module._PROVIDER_LATENCY.items()}
        providers_module._PROVIDER_LATENCY.clear()
        try:
            providers_module.record_provider_latency(ProviderType.NVIDIA, 3.0)
            providers_module.record_provider_latency(ProviderType.TIER_IV, 95.0)
            lines = state_module.route_latency_lines(1)
        finally:
            providers_module._PROVIDER_LATENCY.clear()
            providers_module._PROVIDER_LATENCY.update(previous)
        self.assertTrue(lines)
        self.assertIn("tier_iv", lines[0])
        self.assertIn("p95 95.0s", lines[0])


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
        self.assertIn("rerouted", bar)
        self.assertIn("credits", bar)
        # Owner ruling 2026-09-19: the footer must never name a backend.
        for backend in ("nvidia", "openrouter", "owl", "nim-1"):
            self.assertNotIn(backend, bar.lower())
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


class FooterNeverShowsCoolingTests(RouteEventFixture):
    """Cooling routes must never reach a user-visible footer/status note.

    Live-reported: "⟳ · cooling: nvidia,ollama_cloud⠇" sitting in the footer
    of a session that was working fine. A failed request briefly opens a
    provider's 30s health circuit as part of NORMAL self-healing, so the
    "cooling" label fires constantly, is not actionable, and steals footer
    width from the title and mode. It belongs in /routes only.
    """

    def _open_circuits(self, providers_providers_module, names):
        import time as time_module

        for name in names:
            providers_providers_module._ROUTE_HEALTH[(name, "*")] = (
                providers_providers_module.RouteHealth(
                    circuit_open_until=time_module.monotonic() + 300,
                )
            )

    def test_the_footer_note_ignores_cooling_routes_entirely(self):
        from tamfis_code import providers as providers_module

        state_module.save_session_state(21, workspace_root="/home")
        self._open_circuits(providers_module, ("nvidia", "ollama_cloud"))
        try:
            self.assertEqual(state_module.route_status_compact(21), "")
            self.assertEqual(
                state_module.route_status_line(21, include_current=False), "",
            )
            # /status's own line also stays free of the cooling noise.
            self.assertNotIn("cooling", state_module.route_status_line(21))
        finally:
            providers_module._ROUTE_HEALTH.clear()

    def test_cooling_does_not_dilute_a_real_failover_note(self):
        from tamfis_code import providers as providers_module

        state_module.save_session_state(22, workspace_root="/home")
        state_module.record_route_event(22, provider="nvidia", model="nim-1")
        state_module.record_route_event(
            22, provider="openrouter", model="owl",
            previous_provider="nvidia", reason="402", kind="failover",
        )
        self._open_circuits(providers_module, ("ollama_cloud",))
        try:
            note = state_module.route_status_compact(22)
            self.assertIn("rerouted", note)
            self.assertIn("1 failover", note)
            self.assertNotIn("cooling", note)
            self.assertNotIn("nvidia", note.lower())
            self.assertNotIn("openrouter", note.lower())
        finally:
            providers_module._ROUTE_HEALTH.clear()

    def test_routes_still_reports_cooling(self):
        """The diagnostics surface keeps the detail; only the footer drops it."""
        from tamfis_code import providers as providers_module

        state_module.save_session_state(23, workspace_root="/home")
        self._open_circuits(providers_module, ("nvidia",))
        try:
            diag = state_module.route_diagnostics(23)
            self.assertIn("nvidia", diag["cooling"])
            self.assertIn(
                "nvidia", state_module.cooling_route_names(),
            )
        finally:
            providers_module._ROUTE_HEALTH.clear()


class SessionLatencyPersistenceTests(RouteEventFixture):
    """Per-provider percentiles must belong to the SESSION and survive a
    restart. The in-process registry cannot do either: it is empty in a new
    process (so `/routes --json` has nothing to chart) and it mixes every
    session the process served (so a chart of "this session" would be wrong).
    """

    def setUp(self):
        super().setUp()
        from tamfis_code import providers as providers_module

        self._providers = providers_module
        self._process_snapshot = {
            k: list(v) for k, v in providers_module._PROVIDER_LATENCY.items()
        }
        self._failure_snapshot = dict(providers_module._PROVIDER_LATENCY_FAILURES)
        providers_module._PROVIDER_LATENCY.clear()
        providers_module._PROVIDER_LATENCY_FAILURES.clear()

    def tearDown(self):
        self._providers._PROVIDER_LATENCY.clear()
        self._providers._PROVIDER_LATENCY.update(self._process_snapshot)
        self._providers._PROVIDER_LATENCY_FAILURES.clear()
        self._providers._PROVIDER_LATENCY_FAILURES.update(self._failure_snapshot)
        super().tearDown()

    def test_failed_attempts_are_counted_next_to_the_timings(self):
        state_module.save_session_state(9, workspace_root="/home")
        state_module.record_route_latency(9, "nvidia", 0.02)
        state_module.record_route_latency(9, "nvidia", 0.03, ok=False)
        state_module.record_route_latency(9, "nvidia", 0.04, ok=False)
        stats, _ = state_module.route_latency_stats(9)
        self.assertEqual(stats["nvidia"]["samples"], 3)
        self.assertEqual(stats["nvidia"]["failures"], 2)
        # The line says so, so an instantly-failing route is not read as fast.
        line = state_module.route_latency_lines(9)[0]
        self.assertIn("2 failed", line)

    def test_samples_are_percentiles_and_survive_a_restart(self):
        state_module.save_session_state(7, workspace_root="/home")
        for value in (2.0, 4.0, 6.0, 8.0, 10.0):
            state_module.record_route_latency(7, "nvidia", value)
        stats, source = state_module.route_latency_stats(7)
        self.assertEqual(source, "session")
        self.assertEqual(stats["nvidia"]["samples"], 5)
        self.assertEqual(stats["nvidia"]["p50"], 6.0)
        self.assertEqual(stats["nvidia"]["max"], 10.0)

        # A restart is a fresh process with an empty registry and a re-read
        # state.json -- exactly what the durable copy exists for.
        state_module._STATE_CACHE = None
        self._providers._PROVIDER_LATENCY.clear()
        stats, source = state_module.route_latency_stats(7)
        self.assertEqual(source, "session")
        self.assertEqual(stats["nvidia"]["samples"], 5)
        self.assertEqual(stats["nvidia"]["p95"], 10.0)

    def test_sessions_do_not_share_latency(self):
        state_module.save_session_state(1, workspace_root="/a")
        state_module.save_session_state(2, workspace_root="/b")
        state_module.record_route_latency(1, "nvidia", 3.0)
        state_module.record_route_latency(2, "tier_iv", 90.0)

        session_one, _ = state_module.route_latency_stats(1)
        session_two, _ = state_module.route_latency_stats(2)
        self.assertEqual(set(session_one), {"nvidia"})
        self.assertEqual(set(session_two), {"tier_iv"})
        self.assertEqual(session_one["nvidia"]["p50"], 3.0)

    def test_process_samples_are_a_labelled_fallback_only(self):
        state_module.save_session_state(3, workspace_root="/home")
        self._providers.record_provider_latency("nvidia", 4.0)
        stats, source = state_module.route_latency_stats(3)
        self.assertEqual(source, "process")
        self.assertEqual(stats["nvidia"]["samples"], 1)
        # ...and the report says so, rather than presenting them as the
        # session's own numbers.
        report = state_module.route_report(3)
        self.assertEqual(report["latency_source"], "process")
        lines = state_module.route_latency_lines(3)
        self.assertTrue(any("no samples recorded for this session yet" in line for line in lines))

    def test_a_slow_route_sorts_first_in_the_report(self):
        state_module.save_session_state(4, workspace_root="/home")
        for value in (3.0, 3.5, 4.0):
            state_module.record_route_latency(4, "nvidia", value)
        state_module.record_route_latency(4, "tier_iv", 95.0)
        report = state_module.route_report(4)
        self.assertEqual(report["latency_source"], "session")
        self.assertEqual(report["latency"]["tier_iv"]["p95"], 95.0)
        self.assertEqual(report["latency"]["nvidia"]["samples"], 3)
        lines = state_module.route_latency_lines(4)
        self.assertIn("tier_iv", lines[0])
        self.assertIn("p95 95.0s", lines[0])

    def test_the_json_export_is_chartable(self):
        import json

        state_module.save_session_state(5, workspace_root="/home")
        state_module.record_route_event(5, provider="nvidia", model="nim-1")
        state_module.record_route_latency(5, "nvidia", 12.0)
        state_module.record_route_latency(5, "nvidia", 13.0, ok=False)
        payload = json.loads(state_module.route_report_json(5))
        self.assertEqual(payload["session_id"], 5)
        self.assertEqual(payload["latency_source"], "session")
        self.assertEqual(payload["latency"]["TamfisGPT"]["max"], 13.0)
        self.assertEqual(payload["latency"]["TamfisGPT"]["failures"], 1)
        self.assertEqual(payload["current"]["provider"], "TamfisGPT")
        self.assertNotIn("nvidia", state_module.route_report_json(5).lower())
        # sorted keys + indented, so a diff of two exports is readable.
        self.assertIn('\n  "latency"', state_module.route_report_json(5))

    def test_samples_stay_bounded_per_provider(self):
        from tamfis_code.state import ROUTE_LATENCY_SAMPLES_PER_PROVIDER

        state_module.save_session_state(6, workspace_root="/home")
        for index in range(ROUTE_LATENCY_SAMPLES_PER_PROVIDER + 30):
            state_module.record_route_latency(6, "nvidia", float(index))
        self.assertEqual(
            len(state_module.get_session_state(6).route_latency["nvidia"]),
            ROUTE_LATENCY_SAMPLES_PER_PROVIDER,
        )
        self.assertEqual(
            state_module.route_latency_stats(6)[0]["nvidia"]["samples"],
            ROUTE_LATENCY_SAMPLES_PER_PROVIDER,
        )

    def test_recording_never_raises_on_junk(self):
        state_module.save_session_state(8, workspace_root="/home")
        state_module.record_route_latency(8, None, 1.0)
        state_module.record_route_latency(8, "nvidia", "not-a-number")
        state_module.record_route_latency(8, "nvidia", -5.0)
        self.assertEqual(state_module.get_session_state(8).route_latency, {})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class PublicRouteSurfaceTests(RouteEventFixture):
    """/status and /routes go through the same vendor-free branding as the
    footer: no provider or catalog model id ever reaches the user."""

    def _failed_over_session(self):
        state_module.save_session_state(31, workspace_root="/home")
        state_module.record_route_event(31, provider="nvidia", model="nvidia/nemotron-3-super-120b-a12b")
        state_module.record_route_error(
            31, provider="nvidia",
            error="Error code: 429 - nvidia weekly usage limit, ollama_cloud also cooling",
        )
        state_module.record_route_event(
            31, provider="ollama_cloud", model="kimi-k3:cloud",
            previous_provider="nvidia", reason="429 from nvidia", kind="failover",
        )
        state_module.record_route_event(
            31, provider="hf", model="Qwen/Qwen3.6-35B-A3B",
            previous_provider="ollama_cloud", reason="quota", kind="failover",
        )

    def test_status_route_block_names_no_backend(self):
        self._failed_over_session()
        block = state_module.public_route_status_lines(31)
        self.assertIn("route=TamfisGPT (alt 3)", block)
        self.assertIn("failover=TamfisGPT -> TamfisGPT (alt 2)", block)
        for backend in ("nvidia", "ollama", "hf", "qwen", "kimi", "nemotron"):
            self.assertNotIn(backend, block.lower())

    def test_report_labels_follow_the_order_the_task_moved(self):
        self._failed_over_session()
        report = state_module.public_route_report(31)
        self.assertEqual(
            [entry["provider"] for entry in report["hold_totals"]],
            ["TamfisGPT", "TamfisGPT (alt 2)", "TamfisGPT (alt 3)"],
        )
        blob = str(report).lower()
        for backend in ("nvidia", "ollama", "'hf'", "qwen", "kimi", "nemotron"):
            self.assertNotIn(backend, blob)

    def test_the_same_backend_always_gets_the_same_label(self):
        from tamfis_code.public_identity import RouteLabeler

        label = RouteLabeler()
        self.assertEqual(label("nvidia"), "TamfisGPT")
        self.assertEqual(label("Ollama_Cloud"), "TamfisGPT (alt 2)")
        self.assertEqual(label("NVIDIA"), "TamfisGPT")
        self.assertEqual(label(""), "")

    def test_the_footer_note_after_a_double_failover_is_vendor_free(self):
        self._failed_over_session()
        note = state_module.route_status_compact(31)
        self.assertEqual(note, "⟳ rerouted · credits · 2 failovers")
        self.assertLessEqual(len(note), 56)

    def test_an_exhausted_route_with_no_failover_says_the_route_is_down(self):
        state_module.save_session_state(32, workspace_root="/home")
        state_module.record_route_error(32, provider="nvidia", error="402 credits")
        note = state_module.route_status_compact(32)
        self.assertTrue(note.startswith("⟳ route down"))
        self.assertNotIn("nvidia", note.lower())
