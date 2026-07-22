#!/usr/bin/env python3
"""Regression tests for tamfis-code doctor's session/workspace-snapshot/
event-replay self-diagnosis (session-awareness audit, Phase 17 follow-up):
before this, `tamfis-code doctor` only checked connectivity/auth -- it never
verified the CLI's own claimed state (active session, workspace snapshot,
event replay) actually held up.
"""
import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from tamfis_code import state as state_module
from tamfis_code.doctor import (
    CheckResult, _diagnose_local_providers, _diagnose_local_session,
    _diagnose_session, check_event_sequence_integrity,
)


def _run(coro):
    return asyncio.run(coro)


class EventSequenceIntegrityTests(unittest.TestCase):
    def test_empty_events_is_a_warning_not_a_failure(self):
        result = check_event_sequence_integrity([])
        self.assertEqual(result.status, "WARNING")

    def test_contiguous_sequence_passes(self):
        events = [{"sequence": n} for n in (1, 2, 3, 4)]
        result = check_event_sequence_integrity(events)
        self.assertEqual(result.status, "PASS")

    def test_out_of_order_but_contiguous_still_passes(self):
        events = [{"sequence": n} for n in (3, 1, 4, 2)]
        result = check_event_sequence_integrity(events)
        self.assertEqual(result.status, "PASS")

    def test_duplicate_sequence_fails(self):
        events = [{"sequence": n} for n in (1, 2, 2, 3)]
        result = check_event_sequence_integrity(events)
        self.assertEqual(result.status, "FAIL")
        self.assertIn("duplicate", result.detail.lower())

    def test_gap_is_a_warning_not_a_failure(self):
        """A gap can't be told apart from this check's own window/limit
        truncation from here, so it must not read as a hard failure."""
        events = [{"sequence": n} for n in (1, 2, 5, 6)]
        result = check_event_sequence_integrity(events)
        self.assertEqual(result.status, "WARNING")
        self.assertIn("gap", result.detail.lower())

    def test_missing_sequence_field_fails(self):
        events = [{"sequence": 1}, {"event_type": "assistant_delta"}]
        result = check_event_sequence_integrity(events)
        self.assertEqual(result.status, "FAIL")


class DiagnoseSessionTests(unittest.TestCase):
    def _client(self, session_response, thread_response):
        client = AsyncMock()
        client.get_session.return_value = session_response
        client.get_thread.return_value = thread_response
        return client

    def test_active_session_with_fresh_snapshot_and_clean_sequence_all_pass(self):
        client = self._client(
            session_response={
                "status": "idle",
                "working_directory": "/repo",
                "workspace_snapshot": {
                    "file_index_version": 2, "repository_type": "git",
                    "git_branch": "main", "last_scan_at": "2026-07-12T10:00:00",
                    "scan_reason": "initial_scan",
                },
            },
            thread_response={"events": [{"sequence": n} for n in (1, 2, 3)]},
        )
        results = _run(_diagnose_session(client, 1, Path("/repo")))
        by_name = {r.name: r for r in results}
        self.assertEqual(by_name["Active session"].status, "PASS")
        self.assertEqual(by_name["Session cwd matches local cwd"].status, "PASS")
        self.assertEqual(by_name["Workspace snapshot"].status, "PASS")
        self.assertEqual(by_name["Event replay integrity"].status, "PASS")

    def test_mismatched_cwd_is_a_warning(self):
        client = self._client(
            session_response={"status": "idle", "working_directory": "/somewhere/else", "workspace_snapshot": None},
            thread_response={"events": []},
        )
        results = _run(_diagnose_session(client, 1, Path("/repo")))
        by_name = {r.name: r for r in results}
        self.assertEqual(by_name["Session cwd matches local cwd"].status, "WARNING")

    def test_no_snapshot_yet_is_a_warning_not_a_failure(self):
        client = self._client(
            session_response={"status": "idle", "working_directory": "/repo", "workspace_snapshot": None},
            thread_response={"events": []},
        )
        results = _run(_diagnose_session(client, 1, Path("/repo")))
        by_name = {r.name: r for r in results}
        self.assertEqual(by_name["Workspace snapshot"].status, "WARNING")

    def test_closed_session_is_a_warning(self):
        client = self._client(
            session_response={"status": "closed", "working_directory": "/repo", "workspace_snapshot": None},
            thread_response={"events": []},
        )
        results = _run(_diagnose_session(client, 1, Path("/repo")))
        by_name = {r.name: r for r in results}
        self.assertEqual(by_name["Active session"].status, "WARNING")


class DiagnoseLocalProvidersTests(unittest.TestCase):
    """Before this, doctor never looked at the 3 directly-called providers
    (HF/NVIDIA/OpenRouter) that tamfis-code's default local mode
    actually runs against at all -- these prove it does now, using the
    same get_provider_status() the `providers` command already relies on."""

    def _status(self, *, configured):
        return {
            "available": [],
            "default": "nvidia" if configured else "none",
            "config": {
                "nvidia": {"api_key_set": configured, "key_preview": "x" if configured else "Not set"},
                "hf": {"api_key_set": False, "key_preview": "Not set"},
                "openrouter": {"api_key_set": False, "key_preview": "Not set"},
            },
        }

    def test_at_least_one_configured_provider_passes_routing_check(self):
        with patch("tamfis_code.doctor.get_provider_status", return_value=self._status(configured=True)):
            results = _diagnose_local_providers()
        by_name = {r.name: r for r in results}
        self.assertEqual(by_name["Local provider: nvidia"].status, "PASS")
        self.assertEqual(by_name["Local automatic routing"].status, "PASS")

    def test_no_provider_configured_fails_routing(self):
        # A truly unusable environment: no API keys set for any provider,
        # with no local no-credential fallback to fall back on.
        with patch("tamfis_code.doctor.get_provider_status", return_value=self._status(configured=False)):
            results = _diagnose_local_providers()
        by_name = {r.name: r for r in results}
        self.assertEqual(by_name["Local automatic routing"].status, "FAIL")


class DiagnoseLocalSessionTests(unittest.TestCase):
    """resolve_local_workspace()/save_session_state() write real session
    state -- without redirecting CONFIG_DIR/STATE_PATH (as every other
    stateful test file does), these tests wrote directly into the real
    ~/.config/tamfis-code/state.json on every run, allocating a fresh
    session id each time via _next_local_session_id(). Caught alongside a
    much larger version of the same bug in test_orchestrator.py."""

    def setUp(self):
        self._state_originals = (state_module.CONFIG_DIR, state_module.STATE_PATH)
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        state_module.CONFIG_DIR = base / ".config"
        state_module.STATE_PATH = base / ".config" / "state.json"

    def tearDown(self):
        state_module.CONFIG_DIR, state_module.STATE_PATH = self._state_originals
        self._tmp.cleanup()

    def test_reports_real_persisted_evidence_for_this_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from tamfis_code.workspace import resolve_local_workspace
            ctx = resolve_local_workspace(root, discover=False)
            state_module.save_session_state(
                ctx.session_id,
                estimated_context_tokens=4200,
                completed_actions=[
                    {"type": "tool", "tool_name": "read_file", "success": True},
                    {"type": "tool", "tool_name": "write_file", "success": True},
                    {"type": "tool", "tool_name": "execute_command", "success": False},
                ],
                saved_plans=[{
                    "id": "plan_test1", "objective": "x",
                    "steps": [
                        {"index": 0, "step": "a", "status": "completed"},
                        {"index": 1, "step": "b", "status": "in_progress"},
                        {"index": 2, "step": "c", "status": "pending"},
                    ],
                }],
                active_plan_id="plan_test1",
            )
            results = _diagnose_local_session(root)
        by_name = {r.name: r for r in results}
        self.assertEqual(by_name["Local session context usage"].status, "PASS")
        self.assertIn("4200", by_name["Local session context usage"].detail)
        self.assertEqual(by_name["Local tool-call success rate"].status, "WARNING")
        self.assertIn("2/3", by_name["Local tool-call success rate"].detail)
        self.assertIn("1 completed", by_name["Active plan step progress"].detail)
        self.assertIn("1 in_progress", by_name["Active plan step progress"].detail)
        self.assertIn("1 pending", by_name["Active plan step progress"].detail)

    def test_fresh_session_reports_warnings_not_fabricated_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = _diagnose_local_session(Path(tmp))
        by_name = {r.name: r for r in results}
        self.assertEqual(by_name["Local session context usage"].status, "WARNING")
        self.assertEqual(by_name["Local tool-call success rate"].status, "WARNING")
        self.assertNotIn("Active plan step progress", by_name)


if __name__ == "__main__":
    unittest.main()
