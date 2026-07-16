#!/usr/bin/env python3
"""Regression tests for tamfis-code doctor's session/workspace-snapshot/
event-replay self-diagnosis (session-awareness audit, Phase 17 follow-up):
before this, `tamfis-code doctor` only checked connectivity/auth -- it never
verified the CLI's own claimed state (active session, workspace snapshot,
event replay) actually held up.
"""
import asyncio
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from tamfis_code.doctor import CheckResult, _diagnose_session, check_event_sequence_integrity


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


if __name__ == "__main__":
    unittest.main()
