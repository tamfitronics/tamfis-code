#!/usr/bin/env python3
"""Regression tests for tamfis-code doctor's session/workspace-snapshot/
event-replay self-diagnosis (session-awareness audit, Phase 17 follow-up):
before this, `tamfis-code doctor` only checked connectivity/auth -- it never
verified the CLI's own claimed state (active session, workspace snapshot,
event replay) actually held up.
"""
import asyncio
import os
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from rich.console import Console

from tamfis_code import state as state_module
from tamfis_code.config import Config
from tamfis_code.doctor import (
    CheckResult, _diagnose_local_providers, _diagnose_local_session,
    _diagnose_session, check_event_sequence_integrity, check_path_safety, run_doctor,
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


def _safe_path_check():
    """A GitHub runner's PATH has world-writable, non-sticky directories, which the doctor's
    PATH-safety check rightly FAILs -- so `run_doctor(...)` returned False there and these
    tests (about remote/standalone behaviour, not PATH) failed on CI. Pin the PATH check's
    result; test_check_remote_api_false_still_runs_path_safety covers the real check."""
    from tamfis_code.doctor import CheckResult

    return patch(
        "tamfis_code.doctor.check_path_safety",
        return_value=CheckResult("PATH safety", "PASS", "pinned for this test"),
    )

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
        self.assertEqual(by_name["TamfisGPT model service"].status, "PASS")

    def test_no_provider_configured_fails_routing(self):
        # A truly unusable environment: no API keys set for any provider,
        # with no local no-credential fallback to fall back on.
        with patch("tamfis_code.doctor.get_provider_status", return_value=self._status(configured=False)):
            results = _diagnose_local_providers()
        by_name = {r.name: r for r in results}
        self.assertEqual(by_name["TamfisGPT model service"].status, "FAIL")

    def test_doctor_without_remote_credentials_makes_no_remote_request(self):
        console = Console(file=StringIO(), no_color=True, width=200)
        with patch("tamfis_code.doctor.load_credentials", return_value=None), \
             patch("tamfis_code.doctor.get_provider_status", return_value=self._status(configured=True)), \
             patch("tamfis_code.doctor.RemoteAPIClient") as remote_client, \
             _safe_path_check():
            result = _run(run_doctor(Config(), console))

        self.assertTrue(result)
        remote_client.assert_not_called()
        self.assertIn("not checked without --remote credentials", console.file.getvalue())

    def test_check_remote_api_false_skips_the_remote_backend_even_with_saved_credentials(self):
        """The actual gap this closes: `tamfis-code doctor` with no
        --remote flag used to hand-reimplement only two of run_doctor's
        many checks in cli.py instead of calling run_doctor at all, so
        this parameter (and everything it protects) never had a code path
        reaching it in practice. A user who previously ran `tamfis-code
        login` (so load_credentials() returns real creds) but is now
        running plain `doctor` must still never touch the Remote Workspace
        backend -- check_remote_api=False must hold regardless of whether
        credentials happen to exist on disk, not just when they don't
        (that easier case is test_doctor_without_remote_credentials_makes_
        no_remote_request above)."""
        console = Console(file=StringIO(), no_color=True, width=200)
        fake_creds = SimpleNamespace(email="user@example.com", user_id=None)
        with patch("tamfis_code.doctor.load_credentials", return_value=fake_creds), \
             patch("tamfis_code.doctor.get_provider_status", return_value=self._status(configured=True)), \
             patch("tamfis_code.doctor.RemoteAPIClient") as remote_client, \
             _safe_path_check():
            result = _run(run_doctor(Config(), console, check_remote_api=False))

        self.assertTrue(result)
        remote_client.assert_not_called()
        self.assertIn("not checked in standalone mode", console.file.getvalue())

    def test_check_remote_api_false_still_runs_path_safety(self):
        """Confirmed live: before this fix, `tamfis-code doctor` (no
        --remote) never ran check_path_safety at all -- a real security
        check (a world-writable, non-sticky PATH directory could let
        another local user hijack a command tamfis-code shells out to)
        that silently never executed for the default invocation every
        ordinary user actually takes."""
        console = Console(file=StringIO(), no_color=True, width=200)
        with patch("tamfis_code.doctor.load_credentials", return_value=None), \
             patch("tamfis_code.doctor.get_provider_status", return_value=self._status(configured=True)):
            _run(run_doctor(Config(), console, check_remote_api=False))
        self.assertIn("PATH safety", console.file.getvalue())


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


class PathSafetyTests(unittest.TestCase):
    """Mirrors Codex's own `doctor_path_safety` check, which tamfis-code had
    no equivalent of: execute_command resolves bare command names (git,
    python3, npm, ...) against PATH just like a shell would, so a
    world-writable, non-sticky directory on PATH is a real local
    privilege-escalation vector -- another local user could plant a
    same-named binary there and have it silently run instead of the real
    one on tamfis-code's behalf."""

    def setUp(self):
        self._original_path = os.environ.get("PATH", "")

    def tearDown(self):
        os.environ["PATH"] = self._original_path

    def test_passes_when_every_path_directory_is_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.chmod(tmp, 0o755)
            os.environ["PATH"] = tmp
            result = check_path_safety()
        self.assertEqual(result.status, "PASS")

    def test_fails_on_a_world_writable_non_sticky_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.chmod(tmp, 0o777)
            os.environ["PATH"] = tmp
            result = check_path_safety()
        self.assertEqual(result.status, "FAIL")
        self.assertIn(tmp, result.detail)

    def test_world_writable_but_sticky_directory_is_exempt(self):
        # Mirrors /tmp itself: world-writable but sticky-protected, so
        # another user can create files but can never replace or delete
        # someone else's -- the classic safe shared-tmp pattern.
        with tempfile.TemporaryDirectory() as tmp:
            os.chmod(tmp, 0o1777)
            os.environ["PATH"] = tmp
            result = check_path_safety()
        self.assertEqual(result.status, "PASS")

    def test_empty_path_is_a_warning_not_a_failure(self):
        os.environ["PATH"] = ""
        result = check_path_safety()
        self.assertEqual(result.status, "WARNING")

    def test_a_missing_path_directory_is_skipped_not_raised(self):
        os.environ["PATH"] = "/definitely/does/not/exist/anywhere"
        result = check_path_safety()
        self.assertEqual(result.status, "PASS")


if __name__ == "__main__":
    unittest.main()
