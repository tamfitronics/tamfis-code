"""Coverage for interactive.py's standalone mode (client=None) -- the
interactive REPL calling a provider directly instead of the TamfisGPT
Remote Workspace backend. Uses the same run_interactive test harness as
test_tamfis_code_repl_exit.py (mock PromptSession, feed scripted inputs,
capture everything printed), with real local-state isolation added since
these tests exercise state-backed slash commands (unlike the two exit-path
tests in that file, which never touch state.json).
"""
import asyncio
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from rich.console import Console

from tamfis_code import state as state_module
from tamfis_code.config import Config
from tamfis_code.runner import TaskOutcome
from tamfis_code.workspace import WorkspaceContext


class _StatePatchMixin:
    def setUp(self):
        self._originals = (state_module.CONFIG_DIR, state_module.STATE_PATH)
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        state_module.CONFIG_DIR = base / ".config"
        state_module.STATE_PATH = base / ".config" / "state.json"

    def tearDown(self):
        state_module.CONFIG_DIR, state_module.STATE_PATH = self._originals
        self.tmp.cleanup()


def _run(scripted_inputs, *, session_id=1, workspace_root="/tmp/fake-workspace", config=None):
    buf = io.StringIO()
    fake_console = Console(file=buf, no_color=True, width=200)
    workspace = WorkspaceContext(session_id=session_id, workspace_root=workspace_root)
    config = config or Config()
    prompt_mock = AsyncMock(side_effect=scripted_inputs)

    with patch("tamfis_code.interactive.Console", return_value=fake_console), \
            patch("tamfis_code.interactive.PromptSession") as session_cls, \
            patch("tamfis_code.interactive.print_banner"):
        session_cls.return_value.prompt_async = prompt_mock
        asyncio.run(run_interactive_import(client=None, config=config, workspace=workspace))

    return buf.getvalue()


def run_interactive_import(**kwargs):
    from tamfis_code.interactive import run_interactive
    return run_interactive(**kwargs)


class StandaloneStatusAndToolsTests(_StatePatchMixin, unittest.TestCase):
    def test_status_shows_standalone_not_server_id(self):
        output = _run(["/status", EOFError()])
        self.assertIn("standalone, local session", output)
        self.assertNotIn("server_id=", output)

    def test_tools_shows_real_local_tool_names(self):
        output = _run(["/tools", EOFError()])
        self.assertIn("edit_file", output)
        self.assertIn("execute_command", output)
        self.assertNotIn("glob_files", output)  # old remote-only tool name

    def test_help_mentions_standalone_limitations(self):
        output = _run(["/help", EOFError()])
        self.assertIn("Standalone mode", output)
        self.assertIn("/pty", output)


class StandaloneDiffsAndRevertTests(_StatePatchMixin, unittest.TestCase):
    def test_diffs_empty_session_prints_dim_message(self):
        output = _run(["/diffs", EOFError()])
        self.assertIn("No file mutations recorded", output)

    def test_diffs_and_diff_and_revert_use_local_ledger(self):
        with tempfile.TemporaryDirectory() as ws:
            target = Path(ws) / "app.py"
            target.write_text("new\n")
            from tamfis_code.safety import record_mutation
            entry = record_mutation(1, path=str(target), operation="update", original_content="old\n", new_content="new\n")

            output = _run(["/diffs", f"/diff {entry['mutation_id']}", f"/revert {entry['mutation_id']}", EOFError()], workspace_root=ws)
            self.assertIn(entry["mutation_id"], output)
            self.assertIn("Reverted", output)
            self.assertEqual(target.read_text(), "old\n")

    def test_revert_unknown_mutation_id_shows_error_not_crash(self):
        output = _run(["/revert mut_doesnotexist", EOFError()])
        self.assertIn("No recorded mutation", output)


class StandaloneAgentsAndResumeTests(_StatePatchMixin, unittest.TestCase):
    def test_agents_lists_local_sessions_not_remote_call(self):
        state_module.save_session_state(1, workspace_root="/tmp/fake-workspace")
        state_module.save_session_state(2, workspace_root="/tmp/other-project")
        output = _run(["/agents", EOFError()])
        self.assertIn("/tmp/fake-workspace", output)
        self.assertIn("/tmp/other-project", output)

    def test_resume_with_no_other_sessions(self):
        output = _run(["/resume", EOFError()])
        self.assertIn("No other sessions to resume", output)

    def test_resume_unknown_session_id_shows_error(self):
        output = _run(["/resume 999", EOFError()])
        self.assertIn("No known local session 999", output)


class StandalonePtyAndDoctorTests(_StatePatchMixin, unittest.TestCase):
    def test_pty_unavailable_in_standalone_mode(self):
        output = _run(["/pty start", EOFError()])
        self.assertIn("require --remote", output)

    def test_doctor_shows_provider_status_table(self):
        output = _run(["/doctor", EOFError()])
        self.assertIn("PROVIDER", output)
        self.assertIn("CONFIGURED", output)


class StandaloneAiDispatchTests(_StatePatchMixin, unittest.TestCase):
    def test_natural_language_objective_calls_local_agent_loop(self):
        fake_outcome = TaskOutcome(status="completed", summary="Done.")
        with patch("tamfis_code.interactive.run_local_agent_turn", new=AsyncMock(return_value=fake_outcome)):
            output = _run(["fix the bug", EOFError()])
        self.assertIn("Done.", output)

    def test_retry_with_no_previous_turn(self):
        output = _run(["/retry", EOFError()])
        self.assertIn("No previous turn", output)

    def test_retry_resends_last_objective(self):
        fake_outcome = TaskOutcome(status="completed", summary="Done again.")
        with patch("tamfis_code.interactive.run_local_agent_turn", new=AsyncMock(return_value=fake_outcome)) as mock_turn:
            output = _run(["fix the bug", "/retry", EOFError()])
        self.assertEqual(mock_turn.call_count, 2)
        self.assertIn("Done again.", output)

    def test_plan_mode_saves_plan_on_completion(self):
        fake_outcome = TaskOutcome(status="completed", summary="# Plan\n1. Do X")
        with patch("tamfis_code.interactive.run_local_agent_turn", new=AsyncMock(return_value=fake_outcome)):
            output = _run(["/plan add a feature", EOFError()])
        self.assertIn("Plan saved", output)
        state = state_module.get_session_state(1)
        self.assertEqual(len(state.saved_plans), 1)

    def test_provider_error_does_not_crash_the_repl(self):
        with patch("tamfis_code.interactive.run_local_agent_turn", new=AsyncMock(side_effect=RuntimeError("boom"))):
            output = _run(["fix the bug", "/status", EOFError()])
        self.assertIn("boom", output)
        self.assertIn("standalone, local session", output)  # REPL kept running after the error


class UnsupportedProviderTests(_StatePatchMixin, unittest.TestCase):
    def test_invalid_provider_reported_cleanly_not_a_traceback(self):
        buf = io.StringIO()
        fake_console = Console(file=buf, no_color=True, width=200)
        workspace = WorkspaceContext(session_id=1, workspace_root="/tmp/fake-workspace")
        config = Config()
        with patch("tamfis_code.interactive.Console", return_value=fake_console), \
                patch("tamfis_code.interactive.print_banner"):
            from tamfis_code.interactive import run_interactive
            asyncio.run(run_interactive(client=None, config=config, workspace=workspace, provider="not-a-real-provider"))
        self.assertIn("Unknown local provider", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
