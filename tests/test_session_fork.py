import asyncio
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from click.testing import CliRunner
from rich.console import Console

from tamfis_code import state as state_module
from tamfis_code.cli import cli
from tamfis_code.config import Config
from tamfis_code.interactive import run_interactive
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


def _run(scripted_inputs):
    output = io.StringIO()
    console = Console(file=output, no_color=True, width=200)
    workspace = WorkspaceContext(session_id=1, workspace_root="/tmp/fake-workspace")
    prompt_mock = AsyncMock(side_effect=scripted_inputs)
    with patch("tamfis_code.interactive.Console", return_value=console), patch(
        "tamfis_code.interactive.PromptSession"
    ) as session_cls, patch("tamfis_code.interactive.print_banner"):
        session_cls.return_value.prompt_async = prompt_mock
        asyncio.run(run_interactive(None, Config(), workspace))
    return output.getvalue()


class SessionForkStateTests(_StatePatchMixin, unittest.TestCase):
    def test_fork_copies_context_but_resets_live_execution_state(self):
        state_module.save_session_state(
            4,
            workspace_root="/tmp/project",
            conversation_history=[
                {"role": "user", "content": "inspect auth"},
                {"role": "assistant", "content": "auth uses OAuth"},
            ],
            conversation_summary="Mapped auth",
            active_task={"objective": "still running"},
            execution_status="running",
            current_phase="execute",
            last_task_id="task-old",
            queued_user_instructions=[{"id": "queued-old", "text": "change direction"}],
            turn_checkpoint={"status": "running"},
        )

        forked = state_module.fork_session_state(4)

        self.assertEqual(forked.session_id, 5)
        self.assertEqual(forked.forked_from_session_id, 4)
        self.assertEqual(forked.conversation_summary, "Mapped auth")
        self.assertEqual(len(forked.conversation_history), 2)
        self.assertIsNone(forked.active_task)
        self.assertIsNone(forked.last_task_id)
        self.assertIsNone(forked.turn_checkpoint)
        self.assertEqual(forked.execution_status, "idle")
        self.assertEqual(forked.queued_user_instructions, [])

        forked.conversation_history[0]["content"] = "changed only in memory"
        self.assertEqual(
            state_module.get_session_state(4).conversation_history[0]["content"],
            "inspect auth",
        )

    def test_unknown_source_is_rejected_without_creating_a_session(self):
        with self.assertRaisesRegex(ValueError, "No known local session 99"):
            state_module.fork_session_state(99)
        self.assertEqual(state_module.all_known_session_ids(), [])


class SessionForkSurfaceTests(_StatePatchMixin, unittest.TestCase):
    def test_cli_fork_creates_a_resumable_branch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = str(Path(tmp).resolve())
            state_module.save_session_state(1, workspace_root=root, conversation_summary="source")
            result = CliRunner().invoke(cli, ["--cwd", root, "fork", "1"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("1 -> 2", result.output)
        self.assertEqual(state_module.get_session_state(2).forked_from_session_id, 1)

    def test_interactive_fork_switches_to_the_new_branch(self):
        state_module.save_session_state(
            1,
            workspace_root="/tmp/fake-workspace",
            conversation_history=[{"role": "user", "content": "source context"}],
        )
        outcome = TaskOutcome(status="completed", summary="branch answer")
        with patch(
            "tamfis_code.interactive.run_local_agent_turn",
            new=AsyncMock(return_value=outcome),
        ) as turn:
            output = _run(["/fork", "continue differently", EOFError()])

        self.assertIn("Forked session 1 -> 2", output)
        self.assertEqual(turn.await_args.kwargs["session_id"], 2)
        self.assertEqual(turn.await_args.args[3], [
            {"role": "user", "content": "source context"},
            {"role": "user", "content": "continue differently"},
        ])

    def test_resume_replaces_in_memory_history_instead_of_leaking_source(self):
        state_module.save_session_state(
            1,
            workspace_root="/tmp/fake-workspace",
            conversation_history=[{"role": "user", "content": "source-only context"}],
        )
        state_module.save_session_state(
            2,
            workspace_root="/tmp/other-project",
            conversation_history=[{"role": "assistant", "content": "target-only context"}],
        )
        outcome = TaskOutcome(status="completed", summary="done")
        with patch(
            "tamfis_code.interactive.run_local_agent_turn",
            new=AsyncMock(return_value=outcome),
        ) as turn:
            _run(["/resume 2", "new target turn", EOFError()])

        self.assertEqual(turn.await_args.kwargs["session_id"], 2)
        messages = turn.await_args.args[3]
        self.assertEqual(messages, [
            {"role": "assistant", "content": "target-only context"},
            {"role": "user", "content": "new target turn"},
        ])
