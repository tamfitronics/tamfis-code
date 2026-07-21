"""Tests for the gated plan-mode "execute now?" checkpoint -- a real
Claude-Code-Plan-Mode-style structural gate between "here's the proposed
plan" and any tool actually mutating the workspace. Before this, /plan
mode produced a read-only plan and saved it, but nothing ever asked the
user whether to execute it -- they had to notice the "run /execute-plan"
hint and type a second, separate command themselves.
"""
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from rich.console import Console

from tamfis_code import state as state_module
from tamfis_code.config import Config
from tamfis_code.interactive import run_interactive
from tamfis_code.runner import TaskOutcome
from tamfis_code.workspace import WorkspaceContext


def _run(scripted_inputs, *, console_input_answer=None, run_local_agent_turn_summaries=None):
    """Runs run_interactive with prompt_async yielding scripted_inputs in
    turn, a forced-terminal console whose .input() returns
    console_input_answer, and run_local_agent_turn stubbed to return each
    of run_local_agent_turn_summaries in turn (repeating the last one if
    exhausted). Returns (console_output, run_local_agent_turn_call_count).
    """
    buf = io.StringIO()
    fake_console = Console(file=buf, no_color=True, width=200, force_terminal=True)
    fake_console.input = lambda *a, **k: console_input_answer

    workspace = WorkspaceContext(session_id=1, server_id=1, workspace_root="/tmp/fake-workspace")
    config = Config()

    prompt_mock = AsyncMock(side_effect=scripted_inputs)
    summaries = list(run_local_agent_turn_summaries or ["a plan summary"])
    call_count = {"n": 0}

    async def fake_run_local_agent_turn(*args, **kwargs):
        index = min(call_count["n"], len(summaries) - 1)
        call_count["n"] += 1
        return TaskOutcome(status="completed", summary=summaries[index])

    with patch("tamfis_code.interactive.Console", return_value=fake_console), \
            patch("tamfis_code.interactive.PromptSession") as session_cls, \
            patch("tamfis_code.interactive.print_banner"), \
            patch("tamfis_code.interactive.run_local_agent_turn", new=fake_run_local_agent_turn), \
            patch("tamfis_code.interactive.LiveInputListener") as live_input_cls:
        session_cls.return_value.prompt_async = prompt_mock
        live_input_cls.return_value.start = lambda: None
        live_input_cls.return_value.stop = lambda: None
        import asyncio
        asyncio.run(run_interactive(client=None, config=config, workspace=workspace))

    return buf.getvalue(), call_count["n"]


class PlanModeGateTests(unittest.TestCase):
    def setUp(self):
        self._originals = (state_module.CONFIG_DIR, state_module.STATE_PATH)
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        state_module.CONFIG_DIR = base / ".config"
        state_module.STATE_PATH = base / ".config" / "state.json"

    def tearDown(self):
        state_module.CONFIG_DIR, state_module.STATE_PATH = self._originals
        self.tmp.cleanup()

    def test_plan_saved_and_approving_triggers_real_execution(self):
        output, calls = _run(
            ["/plan add a helper function", EOFError()],
            console_input_answer="y",
            run_local_agent_turn_summaries=["proposed plan text", "executed the plan"],
        )
        self.assertIn("Plan saved", output)
        # One call to generate the plan (read-only), a second real call to
        # actually execute it -- the gate must not just re-print the plan.
        self.assertEqual(calls, 2)
        state = state_module.get_session_state(1)
        self.assertEqual(len(state.saved_plans), 1)
        self.assertEqual(state.saved_plans[0]["status"], "completed")

    def test_declining_leaves_the_plan_saved_but_not_executed(self):
        output, calls = _run(
            ["/plan add a helper function", EOFError()],
            console_input_answer="n",
            run_local_agent_turn_summaries=["proposed plan text"],
        )
        self.assertIn("Plan saved", output)
        self.assertIn("Not executed", output)
        # Only the plan-generation call happened -- nothing was executed.
        self.assertEqual(calls, 1)
        state = state_module.get_session_state(1)
        self.assertEqual(len(state.saved_plans), 1)
        self.assertEqual(state.saved_plans[0]["status"], "ready")

    def test_non_plan_mode_turns_are_never_gated(self):
        # An ordinary coding-mode objective must not trigger any plan-save
        # or approval prompt at all -- the gate is plan-mode-only.
        output, calls = _run(
            ["fix the bug in app.py", EOFError()],
            console_input_answer="y",
            run_local_agent_turn_summaries=["fixed it"],
        )
        self.assertNotIn("Plan saved", output)
        self.assertEqual(calls, 1)
        state = state_module.get_session_state(1)
        self.assertEqual(state.saved_plans, [])


if __name__ == "__main__":
    unittest.main()
