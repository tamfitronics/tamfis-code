"""Cross-agent terminal contracts.

These tests pin the user-visible behaviors shared by Codex, Claude Code, and
Kimi Code: a discoverable workspace, resumable sessions, automation-friendly
input, and an input editor that remains usable while output streams.
"""

import asyncio
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

from tamfis_code import state as state_module
from tamfis_code.config import Config
from tamfis_code.live_input import LiveInputListener
from tamfis_code.render import StreamRenderer


class _StateIsolation(unittest.TestCase):
    def setUp(self):
        self._originals = (state_module.CONFIG_DIR, state_module.STATE_PATH)
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        state_module.CONFIG_DIR = root / ".config"
        state_module.STATE_PATH = root / ".config" / "state.json"

    def tearDown(self):
        state_module.CONFIG_DIR, state_module.STATE_PATH = self._originals
        self.tmp.cleanup()


def _config() -> Config:
    cfg = Config.__new__(Config)
    cfg.approval_policy = "ask"
    return cfg


class LiveTerminalParityTests(_StateIsolation, unittest.IsolatedAsyncioTestCase):
    async def test_follow_up_editor_accepts_multiple_lines_without_ctrl_y(self):
        renderer = StreamRenderer(Console(file=StringIO(), no_color=True, width=160))
        listener = LiveInputListener(session_id=91, renderer=renderer, cli_config=_config())
        listener._active = True
        calls = 0

        async def prompt(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                listener._active = False
            return ["check tests", "then inspect logs"][calls - 1]

        with patch("prompt_toolkit.PromptSession.prompt_async", side_effect=prompt):
            await listener._input_loop()

        queued = state_module.get_session_state(91).queued_user_instructions
        self.assertEqual([item["text"] for item in queued], ["check tests", "then inspect logs"])
        self.assertEqual([item["classification"] for item in queued], ["follow_up", "follow_up"])
        self.assertEqual(calls, 2)

    async def test_escape_queues_a_cancel_for_the_running_turn(self):
        renderer = StreamRenderer(Console(file=StringIO(), no_color=True, width=160))
        listener = LiveInputListener(session_id=92, renderer=renderer, cli_config=_config())

        listener._enqueue_control("cancel")

        queued = state_module.get_session_state(92).queued_user_instructions
        self.assertEqual(len(queued), 1)
        self.assertEqual(queued[0]["classification"], "cancel")

    async def test_ctrl_c_queues_exit_instead_of_cancel(self):
        renderer = StreamRenderer(Console(file=StringIO(), no_color=True, width=160))
        listener = LiveInputListener(session_id=93, renderer=renderer, cli_config=_config())

        listener._enqueue_control("exit")

        queued = state_module.get_session_state(93).queued_user_instructions
        self.assertEqual(queued[0]["classification"], "exit")

    def test_live_footer_keeps_the_current_phase_visible(self):
        renderer = StreamRenderer(Console(file=StringIO(), no_color=True, width=160))
        renderer.handle_event({"event_type": "reasoning_delta", "payload": {"content": "thinking"}})
        self.assertIn("Thinking", renderer.live_input_status())
        self.assertIn("Thinking through the next step", renderer.live_input_status())

    def test_live_footer_explains_the_current_tool_action(self):
        renderer = StreamRenderer(Console(file=StringIO(), no_color=True, width=160))
        renderer.handle_event({
            "event_type": "tool_call_requested",
            "payload": {"name": "read_file", "arguments": {"path": "src/app.py"}},
        })
        self.assertIn("Reading", renderer.live_input_status())
        self.assertIn("src/app.py", renderer.live_input_status())

    def test_streaming_uses_durable_scrollback_when_editor_is_active(self):
        console = Console(file=StringIO(), no_color=True, width=160, force_terminal=True)
        renderer = StreamRenderer(console)
        renderer.live_input_listener = object()

        renderer.handle_event({"event_type": "assistant_delta", "payload": {"content": "A streamed answer."}})
        renderer._flush_assistant(force=True)

        self.assertIn("A streamed answer.", console.file.getvalue())
        self.assertIsNone(renderer._assistant_live)
        renderer.finish()

    def test_submitted_paste_is_echoed_as_a_user_turn(self):
        console = Console(file=StringIO(), no_color=True, width=160)
        renderer = StreamRenderer(console)

        renderer.handle_event({
            "event_type": "user_message",
            "payload": {"content": "Please inspect this pasted configuration."},
        })

        output = console.file.getvalue()
        self.assertIn("You", output)
        self.assertIn("Please inspect this pasted configuration.", output)


class AutomationAndWorkspaceParityTests(_StateIsolation):
    def test_configured_workspace_roots_are_independent_of_launch_cwd(self):
        from tamfis_code.workspace import resolve_local_workspace

        launch = Path(self.tmp.name) / "launch"
        launch.mkdir()
        state = resolve_local_workspace(launch)

        self.assertEqual(Path(state.workspace_root), launch.resolve())
        saved = state_module.get_session_state(state.session_id)
        self.assertIn(str(launch.resolve()), saved.allowed_workspaces)

    def test_automation_input_contract_has_one_source(self):
        # Shell automation must not silently concatenate an objective, stdin,
        # and prompt file. The Click contract rejects multiple sources.
        from click.testing import CliRunner
        from tamfis_code.cli import cli

        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as prompt_file:
            result = CliRunner().invoke(
                cli, ["ask", "fix it", "--stdin", "--prompt-file", prompt_file.name],
                input="also fix this",
            )
        self.assertEqual(result.exit_code, 2)
        self.assertIn("Provide exactly one objective", result.output)


if __name__ == "__main__":
    unittest.main()
