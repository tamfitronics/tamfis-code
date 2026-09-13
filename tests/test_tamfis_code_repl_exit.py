#!/usr/bin/env python3
"""Regression tests for two real, live-reported tamfis-code gaps:

1. Ctrl+C at the idle prompt did nothing except silently redraw the
   prompt (`except KeyboardInterrupt: continue`) -- there was no
   documented or discoverable way to exit except Ctrl+D or typing /exit,
   which reads as "the CLI is stuck" to a user used to Ctrl+C exiting a
   terminal program. Ctrl+C while an AI task/command is actively
   streaming is a separate, already-correct code path (runner.py's
   _install_sigint_watcher cancels just that task) -- untouched by this.

2. Typing "/" alone fell all the way through the command dispatch (every
   check requires an exact match or a prefix with a trailing space and
   content) into parse_intent(), which submitted the bare "/" character
   itself as a one-character AI task objective instead of showing the
   command list the way typing "/" alone is expected to behave.
"""
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from rich.console import Console

from tamfis_code import state as state_module
from tamfis_code.config import Config
from tamfis_code.interactive import run_interactive, HELP_TEXT
from tamfis_code.workspace import WorkspaceContext


def _run(scripted_inputs, workspace_root="/tmp/fake-workspace"):
    """Runs run_interactive with prompt_async yielding each of
    scripted_inputs in turn (raising it directly if it's an Exception),
    and returns everything printed to the console."""
    buf = io.StringIO()
    fake_console = Console(file=buf, no_color=True, width=200)

    workspace = WorkspaceContext(session_id=1, server_id=1, workspace_root=workspace_root)
    config = Config()

    prompt_mock = AsyncMock(side_effect=scripted_inputs)

    with patch("tamfis_code.interactive.Console", return_value=fake_console), \
         patch("tamfis_code.interactive.PromptSession") as session_cls, \
         patch("tamfis_code.interactive.print_banner"):
        session_cls.return_value.prompt_async = prompt_mock
        import asyncio
        asyncio.run(run_interactive(client=None, config=config, workspace=workspace))

    return buf.getvalue()


class ReplExitTests(unittest.TestCase):
    def test_ctrl_c_at_idle_prompt_exits_the_repl(self):
        # KeyboardInterrupt on the very first prompt -- run_interactive
        # must return (the process would exit) rather than looping forever.
        output = _run([KeyboardInterrupt()])
        # No assertion needed beyond "this returns at all" -- the old
        # `continue` behavior would hang this test forever since the mock
        # only has one scripted response and AsyncMock would raise
        # StopAsyncIteration/StopIteration on a second call instead of
        # exiting cleanly, which is exactly the bug being guarded against.
        self.assertIsInstance(output, str)

    def test_bare_slash_shows_the_command_list(self):
        output = _run(["/", EOFError()])
        self.assertIn("show this help", output)  # a line from HELP_TEXT
        self.assertNotIn("task", output.lower().split("show this help")[0][-50:])


class SessionStartEndHookTests(unittest.TestCase):
    """Claude-Code-parity addition: session_start/session_end fire exactly
    once per REPL process lifetime, regardless of which of the loop's many
    exit paths (Ctrl+C, Ctrl+D, /exit, an uncaught exception) is taken --
    proven here via the same run_interactive harness ReplExitTests already
    uses for two different real exit paths, rather than only unit-testing
    hooks.py's run_session_start_hooks/run_session_end_hooks in isolation.
    """

    def test_session_start_hook_output_is_shown_and_session_end_fires_on_ctrl_c(self):
        with tempfile.TemporaryDirectory() as ws:
            marker = Path(ws) / "ended.txt"
            hooks_dir = Path(ws) / ".tamfis"
            hooks_dir.mkdir()
            (hooks_dir / "hooks.toml").write_text(
                '[[session_start]]\n'
                'command = "echo \\"project context loaded\\" 1>&2"\n'
                '\n'
                '[[session_end]]\n'
                f'command = "cat > {marker}"\n'
            )
            output = _run([KeyboardInterrupt()], workspace_root=ws)
            self.assertIn("project context loaded", output)
            self.assertTrue(marker.is_file(), "session_end hook never ran on the Ctrl+C exit path")

    def test_session_end_fires_on_the_eof_exit_path_too(self):
        # A different exit path than Ctrl+C -- proves session_end's
        # try/finally isn't accidentally tied to one specific exception.
        with tempfile.TemporaryDirectory() as ws:
            marker = Path(ws) / "ended.txt"
            hooks_dir = Path(ws) / ".tamfis"
            hooks_dir.mkdir()
            (hooks_dir / "hooks.toml").write_text(
                f'[[session_end]]\ncommand = "cat > {marker}"\n'
            )
            _run([EOFError()], workspace_root=ws)
            self.assertTrue(marker.is_file(), "session_end hook never ran on the EOF exit path")

    def test_no_configured_hooks_is_a_silent_noop(self):
        with tempfile.TemporaryDirectory() as ws:
            output = _run([KeyboardInterrupt()], workspace_root=ws)
        self.assertIsInstance(output, str)


class PreCompactHookTests(unittest.TestCase):
    """Claude-Code-parity addition: a pre_compact hook fires when /compact
    runs, and its output survives the fold into conversation_summary --
    proven here via a real /compact command through the actual REPL, not
    a direct call to compact_session_thread."""

    def setUp(self):
        self._originals = (state_module.CONFIG_DIR, state_module.STATE_PATH)
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        state_module.CONFIG_DIR = base / ".config"
        state_module.STATE_PATH = base / ".config" / "state.json"
        state_module._STATE_CACHE = None
        state_module._STATE_CACHE_KEY = None

    def tearDown(self):
        state_module.CONFIG_DIR, state_module.STATE_PATH = self._originals
        state_module._STATE_CACHE = None
        state_module._STATE_CACHE_KEY = None
        self._tmp.cleanup()

    def test_pre_compact_hook_output_survives_the_fold(self):
        with tempfile.TemporaryDirectory() as ws:
            hooks_dir = Path(ws) / ".tamfis"
            hooks_dir.mkdir()
            (hooks_dir / "hooks.toml").write_text(
                '[[pre_compact]]\ncommand = "echo \\"remember the auth refactor\\" 1>&2"\n'
            )
            history = []
            for i in range(8):
                history.append({"role": "user", "content": f"obj {i}"})
                history.append({"role": "assistant", "content": f"ans {i}"})
            state_module.save_session_state(1, conversation_history=history)

            _run(["/compact", EOFError()], workspace_root=ws)

            state = state_module.get_session_state(1)
            self.assertIn("remember the auth refactor", state.conversation_summary)


if __name__ == "__main__":
    unittest.main()
