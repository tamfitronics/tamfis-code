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
import unittest
from unittest.mock import AsyncMock, patch

from rich.console import Console

from tamfis_code.config import Config
from tamfis_code.interactive import run_interactive, HELP_TEXT
from tamfis_code.workspace import WorkspaceContext


def _run(scripted_inputs):
    """Runs run_interactive with prompt_async yielding each of
    scripted_inputs in turn (raising it directly if it's an Exception),
    and returns everything printed to the console."""
    buf = io.StringIO()
    fake_console = Console(file=buf, no_color=True, width=200)

    workspace = WorkspaceContext(session_id=1, server_id=1, workspace_root="/tmp/fake-workspace")
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


if __name__ == "__main__":
    unittest.main()
