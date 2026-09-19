"""The permission race as the live agent loop actually reaches it.

tests/test_permission_race.py pins the race's own semantics; these tests drive
`run_local_agent_turn` (the real call site inside runner_local's tool dispatch)
with a scripted provider, which is the only way to prove the swap from
"prompt then decide" to "race then decide" did not change what a normal,
already-approved write does -- and that a catastrophic command is stopped
without ever reaching the tool dispatcher.
"""
from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from tamfis_code.mcp import MCPServer
from tamfis_code.providers import ProviderType
from tamfis_code.runner_local import run_local_agent_turn

from test_reasoning_plan import (
    _FakeClient,
    _FakeManager,
    _RecordingRenderer,
    _StatePatchMixin,
    _chunk,
    _delta,
    _tool_call_delta,
)


class RunnerPermissionRaceTests(_StatePatchMixin, unittest.TestCase):
    def _console(self):
        from io import StringIO
        from rich.console import Console
        return Console(file=StringIO(), no_color=True, width=200)

    def test_an_ordinary_approved_write_still_goes_through(self):
        """The race must be transparent for the overwhelmingly common case: a
        policy-approved, in-workspace write is still approved, still renders
        the same approval card, and still reaches the tool.

        The turn itself is allowed to fail here -- this scripted provider
        offers no verification command, and the runner's own
        mutation-verification gate refuses to call an unverified edit done.
        What this test is about is the decision, not the gate."""
        with tempfile.TemporaryDirectory() as ws:
            target = Path(ws) / "hello.py"
            arguments = json.dumps({"path": str(target), "content": "def add(a, b):\n    return a + b\n"})
            client = _FakeClient([
                [_chunk(_delta(tool_calls=[_tool_call_delta(0, call_id="call_1", name="write_file", arguments=arguments)]))],
                [_chunk(_delta(content="Wrote hello.py."))],
            ])
            renderer = _RecordingRenderer()

            outcome = asyncio.run(run_local_agent_turn(
                _FakeManager(client), ProviderType.NVIDIA, None,
                [{"role": "user", "content": f"create {target} with an add function"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
            ))

            self.assertTrue(target.is_file(), "the approved write must actually happen")
            self.assertIn("def add", target.read_text())
            # The approval card is unchanged: the race speeds the decision up,
            # it does not remove the audit trail.
            self.assertTrue(any(
                event.get("event_type") == "approval_required" for event in renderer.events
            ))
            self.assertTrue(any(
                event.get("event_type") == "tool_output"
                and event.get("payload", {}).get("tool") == "write_file"
                and event.get("payload", {}).get("result", {}).get("success") is True
                for event in renderer.events
            ))  # noqa: E501

    def test_a_catastrophic_command_is_blocked_before_dispatch(self):
        """`rm -rf /` is denied by the static process of the race -- with no
        prompt, in every policy tier, and before the tool dispatcher is ever
        called."""
        with tempfile.TemporaryDirectory() as ws:
            dispatched: list[tuple[str, object]] = []
            original_call_tool = MCPServer.call_tool

            async def recording_call_tool(self, name, arguments=None, **kwargs):
                dispatched.append((name, arguments))
                return await original_call_tool(self, name, arguments or {}, **kwargs)

            MCPServer.call_tool = recording_call_tool
            try:
                arguments = json.dumps({"command": "rm -rf /"})
                client = _FakeClient([
                    [_chunk(_delta(tool_calls=[_tool_call_delta(0, call_id="call_1", name="execute_command", arguments=arguments)]))],
                    [_chunk(_delta(content="That command was blocked."))],
                ])
                renderer = _RecordingRenderer()

                outcome = asyncio.run(run_local_agent_turn(
                    _FakeManager(client), ProviderType.NVIDIA, None,
                    [{"role": "user", "content": "clean up the disk for me"}],
                    self._console(), renderer,
                    workspace_root=ws, session_id=1, approval_policy="full-auto", interactive=False,
                ))
            finally:
                MCPServer.call_tool = original_call_tool

            self.assertEqual([name for name, _ in dispatched if name == "execute_command"], [],
                             "a catastrophic command must never reach the tool dispatcher")
            diagnostics = [
                str(event.get("payload", {}).get("content", ""))
                for event in renderer.events
                if event.get("event_type") == "diagnostics"
            ]
            self.assertTrue(
                any("Blocked before prompting" in text for text in diagnostics),
                f"the block must be reported, not silent: {diagnostics}",
            )
            self.assertEqual(outcome.status, "completed")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
