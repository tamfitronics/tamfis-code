"""The always-granted scratch directory (workspace.scratch_root, wired in
via runner_local.py's _apply_mcp_task_scope) -- a local agent turn must be
able to write into it without a workspace-boundary PermissionError, the
same way Claude Code's own scratchpad is always available without an
explicit per-task grant.
"""
from __future__ import annotations

import asyncio
import json
import tempfile
import unittest

from tamfis_code.providers import ProviderType
from tamfis_code.runner_local import run_local_agent_turn
from tamfis_code.workspace import scratch_root

from test_reasoning_plan import (
    _FakeClient,
    _FakeManager,
    _RecordingRenderer,
    _StatePatchMixin,
    _chunk,
    _delta,
    _tool_call_delta,
)


class ScratchRootGrantTests(_StatePatchMixin, unittest.TestCase):
    def _console(self):
        from io import StringIO
        from rich.console import Console
        return Console(file=StringIO(), no_color=True, width=200)

    def test_write_file_into_the_scratch_root_succeeds_without_a_grant(self):
        session_id = 90200
        scratch = scratch_root(session_id)
        target = scratch / "scratch_test.py"
        try:
            with tempfile.TemporaryDirectory() as ws:
                args = json.dumps({"path": str(target), "content": "print('hi')\n"})
                rounds = [
                    [_chunk(_delta(tool_calls=[
                        _tool_call_delta(0, call_id="call_1", name="write_file", arguments=args)
                    ]))],
                    [_chunk(_delta(content="Wrote the scratch file."))],
                ]
                client = _FakeClient(rounds)
                manager = _FakeManager(client)
                renderer = _RecordingRenderer()

                outcome = asyncio.run(run_local_agent_turn(
                    manager, ProviderType.NVIDIA, None,
                    [{"role": "user", "content": "write a throwaway script to /tmp and run it"}],
                    self._console(), renderer,
                    workspace_root=ws, session_id=session_id, approval_policy="auto", interactive=False,
                ))

                self.assertEqual(outcome.status, "completed")
                tool_outputs = [
                    e["payload"]["result"] for e in renderer.events
                    if e["event_type"] == "tool_output"
                ]
                self.assertTrue(
                    any(
                        isinstance(r, dict) and not r.get("error") and "denied" not in str(r).lower()
                        and "outside the workspace" not in str(r).lower()
                        for r in tool_outputs
                    ),
                    f"expected the scratch-root write to succeed, got: {tool_outputs}",
                )
                self.assertTrue(target.is_file())
                self.assertEqual(target.read_text(), "print('hi')\n")
        finally:
            if target.exists():
                target.unlink()
            if scratch.exists():
                scratch.rmdir()


if __name__ == "__main__":
    unittest.main()
