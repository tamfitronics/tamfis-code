"""Auto mode must not present an approval it is not asking for.

Live report (2026-09-21): in mode "auto" a write printed a boxed "Approval required -- risk: medium"
card and then wrote the file at once, which reads as the CLI being blocked on the user.
"""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from io import StringIO

from rich.console import Console

from tamfis_code.providers import ProviderType
from tamfis_code.render import StreamRenderer
from tamfis_code.runner_local import run_local_agent_turn
from tamfis_code.runtime.progress import ExecState, ProgressTracker

from test_reasoning_plan import (
    _FakeClient, _FakeManager, _RecordingRenderer, _StatePatchMixin, _chunk, _delta, _tool_call_delta,
)


def _renderer():
    console = Console(file=StringIO(), no_color=True, width=120)
    return StreamRenderer(console), console


class RenderTests(unittest.TestCase):
    def test_auto_approval_prints_no_card_and_no_repeat_of_the_command(self):
        renderer, console = _renderer()
        renderer.handle_event({"event_type": "approval_auto", "payload": {
            "command": "write_file(path='/home/x/check.txt')", "risk_level": "medium", "diff": None}})
        out = console.file.getvalue()
        self.assertEqual(out.strip(), "")          # the call's own block names it; no "Approval required" card
        self.assertNotIn("Approval required", out)

    def test_auto_approval_still_shows_the_proposed_diff(self):
        renderer, console = _renderer()
        renderer.handle_event({"event_type": "approval_auto", "payload": {
            "command": "write_file(path='x')", "risk_level": "medium",
            "diff": "--- a/x\n+++ b/x\n@@ -0,0 +1 @@\n+hello\n"}})
        out = console.file.getvalue()
        self.assertIn("Proposed change", out)
        self.assertIn("+hello", out)
        self.assertNotIn("Approval required", out)

    def test_a_real_prompt_still_gets_the_full_card(self):
        renderer, console = _renderer()
        renderer.handle_event({"event_type": "approval_required", "payload": {
            "command": "rm -rf build", "risk_level": "high", "reason": "delete", "diff": None}})
        out = console.file.getvalue()
        self.assertIn("Approval required", out)
        self.assertIn("Working directory", out)

    def test_progress_never_reports_waiting_for_the_user_on_an_auto_approval(self):
        tracker = ProgressTracker()
        tracker.observe("approval_auto", {})
        self.assertNotEqual(tracker.state(), ExecState.WAITING_USER)
        tracker.observe("approval_required", {})
        self.assertEqual(tracker.state(), ExecState.WAITING_USER)


class RunnerTests(_StatePatchMixin, unittest.TestCase):
    def _run(self, policy):
        with tempfile.TemporaryDirectory() as ws:
            target = Path(ws) / "auto_display.py"      # inside the workspace: needs an approval decision
            args = json.dumps({"path": str(target), "content": "print('hi')\n"})
            plan = json.dumps({"steps": ["Write the script", "Run it"]})
            rounds = [
                [_chunk(_delta(content=plan))],
                [_chunk(_delta(tool_calls=[_tool_call_delta(0, call_id="call_1", name="write_file", arguments=args)]))],
                [_chunk(_delta(tool_calls=[_tool_call_delta(
                    0, call_id="call_2", name="execute_command",
                    arguments=json.dumps({"command": f"python3 {target}"}))]))],
                [_chunk(_delta(content="Done."))],
            ]
            renderer = _RecordingRenderer()
            asyncio.run(run_local_agent_turn(
                _FakeManager(_FakeClient(rounds)), ProviderType.NVIDIA, None,
                [{"role": "user", "content": "write a throwaway script to /tmp and run it"}],
                Console(file=StringIO(), no_color=True, width=200), renderer,
                workspace_root=ws, session_id=90211, approval_policy=policy, interactive=False,
            ))
            return [e["event_type"] for e in renderer.events]

    def test_auto_mode_announces_instead_of_requesting(self):
        events = self._run("auto")
        self.assertIn("approval_auto", events)
        self.assertNotIn("approval_required", events)

    def test_ask_mode_still_requests(self):
        events = self._run("ask")
        self.assertIn("approval_required", events)
        self.assertNotIn("approval_auto", events)


if __name__ == "__main__":
    unittest.main()
