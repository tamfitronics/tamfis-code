"""runner_local.py's stuck-loop recovery path.

Live-reported (tamfis-code repo, kimi-k2.7-code via ollama_cloud): a model
ran the same execute_command wp-cli cleanup several times in a row across
four real databases (each call actually succeeded), got flagged as stuck,
got one nudge, stayed stuck, and the tools-disabled recovery completion
also came back with empty content -- so the whole turn hard-failed with
"nothing further to try", discarding four real, successful actions. The
fix: when the tools-disabled recovery answer is also empty, reconstruct a
plain-text summary directly from the tool calls/results already recorded
this turn instead of failing outright.
"""
from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

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


class StuckLoopRecoveryTests(_StatePatchMixin, unittest.TestCase):
    def _console(self):
        from io import StringIO
        from rich.console import Console
        return Console(file=StringIO(), no_color=True, width=200)

    def _read_round(self, index, path):
        args = json.dumps({"path": str(path)})
        return [_chunk(_delta(tool_calls=[
            _tool_call_delta(0, call_id=f"call_{index}", name="read_file", arguments=args)
        ]))]

    def test_empty_recovery_answer_falls_back_to_a_reconstructed_summary(self):
        with tempfile.TemporaryDirectory() as ws:
            path = Path(ws) / "file_0.py"
            path.write_text("# real content\n")

            # 5 identical read_file rounds: rounds 0-1 execute for real,
            # round 2 trips the stuck guard and is refused (nudge given),
            # round 3 executes for real again, round 4 trips the guard a
            # second time -- nudge budget (1) is exhausted, so this goes
            # straight to the tools-disabled recovery completion, which
            # returns an empty chunk to simulate the reported failure.
            rounds = [self._read_round(i, path) for i in range(5)]
            rounds.append([_chunk(_delta(content=""))])
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None,
                [{"role": "user", "content": "read file_0.py repeatedly"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
            ))

            self.assertEqual(outcome.status, "completed")
            self.assertIn("read_file", outcome.summary)
            self.assertIn("done", outcome.summary)
            diagnostics = [
                str(e["payload"].get("content"))
                for e in renderer.events
                if e["event_type"] == "diagnostics"
            ]
            self.assertTrue(any("reconstructing a summary" in d for d in diagnostics))


if __name__ == "__main__":
    unittest.main()
