"""runner_local.py's failure-streak stall guard (consecutive_failed_rounds /
FAILURE_DIAGNOSIS_CORRECTION).

Distinct from the pre-existing consecutive_identical_rounds/_is_cycling
guards: those catch the model retrying the SAME (or a short repeating
cycle of) call. This catches the opposite shape of stall -- a *different*
attempt each round, none of which land -- which neither existing guard
would ever trip since the calls are never identical or cycling.
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


class FailureDiagnosisNudgeTests(_StatePatchMixin, unittest.TestCase):
    def _console(self):
        from io import StringIO
        from rich.console import Console
        return Console(file=StringIO(), no_color=True, width=200)

    def _failing_read_round(self, index):
        # A different missing path each round -- varied attempts, never an
        # identical or cycling call -- so this guard is the only one that
        # can catch it.
        args = json.dumps({"path": f"does_not_exist_{index}.py"})
        return [_chunk(_delta(tool_calls=[
            _tool_call_delta(0, call_id=f"call_{index}", name="read_file", arguments=args)
        ]))]

    def test_nudges_after_three_different_failed_attempts_in_a_row(self):
        with tempfile.TemporaryDirectory() as ws:
            rounds = [self._failing_read_round(i) for i in range(3)]
            rounds.append([_chunk(_delta(content="Giving up after repeated failures."))])
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None,
                [{"role": "user", "content": "read a file that doesn't exist, three different ways"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
            ))

            diagnostics = [
                str(e["payload"].get("content"))
                for e in renderer.events
                if e["event_type"] == "diagnostics"
            ]
            self.assertTrue(
                any("varied attempts in a row have all failed" in d for d in diagnostics),
                f"expected a failure-diagnosis nudge diagnostic, got: {diagnostics}",
            )
            # The correction message itself must have actually been sent to
            # the model, not just logged as a diagnostic.
            self.assertTrue(
                any(
                    m.get("role") == "system" and "Stop trying another fix blind" in str(m.get("content"))
                    for call in client.calls for m in call.get("messages", [])
                ),
                "expected FAILURE_DIAGNOSIS_CORRECTION to be appended to working_messages",
            )

    def test_does_not_nudge_when_failures_are_not_consecutive(self):
        with tempfile.TemporaryDirectory() as ws:
            real_file = Path(ws) / "real.py"
            real_file.write_text("# real\n")
            success_args = json.dumps({"path": str(real_file)})

            rounds = [
                self._failing_read_round(0),
                self._failing_read_round(1),
                [_chunk(_delta(tool_calls=[
                    _tool_call_delta(0, call_id="call_ok", name="read_file", arguments=success_args)
                ]))],
                self._failing_read_round(2),
                self._failing_read_round(3),
                [_chunk(_delta(content="Done."))],
            ]
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None,
                [{"role": "user", "content": "mixed attempts"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
            ))

            diagnostics = [
                str(e["payload"].get("content"))
                for e in renderer.events
                if e["event_type"] == "diagnostics"
            ]
            self.assertFalse(
                any("varied attempts in a row have all failed" in d for d in diagnostics),
                f"a successful round in between should reset the streak, got: {diagnostics}",
            )


if __name__ == "__main__":
    unittest.main()
