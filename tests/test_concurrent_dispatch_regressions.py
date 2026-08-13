"""Regression tests for runner_local.py's concurrent tool-call dispatch
(independent calls in one round run via asyncio.gather instead of strictly
one at a time -- see _dispatch_queue/_conflicts/_flush_dispatch_queue).
"""
from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

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


class ConcurrentWriteFileMutationEventTests(_StatePatchMixin, unittest.TestCase):
    def test_two_different_path_writes_in_one_round_report_their_own_mutation(self):
        # Regression: two different-path write_file calls in the same round
        # are dispatched concurrently (no shared path -> not flagged by
        # _conflicts). Both append to the shared session-state
        # modified_files ledger before either call's postprocessing reads
        # it back, so grabbing modified_files[-1] could report call A's
        # file_mutation event using call B's path/diff stats. Each event
        # must match its own call.
        with tempfile.TemporaryDirectory() as workspace:
            root = Path(workspace)
            write_a = json.dumps({"path": "a.py", "content": "print('a')\n"})
            write_b = json.dumps({"path": "b.py", "content": "print('b')\n"})
            verify_args = json.dumps({"command": "true"})
            rounds = [
                [_chunk(_delta(tool_calls=[
                    _tool_call_delta(0, call_id="wa", name="write_file", arguments=write_a),
                    _tool_call_delta(1, call_id="wb", name="write_file", arguments=write_b),
                ]))],
                [_chunk(_delta(tool_calls=[
                    _tool_call_delta(0, call_id="verify", name="execute_command", arguments=verify_args),
                ]))],
                [_chunk(_delta(content="Wrote both files."), finish_reason="stop")],
            ]
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            with patch("tamfis_code.runner_local.should_plan", return_value=False), \
                 patch("tamfis_code.runner_local.detect_validation_commands", return_value=[]):
                outcome = asyncio.run(run_local_agent_turn(
                    manager,
                    ProviderType.NVIDIA,
                    None,
                    [{"role": "user", "content": "add a.py and b.py"}],
                    Console(file=StringIO(), no_color=True, width=200),
                    renderer,
                    workspace_root=str(root),
                    session_id=1,
                    approval_policy="auto",
                    interactive=False,
                ))

        self.assertEqual(outcome.status, "completed")
        mutation_events = [e for e in renderer.events if e.get("event_type") == "file_mutation"]
        self.assertEqual(len(mutation_events), 2)
        reported_paths = {Path(e["payload"]["path"]).name for e in mutation_events}
        self.assertEqual(reported_paths, {"a.py", "b.py"})
        # Each event's own diff stats belong to its own file, not
        # whichever call happened to append to the ledger last.
        for event in mutation_events:
            name = Path(event["payload"]["path"]).name
            self.assertEqual(event["payload"]["lines_added"], 1, name)


if __name__ == "__main__":
    unittest.main()
