"""runner_local.py's MAX_AGENT_ROUND_EXTENSIONS -- a bounded, automatic
extension of the round-cap safety valve.

Live-reported: a genuinely large, multi-feature objective ran 40 rounds of
real, varied tool calls (never tripping MAX_CONSECUTIVE_IDENTICAL_ROUNDS or
any of the narrated-intent/capitulation/fabricated-result stall guards) and
then hard-failed anyway, discarding real completed work -- see
project_tamfis_code_cli_flag_and_xml_fake_tool_call_gap-adjacent session
history. Since dedicated guards already exist to catch a genuinely stuck
model, reaching the round cap without any of them tripping is evidence of
an oversized task, not a stalled one, and now gets one more full round
window at a time (bounded by MAX_AGENT_ROUND_EXTENSIONS) instead of an
immediate hard failure -- mirroring the existing wall-clock runtime budget
extension in orchestrator/engine.py's guard_tool_call.
"""
from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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


class RoundBudgetExtensionTests(_StatePatchMixin, unittest.TestCase):
    def _console(self):
        from io import StringIO
        from rich.console import Console
        return Console(file=StringIO(), no_color=True, width=200)

    def _read_round(self, index, path):
        args = json.dumps({"path": str(path)})
        return [_chunk(_delta(tool_calls=[
            _tool_call_delta(0, call_id=f"call_{index}", name="read_file", arguments=args)
        ]))]

    def test_extends_the_round_budget_instead_of_failing_then_completes(self):
        with tempfile.TemporaryDirectory() as ws:
            files = []
            for i in range(4):
                path = Path(ws) / f"file_{i}.py"
                path.write_text(f"# file {i}\n")
                files.append(path)

            # max_rounds=2 below, so this needs to survive past round index 2
            # (the 3rd round) to prove an extension was actually granted, not
            # just that the base budget happened to be enough.
            rounds = [self._read_round(i, files[i]) for i in range(4)]
            rounds.append([_chunk(_delta(content="All four files inspected. Done."))])
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None,
                [{"role": "user", "content": "read every file in this workspace"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
                max_rounds=2,
            ))

            self.assertEqual(outcome.status, "completed")
            diagnostics = [
                str(e["payload"].get("content"))
                for e in renderer.events
                if e["event_type"] == "diagnostics"
            ]
            self.assertTrue(
                any("Round budget reached" in d for d in diagnostics),
                f"expected a round-budget-extension diagnostic, got: {diagnostics}",
            )

    def test_fails_once_extensions_are_also_exhausted(self):
        with tempfile.TemporaryDirectory() as ws:
            files = []
            for i in range(6):
                path = Path(ws) / f"file_{i}.py"
                path.write_text(f"# file {i}\n")
                files.append(path)

            # max_rounds=1 with MAX_AGENT_ROUND_EXTENSIONS patched to 1 means
            # a hard ceiling of 2 total rounds -- 6 distinct tool-call rounds
            # (never producing a final answer) must still end in failure,
            # not an infinite/unbounded extension loop.
            rounds = [self._read_round(i, files[i]) for i in range(6)]
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            with patch("tamfis_code.runner_local.MAX_AGENT_ROUND_EXTENSIONS", 1):
                outcome = asyncio.run(run_local_agent_turn(
                    manager, ProviderType.NVIDIA, None,
                    [{"role": "user", "content": "read every file in this workspace"}],
                    self._console(), renderer,
                    workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
                    max_rounds=1,
                ))

            self.assertEqual(outcome.status, "failed")
            self.assertIn("tool-call rounds", outcome.error or "")
            self.assertIn("1 round-budget extension already granted", outcome.error or "")


if __name__ == "__main__":
    unittest.main()
