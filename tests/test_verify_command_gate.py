"""runner_local.py's active build/typecheck verification gate: when a turn
mutates files in a JS/TS project with a detectable check/build script
(workspace.py's detect_verify_command), the model must confirm it re-ran
that command since the last edit before the turn is allowed to finish.

Live-reported ("So why can't tamfis-code be efficient at least?" / "That
npm run build trust isn't a model issue; it is system issues"): nothing in
the completion path required a real, re-run, zero-error verification
before a coding task could report done -- confirmed live on TamfisSEO Pro,
where competitorCloner.ts and contentGenerator.ts sat syntactically
truncated while multiple turns reported success. This is a *system* gap,
not a model-capability one: even a fully capable model only verifies what
the harness actually requires before letting it finish.
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


class VerifyCommandGateTests(_StatePatchMixin, unittest.TestCase):
    def _console(self):
        from io import StringIO
        from rich.console import Console
        return Console(file=StringIO(), no_color=True, width=200)

    def _js_workspace(self, tmp: str) -> Path:
        root = Path(tmp)
        (root / "package.json").write_text(
            json.dumps({"name": "x", "scripts": {"check": "tsc -b", "build": "vite build"}}),
            encoding="utf-8",
        )
        return root

    def test_mutation_without_verification_is_nudged_then_completes_after_retries_exhausted(self):
        """The model edits a file but never runs the check command, even
        after being asked -- this must NOT hard-fail the whole task (unlike
        a fabricated/narrated-intent guard) and must NOT loop forever; it
        completes once the bounded retry budget is spent, with diagnostics
        proving the nudge actually fired."""
        with tempfile.TemporaryDirectory() as tmp:
            ws = self._js_workspace(tmp)
            write_args = json.dumps({"path": str(ws / "app.ts"), "content": "export const x = 1;\n"})
            plan_response = json.dumps({"steps": ["Add app.ts"]})

            rounds = [
                [_chunk(_delta(content=plan_response))],
                [_chunk(_delta(tool_calls=[_tool_call_delta(0, call_id="call_1", name="write_file", arguments=write_args)]))],
                # A real tool result triggers one plan-revision round (expects
                # JSON); non-JSON here just falls back to the existing plan
                # silently -- see test_reasoning_plan.py's own malformed-
                # response test for the same behavior.
                [_chunk(_delta(content="Added app.ts."))],
                [_chunk(_delta(content="Added app.ts."))],
                [_chunk(_delta(content="Added app.ts."))],
                [_chunk(_delta(content="Added app.ts."))],
            ]
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None,
                [{"role": "user", "content": "add a small app.ts file"}],
                self._console(), renderer,
                workspace_root=str(ws), session_id=1, approval_policy="auto", interactive=False,
            ))

            self.assertEqual(outcome.status, "completed")
            diagnostics = [
                str(e["payload"].get("content"))
                for e in renderer.events
                if e["event_type"] == "diagnostics"
            ]
            nudges = [d for d in diagnostics if "npm run check" in d and "hasn't been confirmed clean" in d]
            self.assertEqual(len(nudges), 2, f"expected exactly MAX_VERIFY_COMMAND_RETRIES nudges, got: {diagnostics}")

    def test_no_verify_command_detected_never_gates_completion(self):
        """A workspace with no package.json (or no recognised script) must
        behave exactly as before this change -- no nudge, no extra rounds."""
        with tempfile.TemporaryDirectory() as tmp:
            write_args = json.dumps({"path": str(Path(tmp) / "app.py"), "content": "x = 1\n"})
            plan_response = json.dumps({"steps": ["Add app.py"]})
            rounds = [
                [_chunk(_delta(content=plan_response))],
                [_chunk(_delta(tool_calls=[_tool_call_delta(0, call_id="call_1", name="write_file", arguments=write_args)]))],
                [_chunk(_delta(content="Added app.py."))],
                [_chunk(_delta(content="Added app.py."))],
            ]
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None,
                [{"role": "user", "content": "add a small app.py file"}],
                self._console(), renderer,
                workspace_root=tmp, session_id=2, approval_policy="auto", interactive=False,
            ))

            self.assertEqual(outcome.status, "completed")
            diagnostics = [
                str(e["payload"].get("content"))
                for e in renderer.events
                if e["event_type"] == "diagnostics"
            ]
            self.assertFalse(any("hasn't been confirmed clean" in d for d in diagnostics))


if __name__ == "__main__":
    unittest.main()
