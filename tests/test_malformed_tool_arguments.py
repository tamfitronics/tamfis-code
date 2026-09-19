"""A tool call whose argument JSON did not parse must never run.

Live-reported 2026-09-19 (operator transcript): the agent asked to write a
Phase-1 audit report, `write_file` was shown as `write_file({})`, every call
answered "write_file requires path, content; retry with the missing
argument(s)", the model insisted "my write_file calls have been missing the
required path and content parameters" and repeated the identical call until
the repeated-action guard failed the task after 43 minutes.

Root cause: the dispatch loop did

    try:
        arguments = json.loads(tc.arguments or "{}")
    except json.JSONDecodeError:
        arguments = {}

i.e. a TRUNCATED tool call (the document was cut off at the output token
limit, so its JSON never closed) became a call with NO arguments. The model had
in fact sent everything; the runner silently discarded it. The contract now:
an unparseable call is refused with a reason the model can act on (resend valid
JSON, split a large write), never executed with empty arguments.
"""
from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from tamfis_code.mcp import MCPServer
from tamfis_code.providers import ProviderType
from tamfis_code.runner_local import (
    malformed_tool_arguments_result,
    parse_tool_call_arguments,
    run_local_agent_turn,
)

from test_reasoning_plan import (
    _FakeClient,
    _FakeManager,
    _RecordingRenderer,
    _StatePatchMixin,
    _chunk,
    _delta,
    _tool_call_delta,
)


class ParseToolCallArgumentsTests(unittest.TestCase):
    """The parse contract the dispatch loop and the approval batch share."""

    def test_a_valid_object_parses_with_no_reason(self):
        arguments, reason = parse_tool_call_arguments(
            '{"path": "/a/b.txt", "content": "hi"}'
        )
        self.assertEqual(arguments, {"path": "/a/b.txt", "content": "hi"})
        self.assertIsNone(reason)

    def test_no_arguments_is_legitimate_and_not_malformed(self):
        for raw in ("", "   ", None):
            arguments, reason = parse_tool_call_arguments(raw)
            self.assertEqual(arguments, {})
            self.assertIsNone(reason, f"{raw!r} is an empty call, not a broken one")

    def test_truncated_json_is_reported_as_malformed(self):
        arguments, reason = parse_tool_call_arguments(
            '{"path": "/a/b.txt", "content": "the report was cut off here'
        )
        self.assertEqual(arguments, {})
        self.assertIsNotNone(reason, "a truncated argument string must not read as 'no arguments'")

    def test_a_non_object_payload_is_malformed(self):
        arguments, reason = parse_tool_call_arguments('["a", "b"]')
        self.assertEqual(arguments, {})
        self.assertIn("list", reason or "")


class MalformedArgumentsRefusalMessageTests(unittest.TestCase):
    """What the model is told -- the message is the fix."""

    def test_a_truncated_call_names_the_output_token_limit(self):
        result = malformed_tool_arguments_result(
            "write_file", "Unterminated string starting at: line 1 column 33",
            truncated=True,
        )
        self.assertFalse(result["success"])
        self.assertTrue(result["malformed_tool_arguments"])
        self.assertIn("NOT executed", result["error"])
        self.assertIn("token limit", result["error"])
        self.assertIn("split", result["error"].lower())

    def test_a_merely_broken_call_still_says_what_to_do(self):
        result = malformed_tool_arguments_result(
            "write_file", "Expecting ',' delimiter", truncated=False,
        )
        self.assertIn("valid JSON", result["error"])
        self.assertNotIn("token limit", result["error"])
        self.assertNotIn("requires path", result["error"])


class TruncatedWriteIsSalvagedNotLostTests(_StatePatchMixin, unittest.TestCase):
    """A truncated LARGE write is repaired, not discarded.

    Refusing it outright (the first version of this fix) still threw away a
    multi-page document the model had already produced. write_file is the one
    call worth repairing: its arguments were cut off by the output token limit,
    everything before the cut is real work, and a big document is exactly what
    cannot fit in one call. The repaired call still goes through the normal
    guard/approval/dispatch path.
    """

    def _console(self):
        from io import StringIO
        from rich.console import Console
        return Console(file=StringIO(), no_color=True, width=200)

    def test_a_truncated_write_keeps_the_recovered_prefix_and_is_continued(self):
        with tempfile.TemporaryDirectory() as ws:
            target = Path(ws) / "audit.md"
            body = "# Phase 1 audit\n\n" + "finding line\n" * 60
            broken = json.dumps({"path": str(target), "content": body})[:-40]
            client = _FakeClient([
                [_chunk(_delta(content="1. Write the phase 1 audit report"))],
                [_chunk(
                    _delta(tool_calls=[_tool_call_delta(
                        0, call_id="call_1", name="write_file", arguments=broken,
                    )]),
                    finish_reason="length",
                )],
                [_chunk(_delta(content="Continuing with mode=append."))],
            ])
            renderer = _RecordingRenderer()

            asyncio.run(run_local_agent_turn(
                _FakeManager(client), ProviderType.NVIDIA, None,
                [{"role": "user", "content": f"create {target} with the phase 1 findings"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto",
                interactive=False,
            ))

            self.assertTrue(target.is_file(), "the recovered prefix must be kept")
            written = target.read_text()
            self.assertTrue(written.startswith("# Phase 1 audit"))
            self.assertLess(len(written), len(body), "only the recovered prefix is written")
            self.assertGreater(len(written), len(body) // 2)

            diagnostics = [
                str(event.get("payload", {}).get("content", ""))
                for event in renderer.events
                if event.get("event_type") == "diagnostics"
            ]
            self.assertTrue(
                any("truncated" in text and "append" in text for text in diagnostics),
                f"the salvage must be visible and say how to continue: {diagnostics}",
            )

    def test_a_truncated_call_with_nothing_recoverable_is_refused(self):
        with tempfile.TemporaryDirectory() as ws:
            target = Path(ws) / "audit.md"
            # Cut off before any value completed: there is no path to salvage,
            # so there is nothing to repair and the call must be refused.
            broken = '{"path": "' + str(target)[:8]
            client = _FakeClient([
                # Round 1 is the planner's own completion (the runner asks for
                # one before the first tool round), so it must be a plain plan.
                [_chunk(_delta(content="1. Write the phase 1 audit report"))],
                [_chunk(
                    _delta(tool_calls=[_tool_call_delta(
                        0, call_id="call_1", name="write_file", arguments=broken,
                    )]),
                    finish_reason="length",
                )],
                [_chunk(_delta(content="Understood -- I will split the write."))],
            ])
            renderer = _RecordingRenderer()
            dispatched: list[str] = []
            original_call_tool = MCPServer.call_tool

            async def recording_call_tool(self, name, arguments=None, **kwargs):
                dispatched.append(name)
                return await original_call_tool(self, name, arguments or {}, **kwargs)

            MCPServer.call_tool = recording_call_tool
            try:            asyncio.run(run_local_agent_turn(
                _FakeManager(client), ProviderType.NVIDIA, None,
                [{"role": "user", "content": f"create {target} with the findings"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto",
                interactive=False,
            ))
            finally:
                MCPServer.call_tool = original_call_tool

            self.assertFalse(target.exists(), "an unrecoverable write must not create the file")

            tool_results = [
                event.get("payload", {}).get("result", {})
                for event in renderer.events
                if event.get("event_type") == "tool_output"
                and event.get("payload", {}).get("tool") == "write_file"
            ]
            self.assertTrue(tool_results, "the refusal must be reported as a tool result")
            refusal = tool_results[0]
            self.assertFalse(refusal.get("success"))
            self.assertTrue(refusal.get("malformed_tool_arguments"))
            self.assertIn("valid JSON", refusal.get("error", ""))
            # ...and how to recover, instead of the old "you forgot an
            # argument" message that made the model repeat the same call.
            self.assertIn("split", refusal.get("error", "").lower())
            self.assertNotIn("requires path", refusal.get("error", ""))

    def test_a_truncated_non_write_tool_is_never_repaired(self):
        """Only write_file is repaired. Acting on a partially-recovered
        command line could turn a truncated call into a WRONG one."""
        with tempfile.TemporaryDirectory() as ws:
            broken = json.dumps({"command": "rm -rf " + "/" * 3})[:-4]
            client = _FakeClient([
                [_chunk(_delta(content=f"1. create {ws}/cleanup.md with the notes"))],
                [_chunk(
                    _delta(tool_calls=[_tool_call_delta(
                        0, call_id="call_1", name="execute_command", arguments=broken,
                    )]),
                    finish_reason="length",
                )],
                [_chunk(_delta(content="That call was refused."))],
                [_chunk(_delta(content="That call was refused."))],
            ])
            renderer = _RecordingRenderer()
            dispatched: list[str] = []
            original_call_tool = MCPServer.call_tool

            async def recording_call_tool(self, name, arguments=None, **kwargs):
                dispatched.append(name)
                return await original_call_tool(self, name, arguments or {}, **kwargs)

            MCPServer.call_tool = recording_call_tool
            try:
                asyncio.run(run_local_agent_turn(
                    _FakeManager(client), ProviderType.NVIDIA, None,
                    [{"role": "user", "content": f"create {ws}/cleanup.md with the notes"}],
                    self._console(), renderer,
                    workspace_root=ws, session_id=1, approval_policy="auto",
                    interactive=False,
                ))
            finally:
                MCPServer.call_tool = original_call_tool

            self.assertEqual(
                [name for name in dispatched if name == "execute_command"], [],
                "a truncated command must never be repaired and run",
            )
            refusals = [
                event.get("payload", {}).get("result", {})
                for event in renderer.events
                if event.get("event_type") == "tool_output"
                and event.get("payload", {}).get("tool") == "execute_command"
            ]
            self.assertTrue(refusals and refusals[0].get("malformed_tool_arguments"))

            diagnostics = [
                str(event.get("payload", {}).get("content", ""))
                for event in renderer.events
                if event.get("event_type") == "diagnostics"
            ]
            self.assertTrue(
                any("unparseable" in text for text in diagnostics),
                f"the refusal must be visible, not silent: {diagnostics}",
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
