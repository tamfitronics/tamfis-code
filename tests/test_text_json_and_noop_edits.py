"""Regressions from the live auto-blog plugin session (2026-09-24):

1. A weak model wrote its tool call as prose JSON with UNESCAPED nested
   quotes ({"name": "write_todos", "parameters": {"todos": "[{"task": ...
   which is not valid JSON). The extraction only handled parseable objects,
   so the raw JSON rendered verbatim in the Assistant panel while no tool
   ran. Now the malformed candidate is quarantined from the answer and
   answered with the ordinary malformed-arguments refusal.

2. edit_file/write_file with content identical to the file on disk rewrote
   the bytes and recorded a +0/-0 "mutation" that the completion validator
   counted as edit evidence ("✅ Edited ... +0/-0"). Identical content is
   now an honest no-op result, and no mutation is recorded.
"""
import asyncio
import tempfile
import unittest
from pathlib import Path

from tamfis_code.mcp import MCPServer
from tamfis_code.runner_local import _extract_text_json_tool_calls


class MalformedTextJsonToolCallTests(unittest.TestCase):
    OBSERVED = (
        "Here is an example of how we can modify the write_todos function to "
        'include a new task:\n\n{"name": "write_todos", "parameters": '
        '{"todos": "[{"task": "Check plugin settings", "completed": false}, '
        '{"task": "Reset plugin to default settings", "completed": false}, '
        '{"task": "Verify plugin functionality", "completed": false}]"}}'
    )

    def test_observed_malformed_json_is_quarantined_and_refused(self):
        rest, calls = _extract_text_json_tool_calls(self.OBSERVED)
        self.assertNotIn('{"name": "write_todos"', rest)
        self.assertTrue(any(c.name == "malformed_text_tool_call" for c in calls))
        # No call masquerades as an executable write_todos.
        self.assertFalse(any(c.name == "write_todos" for c in calls))

    def test_valid_escaped_json_still_becomes_a_real_call(self):
        text = (
            'do it: {"name": "write_todos", "parameters": '
            '{"todos": "[{\\"task\\": \\"A\\", \\"completed\\": false}]"}}'
        )
        rest, calls = _extract_text_json_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "write_todos")
        self.assertNotIn('{"name"', rest)

    def test_truncated_candidate_is_quarantined(self):
        text = (
            'working on it {"name": "execute_command", '
            '"parameters": {"command": "wp plugin list --format='
        )
        rest, calls = _extract_text_json_tool_calls(text)
        self.assertNotIn('{"name": "execute_command"', rest)
        self.assertTrue(any(c.name == "malformed_text_tool_call" for c in calls))

    def test_ordinary_json_prose_is_untouched(self):
        prose = 'Use a dict like {"a": 1} in Python, or {"key": "value"} config format.'
        rest, calls = _extract_text_json_tool_calls(prose)
        self.assertEqual(rest, prose)
        self.assertEqual(calls, [])

    def test_valid_call_and_malformed_candidate_coexist(self):
        text = (
            'run {"name": "get_git_info", "parameters": {}} then '
            '{"name": "write_todos", "parameters": {"todos": "[{"task": "x"'
        )
        rest, calls = _extract_text_json_tool_calls(text)
        names = [c.name for c in calls]
        self.assertIn("get_git_info", names)
        self.assertIn("malformed_text_tool_call", names)
        self.assertNotIn('"name": "write_todos"', rest)


class _IsolatedMCP(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.server = MCPServer(workspace_root=self.tmp.name)
        self.file = Path(self.tmp.name) / "x.txt"
        self.file.write_text("hello world\n")


class NoOpEditTests(_IsolatedMCP):
    def test_identical_replacement_is_an_honest_no_op(self):
        out = asyncio.run(self.server._edit_file(
            "x.txt", old_string="hello world", new_string="hello world",
        ))
        self.assertIn("No changes", out)
        self.assertNotIn("✅", out)
        self.assertEqual(self.file.read_text(), "hello world\n")

    def test_no_op_edit_records_no_mutation(self):
        asyncio.run(self.server._edit_file(
            "x.txt", old_string="hello world", new_string="hello world",
        ))
        from tamfis_code import state as local_state

        if self.server.session_id is not None:
            state = local_state.get_session_state(self.server.session_id)
            self.assertEqual(
                [m for m in state.modified_files if m.get("path") == str(self.file)],
                [],
            )

    def test_identical_write_is_an_honest_no_op(self):
        out = asyncio.run(self.server._write_file("x.txt", content="hello world\n"))
        self.assertIn("No changes", out)
        self.assertEqual(self.file.read_text(), "hello world\n")

    def test_identical_append_is_an_honest_no_op(self):
        out = asyncio.run(self.server._write_file("x.txt", content="", mode="append"))
        self.assertIn("Nothing appended", out)
        self.assertEqual(self.file.read_text(), "hello world\n")

    def test_real_edit_still_works_and_records_the_mutation(self):
        out = asyncio.run(self.server._edit_file(
            "x.txt", old_string="hello", new_string="goodbye",
        ))
        self.assertTrue(out.startswith("✅"))
        self.assertEqual(self.file.read_text(), "goodbye world\n")
        from tamfis_code import state as local_state

        if self.server.session_id is not None:
            state = local_state.get_session_state(self.server.session_id)
            self.assertTrue(
                any(m.get("path") == str(self.file) for m in state.modified_files)
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
