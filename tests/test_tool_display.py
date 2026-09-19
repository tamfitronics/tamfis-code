"""Claude Code-style tool records in the scrollback.

Owner request 2026-09-19: "Let tamfis-code have such render too". Every tool call
is a durable two-part record -- `● Name(target)` then `⎿  result` -- instead of a
bare arrow line (or, for reads and edits, nothing at all); consecutive reads and
searches collapse into one line.
"""
from __future__ import annotations

import re
import unittest
from io import StringIO

from rich.console import Console

from tamfis_code.render import StreamRenderer
from tamfis_code.tool_display import (
    display_name,
    display_target,
    group_header,
    has_result_content,
    summarize_result,
)


def _renderer(width=110):
    console = Console(file=StringIO(), no_color=True, width=width)
    renderer = StreamRenderer(console)
    renderer.handle_event({"event_type": "task_started", "payload": {"mode": "local"}})
    return console, renderer


def _call(renderer, name, args, result):
    renderer.handle_event({"event_type": "tool_call_requested", "payload": {"name": name, "arguments": args}})
    renderer.handle_event({
        "event_type": "tool_output",
        "payload": {"tool": name, "result": {"tool": name, "success": True, "result": result}},
    })


def _out(console):
    return console.file.getvalue()


class NamesAndTargetsTests(unittest.TestCase):
    def test_tool_names_read_like_actions(self):
        self.assertEqual(display_name("read_file"), "Read")
        self.assertEqual(display_name("execute_command"), "Bash")
        self.assertEqual(display_name("edit_file"), "Update")
        self.assertEqual(display_name("fetch_url"), "Fetch")
        self.assertEqual(display_name("some_new_tool"), "Some New Tool")

    def test_a_command_is_one_line_truncated_with_an_ellipsis(self):
        target = display_target("execute_command", {"command": "cd /x &&\n  python3 -m pytest " + "tests/a.py " * 30}, limit=60)
        self.assertNotIn("\n", target)
        self.assertLessEqual(len(target), 60)
        self.assertTrue(target.endswith("…"))

    def test_secrets_in_a_command_are_redacted(self):
        target = display_target(
            "execute_command", {"command": "mysql --password=hunter2secret db && curl postgres://admin:s3cretpw@db/x"},
        )
        self.assertNotIn("hunter2secret", target)
        self.assertNotIn("s3cretpw", target)

    def test_search_shows_the_pattern_and_only_a_non_default_path(self):
        self.assertEqual(display_target("search_code", {"query": "SLASH", "path": "."}), 'pattern: "SLASH"')
        self.assertEqual(
            display_target("search_code", {"query": "SLASH", "path": "src"}), 'pattern: "SLASH", path: "src"',
        )

    def test_the_group_header_reads_like_a_sentence(self):
        self.assertEqual(group_header({"read_file": 3, "search_code": 2}), "Read 3 files, searched for 2 patterns")
        self.assertEqual(group_header({"read_file": 1}), "Read 1 file")


class ResultSummaryTests(unittest.TestCase):
    def _s(self, tool, result, args=None, success=True):
        return summarize_result(tool, {"tool": tool, "success": success, "result": result}, args)

    def test_a_read_says_how_many_lines(self):
        self.assertEqual(self._s("read_file", "a\nb\nc\n"), (["Read 3 lines"], False))
        self.assertEqual(self._s("read_file", ""), (["Read 0 lines (empty file)"], False))

    def test_a_search_says_matches_and_files(self):
        hits = [{"file": "a.py", "line": 1}, {"file": "b.py", "line": 2}, {"file": "b.py", "line": 9}]
        self.assertEqual(self._s("search_code", hits), (["Found 3 matches in 2 files"], False))
        self.assertEqual(self._s("search_code", []), (["No matches found"], False))

    def test_a_command_shows_its_first_lines_and_a_more_count(self):
        result = {"stdout": "\n".join(f"line {i}" for i in range(10)), "stderr": "", "return_code": 0}
        lines, failed = self._s("execute_command", result)
        self.assertFalse(failed)
        self.assertEqual(lines[:3], ["line 0", "line 1", "line 2"])
        self.assertEqual(lines[3], "… +7 lines")

    def test_a_command_with_no_output_says_so(self):
        self.assertEqual(self._s("execute_command", {"stdout": "", "stderr": "", "return_code": 0}), (["(No output)"], False))

    def test_a_failing_command_shows_the_exit_code_and_the_error(self):
        result = {"stdout": "", "stderr": "ls: cannot access '/nope'\n", "return_code": 2, "success": False}
        lines, failed = self._s("execute_command", result, success=False)
        self.assertTrue(failed)
        self.assertEqual(lines[0], "Exit code 2")
        self.assertIn("cannot access", lines[1])

    def test_a_write_says_how_many_lines_and_where(self):
        lines, _ = self._s("write_file", "✅ Successfully wrote 6 bytes", {"path": "a.txt", "content": "a\nb\nc\n"})
        self.assertEqual(lines, ["Wrote 3 lines to a.txt"])

    def test_a_fetch_says_size_and_status(self):
        lines, _ = self._s("fetch_url", {"status": 200, "bytes": 112600})
        self.assertEqual(lines, ["Received 112.6KB (200 OK)"])

    def test_every_call_gets_a_result_line(self):
        self.assertEqual(self._s("mystery_tool", {})[0], ["Done"])

    def test_an_envelope_with_only_success_carries_no_content(self):
        self.assertFalse(has_result_content({"tool": "glob_files", "success": True}))
        self.assertTrue(has_result_content({"tool": "read_file", "result": "x"}))


class ScrollbackRecordTests(unittest.TestCase):
    def test_a_command_is_a_header_with_a_result_line_under_it(self):
        console, renderer = _renderer()
        _call(renderer, "execute_command", {"command": "pytest -q"},
              {"stdout": "60 passed\n", "stderr": "", "return_code": 0})
        lines = _out(console).splitlines()
        self.assertEqual(lines[0], "● Bash(pytest -q)")
        self.assertEqual(lines[1], "  ⎿  60 passed")

    def test_a_long_command_stays_on_one_line(self):
        console, renderer = _renderer(width=80)
        _call(renderer, "execute_command", {"command": "cd /home/x && " + "python3 -m pytest tests/a.py " * 10},
              {"stdout": "ok\n", "stderr": "", "return_code": 0})
        header = _out(console).splitlines()[0]
        self.assertEqual(len(_out(console).splitlines()), 2)  # header + result, no wrapped continuation
        self.assertTrue(header.endswith("…)"))
        self.assertLessEqual(len(header), 80)

    def test_consecutive_reads_and_searches_collapse_into_one_line(self):
        console, renderer = _renderer()
        _call(renderer, "read_file", {"path": "src/a.py"}, "x\n" * 10)
        _call(renderer, "read_file", {"path": "src/b.py"}, "y\n" * 5)
        _call(renderer, "search_code", {"query": "TODO"}, [{"file": "a.py", "line": 1}])
        self.assertEqual(_out(console), "")  # nothing yet: the run of reads is still open
        renderer.handle_event({"event_type": "assistant_delta", "payload": {"content": "Found it."}})
        text = _out(console)
        self.assertIn("● Read 2 files, searched for 1 pattern", text)
        self.assertIn("⎿  a.py, b.py, pattern: \"TODO\"", text)
        self.assertLess(text.index("● Read 2 files"), text.index("Found it."))

    def test_a_single_read_is_a_full_record(self):
        console, renderer = _renderer()
        _call(renderer, "read_file", {"path": "src/a.py"}, "x\n" * 400)
        renderer.conclude("completed")
        self.assertEqual(_out(console).splitlines()[:2], ["● Read(src/a.py)", "  ⎿  Read 400 lines"])

    def test_a_non_read_call_ends_the_run_of_reads_in_order(self):
        console, renderer = _renderer()
        _call(renderer, "read_file", {"path": "a.py"}, "x\n")
        _call(renderer, "read_file", {"path": "b.py"}, "x\n")
        _call(renderer, "execute_command", {"command": "ls"}, {"stdout": "a\n", "stderr": "", "return_code": 0})
        text = _out(console)
        self.assertLess(text.index("Read 2 files"), text.index("● Bash(ls)"))

    def test_a_failed_read_is_shown_at_once_with_the_real_reason(self):
        console, renderer = _renderer()
        renderer.handle_event({"event_type": "tool_call_requested", "payload": {"name": "read_file", "arguments": {"path": "/etc/shadow"}}})
        renderer.handle_event({"event_type": "tool_output", "payload": {"tool": "read_file", "result": {
            "success": False, "status": "permission_denied", "path": "/etc/shadow"}}})
        text = _out(console)  # no later event: errors are not held behind the group
        self.assertIn("● Read(/etc/shadow)", text)
        self.assertIn("Read failed: Permission denied: /etc/shadow", text)

    def test_a_write_shows_lines_and_the_diff_and_revert_handles(self):
        console, renderer = _renderer()
        _call(renderer, "write_file", {"path": "docs/notes.md", "content": "hello\n" * 30}, "✅ Successfully wrote 180 bytes")
        renderer.handle_event({"event_type": "file_mutation", "payload": {
            "path": "docs/notes.md", "lines_added": 30, "lines_removed": 0, "mutation_id": "m_ab12"}})
        lines = _out(console).splitlines()
        self.assertEqual(lines[0], "● Write(docs/notes.md)")
        self.assertEqual(lines[1], "  ⎿  Wrote 30 lines to docs/notes.md")
        self.assertIn("+30/-0 · /diff m_ab12 to expand · /revert m_ab12", lines[2])

    def test_an_edit_used_to_print_nothing_on_a_terminal_and_now_has_a_record(self):
        console = Console(file=StringIO(), no_color=True, width=110, force_terminal=True)
        renderer = StreamRenderer(console)
        renderer.handle_event({"event_type": "task_started", "payload": {"mode": "local"}})
        _call(renderer, "edit_file", {"path": "src/a.py", "old_string": "a", "new_string": "b"}, "✅ Edited 'src/a.py'")
        plain = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", _out(console))  # a live terminal emits styling
        self.assertIn("● Update(src/a.py)", plain)
        self.assertIn("⎿  Edited src/a.py", plain)
        renderer.finish()

    def test_the_todo_checklist_is_untouched(self):
        console, renderer = _renderer()
        renderer.handle_event({"event_type": "tool_call_requested", "payload": {"name": "write_todos", "arguments": {
            "todos": [{"task": "first", "completed": True}, {"task": "second"}]}}})
        renderer.handle_event({"event_type": "tool_output", "payload": {"tool": "write_todos", "result": {"success": True, "result": "ok"}}})
        text = _out(console)
        self.assertIn("✔ first", text)
        self.assertNotIn("● Write Todos", text)
        self.assertNotIn("Done", text)

    def test_an_empty_completion_envelope_prints_nothing(self):
        console, renderer = _renderer()
        renderer.handle_event({"event_type": "tool_output", "payload": {"tool": "glob_files", "success": True}})
        self.assertEqual(_out(console), "")

    def test_tool_output_is_never_interpreted_as_markup(self):
        console, renderer = _renderer()
        _call(renderer, "execute_command", {"command": "grep '[0-9]+' f"},
              {"stdout": "[bold]not markup[/bold] [red]x\n", "stderr": "", "return_code": 0})
        text = _out(console)
        self.assertIn("[bold]not markup[/bold] [red]x", text)
        self.assertIn("pattern-free" if False else "grep '[0-9]+' f", text)


if __name__ == "__main__":
    unittest.main()
