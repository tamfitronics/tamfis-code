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
    """Codex-style activity blocks: "• Explored / └ Read a, b", "• Ran <cmd> / └ output"."""

    def test_a_command_is_a_bullet_with_its_output_under_a_tree_connector(self):
        console, renderer = _renderer()
        _call(renderer, "execute_command", {"command": "pytest -q"},
              {"stdout": "60 passed\n", "stderr": "", "return_code": 0})
        lines = _out(console).splitlines()
        self.assertEqual(lines[0], "• Ran pytest -q")
        self.assertEqual(lines[1], "  └ 60 passed")

    def test_a_long_command_wraps_and_is_cut_with_a_line_count(self):
        console, renderer = _renderer(width=60)
        _call(renderer, "execute_command", {"command": "cd /home/x && " + "python3 -m pytest tests/a.py " * 12},
              {"stdout": "ok\n", "stderr": "", "return_code": 0})
        lines = _out(console).splitlines()
        self.assertTrue(lines[0].startswith("• Ran cd /home/x"))
        self.assertTrue(lines[1].startswith("  │ "))                      # wrapped continuation
        self.assertTrue(any(line.startswith("  │ … +") and line.endswith(" lines") for line in lines))
        self.assertEqual(lines[-1], "  └ ok")
        self.assertTrue(all(len(line) <= 60 for line in lines))

    def test_long_output_shows_a_preview_and_the_transcript_hint_and_ctrl_t_has_all_of_it(self):
        from tamfis_code.render import TOOL_TRANSCRIPT

        TOOL_TRANSCRIPT.clear()
        console, renderer = _renderer()
        output = "\n".join(f"line {i}" for i in range(1, 137)) + "\n"
        _call(renderer, "execute_command", {"command": "journalctl -u x"},
              {"stdout": output, "stderr": "", "return_code": 0})
        lines = _out(console).splitlines()
        self.assertEqual(lines[1], "  └ line 1")
        self.assertEqual(lines[2], "    line 2")
        self.assertEqual(lines[3], "    … +134 lines (Ctrl+O for full output)")
        self.assertEqual(len(lines), 4)
        (kind, full), = TOOL_TRANSCRIPT.entries()                        # what Ctrl+O shows
        self.assertEqual(kind, "tool")
        self.assertIn("$ journalctl -u x", full)
        self.assertIn("line 1\n", full)
        self.assertIn("line 136", full)

    def test_short_output_is_shown_whole_and_never_advertises_a_transcript(self):
        console, renderer = _renderer()
        _call(renderer, "execute_command", {"command": "ls"},
              {"stdout": "a\nb\nc\nd\n", "stderr": "", "return_code": 0})
        text = _out(console)
        self.assertNotIn("Ctrl+O", text)
        self.assertEqual(text.splitlines()[1:], ["  └ a", "    b", "    c", "    d"])

    def test_a_failing_command_says_so_and_uses_the_error_role(self):
        console, renderer = _renderer()
        _call(renderer, "execute_command", {"command": "false"},
              {"stdout": "", "stderr": "boom\n", "return_code": 2})
        lines = _out(console).splitlines()
        self.assertEqual(lines[0], "• Ran false")
        self.assertEqual(lines[1], "  └ Exit code 2")
        self.assertIn("boom", lines[2])

    def test_secrets_are_redacted_in_the_block_and_in_the_transcript(self):
        from tamfis_code.render import TOOL_TRANSCRIPT

        TOOL_TRANSCRIPT.clear()
        console, renderer = _renderer()
        _call(renderer, "execute_command", {"command": "mysql -pSuperSecret123 -e 'select 1'"},
              {"stdout": "1\n", "stderr": "", "return_code": 0})
        self.assertNotIn("SuperSecret123", _out(console))
        self.assertNotIn("SuperSecret123", TOOL_TRANSCRIPT.entries()[0][1])

    def test_consecutive_reads_and_searches_are_one_explored_block(self):
        console, renderer = _renderer()
        _call(renderer, "read_file", {"path": "src/a.py"}, "x\n" * 10)
        _call(renderer, "read_file", {"path": "src/b.py"}, "y\n" * 5)
        _call(renderer, "search_code", {"query": "TODO", "path": "src"}, [{"file": "a.py", "line": 1}])
        _call(renderer, "list_directory", {"path": "src"}, [{"name": "a.py"}])
        self.assertEqual(_out(console), "")  # nothing yet: the run of reads is still open
        renderer.handle_event({"event_type": "assistant_delta", "payload": {"content": "Found it."}})
        text = _out(console)
        self.assertEqual(
            text.splitlines()[:4],
            ["• Explored", "  └ Read a.py, b.py", "    Search TODO in src", "    List src"],
        )
        self.assertLess(text.index("• Explored"), text.index("Found it."))

    def test_a_file_read_repeatedly_is_shown_with_a_count(self):
        console, renderer = _renderer()
        for _ in range(3):
            _call(renderer, "read_file", {"path": "/home/tamfitronics/mu-plugins/chat.php"}, "x\n")
        _call(renderer, "read_file", {"path": "other.php"}, "x\n")
        renderer.conclude("completed")
        self.assertIn("Read chat.php (×3), other.php", _out(console))

    def test_a_single_read_is_still_an_explored_block(self):
        console, renderer = _renderer()
        _call(renderer, "read_file", {"path": "src/a.py"}, "x\n" * 400)
        renderer.conclude("completed")
        self.assertEqual(_out(console).splitlines()[:2], ["• Explored", "  └ Read a.py"])

    def test_a_non_read_call_ends_the_run_of_reads_in_order(self):
        console, renderer = _renderer()
        _call(renderer, "read_file", {"path": "a.py"}, "x\n")
        _call(renderer, "read_file", {"path": "b.py"}, "x\n")
        _call(renderer, "execute_command", {"command": "ls"}, {"stdout": "a\n", "stderr": "", "return_code": 0})
        text = _out(console)
        self.assertLess(text.index("Read a.py, b.py"), text.index("• Ran ls"))

    def test_a_failed_read_is_shown_at_once_with_the_real_reason(self):
        console, renderer = _renderer()
        renderer.handle_event({"event_type": "tool_call_requested", "payload": {"name": "read_file", "arguments": {"path": "/etc/shadow"}}})
        renderer.handle_event({"event_type": "tool_output", "payload": {"tool": "read_file", "result": {
            "success": False, "status": "permission_denied", "path": "/etc/shadow"}}})
        text = _out(console)  # no later event: errors are not held behind the group
        self.assertIn("• Explored", text)
        self.assertIn("Read failed: Permission denied: /etc/shadow", text)

    def test_a_write_shows_its_summary_and_the_diff_and_revert_handles(self):
        console, renderer = _renderer()
        _call(renderer, "write_file", {"path": "docs/notes.md", "content": "hello\n" * 30}, "✅ Successfully wrote 180 bytes")
        renderer.handle_event({"event_type": "file_mutation", "payload": {
            "path": "docs/notes.md", "lines_added": 30, "lines_removed": 0, "mutation_id": "m_ab12"}})
        lines = _out(console).splitlines()
        self.assertEqual(lines[0], "• Wrote docs/notes.md")
        self.assertEqual(lines[1], "  └ Wrote 30 lines to docs/notes.md")
        self.assertIn("+30/-0 · /diff m_ab12 to expand · /revert m_ab12", lines[2])

    def test_an_edit_has_a_record_on_a_terminal(self):
        console = Console(file=StringIO(), no_color=True, width=110, force_terminal=True)
        renderer = StreamRenderer(console)
        renderer.handle_event({"event_type": "task_started", "payload": {"mode": "local"}})
        _call(renderer, "edit_file", {"path": "src/a.py", "old_string": "a", "new_string": "b"}, "✅ Edited 'src/a.py'")
        plain = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", _out(console))  # a live terminal emits styling
        self.assertIn("• Edited src/a.py", plain)
        self.assertIn("└ Edited src/a.py", plain)
        renderer.finish()

    def test_a_failed_edit_never_wears_the_success_verb(self):
        console, renderer = _renderer()
        _call(renderer, "edit_file", {"path": "src/a.py", "old_string": "a", "new_string": "b"},
              {"success": False, "error": "old_string not found"})
        text = _out(console)
        self.assertIn("• Failed to edit src/a.py", text)
        self.assertNotIn("Edited", text)

    def test_the_todo_checklist_is_untouched(self):
        console, renderer = _renderer()
        renderer.handle_event({"event_type": "tool_call_requested", "payload": {"name": "write_todos", "arguments": {
            "todos": [{"task": "first", "completed": True}, {"task": "second"}]}}})
        renderer.handle_event({"event_type": "tool_output", "payload": {"tool": "write_todos", "result": {"success": True, "result": "ok"}}})
        text = _out(console)
        self.assertIn("✔ first", text)
        self.assertNotIn("Write Todos", text)
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
        self.assertIn("grep '[0-9]+' f", text)


class TranscriptViewerTests(unittest.TestCase):
    """Ctrl+O shows full tool output in the same viewer Ctrl+E uses for long messages -- which is unchanged."""

    def setUp(self):
        from tamfis_code.message_viewer import VIEWER
        from tamfis_code.render import COLLAPSED_MESSAGES, TOOL_TRANSCRIPT

        VIEWER.close()
        TOOL_TRANSCRIPT.clear()
        COLLAPSED_MESSAGES.clear()
        self.addCleanup(VIEWER.close)
        self.addCleanup(TOOL_TRANSCRIPT.clear)
        self.addCleanup(COLLAPSED_MESSAGES.clear)

    def test_ctrl_t_opens_the_newest_tool_output_and_closes_again(self):
        from tamfis_code.message_viewer import VIEWER
        from tamfis_code.render import TOOL_TRANSCRIPT

        TOOL_TRANSCRIPT.add("tool", "$ ls\n\n" + "\n".join(f"row {i}" for i in range(300)))
        self.assertTrue(VIEWER.toggle(TOOL_TRANSCRIPT, key="Ctrl+O"))
        panel = "\n".join(VIEWER.panel_lines(100, 40))
        self.assertIn("Tool output · full transcript", panel)
        self.assertIn("Ctrl+O or Esc to show less", panel)
        self.assertIn("row 0", panel)
        self.assertTrue(VIEWER.toggle(TOOL_TRANSCRIPT, key="Ctrl+O"))
        self.assertFalse(VIEWER.is_open)

    def test_nothing_to_show_is_a_no_op(self):
        from tamfis_code.message_viewer import VIEWER
        from tamfis_code.render import TOOL_TRANSCRIPT

        self.assertFalse(VIEWER.toggle(TOOL_TRANSCRIPT, key="Ctrl+O"))
        self.assertFalse(VIEWER.is_open)

    def test_ctrl_e_for_long_messages_still_works_and_keeps_its_own_key_hint(self):
        from tamfis_code.message_viewer import VIEWER
        from tamfis_code.render import COLLAPSED_MESSAGES

        COLLAPSED_MESSAGES.add("assistant", "word " * 5000)
        self.assertTrue(VIEWER.toggle())
        panel = "\n".join(VIEWER.panel_lines(100, 40))
        self.assertIn("Assistant · full message", panel)
        self.assertIn("Ctrl+E or Esc to show less", panel)

    def test_full_output_is_not_lost_when_the_preview_is_cut(self):
        from tamfis_code import tool_display

        rows = tool_display.ran_block("x", "\n".join(str(i) for i in range(50)), width=100)
        self.assertEqual(rows[-1][0], "more")
        self.assertTrue(tool_display.output_was_cut("\n".join(str(i) for i in range(50))))
        self.assertFalse(tool_display.output_was_cut("a\nb"))


if __name__ == "__main__":
    unittest.main()


class TranscriptKeyIsNotTheSshClientsTests(unittest.IsolatedAsyncioTestCase):
    """Owner report 2026-09-21: Termius (SSH client) takes Ctrl+T for a new tab, so the hint's key never reached
    the app. The hint advertises Ctrl+O, and both keys open the viewer."""

    def test_the_hint_names_ctrl_o_not_ctrl_t(self):
        from tamfis_code.tool_display import TRANSCRIPT_HINT, TRANSCRIPT_KEY

        self.assertEqual(TRANSCRIPT_KEY, "Ctrl+O")
        self.assertIn("Ctrl+O", TRANSCRIPT_HINT)
        self.assertNotIn("Ctrl+T", TRANSCRIPT_HINT)

    async def test_the_live_composer_binds_ctrl_o_and_keeps_ctrl_t_as_an_alias(self):
        from unittest.mock import patch

        from prompt_toolkit.key_binding import KeyBindings

        from tamfis_code.live_input import LiveInputListener
        from test_live_input import _config, _console

        recorded: list[tuple] = []
        real_add = KeyBindings.add

        def spy(self, *keys, **kwargs):
            recorded.append(keys)
            return real_add(self, *keys, **kwargs)

        class Stop(Exception):
            pass

        listener = LiveInputListener(
            session_id=1, renderer=StreamRenderer(_console()), cli_config=_config(),
        )
        with patch.object(KeyBindings, "add", spy), \
             patch("prompt_toolkit.PromptSession", side_effect=Stop):
            try:
                await listener._input_loop()
            except Stop:
                pass
        flat = {key for keys in recorded for key in keys}
        self.assertIn("c-o", flat)
        self.assertIn("c-t", flat)


class RunningComposerLeavesNoStaleFrameTests(unittest.IsolatedAsyncioTestCase):
    """Owner paste 2026-09-21: finished tasks and every follow-up Enter left a copy of the running composer
    (rules, tip, "esc to interrupt" footer) in the scrollback. The running composer must erase itself when done."""

    async def test_the_live_composer_is_erased_when_it_ends(self):
        from unittest.mock import patch

        from test_live_input import _config, _console
        from tamfis_code.live_input import LiveInputListener

        seen = {}

        class Stop(Exception):
            pass

        def fake_session(*args, **kwargs):
            seen.update(kwargs)
            raise Stop

        listener = LiveInputListener(session_id=1, renderer=StreamRenderer(_console()), cli_config=_config())
        with patch("prompt_toolkit.PromptSession", fake_session):
            try:
                await listener._input_loop()
            except Stop:
                pass
        self.assertTrue(seen.get("erase_when_done"))
