import unittest
from io import StringIO

from rich.console import Console

from tamfis_code.render import StreamRenderer, _tool_result_message, print_unified_diff
from tamfis_code.interactive import contextualize_short_reply


def _console() -> Console:
    return Console(file=StringIO(), no_color=True, width=200)


class StreamRendererTests(unittest.TestCase):
    def test_assistant_delta_streams_and_sets_streamed_final_text(self):
        console = _console()
        renderer = StreamRenderer(console)
        self.assertFalse(renderer.streamed_final_text)
        renderer.handle_event({"event_type": "assistant_delta", "payload": {"content": "hello"}})
        self.assertTrue(renderer.streamed_final_text)
        self.assertIn("hello", console.file.getvalue())

    def test_whitespace_delta_does_not_suppress_persisted_final_answer(self):
        console = _console()
        renderer = StreamRenderer(console)

        renderer.handle_event({"event_type": "assistant_delta", "payload": {"content": "  \n"}})

        self.assertFalse(renderer.streamed_final_text)

    def test_assistant_message_and_ai_task_completed_do_not_reprint(self):
        # Regression guard for the "final answer printed twice" bug: the
        # renderer must not re-emit assistant_message/ai_task_completed
        # content -- callers (cli.py/interactive.py) decide whether to
        # print the captured summary based on renderer.streamed_final_text.
        console = _console()
        renderer = StreamRenderer(console)
        renderer.handle_event({"event_type": "assistant_delta", "payload": {"content": "the answer"}})
        renderer.handle_event({"event_type": "assistant_message", "payload": {"visible_content": "the answer"}})
        renderer.handle_event({"event_type": "ai_task_completed", "payload": {"status": "completed"}})
        output = console.file.getvalue()
        self.assertEqual(output.count("the answer"), 1)

    def test_approval_required_uses_top_level_command_text_not_nested_object(self):
        # Regression guard: the real backend payload for approval_required
        # is {"command_id": <int>, "command": "<command text>", ...} --
        # "command" is a plain string, not an object with its own fields.
        # This used to be read as payload["command"]["command_text"],
        # which silently produced nothing for every real event.
        console = _console()
        renderer = StreamRenderer(console)
        renderer.handle_event({
            "event_type": "approval_required",
            "payload": {"command_id": 42, "command": "rm -rf build", "risk_level": "medium"},
        })
        self.assertIn("rm -rf build", console.file.getvalue())

    def test_tool_output_success_and_failure_render_distinct_titles(self):
        console = _console()
        renderer = StreamRenderer(console)
        renderer.handle_event({"event_type": "tool_output", "payload": {"tool": "remote_exec", "content": "ok", "success": True}})
        renderer.handle_event({"event_type": "tool_output", "payload": {"tool": "remote_exec", "content": "Error: bad", "success": False}})
        output = console.file.getvalue()
        self.assertIn("Ran command", output)
        self.assertIn("failed", output)

    def test_empty_tool_completion_envelope_is_not_rendered_as_fake_result(self):
        console = _console()
        renderer = StreamRenderer(console)

        renderer.handle_event({
            "event_type": "tool_output",
            "payload": {"tool": "glob_files", "success": True},
        })

        self.assertNotIn("without a structured result", console.file.getvalue())
        self.assertEqual(console.file.getvalue(), "")

    def test_invalid_tool_result_names_the_broken_tool_and_fails(self):
        message, failed = _tool_result_message({"tool": "read_file", "path": "src/app.py"})
        self.assertTrue(failed)
        self.assertIn("read_file", message)
        self.assertNotIn("Tool completed", message)

    def test_short_contextual_replies_are_expanded(self):
        self.assertIn("step 1", contextualize_short_reply("1", has_context=True))
        self.assertIn("Proceed", contextualize_short_reply("ok", has_context=True))
        self.assertEqual(contextualize_short_reply("1", has_context=False), "1")

    def test_live_panel_is_not_created_for_non_tty_console(self):
        # Regression guard for the TTY fork: redirected/piped output must
        # keep today's plain scrolling behaviour untouched, so the live
        # panel path must not run at all when the console isn't a terminal.
        console = _console()
        renderer = StreamRenderer(console)
        self.assertIsNone(renderer._live)

        renderer.handle_event({
            "event_type": "plan_created",
            "payload": {"title": "Plan", "items": [{"step": "read the file", "status": "pending"}]},
        })
        # Non-TTY keeps the original one-shot print of the plan.
        self.assertIn("read the file", console.file.getvalue())

    def test_live_panel_is_created_when_console_is_a_terminal(self):
        console = Console(file=StringIO(), no_color=True, width=200, force_terminal=True)
        renderer = StreamRenderer(console)
        try:
            self.assertIsNotNone(renderer._live)
            renderer.handle_event({
                "event_type": "plan_created",
                "payload": {"title": "Plan", "items": [{"step": "read the file", "status": "pending"}]},
            })
            self.assertEqual(renderer._plan_steps, [{"step": "read the file", "status": "pending"}])
        finally:
            renderer.finish()
        self.assertIsNone(renderer._live)

    def test_assistant_delta_records_estimated_tokens(self):
        console = Console(file=StringIO(), no_color=True, width=200, force_terminal=True)
        renderer = StreamRenderer(console)
        try:
            self.assertEqual(renderer._metrics.metrics.tokens_used, 0)
            renderer.handle_event({"event_type": "assistant_delta", "payload": {"content": "a" * 40}})
            self.assertEqual(renderer._metrics.metrics.tokens_used, 10)
        finally:
            renderer.finish()

    def test_plan_created_tool_execution_stage_shows_tool_name(self):
        console = _console()
        renderer = StreamRenderer(console)
        renderer.handle_event({
            "event_type": "plan_created",
            "payload": {"stage": "tool_execution", "content": "Using tool: tamgpt/remote_exec...", "tool": "tamgpt/remote_exec"},
        })
        self.assertIn("Running command", console.file.getvalue())
        self.assertNotIn("tamgpt/remote_exec", console.file.getvalue())

    def test_plan_created_non_tool_stage_shows_status_line(self):
        console = _console()
        renderer = StreamRenderer(console)
        renderer.handle_event({
            "event_type": "plan_created",
            "payload": {"stage": "provider_routing", "content": "Routing to provider..."},
        })
        self.assertIn("Routing to provider", console.file.getvalue())

    def test_diff_available_shows_filename(self):
        console = _console()
        renderer = StreamRenderer(console)
        renderer.handle_event({
            "event_type": "diff_available",
            "payload": {"filename": "hello.txt", "size_bytes": 12, "download_url": "https://example.invalid/f"},
        })
        self.assertIn("hello.txt", console.file.getvalue())

    def test_unrecognised_event_type_is_shown_not_silently_dropped(self):
        console = _console()
        renderer = StreamRenderer(console)
        renderer.handle_event({"event_type": "some_future_event_type", "payload": {"content": "new thing"}})
        output = console.file.getvalue()
        self.assertIn("some_future_event_type", output)
        self.assertIn("new thing", output)

    def test_unified_diff_uses_semantic_colours_when_supported(self):
        stream = StringIO()
        console = Console(file=stream, force_terminal=True, no_color=False, color_system="standard", width=200)
        print_unified_diff(console, "--- a/x\n+++ b/x\n-old\n+new\n@@ -1 +1 @@")
        output = stream.getvalue()
        self.assertIn("\x1b[31m", output)
        self.assertIn("\x1b[32m", output)

    def test_unified_diff_no_color_has_no_ansi_sequences(self):
        stream = StringIO()
        console = Console(file=stream, no_color=True, force_terminal=True, width=200)
        print_unified_diff(console, "-old\n+new")
        self.assertNotIn("\x1b[", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
