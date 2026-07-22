import unittest
from io import StringIO

from rich.console import Console

from types import SimpleNamespace

from tamfis_code import __version__
from tamfis_code.render import (
    StreamRenderer,
    _tool_action_label,
    _tool_result_message,
    print_banner,
    print_resume_plan_status,
    print_unified_diff,
    resume_live_if_active,
    suspend_live_if_active,
)
from tamfis_code.interactive import contextualize_short_reply


def _console() -> Console:
    return Console(file=StringIO(), no_color=True, width=200)


class PrintResumePlanStatusTests(unittest.TestCase):
    def test_incomplete_plan_shows_step_markers(self):
        console = _console()
        state = SimpleNamespace(
            saved_plans=[{
                "id": "p1", "objective": "fix the bug",
                "steps": [
                    {"step": "Read the file", "status": "completed"},
                    {"step": "Fix it", "status": "in_progress"},
                    {"step": "Verify", "status": "pending"},
                ],
            }],
            active_plan_id="p1",
        )
        print_resume_plan_status(console, state)
        output = console.file.getvalue()
        self.assertIn("Plan in progress", output)
        self.assertIn("Read the file", output)
        self.assertIn("Fix it", output)
        self.assertIn("Verify", output)

    def test_fully_completed_plan_prints_nothing(self):
        console = _console()
        state = SimpleNamespace(
            saved_plans=[{"id": "p1", "objective": "x", "steps": [{"step": "done thing", "status": "completed"}]}],
            active_plan_id="p1",
        )
        print_resume_plan_status(console, state)
        self.assertEqual(console.file.getvalue(), "")

    def test_no_saved_plans_prints_nothing(self):
        console = _console()
        state = SimpleNamespace(saved_plans=[], active_plan_id=None)
        print_resume_plan_status(console, state)
        self.assertEqual(console.file.getvalue(), "")


class ToolActionLabelSecretRedactionTests(unittest.TestCase):
    """Confirmed live: the "Running command · ..." / "Ran command · ..."
    status line rendered a live DB password verbatim, in cleartext, because
    it prints `arguments["command"]` directly with no redaction."""

    def test_execute_command_label_redacts_inline_password(self):
        label = _tool_action_label("execute_command", {"command": "mysql -u finima -pSECRET123 -e 'SELECT 1'"})
        self.assertNotIn("SECRET123", label)
        self.assertIn("Running command", label)

    def test_non_command_target_is_left_untouched(self):
        label = _tool_action_label("read_file", {"path": "/home/finima/www/wp-config.php"})
        self.assertIn("/home/finima/www/wp-config.php", label)


class StreamRendererTests(unittest.TestCase):
    def test_assistant_delta_streams_and_sets_streamed_final_text(self):
        console = _console()
        renderer = StreamRenderer(console)
        self.assertFalse(renderer.streamed_final_text)
        renderer.handle_event({"event_type": "assistant_delta", "payload": {"content": "hello"}})
        self.assertTrue(renderer.streamed_final_text)
        self.assertIn("hello", console.file.getvalue())

    def test_reasoning_delta_tracks_real_thinking_time_then_freezes(self):
        # reasoning_content is a real, separate pre-answer stream some
        # OpenAI-compatible reasoning models emit (confirmed live against
        # NVIDIA NIM) -- must be timed, kept out of the visible answer, and
        # frozen once real content starts (not kept live-incrementing
        # forever).
        console = _console()
        renderer = StreamRenderer(console)
        self.assertIsNone(renderer._reasoning_start)
        self.assertIsNone(renderer._thought_seconds)

        renderer.handle_event({"event_type": "reasoning_delta", "payload": {"content": "let me think"}})
        self.assertIsNotNone(renderer._reasoning_start)
        self.assertIsNone(renderer._thought_seconds)  # still live, not frozen yet
        self.assertNotIn("let me think", console.file.getvalue())  # not printed by default

        renderer.handle_event({"event_type": "assistant_delta", "payload": {"content": "42"}})
        self.assertIsNotNone(renderer._thought_seconds)
        frozen = renderer._thought_seconds
        renderer.handle_event({"event_type": "assistant_delta", "payload": {"content": " more"}})
        self.assertEqual(renderer._thought_seconds, frozen)  # freezes once, doesn't keep updating

    def test_no_reasoning_delta_means_no_thought_tracking(self):
        console = _console()
        renderer = StreamRenderer(console)
        renderer.handle_event({"event_type": "assistant_delta", "payload": {"content": "hi"}})
        self.assertIsNone(renderer._reasoning_start)
        self.assertIsNone(renderer._thought_seconds)

    def test_status_line_shows_thought_for_once_reasoning_seen(self):
        console = Console(file=StringIO(), no_color=True, width=200, force_terminal=True)
        renderer = StreamRenderer(console)
        renderer.handle_event({"event_type": "reasoning_delta", "payload": {"content": "thinking"}})
        renderer.handle_event({"event_type": "assistant_delta", "payload": {"content": "answer"}})
        status = renderer._build_status()
        label = str(status.text) if hasattr(status, "text") else str(status)
        self.assertIn("thought for", label)
        renderer.finish()

    def test_status_line_shows_no_mode_tag_by_default(self):
        console = Console(file=StringIO(), no_color=True, width=200, force_terminal=True)
        renderer = StreamRenderer(console)
        status = renderer._build_status()
        label = str(status.text) if hasattr(status, "text") else str(status)
        self.assertFalse(label.startswith("["))
        renderer.finish()

    def test_status_line_shows_mode_label_passed_at_construction(self):
        console = Console(file=StringIO(), no_color=True, width=200, force_terminal=True)
        renderer = StreamRenderer(console, mode_label="accept-edits")
        status = renderer._build_status()
        label = str(status.text) if hasattr(status, "text") else str(status)
        self.assertIn("[accept-edits]", label)
        renderer.finish()

    def test_set_mode_label_updates_the_persistent_status_line_immediately(self):
        # Live-reported: switching mode (Shift+Tab) mid-task only ever
        # printed a one-time scrolling diagnostic line that later output
        # pushed off-screen -- unlike Claude Code, where the current mode
        # is a persistent, always-visible part of the UI. set_mode_label
        # must update the SAME persistent status line real-time, not just
        # log a message.
        console = Console(file=StringIO(), no_color=True, width=200, force_terminal=True)
        renderer = StreamRenderer(console, mode_label="manual")
        renderer.set_mode_label("auto")
        status = renderer._build_status()
        label = str(status.text) if hasattr(status, "text") else str(status)
        self.assertIn("[auto]", label)
        self.assertNotIn("[manual]", label)
        renderer.finish()

    def test_status_line_shows_a_rotating_tip_after_a_few_seconds(self):
        import time as _time

        console = Console(file=StringIO(), no_color=True, width=200, force_terminal=True)
        renderer = StreamRenderer(console)
        renderer._task_start = _time.monotonic() - 6  # past the tip start threshold
        status = renderer._build_status()
        rendered = console.render_str  # not used; just checking the Group's text content
        # The status is a Group of renderables when a tip is present -- print
        # it to the console and inspect captured output.
        console.print(status)
        output = console.file.getvalue()
        self.assertIn("Tip:", output)
        renderer.finish()

    def test_status_line_has_no_tip_before_the_start_threshold(self):
        import time as _time

        console = Console(file=StringIO(), no_color=True, width=200, force_terminal=True)
        renderer = StreamRenderer(console)
        renderer._task_start = _time.monotonic()  # 0s elapsed
        status = renderer._build_status()
        console.print(status)
        output = console.file.getvalue()
        self.assertNotIn("Tip:", output)
        renderer.finish()

    def test_running_status_keeps_live_input_affordance_visible(self):
        console = Console(file=StringIO(), no_color=True, width=200, force_terminal=True)
        renderer = StreamRenderer(console)
        try:
            renderer.live_input_listener = object()
            console.print(renderer._build_status())
            output = console.file.getvalue()
            self.assertIn("Type a message and press Enter", output)
            self.assertIn("Ctrl+C/Ctrl+D exit", output)
        finally:
            renderer.finish()

    def test_all_tips_reference_real_commands_only(self):
        # Every tip must name a command that actually exists in cli.py's
        # registered command set (or a real REPL slash command) -- this
        # module has already had one incident of a hand-typed, fictional
        # command list (completion.py's original --help text) drifting
        # from reality; tips must not repeat that mistake.
        from tamfis_code.render import _TIPS
        from tamfis_code.cli import cli

        real_commands = set(cli.commands.keys())
        real_repl_commands = {
            "mode", "compact", "diffs", "diff", "revert", "resume", "retry",
            "plan", "execute-plan", "queue", "model", "pty", "delegate",
            "status", "context", "clear", "exit", "quit", "detach", "help",
        }
        for tip in _TIPS:
            import re as _re
            named = _re.findall(r"`(?:tamfis-code )?[/-]*([a-zA-Z][a-zA-Z-]*)", tip)
            self.assertTrue(named, f"tip names no command: {tip!r}")
            for name in named:
                self.assertTrue(
                    name in real_commands or name in real_repl_commands or name in ("approval", "tamfis-code"),
                    f"tip {tip!r} references unknown command {name!r}",
                )

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

    def test_workspace_setup_is_hidden_but_selected_route_is_visible_by_default(self):
        # Workspace bookkeeping stays quiet, but the actual provider route is
        # user-relevant: without it "local:auto" gave no indication which
        # external provider was actually serving the turn.
        console = _console()
        renderer = StreamRenderer(console)
        renderer.handle_event({"event_type": "workspace_scope", "payload": {"content": "Focused workspace scope: /a"}})
        renderer.handle_event({"event_type": "context_reused", "payload": {}})
        renderer.handle_event({"event_type": "context_rescanned", "payload": {"reason": "git_head_changed"}})
        renderer.handle_event({"event_type": "model_selected", "payload": {"provider": "nvidia", "model": "x", "selection_reason": "r"}})
        output = console.file.getvalue()
        self.assertNotIn("Focused workspace scope", output)
        self.assertNotIn("Reusing workspace context", output)
        self.assertNotIn("Workspace rescanned", output)
        self.assertIn("Using nvidia · x", output)
        self.assertEqual(renderer._selected_provider, "nvidia")

    def test_standalone_banner_does_not_call_auto_provider_a_local_host(self):
        console = _console()
        print_banner(
            console,
            host="local:auto",
            workspace_root="/workspace",
            mode="interactive",
            approval_policy="ask",
        )
        output = console.file.getvalue()
        self.assertIn(f"TamfisGPT Code v{__version__}", output)
        self.assertIn("Runtime: standalone", output)
        self.assertIn("Provider: auto (nvidia, openrouter, hf, in capability-ranked order)", output)
        self.assertNotIn("Host: local:auto", output)

    def test_routine_per_turn_setup_lines_show_in_debug_mode(self):
        console = _console()
        renderer = StreamRenderer(console)
        renderer.debug = True
        renderer.handle_event({"event_type": "workspace_scope", "payload": {"content": "Focused workspace scope: /a"}})
        renderer.handle_event({"event_type": "context_reused", "payload": {}})
        renderer.handle_event({"event_type": "model_selected", "payload": {"provider": "nvidia", "model": "x", "selection_reason": "r"}})
        output = console.file.getvalue()
        self.assertIn("Focused workspace scope", output)
        self.assertIn("Reusing workspace context", output)
        self.assertIn("Provider:", output)

    def test_model_selected_with_empty_model_shows_provider_default_not_unknown(self):
        # Tier IV/NIM routes leave the resolved model blank by design
        # (letting the provider pick its own default) -- this used to
        # print as "Model: unknown" in --debug output, which reads as a
        # real problem when nothing is actually unknown.
        console = _console()
        renderer = StreamRenderer(console)
        renderer.debug = True
        renderer.handle_event({"event_type": "model_selected", "payload": {"provider": "nvidia_nim", "model": "", "selection_reason": "r"}})
        output = console.file.getvalue()
        self.assertNotIn("unknown", output)
        self.assertIn("(provider default)", output)

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

    def test_approval_required_renders_a_diff_when_the_payload_carries_one(self):
        console = _console()
        renderer = StreamRenderer(console)
        renderer.handle_event({
            "event_type": "approval_required",
            "payload": {
                "command": "write_file(path='style.css')", "risk_level": "medium",
                "diff": "--- a/style.css\n+++ b/style.css\n@@ -1 +1 @@\n-old\n+new\n",
            },
        })
        output = console.file.getvalue()
        self.assertIn("Proposed change", output)
        self.assertIn("-old", output)
        self.assertIn("+new", output)

    def test_approval_required_without_a_diff_renders_no_diff_panel(self):
        console = _console()
        renderer = StreamRenderer(console)
        renderer.handle_event({
            "event_type": "approval_required",
            "payload": {"command": "execute_command(command='ls')", "risk_level": "medium"},
        })
        self.assertNotIn("Proposed change", console.file.getvalue())

    def test_tool_output_success_and_failure_render_distinctly(self):
        console = _console()
        renderer = StreamRenderer(console)
        renderer.handle_event({"event_type": "tool_output", "payload": {"tool": "remote_exec", "content": "ok", "success": True}})
        renderer.handle_event({"event_type": "tool_output", "payload": {"tool": "remote_exec", "content": "Error: bad", "success": False}})
        output = console.file.getvalue()
        self.assertIn("Ran command", output)
        self.assertIn("✓", output)
        self.assertIn("✗", output)

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

    def test_plan_snapshot_survives_tty_live_panel_shutdown(self):
        console = Console(file=StringIO(), no_color=True, width=200, force_terminal=True)
        renderer = StreamRenderer(console)
        try:
            renderer.handle_event({
                "event_type": "plan_created",
                "payload": {"title": "Plan", "items": [{"step": "inspect mission pipeline", "status": "pending"}]},
            })
            renderer.handle_event({"event_type": "assistant_delta", "payload": {"content": "I will inspect it."}})
        finally:
            renderer.finish()
        self.assertIn("inspect mission pipeline", console.file.getvalue())

    def test_plan_progress_is_durable_after_live_panel_stops(self):
        console = Console(file=StringIO(), no_color=True, width=200, force_terminal=True)
        renderer = StreamRenderer(console)
        try:
            renderer.handle_event({
                "event_type": "plan_created",
                "payload": {"title": "Plan", "items": [
                    {"step": "Inspect", "status": "pending"},
                    {"step": "Fix", "status": "pending"},
                ]},
            })
            renderer.handle_event({"event_type": "assistant_delta", "payload": {"content": "Working."}})
            renderer.handle_event({
                "event_type": "plan_step_progress",
                "payload": {"items": [
                    {"step": "Inspect", "status": "completed"},
                    {"step": "Fix", "status": "in_progress"},
                ]},
            })
        finally:
            renderer.finish()
        output = console.file.getvalue()
        self.assertIn("Plan progress", output)
        self.assertIn("Inspect · completed", output)
        self.assertIn("Fix · in_progress", output)

    def test_approval_gate_suspends_and_resumes_the_live_status_line(self):
        # Regression guard: Rich's Live redraws on its own timer, independent
        # of a blocking console.input() approval prompt -- without suspending
        # it first, the prompt can be silently overwritten/garbled mid-answer.
        # suspend_live_if_active/resume_live_if_active are exactly what
        # runner_local.py's approval-gate code path calls around
        # resolve_approval_decision.
        console = Console(file=StringIO(), no_color=True, width=200, force_terminal=True)
        renderer = StreamRenderer(console)
        try:
            self.assertIsNotNone(renderer._live)

            suspend_live_if_active(renderer)
            self.assertIsNone(renderer._live, "Live panel must be stopped before a blocking prompt")

            resume_live_if_active(renderer)
            self.assertIsNotNone(renderer._live, "Live panel must resume once the decision is made")
        finally:
            renderer.finish()
        self.assertIsNone(renderer._live)

    def test_suspend_and_resume_are_safe_when_no_live_panel_exists(self):
        # Non-TTY consoles never create a Live panel -- both helpers must be
        # harmless no-ops rather than raising, since runner_local.py calls
        # them unconditionally around every approval prompt.
        console = _console()
        renderer = StreamRenderer(console)
        self.assertIsNone(renderer._live)
        suspend_live_if_active(renderer)
        resume_live_if_active(renderer)
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

    def test_file_image_and_video_generation_aliases_render_their_artifacts(self):
        cases = [
            ("file_generated", "bundle.zip", "File generated"),
            ("image_generated", "diagram.png", "Image generated"),
            ("video_generated", "demo.mp4", "Video generated"),
        ]
        for event_type, filename, title in cases:
            console = _console()
            renderer = StreamRenderer(console)
            renderer.handle_event({
                "event_type": event_type,
                "payload": {"filename": filename, "url": f"/files/{filename}"},
            })
            output = console.file.getvalue()
            self.assertIn(filename, output)
            self.assertIn(title, output)

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
