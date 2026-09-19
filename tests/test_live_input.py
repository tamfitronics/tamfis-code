import asyncio
import os
import re
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from rich.console import Console
from prompt_toolkit.document import Document

from tamfis_code import state as state_module
from tamfis_code.config import Config, next_mode_in_cycle
from tamfis_code.live_input import (
    LiveInputListener,
    _LiveProgressAutoSuggest,
    _CTRL_T,
    _CTRL_Y,
    _ROTATING_TIPS,
    _SHIFT_TAB,
    _active_agent_count,
    _mode_and_agents_html,
    _right_align,
    _right_chip,
    composer_style,
    force_bottom_toolbar_visible,
    idle_bottom_toolbar,
    live_next_message_suggestion,
)
from tamfis_code.render import StreamRenderer


class _StatePatchMixin:
    def setUp(self):
        self._originals = (state_module.CONFIG_DIR, state_module.STATE_PATH)
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        state_module.CONFIG_DIR = base / ".config"
        state_module.STATE_PATH = base / ".config" / "state.json"

    def tearDown(self):
        state_module.CONFIG_DIR, state_module.STATE_PATH = self._originals
        self.tmp.cleanup()


def _console() -> Console:
    return Console(file=StringIO(), no_color=True, width=200)


def _config(approval_policy: str = "ask") -> Config:
    cfg = Config.__new__(Config)
    cfg.approval_policy = approval_policy
    return cfg


class LiveProgressSuggestionTests(unittest.TestCase):
    def test_in_progress_plan_step_drives_live_suggestion(self):
        renderer = StreamRenderer(_console())
        renderer._plan_steps = [
            {"step": "Inspect auth flow", "status": "completed"},
            {"step": "Add logout regression test", "status": "in_progress"},
        ]
        self.assertEqual(
            live_next_message_suggestion(renderer),
            "Finish and verify the active plan step: Add logout regression test",
        )

    def test_live_phase_supplies_a_grounded_fallback(self):
        renderer = StreamRenderer(_console())
        renderer._phase = "validate"
        self.assertEqual(
            live_next_message_suggestion(renderer),
            "Fix any validation failure before declaring the task complete",
        )

    def test_pending_steering_suppresses_duplicate_suggestion(self):
        renderer = StreamRenderer(_console())
        renderer._phase = "validate"
        renderer.request_steering()
        self.assertIsNone(live_next_message_suggestion(renderer))

    def test_live_ghost_only_appears_in_an_empty_composer(self):
        renderer = StreamRenderer(_console())
        renderer._phase = "repair"
        suggest = _LiveProgressAutoSuggest(renderer)
        self.assertIsNotNone(suggest.get_suggestion(None, Document("")))
        self.assertIsNone(suggest.get_suggestion(None, Document("already typing")))


class ShiftTabCyclesModeTests(unittest.TestCase):
    @patch("tamfis_code.live_input._active_agent_count")
    def test_cached_agent_count_keeps_state_reads_out_of_footer_render(
        self,
        active_agent_count,
    ):
        rendered = _mode_and_agents_html(
            _config("ask"),
            1,
            active_agents=2,
        )

        active_agent_count.assert_not_called()
        self.assertIn("2 agents", rendered)

    def test_active_agent_count_reads_session_ledger_once(self):
        sessions = {
            "1": {"is_swarm_child": False, "execution_status": "running"},
            "2": {"is_swarm_child": True, "execution_status": "running"},
            "3": {"is_swarm_child": True, "execution_status": "completed"},
        }
        with patch.object(state_module, "_load_raw", return_value=sessions) as load:
            self.assertEqual(_active_agent_count(1), 1)

        load.assert_called_once()

    def test_idle_toolbar_keeps_ready_status_and_current_mode_below_input(self):
        fragments = idle_bottom_toolbar(
            _config("ask"), 1, provider="ollama_cloud", model="kimi",
        ).__pt_formatted_text__()
        rendered = "".join(text for _style, text in fragments)

        self.assertIn("ready", rendered)
        self.assertIn("TamfisGPT-Ultra", rendered)
        self.assertNotIn("ollama", rendered.lower())
        self.assertIn("⏵⏵ manual", rendered)
        self.assertIn("shift+tab", rendered)

    def test_idle_toolbar_right_aligns_a_rotating_chip(self):
        # A wide terminal: the tip is shown only when it fits whole (see
        # IdleFooterTipFitTests), and the 80-column fallback used in a test
        # process is too narrow for the longest tips.
        with patch("shutil.get_terminal_size", return_value=os.terminal_size((200, 24))):
            fragments = idle_bottom_toolbar(
                _config("ask"), 1, provider="ollama_cloud", model="kimi",
            ).__pt_formatted_text__()
        rendered = "".join(text for _style, text in fragments)

        chip = _right_chip(1, 0)
        # _right_chip returns HTML markup; the plain tip text is what
        # actually lands in the rendered fragments.
        plain_tip = chip.split(">", 1)[1].rsplit("<", 1)[0]
        self.assertIn(plain_tip, rendered)
        self.assertTrue(any(plain_tip == text for _predicate, text in _ROTATING_TIPS))
        # Right-aligned: the chip is the last thing on the line, after padding.
        self.assertTrue(rendered.rstrip().endswith(plain_tip))

    def test_available_update_replaces_tip_with_clickable_install_action(self):
        handler = Mock()
        fragments = idle_bottom_toolbar(
            _config("ask"), 1, model="auto",
            update_version="9.9.9", update_handler=handler,
        ).__pt_formatted_text__()
        rendered = "".join(fragment[1] for fragment in fragments)

        self.assertIn("↑ Install v9.9.9 · click or Ctrl+U", rendered)
        clickable = [fragment for fragment in fragments if len(fragment) == 3]
        self.assertTrue(clickable)
        self.assertTrue(all(fragment[2] is handler for fragment in clickable))

    def test_right_align_pads_to_terminal_width(self):
        import shutil
        from unittest.mock import patch as mock_patch

        with mock_patch("shutil.get_terminal_size", return_value=shutil.os.terminal_size((40, 24))):
            result = _right_align("<ansigray>left</ansigray>", "right")
        plain = result.replace("<ansigray>", "").replace("</ansigray>", "")
        self.assertEqual(len(plain), 40)
        self.assertTrue(plain.endswith("right"))
        self.assertTrue(plain.startswith("left"))

    def test_right_align_never_collapses_the_gap_on_a_narrow_terminal(self):
        import shutil
        from unittest.mock import patch as mock_patch

        with mock_patch("shutil.get_terminal_size", return_value=shutil.os.terminal_size((10, 24))):
            result = _right_align("<ansigray>a very long left side status line</ansigray>", "right")
        self.assertIn("  right", result)

    def test_footer_style_never_uses_reverse_background(self):
        style = composer_style()
        attrs = style.get_attrs_for_style_str(
            "class:bottom-toolbar class:bottom-toolbar.text"
        )
        self.assertFalse(attrs.reverse)
        ghost = style.get_attrs_for_style_str("class:auto-suggestion")
        self.assertEqual(ghost.color, "ansibrightblack")
        self.assertTrue(ghost.italic)

    def _message_text(self, listener) -> str:
        return "".join(text for _style, text in listener._composer_message().__pt_formatted_text__())

    def _toolbar_text(self, listener) -> str:
        return "".join(text for _style, text in listener._bottom_toolbar().__pt_formatted_text__())

    def test_running_status_sits_above_the_input_with_animation_and_phase_activity(self):
        # Claude Code / Codex layout (owner request 2026-09-19): the running
        # status is ABOVE the input, not packed into the footer under a box.
        renderer = StreamRenderer(_console())
        renderer._phase = "validate"
        renderer._model = "TamfisGPT-Pro"
        listener = LiveInputListener(session_id=1, renderer=renderer, cli_config=_config("ask"))
        listener._status_tick = 2
        message = self._message_text(listener)

        self.assertIn("✶", message)  # star glyph for tick 2 (Claude Code's "✢ Mustering…")
        self.assertTrue(any(word in message for word in ("Evaluating", "Checking", "Verifying")))
        self.assertIn("TamfisGPT-Pro", message)
        self.assertRegex(message, r"\(\d.*\)")  # "(5m 45s · ...)" timing block
        # ...and none of it is in the footer any more.
        footer = self._toolbar_text(listener)
        self.assertNotIn("✶", footer)
        self.assertNotIn("TamfisGPT-Pro", footer)

    def test_the_composer_is_split_status_and_tip_above_rules_around_the_input_mode_below(self):
        renderer = StreamRenderer(_console())
        listener = LiveInputListener(session_id=1, renderer=renderer, cli_config=_config("ask"))
        message_lines = self._message_text(listener).split("\n")
        footer_lines = self._toolbar_text(listener).split("\n")

        # above the input: status, tip, then the input's TOP rule, then the prompt
        self.assertGreaterEqual(len(message_lines), 4)
        self.assertIn("Tip:", message_lines[-3])
        self.assertTrue(set(message_lines[-2]) == {"─"}, message_lines[-2])
        self.assertTrue(message_lines[-1].startswith("❯"))
        # below the input: the BOTTOM rule, then exactly one footer line
        self.assertEqual(len(footer_lines), 2)
        self.assertTrue(set(footer_lines[0]) == {"─"}, footer_lines[0])
        self.assertIn("esc to interrupt", footer_lines[1])
        self.assertNotIn("Tip:", footer_lines[1])

    def test_rules_span_the_terminal_width(self):
        from tamfis_code.live_input import composer_rule_html

        with patch("shutil.get_terminal_size", return_value=os.terminal_size((120, 30))):
            rule = re.sub(r"<[^>]+>", "", composer_rule_html())
        self.assertEqual(rule, "─" * 120)

    def test_pending_update_replaces_the_tip_above_the_input_mid_task(self):
        # Informational only: self_update.py never installs/re-execs
        # mid-task, so this chip has no click/Ctrl+U handler, unlike the
        # idle toolbar's clickable install chip.
        renderer = StreamRenderer(_console())
        renderer.pending_update_version = "1.6.99"
        listener = LiveInputListener(session_id=1, renderer=renderer, cli_config=_config("ask"))
        message = self._message_text(listener)

        self.assertIn("1.6.99", message)
        self.assertIn("update when idle", message)
        self.assertNotIn("Tip:", message)

    def test_toolbar_is_not_suppressed_when_terminal_cpr_is_unknown(self):
        from prompt_toolkit import PromptSession

        session = PromptSession(bottom_toolbar="status", show_frame=True)
        before = repr(session.layout.container.children[-1].filter)
        force_bottom_toolbar_visible(session)
        after = repr(session.layout.container.children[-1].filter)

        self.assertIn("renderer_height_is_known", before)
        self.assertNotIn("renderer_height_is_known", after)

    def test_dispatch_cycles_approval_policy(self):
        renderer = StreamRenderer(_console())
        cfg = _config("ask")
        listener = LiveInputListener(session_id=1, renderer=renderer, cli_config=cfg)
        expected = next_mode_in_cycle("ask")

        listener._buf = bytearray(_SHIFT_TAB)
        listener._dispatch()

        self.assertEqual(cfg.approval_policy, expected)
        self.assertEqual(bytes(listener._buf), b"")

    def test_in_task_cycle_updates_config_and_persistent_renderer_mode(self):
        renderer = StreamRenderer(_console(), mode_label="manual")
        cfg = _config("ask")
        listener = LiveInputListener(session_id=1, renderer=renderer, cli_config=cfg)

        label = listener._cycle_mode()

        self.assertEqual(label, "accept-edits")
        self.assertEqual(cfg.approval_policy, "accept-edits")
        self.assertEqual(renderer._mode_label, "accept-edits")

    def test_second_ctrl_c_during_enqueue_control_does_not_crash(self):
        # Regression test: a rapid second/third Ctrl+C while this write is
        # still in flight used to raise a raw KeyboardInterrupt mid-call
        # (confirmed live, inside state.py's redact_secrets) -- this call
        # site is itself already reacting to the first interrupt, so a
        # further one here must not escape as an unhandled crash.
        renderer = StreamRenderer(_console())
        cfg = _config("ask")
        listener = LiveInputListener(session_id=1, renderer=renderer, cli_config=cfg)
        with patch(
            "tamfis_code.live_input.local_state.enqueue_instruction",
            side_effect=KeyboardInterrupt,
        ):
            listener._enqueue_control("cancel")  # must not raise

    def test_incomplete_escape_sequence_is_not_dropped_prematurely(self):
        renderer = StreamRenderer(_console())
        cfg = _config("ask")
        listener = LiveInputListener(session_id=1, renderer=renderer, cli_config=cfg)

        listener._buf = bytearray(b"\x1b")
        listener._dispatch()
        self.assertEqual(bytes(listener._buf), b"\x1b")  # still waiting

        listener._buf.extend(b"[")
        listener._dispatch()
        self.assertEqual(bytes(listener._buf), b"\x1b[")  # still waiting

        listener._buf.extend(b"Z")
        listener._dispatch()
        self.assertEqual(bytes(listener._buf), b"")  # consumed as Shift+Tab
        self.assertEqual(cfg.approval_policy, next_mode_in_cycle("ask"))

    def test_unrecognised_byte_is_dropped_silently(self):
        renderer = StreamRenderer(_console())
        cfg = _config("ask")
        listener = LiveInputListener(session_id=1, renderer=renderer, cli_config=cfg)

        listener._buf = bytearray(b"x")
        listener._dispatch()

        self.assertEqual(bytes(listener._buf), b"")
        self.assertEqual(cfg.approval_policy, "ask")  # unchanged


class SessionTitleFooterPrefixTests(_StatePatchMixin, unittest.TestCase):
    """The session's name must always be visible in the footer -- idle and
    mid-task alike -- so the user always knows which conversation they're
    in, the same way Codex/Claude Code keep a session's title pinned in
    their own status bar."""

    def test_idle_toolbar_shows_the_generic_label_before_any_turn(self):
        fragments = idle_bottom_toolbar(
            _config("ask"), 7, provider="ollama_cloud", model="kimi",
        ).__pt_formatted_text__()
        rendered = "".join(text for _style, text in fragments)
        self.assertIn("Session 7", rendered)

    def test_idle_toolbar_shows_the_persisted_title_once_set(self):
        # Titles are persisted ONLY by the LLM path now (the mechanical
        # generator was removed), so simulate a completed upgrade.
        state_module.save_session_state(
            7, workspace_root="/a", session_title="Fix the flaky auth test",
            title_source="llm",
        )
        fragments = idle_bottom_toolbar(
            _config("ask"), 7, provider="ollama_cloud", model="kimi",
        ).__pt_formatted_text__()
        rendered = "".join(text for _style, text in fragments)
        self.assertIn("Fix the flaky auth test", rendered)
        self.assertNotIn("Session 7", rendered)

    def test_long_title_is_truncated_in_the_footer(self):
        state_module.save_session_state(
            7, workspace_root="/a", session_title="x" * 100, title_source="llm",
        )
        fragments = idle_bottom_toolbar(
            _config("ask"), 7, provider="ollama_cloud", model="kimi",
        ).__pt_formatted_text__()
        rendered = "".join(text for _style, text in fragments)
        self.assertIn("x" * 27 + "…", rendered)
        self.assertNotIn("x" * 28, rendered)

    def test_in_task_toolbar_also_shows_the_session_title(self):
        state_module.save_session_state(
            9, workspace_root="/a", session_title="Refactor the auth middleware",
            title_source="llm",
        )
        renderer = StreamRenderer(_console())
        listener = LiveInputListener(session_id=9, renderer=renderer, cli_config=_config("ask"))
        rendered = "".join(
            text for _style, text in listener._bottom_toolbar().__pt_formatted_text__()
        )
        self.assertIn("Refactor the auth middleware", rendered)


class RotatingChipIsSituationAwareTests(_StatePatchMixin, unittest.TestCase):
    """The corner chip must not advertise a command that has nothing to act
    on in the current session/thread -- e.g. "/diff" with no modified files,
    or "/retry" on a thread that has no prior turn yet."""

    def test_fresh_session_omits_diff_agents_and_retry_tips(self):
        # A brand-new session has no modified files, no running agents, and
        # no prior turn -- none of those situational tips should ever be
        # eligible, regardless of which point in the rotation is sampled.
        for offset in range(len(_ROTATING_TIPS)):
            with patch("time.monotonic", return_value=offset * 8.0):
                tip = _right_chip(1, 0).split(">", 1)[1].rsplit("<", 1)[0]
                self.assertNotIn(tip, {
                    "/diff to review pending changes",
                    "/agents to see what's running",
                    "/retry to rerun the last turn",
                    "/doctor to check unresolved issues",
                })

    def test_agents_tip_only_eligible_when_agents_are_running(self):
        eligible_without = any(
            predicate(state_module.SessionState(session_id=1), 0)
            for predicate, text in _ROTATING_TIPS
            if text == "/agents to see what's running"
        )
        eligible_with = any(
            predicate(state_module.SessionState(session_id=1), 3)
            for predicate, text in _ROTATING_TIPS
            if text == "/agents to see what's running"
        )
        self.assertFalse(eligible_without)
        self.assertTrue(eligible_with)

    def test_diff_and_retry_tips_become_eligible_once_thread_has_history(self):
        state = state_module.get_session_state(2)
        state.modified_files = [{"path": "foo.py"}]
        state.conversation_history = [{"role": "user", "content": "hi"}]
        state_module.put_session_state(state)

        for offset in range(len(_ROTATING_TIPS)):
            with patch("time.monotonic", return_value=offset * 8.0):
                _right_chip(2, 0)  # exercised for side-effect-free coverage

        diff_predicate = next(
            predicate for predicate, text in _ROTATING_TIPS
            if text == "/diff to review pending changes"
        )
        retry_predicate = next(
            predicate for predicate, text in _ROTATING_TIPS
            if text == "/retry to rerun the last turn"
        )
        self.assertTrue(diff_predicate(state, 0))
        self.assertTrue(retry_predicate(state, 0))


class CtrlTInjectsFollowUpTests(_StatePatchMixin, unittest.IsolatedAsyncioTestCase):
    async def test_btw_runs_separately_without_steering_or_queueing(self):
        renderer = StreamRenderer(_console())
        renderer.request_steering = Mock()
        answer = AsyncMock(return_value="A concise side answer.")
        listener = LiveInputListener(
            session_id=42,
            renderer=renderer,
            cli_config=_config(),
            side_question_callback=answer,
        )

        listener._enqueue("/BTW what does this term mean?")
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        answer.assert_awaited_once_with("what does this term mean?")
        renderer.request_steering.assert_not_called()
        self.assertEqual(
            state_module.get_session_state(42).queued_user_instructions, []
        )
        rendered = renderer.console.file.getvalue()
        self.assertIn("A concise side answer.", rendered)
        self.assertIn("BTW", rendered)

    async def test_bare_btw_reports_usage_without_queueing(self):
        renderer = StreamRenderer(_console())
        answer = AsyncMock()
        listener = LiveInputListener(
            session_id=43,
            renderer=renderer,
            cli_config=_config(),
            side_question_callback=answer,
        )

        listener._enqueue("/btw")

        answer.assert_not_awaited()
        self.assertEqual(
            state_module.get_session_state(43).queued_user_instructions, []
        )
        self.assertIn("Usage: /btw", renderer.console.file.getvalue())

    def test_up_recalls_latest_queued_follow_up_for_editing(self):
        first = state_module.enqueue_instruction(42, "check login", classification="follow_up")
        latest = state_module.enqueue_instruction(42, "then inspect billing", classification="follow_up")
        renderer = StreamRenderer(_console())
        listener = LiveInputListener(session_id=42, renderer=renderer, cli_config=_config())
        buffer = SimpleNamespace(text="", cursor_position=0)

        self.assertTrue(listener._recall_latest_queued(buffer))
        self.assertEqual(buffer.text, "then inspect billing")
        self.assertEqual(buffer.cursor_position, len(buffer.text))
        self.assertEqual(listener._editing_instruction_id, latest.id)
        self.assertNotEqual(listener._editing_instruction_id, first.id)

    def test_submitting_recalled_text_updates_queue_without_duplication(self):
        item = state_module.enqueue_instruction(42, "check login", classification="follow_up")
        renderer = StreamRenderer(_console())
        listener = LiveInputListener(session_id=42, renderer=renderer, cli_config=_config())
        listener._editing_instruction_id = item.id

        listener._enqueue("check login and logout")

        queued = state_module.get_session_state(42).queued_user_instructions
        self.assertEqual(len(queued), 1)
        self.assertEqual(queued[0]["text"], "check login and logout")
        self.assertIsNone(listener._editing_instruction_id)
        self.assertIn("Updated queued instruction", renderer.console.file.getvalue())

    def test_agent_count_ignores_stale_top_level_sessions(self):
        stale = state_module.get_session_state(41)
        stale.execution_status = "running"
        state_module.put_session_state(stale)
        child = state_module.get_session_state(42)
        child.execution_status = "running"
        child.is_swarm_child = True
        state_module.put_session_state(child)

        self.assertEqual(_active_agent_count(exclude_session_id=1), 1)

    @patch("prompt_toolkit.PromptSession.prompt_async", new_callable=AsyncMock, return_value="also check the login page")
    async def test_interject_enqueues_a_follow_up_instruction(self, _mock_prompt_async):
        renderer = StreamRenderer(_console())
        cfg = _config("ask")
        listener = LiveInputListener(session_id=42, renderer=renderer, cli_config=cfg)

        await listener._interject()

        queued = state_module.get_session_state(42).queued_user_instructions
        self.assertEqual(len(queued), 1)
        self.assertEqual(queued[0]["text"], "also check the login page")
        self.assertEqual(queued[0]["classification"], "follow_up")
        rendered = renderer.console.file.getvalue()
        self.assertIn("Steering update sent", rendered)
        self.assertIn("also check the login page", rendered)

    @patch("prompt_toolkit.PromptSession.prompt_async", new_callable=AsyncMock, return_value="   ")
    async def test_blank_interject_queues_nothing(self, _mock_prompt_async):
        renderer = StreamRenderer(_console())
        cfg = _config("ask")
        listener = LiveInputListener(session_id=43, renderer=renderer, cli_config=cfg)

        await listener._interject()

        queued = state_module.get_session_state(43).queued_user_instructions
        self.assertEqual(queued, [])

    async def test_interject_wraps_the_prompt_in_patch_stdout(self):
        # Locks in the actual fix for the "other output can corrupt the
        # in-progress typed line" rough edge: the prompt must run inside
        # prompt_toolkit's patch_stdout(), which is what safely coalesces
        # any concurrent console output above the active input line instead
        # of interleaving with it.
        import sys as _sys

        from prompt_toolkit.patch_stdout import StdoutProxy

        renderer = StreamRenderer(_console())
        cfg = _config("ask")
        listener = LiveInputListener(session_id=44, renderer=renderer, cli_config=cfg)
        seen_patched = False
        batch_interval = None

        async def _check_patched(*_args, **_kwargs):
            nonlocal seen_patched, batch_interval
            seen_patched = isinstance(_sys.stdout, StdoutProxy)
            batch_interval = getattr(_sys.stdout, "sleep_between_writes", None)
            return ""

        with patch("prompt_toolkit.PromptSession.prompt_async", side_effect=_check_patched):
            await listener._interject()
        self.assertTrue(seen_patched, "prompt_async did not run under patch_stdout()")
        self.assertLessEqual(batch_interval, 0.01)
        self.assertNotIsInstance(_sys.stdout, StdoutProxy)  # restored afterward

    def test_ctrl_t_byte_schedules_an_interject_task(self):
        # _dispatch() itself must not block -- it only needs to *schedule*
        # the interject coroutine (asyncio.ensure_future), not run it
        # inline, since running it inline would call the blocking `input()`
        # helper synchronously from inside the fd-readable callback.
        async def _noop():
            return None

        async def _run():
            renderer = StreamRenderer(_console())
            cfg = _config("ask")
            listener = LiveInputListener(session_id=1, renderer=renderer, cli_config=cfg)
            with patch.object(listener, "_interject", side_effect=_noop) as mocked:
                listener._buf = bytearray(_CTRL_Y)
                listener._dispatch()
                self.assertEqual(bytes(listener._buf), b"")
                await asyncio.sleep(0)  # let the scheduled task actually run
                mocked.assert_called_once()

    def test_repeated_ctrl_t_does_not_open_competing_editors(self):
        async def _run():
            renderer = StreamRenderer(_console())
            cfg = _config("ask")
            listener = LiveInputListener(session_id=1, renderer=renderer, cli_config=cfg)
            gate = asyncio.Event()

            async def _blocked():
                await gate.wait()

            with patch.object(listener, "_interject", side_effect=_blocked) as mocked:
                listener._buf = bytearray(_CTRL_Y)
                listener._dispatch()
                await asyncio.sleep(0)
                listener._buf = bytearray(_CTRL_T)
                listener._dispatch()
                await asyncio.sleep(0)
                self.assertEqual(mocked.call_count, 1)
                gate.set()
                await asyncio.sleep(0)

        import asyncio
        asyncio.run(_run())

    def test_ctrl_t_is_detected_when_read_with_adjacent_terminal_bytes(self):
        async def _run():
            renderer = StreamRenderer(_console())
            cfg = _config("ask")
            listener = LiveInputListener(session_id=1, renderer=renderer, cli_config=cfg)
            with patch.object(listener, "_interject", new_callable=AsyncMock) as mocked:
                listener._buf = bytearray(b"x" + _CTRL_Y + b"y")
                listener._dispatch()
                await asyncio.sleep(0)
                mocked.assert_awaited_once()

        import asyncio
        asyncio.run(_run())

        import asyncio
        asyncio.run(_run())


class PauseResumeAreSafeOffTtyTests(unittest.TestCase):
    def test_pause_resume_start_stop_are_no_ops_without_a_real_tty(self):
        # Test processes' stdin is never a TTY -- this locks in that every
        # method degrades to a safe no-op rather than raising, matching
        # every other TTY-gated feature in render.py/interactive.py.
        renderer = StreamRenderer(_console())
        cfg = _config("ask")
        listener = LiveInputListener(session_id=1, renderer=renderer, cli_config=cfg)
        self.assertFalse(listener._is_tty)

        listener.start()
        listener.pause()
        listener.resume()
        listener.stop()  # must not raise


class RendererSuspendResumeTouchesListenerTests(unittest.TestCase):
    def test_suspend_and_resume_are_safe_when_no_listener_attached(self):
        renderer = StreamRenderer(_console())
        renderer.suspend_live()
        renderer.resume_live()  # must not raise; live_input_listener is None

    def test_suspend_and_resume_delegate_to_an_attached_listener(self):
        renderer = StreamRenderer(_console())
        calls = []
        renderer.live_input_listener = SimpleNamespaceListener(calls)
        renderer.suspend_live()
        renderer.resume_live()
        self.assertEqual(calls, ["pause", "resume"])


class SimpleNamespaceListener:
    def __init__(self, calls):
        self._calls = calls

    def pause(self):
        self._calls.append("pause")

    def resume(self):
        self._calls.append("resume")


class PauseAsyncActuallyWaitsForShutdownTests(unittest.TestCase):
    """Regression test for the manual-mode approval-gate freeze: `pause()`
    only requests the old prompt exit and returns immediately, so a caller
    that opens a brand new PromptSession right after (every approval gate)
    could start it before the old one had actually released the terminal --
    two prompt_toolkit Applications then raced for the same stdin fd, and
    the new one could be starved of keystrokes entirely (the y/n prompt
    renders but never responds). `pause_async` must not return until the
    old input task has actually finished."""

    def test_pause_async_waits_for_the_input_task_to_actually_finish(self):
        async def _run():
            renderer = StreamRenderer(_console())
            cfg = _config("ask")
            listener = LiveInputListener(session_id=1, renderer=renderer, cli_config=cfg)
            finished = False

            async def _fake_input_loop():
                # Mirrors the real _input_loop's shape: cleanup happens in a
                # `finally` so it runs whether the coroutine exits normally
                # (app.exit()) or is cancelled (the fallback _cancel_prompt()
                # path) -- pause_async must still not return until either
                # way, this has actually happened.
                nonlocal finished
                try:
                    await asyncio.sleep(0.05)
                finally:
                    finished = True

            listener._input_task = asyncio.create_task(_fake_input_loop())
            await asyncio.sleep(0)  # let the fake task actually start running first
            await listener.pause_async()
            self.assertTrue(finished, "pause_async returned before the old prompt task finished")
            self.assertTrue(listener._paused)

        asyncio.run(_run())

    def test_pause_async_is_a_no_op_when_no_input_task_is_running(self):
        async def _run():
            renderer = StreamRenderer(_console())
            cfg = _config("ask")
            listener = LiveInputListener(session_id=1, renderer=renderer, cli_config=cfg)
            await listener.pause_async()  # must not raise
            self.assertTrue(listener._paused)

        asyncio.run(_run())


class ShutdownPromptLetsExitFinalizeBeforeCancellingTests(unittest.TestCase):
    """Regression test for a live-reported freeze: the busy bottom-toolbar
    status line (e.g. "Summarizing… · auto · 11m9s · esc to interrupt")
    stayed on screen forever after a turn finished, even though the CLI was
    actually fine underneath it -- a new prompt worked right away. Root
    cause: `_shutdown_prompt` called `app.exit(result="")` and then
    immediately `task.cancel()`'d the input task on the very next line, with
    no `await` in between to let the event loop actually resume
    `prompt_async()` and run prompt_toolkit's own render finalization
    (erasing the framed composer/toolbar). The cancel interrupted that
    finalization mid-flight, leaving its last rendered frame stuck as
    static scrollback. `_shutdown_prompt` must give the task a bounded
    chance to finish on its own before falling back to cancel."""

    def test_a_task_that_finishes_shortly_after_exit_is_not_cancelled(self):
        async def _run():
            renderer = StreamRenderer(_console())
            cfg = _config("ask")
            listener = LiveInputListener(session_id=1, renderer=renderer, cli_config=cfg)
            was_cancelled = False

            async def _fake_input_loop():
                nonlocal was_cancelled
                try:
                    # Mirrors prompt_toolkit actually resuming prompt_async()
                    # after app.exit() and running its own finalization.
                    await asyncio.sleep(0.05)
                except asyncio.CancelledError:
                    was_cancelled = True
                    raise

            listener._input_task = asyncio.create_task(_fake_input_loop())
            listener._prompt_session = SimpleNamespace(
                app=SimpleNamespace(is_done=False, exit=lambda result="": None)
            )
            await asyncio.sleep(0)  # let the fake task actually start running first
            await listener._shutdown_prompt()
            self.assertFalse(
                was_cancelled,
                "_shutdown_prompt cancelled the input task before its own "
                "exit-triggered finalization could finish",
            )

        asyncio.run(_run())

    def test_a_task_that_never_exits_is_still_cancelled_as_a_fallback(self):
        async def _run():
            renderer = StreamRenderer(_console())
            cfg = _config("ask")
            listener = LiveInputListener(session_id=1, renderer=renderer, cli_config=cfg)

            async def _stuck_input_loop():
                await asyncio.sleep(10)

            listener._input_task = asyncio.create_task(_stuck_input_loop())
            listener._prompt_session = SimpleNamespace(
                app=SimpleNamespace(is_done=False, exit=lambda result="": None)
            )
            await asyncio.sleep(0)
            await asyncio.wait_for(listener._shutdown_prompt(), timeout=2)
            self.assertIsNone(listener._input_task)

        asyncio.run(_run())


class RendererSuspendLiveAsyncTests(unittest.TestCase):
    def test_suspend_live_async_awaits_the_listener_pause_async(self):
        async def _run():
            renderer = StreamRenderer(_console())
            calls = []

            class _AsyncListener:
                async def pause_async(self):
                    calls.append("pause_async")

            renderer.live_input_listener = _AsyncListener()
            await renderer.suspend_live_async()
            self.assertEqual(calls, ["pause_async"])

        asyncio.run(_run())

    def test_suspend_live_async_is_safe_when_no_listener_attached(self):
        async def _run():
            renderer = StreamRenderer(_console())
            await renderer.suspend_live_async()  # must not raise

        asyncio.run(_run())

    def test_suspend_live_async_if_active_falls_back_to_sync_for_doubles(self):
        from tamfis_code.render import suspend_live_async_if_active

        async def _run():
            calls = []

            class _SyncOnlyRendererDouble:
                def suspend_live(self):
                    calls.append("suspend_live")

            await suspend_live_async_if_active(_SyncOnlyRendererDouble())
            self.assertEqual(calls, ["suspend_live"])

        asyncio.run(_run())


class HandlePromptLoopExceptionTests(_StatePatchMixin, unittest.TestCase):
    """Regression test for a live-reported crash: a cancel queued from
    another terminal and prompt_toolkit's own built-in Ctrl+C handler both
    call Application.exit() in the same input cycle. The second call raises
    "Return value already set" from *inside prompt_toolkit's own key
    processor*, which prompt_toolkit hands to a loop-level exception handler
    (not an ordinary raised exception an `await`-side try/except can catch --
    see the docstring on _handle_prompt_loop_exception). Left unhandled, this
    showed up live as a raw traceback and a blocking "Press ENTER to
    continue", i.e. the whole live UI appearing to freeze mid-audit.
    """

    def _listener(self):
        renderer = StreamRenderer(_console())
        cfg = _config("ask")
        return LiveInputListener(session_id=1, renderer=renderer, cli_config=cfg)

    def test_double_exit_race_is_treated_as_an_ordinary_interrupt(self):
        listener = self._listener()
        exit_calls = []
        fake_app = SimpleNamespace(is_done=False, exit=lambda result="": exit_calls.append(result))
        listener._prompt_session = SimpleNamespace(app=fake_app)

        listener._handle_prompt_loop_exception(
            SimpleNamespace(),
            {"exception": Exception("Return value already set. Application.exit() failed.")},
        )

        self.assertEqual(listener.interrupt_classification, "cancel")
        self.assertEqual(exit_calls, [""])

    def test_already_done_app_is_not_re_exited(self):
        listener = self._listener()
        exit_calls = []
        fake_app = SimpleNamespace(is_done=True, exit=lambda result="": exit_calls.append(result))
        listener._prompt_session = SimpleNamespace(app=fake_app)

        listener._handle_prompt_loop_exception(
            SimpleNamespace(),
            {"exception": Exception("Return value already set. Application.exit() failed.")},
        )

        self.assertEqual(listener.interrupt_classification, "cancel")
        self.assertEqual(exit_calls, [])

    def test_unrelated_exception_falls_through_to_the_previous_handler(self):
        listener = self._listener()
        received = []
        listener._previous_loop_exception_handler = lambda loop, context: received.append(context)

        loop = SimpleNamespace()
        context = {"exception": ValueError("something else entirely")}
        listener._handle_prompt_loop_exception(loop, context)

        self.assertIsNone(listener.interrupt_classification)
        self.assertEqual(received, [context])

    def test_unrelated_exception_uses_asyncio_default_when_no_previous_handler(self):
        listener = self._listener()
        listener._previous_loop_exception_handler = None
        seen = []
        loop = SimpleNamespace(default_exception_handler=lambda context: seen.append(context))

        context = {"exception": ValueError("boom")}
        listener._handle_prompt_loop_exception(loop, context)

        self.assertIsNone(listener.interrupt_classification)
        self.assertEqual(seen, [context])


if __name__ == "__main__":
    unittest.main()



class IdleFooterTipFitTests(_StatePatchMixin, unittest.TestCase):
    """The rotating tip is a nicety: shown only when it fits WHOLE. It used to
    be sliced mid-word at the terminal edge ("Tip: Use /btw for a q")."""

    def _footer(self, columns, **kwargs):
        state_module.save_session_state(1, workspace_root="/a")
        with patch("shutil.get_terminal_size", return_value=os.terminal_size((columns, 24))):
            rendered = idle_bottom_toolbar(Config(), 1, model="auto", **kwargs)
        return "".join(text for _style, text in rendered.__pt_formatted_text__()).split("\n")

    def test_the_idle_toolbar_is_a_rule_then_one_footer_line(self):
        lines = self._footer(120)
        self.assertEqual(len(lines), 2)
        self.assertEqual(set(lines[0]), {"─"})
        self.assertEqual(len(lines[0]), 120)

    def test_the_tip_is_dropped_when_the_footer_is_too_narrow_for_it(self):
        footer = self._footer(70)[1]
        self.assertNotIn("/", footer.split("shift+tab", 1)[1])  # nothing after the mode text
        self.assertLessEqual(len(footer), 70)

    def test_the_tip_is_kept_when_it_fits_whole(self):
        footer = self._footer(200)[1]
        self.assertIn("shift+tab", footer)
        self.assertTrue(footer.rstrip().endswith(("task", "model", "diff", "retry", "issues", "plan", "executing", "models", "running", "agents", "cwd", "message")) or "/" in footer.split("shift+tab", 1)[1])

    def test_the_install_chip_is_always_kept(self):
        footer = self._footer(70, update_version="9.9.9")[1]
        self.assertIn("Install v9.9.9", footer)



class RunningComposerStateTests(_StatePatchMixin, unittest.TestCase):
    """Running-composer behaviour that reads or writes session state -- isolated
    in a temp state file (a bare unittest.TestCase here wrote to session 1 and
    leaked route events into unrelated tests)."""

    def _message_text(self, listener) -> str:
        return "".join(text for _style, text in listener._composer_message().__pt_formatted_text__())

    def _toolbar_text(self, listener) -> str:
        return "".join(text for _style, text in listener._bottom_toolbar().__pt_formatted_text__())

    def test_a_route_exception_rides_on_the_status_line_and_names_no_provider(self):
        state_module.save_session_state(1, workspace_root="/home")
        state_module.record_route_event(1, provider="nvidia", model="nim-1")
        state_module.record_route_event(
            1, provider="ollama_cloud", model="m",
            previous_provider="nvidia", reason="429", kind="failover",
        )
        renderer = StreamRenderer(_console())
        listener = LiveInputListener(session_id=1, renderer=renderer, cli_config=_config("ask"))
        status_line = next(
            line for line in self._message_text(listener).split("\n") if "(" in line and ")" in line
        )
        self.assertIn("⟳ rerouted", status_line)
        self.assertIn("1 failover", status_line)
        for backend in ("nvidia", "ollama", "nim-1"):
            self.assertNotIn(backend, self._message_text(listener) + self._toolbar_text(listener))

    def test_the_footer_never_clips_the_mode_line_for_a_long_title(self):
        state_module.save_session_state(
            1, workspace_root="/a", session_title="Refactor the whole authentication middleware stack",
            title_source="llm",
        )
        listener = LiveInputListener(
            session_id=1, renderer=StreamRenderer(_console()), cli_config=_config("ask"),
        )
        with patch("shutil.get_terminal_size", return_value=os.terminal_size((72, 24))):
            footer = self._toolbar_text(listener).split("\n")[1]
        self.assertIn("esc to interrupt", footer)
        self.assertLessEqual(len(footer), 72)



class RunningCommandActivityTests(_StatePatchMixin, unittest.TestCase):
    """While a command runs, the activity shows on two lines like Claude Code:
    the round's title, then "⎿ $ command (12s)" beneath it."""

    def _message(self, width=100):
        renderer = StreamRenderer(_console())
        renderer.handle_event({"event_type": "task_started", "payload": {"mode": "local"}})
        renderer._round_tool_counts = {"read_file": 3, "search_code": 2}
        renderer._running_command = "python3 -m pytest tests/test_live_input.py -q -p no:cacheprovider " * 3
        renderer._running_command_started = __import__("time").monotonic() - 12
        listener = LiveInputListener(session_id=1, renderer=renderer, cli_config=_config("ask"))
        with patch("shutil.get_terminal_size", return_value=os.terminal_size((width, 24))):
            return "".join(text for _s, text in listener._composer_message().__pt_formatted_text__()).split("\n")

    def test_the_title_and_the_running_command_are_separate_lines(self):
        lines = self._message()
        title = next(i for i, line in enumerate(lines) if "Reading 3 files" in line)
        self.assertNotIn("$", lines[title])
        self.assertIn("⎿  $ python3 -m pytest", lines[title + 1])
        self.assertTrue(lines[title + 1].rstrip().endswith("(12s)"), lines[title + 1])

    def test_a_long_command_is_cut_to_the_terminal_width(self):
        for line in self._message(width=72):
            self.assertLessEqual(len(line), 72, line)
