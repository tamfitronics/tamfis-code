"""Regression tests for a session's stable display title.

Introduced alongside the resumable-multi-session UX fix: `tamfis-code`
sessions previously had no persisted name, so the `resume` picker and the
persistent footer could only ever show a bare numeric id. A title is now
derived once, from the opening line of a session's first completed turn
(see state.remember_conversation_turn), and never overwritten afterwards --
the same convention Codex/Claude Code use for naming a conversation.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from tamfis_code import state as state_module


class _StateDirFixture:
    def setUp(self):
        self._originals = (
            state_module.CONFIG_DIR, state_module.STATE_PATH, state_module._LOCK_PATH,
        )
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        state_module.CONFIG_DIR = base / ".config"
        state_module.STATE_PATH = base / ".config" / "state.json"
        # _LOCK_PATH is bound from CONFIG_DIR at import time, so overriding
        # CONFIG_DIR alone leaves the lock pointing at the real user config
        # dir -- under a read-only sandbox every test then emits the
        # "could not acquire the session-state lock" warning.
        state_module._LOCK_PATH = base / ".config" / ".state.lock"
        state_module._STATE_CACHE = None
        state_module._STATE_CACHE_KEY = None

    def tearDown(self):
        (
            state_module.CONFIG_DIR, state_module.STATE_PATH, state_module._LOCK_PATH,
        ) = self._originals
        state_module._STATE_CACHE = None
        state_module._STATE_CACHE_KEY = None
        self._tmp.cleanup()


class DeriveSessionTitleTests(unittest.TestCase):
    def test_collapses_internal_whitespace_including_newlines(self):
        text = "  Fix   the flaky   test\nacross two lines  "
        self.assertEqual(
            state_module._derive_session_title(text),
            "Fix the flaky test across two lines",
        )

    def test_truncates_long_text_with_an_ellipsis(self):
        text = "x" * 100
        title = state_module._derive_session_title(text)
        self.assertEqual(len(title), 61)
        self.assertTrue(title.endswith("…"))

    def test_blank_text_yields_an_empty_title(self):
        self.assertEqual(state_module._derive_session_title("   \n  "), "")

    def test_strips_leading_filler_words(self):
        # The live-reported complaint: the title "just picks the first few
        # words", so "please fix the login bug" showed as "please fix the
        # login bug" instead of leading with the actual work.
        self.assertEqual(
            state_module._derive_session_title("please fix the login bug"),
            "fix the login bug",
        )

    def test_prefers_the_first_sentence_over_later_detail(self):
        # A multi-sentence objective's opening sentence is its intent;
        # later sentences are detail or constraints and must not crowd out
        # the intent in a 60-char title.
        self.assertEqual(
            state_module._derive_session_title(
                "Fix the session leak in resume. Also update the docs. "
                "Finally add tests."
            ),
            "Fix the session leak in resume",
        )

    def test_never_strips_to_an_empty_title(self):
        # An objective made entirely of filler words still yields a
        # non-empty title (the first word is always kept).
        self.assertEqual(state_module._derive_session_title("please"), "please")


class BestEffortSessionLabelTests(_StateDirFixture, unittest.TestCase):
    def test_empty_for_a_session_with_no_recorded_activity(self):
        state_module.save_session_state(1, workspace_root="/a")
        self.assertEqual(
            state_module.best_effort_session_label(state_module.get_session_state(1)), "",
        )

    def test_prefers_the_active_task_objective(self):
        state_module.save_session_state(
            1, workspace_root="/a",
            active_task={"objective": "Fix intelligent routing pipeline"},
            conversation_summary="stale summary that should be skipped",
        )
        self.assertEqual(
            state_module.best_effort_session_label(state_module.get_session_state(1)),
            "Fix intelligent routing pipeline",
        )

    def test_falls_back_to_the_last_user_turn(self):
        state_module.save_session_state(
            1, workspace_root="/a",
            conversation_history=[
                {"role": "user", "content": "Fix the flaky test"},
                {"role": "assistant", "content": "Done"},
            ],
        )
        self.assertEqual(
            state_module.best_effort_session_label(state_module.get_session_state(1)),
            "Fix the flaky test",
        )

    def test_collapses_a_multi_line_objective_onto_one_line(self):
        # Confirmed live: session 1380884423's active_task.objective is a
        # multi-line message ("please finalise the work statred by codex:\n
        # tip: run /review ..."). The resume picker renders row.title into a
        # single line (resume_picker.py render_picker), so a raw newline
        # there would corrupt the picker's layout -- the label must be
        # whitespace-collapsed exactly like a persisted session_title is.
        state_module.save_session_state(
            1, workspace_root="/a",
            active_task={"objective": "please finalise the work statred by codex:\n  tip: run /review to get a code review of your current work"},
        )
        label = state_module.best_effort_session_label(state_module.get_session_state(1))
        self.assertNotIn("\n", label)
        # The smart title drops the leading filler word "please" and caps at
        # 60 chars -- the same _derive_session_title the persisted
        # session_title uses.
        self.assertEqual(label, "finalise the work statred by codex: tip: run /review to get …")


class SessionDisplayTitleTests(_StateDirFixture, unittest.TestCase):
    def test_falls_back_to_a_generic_label_when_nothing_is_recorded_at_all(self):
        state_module.save_session_state(3, workspace_root="/a")
        self.assertEqual(state_module.session_display_title(3), "Session 3")

    def test_unknown_session_also_gets_the_generic_label(self):
        self.assertEqual(state_module.session_display_title(999), "Session 999")

    def test_falls_back_to_live_activity_before_the_first_turn_completes(self):
        # Confirmed live: a session mid-task (or one that predates the
        # session_title feature) showed as a bare "Session 1380884423" in
        # the resume picker and footer -- indistinguishable from every
        # other such session -- even though its active_task objective was
        # already sitting in state.json. session_title is deliberately left
        # unset here (only remember_conversation_turn ever sets it) to
        # reproduce that exact gap.
        state_module.save_session_state(
            3, workspace_root="/a",
            active_task={"objective": "Fix intelligent routing pipeline"},
        )
        self.assertEqual(
            state_module.session_display_title(3), "Fix intelligent routing pipeline",
        )

    def test_reflects_the_persisted_title_once_set(self):
        state_module.save_session_state(3, workspace_root="/a")
        state_module.remember_conversation_turn(
            3, objective="Refactor the auth middleware", answer="Done.",
        )
        self.assertEqual(state_module.session_display_title(3), "Refactor the auth middleware")


class SessionHasRecordedActivityTests(_StateDirFixture, unittest.TestCase):
    """Live-reported: "whenever you reinstall the sessions titles
    disappear and only the session ID remains" -- traced to
    resolve_local_workspace() being called by read-only, non-
    conversational commands too (`doctor`, `sessions`), so a fresh
    directory permanently registers an empty, title-less session id with
    nothing to ever resume. session_has_recorded_activity is the building
    block for surfacing a real prior session instead (see cli.py's
    _print_resumable_session_hint) -- it must agree exactly with whatever
    session_display_title would otherwise fall back to a bare "Session N"
    for.
    """

    def test_false_for_a_session_with_nothing_recorded_at_all(self):
        state_module.save_session_state(3, workspace_root="/a")
        state = state_module.get_session_state(3)
        self.assertFalse(state_module.session_has_recorded_activity(state))

    def test_true_once_a_persisted_title_exists(self):
        state_module.save_session_state(3, workspace_root="/a")
        state_module.remember_conversation_turn(3, objective="Add dark mode", answer="Done.")
        state = state_module.get_session_state(3)
        self.assertTrue(state_module.session_has_recorded_activity(state))

    def test_true_for_a_session_mid_task_with_no_title_yet(self):
        state_module.save_session_state(
            3, workspace_root="/a", active_task={"objective": "Fix intelligent routing pipeline"},
        )
        state = state_module.get_session_state(3)
        self.assertTrue(state_module.session_has_recorded_activity(state))


class RememberConversationTurnSetsTitleOnceTests(_StateDirFixture, unittest.TestCase):
    def test_first_completed_turn_sets_the_title(self):
        state_module.save_session_state(1, workspace_root="/a")
        state_module.remember_conversation_turn(1, objective="Add dark mode", answer="Done.")
        self.assertEqual(state_module.get_session_state(1).session_title, "Add dark mode")

    def test_later_turns_never_overwrite_an_already_set_title(self):
        state_module.save_session_state(1, workspace_root="/a")
        state_module.remember_conversation_turn(1, objective="Add dark mode", answer="Done.")
        state_module.remember_conversation_turn(1, objective="Now fix the footer too", answer="Done.")
        self.assertEqual(state_module.get_session_state(1).session_title, "Add dark mode")

    def test_blank_objective_leaves_the_title_unset(self):
        state_module.save_session_state(1, workspace_root="/a")
        state_module.remember_conversation_turn(1, objective="   ", answer="Done.")
        self.assertEqual(state_module.get_session_state(1).session_title, "")


class EnsureSessionTitleTests(_StateDirFixture, unittest.TestCase):
    """Confirmed live: three of the four call sites that clear active_task on
    task completion (Remote Workspace reattach, `status` polling a completed
    remote task, and runner.py's remote task-stream completion) never went
    through remember_conversation_turn, so those sessions kept showing as a
    bare "Session N" forever. Each such site now calls ensure_session_title
    directly before clearing active_task -- these tests stand in for those
    three sites without needing to drive the full CLI/runner machinery."""

    def test_sets_the_title_from_the_objective_when_unset(self):
        state_module.save_session_state(1, workspace_root="/a")
        state_module.ensure_session_title(1, "Fix the flaky test")
        self.assertEqual(state_module.get_session_state(1).session_title, "Fix the flaky test")

    def test_never_overwrites_an_already_set_title(self):
        state_module.save_session_state(1, workspace_root="/a")
        state_module.ensure_session_title(1, "Fix the flaky test")
        state_module.ensure_session_title(1, "Something else entirely")
        self.assertEqual(state_module.get_session_state(1).session_title, "Fix the flaky test")

    def test_blank_objective_is_a_no_op(self):
        state_module.save_session_state(1, workspace_root="/a")
        state_module.ensure_session_title(1, "   ")
        self.assertEqual(state_module.get_session_state(1).session_title, "")


def _fake_providers(content_or_exc, *, provider_name="nvidia"):
    """Stand in for tamfis_code.providers.ProviderManager inside
    _generate_session_title: a manager whose first routing provider returns
    `content_or_exc` (a content string, or an exception instance to raise).
    Returns (patcher, create_calls) -- the caller must invoke the patcher's
    stop(); create_calls records each chat.completions.create kwargs dict."""
    create_calls = []

    class FakeCompletions:
        async def create(self, **kwargs):
            create_calls.append(kwargs)
            if isinstance(content_or_exc, Exception):
                raise content_or_exc
            resp = MagicMock()
            choice = MagicMock()
            choice.message.content = content_or_exc
            resp.choices = [choice]
            return resp

    fake_client = MagicMock()
    fake_client.chat.completions = FakeCompletions()

    manager = MagicMock()
    config = MagicMock()
    config.models = ["kimi-k3"]
    config.default_model = "kimi-k3"
    config.free_model = None
    manager.routing_order = (provider_name,)
    manager.get_client.return_value = fake_client
    manager.PROVIDERS = {provider_name: config}
    manager.select_model.return_value = "kimi-k3"

    patcher = patch("tamfis_code.providers.ProviderManager", return_value=manager)
    return patcher, create_calls


class UpgradeSessionTitleWithAiTests(_StateDirFixture, unittest.TestCase):
    """The mechanical title (first ~60 chars of the objective) is upgraded,
    best-effort, to a short LLM-written one via the providers system
    (ProviderManager with main + fallback providers). This must never raise
    or block completion: an unreachable or broken provider should simply
    leave the mechanical title in place -- visibly, not silently."""

    def _run(self, coro):
        import asyncio
        return asyncio.run(coro)

    def _upgrade(self, content_or_exc, objective="Fix the flaky auth test", session_id=1):
        patcher, calls = _fake_providers(content_or_exc)
        with patcher:
            self._run(state_module.upgrade_session_title_with_ai(session_id, objective))
        patcher.stop()
        return calls

    def test_replaces_the_mechanical_title_on_success(self):
        state_module.save_session_state(1, workspace_root="/a")
        state_module.ensure_session_title(1, "Fix the flaky auth test")
        calls = self._upgrade("Fix flaky auth test")
        self.assertEqual(len(calls), 1)
        self.assertEqual(state_module.get_session_state(1).session_title, "Fix flaky auth test")

    def test_llm_call_uses_a_generous_token_budget_not_a_starving_20(self):
        """Reasoning models (kimi-k3, glm) can spend the completion budget on
        reasoning before writing a 4-word title. The old max_tokens=20 made
        those calls return EMPTY content -- indistinguishable from a provider
        outage, silently leaving the mechanical title. The providers path
        must request enough tokens for a reasoning model to actually answer."""
        state_module.save_session_state(1, workspace_root="/a")
        state_module.ensure_session_title(1, "Fix the flaky auth test")
        calls = self._upgrade("Fix flaky auth test")
        self.assertTrue(calls)
        self.assertGreaterEqual(calls[0].get("max_tokens", 0), 100)

    def test_rejects_a_response_that_answers_the_objective_instead_of_titling_it(self):
        """Confirmed live (2026-09-15): the model does not reliably follow
        the "respond with ONLY the title" system prompt -- it can start
        answering the objective's actual content instead. Before validation,
        that full-sentence response got blindly 60-char-truncated,
        producing a title visually indistinguishable from the mechanical
        "first few words" title -- the exact live-reported "still not using
        LLM" symptom, even though a real model call did happen."""
        state_module.save_session_state(1, workspace_root="/a")
        state_module.ensure_session_title(1, "What does a semicolon do in Python?")
        self._upgrade(
            "In Python, a semicolon is used to separate multiple statements "
            "written on a single line, though it is rarely used because "
            "newlines already terminate statements.",
            objective="What does a semicolon do in Python?",
        )
        self.assertEqual(
            state_module.get_session_state(1).session_title,
            "What does a semicolon do in Python?",
        )

    def test_rejects_a_response_ending_in_terminal_punctuation_even_if_short(self):
        state_module.save_session_state(1, workspace_root="/a")
        state_module.ensure_session_title(1, "Fix the flaky auth test")
        self._upgrade("It fixes the test.")
        self.assertEqual(state_module.get_session_state(1).session_title, "Fix the flaky auth test")

    def test_accepts_a_genuinely_short_compliant_title(self):
        state_module.save_session_state(1, workspace_root="/a")
        state_module.ensure_session_title(1, "Investigate the timeout bug in the retry loop")
        self._upgrade("Investigate retry loop timeout", objective="Investigate the timeout bug in the retry loop")
        self.assertEqual(state_module.get_session_state(1).session_title, "Investigate retry loop timeout")

    def test_leaves_the_mechanical_title_when_provider_fails(self):
        state_module.save_session_state(1, workspace_root="/a")
        state_module.ensure_session_title(1, "Fix the flaky auth test")
        self._upgrade(ConnectionError("boom"))
        self.assertEqual(state_module.get_session_state(1).session_title, "Fix the flaky auth test")

    def test_blank_objective_is_a_no_op(self):
        state_module.save_session_state(1, workspace_root="/a")
        self._run(state_module.upgrade_session_title_with_ai(1, "   "))
        self.assertEqual(state_module.get_session_state(1).session_title, "")

    def test_a_successful_upgrade_is_never_retried_on_a_later_turn(self):
        """Every call site awaits this after each completed turn, not just
        the session's first. A successful AI title must never be replaced
        by a later turn's own objective -- a session about "fix the flaky
        auth test" must not retitle itself after an unrelated later
        message in the same thread."""
        state_module.save_session_state(1, workspace_root="/a")
        state_module.ensure_session_title(1, "Fix the flaky auth test")
        self._upgrade("Fix flaky auth test")
        self.assertEqual(state_module.get_session_state(1).session_title, "Fix flaky auth test")

        patcher, _calls = _fake_providers("Unrelated later message")
        with patcher as factory:
            self._run(state_module.upgrade_session_title_with_ai(1, "what's the weather like"))
            factory.assert_not_called()
        patcher.stop()
        self.assertEqual(state_module.get_session_state(1).session_title, "Fix flaky auth test")

    def test_a_failed_attempt_does_not_burn_the_one_shot_budget(self):
        """A provider outage on the first turn must not leave the session
        stuck with its mechanical title forever: the attempt is un-marked,
        so the next turn retries. (The old behavior burned the budget on
        ANY first attempt, successful or not -- a transient outage on turn
        one permanently disabled LLM titling for the session.)"""
        state_module.save_session_state(1, workspace_root="/a")
        state_module.ensure_session_title(1, "Fix the flaky auth test")

        self._upgrade(ConnectionError("provider down"))
        self.assertEqual(state_module.get_session_state(1).session_title, "Fix the flaky auth test")
        self.assertFalse(state_module.get_session_state(1).ai_title_attempted)

        # The next turn retries and succeeds.
        self._upgrade("Fix flaky auth test", objective="Fix the flaky auth test")
        self.assertEqual(state_module.get_session_state(1).session_title, "Fix flaky auth test")
        self.assertTrue(state_module.get_session_state(1).ai_title_attempted)

    def test_a_validation_reject_also_frees_the_budget_for_retry(self):
        """An answer-shaped response is a MODEL behavior failure, not a
        permanent property of the session -- the next turn's objective may
        title perfectly well, so a rejected attempt must also retry."""
        state_module.save_session_state(1, workspace_root="/a")
        state_module.ensure_session_title(1, "What does a semicolon do in Python?")
        self._upgrade(
            "A semicolon separates statements on one line, though it is "
            "rarely used because newlines already separate statements.",
            objective="What does a semicolon do in Python?",
        )
        self.assertFalse(state_module.get_session_state(1).ai_title_attempted)

    def test_a_successful_attempt_marks_the_budget_spent(self):
        state_module.save_session_state(1, workspace_root="/a")
        state_module.ensure_session_title(1, "Fix the flaky auth test")
        self._upgrade("Fix flaky auth test")
        self.assertTrue(state_module.get_session_state(1).ai_title_attempted)

    def test_provider_failure_is_diagnosed_not_silently_swallowed(self):
        """The old blanket `except Exception: pass` made every failure mode
        (unconfigured provider, starved model, validation reject) look
        identical from the outside -- the exact "is this even using the
        LLM?" debugging black hole. Each skipped provider now logs one
        diagnostic line."""
        import contextlib, io
        state_module.save_session_state(1, workspace_root="/a")
        state_module.ensure_session_title(1, "Fix the flaky auth test")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self._upgrade(ConnectionError("boom"))
        self.assertIn("[title]", err.getvalue())
        self.assertIn("provider attempt failed", err.getvalue())
