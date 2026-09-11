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

from tamfis_code import state as state_module


class _StateDirFixture:
    def setUp(self):
        self._originals = (state_module.CONFIG_DIR, state_module.STATE_PATH)
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        state_module.CONFIG_DIR = base / ".config"
        state_module.STATE_PATH = base / ".config" / "state.json"
        state_module._STATE_CACHE = None
        state_module._STATE_CACHE_KEY = None

    def tearDown(self):
        state_module.CONFIG_DIR, state_module.STATE_PATH = self._originals
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


class SessionDisplayTitleTests(_StateDirFixture, unittest.TestCase):
    def test_falls_back_to_a_generic_label_before_any_turn_completes(self):
        state_module.save_session_state(3, workspace_root="/a")
        self.assertEqual(state_module.session_display_title(3), "Session 3")

    def test_unknown_session_also_gets_the_generic_label(self):
        self.assertEqual(state_module.session_display_title(999), "Session 999")

    def test_reflects_the_persisted_title_once_set(self):
        state_module.save_session_state(3, workspace_root="/a")
        state_module.remember_conversation_turn(
            3, objective="Refactor the auth middleware", answer="Done.",
        )
        self.assertEqual(state_module.session_display_title(3), "Refactor the auth middleware")


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
