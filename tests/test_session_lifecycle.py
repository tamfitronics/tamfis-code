"""Tests for a session's created_at (set once, ever) and archived flag.

Both back the `tamfis-code resume` full-screen picker: created_at backs
its "Sort: Created" option, archived backs Ctrl+A there (a soft,
reversible hide -- never the same thing as `tamfis-code clear-session`,
which is the only operation that actually erases a session).
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


class CreatedAtTests(_StateDirFixture, unittest.TestCase):
    def test_set_on_first_write(self):
        state_module.save_session_state(1, workspace_root="/a")
        self.assertTrue(state_module.get_session_state(1).created_at)

    def test_never_changes_on_later_writes(self):
        state_module.save_session_state(1, workspace_root="/a")
        first = state_module.get_session_state(1).created_at
        state_module.save_session_state(1, workspace_root="/b")
        state_module.save_session_state(1, current_working_directory="/b")
        self.assertEqual(state_module.get_session_state(1).created_at, first)

    def test_differs_from_updated_at_after_a_later_write(self):
        import time

        state_module.save_session_state(1, workspace_root="/a")
        created = state_module.get_session_state(1).created_at
        time.sleep(0.01)
        state_module.save_session_state(1, workspace_root="/b")
        state = state_module.get_session_state(1)
        self.assertEqual(state.created_at, created)
        self.assertNotEqual(state.updated_at, created)


class SetSessionArchivedTests(_StateDirFixture, unittest.TestCase):
    def test_defaults_to_not_archived(self):
        state_module.save_session_state(1, workspace_root="/a")
        self.assertFalse(state_module.get_session_state(1).archived)

    def test_archiving_sets_the_flag(self):
        state_module.save_session_state(1, workspace_root="/a")
        state_module.set_session_archived(1, True)
        self.assertTrue(state_module.get_session_state(1).archived)

    def test_unarchiving_clears_the_flag(self):
        state_module.save_session_state(1, workspace_root="/a")
        state_module.set_session_archived(1, True)
        state_module.set_session_archived(1, False)
        self.assertFalse(state_module.get_session_state(1).archived)

    def test_archiving_never_touches_conversation_history(self):
        state_module.save_session_state(1, workspace_root="/a")
        state_module.remember_conversation_turn(1, objective="hi", answer="hello")
        state_module.set_session_archived(1, True)
        state = state_module.get_session_state(1)
        self.assertTrue(state.archived)
        self.assertEqual(state.session_title, "hi")
        self.assertTrue(state.conversation_history)

    def test_session_stays_in_all_known_session_ids_once_archived(self):
        state_module.save_session_state(1, workspace_root="/a")
        state_module.set_session_archived(1, True)
        self.assertIn(1, state_module.all_known_session_ids())
