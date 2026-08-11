#!/usr/bin/env python3
"""Regression tests for the state.json UI-freeze fix (tamfis-code 1.6.1).

Root cause (measured live on a real install): state.json is a single file
shared by every session ever created, several SessionState list/dict fields
(modified_files, inspected_files, ...) had no call site that ever bounded
them, and put_session_state did a synchronous full read+json.dump+fsync of
the *entire* multi-session file on every single event -- often many times
per turn during a long audit -- on the same thread driving the live
prompt_toolkit UI. A live install had grown to 15.7MB/80 sessions, several
individual sessions multiple MB each, which made every write visibly stall
the terminal.

This covers the three independent fixes: per-field caps enforced centrally
(not relying on every call site to remember to slice), stale-session
eviction from the hot file, and the in-memory parse cache.
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


class EnforceStateCapsTests(_StateDirFixture, unittest.TestCase):
    def test_modified_files_is_capped_regardless_of_call_site(self):
        # This is the exact bug: runner.py's file_mutation handler appended
        # to modified_files with no [-N:] slice, unlike every other list
        # field it touches.
        oversized = [{"mutation_id": str(i), "path": f"f{i}.py"} for i in range(1000)]
        state_module.save_session_state(1, modified_files=oversized)
        loaded = state_module.get_session_state(1)
        self.assertEqual(len(loaded.modified_files), state_module.MAX_MODIFIED_FILES)
        # Keeps the most recent entries, not the oldest.
        self.assertEqual(loaded.modified_files[-1]["mutation_id"], "999")

    def test_inspected_files_dict_is_capped_keeping_most_recent(self):
        oversized = {f"/repo/f{i}.py": {"path": f"/repo/f{i}.py"} for i in range(1000)}
        state_module.save_session_state(2, inspected_files=oversized)
        loaded = state_module.get_session_state(2)
        self.assertEqual(len(loaded.inspected_files), state_module.MAX_INSPECTED_FILES)
        self.assertIn("/repo/f999.py", loaded.inspected_files)
        self.assertNotIn("/repo/f0.py", loaded.inspected_files)

    def test_small_session_is_unaffected(self):
        state_module.save_session_state(3, modified_files=[{"mutation_id": "1", "path": "a.py"}])
        loaded = state_module.get_session_state(3)
        self.assertEqual(len(loaded.modified_files), 1)


class PruneStaleSessionsTests(_StateDirFixture, unittest.TestCase):
    def test_session_untouched_past_max_age_is_dropped_on_next_write(self):
        state_module.save_session_state(10, workspace_root="/repo")
        raw = state_module._load_raw()
        from datetime import datetime, timezone, timedelta
        stale_at = (datetime.now(timezone.utc) - state_module.STALE_SESSION_MAX_AGE - timedelta(days=1)).isoformat()
        raw["10"]["updated_at"] = stale_at
        state_module._save_raw(raw)

        # A write for a *different* session should evict the stale one.
        state_module.save_session_state(11, workspace_root="/repo2")

        raw_after = state_module._load_raw()
        self.assertNotIn("10", raw_after)
        self.assertIn("11", raw_after)

    def test_the_session_being_written_is_never_pruned_even_if_its_own_updated_at_is_old(self):
        state_module.save_session_state(20, workspace_root="/repo")
        raw = state_module._load_raw()
        from datetime import datetime, timezone, timedelta
        raw["20"]["updated_at"] = (
            datetime.now(timezone.utc) - state_module.STALE_SESSION_MAX_AGE - timedelta(days=1)
        ).isoformat()
        state_module._save_raw(raw)

        # Writing to session 20 itself refreshes updated_at before pruning
        # runs, so it must survive its own write.
        state_module.save_session_state(20, current_phase="thinking")
        raw_after = state_module._load_raw()
        self.assertIn("20", raw_after)


class LoadRawCacheTests(_StateDirFixture, unittest.TestCase):
    def test_repeated_reads_without_a_write_do_not_reparse(self):
        state_module.save_session_state(30, workspace_root="/repo")
        first = state_module._load_raw()
        second = state_module._load_raw()
        # Same object identity -- the second call returned the cached dict
        # instead of re-parsing the file from disk.
        self.assertIs(first, second)

    def test_external_write_invalidates_the_cache(self):
        state_module.save_session_state(31, workspace_root="/repo")
        state_module._load_raw()
        # Simulate a second process (e.g. `tamfis-code queue`) writing the
        # file directly, changing its mtime.
        raw = state_module._load_raw()
        raw["32"] = {"session_id": 32, "workspace_root": "/other"}
        state_module._save_raw(raw)
        reloaded = state_module._load_raw()
        self.assertIn("32", reloaded)


class ClearSessionStateTests(_StateDirFixture, unittest.TestCase):
    def test_clear_removes_hot_state_but_retains_recovery_snapshot(self):
        state_module.save_session_state(40, workspace_root="/repo")
        snapshot = state_module.CONFIG_DIR / ".memory" / "session-40.json"
        self.assertTrue(snapshot.is_file())

        self.assertTrue(state_module.clear_session_state(40))

        self.assertNotIn(40, state_module.all_known_session_ids())
        self.assertTrue(snapshot.is_file())
        self.assertFalse(state_module.clear_session_state(40))


if __name__ == "__main__":
    unittest.main()
