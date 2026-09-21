"""Saving one session must not re-sanitize every OTHER session.

Measured 2026-09-21: one state save cost ~7 s of blocking CPU on a 63 MB / 94-session store. The per-row
sanitize cache is keyed by object identity, so whenever another process rewrote state.json this process
re-parsed it and re-ran the redaction regexes over every row.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tamfis_code import state as state_module


class StateSaveCostTests(unittest.TestCase):
    def setUp(self):
        self._orig = (state_module.CONFIG_DIR, state_module.STATE_PATH)
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        state_module.CONFIG_DIR = base / ".config"
        state_module.STATE_PATH = base / ".config" / "state.json"
        state_module._STATE_CACHE, state_module._STATE_CACHE_KEY = None, None
        state_module._SANITIZED_ROW_CACHE.clear()

    def tearDown(self):
        state_module.CONFIG_DIR, state_module.STATE_PATH = self._orig
        state_module._STATE_CACHE, state_module._STATE_CACHE_KEY = None, None
        state_module._SANITIZED_ROW_CACHE.clear()
        self.tmp.cleanup()

    def _seed(self, count=30):
        for sid in range(1, count + 1):
            state_module.save_session_state(sid, workspace_root="/w", session_title=f"session {sid}")

    def _other_process_rewrites_the_file(self):
        """Another process saved: the file changes underneath this one (new mtime, same content shape)."""
        raw = json.loads(state_module.STATE_PATH.read_text())
        raw["1"]["session_title"] = "changed elsewhere"
        state_module.STATE_PATH.write_text(json.dumps(raw))
        import os
        stat = state_module.STATE_PATH.stat()
        os.utime(state_module.STATE_PATH, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))

    def test_a_save_after_another_process_wrote_only_sanitizes_the_row_being_saved(self):
        self._seed(30)
        self._other_process_rewrites_the_file()          # cold: this process must re-parse the file
        real = state_module._sanitize
        calls = []

        def counting(value):
            if isinstance(value, dict) and "workspace_root" in value:   # a whole session row (not a nested value)
                calls.append(value.get("session_title"))
            return real(value)

        with patch.object(state_module, "_sanitize", counting):
            state_module.save_session_state(2, workspace_root="/w", session_title="edited here")
        # Only the session being saved (save_session_state writes twice): never the other 29.
        self.assertEqual(set(calls), {"edited here"}, "other sessions were re-sanitized after another process wrote the file")
        self.assertLessEqual(len(calls), 2)

    def test_the_row_being_written_is_still_sanitized(self):
        self._seed(3)
        self._other_process_rewrites_the_file()
        state_module.save_session_state(2, workspace_root="/w", session_title="deploy with password=SuperSecret123456")
        on_disk = state_module.STATE_PATH.read_text()
        self.assertNotIn("SuperSecret123456", on_disk)

    def test_other_sessions_survive_the_save_unchanged(self):
        self._seed(5)
        self._other_process_rewrites_the_file()
        state_module.save_session_state(3, workspace_root="/w", session_title="edited")
        raw = json.loads(state_module.STATE_PATH.read_text())
        self.assertEqual(raw["1"]["session_title"], "changed elsewhere")     # the other process's write is kept
        self.assertEqual(raw["3"]["session_title"], "edited")
        self.assertEqual(sorted(raw), ["1", "2", "3", "4", "5"])

    def test_a_row_edited_in_place_is_sanitized_again(self):
        self._seed(2)
        state_module._SANITIZED_ROW_CACHE.clear()
        state_module._STATE_CACHE, state_module._STATE_CACHE_KEY = None, None
        state_module._load_raw()                                    # rows now trusted as already sanitized
        self.assertIn("1", state_module._SANITIZED_ROW_CACHE)
        self.assertTrue(state_module.reset_session_task_state(1))    # edits row 1 in place, then saves
        cached = state_module._SANITIZED_ROW_CACHE.get("1")
        self.assertIsNotNone(cached)
        self.assertIsNot(cached[1], state_module._load_raw().get("1"))   # re-sanitized: a fresh copy, not the trusted row


if __name__ == "__main__":
    unittest.main()
