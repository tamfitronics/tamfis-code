"""The session store stays small: byte budgets, cold-session compaction, and revert safety.

Live report 2026-09-21: the real state.json had grown to 62 MB (single sessions of 3-5 MB) because the
caps counted entries, not bytes, and every save rewrote all of it. Codex/Claude Code never rewrite history.
"""
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tamfis_code import state as st
from tamfis_code.safety import record_mutation, revert_mutation


def _iso(**delta):
    return (datetime.now(timezone.utc) - timedelta(**delta)).isoformat()


class StoreCompactionTests(unittest.TestCase):
    def setUp(self):
        self._orig = (st.CONFIG_DIR, st.STATE_PATH)
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        st.CONFIG_DIR = base / ".config"
        st.STATE_PATH = base / ".config" / "state.json"
        self.base = base

    def tearDown(self):
        st.CONFIG_DIR, st.STATE_PATH = self._orig
        self.tmp.cleanup()

    def test_only_the_newest_revert_bodies_survive_a_budget(self):
        entries = [
            {"mutation_id": f"m{i}", "path": f"/x/{i}", "original_content": "a" * 100_000, "unified_diff": "d" * 50_000}
            for i in range(5)
        ]
        out = st._budget_revert_bodies(entries, 300_000)  # room for two 150k entries
        trimmed = [e["mutation_id"] for e in out if e.get("body_trimmed")]
        self.assertEqual(trimmed, ["m0", "m1", "m2"])
        self.assertEqual(out[4]["original_content"], "a" * 100_000)
        self.assertIsNone(out[0]["original_content"])
        self.assertEqual(out[0]["path"], "/x/0")  # metadata is kept

    def test_a_trimmed_mutation_refuses_to_revert_instead_of_deleting_the_file(self):
        target = self.base / "keep.txt"
        target.write_text("precious\n")
        record_mutation(5, path=str(target), operation="update", original_content="before\n", new_content="precious\n")
        state = st.get_session_state(5)
        mutation_id = state.modified_files[0]["mutation_id"]
        state.modified_files[0].update(original_content=None, unified_diff="", body_trimmed=True)
        st.put_session_state(state)
        with self.assertRaises(ValueError) as caught:
            revert_mutation(5, mutation_id)
        self.assertIn("trimmed", str(caught.exception))
        self.assertTrue(target.exists())
        self.assertEqual(target.read_text(), "precious\n")

    def test_recent_mutations_stay_revertible_through_a_save(self):
        target = self.base / "f.txt"
        target.write_text("after\n")
        record_mutation(6, path=str(target), operation="update", original_content="before\n", new_content="after\n")
        state = st.get_session_state(6)
        st.put_session_state(state)  # a later save must not trim a small, recent pre-image
        revert_mutation(6, st.get_session_state(6).modified_files[0]["mutation_id"])
        self.assertEqual(target.read_text(), "before\n")

    def test_older_checkpoints_drop_their_embedded_ledger(self):
        cps = [{"checkpoint_id": str(i), "task_state": {"big": "x" * 1000}} for i in range(8)]
        out = st._slim_checkpoints(cps, 3)
        self.assertEqual([("task_state" in c) for c in out], [False] * 5 + [True] * 3)
        self.assertEqual(out[0]["checkpoint_id"], "0")

    def test_long_strings_are_clipped_with_a_marker(self):
        clipped = st._clip_strings({"stdout": "z" * 10_000, "n": 3, "l": ["ok", "y" * 5_000]}, 1_000)
        self.assertLess(len(clipped["stdout"]), 1_200)
        self.assertIn("chars trimmed", clipped["stdout"])
        self.assertEqual(clipped["n"], 3)
        self.assertEqual(clipped["l"][0], "ok")

    def test_an_idle_session_is_compacted_once_on_the_next_save_of_another_session(self):
        cold = st.SessionState(session_id=1, execution_status="completed")
        cold.modified_files = [{"mutation_id": "a", "path": "/p", "original_content": "o" * 90_000, "unified_diff": "d" * 90_000}]
        cold.turn_checkpoint = {"messages": [{"content": "m"}] * 40, "status": "completed"}
        cold.context_checkpoints = [{"checkpoint_id": str(i), "task_state": {"k": "v" * 500}} for i in range(20)]
        st.put_session_state(cold)
        raw = json.loads(st.STATE_PATH.read_text())
        raw["1"]["updated_at"] = _iso(days=2)
        st.STATE_PATH.write_text(json.dumps(raw))
        st._STATE_CACHE = None

        st.put_session_state(st.SessionState(session_id=2))  # another session saves; session 1 goes cold
        row = json.loads(st.STATE_PATH.read_text())["1"]
        self.assertEqual(row[st.COLD_COMPACT_MARKER], 1)
        self.assertTrue(row["modified_files"][0]["body_trimmed"])
        self.assertIsNone(row["turn_checkpoint"])
        self.assertEqual(len(row["context_checkpoints"]), st.COLD_CHECKPOINTS_KEPT)
        self.assertNotIn("task_state", row["context_checkpoints"][0])

        # the marker is a private detail: the row still loads, and touching the session rebuilds it hot
        loaded = st.get_session_state(1)
        self.assertEqual(loaded.session_id, 1)
        st.put_session_state(loaded)
        self.assertNotIn(st.COLD_COMPACT_MARKER, json.loads(st.STATE_PATH.read_text())["1"])

    def test_a_fresh_session_is_not_treated_as_cold(self):
        st.put_session_state(st.SessionState(session_id=3, modified_files=[
            {"mutation_id": "a", "path": "/p", "original_content": "o" * 1000, "unified_diff": "d"},
        ]))
        st.put_session_state(st.SessionState(session_id=4))
        row = json.loads(st.STATE_PATH.read_text())["3"]
        self.assertNotIn(st.COLD_COMPACT_MARKER, row)
        self.assertEqual(row["modified_files"][0]["original_content"], "o" * 1000)

    def test_the_file_is_written_compact(self):
        st.put_session_state(st.SessionState(session_id=7))
        text = st.STATE_PATH.read_text()
        self.assertNotIn("\n  ", text)
        self.assertEqual(json.loads(text)["7"]["session_id"], 7)


if __name__ == "__main__":
    unittest.main()
