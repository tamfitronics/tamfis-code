#!/usr/bin/env python3
"""Tests for tamfis-code's thread compression/summarization (state.py) and
the deep self-health-check (doctor.py) -- the Claude Code/Codex-parity
features that let a long REPL thread stay light in the terminal and let the
CLI verify its own subsystems actually work.

state.py's compact_session_thread/summarize_thread are pure functions over
durable SessionState, so they're tested directly without a provider or
network. doctor.py's _diagnose_self_health is tested against a temp
CONFIG_DIR so it never touches the real ~/.config/tamfis-code.
"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tamfis_code import state as state_module
from tamfis_code.doctor import _diagnose_self_health, CheckResult


class _StateDirFixture:
    """Redirect state.py's CONFIG_DIR/STATE_PATH to a temp dir, the same
    convention test_doctor_session_diagnostics.py uses."""

    def setUp(self):
        self._state_originals = (state_module.CONFIG_DIR, state_module.STATE_PATH)
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        state_module.CONFIG_DIR = base / ".config"
        state_module.STATE_PATH = base / ".config" / "state.json"

    def tearDown(self):
        state_module.CONFIG_DIR, state_module.STATE_PATH = self._state_originals
        self._tmp.cleanup()


class SummarizeThreadTests(_StateDirFixture, unittest.TestCase):
    def _seed_history(self, session_id, turns):
        """Build conversation_history from (objective, answer) pairs."""
        history = []
        for objective, answer in turns:
            history.append({"role": "user", "content": objective})
            if answer:
                history.append({"role": "assistant", "content": answer})
        state_module.save_session_state(session_id, conversation_history=history)

    def test_empty_session_returns_no_conversation_message(self):
        recap = state_module.summarize_thread(1)
        self.assertIn("No conversation", recap)

    def test_short_thread_preserves_all_turns_verbatim(self):
        self._seed_history(2, [("fix the bug", "done"), ("add tests", "added")])
        recap = state_module.summarize_thread(2)
        self.assertIn("fix the bug", recap)
        self.assertIn("done", recap)
        self.assertIn("add tests", recap)
        self.assertIn("added", recap)
        # A short thread (2 turns, under the keep_recent default) should not
        # label anything as "earlier turn(s)".
        self.assertNotIn("earlier turn", recap)

    def test_long_thread_folds_older_turns_into_summary(self):
        turns = [(f"objective {i}", f"answer {i}") for i in range(10)]
        self._seed_history(3, turns)
        recap = state_module.summarize_thread(3, keep_recent=4)
        # The recent 4 turns are preserved verbatim.
        self.assertIn("objective 9", recap)
        self.assertIn("answer 9", recap)
        self.assertIn("objective 6", recap)
        # Older turns are folded into a summary section, not verbatim.
        self.assertIn("Summary of 6 earlier turn", recap)
        self.assertIn("objective 0", recap)
        self.assertIn("objective 5", recap)

    def test_summarize_includes_modified_files_and_plan_progress(self):
        self._seed_history(4, [("do work", "did it")])
        state_module.save_session_state(
            4,
            modified_files=[{"path": "/repo/src/app.py", "lines_added": 5, "lines_removed": 1}],
            saved_plans=[{
                "id": "plan_x", "objective": "build it",
                "steps": [
                    {"step": "a", "status": "completed"},
                    {"step": "b", "status": "pending"},
                ],
            }],
            active_plan_id="plan_x",
        )
        recap = state_module.summarize_thread(4)
        self.assertIn("src/app.py", recap)
        self.assertIn("plan_x", recap)
        self.assertIn("1/2 steps done", recap)


class CompactSessionThreadTests(_StateDirFixture, unittest.TestCase):
    def _seed_history(self, session_id, turns):
        history = []
        for objective, answer in turns:
            history.append({"role": "user", "content": objective})
            if answer:
                history.append({"role": "assistant", "content": answer})
        state_module.save_session_state(session_id, conversation_history=history)

    def test_compact_folds_older_turns_and_keeps_recent(self):
        turns = [(f"obj {i}", f"ans {i}") for i in range(8)]
        self._seed_history(5, turns)
        recap = state_module.compact_session_thread(5, keep_recent=3)
        # After compaction, conversation_history should only contain the
        # recent 3 turns (6 messages: 3 user + 3 assistant).
        state = state_module.get_session_state(5)
        self.assertEqual(len(state.conversation_history), 6)
        self.assertIn("obj 7", state.conversation_history[-2]["content"])
        self.assertIn("ans 7", state.conversation_history[-1]["content"])
        # The older turns are folded into conversation_summary.
        self.assertIn("obj 0", state.conversation_summary)
        self.assertIn("obj 4", state.conversation_summary)
        # The recap reflects the post-compaction state: only the recent 3
        # turns remain in history, so the recap shows them (the older turns
        # are now in conversation_summary, asserted above).
        self.assertIn("obj 7", recap)
        self.assertIn("Recent 3 turn", recap)

    def test_compact_short_thread_is_a_noop_on_history(self):
        self._seed_history(6, [("one", "1"), ("two", "2")])
        recap = state_module.compact_session_thread(6, keep_recent=4)
        state = state_module.get_session_state(6)
        # Nothing to fold -- history is unchanged.
        self.assertEqual(len(state.conversation_history), 4)
        self.assertIn("one", recap)
        self.assertIn("two", recap)

    def test_compact_is_idempotent(self):
        turns = [(f"obj {i}", f"ans {i}") for i in range(10)]
        self._seed_history(7, turns)
        first = state_module.compact_session_thread(7, keep_recent=3)
        state_after_first = state_module.get_session_state(7)
        second = state_module.compact_session_thread(7, keep_recent=3)
        state_after_second = state_module.get_session_state(7)
        # A second compact on the already-compacted thread doesn't lose more
        # recent turns or duplicate the summary.
        self.assertEqual(len(state_after_second.conversation_history), 6)
        self.assertEqual(state_after_first.conversation_summary, state_after_second.conversation_summary)


class DiagnoseSelfHealthTests(_StateDirFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self._workspace_tmp = tempfile.TemporaryDirectory()
        self.workspace_root = Path(self._workspace_tmp.name)

    def tearDown(self):
        super().tearDown()
        self._workspace_tmp.cleanup()

    def test_fresh_environment_reports_state_and_journal_as_pass_or_warning(self):
        results = _diagnose_self_health(self.workspace_root)
        by_name = {r.name: r for r in results}
        # State writability should pass in a fresh temp dir.
        self.assertEqual(by_name["Session state writability"].status, "PASS")
        # No executions yet is a warning, not a failure.
        self.assertEqual(by_name["Runtime execution journal"].status, "WARNING")
        # Evidence store dir is created on demand.
        self.assertEqual(by_name["Context-rollover evidence store"].status, "PASS")
        # The core tool set must be registered.
        self.assertEqual(by_name["Local tool registry"].status, "PASS")

    def test_missing_core_tool_fails(self):
        # Patch MCPServer so a core tool is missing.
        from tamfis_code import mcp as mcp_module
        original_init = mcp_module.MCPServer.__init__

        def _broken_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            # Simulate edit_file being unavailable.
            if hasattr(self, "tools") and "edit_file" in self.tools:
                del self.tools["edit_file"]

        mcp_module.MCPServer.__init__ = _broken_init
        try:
            results = _diagnose_self_health(self.workspace_root)
        finally:
            mcp_module.MCPServer.__init__ = original_init
        by_name = {r.name: r for r in results}
        self.assertEqual(by_name["Local tool registry"].status, "FAIL")
        self.assertIn("edit_file", by_name["Local tool registry"].detail)


if __name__ == "__main__":
    unittest.main()