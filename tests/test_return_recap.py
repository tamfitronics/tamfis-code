"""The short "Conversation recap" shown when coming back to a session (resume, /recap, idle-away)."""
import asyncio
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

from tamfis_code import state as st
from tamfis_code.return_recap import (
    NO_NEXT_STEP, away_threshold_seconds, build_return_recap, print_return_recap, render_return_recap,
)


class RecapTests(unittest.TestCase):
    def setUp(self):
        self._orig = (st.CONFIG_DIR, st.STATE_PATH)
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        st.CONFIG_DIR = base / ".config"
        st.STATE_PATH = base / ".config" / "state.json"

    def tearDown(self):
        st.CONFIG_DIR, st.STATE_PATH = self._orig
        self.tmp.cleanup()

    def _history(self, sid, objective, answer, **extra):
        st.save_session_state(sid, conversation_history=[
            {"role": "user", "content": objective}, {"role": "assistant", "content": answer},
        ], **extra)

    def test_an_empty_session_has_no_recap(self):
        self.assertIsNone(build_return_recap(1))
        console = Console(file=io.StringIO())
        self.assertFalse(print_return_recap(console, 1))

    def test_objective_where_it_stands_and_an_honest_missing_next_step(self):
        self._history(2, "Build a geo-aware BetPredict operator list for five bookmakers.",
                      "A first compliant version was built and deployed with timezone detection. It has a manual override.")
        recap = build_return_recap(2)
        self.assertIn("geo-aware BetPredict operator list", recap.objective)
        self.assertIn("first compliant version was built", recap.standing)
        self.assertEqual(recap.next_step, NO_NEXT_STEP)

    def test_a_recorded_next_step_in_the_answer_is_used_never_invented(self):
        self._history(3, "Fix login", "Fixed the token bug.\nNext step: add a regression test for expired tokens.")
        self.assertIn("regression test", build_return_recap(3).next_step)

    def test_an_interrupted_run_and_a_plan_show_progress_and_the_pending_step(self):
        self._history(4, "Migrate the database", "Started the migration.", execution_status="interrupted",
                      saved_plans=[{"id": "p1", "objective": "Migrate the database to v2", "steps": [
                          {"status": "completed", "description": "backup"},
                          {"status": "pending", "description": "run migration 0042"}]}],
                      active_plan_id="p1")
        recap = build_return_recap(4)
        self.assertEqual(recap.objective, "Migrate the database to v2")
        self.assertIn("the last run interrupted", recap.standing)
        self.assertIn("plan 1/2 steps done", recap.standing)
        self.assertEqual(recap.next_step, "run migration 0042")

    def test_changed_files_are_named_but_reverted_ones_are_not(self):
        self._history(5, "Edit stuff", "Done.", modified_files=[
            {"path": "/w/a.py", "revert_status": "reverted"}, {"path": "/w/b.py"}])
        recap = build_return_recap(5)
        self.assertIn("changed b.py", recap.standing)
        self.assertNotIn("a.py", recap.standing)

    def test_render_prints_the_titled_block(self):
        self._history(6, "Do a thing", "It is done.")
        console = Console(file=io.StringIO(), width=100)
        render_return_recap(console, build_return_recap(6))
        out = console.file.getvalue()
        self.assertIn("Conversation recap", out)
        self.assertIn("Objective: Do a thing", out)
        self.assertIn(f"Next: {NO_NEXT_STEP}", out)

    def test_the_away_threshold_is_configurable_and_can_be_disabled(self):
        with patch.dict(os.environ, {"TAMFIS_CODE_AWAY_RECAP_MINUTES": "0"}):
            self.assertEqual(away_threshold_seconds(), 0.0)
        with patch.dict(os.environ, {"TAMFIS_CODE_AWAY_RECAP_MINUTES": "2"}):
            self.assertEqual(away_threshold_seconds(), 120.0)
        with patch.dict(os.environ, {"TAMFIS_CODE_AWAY_RECAP_MINUTES": "junk"}):
            self.assertEqual(away_threshold_seconds(), 600.0)


class AwayWatcherTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._orig = (st.CONFIG_DIR, st.STATE_PATH)
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        st.CONFIG_DIR = base / ".config"
        st.STATE_PATH = base / ".config" / "state.json"
        st.save_session_state(9, conversation_history=[
            {"role": "user", "content": "Ship it"}, {"role": "assistant", "content": "Shipped."}])

    def tearDown(self):
        st.CONFIG_DIR, st.STATE_PATH = self._orig
        self.tmp.cleanup()

    async def _run(self, buffer_text):
        from types import SimpleNamespace

        from tamfis_code.interactive import _show_recap_when_away

        printed = []

        async def fake_run_in_terminal(func):
            printed.append(func)
            func()

        session = SimpleNamespace(default_buffer=SimpleNamespace(text=buffer_text), app=SimpleNamespace(is_running=True))
        console = Console(file=io.StringIO(), width=100)
        real_sleep = asyncio.sleep

        async def instant_sleep(*_a, **_k):
            await real_sleep(0)

        with patch.dict(os.environ, {"TAMFIS_CODE_AWAY_RECAP_MINUTES": "0.0005"}), \
             patch("asyncio.sleep", new=instant_sleep), \
             patch("prompt_toolkit.application.run_in_terminal", new=fake_run_in_terminal):
            await _show_recap_when_away(9, session, console)
        return printed, console.file.getvalue()

    async def test_an_idle_prompt_prints_the_recap_once(self):
        printed, out = await self._run("")
        self.assertEqual(len(printed), 1)
        self.assertIn("Conversation recap", out)

    async def test_a_user_who_is_typing_is_not_interrupted(self):
        printed, out = await self._run("half a message")
        self.assertEqual(printed, [])
        self.assertEqual(out, "")


if __name__ == "__main__":
    unittest.main()
