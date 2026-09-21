"""Launching the CLI must not silently re-run an OLD interrupted plan.

Live report 2026-09-21: a bare `tamfis-code` resumed an old session and auto-queued `continue`, so a stale
plan ("Backup tamfitronics database") started executing before the user had typed anything.
"""
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tamfis_code import state as state_module
from tamfis_code.cli import (
    AUTO_CONTINUE_WINDOW_SECONDS, _age_words, _interruption_time, _mark_auto_resumed, _should_auto_continue,
)


def _ago(seconds):
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()


class AutoContinueTests(unittest.TestCase):
    def test_a_recent_interruption_recovers_on_its_own(self):
        self.assertTrue(_should_auto_continue(_ago(60)))
        self.assertTrue(_should_auto_continue(_ago(AUTO_CONTINUE_WINDOW_SECONDS - 30)))

    def test_an_old_interruption_waits_for_the_user(self):
        self.assertFalse(_should_auto_continue(_ago(AUTO_CONTINUE_WINDOW_SECONDS + 60)))
        self.assertFalse(_should_auto_continue(_ago(6 * 3600)))
        self.assertFalse(_should_auto_continue(_ago(3 * 86400)))

    def test_unknown_or_garbage_timestamps_never_auto_run(self):
        for value in (None, "", "not a date"):
            self.assertFalse(_should_auto_continue(value))

    def test_naive_timestamps_are_read_as_utc(self):
        naive = (datetime.now(timezone.utc) - timedelta(seconds=30)).replace(tzinfo=None).isoformat()
        self.assertTrue(_should_auto_continue(naive))

    def test_the_window_is_configurable_and_zero_disables_it(self):
        with patch.dict(os.environ, {"TAMFIS_CODE_AUTO_RESUME_WINDOW": "0"}):
            self.assertFalse(_should_auto_continue(_ago(1)))
        with patch.dict(os.environ, {"TAMFIS_CODE_AUTO_RESUME_WINDOW": "3600"}):
            self.assertTrue(_should_auto_continue(_ago(1800)))
        with patch.dict(os.environ, {"TAMFIS_CODE_AUTO_RESUME_WINDOW": "junk"}):
            self.assertTrue(_should_auto_continue(_ago(60)))            # falls back to the default

    def test_age_wording(self):
        self.assertEqual(_age_words(_ago(5)), "moments ago")
        self.assertEqual(_age_words(_ago(600)), "10 minutes ago")
        self.assertEqual(_age_words(_ago(7200)), "2 hours ago")
        self.assertEqual(_age_words(_ago(86400)), "1 day ago")
        self.assertEqual(_age_words(None), "earlier")


class AnchorAndOneShotTests(unittest.TestCase):
    """Live report 2026-09-21 (again): the resume still auto-continued. The rule read the session's
    ``updated_at``, which EVERY launch refreshes, so an hours-old session always looked "recent"."""

    def setUp(self):
        self._orig = state_module.CONFIG_DIR
        self.tmp = tempfile.TemporaryDirectory()
        state_module.CONFIG_DIR = Path(self.tmp.name)

    def tearDown(self):
        state_module.CONFIG_DIR = self._orig
        self.tmp.cleanup()

    def test_the_clock_is_the_checkpoints_not_the_sessions(self):
        old_checkpoint = _ago(3 * 3600)
        state = SimpleNamespace(updated_at=_ago(5), turn_checkpoint={"updated_at": old_checkpoint})
        self.assertEqual(_interruption_time(state), old_checkpoint)
        self.assertFalse(_should_auto_continue(_interruption_time(state)))       # hours old: wait for the user
        self.assertIsNone(_interruption_time(SimpleNamespace(updated_at=_ago(5), turn_checkpoint={})))
        self.assertFalse(_should_auto_continue(_interruption_time(SimpleNamespace(updated_at=_ago(5), turn_checkpoint=None))))

    def test_one_automatic_resume_per_session_per_window(self):
        recent = _ago(60)
        self.assertTrue(_should_auto_continue(recent, session_id=7))
        _mark_auto_resumed(7)
        self.assertFalse(_should_auto_continue(recent, session_id=7))            # user stopped it and relaunched
        self.assertTrue(_should_auto_continue(recent, session_id=8))             # other sessions unaffected
        later = datetime.now(timezone.utc) + timedelta(seconds=AUTO_CONTINUE_WINDOW_SECONDS + 5)
        self.assertTrue(_should_auto_continue((later - timedelta(seconds=30)).isoformat(), session_id=7, now=later))

    def test_a_broken_marker_file_never_blocks_or_breaks(self):
        (Path(self.tmp.name) / "auto_resume.json").write_text("not json")
        self.assertTrue(_should_auto_continue(_ago(60), session_id=1))


if __name__ == "__main__":
    unittest.main()
