"""A cancel queued for a PREVIOUS run must not stop the run that was auto-resumed afterwards.

Live report 2026-09-21: after `continue` the resumed run printed "Stopped (27s) ... Error: Task canceld by
user request." although the user had not cancelled it -- a leftover cancel from an earlier process.
"""
import tempfile
import time
import unittest
from pathlib import Path

from tamfis_code import state as state_module
from tamfis_code.runner_local import _apply_live_queued_instruction, _claim_live_queued_instructions


class _Renderer:
    def __init__(self):
        self.events = []

    def handle_event(self, event):
        self.events.append(event)


class StaleStopRequestTests(unittest.TestCase):
    def setUp(self):
        self._orig = (state_module.CONFIG_DIR, state_module.STATE_PATH)
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        state_module.CONFIG_DIR = base / ".config"
        state_module.STATE_PATH = base / ".config" / "state.json"

    def tearDown(self):
        state_module.CONFIG_DIR, state_module.STATE_PATH = self._orig
        self.tmp.cleanup()

    def _status(self, session_id, item_id):
        return next(i["status"] for i in state_module.get_session_state(session_id).queued_user_instructions if i["id"] == item_id)

    def test_a_cancel_from_before_this_run_is_discarded(self):
        stale = state_module.enqueue_instruction(1, "", classification="cancel")
        time.sleep(0.01)
        since = state_module._now()                       # the (resumed) run starts now
        self.assertEqual(_claim_live_queued_instructions(1, since=since), [])
        self.assertEqual(self._status(1, stale.id), "completed")

    def test_a_cancel_made_during_this_run_still_stops_it_including_from_another_terminal(self):
        since = state_module._now()
        time.sleep(0.01)
        live = state_module.enqueue_instruction(1, "", classification="cancel")
        claimed = _claim_live_queued_instructions(1, since=since)
        self.assertEqual([c["id"] for c in claimed], [live.id])
        outcome = _apply_live_queued_instruction(claimed[0], session_id=1, working_messages=[], renderer=_Renderer())
        self.assertEqual(outcome.status, "cancelled")
        self.assertIn("cancelled by user request", outcome.error)
        self.assertNotIn("canceld", outcome.error)

    def test_old_pause_and_exit_requests_expire_too_but_old_follow_ups_still_apply(self):
        pause = state_module.enqueue_instruction(1, "", classification="pause")
        exit_ = state_module.enqueue_instruction(1, "", classification="exit")
        note = state_module.enqueue_instruction(1, "also check the login page", classification="follow_up")
        time.sleep(0.01)
        claimed = _claim_live_queued_instructions(1, since=state_module._now())
        self.assertEqual([c["id"] for c in claimed], [note.id])   # typed while the app was closed: still wanted
        self.assertEqual(self._status(1, pause.id), "completed")
        self.assertEqual(self._status(1, exit_.id), "completed")

    def test_without_a_start_time_behaviour_is_unchanged(self):
        item = state_module.enqueue_instruction(1, "", classification="cancel")
        self.assertEqual([c["id"] for c in _claim_live_queued_instructions(1)], [item.id])

    def test_past_tense_wording(self):
        for classification, word in (("pause", "paused"), ("cancel", "cancelled")):
            item = state_module.enqueue_instruction(2, "", classification=classification)
            outcome = _apply_live_queued_instruction(dict(id=item.id, text="", classification=classification),
                                                     session_id=2, working_messages=[], renderer=_Renderer())
            self.assertIn(word, outcome.error)


if __name__ == "__main__":
    unittest.main()
