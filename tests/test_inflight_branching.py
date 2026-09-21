"""Mid-task messages branch instead of waiting: every one is answered at once, none is silent.

Owner report 2026-09-21: "/**** does nothing in-flight ... even other follow-ups, not just the commands ... it
should not be linear". A slash command used to reach the model as the prose "/diff" and do nothing; a plain
follow-up got one "queued" line and then silence until the task's next safe step.
"""
import asyncio
import unittest
from unittest.mock import AsyncMock, Mock

from test_live_input import _StatePatchMixin, _config, _console

from tamfis_code import state as state_module
from tamfis_code.live_input import LiveInputListener
from tamfis_code.render import StreamRenderer


def _listener(session_id, callback=None):
    renderer = StreamRenderer(_console())
    renderer.request_steering = Mock()
    return renderer, LiveInputListener(
        session_id=session_id, renderer=renderer, cli_config=_config(), side_question_callback=callback,
    )


class SlashCommandsMidTaskTests(_StatePatchMixin, unittest.IsolatedAsyncioTestCase):
    async def test_a_command_that_cannot_run_now_is_acknowledged_and_deferred_not_sent_to_the_model(self):
        renderer, listener = _listener(60)
        listener._enqueue("/diff")
        queued = state_module.get_session_state(60).queued_user_instructions
        self.assertEqual([(i["text"], i["classification"]) for i in queued], [("/diff", "command")])
        renderer.request_steering.assert_not_called()          # the running model never sees "/diff"
        self.assertIn("runs as soon as the task finishes", renderer.console.file.getvalue())

    async def test_a_deferred_command_is_not_claimed_by_the_running_turn(self):
        from tamfis_code.runner_local import _claim_live_queued_instructions

        _, listener = _listener(61)
        listener._enqueue("/diffs")
        self.assertEqual(_claim_live_queued_instructions(61), [])
        self.assertEqual(state_module.get_session_state(61).queued_user_instructions[0]["status"], "queued")

    async def test_an_unknown_command_is_reported_with_a_suggestion_and_not_queued(self):
        renderer, listener = _listener(62)
        listener._enqueue("/dif")
        self.assertEqual(state_module.get_session_state(62).queued_user_instructions, [])
        self.assertIn("Unknown command /dif", renderer.console.file.getvalue())
        self.assertIn("did you mean", renderer.console.file.getvalue())

    async def test_stop_cancels_the_running_task_at_once(self):
        interrupted = []
        renderer = StreamRenderer(_console())
        listener = LiveInputListener(
            session_id=63, renderer=renderer, cli_config=_config(), interrupt_callback=interrupted.append,
        )
        listener._enqueue("/stop")
        self.assertEqual(interrupted, ["cancel"])

    async def test_help_and_queue_answer_locally(self):
        renderer, listener = _listener(64)
        listener._enqueue("/help")
        listener._enqueue("/queue")
        out = renderer.console.file.getvalue()
        self.assertIn("run at once", out)
        self.assertIn("Nothing is queued", out)

    async def test_a_path_is_not_mistaken_for_a_command(self):
        renderer, listener = _listener(65)
        listener._enqueue("/home/x/notes.txt has the answer")
        queued = state_module.get_session_state(65).queued_user_instructions
        self.assertEqual(queued[0]["classification"], "follow_up")


class FollowUpAcknowledgementTests(_StatePatchMixin, unittest.IsolatedAsyncioTestCase):
    async def test_a_plain_follow_up_is_answered_at_once_by_a_side_branch_and_still_queued_for_the_task(self):
        answer = AsyncMock(return_value="Understood: add tests too. I'll fold that in at the next step.")
        renderer, listener = _listener(66, answer)
        listener._enqueue("also add tests for it")
        for _ in range(5):
            await asyncio.sleep(0)
        self.assertEqual(state_module.get_session_state(66).queued_user_instructions[0]["classification"], "follow_up")
        renderer.request_steering.assert_called_once()
        answer.assert_awaited_once()
        self.assertIn("also add tests for it", answer.await_args.args[0])
        out = renderer.console.file.getvalue()
        self.assertIn("Follow-up", out)
        self.assertIn("I'll fold that in", out)

    async def test_the_callback_is_told_it_is_a_follow_up_when_it_can_take_the_flag(self):
        seen = {}

        async def answer(question, *, followup=False):
            seen["followup"] = followup
            return "ok"

        _, listener = _listener(67, answer)
        listener._enqueue("switch to the other file")
        for _ in range(5):
            await asyncio.sleep(0)
        self.assertTrue(seen["followup"])

    async def test_a_failing_side_branch_leaves_the_queued_line_and_never_raises(self):
        answer = AsyncMock(side_effect=RuntimeError("provider down"))
        renderer, listener = _listener(68, answer)
        listener._enqueue("do it differently")
        for _ in range(5):
            await asyncio.sleep(0)
        out = renderer.console.file.getvalue()
        self.assertIn("Follow-up queued", out)
        self.assertNotIn("provider down", out)

    async def test_the_acknowledgement_can_be_switched_off(self):
        import os
        from unittest.mock import patch

        answer = AsyncMock(return_value="x")
        with patch.dict(os.environ, {"TAMFIS_CODE_FOLLOWUP_ACK": "0"}):
            _, listener = _listener(69, answer)
            listener._enqueue("hello there")
            await asyncio.sleep(0.01)
        answer.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
