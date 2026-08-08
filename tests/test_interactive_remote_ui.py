import unittest
from unittest.mock import MagicMock, patch

from tamfis_code.config import Config
from tamfis_code.interactive import _run_remote_turn_with_live_ui
from tamfis_code.runner import TaskOutcome


class RemoteTurnLiveUITests(unittest.IsolatedAsyncioTestCase):
    async def test_remote_turn_starts_and_stops_live_footer(self):
        listener = MagicMock()
        listener.stop.return_value = None
        outcome = TaskOutcome(status="completed", summary="done")

        async def turn():
            return outcome

        with patch("tamfis_code.interactive.LiveInputListener", return_value=listener):
            result = await _run_remote_turn_with_live_ui(
                session_id=55,
                renderer=MagicMock(),
                config=Config(),
                turn_coro=turn(),
            )

        self.assertIs(result, outcome)
        listener.start.assert_called_once_with()
        listener.set_outcome_status.assert_called_once_with("completed")
        listener.stop.assert_called_once_with()

    async def test_remote_turn_marks_footer_failed_on_exception(self):
        listener = MagicMock()
        listener.stop.return_value = None

        async def turn():
            raise RuntimeError("stream failed")

        with patch("tamfis_code.interactive.LiveInputListener", return_value=listener):
            with self.assertRaisesRegex(RuntimeError, "stream failed"):
                await _run_remote_turn_with_live_ui(
                    session_id=55,
                    renderer=MagicMock(),
                    config=Config(),
                    turn_coro=turn(),
                )

        listener.set_outcome_status.assert_called_once_with("failed")
        listener.stop.assert_called_once_with()
