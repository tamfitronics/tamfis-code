import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from tamfis_code import config as config_module
from tamfis_code import state as state_module
from tamfis_code.cli import _resume_interrupted_task_if_any


class ResumeInterruptedTaskTests(unittest.TestCase):
    """cli.py's _resume_interrupted_task_if_any: on interactive startup,
    read back the same realtime .memory/state.json mirror this session was
    already writing continuously, and either reattach (task still active
    server-side) or report the outcome once (task ended while no CLI was
    attached) -- instead of silently landing at a blank prompt either way.
    """

    def setUp(self):
        self._config_originals = (
            config_module.CONFIG_DIR, config_module.CREDENTIALS_PATH, config_module.USER_CONFIG_PATH,
        )
        self._state_originals = (state_module.CONFIG_DIR, state_module.STATE_PATH)
        self.tmp = tempfile.TemporaryDirectory()
        tmp_path = Path(self.tmp.name)
        config_module.CONFIG_DIR = tmp_path
        config_module.CREDENTIALS_PATH = tmp_path / "credentials.json"
        config_module.USER_CONFIG_PATH = tmp_path / "config.toml"
        state_module.CONFIG_DIR = tmp_path / "state"
        state_module.STATE_PATH = tmp_path / "state" / "state.json"

    def tearDown(self):
        (config_module.CONFIG_DIR, config_module.CREDENTIALS_PATH,
         config_module.USER_CONFIG_PATH) = self._config_originals
        state_module.CONFIG_DIR, state_module.STATE_PATH = self._state_originals
        self.tmp.cleanup()

    def _workspace(self, session_id=50):
        workspace = MagicMock()
        workspace.session_id = session_id
        return workspace

    def test_idle_session_does_nothing(self):
        state_module.save_session_state(50, execution_status="idle")
        client = AsyncMock()
        console = MagicMock()
        asyncio.run(_resume_interrupted_task_if_any(client, console, MagicMock(), self._workspace()))
        client.get_task.assert_not_called()
        console.print.assert_not_called()

    def test_still_running_task_reattaches(self):
        state_module.save_session_state(
            50, execution_status="running", last_task_id="t-1",
            active_task={"id": "t-1", "objective": "fix the bug"},
        )
        client = AsyncMock()
        client.get_task.return_value = {"status": "running", "objective": "fix the bug"}
        console = MagicMock()
        with patch("tamfis_code.cli.attach_and_stream", new=AsyncMock()) as attach_mock:
            asyncio.run(_resume_interrupted_task_if_any(client, console, MagicMock(), self._workspace()))
        attach_mock.assert_awaited_once()
        _, kwargs = attach_mock.call_args
        self.assertEqual(kwargs["session_id"], 50)
        self.assertEqual(kwargs["task_id"], "t-1")

    def test_task_finished_while_away_reports_and_clears_state(self):
        state_module.save_session_state(
            50, execution_status="backgrounded", last_task_id="t-2",
            active_task={"id": "t-2", "objective": "add tests"},
        )
        client = AsyncMock()
        client.get_task.return_value = {"status": "completed", "objective": "add tests"}
        console = MagicMock()
        with patch("tamfis_code.cli.attach_and_stream", new=AsyncMock()) as attach_mock:
            asyncio.run(_resume_interrupted_task_if_any(client, console, MagicMock(), self._workspace()))
        attach_mock.assert_not_called()
        console.print.assert_called_once()
        self.assertIn("completed", console.print.call_args[0][0])
        state = state_module.get_session_state(50)
        self.assertEqual(state.execution_status, "completed")
        self.assertIsNone(state.active_task)

    def test_unreachable_server_fails_silent_not_fatal(self):
        from tamfis_code.api_client import RemoteAPIError

        state_module.save_session_state(
            50, execution_status="running", last_task_id="t-3",
            active_task={"id": "t-3", "objective": "x"},
        )
        client = AsyncMock()
        client.get_task.side_effect = RemoteAPIError(503, "offline")
        console = MagicMock()
        # Must not raise -- an unreachable server on startup shouldn't block
        # the ordinary interactive REPL from opening at all.
        asyncio.run(_resume_interrupted_task_if_any(client, console, MagicMock(), self._workspace()))


if __name__ == "__main__":
    unittest.main()
