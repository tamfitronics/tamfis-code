"""Regression test for the real ~10-minute hang bug: _stream_task must
extract the command id from payload["command_id"] (a top-level int) and
call approve_command with it, not attempt to read payload["command"]["id"]
(payload["command"] is a plain string -- the command text -- not an
object). Reading the wrong field silently found nothing, so
approve_command() was never called and the server's wait_for_approval()
blocked for its full timeout on every real MEDIUM/DANGEROUS command.

Uses a fake client (no real network/DB) so this test runs fast and
deterministically -- it isn't a substitute for the live verification this
fix already got, just a way to make sure it can't silently regress.
"""

import asyncio
import tempfile
import unittest
from io import StringIO
from pathlib import Path

from rich.console import Console

from tamfis_code.render import StreamRenderer
from tamfis_code.runner import _stream_task
from tamfis_code import state as state_module


class FakeStreamClient:
    def __init__(self, events):
        self._events = events
        self.approve_calls: list[tuple[int, str]] = []

    async def stream_session(self, session_id, last_event_id):
        for event in self._events:
            yield event

    async def approve_command(self, command_id, decision):
        self.approve_calls.append((command_id, decision))
        return {"id": command_id, "status": "denied" if decision == "deny" else "approved"}

    async def get_task(self, task_id):
        raise AssertionError("get_task should not be called -- the stream already reached a terminal event")


class StreamTaskApprovalTests(unittest.TestCase):
    def test_canonical_payload_sequence_cannot_poison_sse_replay_cursor(self):
        events = [
            {
                # A legacy canonical frame has no SSE id and its global
                # sequence can be much larger than this session's cursor.
                "task_id": None, "sequence": 2099,
                "event_type": "tool_result", "payload": {},
            },
            {
                "task_id": "t1", "sequence": 500, "stream_sequence": 500,
                "event_type": "ai_task_completed", "payload": {"status": "completed"},
            },
        ]
        client = FakeStreamClient(events)
        console = Console(file=StringIO(), no_color=True, width=200)

        with tempfile.TemporaryDirectory() as tmp:
            original_dir, original_path = state_module.CONFIG_DIR, state_module.STATE_PATH
            state_module.CONFIG_DIR, state_module.STATE_PATH = Path(tmp), Path(tmp) / "state.json"
            try:
                outcome = asyncio.run(_stream_task(
                    client, StreamRenderer(console), console,
                    session_id=1, task_id="t1", approval_policy="ask", interactive=False,
                ))
                self.assertEqual(outcome.status, "completed")
                self.assertEqual(state_module.get_session_state(1).last_event_id, 500)
            finally:
                state_module.CONFIG_DIR, state_module.STATE_PATH = original_dir, original_path

    def test_active_task_reconnects_after_stream_closes(self):
        class ReconnectingClient(FakeStreamClient):
            def __init__(self):
                super().__init__([])
                self.stream_calls = 0

            async def stream_session(self, session_id, last_event_id):
                self.stream_calls += 1
                if self.stream_calls == 1:
                    return
                yield {
                    "task_id": "t1", "sequence": 2,
                    "event_type": "assistant_message",
                    "payload": {"visible_content": "done after reconnect"},
                }
                yield {
                    "task_id": "t1", "sequence": 3,
                    "event_type": "ai_task_completed", "payload": {"status": "completed"},
                }

            async def get_task(self, task_id):
                return {"id": task_id, "status": "running"}

        client = ReconnectingClient()
        console = Console(file=StringIO(), no_color=True, width=200)
        renderer = StreamRenderer(console)

        outcome = asyncio.run(_stream_task(
            client, renderer, console,
            session_id=1, task_id="t1", approval_policy="ask", interactive=False,
        ))

        self.assertEqual(client.stream_calls, 2)
        self.assertEqual(outcome.status, "completed")
        self.assertEqual(outcome.summary, "done after reconnect")

    def test_extracts_command_id_from_top_level_field_and_approves(self):
        events = [
            {"task_id": "t1", "event_type": "approval_required", "payload": {
                "command_id": 99, "command": "sleep 30 && echo done", "risk_level": "medium",
            }},
            {"task_id": "t1", "event_type": "ai_task_failed", "payload": {"error": "denied"}},
        ]
        client = FakeStreamClient(events)
        console = Console(file=StringIO(), no_color=True, width=200)
        renderer = StreamRenderer(console)

        outcome = asyncio.run(_stream_task(
            client, renderer, console,
            session_id=1, task_id="t1", approval_policy="never", interactive=False,
        ))

        self.assertEqual(client.approve_calls, [(99, "deny")])
        self.assertEqual(outcome.status, "failed")

    def test_same_command_id_is_only_prompted_once(self):
        events = [
            {"task_id": "t1", "event_type": "approval_required", "payload": {
                "command_id": 1, "command": "rm file", "risk_level": "medium",
            }},
            {"task_id": "t1", "event_type": "approval_required", "payload": {
                "command_id": 1, "command": "rm file", "risk_level": "medium",
            }},
            {"task_id": "t1", "event_type": "ai_task_completed", "payload": {"status": "completed"}},
        ]
        client = FakeStreamClient(events)
        console = Console(file=StringIO(), no_color=True, width=200)
        renderer = StreamRenderer(console)

        asyncio.run(_stream_task(
            client, renderer, console,
            session_id=1, task_id="t1", approval_policy="full-auto", interactive=False,
        ))

        self.assertEqual(client.approve_calls, [(1, "approve_once")])

    def test_events_for_other_tasks_on_same_session_are_ignored(self):
        events = [
            {"task_id": "other-task", "event_type": "approval_required", "payload": {
                "command_id": 5, "command": "rm -rf /", "risk_level": "dangerous",
            }},
            {"task_id": "t1", "event_type": "ai_task_completed", "payload": {"status": "completed"}},
        ]
        client = FakeStreamClient(events)
        console = Console(file=StringIO(), no_color=True, width=200)
        renderer = StreamRenderer(console)

        outcome = asyncio.run(_stream_task(
            client, renderer, console,
            session_id=1, task_id="t1", approval_policy="ask", interactive=False,
        ))

        self.assertEqual(client.approve_calls, [])  # never called -- that event belonged to a different task
        self.assertEqual(outcome.status, "completed")

    def test_events_with_no_task_id_are_ignored_not_treated_as_a_match(self):
        # Regression guard for a real bug: manually-submitted commands
        # (the `$ cmd` / `run` path) have task_id=None, not a DIFFERENT
        # task_id -- an earlier version of this filter only skipped events
        # where task_id was set AND different, letting every historical
        # manual command replay into every subsequent AI task's stream and
        # attempt a spurious re-approval of an already-resolved command.
        events = [
            {"task_id": None, "event_type": "approval_required", "payload": {
                "command_id": 7, "command": "some old manual command", "risk_level": "medium",
            }},
            {"task_id": "t1", "event_type": "ai_task_completed", "payload": {"status": "completed"}},
        ]
        client = FakeStreamClient(events)
        console = Console(file=StringIO(), no_color=True, width=200)
        renderer = StreamRenderer(console)

        outcome = asyncio.run(_stream_task(
            client, renderer, console,
            session_id=1, task_id="t1", approval_policy="ask", interactive=False,
        ))

        self.assertEqual(client.approve_calls, [])
        self.assertEqual(outcome.status, "completed")

    def test_reprioritise_instruction_cancels_current_task_at_safe_boundary(self):
        class WaitingClient(FakeStreamClient):
            def __init__(self):
                super().__init__([])
                self.cancelled = []

            async def stream_session(self, session_id, last_event_id):
                await asyncio.sleep(5)
                if False:
                    yield {}

            async def cancel_task(self, task_id):
                self.cancelled.append(task_id)
                return {"id": task_id, "status": "cancelled"}

        with tempfile.TemporaryDirectory() as tmp:
            original_dir, original_path = state_module.CONFIG_DIR, state_module.STATE_PATH
            state_module.CONFIG_DIR, state_module.STATE_PATH = Path(tmp), Path(tmp) / "state.json"
            try:
                state_module.enqueue_instruction(55, "check authentication first", classification="reprioritise", priority=1)
                client = WaitingClient()
                console = Console(file=StringIO(), no_color=True)
                outcome = asyncio.run(_stream_task(
                    client, StreamRenderer(console), console, session_id=55, task_id="task-a",
                    approval_policy="ask", interactive=False,
                ))
                self.assertEqual(outcome.status, "cancelled")
                self.assertEqual(client.cancelled, ["task-a"])
                queued = state_module.get_session_state(55).queued_user_instructions[0]
                self.assertEqual(queued["status"], "queued")
            finally:
                state_module.CONFIG_DIR, state_module.STATE_PATH = original_dir, original_path


if __name__ == "__main__":
    unittest.main()
