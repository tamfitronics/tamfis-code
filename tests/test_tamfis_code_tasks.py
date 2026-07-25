import asyncio
import unittest

from tamfis_code.tasks import find_recent_task


class FakeThreadClient:
    def __init__(self, messages, tasks_by_id):
        self._messages = messages
        self._tasks = tasks_by_id

    async def get_thread(self, session_id, after_sequence=0):
        return {"messages": self._messages}

    async def get_task(self, task_id):
        return self._tasks[task_id]


class FindRecentTaskTests(unittest.TestCase):
    def test_finds_most_recent_task_matching_status(self):
        messages = [
            {"task_id": "t1", "role": "user"},
            {"task_id": "t1", "role": "assistant"},
            {"task_id": "t2", "role": "user"},
            {"task_id": "t2", "role": "assistant"},
            {"task_id": "t3", "role": "user"},
        ]
        tasks = {
            "t1": {"id": "t1", "status": "completed"},
            "t2": {"id": "t2", "status": "failed"},
            "t3": {"id": "t3", "status": "running"},
        }
        client = FakeThreadClient(messages, tasks)
        result = asyncio.run(find_recent_task(client, 1, only_status={"failed", "cancelled"}))
        self.assertEqual(result["id"], "t2")

    def test_returns_none_when_nothing_matches(self):
        messages = [{"task_id": "t1", "role": "user"}]
        tasks = {"t1": {"id": "t1", "status": "completed"}}
        client = FakeThreadClient(messages, tasks)
        result = asyncio.run(find_recent_task(client, 1, only_status={"failed"}))
        self.assertIsNone(result)

    def test_no_status_filter_returns_most_recent_task(self):
        messages = [
            {"task_id": "t1", "role": "user"},
            {"task_id": "t2", "role": "user"},
        ]
        tasks = {"t1": {"id": "t1", "status": "completed"}, "t2": {"id": "t2", "status": "completed"}}
        client = FakeThreadClient(messages, tasks)
        result = asyncio.run(find_recent_task(client, 1))
        self.assertEqual(result["id"], "t2")

    def test_empty_thread_returns_none(self):
        client = FakeThreadClient([], {})
        result = asyncio.run(find_recent_task(client, 1))
        self.assertIsNone(result)

    def test_duplicate_task_ids_across_messages_only_checked_once(self):
        messages = [
            {"task_id": "t1", "role": "user"},
            {"task_id": "t1", "role": "assistant"},
        ]
        calls = []

        class CountingClient(FakeThreadClient):
            async def get_task(self, task_id):
                calls.append(task_id)
                return await super().get_task(task_id)

        client = CountingClient(messages, {"t1": {"id": "t1", "status": "failed"}})
        asyncio.run(find_recent_task(client, 1, only_status={"failed"}))
        self.assertEqual(calls, ["t1"])


if __name__ == "__main__":
    unittest.main()
