import asyncio
import time
import unittest
from unittest.mock import MagicMock, patch

from tamfis_code.agents import AgentManager


class ExecuteTasksConcurrencyTests(unittest.TestCase):
    def test_concurrent_tasks_actually_overlap(self):
        fake_workspace = MagicMock(session_id=1, workspace_root="/tmp")

        async def fake_execute(self, task):
            await asyncio.sleep(0.2)
            return {"status": "completed", "summary": f"done:{task.description}"}

        with patch("tamfis_code.workspace.resolve_local_workspace", return_value=fake_workspace), \
                patch("tamfis_code.agents.DelegatedCodingAgent.execute", new=fake_execute):
            manager = AgentManager()
            started = time.monotonic()
            results = asyncio.run(manager.execute_tasks(
                ["task a", "task b", "task c"],
                manager=object(), provider=object(), model=None, console=object(), workspace_root="/tmp",
                max_concurrency=3,
            ))
            elapsed = time.monotonic() - started

        self.assertEqual(len(results), 3)
        self.assertTrue(all(r["status"] == "completed" for r in results))
        # Sequential execution would take ~0.6s (3 x 0.2s); real overlap
        # should keep this close to a single 0.2s sleep.
        self.assertLess(elapsed, 0.45)

    def test_max_concurrency_one_runs_sequentially(self):
        fake_workspace = MagicMock(session_id=1, workspace_root="/tmp")

        async def fake_execute(self, task):
            await asyncio.sleep(0.15)
            return {"status": "completed", "summary": "done"}

        with patch("tamfis_code.workspace.resolve_local_workspace", return_value=fake_workspace), \
                patch("tamfis_code.agents.DelegatedCodingAgent.execute", new=fake_execute):
            manager = AgentManager()
            started = time.monotonic()
            results = asyncio.run(manager.execute_tasks(
                ["task a", "task b"],
                manager=object(), provider=object(), model=None, console=object(), workspace_root="/tmp",
                max_concurrency=1,
            ))
            elapsed = time.monotonic() - started

        self.assertEqual(len(results), 2)
        self.assertGreaterEqual(elapsed, 0.28)

    def test_failed_delegated_task_is_reported_not_raised(self):
        fake_workspace = MagicMock(session_id=1, workspace_root="/tmp")

        async def fake_execute(self, task):
            raise RuntimeError("boom")

        with patch("tamfis_code.workspace.resolve_local_workspace", return_value=fake_workspace), \
                patch("tamfis_code.agents.DelegatedCodingAgent.execute", new=fake_execute):
            manager = AgentManager()
            results = asyncio.run(manager.execute_tasks(
                ["broken task"], manager=object(), provider=object(), model=None, console=object(), workspace_root="/tmp",
            ))

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "failed")
        self.assertIn("boom", results[0]["result"]["error"])

    def test_default_concurrency_is_one(self):
        # execute_tasks defaults to sequential -- concurrent tool execution
        # against the same workspace (two sub-tasks editing overlapping
        # files, concurrent state.json writers) hasn't been stress-tested,
        # so it's opt-in via max_concurrency, not the default.
        import inspect
        signature = inspect.signature(AgentManager.execute_tasks)
        self.assertEqual(signature.parameters["max_concurrency"].default, 1)


if __name__ == "__main__":
    unittest.main()
