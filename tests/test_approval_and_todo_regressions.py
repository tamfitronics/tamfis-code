"""Live report 2026-09-21: after an approval prompt the composer was dead (Enter did nothing, typed keys
were echoed, follow-ups never sent), and a run sat on "Reviewing the tool result..." for 260 minutes.
The approval was for ``write_todos`` -- a to-do list edit rated "dangerous".
"""
import asyncio
import json
import time
import unittest

from tamfis_code import permission_race
from tamfis_code.mcp import MCPServer
from tamfis_code.safety import classify_tool_call_risk


class WriteTodosRiskTests(unittest.TestCase):
    def test_write_todos_is_never_dangerous(self):
        for args in (
            {"todos": json.dumps([{"task": "x", "completed": False}])},
            {"todos": [{"task": "x", "completed": False}]},
            {},
        ):
            self.assertEqual(classify_tool_call_risk("write_todos", args, workspace_root="/home"), "read_only")

    def test_unknown_tools_still_fail_safe(self):
        self.assertEqual(classify_tool_call_risk("totally_new_tool", {}, workspace_root="/home"), "dangerous")


class WriteTodosParsingTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, todos):
        server = MCPServer.__new__(MCPServer)
        server.session_id = None
        return await MCPServer._write_todos(server, todos)

    async def test_a_json_string_is_parsed_not_iterated_character_by_character(self):
        # Used to answer "Todo list cleared." for a perfectly good list.
        result = await self._run(json.dumps([{"task": "Check readiness", "completed": False}]))
        self.assertIn("0/1", result)

    async def test_accepts_a_real_list_a_single_object_and_the_content_status_shape(self):
        self.assertIn("1/2", await self._run([{"task": "a", "completed": True}, {"task": "b"}]))
        self.assertIn("0/1", await self._run({"task": "only"}))
        self.assertIn("1/2", await self._run([{"content": "a", "status": "completed"}, {"content": "b", "status": "pending"}]))

    async def test_plain_text_lines_become_open_tasks_and_garbage_clears_nothing_silently(self):
        self.assertIn("0/2", await self._run("- first\n- second"))
        self.assertIn("cleared", await self._run(None))


class RaceCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_losing_racer_that_ignores_cancellation_cannot_block_the_decision(self):
        """The cleanup awaited every cancelled racer with no bound: one that swallowed the cancel (a
        prompt app tearing down, a classifier inside a client call) meant the approved tool never ran."""
        stubborn_started = asyncio.Event()
        release = asyncio.Event()

        async def stubborn_ui():
            stubborn_started.set()
            while not release.is_set():
                try:
                    await asyncio.wait_for(release.wait(), timeout=3600)
                except asyncio.CancelledError:
                    pass  # ignores cancellation until the test lets it go
                except asyncio.TimeoutError:
                    pass

        async def denying_classifier(_payload):
            await stubborn_started.wait()
            return "UNSAFE"

        original = permission_race.CANCEL_GRACE_SECONDS
        permission_race.CANCEL_GRACE_SECONDS = 0.3
        try:
            started = time.monotonic()
            outcome = await asyncio.wait_for(permission_race.race_permission(
                "execute_command", {"command": "rm -rf /tmp/x"}, risk="medium", policy="ask", interactive=True,
                ui_prompt=stubborn_ui, classifier=denying_classifier,
                policy_decision=lambda policy, risk, interactive: None, read_only_tools=set(),
            ), timeout=10)
            elapsed = time.monotonic() - started
        finally:
            permission_race.CANCEL_GRACE_SECONDS = original
            release.set()            # let the stubborn racer finish so the test loop can close
            await asyncio.sleep(0.05)
        self.assertLess(elapsed, 3.0, "the decision waited on a racer that ignores cancellation")
        self.assertIsNotNone(outcome)


if __name__ == "__main__":
    unittest.main()
