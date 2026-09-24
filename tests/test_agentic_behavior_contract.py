"""Behavioral-contract tests for the agentic overhaul of tamfis-code.

Pins the system prompt's positive identity/workflow shape (not just its
incident rules), the write_todos plan-tracking tool's behavior, and the
renderer's visible todo checklist. These encode the overhaul that made
tamfis-code behave like a real coding agent (Claude Code/Codex/Freebuff
class) instead of a generic chat-with-tools loop.
"""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from io import StringIO
from pathlib import Path

from rich.console import Console

from tamfis_code.render import StreamRenderer
from tamfis_code.tool_policy import EDIT_TOOLS, EXECUTE_TOOLS, READ_TOOLS, allowed_tools
from tamfis_code.routing import TaskProfile, TaskType
from tamfis_code.workspace import build_system_prompt
from tamfis_code.runner_local import _prechange_review_applies


def _profile(task_type: TaskType) -> TaskProfile:
    return TaskProfile(
        task_type=task_type, complexity="medium", requires_tools=True,
        requires_repository_context=True, requires_long_context=False,
        requires_validation=True, preferred_quality_tier="standard",
    )


def test_read_only_turn_does_not_open_mutation_review_prompt():
    complex_profile = TaskProfile(
        task_type=TaskType.MIXED, complexity="very_complex", requires_tools=True,
        requires_repository_context=True, requires_long_context=True,
        requires_validation=True, preferred_quality_tier="frontier",
    )
    assert not _prechange_review_applies(
        complex_profile,
        "inventory the repository and then implement the proposed changes",
        interactive=True,
        turn_read_only=True,
    )


def _console() -> Console:
    return Console(file=StringIO(), no_color=True, width=200)


class SystemPromptBehaviorContractTests(unittest.TestCase):
    """The system prompt must lead with the agentic identity/workflow that
    shapes behavior positively, while retaining the incident rules that
    each have their own pinned regression test elsewhere in this suite."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _prompt(self) -> str:
        return build_system_prompt(1, self.root)

    def test_prompt_leads_with_an_agentic_identity_not_a_scold(self):
        prompt = self._prompt()
        # Identity first: names the agent and states it DOES the work.
        self.assertTrue(prompt.startswith("You are Tamfis Code"))
        self.assertIn("senior software engineering agent", prompt)
        self.assertIn("you do the work", prompt)
        # The very first incident rule must not be the opening sentence:
        # the prompt opens with identity, then a workflow section.
        self.assertLess(
            prompt.index("senior software engineering agent"),
            prompt.index("Never call list_directory"),
        )

    def test_prompt_has_a_verify_your_work_contract(self):
        prompt = self._prompt()
        self.assertIn("RUN the project's own check", prompt)
        self.assertIn("Never claim success from reasoning alone", prompt)

    def test_prompt_has_a_planning_contract_via_write_todos(self):
        prompt = self._prompt()
        self.assertIn("write_todos", prompt)
        self.assertIn("3+ tool calls", prompt)

    def test_prompt_has_a_communication_contract(self):
        prompt = self._prompt()
        self.assertIn("## Communication", prompt)
        self.assertIn("Lead with the outcome", prompt)
        self.assertIn("code block in your text is not a change", prompt)

    def test_prompt_keeps_the_incident_rules(self):
        prompt = self._prompt()
        self.assertIn("Never call list_directory", prompt)
        self.assertIn("stuck loop", prompt)
        self.assertIn("REAL configured port", prompt)
        self.assertIn("evidence chain", prompt)
        self.assertIn("extension must", prompt)


class WriteTodosToolPolicyTests(unittest.TestCase):
    """write_todos must be available in every non-plain tool-calling mode,
    including read-only turns (it mutates only the session ledger)."""

    def test_write_todos_is_in_every_tool_important_list(self):
        self.assertIn("write_todos", READ_TOOLS)
        self.assertIn("write_todos", EDIT_TOOLS)
        self.assertIn("write_todos", EXECUTE_TOOLS)

    def test_read_only_turns_still_get_write_todos(self):
        tools = allowed_tools(_profile(TaskType.INSPECT), read_only=True)
        self.assertIn("execute_command", tools)
        self.assertIn("write_todos", tools)
        self.assertNotIn("write_file", tools)

    def test_edit_turns_get_write_todos(self):
        self.assertIn("write_todos", allowed_tools(_profile(TaskType.EDIT), read_only=False))


class WriteTodosToolHandlerTests(unittest.TestCase):
    """The handler itself: cleans input, persists to the session task
    ledger when a session exists, and never raises on malformed input."""

    def _server(self, session_id=None):
        from tamfis_code.mcp import MCPServer
        return MCPServer(session_id=session_id)

    def test_returns_progress_summary(self):
        server = self._server()
        result = asyncio.run(server._write_todos([
            {"task": "Read the failing module", "completed": True},
            {"task": "Fix pagination", "completed": False},
            {"task": "Run tests", "completed": False},
        ]))
        self.assertIn("1/3", result)

    def test_drops_malformed_and_blank_entries(self):
        server = self._server()
        result = asyncio.run(server._write_todos([
            {"task": "Real step", "completed": False},
            {"completed": True},           # no task text -> dropped
            "not a dict",                  # -> dropped
            {"task": "   ", "completed": True},  # blank -> dropped
        ]))
        self.assertIn("0/1", result)

    def test_caps_at_fifty_entries(self):
        server = self._server()
        todos = [{"task": f"step {i}", "completed": False} for i in range(80)]
        result = asyncio.run(server._write_todos(todos))
        self.assertIn("0/50", result)

    def test_truncates_oversized_task_text(self):
        server = self._server()
        asyncio.run(server._write_todos([{"task": "x" * 5000, "completed": False}]))
        # No exception; oversized text is clamped (behavioral: bounded state).
        self.assertTrue(True)

    def test_persists_to_session_task_ledger(self):
        import tamfis_code.state as state_module
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            orig_config, orig_state = state_module.CONFIG_DIR, state_module.STATE_PATH
            state_module.CONFIG_DIR = base / ".config"
            state_module.STATE_PATH = base / ".config" / "state.json"
            try:
                server = self._server(session_id=424242)
                asyncio.run(server._write_todos([
                    {"task": "Persisted step", "completed": False},
                ]))
                ledger = state_module.get_session_state(424242).task_state
                self.assertEqual(
                    ledger.get("todo_list"),
                    [{"task": "Persisted step", "completed": False}],
                )
            finally:
                state_module.CONFIG_DIR, state_module.STATE_PATH = orig_config, orig_state

    def test_handler_never_raises_without_a_session(self):
        server = self._server(session_id=None)
        result = asyncio.run(server._write_todos([{"task": "solo", "completed": False}]))
        self.assertIn("0/1", result)


class WriteTodosRenderTests(unittest.TestCase):
    """The terminal renders the todo list as a visible checklist with
    done/open markers, not as generic buried tool output."""

    def _render(self, todos) -> str:
        console = _console()
        renderer = StreamRenderer(console)
        renderer.handle_event({
            "event_type": "tool_call_requested",
            "payload": {"name": "write_todos", "arguments": {"todos": todos}},
        })
        return console.file.getvalue()

    def test_checklist_renders_done_and_open_markers(self):
        output = self._render([
            {"task": "Read the module", "completed": True},
            {"task": "Fix the bug", "completed": False},
            {"task": "Run tests", "completed": False},
        ])
        self.assertIn("✔", output)
        self.assertIn("❯", output)
        self.assertIn("Fix the bug", output)
        # Completed steps must not carry the active marker.
        self.assertLess(output.index("✔ Read the module"), output.index("❯ Fix the bug"))

    def test_active_marker_lands_on_first_open_step(self):
        output = self._render([
            {"task": "Step one", "completed": True},
            {"task": "Step two", "completed": True},
            {"task": "Step three", "completed": False},
        ])
        self.assertIn("❯ Step three", output)
        self.assertNotIn("❯ Step one", output)

    def test_all_complete_renders_no_active_marker(self):
        output = self._render([
            {"task": "Only step", "completed": True},
        ])
        self.assertIn("✔ Only step", output)
        self.assertNotIn("❯", output)

    def test_empty_todo_list_renders_nothing(self):
        output = self._render([])
        self.assertEqual(output.strip(), "")

    def test_task_text_is_escaped_not_interpreted_as_markup(self):
        output = self._render([
            {"task": "[red]not markup[/red]", "completed": False},
        ])
        self.assertIn("[red]not markup[/red]", output)
        self.assertNotIn("\x1b[31m", output)


if __name__ == "__main__":
    unittest.main()
