"""Reasoning-based plan generation (orchestrator/planner.py's
build_reasoning_plan_prompt/parse_reasoning_plan) and its wiring into
runner_local.py's run_local_agent_turn -- replacing the fixed template plan
with one grounded in the real objective and real workspace facts, and
revising it once real tool evidence exists (adaptive replanning).
"""
from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console

from tamfis_code import state as state_module
from tamfis_code.orchestrator.planner import (
    ExecutionPlan,
    build_reasoning_plan_prompt,
    parse_reasoning_plan,
)
from tamfis_code.providers import ProviderType
from tamfis_code.routing import TaskProfile, TaskType
from tamfis_code.runner_local import run_local_agent_turn


class ParseReasoningPlanTests(unittest.TestCase):
    def test_parses_a_well_formed_plan(self):
        raw = json.dumps({
            "steps": ["Read calc.py to find the off-by-one error", "Fix the bounds check", "Run pytest"],
            "assumptions": ["calc.py is the only file involved"],
            "risks": ["Hidden second bug elsewhere"],
        })
        plan = parse_reasoning_plan(raw, objective="fix calc.py")
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.steps), 3)
        self.assertEqual(plan.steps[0].name, "Read calc.py to find the off-by-one error")
        self.assertEqual(plan.steps[0].index, 1)
        self.assertEqual(plan.assumptions, ["calc.py is the only file involved"])
        self.assertEqual(plan.risks, ["Hidden second bug elsewhere"])

    def test_strips_a_markdown_code_fence(self):
        raw = "```json\n" + json.dumps({"steps": ["Do the one thing"]}) + "\n```"
        plan = parse_reasoning_plan(raw, objective="x")
        self.assertIsNotNone(plan)
        self.assertEqual(plan.steps[0].name, "Do the one thing")

    def test_missing_steps_key_returns_none(self):
        self.assertIsNone(parse_reasoning_plan(json.dumps({"assumptions": ["x"]}), objective="x"))

    def test_empty_steps_list_returns_none(self):
        self.assertIsNone(parse_reasoning_plan(json.dumps({"steps": []}), objective="x"))

    def test_malformed_json_returns_none_not_raises(self):
        self.assertIsNone(parse_reasoning_plan("not json at all {{{", objective="x"))

    def test_empty_string_returns_none(self):
        self.assertIsNone(parse_reasoning_plan("", objective="x"))

    def test_non_dict_json_returns_none(self):
        self.assertIsNone(parse_reasoning_plan(json.dumps(["steps", "not", "a", "dict"]), objective="x"))

    def test_missing_optional_fields_get_sensible_defaults(self):
        plan = parse_reasoning_plan(json.dumps({"steps": ["Only step"]}), objective="x")
        self.assertIsNotNone(plan)
        self.assertTrue(plan.assumptions)
        self.assertTrue(plan.risks)

    def test_steps_beyond_the_cap_are_truncated_not_rejected(self):
        raw = json.dumps({"steps": [f"step {i}" for i in range(20)]})
        plan = parse_reasoning_plan(raw, objective="x")
        self.assertIsNotNone(plan)
        self.assertLessEqual(len(plan.steps), 8)


class BuildReasoningPlanPromptTests(unittest.TestCase):
    def _profile(self):
        return TaskProfile(TaskType.DEBUG, "high", True, True, True, True, "frontier")

    def test_prompt_includes_the_real_objective_and_workspace_facts(self):
        messages = build_reasoning_plan_prompt(
            "fix the crash in calc.py", self._profile(),
            {"detected_languages": ["Python"], "frameworks": ["Django"], "test_commands": ["pytest -q"]},
        )
        user_content = messages[-1]["content"]
        self.assertIn("fix the crash in calc.py", user_content)
        self.assertIn("Python", user_content)
        self.assertIn("Django", user_content)
        self.assertIn("pytest -q", user_content)

    def test_evidence_summary_is_included_for_a_revision_and_asks_for_grounding(self):
        messages = build_reasoning_plan_prompt(
            "fix the crash", self._profile(), {}, evidence_summary="Files inspected so far: calc.py",
        )
        user_content = messages[-1]["content"]
        self.assertIn("Files inspected so far: calc.py", user_content)
        self.assertIn("REVISION", user_content)

    def test_no_evidence_summary_omits_the_revision_language(self):
        messages = build_reasoning_plan_prompt("fix the crash", self._profile(), {})
        self.assertNotIn("REVISION", messages[-1]["content"])


def _delta(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _tool_call_delta(index, call_id=None, name=None, arguments=None):
    function = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=call_id, function=function)


def _chunk(delta, finish_reason=None):
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)])


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for chunk in self._chunks:
            yield chunk


class _FakeClient:
    def __init__(self, rounds):
        self._rounds = list(rounds)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeStream(self._rounds.pop(0))


class _FakeManager:
    def __init__(self, client):
        self._client = client
        self.PROVIDERS = {ProviderType.NVIDIA: SimpleNamespace(default_model="fake-model", context_window=32768)}

    def get_client(self, provider):
        return self._client


class _RecordingRenderer:
    def __init__(self):
        self.events = []

    def handle_event(self, event):
        self.events.append(event)


class _StatePatchMixin:
    def setUp(self):
        self._originals = (state_module.CONFIG_DIR, state_module.STATE_PATH)
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        state_module.CONFIG_DIR = base / ".config"
        state_module.STATE_PATH = base / ".config" / "state.json"

    def tearDown(self):
        state_module.CONFIG_DIR, state_module.STATE_PATH = self._originals
        self.tmp.cleanup()


class ReasoningPlanIntegrationTests(_StatePatchMixin, unittest.TestCase):
    def _console(self):
        from io import StringIO
        return Console(file=StringIO(), no_color=True, width=200)

    def test_plan_worthy_task_gets_a_real_task_specific_plan_not_the_generic_template(self):
        """Confirmed the old behaviour: every plan-worthy task got the exact
        same "Inspect the relevant repository context and manifests" /
        "Select a capable provider/model..." boilerplate regardless of what
        was actually being asked. The reasoning plan must replace it with
        something grounded in the objective."""
        with tempfile.TemporaryDirectory() as ws:
            plan_response = json.dumps({
                "steps": [
                    "Open calc.py and locate the addition in the total() function",
                    "Change the off-by-one increment to the correct value",
                    "Re-run the failing test to confirm the fix",
                ],
                "risks": ["A second unrelated bug in the same function"],
            })
            rounds = [
                [_chunk(_delta(content=plan_response))],
                [_chunk(_delta(content="Fixed."))],
            ]
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None,
                [{"role": "user", "content": "fix the bug in calc.py"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
            ))

            self.assertEqual(outcome.status, "completed")
            plan_events = [e for e in renderer.events if e["event_type"] == "plan_created"]
            self.assertEqual(len(plan_events), 1)
            steps = [item["step"] for item in plan_events[0]["payload"]["items"]]
            self.assertIn("Open calc.py and locate the addition in the total() function", steps)
            self.assertNotIn("Inspect the relevant repository context and manifests", steps)

            # The plan must also actually reach the model as prompt context.
            plan_messages = [
                m["content"] for m in client.calls[-1]["messages"]
                if m.get("role") == "system" and "TASK PLAN" in str(m.get("content"))
            ]
            self.assertTrue(plan_messages)
            self.assertIn("Open calc.py and locate the addition", plan_messages[0])

    def test_malformed_planning_response_falls_back_silently_turn_still_completes(self):
        with tempfile.TemporaryDirectory() as ws:
            rounds = [
                [_chunk(_delta(content="I am not JSON, I am just talking."))],
                [_chunk(_delta(content="Fixed anyway."))],
            ]
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None,
                [{"role": "user", "content": "fix the bug in calc.py"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
            ))

            self.assertEqual(outcome.status, "completed")
            self.assertIn("Fixed anyway.", outcome.summary)
            diagnostics = [e["payload"].get("content", "") for e in renderer.events if e["event_type"] == "diagnostics"]
            self.assertTrue(any("using the existing plan" in d for d in diagnostics))

    def test_question_type_task_never_triggers_a_planning_call(self):
        """should_plan(QUESTION) is False -- a plain question must not pay
        for (or wait on) an extra planning completion at all."""
        with tempfile.TemporaryDirectory() as ws:
            rounds = [[_chunk(_delta(content="This project is a coding agent CLI."))]]
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None,
                [{"role": "user", "content": "what does this project do?"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
            ))

            self.assertEqual(outcome.status, "completed")
            self.assertEqual(len(client.calls), 1)
            self.assertFalse([e for e in renderer.events if e["event_type"] == "plan_created"])

    def test_plan_is_revised_once_real_tool_evidence_exists(self):
        """The initial plan is necessarily a guess (made before any tool has
        run) -- once a real tool result exists, it must be revised, grounded
        in what was actually found, not left as the original guess for the
        rest of the turn."""
        with tempfile.TemporaryDirectory() as ws:
            (Path(ws) / "calc.py").write_text("def total(n):\n    return n + 2\n")
            read_args = json.dumps({"path": str(Path(ws) / "calc.py")})
            initial_plan = json.dumps({"steps": ["Read calc.py", "Fix it", "Verify"]})
            revised_plan = json.dumps({
                "steps": ["Change n + 2 to n + 1 in total()", "Re-run the failing test"],
            })
            rounds = [
                [_chunk(_delta(content=initial_plan))],
                [_chunk(_delta(tool_calls=[_tool_call_delta(0, call_id="call_1", name="read_file", arguments=read_args)]))],
                [_chunk(_delta(content=revised_plan))],
                [_chunk(_delta(content="Done."))],
            ]
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            outcome = asyncio.run(run_local_agent_turn(
                manager, ProviderType.NVIDIA, None,
                [{"role": "user", "content": "fix the bug in calc.py"}],
                self._console(), renderer,
                workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
            ))

            self.assertEqual(outcome.status, "completed")
            plan_events = [e for e in renderer.events if e["event_type"] == "plan_created"]
            self.assertEqual(len(plan_events), 2, "expected one initial plan_created and one revision")
            revised_steps = [item["step"] for item in plan_events[1]["payload"]["items"]]
            self.assertIn("Change n + 2 to n + 1 in total()", revised_steps)
            self.assertEqual(plan_events[1]["payload"]["title"], "Plan (revised)")

            # Only ever revised once, even though this turn had more than
            # one tool-bearing round available in principle.
            revise_calls = [
                c for c in client.calls
                if any("REVISION" in str(m.get("content")) for m in c["messages"] if m.get("role") == "user")
            ]
            self.assertEqual(len(revise_calls), 1)


if __name__ == "__main__":
    unittest.main()
