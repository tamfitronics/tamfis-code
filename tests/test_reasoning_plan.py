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
from unittest.mock import patch

from rich.console import Console

from tamfis_code import state as state_module
from tamfis_code.orchestrator.planner import (
    ExecutionPlan,
    build_reasoning_plan_prompt,
    create_plan,
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

    def test_purpose_is_not_appended_to_the_rendered_step_name(self):
        # Regression: every step used to render as "<action> — <purpose>.",
        # doubling (or tripling) the length of every single plan item. Claude
        # Code/Codex-style plans are short, scannable action lines; `purpose`
        # is accepted from the model but must not widen what's displayed.
        raw = json.dumps({
            "steps": [{
                "action": "Read calc.py",
                "purpose": "to find the off-by-one error causing the reported crash",
            }],
        })
        plan = parse_reasoning_plan(raw, objective="x")
        self.assertIsNotNone(plan)
        self.assertEqual(plan.steps[0].name, "Read calc.py")
        self.assertNotIn("off-by-one", plan.steps[0].name)


class ParseReasoningPlanStrictEvidenceTests(unittest.TestCase):
    """parse_reasoning_plan's evidence-validated path (strict_evidence=True),
    exercised via real filesystem paths under scope_roots -- the path
    validation in orchestrator/planner.py's PlannerEvidence.path_was_discovered
    had no direct test coverage at all before this. Regression coverage for
    the "always the same generic plan" bug: a plan step naming a file a
    create/add objective is about to bring into existence was silently
    dropped because reconnaissance is read-only and runs before anything is
    created, so it can never have seen a not-yet-existing path. When every
    step in a plan named only such paths, the whole plan was rejected and the
    turn fell back to the fixed template -- confirmed live against the
    installed CLI with a real "create a new file" objective.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        (self.root / "main.py").write_text("print('hi')\n")
        (self.root / "services").mkdir()
        (self.root / "services" / "existing_service.py").write_text("x = 1\n")

    def test_step_targeting_an_existing_file_is_accepted(self):
        raw = json.dumps({"steps": [{
            "action": "Read main.py to find the bug",
            "targets": [str(self.root / "main.py")],
        }]})
        plan = parse_reasoning_plan(raw, objective="x", scope_roots=[self.root])
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.steps), 1)

    def test_step_creating_a_new_file_directly_in_the_root_is_accepted(self):
        # Regression: this is the exact shape confirmed live -- "Create
        # utils.py" naming a file that does not exist yet, in a workspace
        # with no manifests/graph evidence at all.
        raw = json.dumps({"steps": [{
            "action": "Create utils.py with a square(n) helper",
            "targets": [str(self.root / "utils.py")],
        }]})
        plan = parse_reasoning_plan(raw, objective="x", scope_roots=[self.root])
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.steps), 1)
        self.assertFalse((self.root / "utils.py").exists())  # planning never creates it

    def test_step_naming_a_path_outside_any_authorised_root_is_still_rejected(self):
        outside = Path(tempfile.gettempdir()) / "definitely-not-authorised-elsewhere.py"
        raw = json.dumps({"steps": [{
            "action": "Create a file outside the workspace",
            "targets": [str(outside)],
        }]})
        plan = parse_reasoning_plan(raw, objective="x", scope_roots=[self.root])
        self.assertIsNone(plan)


class GroundedFallbackPlanTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        (self.root / "main.py").write_text("print('hi')\n")
        (self.root / "services").mkdir()
        (self.root / "services" / "existing_service.py").write_text("x = 1\n")

    def test_fallback_names_objective_matching_files_instead_of_generic_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "services" / "mission_pipeline.py"
            target.parent.mkdir()
            target.write_text("pass\n")
            manifest = root / "pyproject.toml"
            manifest.write_text("[project]\nname='demo'\n")
            profile = TaskProfile(
                task_type=TaskType.DEBUG, complexity="high", requires_tools=True,
                requires_validation=True, requires_repository_context=True,
                requires_long_context=True, preferred_quality_tier="frontier",
            )
            reconnaissance = (
                f"ROOT: {root}\n"
                "  manifests: pyproject.toml\n"
                "  objective_matching_paths: services/mission_pipeline.py\n"
            )
            plan = create_plan("fix the mission pipeline", profile, reconnaissance_summary=reconnaissance)

        self.assertIsNotNone(plan)
        rendered = "\n".join(step.name for step in plan.steps)
        self.assertIn(str(target), rendered)
        self.assertNotIn("Trace objective-relevant code paths from reconnaissance", rendered)

    def test_deeply_fabricated_nonexistent_path_is_rejected_with_no_graph_evidence(self):
        # Parent directory doesn't exist either -- not a plausible creation
        # target, distinct from "new file in an existing directory".
        fabricated = self.root / "made" / "up" / "nested" / "dirs" / "file.py"
        raw = json.dumps({"steps": [{
            "action": "Create a deeply nested new file",
            "targets": [str(fabricated)],
        }]})
        plan = parse_reasoning_plan(raw, objective="x", scope_roots=[self.root])
        self.assertIsNone(plan)

    def test_new_file_sibling_to_a_connected_existing_file_is_accepted(self):
        raw = json.dumps({"steps": [{
            "action": "Create a new service module next to the existing one",
            "targets": [str(self.root / "services" / "new_service.py")],
        }]})
        plan = parse_reasoning_plan(
            raw, objective="x", scope_roots=[self.root],
            workspace_summary={"objective_matching_paths": [str(self.root / "services" / "existing_service.py")]},
        )
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.steps), 1)

    def test_unrelated_existing_file_still_rejected_when_graph_evidence_present(self):
        # Guards against the fix over-widening: an existing file that graph
        # evidence never connected to the objective is still not evidence-backed.
        (self.root / "unrelated.py").write_text("y = 2\n")
        raw = json.dumps({"steps": [{
            "action": "Edit unrelated.py",
            "targets": [str(self.root / "unrelated.py")],
        }]})
        plan = parse_reasoning_plan(
            raw, objective="x", scope_roots=[self.root],
            workspace_summary={"objective_matching_paths": [str(self.root / "services" / "existing_service.py")]},
        )
        self.assertIsNone(plan)

    def test_plan_survives_when_only_some_steps_are_creation_targets(self):
        raw = json.dumps({"steps": [
            {"action": "Read main.py to understand the current entrypoint", "targets": [str(self.root / "main.py")]},
            {"action": "Create utils.py with a square(n) helper", "targets": [str(self.root / "utils.py")]},
        ]})
        plan = parse_reasoning_plan(raw, objective="x", scope_roots=[self.root])
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.steps), 2)


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
        self.background_requested = asyncio.Event()

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

    def _no_real_validation_commands(self):
        # detect_validation_commands would otherwise pick a real language
        # check (e.g. "python -m compileall -q .") for the calc.py fixture
        # -- that depends on the host actually having a `python` binary on
        # PATH (not guaranteed; some hosts only have `python3`). These tests
        # care about the reasoning-plan/mutation-evidence wiring, not real
        # Python tooling, so replace it with an always-succeeding no-op.
        return patch(
            "tamfis_code.runner_local.detect_validation_commands",
            return_value=[("no-op check", "true")],
        )

    def test_plan_worthy_task_gets_a_real_task_specific_plan_not_the_generic_template(self):
        """Confirmed the old behaviour: every plan-worthy task got the exact
        same "Inspect the relevant repository context and manifests" /
        "Select a capable provider/model..." boilerplate regardless of what
        was actually being asked. The reasoning plan must replace it with
        something grounded in the objective."""
        with tempfile.TemporaryDirectory() as ws:
            calc_path = Path(ws) / "calc.py"
            calc_path.write_text("def total(n):\n    return n + 2\n")
            plan_response = json.dumps({
                "steps": [
                    "Open calc.py and locate the addition in the total() function",
                    "Change the off-by-one increment to the correct value",
                    "Re-run the failing test to confirm the fix",
                ],
                "risks": ["A second unrelated bug in the same function"],
            })
            edit_args = json.dumps({
                "path": str(calc_path),
                "old_string": "return n + 2", "new_string": "return n + 1",
            })
            verify_args = json.dumps({"command": "true"})
            rounds = [
                [_chunk(_delta(content=plan_response))],
                [_chunk(_delta(tool_calls=[_tool_call_delta(0, call_id="call_1", name="edit_file", arguments=edit_args)]))],
                # The first successful tool call triggers one automatic
                # plan revision round (see runner_local.py's "Adaptive
                # replanning") before the turn continues.
                [_chunk(_delta(content=plan_response))],
                [_chunk(_delta(tool_calls=[_tool_call_delta(0, call_id="call_2", name="execute_command", arguments=verify_args)]))],
                [_chunk(_delta(content="Fixed."))],
            ]
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            with self._no_real_validation_commands():
                outcome = asyncio.run(run_local_agent_turn(
                    manager, ProviderType.NVIDIA, None,
                    [{"role": "user", "content": "fix the bug in calc.py"}],
                    self._console(), renderer,
                    workspace_root=ws, session_id=1, approval_policy="auto", interactive=False,
                ))

            self.assertEqual(outcome.status, "completed")
            plan_events = [e for e in renderer.events if e["event_type"] == "plan_created"]
            self.assertEqual(len(plan_events), 2, "expected the initial plan plus one evidence-triggered revision")
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
            calc_path = Path(ws) / "calc.py"
            calc_path.write_text("def total(n):\n    return n + 2\n")
            edit_args = json.dumps({
                "path": str(calc_path),
                "old_string": "return n + 2", "new_string": "return n + 1",
            })
            verify_args = json.dumps({"command": "true"})
            rounds = [
                [_chunk(_delta(content="I am not JSON, I am just talking."))],
                [_chunk(_delta(tool_calls=[_tool_call_delta(0, call_id="call_1", name="edit_file", arguments=edit_args)]))],
                # The first successful tool call triggers one automatic
                # plan revision round; keep it malformed too so this stays
                # a test of "the fallback holds even when nothing ever
                # parses as a real plan."
                [_chunk(_delta(content="Still not JSON."))],
                [_chunk(_delta(tool_calls=[_tool_call_delta(0, call_id="call_2", name="execute_command", arguments=verify_args)]))],
                [_chunk(_delta(content="Fixed anyway."))],
            ]
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            with self._no_real_validation_commands():
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
            calc_path = Path(ws) / "calc.py"
            calc_path.write_text("def total(n):\n    return n + 2\n")
            read_args = json.dumps({"path": str(calc_path)})
            edit_args = json.dumps({
                "path": str(calc_path),
                "old_string": "return n + 2", "new_string": "return n + 1",
            })
            verify_args = json.dumps({"command": "true"})
            initial_plan = json.dumps({"steps": ["Read calc.py", "Fix it", "Verify"]})
            revised_plan = json.dumps({
                "steps": ["Change n + 2 to n + 1 in total()", "Re-run the failing test"],
            })
            rounds = [
                [_chunk(_delta(content=initial_plan))],
                [_chunk(_delta(tool_calls=[_tool_call_delta(0, call_id="call_1", name="read_file", arguments=read_args)]))],
                [_chunk(_delta(content=revised_plan))],
                [_chunk(_delta(tool_calls=[_tool_call_delta(0, call_id="call_2", name="edit_file", arguments=edit_args)]))],
                [_chunk(_delta(tool_calls=[_tool_call_delta(0, call_id="call_3", name="execute_command", arguments=verify_args)]))],
                [_chunk(_delta(content="Done."))],
            ]
            client = _FakeClient(rounds)
            manager = _FakeManager(client)
            renderer = _RecordingRenderer()

            with self._no_real_validation_commands():
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
