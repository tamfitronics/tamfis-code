"""Tests for the multi-phase formal-planning extension: is_formal_planning_
objective's gate, extract_phase_outline's parsing, merge_phase_plans'
combination logic, ExecutionPlan's phase-aware step queries, and
AgentOrchestrator._scale_budgets_for_plan_size's round-budget scaling for
an oversized plan.

Covers the pure/deterministic pieces of that pipeline; the LLM-calling
passes themselves (_attempt_phase_plans, _verify_and_critique_plan) are
exercised indirectly through this module's building blocks rather than
re-mocked here.
"""
import dataclasses
import json
import types
import unittest

from tamfis_code.orchestrator.engine import AgentOrchestrator
from tamfis_code.orchestrator.planner import (
    MAX_PLAN_PHASES,
    ExecutionPlan,
    PlanStep,
    extract_phase_outline,
    is_formal_planning_objective,
    merge_phase_plans,
    plan_phase_count,
)
from tamfis_code.routing import TaskProfile, TaskType
from tamfis_code.runtime.budgets import RuntimeBudgets


def _profile(task_type: TaskType) -> TaskProfile:
    return TaskProfile(
        task_type=task_type, complexity="medium", requires_tools=True,
        requires_repository_context=True, requires_long_context=False,
        requires_validation=True, preferred_quality_tier="standard",
    )


class IsFormalPlanningObjectiveTests(unittest.TestCase):
    def test_plan_audit_and_mixed_are_always_formal(self):
        for task_type in (TaskType.PLAN, TaskType.AUDIT, TaskType.MIXED):
            self.assertTrue(is_formal_planning_objective(_profile(task_type), "fix typo"))

    def test_a_short_edit_objective_is_not_formal(self):
        self.assertFalse(is_formal_planning_objective(_profile(TaskType.EDIT), "fix the login bug"))

    def test_an_edit_objective_naming_architecture_scale_is_formal(self):
        self.assertTrue(is_formal_planning_objective(
            _profile(TaskType.EDIT), "redesign the authentication architecture across the repository",
        ))

    def test_a_long_edit_objective_is_formal_purely_on_length(self):
        long_objective = "fix the bug where " + "the retry logic misbehaves " * 20
        self.assertGreaterEqual(len(long_objective), 320)
        self.assertTrue(is_formal_planning_objective(_profile(TaskType.EDIT), long_objective))

    def test_conversation_and_research_task_types_are_never_formal(self):
        self.assertFalse(is_formal_planning_objective(_profile(TaskType.CONVERSATION), "x" * 400))
        self.assertFalse(is_formal_planning_objective(_profile(TaskType.RESEARCH), "x" * 400))


class ExtractPhaseOutlineTests(unittest.TestCase):
    def test_extracts_a_well_formed_outline(self):
        raw = json.dumps({
            "multi_phase": True,
            "phase_outline": [
                {"name": "Backend", "description": "Add the API endpoint"},
                {"name": "Frontend", "description": "Wire up the UI"},
            ],
        })
        outline = extract_phase_outline(raw)
        self.assertEqual(outline, [
            {"name": "Backend", "description": "Add the API endpoint"},
            {"name": "Frontend", "description": "Wire up the UI"},
        ])

    def test_multi_phase_false_yields_none_even_with_an_outline_present(self):
        raw = json.dumps({"multi_phase": False, "phase_outline": [{"name": "X", "description": "Y"}]})
        self.assertIsNone(extract_phase_outline(raw))

    def test_missing_phase_outline_yields_none(self):
        self.assertIsNone(extract_phase_outline(json.dumps({"multi_phase": True})))

    def test_non_dict_items_are_skipped_and_nameless_items_are_dropped(self):
        raw = json.dumps({
            "multi_phase": True,
            "phase_outline": ["not a dict", {"description": "no name"}, {"name": "Real phase", "description": "ok"}],
        })
        self.assertEqual(extract_phase_outline(raw), [{"name": "Real phase", "description": "ok"}])

    def test_an_outline_left_with_nothing_usable_yields_none(self):
        raw = json.dumps({"multi_phase": True, "phase_outline": [{"description": "no name"}]})
        self.assertIsNone(extract_phase_outline(raw))

    def test_outline_is_bounded_to_max_plan_phases(self):
        raw = json.dumps({
            "multi_phase": True,
            "phase_outline": [{"name": f"Phase {i}", "description": ""} for i in range(MAX_PLAN_PHASES + 5)],
        })
        outline = extract_phase_outline(raw)
        self.assertEqual(len(outline), MAX_PLAN_PHASES)

    def test_unparseable_content_yields_none(self):
        self.assertIsNone(extract_phase_outline("not json at all"))


class MergePhasePlansTests(unittest.TestCase):
    def _plan(self, *step_names: str, assumptions=(), risks=()) -> ExecutionPlan:
        steps = [PlanStep(i, name) for i, name in enumerate(step_names, start=1)]
        return ExecutionPlan(
            objective="obj", assumptions=list(assumptions), components=[], steps=steps,
            validation_criteria=[], risks=list(risks),
        )

    def test_concatenates_steps_and_tags_each_with_its_1_based_phase(self):
        merged = merge_phase_plans("obj", [
            ("Backend", self._plan("add endpoint")),
            ("Frontend", self._plan("wire up UI", "add loading state")),
        ])
        self.assertEqual(plan_phase_count(merged), 2)
        self.assertEqual(merged.phase_names, ["Backend", "Frontend"])
        self.assertEqual([s.phase for s in merged.steps], [1, 2, 2])
        self.assertEqual([s.name for s in merged.steps], ["add endpoint", "wire up UI", "add loading state"])

    def test_a_phase_with_no_steps_is_dropped_and_does_not_break_numbering(self):
        merged = merge_phase_plans("obj", [
            ("Backend", self._plan("add endpoint")),
            ("Empty phase", self._plan()),
            ("Frontend", self._plan("wire up UI")),
        ])
        self.assertEqual(merged.phase_names, ["Backend", "Frontend"])
        self.assertEqual([s.phase for s in merged.steps], [1, 2])

    def test_assumptions_and_risks_are_deduplicated_across_phases(self):
        merged = merge_phase_plans("obj", [
            ("A", self._plan("s1", assumptions=["shared assumption"], risks=["shared risk"])),
            ("B", self._plan("s2", assumptions=["shared assumption"], risks=["shared risk"])),
        ])
        self.assertEqual(merged.assumptions.count("shared assumption"), 1)
        self.assertEqual(merged.risks.count("shared risk"), 1)

    def test_raises_when_every_phase_produced_no_steps(self):
        with self.assertRaises(ValueError):
            merge_phase_plans("obj", [("A", self._plan()), ("B", self._plan())])

    def test_step_indexes_are_reindexed_contiguously_across_phases(self):
        merged = merge_phase_plans("obj", [
            ("A", self._plan("s1", "s2")),
            ("B", self._plan("s3")),
        ])
        self.assertEqual([s.index for s in merged.steps], [1, 2, 3])


class ExecutionPlanPhaseQueryTests(unittest.TestCase):
    def _phased_plan(self) -> ExecutionPlan:
        steps = [
            PlanStep(1, "phase1-a", status="completed", phase=1),
            PlanStep(2, "phase1-b", status="pending", phase=1),
            PlanStep(3, "phase2-a", status="pending", phase=2),
        ]
        return ExecutionPlan(
            objective="obj", assumptions=[], components=[], steps=steps,
            validation_criteria=[], risks=[], phase_names=["One", "Two"],
        )

    def test_next_pending_with_no_phase_walks_in_plan_order(self):
        plan = self._phased_plan()
        self.assertEqual(plan.next_pending().name, "phase1-b")

    def test_next_pending_scoped_to_a_phase_ignores_other_phases(self):
        plan = self._phased_plan()
        self.assertEqual(plan.next_pending(phase=2).name, "phase2-a")
        self.assertEqual(plan.next_pending(phase=1).name, "phase1-b")

    def test_next_pending_scoped_to_a_phase_with_nothing_left_is_none(self):
        plan = self._phased_plan()
        plan.steps[2].status = "completed"  # complete the only phase-2 step
        self.assertIsNone(plan.next_pending(phase=2))

    def test_remaining_step_count_scoped_to_a_phase(self):
        plan = self._phased_plan()
        self.assertEqual(plan.remaining_step_count(), 2)
        self.assertEqual(plan.remaining_step_count(phase=1), 1)
        self.assertEqual(plan.remaining_step_count(phase=2), 1)

    def test_plan_phase_count_is_1_for_an_ordinary_flat_plan(self):
        flat = ExecutionPlan(
            objective="obj", assumptions=[], components=[],
            steps=[PlanStep(1, "only step")], validation_criteria=[], risks=[],
        )
        self.assertEqual(plan_phase_count(flat), 1)


class ScaleBudgetsForPlanSizeTests(unittest.TestCase):
    def _fake_orchestrator(self, budgets: RuntimeBudgets, *, step_count: int):
        events: list[dict] = []
        fake_self = types.SimpleNamespace(
            run=types.SimpleNamespace(runtime=types.SimpleNamespace(budgets=budgets)),
            emit=events.append,
        )
        plan = ExecutionPlan(
            objective="obj", assumptions=[], components=[],
            steps=[PlanStep(i, f"step {i}") for i in range(step_count)],
            validation_criteria=[], risks=[],
        )
        return fake_self, plan, events

    def test_a_small_plan_leaves_the_default_budget_untouched(self):
        fake_self, plan, events = self._fake_orchestrator(RuntimeBudgets(), step_count=10)
        AgentOrchestrator._scale_budgets_for_plan_size(fake_self, plan)
        self.assertEqual(fake_self.run.runtime.budgets.max_round_extensions, 2)
        self.assertEqual(events, [])

    def test_a_plan_just_over_80_steps_widens_the_budget_and_reports_the_true_before_value(self):
        fake_self, plan, events = self._fake_orchestrator(RuntimeBudgets(), step_count=81)
        AgentOrchestrator._scale_budgets_for_plan_size(fake_self, plan)
        self.assertEqual(fake_self.run.runtime.budgets.max_round_extensions, 3)
        self.assertEqual(len(events), 1)
        message = events[0]["payload"]["content"]
        # Regression test: this diagnostic used to read the budget field
        # AFTER already overwriting it, so "from X" always printed the new
        # value instead of the real previous one (e.g. "from 3 to 3").
        self.assertIn("from 2", message)
        self.assertIn("to 3", message)

    def test_widening_is_capped_at_10_extensions_for_a_very_large_plan(self):
        fake_self, plan, events = self._fake_orchestrator(RuntimeBudgets(), step_count=1000)
        AgentOrchestrator._scale_budgets_for_plan_size(fake_self, plan)
        self.assertEqual(fake_self.run.runtime.budgets.max_round_extensions, 10)

    def test_never_narrows_an_already_wider_budget(self):
        budgets = dataclasses.replace(RuntimeBudgets(), max_round_extensions=8)
        fake_self, plan, events = self._fake_orchestrator(budgets, step_count=90)
        AgentOrchestrator._scale_budgets_for_plan_size(fake_self, plan)
        self.assertEqual(fake_self.run.runtime.budgets.max_round_extensions, 8)
        self.assertEqual(events, [])

    def test_the_budgets_object_itself_is_frozen_so_replacement_must_be_used(self):
        """RuntimeBudgets is deliberately immutable (runtime/budgets.py) --
        this pins that invariant so a future refactor that tries direct
        field assignment again (the bug _scale_budgets_for_plan_size itself
        used to have) fails loudly here instead of only at run time on a
        real oversized plan."""
        with self.assertRaises(dataclasses.FrozenInstanceError):
            RuntimeBudgets().max_round_extensions = 9


if __name__ == "__main__":
    unittest.main()
