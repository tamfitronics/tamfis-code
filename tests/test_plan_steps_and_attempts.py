import tempfile
import unittest
from pathlib import Path

from tamfis_code import state as state_module
from tamfis_code.state import (
    finish_action,
    get_session_state,
    save_plan,
    start_action,
    update_plan_steps,
)


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


class PlanStepsTests(_StatePatchMixin, unittest.TestCase):
    def test_save_plan_persists_steps_with_index(self):
        items = [{"step": "read the file", "status": "pending"}, {"step": "edit it", "status": "pending"}]
        plan = save_plan(1, objective="do the thing", content="# plan", steps=items)
        self.assertEqual(plan.steps[0]["index"], 0)
        self.assertEqual(plan.steps[1]["index"], 1)
        self.assertEqual(plan.steps[0]["step"], "read the file")

        reloaded = get_session_state(1)
        self.assertEqual(reloaded.saved_plans[-1]["steps"], plan.steps)

    def test_save_plan_with_no_steps_defaults_to_empty_list(self):
        plan = save_plan(1, objective="do the thing", content="# plan")
        self.assertEqual(plan.steps, [])

    def test_update_plan_steps_overwrites_existing_plan(self):
        plan = save_plan(1, objective="do the thing", content="# plan", steps=[{"step": "a", "status": "pending"}])
        update_plan_steps(1, plan.id, [{"step": "a", "status": "completed"}, {"step": "b", "status": "pending"}])

        reloaded = get_session_state(1)
        steps = reloaded.saved_plans[-1]["steps"]
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0]["status"], "completed")
        self.assertEqual(steps[1]["index"], 1)

    def test_update_plan_steps_unknown_plan_id_is_a_noop(self):
        result = update_plan_steps(1, "plan_does_not_exist", [{"step": "a"}])
        self.assertIsNone(result)


class AttemptTrackingTests(_StatePatchMixin, unittest.TestCase):
    def test_single_failure_does_not_escalate(self):
        action = start_action(1, action_type="ai_task", purpose="fix the bug")
        finish_action(1, action.id, status="failed", summary="boom")

        state = get_session_state(1)
        self.assertEqual(state.completed_actions[-1]["attempts"], 1)
        self.assertEqual(state.completed_actions[-1]["last_error"], "boom")
        self.assertEqual(state.unresolved_issues, [])

    def test_repeated_failure_escalates_to_unresolved_issues(self):
        for _ in range(2):
            action = start_action(1, action_type="ai_task", purpose="fix the bug")
            finish_action(1, action.id, status="failed", summary="boom")

        state = get_session_state(1)
        self.assertEqual(state.completed_actions[-1]["attempts"], 2)
        matching = [i for i in state.unresolved_issues if i.get("type") == "repeated_action_failure"]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["purpose"], "fix the bug")
        self.assertEqual(matching[0]["attempts"], 2)

    def test_escalation_is_not_duplicated_on_further_failures(self):
        for _ in range(3):
            action = start_action(1, action_type="ai_task", purpose="fix the bug")
            finish_action(1, action.id, status="failed", summary="boom")

        state = get_session_state(1)
        matching = [i for i in state.unresolved_issues if i.get("type") == "repeated_action_failure"]
        self.assertEqual(len(matching), 1)

    def test_successful_action_does_not_increment_attempts(self):
        action = start_action(1, action_type="ai_task", purpose="fix the bug")
        finish_action(1, action.id, status="completed", summary="done")

        state = get_session_state(1)
        self.assertEqual(state.completed_actions[-1]["attempts"], 0)
        self.assertEqual(state.unresolved_issues, [])

    def test_different_purposes_track_attempts_independently(self):
        action_a = start_action(1, action_type="ai_task", purpose="fix bug A")
        finish_action(1, action_a.id, status="failed", summary="boom")
        action_b = start_action(1, action_type="ai_task", purpose="fix bug B")
        finish_action(1, action_b.id, status="failed", summary="boom")

        state = get_session_state(1)
        self.assertEqual(state.completed_actions[0]["attempts"], 1)
        self.assertEqual(state.completed_actions[1]["attempts"], 1)
        self.assertEqual(state.unresolved_issues, [])


if __name__ == "__main__":
    unittest.main()
