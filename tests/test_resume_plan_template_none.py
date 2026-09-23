"""Resume must tolerate a legacy plan when the current task needs no plan."""
from __future__ import annotations

import unittest

from tamfis_code.orchestrator.engine import AgentOrchestrator
from tamfis_code.routing import classify_task
from tamfis_code.runtime.resume import ResumeSnapshot


class ResumePlanTemplateNoneTests(unittest.TestCase):
    def test_plain_conversation_resume_does_not_dereference_missing_template(self):
        snapshot = ResumeSnapshot(
            session_id=507,
            plan_id="legacy-plan",
            objective="hello",
            steps=[{"name": "finish the interrupted action", "status": "pending"}],
            static={},
            ledger_next_action="continue",
        )
        orchestrator = AgentOrchestrator(session_id=507, workspace_root="/tmp", emit=lambda _: None)

        plan = orchestrator._plan_from_snapshot("hello", classify_task("hello"), snapshot)

        self.assertEqual(plan.assumptions, [])
        self.assertEqual(plan.components, [])
        self.assertEqual(plan.validation_criteria, [])
        self.assertEqual(plan.risks, [])
        self.assertEqual(plan.steps[0].name, "finish the interrupted action")


if __name__ == "__main__":
    unittest.main()
