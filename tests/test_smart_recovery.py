"""Orchestration gaps found in one owner transcript (2026-09-21) -- each was a place where the planner or the
recovery machinery contradicted or buried the user's real task."""
import tempfile
import unittest
from pathlib import Path

from tamfis_code import state as state_module
from tamfis_code.orchestrator import AgentOrchestrator
from tamfis_code.orchestrator.validator import changed_paths_from_evidence
from tamfis_code.return_recap import _first_sentences, _strip_context_chain, build_return_recap
from tamfis_code.runner_local import (
    _checkpoint_resume_objective, _is_machine_generated_objective, _is_real_resume_objective, _is_resume_request,
)


class MachineTextIsNotAnObjectiveTests(unittest.TestCase):
    MACHINE = (
        "Continue from the saved checkpoint and resolve: execution cancelled by user.",
        "Repair the failed plan step, then revalidate it: Read pyproject.toml and requirements.txt",
        "Continue the active plan with: Read requirements.txt for declared dependencies",
        "Continue the interrupted task from the latest saved checkpoint",
        "/retry",
    )

    def test_composer_suggestions_and_recovery_wording_are_recognised(self):
        for text in self.MACHINE:
            self.assertTrue(_is_machine_generated_objective(text), text)
            self.assertFalse(_is_real_resume_objective(text), text)
        self.assertFalse(_is_machine_generated_objective("Fix the TypeError in serve2.py streaming"))
        self.assertTrue(_is_real_resume_objective("Fix the TypeError in serve2.py streaming"))

    def test_a_submitted_suggestion_does_not_replace_or_snowball_the_real_task(self):
        checkpoint = {
            "objective": "Fix the TypeError in serve2.py SSE streaming",
            "messages": [
                {"role": "user", "content": "Fix the TypeError in serve2.py SSE streaming"},
                {"role": "user", "content": self.MACHINE[0]},
                {"role": "user", "content": self.MACHINE[1]},
            ],
        }
        recovered = _checkpoint_resume_objective(checkpoint)
        self.assertEqual(recovered, "Fix the TypeError in serve2.py SSE streaming")
        self.assertNotIn("Additional user context", recovered)


class UserStopIsNotAFailureTests(unittest.TestCase):
    def setUp(self):
        self._orig = (state_module.CONFIG_DIR, state_module.STATE_PATH)
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        state_module.CONFIG_DIR = base / ".config"
        state_module.STATE_PATH = base / ".config" / "state.json"

    def tearDown(self):
        state_module.CONFIG_DIR, state_module.STATE_PATH = self._orig
        self.tmp.cleanup()

    def _run_then_fail(self, error):
        with tempfile.TemporaryDirectory() as root:
            Path(root, "pyproject.toml").write_text("[project]\nname='x'\n")
            engine = AgentOrchestrator(session_id=7100, workspace_root=root, emit=lambda e: None)
            run = engine.begin(objective="implement a multi-file fix and run tests",
                               messages=[{"role": "user", "content": "implement a multi-file fix and run tests"}],
                               read_only=False)
            run.plan.steps[0].status = "in_progress"
            engine.fail(error)
            return [s.status for s in run.plan.steps]

    def test_a_cancel_leaves_the_interrupted_step_pending_not_failed(self):
        self.assertEqual(self._run_then_fail("Execution cancelled by user.")[0], "pending")

    def test_a_real_error_still_fails_the_step(self):
        self.assertEqual(self._run_then_fail("TypeError: can only concatenate str")[0], "failed")


class NoFalseNoFilesChangedTests(unittest.TestCase):
    def test_a_successful_git_diff_is_evidence_of_a_change(self):
        records = [{
            "tool_name": "execute_command", "success": True, "exit_code": 0,
            "arguments": {"command": "git diff --stat"},
            "output": " tamgpt/serve2.py | 2 +-\n 1 file changed, 1 insertion(+), 1 deletion(-)\n",
            "stdout": " tamgpt/serve2.py | 2 +-\n 1 file changed, 1 insertion(+), 1 deletion(-)\n",
        }]
        self.assertEqual([Path(p).name for p in changed_paths_from_evidence(records, "/home/tamgpt")], ["serve2.py"])

    def test_no_evidence_means_no_paths(self):
        self.assertEqual(changed_paths_from_evidence([{"tool_name": "read_file", "success": True}], "/w"), [])


class RecapTextTests(unittest.TestCase):
    def test_identifiers_with_underscores_survive(self):
        out = _first_sentences("Summary - line 431 now reads piece = _decode([token_id]) or \"\".", 260)
        self.assertIn("_decode([token_id])", out)
        self.assertFalse(out.lower().startswith("summary"))

    def test_a_snowballed_objective_shows_only_the_original_task(self):
        chain = ("Fix the TypeError in serve2.py\n\nAdditional user context: continue from the saved checkpoint and "
                 "resolve: x\n\nAdditional user context: continue from the saved checkpoint and resolve: x")
        self.assertEqual(_strip_context_chain(chain), "Fix the TypeError in serve2.py")

    def test_the_recap_objective_skips_submitted_suggestions(self):
        orig = (state_module.CONFIG_DIR, state_module.STATE_PATH)
        with tempfile.TemporaryDirectory() as t:
            state_module.CONFIG_DIR = Path(t) / "c"
            state_module.STATE_PATH = state_module.CONFIG_DIR / "state.json"
            try:
                state_module.save_session_state(3, conversation_history=[
                    {"role": "user", "content": "Fix the TypeError in serve2.py"},
                    {"role": "assistant", "content": "Working on it."},
                    {"role": "user", "content": "Repair the failed plan step, then revalidate it: Read pyproject.toml"},
                    {"role": "assistant", "content": "Done."},
                ])
                self.assertEqual(build_return_recap(3).objective, "Fix the TypeError in serve2.py")
            finally:
                state_module.CONFIG_DIR, state_module.STATE_PATH = orig


if __name__ == "__main__":
    unittest.main()
