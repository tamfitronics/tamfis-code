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
    _completed_saved_plan, _enum_wire_value, _normalize_provider_type, _resume_instruction_for_model,
    _resume_step_contract, _resume_step_is_read_only,
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
        self.assertFalse(_is_real_resume_objective("clear"))
        self.assertFalse(_is_real_resume_objective("/clear"))

    def test_continue_with_a_new_objective_is_not_a_checkpoint_resume(self):
        self.assertFalse(_is_resume_request("continue full finitron codebase improvement"))
        self.assertTrue(_is_resume_request("Continue the interrupted task from the latest saved checkpoint"))
        self.assertTrue(_is_resume_request("continue with steps 1 and 2"))

    def test_completed_saved_plan_is_detected_without_indexing_a_next_step(self):
        from types import SimpleNamespace

        plan = {"id": "plan-1", "steps": [
            {"step": "Inspect files", "status": "completed"},
            {"step": "Run tests", "status": "completed"},
        ]}
        self.assertEqual(_completed_saved_plan(SimpleNamespace(saved_plans=[plan])), plan)
        self.assertIsNone(_completed_saved_plan(SimpleNamespace(saved_plans=[{
            "steps": [{"step": "Run tests", "status": "pending"}],
        }])))


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

    def test_control_word_checkpoint_can_recover_the_original_action_objective(self):
        from tamfis_code.runner_local import _objective_from_completed_actions

        recovered = _objective_from_completed_actions([
            {"purpose": "Execute write_todos for: keep training Finitron models in /home/finitron"},
            {"purpose": "Execute ask_user_question for: clear"},
        ])
        self.assertIn("Finitron", recovered)

    def test_read_only_saved_step_cannot_escalate_repair_wording(self):
        from types import SimpleNamespace

        snapshot = SimpleNamespace(
            resume_step_name="Read pyproject.toml and requirements.txt for declared entry points and dependencies",
        )
        self.assertTrue(_resume_step_is_read_only(self.MACHINE[1], snapshot))
        self.assertIn("Work on this step only", _resume_step_contract(snapshot))

    def test_mutating_saved_step_remains_executable(self):
        from types import SimpleNamespace

        snapshot = SimpleNamespace(resume_step_name="Implement the fix in serve2.py")
        self.assertFalse(_resume_step_is_read_only(self.MACHINE[1], snapshot))

    def test_persisted_provider_wire_name_is_normalized_before_value_access(self):
        from tamfis_code.providers import ProviderType

        self.assertIs(_normalize_provider_type("nvidia"), ProviderType.NVIDIA)
        self.assertIs(_normalize_provider_type(ProviderType.HF.value), ProviderType.HF)
        self.assertIs(_normalize_provider_type("stale-provider", default=ProviderType.AUTO), ProviderType.AUTO)

    def test_machine_repeated_action_error_is_not_replayed_as_the_objective(self):
        prompt = (
            "Continue from the saved checkpoint and resolve: Blocked repeated action: "
            "list_agent_types with identical arguments was already attempted 2 times "
            "without sufficient progress."
        )
        safe = _resume_instruction_for_model(prompt)
        self.assertIn("Continue the original task", safe)
        self.assertNotIn("list_agent_types", safe)

    def test_legacy_resume_does_not_append_guard_error_to_provider_transcript(self):
        from types import SimpleNamespace
        from tamfis_code.runner_local import _legacy_resume_messages

        error_prompt = (
            "Continue from the saved checkpoint and resolve: Blocked repeated action: "
            "list_agent_types with identical arguments was already attempted 2 times"
        )
        state = SimpleNamespace(
            conversation_history=[
                {"role": "user", "content": "Train Finitron Models"},
                {"role": "user", "content": error_prompt},
            ],
            completed_actions=[], active_task={"objective": "Train Finitron Models"},
            modified_files=[], validation_results=[], context_checkpoints=[],
            conversation_summary="",
        )
        messages, objective = _legacy_resume_messages(state, error_prompt)
        assert objective == "Train Finitron Models"
        assert all("list_agent_types" not in str(item.get("content")) for item in messages)


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


class PollutedObjectiveRepairTests(unittest.TestCase):
    """Existing checkpoints written before machine wording was filtered are repaired, not just future ones."""

    SNOWBALL = (
        "Fix the TypeError in serve2.py streaming\n\nAdditional user context: Repair the failed plan step, then "
        "revalidate it: Read pyproject.toml\n\nAdditional user context: continue from the saved checkpoint and "
        "resolve: execution cancelled by user\n\nAdditional user context: continue from the saved checkpoint and "
        "resolve: execution cancelled by user\n\nAdditional user context: continue"
    )

    def test_the_chain_collapses_to_the_real_task(self):
        self.assertEqual(state_module.clean_objective_chain(self.SNOWBALL), "Fix the TypeError in serve2.py streaming")

    def test_a_genuine_clarification_survives(self):
        text = "Fix the login bug\n\nAdditional user context: it only happens on Safari\n\nAdditional user context: continue"
        self.assertEqual(
            state_module.clean_objective_chain(text),
            "Fix the login bug\n\nAdditional user context: it only happens on Safari",
        )

    def test_clean_text_is_unchanged_and_the_repair_is_idempotent(self):
        clean = "Add a login page"
        self.assertEqual(state_module.clean_objective_chain(clean), clean)
        once = state_module.clean_objective_chain(self.SNOWBALL)
        self.assertEqual(state_module.clean_objective_chain(once), once)

    def test_a_stored_row_is_repaired_everywhere_the_objective_lives(self):
        row = {
            "turn_checkpoint": {"objective": self.SNOWBALL, "status": "interrupted"},
            "active_task": {"objective": self.SNOWBALL},
            "saved_plans": [{"id": "p", "objective": self.SNOWBALL, "steps": []}],
        }
        fixed = state_module._clean_row_objectives(row)
        for value in (fixed["turn_checkpoint"]["objective"], fixed["active_task"]["objective"],
                      fixed["saved_plans"][0]["objective"]):
            self.assertEqual(value, "Fix the TypeError in serve2.py streaming")
        self.assertEqual(fixed["turn_checkpoint"]["status"], "interrupted")

    def test_the_resume_objective_of_a_polluted_checkpoint_is_the_real_task(self):
        checkpoint = {"objective": self.SNOWBALL, "messages": []}
        self.assertEqual(_checkpoint_resume_objective(checkpoint), "Fix the TypeError in serve2.py streaming")


class CleanerKeepsWhatAPersonWroteTests(unittest.TestCase):
    def test_words_a_person_added_to_a_suggestion_survive(self):
        text = ("pleae continue /home/x/prompt.docx\n\nAdditional user context: continue the interrupted task from the "
                "latest saved checkpoint; you were here and also check /tmp/tamfis-code for what you prepared\n\n"
                "Additional user context: continue the interrupted task from the latest saved checkpoint")
        cleaned = state_module.clean_objective_chain(text)
        self.assertIn("also check /tmp/tamfis-code for what you prepared", cleaned)
        self.assertNotIn("continue the interrupted task", cleaned.lower())

    def test_an_objective_that_is_only_machinery_is_left_alone_not_blanked(self):
        only = ("Repair the failed plan step, then revalidate it: inspect the file\n\n"
                "Additional user context: continue from the saved checkpoint and resolve: x")
        self.assertEqual(state_module.clean_objective_chain(only), only)
        row = {"active_task": {"objective": only}}
        self.assertEqual(state_module._clean_row_objectives(row)["active_task"]["objective"], only)

    def test_a_slash_command_objective_keeps_its_own_text(self):
        self.assertEqual(
            state_module.clean_objective_chain(
                "/background\n\nAdditional user context: continue from the saved checkpoint and resolve: blocked"),
            "/background",
        )


class CheckpointEnumCompatibilityTests(unittest.TestCase):
    def test_checkpointed_task_profile_strings_render_without_value_error(self):
        self.assertEqual(_enum_wire_value("very_complex"), "very_complex")
