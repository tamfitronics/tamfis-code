import tempfile
import unittest
from pathlib import Path

from tamfis_code import state as state_module
from tamfis_code.orchestrator import AgentOrchestrator, AgentPhase, ToolEnvelope
from tamfis_code.routing import TaskType, classify_task
from tamfis_code.tool_policy import allowed_tools
from tamfis_code.model_registry import get_model


class OrchestratorTests(unittest.TestCase):
    """Every AgentOrchestrator call in this file persists real session
    state via state.py -- without redirecting CONFIG_DIR/STATE_PATH to a
    temp dir (as every other test file's isolation mixin already does),
    these tests wrote directly into the real ~/.config/tamfis-code/
    state.json on every run. One test in particular
    (test_huge_objective_is_not_duplicated_unbounded_into_the_system_prompt,
    session_id 9013) intentionally uses a ~300KB objective; with no
    isolation, repeated suite runs kept appending another huge entry to
    that session's saved_plans (capped at 50, ~150KB each -- confirmed
    live: grew the real state.json to 14.8MB, session 9013 alone
    accounting for 7.5MB, degrading every real `tamfis-code` command's
    save_session_state call, not just these tests)."""

    def setUp(self):
        self._state_originals = (state_module.CONFIG_DIR, state_module.STATE_PATH)
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        state_module.CONFIG_DIR = base / ".config"
        state_module.STATE_PATH = base / ".config" / "state.json"

    def tearDown(self):
        state_module.CONFIG_DIR, state_module.STATE_PATH = self._state_originals
        self._tmp.cleanup()

    def test_complex_edit_creates_persistent_plan_and_context(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root, "pyproject.toml").write_text("[project]\nname='x'\n")
            events = []
            engine = AgentOrchestrator(session_id=9011, workspace_root=root, emit=events.append)
            run = engine.begin(
                objective="implement a multi-file fix and run tests",
                messages=[{"role": "user", "content": "implement a multi-file fix and run tests"}],
                read_only=False,
            )
            self.assertIn(run.profile.task_type, {TaskType.EDIT, TaskType.DEBUG})
            self.assertIsNotNone(run.plan)
            self.assertIsNotNone(run.context)
            self.assertIn("workspace_summary", run.context.layers)
            self.assertEqual(run.phase, AgentPhase.PLAN)

    def test_huge_objective_is_not_duplicated_unbounded_into_the_system_prompt(self):
        """Confirmed live: for a plan-worthy task (complexity=="high", e.g. a
        "fix"/"debug" objective), build_context_bundle used to embed the
        FULL objective twice more inside the leading system message beyond
        its one real appearance as the latest user message -- once directly
        in the "Active orchestration context" supplemental text, and again
        inside the plan dict's own `objective` field via `f"Active plan:
        {plan}"`. None of runner_local.py's compaction passes touch
        role=="system" content (it carries essential workspace instructions
        that must survive compaction), so a large pasted objective (e.g. a
        long log/diff as the "fix this" request) kept blowing the token
        budget even after the real user-facing copy was compacted -- three
        full copies of the same huge text were being sent, only one of
        which compaction could ever reach."""
        with tempfile.TemporaryDirectory() as root:
            huge_objective = "fix the crash caused by this huge log:\n" + ("z" * 300_000)
            events = []
            engine = AgentOrchestrator(session_id=9013, workspace_root=root, emit=events.append)
            run = engine.begin(
                objective=huge_objective,
                messages=[{"role": "user", "content": huge_objective}],
                read_only=False,
            )

            self.assertIsNotNone(run.plan, "this objective must classify as plan-worthy for the test to be meaningful")
            system_message = run.context.messages[0]
            self.assertEqual(system_message["role"], "system")
            # The real, full objective is only supposed to appear once --
            # as the actual latest user message that follows the system
            # message, not duplicated inside the system message itself.
            self.assertLess(len(system_message["content"]), 10_000)

    def test_tool_result_is_persisted_and_observed(self):
        with tempfile.TemporaryDirectory() as root:
            events = []
            engine = AgentOrchestrator(session_id=9012, workspace_root=root, emit=events.append)
            engine.begin(objective="inspect repository", messages=[{"role":"user","content":"inspect repository"}], read_only=True)
            envelope = ToolEnvelope("c1", "read_file", {"path": "x"}, "inspect")
            envelope.finish(result={"success": True, "result": "ok"}, success=True)
            engine.record_tool(envelope)
            self.assertEqual(engine.run.phase, AgentPhase.OBSERVE)
            self.assertEqual(engine.run.tool_records[0].success, True)

    def test_tool_results_advance_plan_step_status(self):
        # Local runs used to create a plan once in begin() and never touch
        # its step statuses again -- render.py was fully built to show
        # in_progress/completed markers, but nothing ever sent them, so
        # local turns always showed every step as pending regardless of
        # real progress. Each observed tool result should now advance a
        # best-effort cursor through the plan, and it must be visible both
        # in-memory and in the persisted saved plan (state.get_plan).
        import tamfis_code.state as state_module

        with tempfile.TemporaryDirectory() as root:
            events = []
            engine = AgentOrchestrator(session_id=90021, workspace_root=root, emit=events.append)
            run = engine.begin(
                objective="implement a multi-file fix and run tests",
                messages=[{"role": "user", "content": "implement a multi-file fix and run tests"}],
                read_only=False,
            )
            self.assertGreaterEqual(len(run.plan.steps), 3, "need >=3 steps for this test to be meaningful")
            self.assertTrue(all(step.status == "pending" for step in run.plan.steps))
            self.assertIsNotNone(run.plan_id)

            envelope = ToolEnvelope("c1", "read_file", {"path": "x"}, "inspect")
            envelope.finish(result={"success": True, "result": "ok"}, success=True)
            engine.record_tool(envelope)

            self.assertEqual(run.plan.steps[0].status, "in_progress")
            self.assertTrue(all(step.status == "pending" for step in run.plan.steps[1:]))

            envelope2 = ToolEnvelope("c2", "write_file", {"path": "x"}, "edit")
            envelope2.finish(result={"success": True, "result": "ok"}, success=True)
            engine.record_tool(envelope2)

            # Tool count is not proof that any particular plan step completed.
            # Until validation establishes completion, the first step remains
            # in progress and later steps remain pending.
            self.assertEqual(run.plan.steps[0].status, "in_progress")
            self.assertTrue(
                all(step.status == "pending" for step in run.plan.steps[1:])
            )
            # The final (report) step must never be auto-advanced into --
            # only complete()/fail() may resolve it.
            self.assertEqual(run.plan.steps[-1].status, "pending")

            persisted = state_module.get_plan(90021, run.plan_id)
            self.assertEqual(
                [s["status"] for s in persisted["steps"]],
                [step.status for step in run.plan.steps],
            )
            # A distinct event type from "plan_created" -- that means "a
            # new/revised plan exists" (reprints the banner, resets the
            # spinner phase); this only means step statuses changed in
            # place, so it must not be conflated with plan_created.
            self.assertTrue(any(e.get("event_type") == "plan_step_progress" for e in events))
            self.assertFalse(any(e.get("event_type") == "plan_created" for e in events))

    def test_complete_marks_all_plan_steps_completed_on_pass(self):
        with tempfile.TemporaryDirectory() as root:
            engine = AgentOrchestrator(session_id=90022, workspace_root=root, emit=lambda event: None)
            run = engine.begin(
                objective="implement a multi-file fix and run tests",
                messages=[{"role": "user", "content": "implement a multi-file fix and run tests"}],
                read_only=False,
            )
            envelope = ToolEnvelope("c1", "write_file", {"path": "x"}, "edit")
            envelope.finish(result={"success": True, "result": "ok"}, success=True)
            engine.record_tool(envelope)
            envelope2 = ToolEnvelope("c2", "execute_command", {"command": "pytest"}, "validate")
            envelope2.finish(result={"success": True, "result": "ok"}, success=True)
            engine.record_tool(envelope2)
            report = engine.complete(final_text="Done, tests pass.", any_mutation=True)
            self.assertTrue(report.passed, report.unresolved)
            self.assertTrue(all(step.status == "completed" for step in run.plan.steps))

    def test_fail_marks_in_progress_step_failed(self):
        with tempfile.TemporaryDirectory() as root:
            engine = AgentOrchestrator(session_id=90023, workspace_root=root, emit=lambda event: None)
            run = engine.begin(
                objective="implement a multi-file fix and run tests",
                messages=[{"role": "user", "content": "implement a multi-file fix and run tests"}],
                read_only=False,
            )
            envelope = ToolEnvelope("c1", "read_file", {"path": "x"}, "inspect")
            envelope.finish(result={"success": True, "result": "ok"}, success=True)
            engine.record_tool(envelope)
            engine.fail("provider unavailable")
            self.assertEqual(run.plan.steps[0].status, "failed")

    def test_replace_plan_persists_new_plan_as_the_active_saved_plan(self):
        # Before this fix, runner_local.py swapped a real reasoning plan
        # into orchestrator.run.plan directly, so the model saw it and the
        # renderer showed it -- but state.saved_plans (what get_plan()/
        # `tamfis-code plan` return) kept the generic synchronous template
        # from begin() forever.
        import tamfis_code.state as state_module
        from tamfis_code.orchestrator import ExecutionPlan
        from tamfis_code.orchestrator.planner import PlanStep

        with tempfile.TemporaryDirectory() as root:
            engine = AgentOrchestrator(session_id=90024, workspace_root=root, emit=lambda event: None)
            run = engine.begin(
                objective="fix the login bug",
                messages=[{"role": "user", "content": "fix the login bug"}],
                read_only=False,
            )
            original_plan_id = run.plan_id
            real_plan = ExecutionPlan(
                objective="fix the login bug",
                assumptions=[], components=[],
                steps=[PlanStep(1, "Read auth.py to find the actual bug"), PlanStep(2, "Fix and verify")],
                validation_criteria=[], risks=[],
            )
            engine.replace_plan(real_plan)
            self.assertNotEqual(run.plan_id, original_plan_id)
            self.assertIs(run.plan, real_plan)
            persisted = state_module.get_plan(90024)
            self.assertEqual(persisted["id"], run.plan_id)
            self.assertEqual([s["step"] for s in persisted["steps"]], ["Read auth.py to find the actual bug", "Fix and verify"])

    def test_validation_rejects_change_claim_without_mutation(self):
        with tempfile.TemporaryDirectory() as root:
            engine = AgentOrchestrator(session_id=9013, workspace_root=root, emit=lambda event: None)
            engine.begin(objective="fix app.py", messages=[{"role":"user","content":"fix app.py"}], read_only=False)
            report = engine.complete(final_text="Fixed.", any_mutation=False)
            self.assertFalse(report.passed)
            self.assertTrue(report.unresolved)

    def test_plain_chat_has_no_tools(self):
        profile = classify_task("hello")
        self.assertEqual(allowed_tools(profile, read_only=False), [])

    def test_audit_has_read_only_tools_in_read_only_mode(self):
        profile = classify_task("audit the entire repository", read_only=True)
        tools = allowed_tools(profile, read_only=True)
        self.assertIn("read_file", tools)
        self.assertNotIn("write_file", tools)

    def test_ask_user_question_is_offered_even_in_read_only_mode(self):
        """User-requested: the agent should be able to pause and ask instead
        of guessing, even during a read-only audit -- exactly the scenario
        that motivated it (an audit that couldn't verify the real project
        type and defaulted to a Node/React guess instead)."""
        profile = classify_task("audit the entire repository", read_only=True)
        self.assertIn("ask_user_question", allowed_tools(profile, read_only=True))

    def test_misclassified_edit_request_still_gets_edit_tools_outside_read_only_mode(self):
        """Regression: confirmed live -- "make the TamfisPress child theme
        full-width" (a genuine edit request) doesn't contain any of
        classify_task's hardcoded EDIT keywords ("edit", "modify", "add ",
        "create ", "implement", etc.), so it fell through to the QUESTION
        catch-all. allowed_tools then forced READ_TOOLS for *any* QUESTION
        profile regardless of read_only, even in the default coding/auto
        mode (read_only=False) -- so the model had no write_file/edit_file
        tool and told the user it couldn't edit files directly instead of
        just doing the edit."""
        objective = "make the TamfisPress child theme full-width"
        profile = classify_task(objective, read_only=False)
        self.assertEqual(profile.task_type, TaskType.QUESTION)
        tools = allowed_tools(profile, read_only=False)
        self.assertIn("write_file", tools)
        self.assertIn("edit_file", tools)
        self.assertIn("extract_archive", tools)
        self.assertIn("repackage_archive", tools)

    def test_question_is_still_read_only_when_explicitly_in_read_only_mode(self):
        profile = classify_task("what does this function do", read_only=True)
        tools = allowed_tools(profile, read_only=True)
        self.assertNotIn("write_file", tools)
        self.assertNotIn("edit_file", tools)

    def test_research_request_is_offered_web_search_and_browser(self):
        # Regression: TaskType.RESEARCH previously had no classify_task
        # branch, so this path (and RESEARCH_TOOLS) was unreachable dead
        # code -- the agent loop could never actually offer web_search or
        # browser to the model, only `tamfis-code tools call` could.
        profile = classify_task("search the web for the current release of FastAPI", read_only=False)
        self.assertEqual(profile.task_type, TaskType.RESEARCH)
        tools = allowed_tools(profile, read_only=False)
        self.assertIn("web_search", tools)
        self.assertIn("browser", tools)
        self.assertNotIn("write_file", tools)

    def test_kimi_k2_6_registered_for_both_nvidia_and_hf_casings(self):
        # NVIDIA's own kimi-k2.6 route is a per-account entitlement gap
        # (see test_routing.py), not a reason to drop the model from the
        # registry -- confirmed live it also works on HF, under a
        # different, case-sensitive id.
        nvidia_entry = get_model("moonshotai/kimi-k2.6")
        self.assertIsNotNone(nvidia_entry)
        self.assertEqual(nvidia_entry.provider, "nvidia")
        hf_entry = get_model("moonshotai/Kimi-K2.6")
        self.assertIsNotNone(hf_entry)
        self.assertEqual(hf_entry.provider, "hf")

    def test_plain_question_in_read_only_mode_validates_without_tool_evidence(self):
        # Regression: classify_task used to set requires_tools=read_only for
        # the generic QUESTION fallback, so a trivial chat-mode question
        # (which needs no tool call) always failed validate_completion's
        # tool_evidence_recorded check -- and since that check never
        # explained itself via `unresolved`, the CLI printed a bare,
        # unexplained "Validation incomplete" caveat on every plain chat
        # answer.
        with tempfile.TemporaryDirectory() as root:
            engine = AgentOrchestrator(session_id=9014, workspace_root=root, emit=lambda event: None)
            engine.begin(
                objective="reply with exactly the word PONG and nothing else",
                messages=[{"role": "user", "content": "reply with exactly the word PONG and nothing else"}],
                read_only=True,
            )
            report = engine.complete(final_text="PONG", any_mutation=False)
        self.assertTrue(report.passed)
        self.assertEqual(report.unresolved, [])

    def test_tool_evidence_failure_explains_itself_in_unresolved(self):
        # If a task type that genuinely requires_tools=True completes with
        # no successful tool call, the failure must be explained, not just
        # silently fail the boolean check (which used to render as a bare
        # "Validation incomplete: " with nothing after the colon).
        from tamfis_code.orchestrator.validator import validate_completion
        from tamfis_code.routing import TaskProfile, TaskType

        profile = TaskProfile(TaskType.INSPECT, "medium", True, True, False, False, "high")
        report = validate_completion(profile=profile, tool_records=[], any_mutation=False, final_text="answer")
        self.assertFalse(report.passed)
        self.assertTrue(report.unresolved)
