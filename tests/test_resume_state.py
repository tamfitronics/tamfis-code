"""A `continue` must RESUME where the task stopped, not re-evaluate it.

Live-reported 2026-09-19: tamfis-code did not keep accurate state for resuming
exactly where it left off -- it "has to re-evaluate again and again". Reproduced:
AgentOrchestrator.begin() ran at the start of EVERY turn, resume included, and
reset the durable task record, built a fresh all-pending template plan, saved it
as a second plan and wrote the ledger from that: a task interrupted at 3 of 4
steps came back as 0 of 4 and /status said nothing was done.

These tests pin the contract: a resume RESTORES the saved plan (each step keeps
its status), keeps the task record, does not re-plan on a bare "continue", and
/status says where it will pick up. A genuinely new task still starts clean.
"""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import AsyncMock, patch

from rich.console import Console

from tamfis_code import state as state_module
from tamfis_code.orchestrator.engine import AgentOrchestrator
from tamfis_code.providers import ProviderType
from tamfis_code.runner_local import (
    _is_bare_continue,
    _resume_plan_message_content,
    run_local_agent_turn,
)
from tamfis_code.runtime import ledger as ledger_module
from tamfis_code.runtime.ledger import load_ledger
from tamfis_code.runtime.resume import describe_resume_point, load_resume_snapshot

from test_reasoning_plan import (
    _FakeClient,
    _FakeManager,
    _RecordingRenderer,
    _chunk,
    _delta,
)

OBJECTIVE = "Audit the repository, inspect src/a.py and src/b.py, then fix all bugs and write a report"


class _IsolatedState:
    def setUp(self):
        self._originals = (
            state_module.CONFIG_DIR, state_module.STATE_PATH, state_module._LOCK_PATH,
            ledger_module.LEDGER_DIR,
        )
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        state_module.CONFIG_DIR = base / ".config"
        state_module.STATE_PATH = base / ".config" / "state.json"
        state_module._LOCK_PATH = base / ".config" / ".state.lock"
        state_module._STATE_CACHE = None
        ledger_module.LEDGER_DIR = base / "ledgers"

    def tearDown(self):
        (
            state_module.CONFIG_DIR, state_module.STATE_PATH, state_module._LOCK_PATH,
            ledger_module.LEDGER_DIR,
        ) = self._originals
        state_module._STATE_CACHE = None
        self._tmp.cleanup()

    def _interrupt_after(self, session_id, done=3):
        """A task that ran `done` plan steps and was then interrupted."""
        state_module.save_session_state(session_id, workspace_root="/tmp")
        first = AgentOrchestrator(session_id=session_id, workspace_root="/tmp", emit=lambda e: None)
        first.begin(objective=OBJECTIVE, messages=[{"role": "user", "content": OBJECTIVE}], read_only=False)
        for step in first.run.plan.steps[:done]:
            step.status = "completed"
            step.evidence.append(f"done: {step.name}")
        first._sync_plan_progress()
        first.save_task_ledger(status="partial", next_action="continue")
        state_module.update_task_state(
            session_id, files_read=["src/a.py", "src/b.py"], decisions=["bug is in the retry loop"],
        )
        return first

    def _progress(self, session_id):
        ledger = load_ledger(str(session_id))
        statuses = [step.status for step in ledger.plan_steps]
        task_state = state_module.get_session_state(session_id).task_state or {}
        return {
            "done": sum(1 for status in statuses if status == "completed"),
            "total": len(statuses),
            "files_read": list(task_state.get("files_read") or []),
            "decisions": list(task_state.get("decisions") or []),
            "saved_plans": len(state_module.get_session_state(session_id).saved_plans),
            "next_action": ledger.next_action,
        }


class BeginRestoresInsteadOfResettingTests(_IsolatedState, unittest.TestCase):
    def test_a_resume_keeps_the_plan_progress_the_task_record_and_a_single_plan(self):
        self._interrupt_after(501, done=3)
        before = self._progress(501)
        self.assertEqual((before["done"], before["total"]), (3, 4))

        resumed = AgentOrchestrator(session_id=501, workspace_root="/tmp", emit=lambda e: None)
        resumed.begin(
            objective=OBJECTIVE, messages=[{"role": "user", "content": OBJECTIVE}], read_only=False,
            restore=load_resume_snapshot(501),
        )
        after = self._progress(501)
        # The reproduced bug: this came back as 0/4 with every list emptied.
        self.assertEqual((after["done"], after["total"]), (3, 4))
        self.assertEqual(after["files_read"], ["src/a.py", "src/b.py"])
        self.assertEqual(after["decisions"], ["bug is in the retry loop"])
        self.assertEqual(after["saved_plans"], 1)  # no second copy of the plan
        self.assertTrue(resumed.run.plan_restored)
        self.assertEqual([s.status for s in resumed.run.plan.steps][:3], ["completed"] * 3)
        self.assertTrue(resumed.run.plan.steps[0].evidence)  # evidence survives too

    def test_next_action_says_where_the_task_stands_not_begin_executing(self):
        self._interrupt_after(502, done=2)
        resumed = AgentOrchestrator(session_id=502, workspace_root="/tmp", emit=lambda e: None)
        resumed.begin(objective=OBJECTIVE, messages=[], read_only=False, restore=load_resume_snapshot(502))
        next_action = load_ledger("502").next_action
        self.assertTrue(next_action.startswith("Resume at step 3/4:"), next_action)

    def test_resume_keeps_saved_objective_when_wrapper_prompt_is_used(self):
        """Recovery wording must not become a new task objective."""
        self._interrupt_after(506, done=2)
        snapshot = load_resume_snapshot(506)
        wrapper = "Continue from the saved checkpoint and resolve: provider failure"
        resumed = AgentOrchestrator(session_id=506, workspace_root="/tmp", emit=lambda e: None)
        resumed.begin(objective=wrapper, messages=[], read_only=False, restore=snapshot)
        state = state_module.get_session_state(506)
        self.assertEqual(resumed.run.objective, OBJECTIVE)
        self.assertEqual(state.active_task["objective"], OBJECTIVE)
        self.assertEqual(load_ledger("506").objective, OBJECTIVE)

    def test_the_restored_plan_keeps_its_non_step_parts(self):
        first = self._interrupt_after(503, done=1)
        first.run.plan.risks.append("a risk the planner found")
        first.save_task_ledger(status="partial", next_action="x")
        resumed = AgentOrchestrator(session_id=503, workspace_root="/tmp", emit=lambda e: None)
        resumed.begin(objective=OBJECTIVE, messages=[], read_only=False, restore=load_resume_snapshot(503))
        self.assertIn("a risk the planner found", resumed.run.plan.risks)

    def test_a_new_task_without_a_snapshot_still_starts_clean(self):
        self._interrupt_after(504, done=3)
        fresh = AgentOrchestrator(session_id=504, workspace_root="/tmp", emit=lambda e: None)
        fresh.begin(objective=OBJECTIVE + " -- a different task", messages=[], read_only=False)
        after = self._progress(504)
        self.assertEqual(after["files_read"], [])
        self.assertEqual(after["decisions"], [])
        self.assertEqual(after["done"], 0)
        self.assertFalse(fresh.run.plan_restored)

    def test_a_plan_picked_up_from_another_session_brings_its_task_record_with_it(self):
        """`resume` selects the newest interrupted session for the workspace, so
        the interrupted plan can live in a DIFFERENT session than the one that
        runs the resume turn."""
        self._interrupt_after(801, done=3)
        state_module.save_session_state(802, workspace_root="/tmp")
        snapshot = load_resume_snapshot(801)
        resumed = AgentOrchestrator(session_id=802, workspace_root="/tmp", emit=lambda e: None)
        resumed.begin(objective=OBJECTIVE, messages=[], read_only=False, restore=snapshot)
        after = self._progress(802)
        self.assertEqual((after["done"], after["total"]), (3, 4))
        self.assertEqual(after["files_read"], ["src/a.py", "src/b.py"])
        self.assertEqual(after["decisions"], ["bug is in the retry loop"])
        self.assertEqual(after["saved_plans"], 1)  # the plan was saved into THIS session, statuses intact
        saved_steps = state_module.get_session_state(802).saved_plans[0]["steps"]
        self.assertEqual([s["status"] for s in saved_steps][:3], ["completed"] * 3)

    def test_a_plan_less_task_does_not_inherit_the_previous_tasks_progress(self):
        """The ledger is keyed by session: a one-line task has NO plan, and used
        to keep the previous task's "3/4 steps done" in /status."""
        self._interrupt_after(505, done=3)
        fresh = AgentOrchestrator(session_id=505, workspace_root="/tmp", emit=lambda e: None)
        fresh.begin(objective="Write a haiku about rain", messages=[], read_only=False)
        self.assertIsNone(fresh.run.plan)
        self.assertEqual(load_ledger("505").plan_steps, [])


class SnapshotTests(_IsolatedState, unittest.TestCase):
    def test_nothing_to_resume_when_every_step_is_done_or_there_is_no_plan(self):
        state_module.save_session_state(510, workspace_root="/tmp")
        self.assertIsNone(load_resume_snapshot(510))
        self._interrupt_after(511, done=4)
        self.assertIsNone(load_resume_snapshot(511))
        self.assertIsNone(describe_resume_point(511))

    def test_the_resume_point_is_the_first_step_not_completed(self):
        self._interrupt_after(512, done=2)
        point = describe_resume_point(512)
        self.assertEqual((point["step"], point["total"], point["done"]), (3, 4, 2))
        self.assertTrue(point["next_action"].startswith("Resume at step 3/4"))

    def test_an_interrupted_in_progress_step_is_where_the_resume_starts(self):
        first = self._interrupt_after(513, done=2)
        first.run.plan.steps[2].status = "in_progress"
        first._sync_plan_progress()
        self.assertEqual(describe_resume_point(513)["step"], 3)

    def test_the_banner_names_the_step_and_says_it_is_not_replanning(self):
        self._interrupt_after(514, done=3)
        banner = load_resume_snapshot(514).banner()
        self.assertTrue(banner.startswith("◆ Resuming at step 4/4"))
        self.assertIn("3 done", banner)
        self.assertIn("instead of re-planning", banner)


class BareContinueTests(unittest.TestCase):
    def test_a_bare_continue_is_recognised(self):
        for text in ("continue", "Continue", "please continue", "resume", "keep going", "go on"):
            with self.subTest(text=text):
                self.assertTrue(_is_bare_continue(text))

    def test_a_continue_that_carries_an_instruction_is_not_bare(self):
        for text in (
            "proceed with 1, 2 and 3",
            "continue, but also add tests to auth.py",
            "fix the failing test",
            "",
        ):
            with self.subTest(text=text):
                self.assertFalse(_is_bare_continue(text))

    def test_the_resume_plan_message_marks_done_next_and_remaining(self):
        from tamfis_code.orchestrator.planner import ExecutionPlan, PlanStep

        plan = ExecutionPlan(
            objective="o", assumptions=[], components=[], validation_criteria=[], risks=[],
            steps=[
                PlanStep(1, "inspect", "completed", ["read a.py"]),
                PlanStep(2, "fix", "in_progress"),
                PlanStep(3, "report", "pending"),
            ],
        )
        lines = _resume_plan_message_content(plan).splitlines()
        self.assertIn("RESUMED", lines[0])
        self.assertEqual(lines[1], "✓ 1. inspect")
        self.assertIn("read a.py", lines[2])
        self.assertEqual(lines[3], "▶ 2. fix")
        self.assertEqual(lines[4], "· 3. report")


class RunnerResumeTests(_IsolatedState, unittest.TestCase):
    """End to end through run_local_agent_turn with a scripted model."""

    def _resumable_session(self, session_id):
        self._interrupt_after(session_id, done=3)
        state_module.save_session_state(
            session_id, workspace_root="/tmp", execution_status="interrupted",
            turn_checkpoint={
                "objective": OBJECTIVE,
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": OBJECTIVE},
                    {"role": "assistant", "content": "I read src/a.py and src/b.py; the bug is in the retry loop."},
                ],
                "partial_assistant": "",
            },
        )

    def _turn(self, session_id, text):
        client = _FakeClient([[_chunk(_delta(content="Done."), finish_reason="stop")]] * 12)
        renderer = _RecordingRenderer()
        planner = AsyncMock(return_value=None)
        recon = patch("tamfis_code.runner_local._build_planning_reconnaissance", return_value="recon")
        with patch("tamfis_code.runner_local._attempt_reasoning_plan", planner), recon, patch(
            "tamfis_code.state.upgrade_session_title_with_ai", new=AsyncMock(),
        ):
            asyncio.run(run_local_agent_turn(
                _FakeManager(client), ProviderType.NVIDIA, None,
                [{"role": "user", "content": text}],
                Console(file=StringIO(), no_color=True, width=200), renderer,
                workspace_root="/tmp", session_id=session_id, approval_policy="auto",
                interactive=False,
            ))
        return client, renderer, planner

    def test_a_bare_continue_does_not_re_plan_and_hands_the_model_the_saved_plan(self):
        self._resumable_session(601)
        client, renderer, planner = self._turn(601, "continue")

        planner.assert_not_called()  # no fresh LLM plan, no repository walk
        plan_events = [e for e in renderer.events if e["event_type"] == "plan_created"]
        self.assertEqual(len(plan_events), 1)
        payload = plan_events[0]["payload"]
        self.assertTrue(payload["continuation"])
        self.assertEqual([i["status"] for i in payload["items"]][:3], ["completed"] * 3)
        sent = "\n".join(str(m.get("content")) for m in client.calls[0]["messages"])
        self.assertIn("RESUMED", sent)
        self.assertIn("✓ 1.", sent)
        self.assertIn("▶ 4.", sent)
        banner = [
            e["payload"]["content"] for e in renderer.events
            if e["event_type"] == "diagnostics" and "Resuming at step" in e["payload"].get("content", "")
        ]
        self.assertTrue(banner, [e for e in renderer.events if e["event_type"] == "diagnostics"])

    def test_the_task_record_and_single_plan_survive_the_resume_turn(self):
        self._resumable_session(602)
        self._turn(602, "continue")
        progress = self._progress(602)
        self.assertEqual(progress["files_read"], ["src/a.py", "src/b.py"])
        self.assertEqual(progress["decisions"], ["bug is in the retry loop"])
        self.assertEqual(progress["saved_plans"], 1)
        self.assertGreaterEqual(progress["done"], 3)

    def test_a_resume_that_carries_a_new_instruction_still_re_plans(self):
        self._resumable_session(603)
        _client, renderer, planner = self._turn(603, "proceed with 1, 2 and 3")
        planner.assert_called()  # a real instruction: plan for it, as before
        # ...but the durable record is still kept, not wiped.
        self.assertEqual(self._progress(603)["files_read"], ["src/a.py", "src/b.py"])


class StatusShowsTheResumePointTests(_IsolatedState, unittest.TestCase):
    def test_status_reports_the_step_a_continue_picks_up_at(self):
        import io
        from unittest.mock import MagicMock

        from tamfis_code.config import Config
        from tamfis_code.interactive import run_interactive
        from tamfis_code.workspace import WorkspaceContext

        self._interrupt_after(701, done=2)
        console = Console(file=io.StringIO(), no_color=True, width=200)
        inputs = iter(["/status", EOFError()])

        async def _prompt(*args, **kwargs):
            value = next(inputs)
            if isinstance(value, BaseException):
                raise value
            return value

        with patch("tamfis_code.interactive.Console", return_value=console), \
                patch("tamfis_code.interactive.PromptSession") as session_cls, \
                patch("tamfis_code.interactive.print_banner"):
            session_cls.return_value.prompt_async = _prompt
            asyncio.run(run_interactive(
                client=None, config=Config(),
                workspace=WorkspaceContext(session_id=701, workspace_root="/tmp"),
            ))
        output = console.file.getvalue()
        self.assertIn("resume_at=step 3/4", output)
        self.assertIn("2 done", output)
        self.assertIn("continue", output)


if __name__ == "__main__":
    unittest.main()
