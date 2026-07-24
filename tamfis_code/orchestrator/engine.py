"""Persistent Claude Code/Codex-style orchestration state machine."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .. import state as local_state
from ..routing import TaskProfile, classify_task
from .context import ContextBundle, build_context_bundle
from .planner import ExecutionPlan, create_plan
from .protocols import AgentPhase, ToolEnvelope
from .validator import ValidationReport, validate_completion
from ..runtime import ExecutionController, GuardDecision, ObservationDecision


@dataclass
class OrchestrationRun:
    session_id: int
    objective: str
    profile: TaskProfile
    phase: AgentPhase = AgentPhase.UNDERSTAND
    plan: ExecutionPlan | None = None
    plan_id: str | None = None
    context: ContextBundle | None = None
    tool_records: list[ToolEnvelope] = field(default_factory=list)
    validation: ValidationReport | None = None
    route: dict[str, Any] = field(default_factory=dict)
    repair_attempts: int = 0
    reasoning_plan: bool = False
    runtime: ExecutionController = field(default_factory=ExecutionController)


class AgentOrchestrator:
    def __init__(self, *, session_id: int, workspace_root: str, emit: Callable[[dict[str, Any]], None]):
        self.session_id = session_id
        self.workspace_root = workspace_root
        self.emit = emit
        self.run: OrchestrationRun | None = None

    def transition(self, phase: AgentPhase, *, action: str = "") -> None:
        if self.run is None:
            raise RuntimeError("orchestration run has not started")
        self.run.phase = phase
        local_state.save_session_state(
            self.session_id, current_phase=phase.value,
            execution_status="failed" if phase == AgentPhase.FAILED else (
                "completed" if phase == AgentPhase.COMPLETED else "running"
            ),
            running_action={"purpose": action or phase.value, "phase": phase.value},
        )
        self.emit({"event_type": f"orchestrator_{phase.value}", "payload": {"phase": phase.value, "action": action}})

    def begin(self, *, objective: str, messages: list[dict[str, Any]], read_only: bool) -> OrchestrationRun:
        profile = classify_task(objective, read_only=read_only)
        self.run = OrchestrationRun(self.session_id, objective, profile)
        local_state.save_session_state(
            self.session_id,
            active_task={"objective": objective, "task_type": profile.task_type.value, "complexity": profile.complexity},
            current_phase=AgentPhase.UNDERSTAND.value, execution_status="running",
        )
        self.transition(AgentPhase.UNDERSTAND, action="Classify the request deterministically")
        if profile.requires_repository_context:
            self.transition(AgentPhase.INSPECT, action="Load or refresh repository context")
        self.run.plan = create_plan(objective, profile)
        self.run.runtime.start_planning()
        plan_dict = self.run.plan.to_dict() if self.run.plan else None
        self.run.context = build_context_bundle(
            session_id=self.session_id, workspace_root=self.workspace_root,
            objective=objective, profile=profile, conversation_messages=messages, plan=plan_dict,
        )
        if self.run.plan:
            self.transition(AgentPhase.PLAN, action="Persist an executable plan")
            saved = local_state.save_plan(
                self.session_id, objective=objective,
                content="\n".join(f"{s.index}. {s.name}" for s in self.run.plan.steps),
                steps=[{"index": s.index, "step": s.name, "status": s.status} for s in self.run.plan.steps],
            )
            self.run.plan_id = saved.id
        return self.run

    def replace_plan(self, plan: ExecutionPlan) -> None:
        """Swap in a plan grounded in real evidence (the initial reasoning
        plan, or a mid-turn revision) and persist it under a fresh plan id --
        keeping `state.saved_plans`/`get_plan()` in sync with whatever plan
        is actually driving the turn, instead of leaving the synchronous
        deterministic-template plan from begin() as the persisted record of
        record. Callers still emit their own `plan_created` renderer event
        for the "here is the new plan" banner; this only handles state.
        """
        assert self.run is not None
        if not self.run.runtime.record_plan_revision():
            self.fail(self.run.runtime.snapshot.failure_reason)
            return
        saved = local_state.save_plan(
            self.session_id, objective=self.run.objective,
            content="\n".join(f"{s.index}. {s.name}" for s in plan.steps),
            steps=[{"index": s.index, "step": s.name, "status": s.status} for s in plan.steps],
        )
        self.run.plan = plan
        self.run.plan_id = saved.id

    def _sync_plan_progress(self) -> None:
        """Persist current step statuses and let the renderer live-update
        the same way it already does for a freshly created plan (render.py
        explicitly documents step statuses beyond the initial plan_created
        payload as a best-effort approximation, not precise per-step
        completion tracking -- this keeps that promise honest rather than
        inventing false precision)."""
        assert self.run is not None
        if self.run.plan is None or self.run.plan_id is None:
            return
        items = [{"step": s.name, "status": s.status} for s in self.run.plan.steps]
        local_state.update_plan_steps(self.session_id, self.run.plan_id, items)
        # Deliberately a distinct event type from "plan_created" -- that
        # event means "a new/revised plan now exists" (renderer reprints
        # the plan banner and resets the spinner phase to "plan" on it);
        # this only means "the existing plan's step statuses changed",
        # which should update the live step markers in place with none of
        # that -- no banner reprint, no spinner phase change, every round.
        self.emit({"event_type": "plan_step_progress", "payload": {"items": items}})

    def record_route(self, *, provider: str, model: str, reason: str, fallback_chain: list[str] | None = None) -> None:
        assert self.run is not None
        self.transition(AgentPhase.ROUTE, action="Select a capability-matched provider and model")
        self.run.route = {"provider": provider, "model": model, "reason": reason, "fallback_chain": fallback_chain or []}
        local_state.save_session_state(self.session_id, selected_provider=provider, selected_model=model)

    def start_execution(self) -> None:
        assert self.run is not None
        self.run.runtime.start_execution()
        self.transition(AgentPhase.EXECUTE, action="Execute the model/tool loop")

    def guard_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> GuardDecision:
        assert self.run is not None
        decision = self.run.runtime.guard_action(tool_name, arguments)
        if not decision.allowed:
            self.emit({"event_type": "diagnostics", "payload": {"content": decision.reason}})
        return decision

    def waiting_for_approval(self, purpose: str) -> None:
        self.transition(AgentPhase.WAITING_FOR_APPROVAL, action=purpose)

    def record_tool(self, envelope: ToolEnvelope) -> ObservationDecision:
        assert self.run is not None
        self.run.tool_records.append(envelope)
        self.transition(AgentPhase.OBSERVE, action=f"Observe {envelope.tool_name} result")
        result = {
            "success": bool(envelope.success),
            "result": {
                "stdout": envelope.stdout,
                "stderr": envelope.stderr,
                "exit_code": envelope.exit_code,
                "files_changed": list(envelope.files_changed),
                "path": envelope.arguments.get("path") or envelope.arguments.get("destination"),
            },
        }
        decision = self.run.runtime.observe(envelope.tool_name, envelope.arguments, result)
        state = local_state.get_session_state(self.session_id)
        records = state.completed_actions + [{"type": "tool", **envelope.to_dict()}]
        local_state.save_session_state(
            self.session_id, completed_actions=records[-250:],
            running_action={
                "purpose": decision.reason or f"Observed {envelope.tool_name}",
                "phase": self.run.runtime.snapshot.phase.value,
                "runtime": self.run.runtime.snapshot.to_dict(),
            },
        )
        self._advance_plan_step(decision)
        if decision.terminal:
            self.fail(decision.reason)
        return decision

    def _advance_plan_step(self, decision: ObservationDecision) -> None:
        """Advance only when a successful observation gained useful evidence."""
        assert self.run is not None
        if self.run.plan is None or not self.run.plan.steps:
            return
        active = next((step for step in self.run.plan.steps if step.status == "in_progress"), None)
        if active is None:
            active = next((step for step in self.run.plan.steps if step.status == "pending"), None)
            if active is not None:
                active.status = "in_progress"
        if active is not None and decision.useful:
            active.status = "completed"
            active.evidence.extend(item for item in decision.evidence if item not in active.evidence)
            nxt = next((step for step in self.run.plan.steps if step.status == "pending"), None)
            if nxt is not None:
                nxt.status = "in_progress"
        self._sync_plan_progress()

    def mark_repair(self, reason: str) -> None:
        assert self.run is not None
        self.run.repair_attempts += 1
        if not self.run.runtime.record_repair():
            self.fail(self.run.runtime.snapshot.failure_reason)
            return
        self.transition(AgentPhase.REPAIR, action=reason)

    def validate(self, *, final_text: str, any_mutation: bool) -> ValidationReport:
        assert self.run is not None
        self.run.runtime.begin_validation()
        self.transition(AgentPhase.VALIDATE, action="Validate evidence and completion claims")
        report = validate_completion(
            profile=self.run.profile,
            tool_records=[item.to_dict() for item in self.run.tool_records],
            any_mutation=any_mutation, final_text=final_text,
        )
        if (
            self.run.reasoning_plan
            and self.run.profile.task_type.value == "audit"
            and self.run.plan is not None
        ):
            pending = [
                step.name for step in self.run.plan.steps
                if step.status in {"pending", "in_progress"}
            ]
            if pending:
                report.passed = False
                report.unresolved.append(
                    "Execution plan incomplete; pending steps: " + "; ".join(pending)
                )
                if report.severity == "pass":
                    report.severity = "warning"
        self.run.validation = report
        state = local_state.get_session_state(self.session_id)
        local_state.save_session_state(
            self.session_id,
            validation_results=(state.validation_results + [report.to_dict()])[-100:],
            unresolved_issues=[{"issue": item} for item in report.unresolved],
        )
        return report

    def complete(self, *, final_text: str, any_mutation: bool) -> ValidationReport:
        report = self.validate(final_text=final_text, any_mutation=any_mutation)
        if self.run is not None and self.run.plan is not None:
            for step in self.run.plan.steps:
                step.status = "completed" if report.passed else (
                    "failed" if step.status == "in_progress" else step.status
                )
            self._sync_plan_progress()
        self.transition(AgentPhase.REPORT, action="Report only evidence-supported outcomes")
        if report.severity == "error":
            self.run.runtime.fail("Completion validation failed.")
        else:
            self.run.runtime.complete()
        self.transition(AgentPhase.FAILED if report.severity == "error" else AgentPhase.COMPLETED)
        local_state.checkpoint(self.session_id, reason="orchestrator_complete", summary=final_text[-1000:])
        return report

    def fail(self, error: str) -> None:
        if self.run is not None:
            self.run.runtime.fail(error)
            if self.run.plan is not None:
                for step in self.run.plan.steps:
                    if step.status == "in_progress":
                        step.status = "failed"
                self._sync_plan_progress()
            self.transition(AgentPhase.FAILED, action=error)
        local_state.checkpoint(self.session_id, reason="orchestrator_failed", summary=error[-1000:])
