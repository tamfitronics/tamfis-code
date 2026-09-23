"""Persistent Claude Code/Codex-style orchestration state machine."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePath
import os
import re
from typing import Any, Callable, Optional

from .. import state as local_state
from ..routing import TaskProfile, classify_task
from .context import ContextBundle, build_context_bundle
from .planner import ExecutionPlan, PlanStep, create_plan
from .protocols import AgentPhase, ToolEnvelope, classify_failure
from .validator import ValidationReport, validate_completion
from .production import DeliveryPolicy, build_delivery_policy, completion_is_evidence_bound
from ..workspace import load_instruction_text
from ..runtime import ExecutionController, GuardDecision, ObservationDecision
from ..runtime.resume import ResumeSnapshot
from ..runtime.budgets import RuntimeBudgets


# The durable task-record lists that describe WHAT HAS BEEN DONE so far.
_TASK_RECORD_KEYS = (
    "blocked_steps", "assumptions", "decisions", "files_read", "files_modified",
    "artifacts_created", "commands_run", "tests", "failures", "retries",
    "completion_evidence",
)


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
    # True when this turn CONTINUES a saved plan (steps keep their statuses)
    # instead of starting from a fresh template. See begin(restore=...).
    plan_restored: bool = False
    runtime: ExecutionController = field(default_factory=ExecutionController)
    delivery_policy: DeliveryPolicy | None = None

    def __post_init__(self) -> None:
        # Durable checkpoints and older callers serialize enum values as
        # strings.  Keep the in-memory state typed so every later
        # ``phase.value`` access is safe during resume and plan execution.
        if isinstance(self.phase, str):
            self.phase = AgentPhase(self.phase)


class AgentOrchestrator:
    def __init__(
        self,
        *,
        session_id: int,
        workspace_root: str,
        emit: Callable[[dict[str, Any]], None],
        budgets: RuntimeBudgets | None = None,
    ):
        self.session_id = session_id
        self.workspace_root = workspace_root
        self.emit = emit
        self.budgets = budgets or RuntimeBudgets()
        self.run: OrchestrationRun | None = None
        # A session can execute several independent tasks.  The durable
        # safety mutation history is intentionally broader than one task, but
        # the user-facing task ledger/recap must not present an older task's
        # files as changes from the current one.
        self._ledger_is_new_task = False
        self._ledger_initialized = False
        try:
            self._baseline_mutation_ids = {
                str(item.get("mutation_id"))
                for item in local_state.get_session_state(session_id).modified_files
                if item.get("mutation_id")
            }
        except Exception:
            self._baseline_mutation_ids = set()

    def transition(self, phase: AgentPhase, *, action: str = "") -> None:
        if self.run is None:
            raise RuntimeError("orchestration run has not started")
        if isinstance(phase, str):
            phase = AgentPhase(phase)
        self.run.phase = phase
        local_state.save_session_state(
            self.session_id, current_phase=phase.value,
            execution_status="failed" if phase == AgentPhase.FAILED else (
                "completed" if phase == AgentPhase.COMPLETED else "running"
            ),
            owner_pid=None if phase in {AgentPhase.FAILED, AgentPhase.COMPLETED} else os.getpid(),
            running_action={"purpose": action or phase.value, "phase": phase.value},
        )
        self.emit({"event_type": f"orchestrator_{phase.value}", "payload": {"phase": phase.value, "action": action}})
        local_state.task_checkpoint(
            self.session_id, reason=f"phase_{phase.value}", next_action=action,
            phase=phase.value, current_step=action,
        )

    def begin(
        self, *, objective: str, messages: list[dict[str, Any]], read_only: bool,
        restore: Optional["ResumeSnapshot"] = None,
    ) -> OrchestrationRun:
        """Start a turn.

        With `restore` (a ResumeSnapshot of an interrupted task) this CONTINUES
        that task: the saved plan keeps every step's status, the durable task
        record (files read, decisions, commands, failures...) is kept, and the
        ledger's next_action says where the task stands. Without it this is a
        new task and the record is reset, as before.

        Live-reported 2026-09-19: begin() used to reset all of that on EVERY
        turn, resume included, so a task interrupted at 3 of 4 steps came back
        as 0 of 4 and had to be re-evaluated from scratch, again and again.
        """
        # A recovery prompt is transport/control data, not a replacement for
        # the user's task.  Older resume paths passed strings such as
        # "Continue from the saved checkpoint and resolve: ..." here and then
        # persisted that wrapper as active_task.objective.  Subsequent resume
        # selection consequently compared the wrapper against the saved plan,
        # created a second plan, and displayed progress for the wrong task.
        # The saved plan objective is authoritative whenever a valid snapshot
        # is present; the current messages still carry any explicit follow-up
        # requirements to the model.
        canonical_objective = (
            str(getattr(restore, "objective", "") or objective).strip()
            if restore is not None and str(getattr(restore, "objective", "") or "").strip()
            else objective
        )
        profile = classify_task(canonical_objective, read_only=read_only)
        self.run = OrchestrationRun(self.session_id, canonical_objective, profile, runtime=ExecutionController(self.budgets))
        self.run.delivery_policy = build_delivery_policy(
            read_only=read_only,
            requires_validation=profile.requires_validation,
            complexity=str(profile.complexity),
        )
        restored = restore is not None and restore.resume_index is not None
        local_state.save_session_state(
            self.session_id,
            active_task={
                "objective": canonical_objective,
                "task_type": getattr(profile.task_type, "value", profile.task_type),
                "complexity": getattr(profile.complexity, "value", profile.complexity),
            },
            current_phase=AgentPhase.UNDERSTAND.value, execution_status="running",
            owner_pid=os.getpid(),
        )
        if restored:
            names = [step["name"] for step in restore.steps]
            # The plan can be picked up from ANOTHER session (`resume` selects
            # the newest interrupted one for this workspace); its durable task
            # record must come with it, or only the plan would survive.
            carried = (
                {key: restore.task_state[key] for key in _TASK_RECORD_KEYS if key in restore.task_state}
                if restore.session_id != self.session_id else {}
            )
            local_state.update_task_state(
                self.session_id, task_id=str(self.session_id), objective=canonical_objective,
                status="running", phase=AgentPhase.UNDERSTAND.value,
                delivery_policy=self.run.delivery_policy.to_dict(),
                **carried,
                plan=names,
                completed_steps=[s["name"] for s in restore.steps if s["status"] == "completed"],
                pending_steps=[s["name"] for s in restore.steps if s["status"] != "completed"],
                next_action=restore.next_action(),
            )
        else:
            local_state.update_task_state(
                self.session_id, task_id=str(self.session_id), objective=objective,
                status="running", phase=AgentPhase.UNDERSTAND.value,
                delivery_policy=self.run.delivery_policy.to_dict(),
                plan=[], completed_steps=[], pending_steps=[], blocked_steps=[],
                assumptions=[], decisions=[], files_read=[], files_modified=[],
                artifacts_created=[], commands_run=[], tests=[], failures=[],
                retries=[], completion_evidence=[], next_action="Classify the request",
            )
        self.transition(AgentPhase.UNDERSTAND, action="Classify the request deterministically")
        if profile.requires_repository_context:
            self.transition(AgentPhase.INSPECT, action="Load or refresh repository context")
        if restored:
            self.run.plan = self._plan_from_snapshot(canonical_objective, profile, restore)
            self.run.plan_restored = True
            # Restore durable tool evidence before the next provider request.
            # Without this, a failover/resume could retain the plan while
            # losing the successful write/read/validation records that prove
            # the plan's completed work.  In particular, `compileall -q` is
            # intentionally silent; exit_code=0 is the evidence.
            for raw in restore.tool_records:
                try:
                    fields = {
                        key: raw.get(key)
                        for key in ToolEnvelope.__dataclass_fields__
                        if key in raw
                    }
                    fields.setdefault("tool_call_id", f"restored_{len(self.run.tool_records)}")
                    fields.setdefault("tool_name", "unknown")
                    fields.setdefault("arguments", {})
                    fields.setdefault("purpose", "restored tool evidence")
                    fields["arguments"] = fields["arguments"] if isinstance(fields["arguments"], dict) else {}
                    self.run.tool_records.append(ToolEnvelope(**fields))
                except (TypeError, ValueError):
                    # A malformed legacy record must not crash resume or be
                    # treated as evidence.  The next live tool result can
                    # still rebuild the current checkpoint.
                    continue
        else:
            self.run.plan = create_plan(objective, profile)
        self.run.runtime.start_planning()
        plan_dict = self.run.plan.to_dict() if self.run.plan else None
        self.run.context = build_context_bundle(
            session_id=self.session_id, workspace_root=self.workspace_root,
            objective=objective, profile=profile, conversation_messages=messages, plan=plan_dict,
        )
        if self.run.plan:
            self.transition(AgentPhase.PLAN, action="Restore the saved plan" if restored else "Persist an executable plan")
            if restored and restore.session_id == self.session_id and restore.plan_id:
                # Same session: keep the plan that is already persisted (and its
                # id) instead of saving a second copy with every step reset.
                self.run.plan_id = restore.plan_id
                local_state.save_session_state(self.session_id, active_plan_id=restore.plan_id)
            else:
                saved = local_state.save_plan(
                    self.session_id, objective=objective,
                    content="\n".join(f"{s.index}. {s.name}" for s in self.run.plan.steps),
                    steps=self._plan_items(self.run.plan),
                )
                self.run.plan_id = saved.id
        # Re-anchor the first epoch's clock now that one-time task setup
        # (state persistence, planning, context building) is actually done,
        # so that setup latency is never silently deducted from the first
        # epoch's own execution budget.
        self.run.runtime.begin_epoch_clock()
        try:
            self.save_task_ledger(
                status="running",
                next_action=restore.next_action() if restored else "begin executing the plan",
            )
        except Exception:
            pass
        return self.run

    @staticmethod
    def _plan_items(plan: ExecutionPlan) -> list[dict[str, Any]]:
        """A plan's steps as persisted: name, status, and the evidence/phase a
        resume needs to rebuild the same plan."""
        return [
            {
                "index": s.index, "step": s.name, "status": s.status,
                "evidence": list(s.evidence[-5:]), "phase": s.phase,
            }
            for s in plan.steps
        ]

    def _plan_from_snapshot(
        self, objective: str, profile: TaskProfile, restore: "ResumeSnapshot",
    ) -> ExecutionPlan:
        """Rebuild the interrupted plan: its own steps and statuses, with the
        non-step parts (assumptions, risks, validation criteria) from what the
        ledger kept, falling back to the template's for a legacy record."""
        template = create_plan(objective, profile)

        # ``create_plan`` legitimately returns ``None`` for a plain
        # conversation.  A saved plan can still reach this path when a
        # provider/failover resume carries an older interrupted plan whose
        # current objective now classifies as conversation.  Never let a
        # missing template turn resume into ``NoneType has no attribute
        # assumptions`` (or the equivalent for another static field).
        static = restore.static if isinstance(restore.static, dict) else {}

        def restored_list(key: str) -> list[Any]:
            value = static.get(key)
            if isinstance(value, (list, tuple)) and value:
                return list(value)
            fallback = getattr(template, key, []) if template is not None else []
            return list(fallback or [])

        return ExecutionPlan(
            objective=objective,
            assumptions=restored_list("assumptions"),
            components=restored_list("components"),
            steps=[
                PlanStep(
                    index=position, name=step["name"], status=step["status"],
                    evidence=list(step.get("evidence") or []), phase=int(step.get("phase") or 0),
                )
                for position, step in enumerate(restore.steps, start=1)
            ],
            validation_criteria=restored_list("validation_criteria"),
            risks=restored_list("risks"),
            phase_names=restored_list("phase_names"),
        )

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
        plan.deduplicate_steps()
        if not self.run.runtime.record_plan_revision():
            # Same reasoning as mark_repair's extension below: a genuinely
            # evolving task can legitimately need more than
            # max_plan_revisions replans as it learns more about the real
            # codebase. Grant a fresh window instead of ending the task on
            # this accounting ceiling.
            if self.run.runtime.extend_plan_revision_budget():
                extensions = self.run.runtime.snapshot.plan_revision_extensions
                limit = self.run.runtime.budgets.max_plan_revision_extensions
                self.emit({
                    "event_type": "diagnostics",
                    "payload": {
                        "content": (
                            f"Plan revision budget reached -- granting another "
                            f"{self.run.runtime.budgets.max_plan_revisions} revisions "
                            f"(extension {extensions}/{limit}) instead of ending the task."
                        ),
                    },
                })
                self.run.runtime.record_plan_revision()
            else:
                self.fail(self.run.runtime.snapshot.failure_reason)
                return
        saved = local_state.save_plan(
            self.session_id, objective=self.run.objective,
            content="\n".join(f"{s.index}. {s.name}" for s in plan.steps),
            steps=self._plan_items(plan),
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
        items = [
            {"step": s.name, "status": s.status, "evidence": list(s.evidence[-5:]), "phase": s.phase}
            for s in self.run.plan.steps
        ]
        local_state.update_plan_steps(self.session_id, self.run.plan_id, items)
        # Deliberately a distinct event type from "plan_created" -- that
        # event means "a new/revised plan now exists" (renderer reprints
        # the plan banner and resets the spinner phase to "plan" on it);
        # this only means "the existing plan's step statuses changed",
        # which should update the live step markers in place with none of
        # that -- no banner reprint, no spinner phase change, every round.
        self.emit({"event_type": "plan_step_progress", "payload": {"items": items}})

    def edit_plan_step(self, index: int, *, name: str | None = None, status: str | None = None) -> None:
        assert self.run is not None and self.run.plan is not None
        self.run.plan.edit_step(index, name=name, status=status)
        self._sync_plan_progress()

    def add_plan_step(self, name: str, *, after: int | None = None) -> None:
        assert self.run is not None and self.run.plan is not None
        self.run.plan.add_step(name, after=after)
        self._sync_plan_progress()

    def remove_plan_step(self, index: int) -> None:
        assert self.run is not None and self.run.plan is not None
        self.run.plan.remove_step(index)
        self._sync_plan_progress()

    def record_route(
        self,
        *,
        provider: str,
        model: str,
        reason: str,
        fallback_chain: list[str] | None = None,
        requested_provider: str | None = None,
        requested_model: str | None = None,
        fallback_reason: str | None = None,
    ) -> None:
        assert self.run is not None
        self.transition(AgentPhase.ROUTE, action="Select a capability-matched provider and model")
        prior = self.run.route
        immutable_requested_provider = prior.get("requested_provider") or requested_provider or provider
        immutable_requested_model = prior.get("requested_model") or requested_model or model
        self.run.route = {
            # Compatibility names retained for renderers and older state.
            "provider": provider,
            "model": model,
            "requested_provider": immutable_requested_provider,
            "requested_model": immutable_requested_model,
            "effective_provider": provider,
            "effective_model": model,
            "fallback_reason": fallback_reason or (
                reason if prior and (
                    provider != immutable_requested_provider
                    or model != immutable_requested_model
                ) else ""
            ),
            "reason": reason,
            "fallback_chain": fallback_chain or [],
        }
        # FIX: this used to also call
        # local_state.save_session_state(self.session_id, selected_provider=
        # provider, selected_model=model) on every single round -- including
        # every automatic AUTO-mode resolution and every mid-turn fallback,
        # not just an explicit `/model` selection. selected_provider/
        # selected_model is the field `/model <provider> <model>` sets to
        # record the user's deliberate override (read back by, e.g.,
        # interactive.py's `/btw` side-question routing and the remote-mode
        # turn dispatch); overwriting it here on every round made a plain
        # AUTO session that never touched `/model` silently look, to any
        # later reader of that field, exactly like the user had explicitly
        # pinned whatever provider AUTO most recently happened to resolve
        # to (deterministically NVIDIA most of the time -- see providers.py).
        # This run's actual requested-vs-effective route is already tracked
        # correctly and separately in `self.run.route` above; nothing needs
        # a second, conflated copy in the explicit-preference field.

        # Route provenance for /status and the persistent footer. record_route
        # is the single authoritative place a route changes -- initial
        # selection, automatic failover, and mid-turn recovery all come through
        # here -- so recording it once here means the user can see WHICH route
        # a task is actually running on and whether it failed over at all,
        # instead of that being visible only as a debug diagnostic scrolling
        # past. Never allowed to raise: route bookkeeping must not be able to
        # break a task.
        try:
            from ..state import record_route_event

            previous_provider = str(
                prior.get("effective_provider") or prior.get("provider") or ""
            )
            previous_model = str(prior.get("effective_model") or prior.get("model") or "")
            changed = bool(previous_provider) and previous_provider != provider
            fallback_reason = str(self.run.route.get("fallback_reason") or "")
            if changed or fallback_reason:
                record_route_event(
                    self.session_id,
                    provider=provider,
                    model=model,
                    previous_provider=previous_provider,
                    previous_model=previous_model,
                    reason=fallback_reason or reason,
                    kind="failover" if fallback_reason else "select",
                )
            elif not prior:
                record_route_event(
                    self.session_id, provider=provider, model=model, reason=reason,
                    kind="select",
                )
        except Exception:
            pass

    def start_execution(self) -> None:
        assert self.run is not None
        self.run.runtime.start_execution()
        self.transition(AgentPhase.EXECUTE, action="Execute the model/tool loop")

    def guard_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> GuardDecision:
        assert self.run is not None
        decision = self.run.runtime.guard_action(tool_name, arguments)
        if not decision.allowed and decision.time_budget_exhausted:
            # Running out of wall-clock time mid-task isn't the same kind of
            # failure as a stalled or looping agent -- it just means the
            # execution epoch expired. Checkpoint the current task state,
            # renew the epoch, and continue the same task (see
            # ExecutionController.renew_epoch). Bounded by
            # max_runtime_extensions before treating it as final.
            if self.run.runtime.epoch_renewal_due() or decision.epoch_renewal_requested:
                self._checkpoint_before_epoch_renewal()
            if self.run.runtime.renew_epoch():
                extensions = self.run.runtime.snapshot.runtime_extensions
                limit = self.run.runtime.budgets.max_runtime_extensions
                self.emit({
                    "event_type": "diagnostics",
                    "payload": {
                        "content": (
                            f"↻ Runtime epoch renewed; continuing current task "
                            f"(epoch {self.run.runtime.epoch_index}, renewal {extensions}/{limit})."
                        ),
                    },
                })
                decision = self.run.runtime.guard_action(tool_name, arguments)
        elif not decision.allowed and decision.tool_call_budget_exhausted:
            # Same reasoning as the wall-clock extension above, for the raw
            # tool-call count: a genuine stall is still caught independently
            # by the repeated-action/empty-observation guards, so a large
            # but genuinely productive audit shouldn't hard-fail here and
            # force the user to go edit config.toml before it can finish.
            if self.run.runtime.extend_tool_call_budget():
                extensions = self.run.runtime.snapshot.tool_call_extensions
                limit = self.run.runtime.budgets.max_tool_call_extensions
                self.emit({
                    "event_type": "diagnostics",
                    "payload": {
                        "content": (
                            f"Tool-call budget reached -- granting more headroom "
                            f"(extension {extensions}/{limit}) instead of ending the task."
                        ),
                    },
                })
                decision = self.run.runtime.guard_action(tool_name, arguments)
        if not decision.allowed:
            self.emit({"event_type": "diagnostics", "payload": {"content": decision.reason}})
        return decision

    def _checkpoint_before_epoch_renewal(self) -> None:
        """Atomically persist task state before closing an exhausted epoch.

        Uses the session's existing durable task ledger (state.task_checkpoint)
        so the checkpoint survives process interruption and provider switches,
        and records the next intended action so the resumed epoch (or a fresh
        process) can continue without rediscovery.
        """
        assert self.run is not None
        try:
            local_state.task_checkpoint(
                self.session_id,
                reason="epoch_renewal",
                next_action=self.run.objective[-500:],
                phase=self.run.phase.value,
                epoch_index=self.run.runtime.epoch_index,
                runtime_extensions=self.run.runtime.snapshot.runtime_extensions,
                tool_calls=self.run.runtime.snapshot.tool_calls,
                evidence_items=self.run.runtime.snapshot.evidence_items,
            )
        except Exception:
            # Checkpointing must never block the renewal itself; the live
            # runtime state remains the source of truth for this process.
            pass
        try:
            self.save_task_ledger(
                status="checkpointing",
                next_action=f"renew execution epoch {self.run.runtime.epoch_index + 1} and continue",
            )
        except Exception:
            pass

    def save_task_ledger(self, *, status: str, next_action: str) -> None:
        """Persist the structured TaskLedger (Layer A durable facts) at a
        checkpoint boundary -- the compaction-safe anchor thread compression
        rebuilds active context from (see runtime/ledger.py's module
        docstring). Reuses self.run's own plan/route/objective as the single
        source of truth rather than tracking a second, parallel copy of the
        same facts; never raises (checkpointing must not block execution).
        """
        assert self.run is not None
        from ..runtime.ledger import LedgerEdit, PlanStep as _LedgerPlanStep, TaskLedger, load_ledger, save_ledger

        task_id = str(self.session_id)
        existing_ledger = load_ledger(task_id)
        if self._ledger_initialized:
            ledger = existing_ledger or TaskLedger(
                task_id=task_id, session_id=self.session_id, objective=self.run.objective,
                repo_roots=[self.workspace_root], cwd=self.workspace_root,
            )
        elif existing_ledger is None:
            ledger = TaskLedger(
                task_id=task_id, session_id=self.session_id, objective=self.run.objective,
                repo_roots=[self.workspace_root], cwd=self.workspace_root,
            )
            self._ledger_is_new_task = True
            self._ledger_initialized = True
        elif not local_state.task_objectives_compatible(
            str(existing_ledger.objective or ""), self.run.objective,
        ):
            # Do not merge an unrelated task into the prior session ledger.
            # This is the source of misleading recaps such as a Finitron task
            # showing old registry/test files as if the current turn changed
            # them.  Resume/continuation objectives remain compatible and keep
            # their plan and evidence intact.
            ledger = TaskLedger(
                task_id=task_id, session_id=self.session_id, objective=self.run.objective,
                repo_roots=[self.workspace_root], cwd=self.workspace_root,
            )
            self._ledger_is_new_task = True
            self._ledger_initialized = True
        else:
            ledger = existing_ledger
            self._ledger_initialized = True
        ledger.objective = self.run.objective
        ledger.status = status
        if self.run.plan is not None:
            ledger.plan_steps = [
                _LedgerPlanStep(index=step.index, name=step.name, status=step.status)
                for step in self.run.plan.steps
            ]
            completed = sum(1 for s in ledger.plan_steps if s.status == "completed")
            ledger.current_step_index = min(completed, max(len(ledger.plan_steps) - 1, 0))
            ledger.plan_static = {
                "assumptions": list(self.run.plan.assumptions or []),
                "components": list(self.run.plan.components or []),
                "validation_criteria": list(self.run.plan.validation_criteria or []),
                "risks": list(self.run.plan.risks or []),
                "phase_names": list(self.run.plan.phase_names or []),
                "delivery_policy": (
                    self.run.delivery_policy.to_dict()
                    if self.run.delivery_policy is not None else {}
                ),
            }
        else:
            # A task with no plan has no plan steps. The ledger is keyed by
            # session, so without this a new one-line task inherited the PREVIOUS
            # task's "3/4 steps done" and /status reported progress that
            # belonged to different work.
            ledger.plan_steps = []
            ledger.current_step_index = 0
            ledger.plan_static = {}
            if self.run.delivery_policy is not None:
                ledger.plan_static = {"delivery_policy": self.run.delivery_policy.to_dict()}
        # Reuses the same modified_files list safety.py's record_mutation
        # already maintains per session -- a real recap needs to show real
        # changed files, not a second, separately-tracked copy that could
        # drift from what actually happened.
        try:
            modified = local_state.get_session_state(self.session_id).modified_files
        except Exception:
            modified = []
        ledger.edits = [
            LedgerEdit(
                file=str(item.get("path") or ""),
                operation=str(item.get("operation") or "update"),
                description=(
                    f"{item.get('operation', 'edit')} "
                    f"(+{item.get('lines_added', 0)}/-{item.get('lines_removed', 0)})"
                ),
                applied=True,
                mutation_id=str(item.get("mutation_id") or ""),
            )
            for item in modified
            if item.get("path") and not (
                self._ledger_is_new_task
                and item.get("mutation_id") in self._baseline_mutation_ids
            )
        ]
        if self.run.route:
            ledger.current_provider = str(self.run.route.get("provider") or ledger.current_provider)
            ledger.current_model = str(self.run.route.get("model") or ledger.current_model)
        ledger.current_action = f"epoch {self.run.runtime.epoch_index} ({self.run.phase.value})"
        ledger.next_action = next_action
        ledger.checkpoint_version += 1
        save_ledger(ledger)

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
        ledger = local_state.get_session_state(self.session_id).task_state or {}
        failures = list(ledger.get("failures") or [])
        retries = list(ledger.get("retries") or [])
        if not envelope.success:
            failure = {
                "category": classify_failure(envelope.stderr, tool_name=envelope.tool_name),
                "action": envelope.tool_name,
                "error": envelope.stderr[-1200:],
                "at": envelope.completed_at,
            }
            failures.append(failure)
            retries.append({"retry_number": len(retries) + 1, "failure": failure, "disposition": "pending_diagnosis"})
        local_state.update_task_state(
            self.session_id, phase=self.run.phase.value,
            commands_run=list(ledger.get("commands_run") or []) + ([envelope.arguments.get("command")] if envelope.tool_name == "execute_command" and envelope.arguments.get("command") else []),
            failures=failures, retries=retries,
            files_modified=list(ledger.get("files_modified") or []) + list(envelope.files_changed),
        )
        self._advance_plan_step(decision, envelope)
        if decision.terminal:
            self.fail(decision.reason)
        return decision

    @staticmethod
    def _tool_matches_plan_step(tool: ToolEnvelope, step_name: str) -> bool:
        """Return whether a useful tool result is evidence for this step.

        Sequential observations do not necessarily map one-to-one to plan
        steps. Keep unmatched evidence in the ledger without letting it claim
        an unrelated milestone.
        """
        name = tool.tool_name.casefold()
        step = step_name.casefold()
        # Keep real extensions (``status.json``) together without treating
        # sentence-ending punctuation (``components.``) as a filename.
        words = set(re.findall(r"[a-z0-9_]+(?:\.[a-z0-9_]+)*", step))
        target = ""
        for key in ("path", "directory", "destination", "output_path", "query", "symbol"):
            value = tool.arguments.get(key)
            if isinstance(value, str) and value.strip():
                target = value.strip().casefold()
                break
        target_name = PurePath(target).name if target else ""
        target_words = set(re.findall(r"[a-z0-9_-]+", target_name))
        target_matches = bool(
            target and (
                target in step
                or (target_name and target_name in words)
                or (target_name and target_name in step)
                or any(word in words for word in target_words if len(word) >= 3)
            )
        )
        names_a_file_kind = any("." in word for word in words) or bool(
            words & {"json", "yaml", "yml", "toml", "markdown", "md", "txt", "log", "pdf", "docx", "file"}
        )

        if name in {"read_file", "inspect_artifact"}:
            return target_matches or (
                bool(words & {"read", "review", "inspect", "examine"}) and not names_a_file_kind
            )
        if name == "list_directory":
            return bool(words & {"list", "inventory", "directory", "folder", "contents", "locate", "discover"}) and (
                not target or target_matches or not names_a_file_kind
            )
        if name in {"search_code", "find_references"}:
            return bool(words & {"search", "find", "locate", "identify", "references", "usages"})
        if name == "get_git_info":
            return bool(words & {"git", "branch", "commit", "repository", "status"})
        if name in {"write_file", "create_file", "create_artifact"}:
            return target_matches or bool(words & {"write", "create", "add", "generate"})
        if name in {"edit_file", "patch_file"}:
            return target_matches or bool(words & {"edit", "change", "modify", "fix", "patch", "update", "refactor"})
        if name in {"extract_archive", "repackage_archive"}:
            return target_matches or bool(words & {"extract", "archive", "package", "repackage"})
        if name == "execute_command":
            return bool(words & {"run", "execute", "test", "verify", "validate", "check", "build", "lint", "typecheck", "install"})
        if name == "ask_user_question":
            return bool(words & {"ask", "clarify", "confirm"})
        return False

    def _advance_plan_step(self, decision: ObservationDecision, tool: ToolEnvelope) -> None:
        """Advance only when useful evidence corresponds to the active step."""
        assert self.run is not None
        if self.run.plan is None or not self.run.plan.steps:
            return
        changed = False
        active = next((step for step in self.run.plan.steps if step.status == "in_progress"), None)
        if active is None:
            active = next((step for step in self.run.plan.steps if step.status == "pending"), None)
            if active is not None:
                active.status = "in_progress"
                changed = True
        if active is not None and decision.useful and self._tool_matches_plan_step(tool, active.name):
            active.status = "completed"
            active.evidence.extend(item for item in decision.evidence if item not in active.evidence)
            changed = True
            nxt = next((step for step in self.run.plan.steps if step.status == "pending"), None)
            if nxt is not None:
                nxt.status = "in_progress"
        if changed:
            self._sync_plan_progress()

    def mark_repair(self, reason: str, *, provider_switch: bool = False) -> None:
        assert self.run is not None
        self.run.repair_attempts += 1
        if not self.run.runtime.record_repair():
            # The shared repair counter ran out -- not necessarily because
            # THIS repair is unproductive, possibly because unrelated infra
            # recovery earlier in the turn (provider fallback, empty-
            # continuation recovery, ...) already spent most of it. Grant a
            # fresh window instead of failing the whole task on that
            # accounting artifact, bounded by max_repair_extensions.
            if self.run.runtime.extend_repair_budget():
                extensions = self.run.runtime.snapshot.repair_extensions
                limit = self.run.runtime.budgets.max_repair_extensions
                self.emit({
                    "event_type": "diagnostics",
                    "payload": {
                        "content": (
                            f"Repair budget reached -- granting another "
                            f"{self.run.runtime.budgets.max_repair_rounds} attempts "
                            f"(extension {extensions}/{limit}) instead of ending the task."
                        ),
                    },
                })
                self.run.runtime.record_repair()
            else:
                self.fail(self.run.runtime.snapshot.failure_reason)
                return
        # A successful switch to a genuinely untested provider is progress,
        # not a repeat of whatever the previous provider kept getting wrong
        # -- it hasn't had a single attempt charged against it yet. Without
        # this, a run configured with many fallback candidates could still
        # exhaust the shared repair counter (and its limited extensions)
        # purely from cycling through providers, before any of the later,
        # untried ones ever got a real shot. Reset the round counter (not
        # the capped extensions counter) so each newly-adopted provider
        # starts its own attempts from zero, same as the very first
        # provider of the run did.
        if provider_switch:
            self.run.runtime.snapshot.repair_rounds = 0
        self.transition(AgentPhase.REPAIR, action=reason)

    def validate(self, *, final_text: str, any_mutation: bool, read_only: bool = False) -> ValidationReport:
        assert self.run is not None
        self.run.runtime.begin_validation()
        self.transition(AgentPhase.VALIDATE, action="Validate evidence and completion claims")
        try:
            project_instructions = load_instruction_text(Path(self.workspace_root))
        except Exception:
            # Best-effort only -- a workspace with no instructions, an
            # unreadable file, or a non-git directory must never block
            # validation itself; it just means the deploy-recorded check
            # below has nothing to enforce this turn.
            project_instructions = ""
        report = validate_completion(
            profile=self.run.profile,
            tool_records=[item.to_dict() for item in self.run.tool_records],
            any_mutation=any_mutation, final_text=final_text,
            objective=self.run.objective, workspace_root=self.workspace_root,
            project_instructions=project_instructions, read_only=read_only,
        )
        if (
            self.run.reasoning_plan
            and getattr(self.run.profile.task_type, "value", self.run.profile.task_type) == "audit"
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
        ledger = local_state.get_session_state(self.session_id).task_state or {}
        evidence = list(ledger.get("completion_evidence") or [])
        if report.passed:
            evidence.append("completion validation passed")
        local_state.update_task_state(
            self.session_id, phase=AgentPhase.VALIDATE.value,
            status="running" if report.severity != "error" else "failed",
            tests=list(ledger.get("tests") or []) + [report.to_dict()],
            completion_evidence=evidence,
            unresolved_issues=report.unresolved,
        )
        return report

    def complete(self, *, final_text: str, any_mutation: bool) -> ValidationReport:
        report = self.validate(final_text=final_text, any_mutation=any_mutation)
        if self.run is not None and self.run.delivery_policy is not None:
            observed_evidence = sum(1 for record in self.run.tool_records if record.success)
            if not completion_is_evidence_bound(
                self.run.delivery_policy,
                evidence_count=observed_evidence,
                passed=report.passed,
            ):
                report.passed = False
                report.severity = "error"
                report.unresolved.append(
                    "Completion requires at least one observed verification result."
                )
                report.checks.append({
                    "name": "evidence_bound_completion",
                    "passed": False,
                    "detail": "No successful tool/verification evidence was observed.",
                })
        if self.run is not None and self.run.plan is not None:
            for step in self.run.plan.steps:
                step.status = "completed" if report.passed else (
                    "failed" if step.status == "in_progress" else step.status
                )
            self._sync_plan_progress()
        self.transition(AgentPhase.REPORT, action="Report only evidence-supported outcomes")
        # `passed` is the completion authority.  Severity describes how the
        # caller should present the failure; a warning still contains an
        # unresolved acceptance criterion (for example a pending plan step)
        # and must never become a successful task merely because it is not an
        # error-level finding.
        if not report.passed:
            self.run.runtime.fail("Completion validation failed.")
        else:
            self.run.runtime.complete()
        terminal_phase = AgentPhase.COMPLETED if report.passed else AgentPhase.FAILED
        terminal_status = "completed" if report.passed else "failed"
        self.transition(terminal_phase)
        local_state.checkpoint(
            self.session_id,
            reason="orchestrator_complete" if report.passed else "orchestrator_validation_failed",
            summary=final_text[-1000:],
        )
        local_state.task_checkpoint(
            self.session_id,
            reason="delivery_ready" if report.passed else "completion_validation_failed",
            next_action="none" if report.passed else "resume and satisfy unresolved acceptance criteria",
            phase=terminal_phase.value,
            status=terminal_status,
            completion_evidence=(
                list((local_state.get_session_state(self.session_id).task_state or {}).get("completion_evidence") or [])
                + (["final delivery emitted"] if report.passed else [])
            ),
        )
        try:
            self.save_task_ledger(
                status=terminal_status,
                next_action="none" if report.passed else "resume and satisfy unresolved acceptance criteria",
            )
        except Exception:
            pass
        return report

    _USER_STOP_RE = re.compile(
        r"\b(?:cancel+ed|stopped|interrupted|paused)\b[^.\n]{0,40}\b(?:by\s+(?:the\s+)?user|user\s+request)\b|"
        r"\buser[- ](?:cancel+ed|interrupted|stopped)\b|\bexecution\s+cancel+ed\b",
        re.IGNORECASE,
    )

    def fail(self, error: str) -> None:
        # A user's own Esc/Ctrl+C is not a failure of the step it interrupted. Marking that step "failed"
        # made the next-message box offer "Repair the failed plan step ..." and later turns treat the cut-off
        # step as broken work to fix (owner report 2026-09-21); it is simply not finished yet.
        user_stop = bool(self._USER_STOP_RE.search(error or ""))
        if self.run is not None:
            self.run.runtime.fail(error)
            if self.run.plan is not None:
                for step in self.run.plan.steps:
                    if step.status == "in_progress":
                        step.status = "pending" if user_stop else "failed"
                self._sync_plan_progress()
            self.transition(AgentPhase.FAILED, action=error)
        local_state.checkpoint(self.session_id, reason="orchestrator_failed", summary=error[-1000:])
        prior_failures = list((local_state.get_session_state(self.session_id).task_state or {}).get("failures") or [])
        local_state.task_checkpoint(
            self.session_id, reason="task_stopped" if user_stop else "task_failed",
            next_action="resume where it stopped" if user_stop else "diagnose and resume",
            phase=AgentPhase.FAILED.value, status="interrupted" if user_stop else "failed",
            failures=prior_failures if user_stop else prior_failures + [{"category": classify_failure(error), "error": error[-1200:]}],
        )
        try:
            self.save_task_ledger(status="failed", next_action="diagnose and resume")
        except Exception:
            pass
