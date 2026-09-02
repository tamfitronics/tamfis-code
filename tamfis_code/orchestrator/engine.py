"""Persistent Claude Code/Codex-style orchestration state machine."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePath
import re
from typing import Any, Callable

from .. import state as local_state
from ..routing import TaskProfile, classify_task
from .context import ContextBundle, build_context_bundle
from .planner import ExecutionPlan, create_plan
from .protocols import AgentPhase, ToolEnvelope, classify_failure
from .validator import ValidationReport, validate_completion
from ..runtime import ExecutionController, GuardDecision, ObservationDecision
from ..runtime.budgets import RuntimeBudgets


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
        local_state.task_checkpoint(
            self.session_id, reason=f"phase_{phase.value}", next_action=action,
            phase=phase.value, current_step=action,
        )

    def begin(self, *, objective: str, messages: list[dict[str, Any]], read_only: bool) -> OrchestrationRun:
        profile = classify_task(objective, read_only=read_only)
        self.run = OrchestrationRun(self.session_id, objective, profile, runtime=ExecutionController(self.budgets))
        local_state.save_session_state(
            self.session_id,
            active_task={"objective": objective, "task_type": profile.task_type.value, "complexity": profile.complexity},
            current_phase=AgentPhase.UNDERSTAND.value, execution_status="running",
        )
        local_state.update_task_state(
            self.session_id, task_id=str(self.session_id), objective=objective,
            status="running", phase=AgentPhase.UNDERSTAND.value,
            plan=[], completed_steps=[], pending_steps=[], blocked_steps=[],
            assumptions=[], decisions=[], files_read=[], files_modified=[],
            artifacts_created=[], commands_run=[], tests=[], failures=[],
            retries=[], completion_evidence=[], next_action="Classify the request",
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
        if not decision.allowed and decision.time_budget_exhausted:
            # Running out of wall-clock time mid-task isn't the same kind of
            # failure as a stalled or looping agent -- it just means the
            # turn needs to keep going. Grant a fresh budget and re-check
            # (which still applies every other guard -- tool-call ceiling,
            # repeated-action detection -- against the unmodified state) up
            # to max_runtime_extensions before treating it as final.
            if self.run.runtime.extend_runtime():
                extensions = self.run.runtime.snapshot.runtime_extensions
                limit = self.run.runtime.budgets.max_runtime_extensions
                self.emit({
                    "event_type": "diagnostics",
                    "payload": {
                        "content": (
                            f"Turn budget reached -- starting turn {extensions + 1} "
                            f"(extension {extensions}/{limit}) instead of ending the task."
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

    def validate(self, *, final_text: str, any_mutation: bool) -> ValidationReport:
        assert self.run is not None
        self.run.runtime.begin_validation()
        self.transition(AgentPhase.VALIDATE, action="Validate evidence and completion claims")
        report = validate_completion(
            profile=self.run.profile,
            tool_records=[item.to_dict() for item in self.run.tool_records],
            any_mutation=any_mutation, final_text=final_text,
            objective=self.run.objective, workspace_root=self.workspace_root,
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
        local_state.task_checkpoint(
            self.session_id, reason="delivery_ready", next_action="none",
            phase=AgentPhase.COMPLETED.value, status="completed",
            completion_evidence=list((local_state.get_session_state(self.session_id).task_state or {}).get("completion_evidence") or []) + ["final delivery emitted"],
        )
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
        local_state.task_checkpoint(
            self.session_id, reason="task_failed", next_action="diagnose and resume",
            phase=AgentPhase.FAILED.value, status="failed",
            failures=list((local_state.get_session_state(self.session_id).task_state or {}).get("failures") or []) + [{"category": classify_failure(error), "error": error[-1200:]}],
        )
