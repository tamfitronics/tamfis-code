"""What a `continue` RESTORES instead of re-deriving.

Live-reported 2026-09-19: tamfis-code did not keep accurate state for resuming
exactly where it left off -- after an interruption it "re-evaluates again and
again". Cause (reproduced): AgentOrchestrator.begin() runs at the start of EVERY
turn, and it reset the durable task record (files read, completed steps,
decisions...), built a fresh all-pending template plan, saved it as a second
plan, and wrote the ledger from that -- so a task interrupted at 3 of 4 steps
came back as 0 of 4 and /status said nothing was done. The real progress was
sitting in the saved plan (step statuses are persisted as the turn advances) and
in the ledger; the resume turn simply threw it away.

A ResumeSnapshot is the plan-and-progress state to carry across that boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# next_action values that carry no information about WHERE the task stands.
_UNINFORMATIVE_NEXT_ACTIONS = frozenset({
    "", "begin executing the plan", "classify the request",
})


@dataclass
class ResumeSnapshot:
    """The plan, its per-step progress, and the durable task record of a task
    that was interrupted before all its steps were done."""

    session_id: int
    plan_id: str
    objective: str
    steps: list[dict[str, Any]]  # {"name", "status", "evidence", "phase"}
    static: dict[str, Any] = field(default_factory=dict)  # assumptions/components/validation_criteria/risks/phase_names
    task_state: dict[str, Any] = field(default_factory=dict)
    ledger_next_action: str = ""
    tool_records: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.steps)

    @property
    def done(self) -> int:
        return sum(1 for step in self.steps if step.get("status") == "completed")

    @property
    def resume_index(self) -> Optional[int]:
        """0-based index of the first step that is not completed."""
        for index, step in enumerate(self.steps):
            if step.get("status") != "completed":
                return index
        return None

    @property
    def resume_step_name(self) -> str:
        index = self.resume_index
        return str(self.steps[index].get("name") or "") if index is not None else ""

    def next_action(self) -> str:
        index = self.resume_index
        if index is None:
            return "Validate and report"
        return f"Resume at step {index + 1}/{self.total}: {self.resume_step_name}"

    def banner(self) -> str:
        """The one line shown when a resume starts. "◆" marks it as always
        shown, not gated behind --debug."""
        index = self.resume_index
        if index is None:
            return "◆ Resuming: every plan step is done; validating and reporting."
        return (
            f"◆ Resuming at step {index + 1}/{self.total}: {self.resume_step_name} "
            f"({self.done} done) -- continuing the saved plan instead of re-planning."
        )


def _normalise_steps(raw_steps: list[Any]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for item in raw_steps or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("step") or item.get("name") or "").strip()
        if not name:
            continue
        steps.append({
            "name": name,
            "status": str(item.get("status") or "pending"),
            "evidence": [str(e) for e in (item.get("evidence") or [])][:5],
            "phase": int(item.get("phase") or 0),
        })
    return steps


def _record_identity(record: dict[str, Any]) -> tuple[Any, ...]:
    """Return a stable identity for a checkpoint/completed-action record.

    Provider retries can write the same result more than once.  Prefer the
    provider/tool id, but retain a deterministic fallback for older state
    files that predate that field.
    """
    return (
        record.get("tool_call_id")
        or record.get("mutation_id")
        or record.get("id")
        or (
            record.get("tool_name"),
            record.get("completed_at"),
            (record.get("arguments") or {}).get("path"),
            (record.get("arguments") or {}).get("command"),
        )
    )


def _same_objective_record(record: dict[str, Any], objective: str) -> bool:
    """Whether a durable action was recorded for this resumed objective.

    Tool records historically stored the objective in ``purpose`` as a
    bounded prefix (``Execute ... for: <objective[:160]>``).  Matching that
    prefix lets us recover evidence across a provider failover while keeping
    records from unrelated plans out of the resumed turn.
    """
    if not objective:
        return False
    purpose = str(record.get("purpose") or "")
    stored_objective = str(record.get("objective") or "")
    needle = " ".join(objective.split()).casefold()
    if not needle:
        return False
    for candidate in (purpose, stored_objective):
        haystack = " ".join(candidate.split()).casefold()
        if haystack and (needle[:160] in haystack or needle[:96] in haystack):
            return True
    return False


def _resume_tool_records(
    checkpoint_records: list[Any],
    completed_actions: list[Any],
    objective: str,
) -> list[dict[str, Any]]:
    """Merge checkpoint records with verified same-task durable history.

    A checkpoint is deliberately rewritten during failover.  It can therefore
    contain only the latest route's failed call even though earlier routes
    successfully read or mutated files.  ``completed_actions`` is the durable
    evidence ledger; restore its same-objective successful tool records and
    deduplicate them by tool-call identity.
    """
    merged: list[dict[str, Any]] = [
        item for item in checkpoint_records if isinstance(item, dict)
    ]
    for item in completed_actions:
        if not isinstance(item, dict) or item.get("type") != "tool":
            continue
        if item.get("success") is not True or not _same_objective_record(item, objective):
            continue
        merged.append(item)

    result: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for item in merged:
        identity = _record_identity(item)
        if identity in seen:
            continue
        seen.add(identity)
        result.append(item)
    return result


def load_resume_snapshot(session_id: int) -> Optional[ResumeSnapshot]:
    """The resumable state of `session_id`'s latest plan, or None when there is
    nothing to resume (no plan, or every step already completed)."""
    from .. import state as local_state
    from .ledger import load_ledger

    try:
        state = local_state.get_session_state(session_id)
    except Exception:
        return None
    plans = list(state.saved_plans or [])
    if not plans:
        return None
    plan = next(
        (item for item in reversed(plans) if item.get("id") == state.active_plan_id),
        plans[-1],
    ) or {}
    steps = _normalise_steps(plan.get("steps") or [])
    if not steps or all(step["status"] == "completed" for step in steps):
        return None
    try:
        ledger = load_ledger(str(session_id))
    except Exception:
        ledger = None
    static = dict(getattr(ledger, "plan_static", None) or {})
    ledger_next = str(getattr(ledger, "next_action", "") or "")
    if ledger_next.strip().casefold() in _UNINFORMATIVE_NEXT_ACTIONS:
        ledger_next = ""
    checkpoint = state.turn_checkpoint if isinstance(state.turn_checkpoint, dict) else {}
    checkpoint_records = checkpoint.get("tool_records") or []
    # A later failover checkpoint may contain only its own failed/partial
    # records.  Merge the durable, objective-matched ledger so validation can
    # see successful work from earlier routes without importing unrelated
    # session history.
    objective = str(
        checkpoint.get("objective")
        or plan.get("objective")
        or getattr(state, "active_task", None) and state.active_task.get("objective")
        or ""
    )
    checkpoint_records = _resume_tool_records(
        checkpoint_records,
        state.completed_actions or [],
        objective,
    )
    return ResumeSnapshot(
        session_id=session_id,
        plan_id=str(plan.get("id") or ""),
        objective=objective or str(plan.get("objective") or ""),
        steps=steps,
        static=static,
        task_state=dict(state.task_state or {}),
        ledger_next_action=ledger_next,
        tool_records=[item for item in checkpoint_records if isinstance(item, dict)],
    )


def describe_resume_point(session_id: int) -> Optional[dict[str, Any]]:
    """Where a `continue` would pick up, for /status. None when nothing is
    unfinished."""
    snapshot = load_resume_snapshot(session_id)
    index = snapshot.resume_index if snapshot else None
    if snapshot is None or index is None:
        return None
    return {
        "step": index + 1,
        "total": snapshot.total,
        "done": snapshot.done,
        "name": snapshot.resume_step_name,
        "next_action": snapshot.next_action(),
    }
