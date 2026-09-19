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
    return ResumeSnapshot(
        session_id=session_id,
        plan_id=str(plan.get("id") or ""),
        objective=str(plan.get("objective") or ""),
        steps=steps,
        static=static,
        task_state=dict(state.task_state or {}),
        ledger_next_action=ledger_next,
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
