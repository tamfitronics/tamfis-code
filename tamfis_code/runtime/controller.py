"""Deterministic controller that owns progress, budgets and stall detection."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .budgets import RuntimeBudgets
from .evidence import action_fingerprint, evidence_labels, is_empty_result, observation_fingerprint
from .state import RuntimePhase, RuntimeSnapshot


@dataclass(frozen=True)
class GuardDecision:
    allowed: bool
    terminal: bool = False
    reason: str = ""
    fingerprint: str = ""
    # True only when `terminal` was caused by the wall-clock budget running
    # out -- the one failure mode a caller may recover from by granting a
    # fresh turn (extend_runtime) instead of ending the task. Every other
    # terminal reason (repeated actions, stalls, tool-call ceiling) reflects
    # the agent making no real progress and must stay a hard failure.
    time_budget_exhausted: bool = False


@dataclass(frozen=True)
class ObservationDecision:
    useful: bool
    terminal: bool = False
    reason: str = ""
    evidence: tuple[str, ...] = field(default_factory=tuple)


class ExecutionController:
    def __init__(self, budgets: RuntimeBudgets | None = None) -> None:
        self.budgets = budgets or RuntimeBudgets()
        self.snapshot = RuntimeSnapshot()
        self.started_at = time.monotonic()
        self._last_action = ""
        self._last_observation = ""

    def _fail(self, reason: str) -> None:
        if not self.snapshot.terminal:
            self.snapshot.failure_reason = reason
            self.snapshot.transition(RuntimePhase.FAILED)

    def extend_runtime(self) -> bool:
        """Grant one more turn's worth of wall-clock budget instead of
        ending the task, when the only reason execution stopped is time.

        Returns False once the extension allowance itself is exhausted --
        at that point the caller must treat the timeout as final. A
        successful extension clears a FAILED phase that was set purely by
        `_check_time()` so execution can resume; it bypasses the normal
        transition table on purpose, since FAILED has no legal outgoing
        transitions and this is the one case where that's meant to be
        reversible.
        """
        if self.snapshot.runtime_extensions >= self.budgets.max_runtime_extensions:
            return False
        self.snapshot.runtime_extensions += 1
        self.started_at = time.monotonic()
        if self.snapshot.phase == RuntimePhase.FAILED and self.snapshot.failure_reason.startswith(
            "Runtime budget exhausted"
        ):
            self.snapshot.phase = RuntimePhase.EXECUTE
            self.snapshot.failure_reason = ""
        return True

    def _check_time(self) -> str:
        elapsed = time.monotonic() - self.started_at
        if elapsed >= self.budgets.max_runtime_seconds:
            return f"Runtime budget exhausted after {int(elapsed)} seconds."
        return ""

    def start_planning(self) -> None:
        if self.snapshot.phase == RuntimePhase.DISCOVER:
            self.snapshot.transition(RuntimePhase.PLAN)

    def start_execution(self) -> None:
        if self.snapshot.phase in {RuntimePhase.DISCOVER, RuntimePhase.PLAN, RuntimePhase.OBSERVE, RuntimePhase.REPAIR}:
            self.snapshot.transition(RuntimePhase.EXECUTE)

    def guard_action(self, tool_name: str, arguments: dict[str, Any]) -> GuardDecision:
        timeout = self._check_time()
        if timeout:
            self._fail(timeout)
            return GuardDecision(False, True, timeout, time_budget_exhausted=True)
        if self.snapshot.terminal:
            return GuardDecision(False, True, self.snapshot.failure_reason or "Runtime is terminal.")
        if self.snapshot.tool_calls >= self.budgets.max_tool_calls:
            reason = f"Tool-call budget exhausted ({self.budgets.max_tool_calls})."
            self._fail(reason)
            return GuardDecision(False, True, reason)

        fingerprint = action_fingerprint(tool_name, arguments)
        count = self.snapshot.action_counts.get(fingerprint, 0) + 1
        self.snapshot.action_counts[fingerprint] = count
        self.snapshot.repeated_actions = count
        if count > self.budgets.max_identical_actions:
            reason = (
                f"Blocked repeated action: {tool_name} with identical arguments was already attempted "
                f"{count - 1} times without sufficient progress."
            )
            # Repeated action is refused first; terminal failure is reserved for
            # continued no-progress pressure, so the model can choose a genuinely
            # different strategy once.
            terminal = self.snapshot.consecutive_empty_observations >= self.budgets.max_consecutive_empty_observations
            if terminal:
                self._fail(reason)
            return GuardDecision(False, terminal, reason, fingerprint)

        self.snapshot.tool_calls += 1
        self._last_action = fingerprint
        self.start_execution()
        return GuardDecision(True, fingerprint=fingerprint)

    def observe(self, tool_name: str, arguments: dict[str, Any], result: dict[str, Any]) -> ObservationDecision:
        if self.snapshot.terminal:
            return ObservationDecision(False, True, self.snapshot.failure_reason)
        if self.snapshot.phase == RuntimePhase.EXECUTE:
            self.snapshot.transition(RuntimePhase.OBSERVE)

        empty = is_empty_result(tool_name, result)
        fingerprint = observation_fingerprint(tool_name, result)
        observation_count = self.snapshot.observation_counts.get(fingerprint, 0) + 1
        self.snapshot.observation_counts[fingerprint] = observation_count
        self._last_observation = fingerprint

        if empty:
            self.snapshot.empty_observations += 1
            self.snapshot.consecutive_empty_observations += 1
            count = self.snapshot.consecutive_empty_observations
            if count >= self.budgets.max_consecutive_empty_observations:
                reason = (
                    f"Agent stalled after {count} consecutive tool results produced no useful evidence. "
                    "The runtime stopped the loop instead of repeating reconnaissance."
                )
                self._fail(reason)
                return ObservationDecision(False, True, reason)
            self.snapshot.transition(RuntimePhase.EXECUTE)
            return ObservationDecision(False, False, f"No useful evidence gained ({count}/{self.budgets.max_consecutive_empty_observations}).")

        labels = tuple(evidence_labels(tool_name, arguments, result))
        self.snapshot.evidence_items += len(labels) or 1
        self.snapshot.consecutive_empty_observations = 0
        # An action is only a loop when it repeats without changing the
        # evidence state.  Keep the guard strict for identical observations,
        # but forget an action's old count after genuinely novel evidence so a
        # model can sensibly re-run a check after editing or discovering a new
        # dependency.  The old implementation counted for the whole turn,
        # which made valid "inspect -> change -> verify" sequences look like
        # pathological repetition.
        if observation_count == 1:
            # New evidence can invalidate more than the immediately preceding
            # action (for example, a file edit changes which search commands
            # are meaningful), so reset the stale repetition window globally.
            self.snapshot.action_counts.clear()
        self.snapshot.transition(RuntimePhase.EXECUTE)
        return ObservationDecision(True, False, evidence=labels)

    def begin_validation(self) -> None:
        if self.snapshot.phase in {RuntimePhase.EXECUTE, RuntimePhase.OBSERVE, RuntimePhase.REPAIR}:
            self.snapshot.transition(RuntimePhase.VALIDATE)

    def record_repair(self) -> bool:
        self.snapshot.repair_rounds += 1
        if self.snapshot.repair_rounds > self.budgets.max_repair_rounds:
            self._fail(f"Repair budget exhausted ({self.budgets.max_repair_rounds}).")
            return False
        if self.snapshot.phase in {RuntimePhase.EXECUTE, RuntimePhase.OBSERVE, RuntimePhase.VALIDATE}:
            self.snapshot.transition(RuntimePhase.REPAIR)
        return True

    def extend_repair_budget(self) -> bool:
        """Grant a fresh max_repair_rounds window instead of ending the
        task, when record_repair() just ran out. Bounded by
        max_repair_extensions -- see budgets.py for why record_repair's
        counter is shared across unrelated recovery classes and can run out
        on infra noise before the model gets a real shot at the actual
        failure. Like extend_runtime(), reverses a FAILED phase that was
        set purely by this exhaustion so execution can resume."""
        if self.snapshot.repair_extensions >= self.budgets.max_repair_extensions:
            return False
        self.snapshot.repair_extensions += 1
        self.snapshot.repair_rounds = 0
        if self.snapshot.phase == RuntimePhase.FAILED and self.snapshot.failure_reason.startswith(
            "Repair budget exhausted"
        ):
            self.snapshot.phase = RuntimePhase.EXECUTE
            self.snapshot.failure_reason = ""
        return True

    def record_plan_revision(self) -> bool:
        self.snapshot.plan_revisions += 1
        if self.snapshot.plan_revisions > self.budgets.max_plan_revisions:
            self._fail(f"Plan revision budget exhausted ({self.budgets.max_plan_revisions}).")
            return False
        return True

    def complete(self) -> None:
        self.begin_validation()
        if self.snapshot.phase == RuntimePhase.VALIDATE:
            self.snapshot.transition(RuntimePhase.COMPLETE)

    def fail(self, reason: str) -> None:
        self._fail(reason)
