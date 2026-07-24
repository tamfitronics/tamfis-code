"""Runtime state and legal transition rules."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class RuntimePhase(str, Enum):
    DISCOVER = "discover"
    PLAN = "plan"
    EXECUTE = "execute"
    OBSERVE = "observe"
    VALIDATE = "validate"
    REPAIR = "repair"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


_TERMINAL = {RuntimePhase.COMPLETE, RuntimePhase.FAILED, RuntimePhase.CANCELLED}
_ALLOWED: dict[RuntimePhase, set[RuntimePhase]] = {
    RuntimePhase.DISCOVER: {RuntimePhase.PLAN, RuntimePhase.EXECUTE, RuntimePhase.FAILED, RuntimePhase.CANCELLED},
    RuntimePhase.PLAN: {RuntimePhase.EXECUTE, RuntimePhase.FAILED, RuntimePhase.CANCELLED},
    RuntimePhase.EXECUTE: {RuntimePhase.OBSERVE, RuntimePhase.VALIDATE, RuntimePhase.REPAIR, RuntimePhase.FAILED, RuntimePhase.CANCELLED},
    RuntimePhase.OBSERVE: {RuntimePhase.EXECUTE, RuntimePhase.VALIDATE, RuntimePhase.REPAIR, RuntimePhase.FAILED, RuntimePhase.CANCELLED},
    RuntimePhase.VALIDATE: {RuntimePhase.COMPLETE, RuntimePhase.REPAIR, RuntimePhase.FAILED, RuntimePhase.CANCELLED},
    RuntimePhase.REPAIR: {RuntimePhase.EXECUTE, RuntimePhase.VALIDATE, RuntimePhase.FAILED, RuntimePhase.CANCELLED},
    RuntimePhase.COMPLETE: set(), RuntimePhase.FAILED: set(), RuntimePhase.CANCELLED: set(),
}


@dataclass
class RuntimeSnapshot:
    phase: RuntimePhase = RuntimePhase.DISCOVER
    tool_calls: int = 0
    empty_observations: int = 0
    consecutive_empty_observations: int = 0
    plan_revisions: int = 0
    repair_rounds: int = 0
    evidence_items: int = 0
    repeated_actions: int = 0
    failure_reason: str = ""
    action_counts: dict[str, int] = field(default_factory=dict)
    observation_counts: dict[str, int] = field(default_factory=dict)

    @property
    def terminal(self) -> bool:
        return self.phase in _TERMINAL

    def transition(self, target: RuntimePhase) -> None:
        if target == self.phase:
            return
        if target not in _ALLOWED[self.phase]:
            raise RuntimeError(f"illegal runtime transition: {self.phase.value} -> {target.value}")
        self.phase = target

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["phase"] = self.phase.value
        return data
