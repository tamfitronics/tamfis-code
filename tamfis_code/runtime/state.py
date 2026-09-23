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
    novel_observations: int = 0
    repeated_actions: int = 0
    runtime_extensions: int = 0
    repair_extensions: int = 0
    tool_call_extensions: int = 0
    plan_revision_extensions: int = 0
    failure_reason: str = ""
    action_counts: dict[str, int] = field(default_factory=dict)
    observation_counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Checkpoint/API payloads carry the wire value, while the controller
        # uses RuntimePhase for transition rules and rendering. Normalize at
        # construction so every later ``phase.value`` access is safe.
        if not isinstance(self.phase, RuntimePhase):
            self.phase = RuntimePhase(str(self.phase))

    @property
    def terminal(self) -> bool:
        return self.phase in _TERMINAL

    def transition(self, target: RuntimePhase) -> None:
        # Checkpoints and API callers may provide the wire value instead of
        # the enum. Normalize before consulting transition rules or rendering
        # diagnostics; otherwise an invalid-transition error itself crashes
        # with ``'str' object has no attribute 'value'``.
        if isinstance(self.phase, str):
            self.phase = RuntimePhase(self.phase)
        if isinstance(target, str):
            target = RuntimePhase(target)
        if target == self.phase:
            return
        if target not in _ALLOWED[self.phase]:
            raise RuntimeError(f"illegal runtime transition: {self.phase.value} -> {target.value}")
        self.phase = target

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["phase"] = self.phase.value
        return data
