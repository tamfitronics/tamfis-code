"""Grouped approval primitives for Claude Code-style execution."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


_RISK_ORDER = {"read_only": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


@dataclass(slots=True)
class ApprovalAction:
    tool_name: str
    arguments: dict[str, Any]
    purpose: str
    risk: str = "read_only"
    cwd: str | None = None


@dataclass(slots=True)
class ApprovalBatch:
    actions: list[ApprovalAction] = field(default_factory=list)
    decision: str = "pending"

    @property
    def highest_risk(self) -> str:
        return max((a.risk for a in self.actions), key=lambda r: _RISK_ORDER.get(r, 99), default="read_only")

    @property
    def requires_prompt(self) -> bool:
        return any(action.risk != "read_only" for action in self.actions)

    def add(self, action: ApprovalAction) -> None:
        self.actions.append(action)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "highest_risk": self.highest_risk,
            "requires_prompt": self.requires_prompt,
            "actions": [asdict(action) for action in self.actions],
        }

    @property
    def risky_actions(self) -> list[ApprovalAction]:
        return [action for action in self.actions if action.risk != "read_only"]


def describe_batch(batch: ApprovalBatch) -> str:
    """Render a single numbered prompt body for every risky action in a turn.

    Used to fold several tool calls from the same model turn into one
    approval decision (Claude-Code-style batching) instead of prompting once
    per call -- callers pass this combined text into the same
    ``resolve_approval_decision[_async]`` used for a single-action prompt, so
    policy handling (auto/safe/deny, session-scoped approval) is unchanged.
    """
    import json as _json

    lines = []
    for index, action in enumerate(batch.risky_actions, start=1):
        rendered_args = _json.dumps(action.arguments, default=str)
        lines.append(f"{index}. {action.tool_name}({rendered_args})  [{action.risk}]")
    return "\n".join(lines)
