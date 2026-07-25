"""Phase 5 cognitive orchestration primitives.

These structures make the runtime reason against an explicit task contract,
record requirement-to-evidence links, revise plans when evidence changes, and
block unsupported completion claims.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable


class RequirementStatus(str, Enum):
    PENDING = "pending"
    SATISFIED = "satisfied"
    BLOCKED = "blocked"
    FAILED = "failed"


@dataclass(slots=True)
class SuccessCriterion:
    criterion_id: str
    description: str
    required_evidence: list[str] = field(default_factory=list)
    status: RequirementStatus = RequirementStatus.PENDING


@dataclass(slots=True)
class TaskContract:
    objective: str
    intent: str
    requested_mutation: bool
    constraints: list[str] = field(default_factory=list)
    criteria: list[SuccessCriterion] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def derive(
        cls,
        objective: str,
        *,
        read_only: bool,
        approval_policy: str,
        metadata: dict[str, Any] | None = None,
    ) -> "TaskContract":
        text = " ".join(objective.split())
        lowered = text.lower()
        mutation_words = ("fix", "edit", "change", "implement", "create", "write", "refactor", "remove", "add")
        requested_mutation = (not read_only) and any(word in lowered for word in mutation_words)
        if read_only:
            intent = "audit"
        elif requested_mutation:
            intent = "engineering_change"
        else:
            intent = "analysis"
        criteria = [
            SuccessCriterion("objective", text or "Complete the requested task", ["completion_summary"]),
            SuccessCriterion("evidence", "Support completion with concrete evidence", ["tool_observation"]),
        ]
        if requested_mutation:
            criteria.extend([
                SuccessCriterion("mutation", "Record every changed file", ["file_mutation"]),
                SuccessCriterion("validation", "Validate the requested outcome", ["validation_result"]),
            ])
        return cls(
            objective=text,
            intent=intent,
            requested_mutation=requested_mutation,
            constraints=[f"approval_policy={approval_policy}", "preserve user corrections", "do not claim unsupported completion"],
            criteria=criteria,
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for criterion in data["criteria"]:
            criterion["status"] = criterion["status"].value if hasattr(criterion["status"], "value") else criterion["status"]
        return data


@dataclass(slots=True)
class EvidenceNode:
    evidence_id: str
    kind: str
    summary: str
    source: str
    requirement_ids: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvidenceGraph:
    nodes: dict[str, EvidenceNode] = field(default_factory=dict)

    def add(self, node: EvidenceNode) -> None:
        self.nodes[node.evidence_id] = node

    def evidence_for(self, criterion_id: str) -> list[EvidenceNode]:
        return [node for node in self.nodes.values() if criterion_id in node.requirement_ids]

    def satisfy_contract(self, contract: TaskContract) -> None:
        for criterion in contract.criteria:
            nodes = self.evidence_for(criterion.criterion_id)
            kinds = {node.kind for node in nodes}
            required = set(criterion.required_evidence)
            if required and required.issubset(kinds):
                criterion.status = RequirementStatus.SATISFIED

    def missing(self, contract: TaskContract) -> list[str]:
        self.satisfy_contract(contract)
        return [criterion.description for criterion in contract.criteria if criterion.status != RequirementStatus.SATISFIED]

    def to_dict(self) -> dict[str, Any]:
        return {key: asdict(value) for key, value in self.nodes.items()}


@dataclass(slots=True)
class PlanRevision:
    revision: int
    reason: str
    previous_steps: list[str]
    new_steps: list[str]
    evidence_ids: list[str] = field(default_factory=list)


class ReplanningEngine:
    """Deterministically revises plans when new evidence invalidates them."""

    def revise(
        self,
        *,
        revision: int,
        reason: str,
        previous_steps: Iterable[str],
        replacement_steps: Iterable[str],
        evidence_ids: Iterable[str] = (),
    ) -> PlanRevision:
        old = [step.strip() for step in previous_steps if step.strip()]
        new = [step.strip() for step in replacement_steps if step.strip()]
        if not reason.strip():
            raise ValueError("plan revision requires a reason")
        if not new:
            raise ValueError("revised plan cannot be empty")
        if old == new:
            raise ValueError("revised plan must materially differ")
        return PlanRevision(revision, reason.strip(), old, new, list(evidence_ids))
