"""Independent requirement-to-evidence completion review."""
from __future__ import annotations

from dataclasses import dataclass, field

from .cognitive import EvidenceGraph, TaskContract


@dataclass(slots=True)
class ReviewResult:
    approved: bool
    missing_requirements: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class IndependentReviewer:
    def review(self, contract: TaskContract, graph: EvidenceGraph) -> ReviewResult:
        missing = graph.missing(contract)
        warnings: list[str] = []
        if contract.requested_mutation and not any(node.kind == "file_mutation" for node in graph.nodes.values()):
            warnings.append("A mutation was requested but no changed-file evidence exists.")
        if contract.requested_mutation and not any(node.kind == "validation_result" for node in graph.nodes.values()):
            warnings.append("A mutation was requested but no validation evidence exists.")
        return ReviewResult(not missing and not warnings, missing, warnings)
