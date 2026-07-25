"""Truthful task completion semantics."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CompletionStatus(str, Enum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    BLOCKED = "blocked"
    FAILED = "failed"
    NO_CHANGES_REQUIRED = "no_changes_required"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class CompletionSummary:
    status: CompletionStatus
    summary: str
    changed_files: list[str] = field(default_factory=list)
    validations: list[dict[str, Any]] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "summary": self.summary,
            "changed_files": self.changed_files,
            "validations": self.validations,
            "unresolved": self.unresolved,
        }


def determine_completion(*, requested_mutation: bool, changed_files: list[str], validation_passed: bool,
                         unresolved: list[str], error: str | None = None, cancelled: bool = False) -> CompletionStatus:
    if cancelled:
        return CompletionStatus.CANCELLED
    if error and not changed_files:
        return CompletionStatus.FAILED
    if unresolved:
        return CompletionStatus.PARTIAL if changed_files else CompletionStatus.BLOCKED
    if requested_mutation and not changed_files:
        return CompletionStatus.NO_CHANGES_REQUIRED if validation_passed else CompletionStatus.BLOCKED
    if validation_passed:
        return CompletionStatus.COMPLETED
    return CompletionStatus.PARTIAL
