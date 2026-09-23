"""Production delivery invariants shared by every Tamfis-Code run.

This is deliberately provider-neutral. Models may propose work, but the
runtime owns whether a task is complete: mutations need scope/evidence,
multi-step work needs a plan, and completion needs verification records.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DeliveryPolicy:
    version: str
    planning: str
    mutation_requires_scope: bool
    verification_required: bool
    review_required: bool
    resumable: bool
    max_unverified_retries: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_delivery_policy(*, read_only: bool, requires_validation: bool, complexity: str) -> DeliveryPolicy:
    """Return deterministic safeguards for the current task profile."""
    mutating = not read_only
    substantial = complexity in {"moderate", "complex", "very_complex", "high"}
    return DeliveryPolicy(
        version="production-delivery-v1",
        planning="required" if substantial or mutating else "proportional",
        mutation_requires_scope=mutating,
        verification_required=bool(requires_validation or mutating),
        review_required=bool(substantial or mutating),
        resumable=True,
        max_unverified_retries=2,
    )


def completion_is_evidence_bound(policy: DeliveryPolicy, *, evidence_count: int, passed: bool) -> bool:
    """Prevent a model's prose from becoming a false completion claim."""
    if not passed:
        return False
    if policy.verification_required and evidence_count < 1:
        return False
    return True

