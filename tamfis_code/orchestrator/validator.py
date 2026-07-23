"""Evidence-based validation and completion integrity."""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from ..routing import TaskProfile, TaskType


@dataclass
class ValidationReport:
    passed: bool
    checks: list[dict[str, Any]] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    severity: str = "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": self.checks,
            "unresolved": self.unresolved,
            "severity": self.severity,
        }


_UNSUPPORTED_INSPECTION_CLAIM_RE = re.compile(
    r"\b(?:i\s+(?:have\s+)?(?:reviewed|examined|inspected|audited|analysed|analyzed)|"
    r"i\s+now\s+have\s+the\s+full\s+picture|"
    r"after\s+(?:reviewing|examining|inspecting|auditing|analysing|analyzing)|"
    r"the\s+(?:review|inspection|audit|analysis)\s+(?:shows|found|identified|revealed)|"
    # Confirmed live (meta/llama-3.1-70b-instruct on NVIDIA NIM): a weak
    # model can fabricate a *specific* tool's past-tense result -- "the
    # search_code tool has found several references to..." -- without ever
    # issuing that tool call. This reads as reported evidence exactly like
    # the phrases above, just naming the tool instead of saying "I".
    r"the\s+\w+\s+tool\s+(?:has\s+)?(?:found|returned|shows?|revealed|indicates?)|"
    r"(?:the\s+)?results?\s+(?:suggest|indicate)s?\b)\b",
    re.IGNORECASE,
)


def _claims_completed_inspection(final_text: str) -> bool:
    """Return True only for prose that asserts a real inspection occurred.

    Generic completion words such as ``Fixed`` or ``Done`` remain warning-level
    when evidence is missing, preserving the existing caveat behaviour.  This
    targets the dangerous cross-workspace failure where a provider says it
    reviewed or inspected code despite recording no successful tool result.
    """
    return bool(_UNSUPPORTED_INSPECTION_CLAIM_RE.search(final_text or ""))


def validate_completion(
    *, profile: TaskProfile, tool_records: list[dict[str, Any]],
    any_mutation: bool, final_text: str,
) -> ValidationReport:
    checks: list[dict[str, Any]] = []
    unresolved: list[str] = []
    successful_tools = [item for item in tool_records if item.get("success") is True]
    tool_evidence_passed = bool(successful_tools) or not profile.requires_tools
    checks.append({"name": "tool_evidence_recorded", "passed": tool_evidence_passed})
    if not tool_evidence_passed:
        unresolved.append("This task type requires tool evidence, but no successful tool call was recorded.")

    if profile.task_type in {TaskType.EDIT, TaskType.DEBUG}:
        checks.append({"name": "mutation_recorded", "passed": any_mutation})
        if not any_mutation:
            unresolved.append("The request required a code change, but no successful file mutation was recorded.")

    if profile.requires_validation:
        validation_tools = {
            "execute_command", "get_git_info", "read_file", "search_code", "list_directory",
        }
        validated = any(item.get("tool_name") in validation_tools and item.get("success") for item in tool_records)
        # A successful recorded mutation is proportionate validation for a
        # trivial single-file creation/edit. Complex audit/debug/test turns
        # still require an independent inspection or command result.
        if profile.task_type == TaskType.EDIT and any_mutation:
            validated = True
        checks.append({"name": "validation_evidence", "passed": validated})
        if not validated:
            unresolved.append("No successful validation or inspection tool result was recorded.")

    checks.append({"name": "non_empty_report", "passed": bool(final_text.strip())})

    passed = not unresolved and all(c["passed"] for c in checks)
    severity = "pass"
    if not passed:
        severity = "warning"
        if profile.requires_tools and not successful_tools and _claims_completed_inspection(final_text):
            severity = "error"
            unresolved.append(
                "The response claims repository inspection or review, but no successful tool evidence supports that claim."
            )

    return ValidationReport(passed, checks, unresolved, severity=severity)
