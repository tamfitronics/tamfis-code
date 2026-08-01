"""Evidence-based validation and completion integrity."""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any
from pathlib import Path

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

_VERIFIED_NO_CHANGE_RE = re.compile(
    r"\b(?:"
    r"(?:issue|error|bug|failure|problem)\s+(?:is|was|has\s+been)\s+"
    r"(?:now\s+)?(?:already\s+)?(?:fixed|resolved|working)|"
    r"already\s+(?:fixed|resolved|implemented|correct|present|working)|"
    r"no\s+(?:code|file|source)?\s*changes?\s+(?:were\s+)?(?:needed|required|necessary)|"
    r"(?:fix|change|implementation|conflict\s+clause)\s+(?:is|was)\s+already\s+"
    r"(?:present|applied|deployed|correct)|"
    r"running\s+(?:server|service|deployment)\b[^.!?\n]{0,120}\b(?:correct|fixed|resolved)"
    r")\b",
    re.IGNORECASE,
)

_MUTATING_TOOLS = {
    "write_file", "edit_file", "create_file", "patch_file",
    "extract_archive", "repackage_archive",
    "create_artifact",
}

_VALIDATION_EVIDENCE_TOOLS = {
    "execute_command", "get_git_info", "read_file", "search_code", "list_directory",
}


def verified_no_change_completion(
    *, tool_records: list[dict[str, Any]], final_text: str,
) -> bool:
    """Accept a genuine evidence-backed no-op, not a narrated fake edit.

    Coding agents sometimes discover that the requested fix is already in
    the checked-out code and deployed runtime. Requiring a fresh mutation in
    that case encourages meaningless file touches and turns a truthful
    verification into a false failure. The exception is deliberately narrow:
    the report must explicitly say no change was needed/already resolved,
    inspection or validation must have succeeded, no mutating tool may have
    been attempted, and the latest command (if any) must be green.
    """
    if not _VERIFIED_NO_CHANGE_RE.search(final_text or ""):
        return False
    if any(item.get("tool_name") in _MUTATING_TOOLS for item in tool_records):
        return False
    if not any(
        item.get("tool_name") in _VALIDATION_EVIDENCE_TOOLS
        and item.get("success") is True
        for item in tool_records
    ):
        return False
    commands = [item for item in tool_records if item.get("tool_name") == "execute_command"]
    if commands:
        latest = commands[-1]
        if latest.get("success") is not True or latest.get("exit_code") not in (None, 0):
            return False
    return True


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
    any_mutation: bool, final_text: str, objective: str = "", workspace_root: str = "",
) -> ValidationReport:
    checks: list[dict[str, Any]] = []
    unresolved: list[str] = []
    successful_tools = [item for item in tool_records if item.get("success") is True]
    verified_no_change = verified_no_change_completion(
        tool_records=tool_records, final_text=final_text,
    )
    tool_evidence_passed = bool(successful_tools) or not profile.requires_tools
    checks.append({"name": "tool_evidence_recorded", "passed": tool_evidence_passed})
    if not tool_evidence_passed:
        unresolved.append("This task type requires tool evidence, but no successful tool call was recorded.")

    if profile.task_type in {TaskType.EDIT, TaskType.DEBUG}:
        mutation_requirement_met = any_mutation or verified_no_change
        checks.append({
            "name": "mutation_recorded",
            "passed": mutation_requirement_met,
            "accepted_verified_no_change": verified_no_change,
        })
        if not mutation_requirement_met:
            unresolved.append("The request required a code change, but no successful file mutation was recorded.")

        # A successful transport response is not enough for a project build:
        # every explicitly requested output path must exist and be non-empty.
        # This catches the common long-generation failure where a model builds
        # the first few files in a supplied tree and then declares the whole
        # theme/plugin complete.
        expected_paths = _explicit_output_paths(objective)
        if expected_paths and workspace_root:
            root = Path(workspace_root).resolve()
            missing: list[str] = []
            for requested in expected_paths:
                candidate = (root / requested).resolve()
                if root not in candidate.parents and candidate != root:
                    continue
                try:
                    valid = candidate.is_file() and candidate.stat().st_size > 0
                except OSError:
                    valid = False
                if not valid:
                    missing.append(requested)
            checks.append({"name": "explicit_outputs_exist", "passed": not missing, "expected": expected_paths})
            if missing:
                unresolved.append("Explicitly requested output files are missing or empty: " + ", ".join(missing[:30]))

    if profile.requires_validation:
        validated = any(
            item.get("tool_name") in _VALIDATION_EVIDENCE_TOOLS and item.get("success")
            for item in tool_records
        )

        # Verification is ordered evidence.  A later failed check/build must
        # invalidate an earlier successful command; otherwise a model can run
        # `build`, fail `check`, and still be allowed to report "all checks
        # passed".  The runner may ask the model to repair and retry, but it
        # must never call the turn complete while the latest verification is
        # red.
        commands = [item for item in tool_records if item.get("tool_name") == "execute_command"]
        latest_command = commands[-1] if commands else None
        latest_command_failed = bool(
            latest_command
            and (
                latest_command.get("success") is not True
                or latest_command.get("exit_code") not in (None, 0)
            )
        )
        checks.append({
            "name": "latest_command_clean",
            "passed": not latest_command_failed,
        })
        if latest_command_failed:
            command = latest_command.get("arguments", {}).get("command", "verification command")
            unresolved.append(
                f"The latest verification command failed: {str(command)[:160]}. "
                "The task cannot be reported complete until it is repaired and rerun successfully."
            )
            validated = False

        # A successful mutation alone is not validation for code edits.  A
        # read/inspection or command result is required, and failed latest
        # verification always wins over earlier evidence.
        checks.append({"name": "validation_evidence", "passed": validated})
        if not validated:
            unresolved.append("No successful validation or inspection tool result was recorded.")

    checks.append({"name": "non_empty_report", "passed": bool(final_text.strip())})

    passed = not unresolved and all(c["passed"] for c in checks)
    severity = "pass"
    if not passed:
        severity = "warning"
        if profile.requires_validation and any(
            check["name"] == "latest_command_clean" and not check["passed"]
            for check in checks
        ):
            severity = "error"
        if (
            profile.task_type in {TaskType.EDIT, TaskType.DEBUG}
            and not any_mutation
            and not verified_no_change
        ):
            # A requested edit/debug task without a recorded file mutation is
            # not a completed task. Warnings used to flow through
            # orchestrator.complete(), causing the CLI to claim success after
            # the model merely described a change.
            severity = "error"
        if profile.requires_tools and not successful_tools and _claims_completed_inspection(final_text):
            severity = "error"
            unresolved.append(
                "The response claims repository inspection or review, but no successful tool evidence supports that claim."
            )

    return ValidationReport(passed, checks, unresolved, severity=severity)


_OUTPUT_PATH_RE = re.compile(
    r"(?<![\w./-])([\w.-]+(?:/[\w.@+-]+)+\.(?:php|css|scss|sass|js|jsx|ts|tsx|json|xml|yaml|yml|toml|md|txt|pot|po|mo|py|go|rs|java|rb|sh|html|htm|svg|png|jpe?g|webp|docx|xlsx|pptx|pdf))\b",
    re.IGNORECASE,
)


def _explicit_output_paths(objective: str) -> list[str]:
    paths: list[str] = []
    for match in _OUTPUT_PATH_RE.finditer(objective or ""):
        value = match.group(1).strip().replace("\\", "/")
        if value.startswith(("http://", "https://")) or ".." in Path(value).parts:
            continue
        if value not in paths:
            paths.append(value)
    return paths[:200]
