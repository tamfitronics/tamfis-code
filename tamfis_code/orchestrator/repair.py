"""Deterministic repair classification and strategy selection."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class FailureClass(str, Enum):
    TOOL_SCHEMA_ERROR = "tool_schema_error"
    SHELL_QUOTING_ERROR = "shell_quoting_error"
    FILE_NOT_FOUND = "file_not_found"
    PERMISSION_DENIED = "permission_denied"
    VALIDATION_FAILURE = "validation_failure"
    COMMAND_FAILURE = "command_failure"
    PROVIDER_STALL = "provider_stall"
    DUPLICATE_ACTION = "duplicate_action"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class RepairDecision:
    failure_class: FailureClass
    strategy: str
    retry_allowed: bool
    force_different_tool: bool = False


def classify_failure(*, tool_name: str, result: dict[str, Any] | str) -> FailureClass:
    text = str(result).lower()
    if "unexpected keyword argument" in text or "schema" in text and "argument" in text:
        return FailureClass.TOOL_SCHEMA_ERROR
    if "syntax error near unexpected token" in text or (tool_name == "execute_command" and "printf" in text):
        return FailureClass.SHELL_QUOTING_ERROR
    if (
        "file not found" in text
        or "no such file" in text
        or ("file" in text and "not found" in text)
    ):
        return FailureClass.FILE_NOT_FOUND
    if "permission denied" in text:
        return FailureClass.PERMISSION_DENIED
    if "validation" in text and ("failed" in text or "error" in text):
        return FailureClass.VALIDATION_FAILURE
    if "stalled" in text or "timeout" in text:
        return FailureClass.PROVIDER_STALL
    if "repeated action" in text or "identical arguments" in text:
        return FailureClass.DUPLICATE_ACTION
    if "exit" in text or "return_code" in text or "command failed" in text:
        return FailureClass.COMMAND_FAILURE
    return FailureClass.UNKNOWN


def choose_repair(*, tool_name: str, result: dict[str, Any] | str, attempt: int) -> RepairDecision:
    failure = classify_failure(tool_name=tool_name, result=result)
    if failure == FailureClass.TOOL_SCHEMA_ERROR:
        return RepairDecision(failure, "normalise arguments against the canonical tool schema", attempt < 2)
    if failure == FailureClass.SHELL_QUOTING_ERROR:
        return RepairDecision(failure, "switch to native write_file/edit_file; do not retry shell source construction", True, True)
    if failure == FailureClass.FILE_NOT_FOUND:
        return RepairDecision(failure, "refresh workspace inventory and resolve the canonical path", attempt < 2)
    if failure == FailureClass.PERMISSION_DENIED:
        return RepairDecision(
            failure,
            (
                "preserve the canonical workspace; report the exact denied path and command "
                "through the approval boundary without sudo, password guessing, ownership "
                "changes, or copying the project"
            ),
            attempt < 1,
        )
    if failure == FailureClass.VALIDATION_FAILURE:
        return RepairDecision(failure, "rollback the mutation, inspect validator output, then generate a corrected change", attempt < 2)
    if failure == FailureClass.PROVIDER_STALL:
        return RepairDecision(failure, "cancel the stream, preserve partial output, and route through the configured fallback chain", attempt < 2)
    if failure == FailureClass.DUPLICATE_ACTION:
        return RepairDecision(failure, "require a materially different tool or argument set", False, True)
    if failure == FailureClass.COMMAND_FAILURE:
        return RepairDecision(failure, "inspect stderr and select a different evidence-backed command", attempt < 2)
    return RepairDecision(failure, "surface the failure and preserve partial completion evidence", False)
