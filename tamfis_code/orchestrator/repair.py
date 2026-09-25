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
    TEST_FAILURE = "test_failure"
    SYNTAX_ERROR = "syntax_error"
    DEPENDENCY_MISSING = "dependency_missing"
    GIT_USAGE_ERROR = "git_usage_error"
    AUTHENTICATION_FAILED = "authentication_failed"
    NETWORK_FAILURE = "network_failure"
    RATE_LIMITED = "rate_limited"
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
    if (
        "fatal: unrecognized argument" in text
        or "unknown option" in text and "git" in text
        or "ambiguous argument" in text and "git" in text
    ):
        return FailureClass.GIT_USAGE_ERROR
    if (
        "modulenotfounderror" in text
        or "no module named" in text
        or "cannot find module" in text
        or "command not found" in text
        or "is not recognized as an internal or external command" in text
    ):
        return FailureClass.DEPENDENCY_MISSING
    if "syntaxerror" in text or "parse error" in text or "unexpected token" in text:
        return FailureClass.SYNTAX_ERROR
    if (
        "assertionerror" in text
        or "tests failed" in text
        or "test failed" in text
        or (" failed" in text and ("pytest" in text or "jest" in text or "vitest" in text))
    ):
        return FailureClass.TEST_FAILURE
    if "401 unauthorized" in text or "authentication failed" in text or "invalid api key" in text:
        return FailureClass.AUTHENTICATION_FAILED
    if "429" in text or "rate limit" in text or "too many requests" in text:
        return FailureClass.RATE_LIMITED
    if (
        "could not resolve host" in text
        or "connection refused" in text
        or "network is unreachable" in text
        or "temporary failure in name resolution" in text
    ):
        return FailureClass.NETWORK_FAILURE
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
        return RepairDecision(
            failure,
            (
                "inspect the workspace tree and search for the requested file, then retry "
                "with one exact canonical path; never repeat the guessed path or declare "
                "the file absent after one miss"
            ),
            attempt < 2,
            True,
        )
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
    if failure == FailureClass.TEST_FAILURE:
        return RepairDecision(
            failure,
            "isolate the first failing test, map its assertion to the changed code, fix the root cause, then rerun the focused test before the full suite",
            attempt < 3,
        )
    if failure == FailureClass.SYNTAX_ERROR:
        return RepairDecision(
            failure,
            "open the exact reported file and line with surrounding context, repair syntax using the language-native edit tool, then run the narrow parser or linter",
            attempt < 2,
            True,
        )
    if failure == FailureClass.DEPENDENCY_MISSING:
        return RepairDecision(
            failure,
            "inspect the project manifest and lockfile; install only through the declared package manager, then retry the original validation once",
            attempt < 1,
        )
    if failure == FailureClass.GIT_USAGE_ERROR:
        return RepairDecision(
            failure,
            "inspect the installed Git command syntax, remove or replace the unsupported flag, and retry the same read without changing repository state",
            attempt < 2,
        )
    if failure == FailureClass.AUTHENTICATION_FAILED:
        return RepairDecision(
            failure,
            "stop automatic retries, preserve the exact service and credential name, and request the missing authentication through the normal approval/configuration boundary",
            False,
        )
    if failure == FailureClass.RATE_LIMITED:
        return RepairDecision(
            failure,
            "honour Retry-After when present and route once through a configured fallback instead of hammering the same endpoint",
            attempt < 1,
            True,
        )
    if failure == FailureClass.NETWORK_FAILURE:
        return RepairDecision(
            failure,
            "separate DNS, connection, and service-health checks; preserve partial work and retry only after a read-only connectivity check",
            attempt < 1,
        )
    if failure == FailureClass.PROVIDER_STALL:
        return RepairDecision(failure, "cancel the stream, preserve partial output, and route through the configured fallback chain", attempt < 2)
    if failure == FailureClass.DUPLICATE_ACTION:
        return RepairDecision(failure, "require a materially different tool or argument set", False, True)
    if failure == FailureClass.COMMAND_FAILURE:
        return RepairDecision(failure, "inspect stderr and select a different evidence-backed command", attempt < 2)
    return RepairDecision(failure, "surface the failure and preserve partial completion evidence", False)
