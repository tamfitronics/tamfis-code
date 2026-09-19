"""Claude Code-style tool records for the scrollback.

Every tool call is a durable two-part record instead of a bare arrow line (or,
for reads and edits, nothing at all):

    ● Read(tamfis_code/interactive.py)
      ⎿  Read 400 lines

    ● Bash(cd /home/x && python3 -m pytest tests/test_live_input.py -q …)
      ⎿  60 passed in 1.7s

    ● Read 3 files, searched for 2 patterns          <- consecutive reads/searches, grouped
      ⎿  interactive.py, render.py, state.py … +2 more

Pure functions only (no Rich, no I/O), so the wording is unit-testable and the
renderer stays a thin printer.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from .safety import redact_secrets

# Tool name -> the short title shown before the parenthesised target.
_DISPLAY_NAMES = {
    "read_file": "Read",
    "list_directory": "List",
    "create_directory": "Mkdir",
    "search_code": "Search",
    "search_files": "Search",
    "grep_files": "Search",
    "glob_files": "Glob",
    "find_references": "References",
    "get_git_info": "Git",
    "write_file": "Write",
    "create_file": "Write",
    "edit_file": "Update",
    "file_edit": "Update",
    "update_file": "Update",
    "execute_command": "Bash",
    "run_command": "Bash",
    "remote_exec": "Bash",
    "read_background_job": "BackgroundJob",
    "web_search": "Web Search",
    "web_fetch": "Fetch",
    "fetch_url": "Fetch",
    "fetch": "Fetch",
    "inspect_artifact": "Inspect",
    "create_artifact": "Artifact",
    "extract_archive": "Extract",
    "repackage_archive": "Repackage",
    "retrieve_evidence": "Recall",
    "ask_user_question": "Ask",
    "list_external_agent_sessions": "Sessions",
    "read_external_agent_session": "Session",
}

# Read-only tools that are GROUPED into one collapsed line when consecutive:
# tool -> (past verb, singular noun, plural noun).
_GROUP_CATEGORY = {
    "read_file": ("Read", "file", "files"),
    "list_directory": ("Listed", "directory", "directories"),
    "search_code": ("Searched for", "pattern", "patterns"),
    "search_files": ("Searched for", "pattern", "patterns"),
    "grep_files": ("Searched for", "pattern", "patterns"),
    "glob_files": ("Found", "file pattern", "file patterns"),
    "find_references": ("Found references for", "symbol", "symbols"),
    "get_git_info": ("Read", "Git repository", "Git repositories"),
    "read_background_job": ("Checked", "background job", "background jobs"),
}

_MAX_TARGET_CHARS = 100
_MAX_RESULT_LINES = 3
_MAX_RESULT_LINE_CHARS = 120
_LEADING_STATUS_GLYPH = re.compile(r"^\s*(?:✅|✓|☑️|✔️?|❌|✗|⚠️?)\s*")


def normalized_name(tool: str) -> str:
    return (tool or "tool").strip().lower().replace("-", "_").rsplit("/", 1)[-1]


def display_name(tool: str) -> str:
    name = normalized_name(tool)
    return _DISPLAY_NAMES.get(name) or name.replace("_", " ").strip().title() or "Tool"


def _one_line(text: Any, limit: int) -> str:
    """Collapse whitespace/newlines to single spaces and truncate with "…"."""
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[: max(1, limit - 1)].rstrip() + "…"


def display_target(tool: str, arguments: Optional[dict[str, Any]], *, limit: int = _MAX_TARGET_CHARS) -> str:
    """The one-line argument summary inside the parentheses. Secrets redacted."""
    args = arguments if isinstance(arguments, dict) else {}
    name = normalized_name(tool)
    if name in {"execute_command", "run_command", "remote_exec"}:
        return _one_line(redact_secrets(str(args.get("command") or "")), limit)
    if name in {"search_code", "search_files", "grep_files", "glob_files", "find_references"}:
        pattern = args.get("query") or args.get("pattern") or args.get("symbol") or args.get("name") or ""
        parts = [f'pattern: "{_one_line(pattern, 60)}"'] if pattern else []
        path = args.get("path") or args.get("directory") or ""
        if path and str(path) not in {".", "./"}:
            parts.append(f"path: \"{_one_line(path, 40)}\"")
        return _one_line(", ".join(parts), limit)
    if name == "ask_user_question":
        asked = args.get("questions")
        if isinstance(asked, list) and len(asked) > 1:
            return f"{len(asked)} questions"
        if isinstance(asked, list) and asked and isinstance(asked[0], dict):
            return _one_line(asked[0].get("question") or "", limit)
        return _one_line(args.get("question") or "", limit)
    if name in {"web_fetch", "fetch_url", "fetch"}:
        return _one_line(args.get("url") or "", limit)
    if name == "web_search":
        return _one_line(args.get("query") or "", limit)
    target = (
        args.get("path") or args.get("file_path") or args.get("url")
        or args.get("query") or args.get("pattern") or args.get("command") or args.get("name") or ""
    )
    if not target:
        # First scalar argument, so an unfamiliar tool still says what it acted on.
        for key, value in args.items():
            if isinstance(value, (str, int, float)) and str(value).strip():
                return _one_line(f"{key}: {value}", limit)
    return _one_line(redact_secrets(str(target)), limit)


def group_header(counts: dict[str, int]) -> str:
    """"Read 3 files, searched for 2 patterns" from {tool: count} (first-used order)."""
    merged: dict[tuple[str, str, str], int] = {}
    for tool, count in counts.items():
        category = _GROUP_CATEGORY.get(normalized_name(tool))
        if category and count > 0:
            merged[category] = merged.get(category, 0) + count
    parts = []
    for (verb, singular, plural), count in merged.items():
        parts.append(f"{verb} {count} {singular if count == 1 else plural}")
    if not parts:
        return ""
    return ", ".join([parts[0], *(p[0].lower() + p[1:] for p in parts[1:])])


_FAILURE_PREFIX = {
    "read_file": "Read failed: ",
    "list_directory": "List failed: ",
    "write_file": "Write failed: ",
    "create_file": "Write failed: ",
    "edit_file": "Edit failed: ",
    "file_edit": "Edit failed: ",
    "update_file": "Edit failed: ",
}


def failure_line(tool: str, message: str) -> str:
    """One red result line for a failed call, naming WHAT failed and the real
    reason ("Read failed: Permission denied: /etc/shadow")."""
    cleaned = re.sub(r"^(?:error|failed)\s*:\s*", "", _clean_status(message), flags=re.IGNORECASE)
    return f"{_FAILURE_PREFIX.get(normalized_name(tool), 'Error: ')}{_one_line(cleaned or message, 200)}"


def has_result_content(payload: dict[str, Any]) -> bool:
    """False for an envelope that carries nothing but "success": rendering one
    produced the misleading "Tool completed without a structured result" card."""
    envelope = _envelope(payload)
    keys = ("result", "content", "stdout", "stderr", "message", "error", "error_code",
            "status", "exit_code", "return_code", "resolved_path", "path", "requested_path")
    return any(envelope.get(key) not in (None, "", [], {}) for key in keys)


def is_groupable(tool: str) -> bool:
    return normalized_name(tool) in _GROUP_CATEGORY


def _envelope(payload: dict[str, Any]) -> dict[str, Any]:
    result = payload.get("result") if isinstance(payload.get("result"), dict) else payload
    return result if isinstance(result, dict) else {}


def _inner(envelope: dict[str, Any]) -> Any:
    """The tool's own result value (the envelope's "result", else the envelope)."""
    return envelope.get("result", envelope)


def _failure_text(envelope: dict[str, Any], inner: Any) -> str:
    holder = inner if isinstance(inner, dict) else envelope
    for key in ("message", "error", "stderr"):
        value = str(holder.get(key) or envelope.get(key) or "").strip()
        if value:
            return value
    if isinstance(inner, str) and inner.strip():
        return inner.strip()
    code = holder.get("return_code", holder.get("exit_code"))
    return f"Command failed with exit code {code}" if code not in (None, 0) else "Tool operation failed"


def _is_failure(envelope: dict[str, Any], inner: Any) -> bool:
    if envelope.get("success") is False or envelope.get("ok") is False:
        return True
    if isinstance(inner, dict):
        if inner.get("success") is False or inner.get("ok") is False:
            return True
        status = str(inner.get("status") or "").lower()
        if status in {"failed", "error", "not_found", "permission_denied", "timed_out", "cancelled"}:
            return True
        if inner.get("error_code"):
            return True
    return False


def _count_lines(text: str) -> int:
    if not text:
        return 0
    return text.count("\n") + (0 if text.endswith("\n") else 1)


def _clean_status(text: str) -> str:
    return _LEADING_STATUS_GLYPH.sub("", str(text or "")).strip()


def _output_lines(text: str) -> list[str]:
    """First few lines of command output, each trimmed, with a "+N lines" tail."""
    lines = [ln.rstrip() for ln in str(text or "").strip("\n").split("\n")]
    lines = [ln for ln in lines if ln.strip()] or []
    shown = [_one_line(ln, _MAX_RESULT_LINE_CHARS) for ln in lines[:_MAX_RESULT_LINES]]
    if len(lines) > _MAX_RESULT_LINES:
        extra = len(lines) - _MAX_RESULT_LINES
        shown.append(f"… +{extra} line{'s' if extra != 1 else ''}")
    return shown


def summarize_result(
    tool: str, payload: dict[str, Any], arguments: Optional[dict[str, Any]] = None,
) -> tuple[list[str], bool]:
    """The `⎿` lines for a completed tool call, and whether it FAILED."""
    name = normalized_name(tool)
    args = arguments if isinstance(arguments, dict) else {}
    envelope = _envelope(payload)
    inner = _inner(envelope)

    if name in {"execute_command", "run_command", "remote_exec"} and isinstance(inner, dict):
        code = inner.get("return_code", inner.get("exit_code"))
        stdout = str(inner.get("stdout") or inner.get("content") or "")
        stderr = str(inner.get("stderr") or "")
        if code not in (None, 0):
            return [f"Exit code {code}", *_output_lines(stderr or stdout)], True
        if inner.get("success") is False or envelope.get("success") is False:
            # Failed without an exit code (timed out, blocked, refused): the error text.
            body = _output_lines(stderr or stdout)
            return ([f"Error: {body[0]}", *body[1:]] if body else ["Command failed"]), True
        body = _output_lines(stdout or stderr)
        return (body or ["(No output)"]), False

    if _is_failure(envelope, inner):
        return [f"Error: {_one_line(_failure_text(envelope, inner), 200)}"], True

    if name == "read_file":
        text = inner if isinstance(inner, str) else str(inner.get("content") or "") if isinstance(inner, dict) else ""
        count = _count_lines(text)
        return [f"Read {count} line{'s' if count != 1 else ''}" if count else "Read 0 lines (empty file)"], False

    if name == "list_directory" and isinstance(inner, list):
        return [f"Listed {len(inner)} entr{'y' if len(inner) == 1 else 'ies'}"], False

    if name in {"search_code", "search_files", "grep_files", "find_references"} and isinstance(inner, list):
        if not inner:
            return ["No matches found"], False
        files = {str(item.get("file") or item.get("path") or "") for item in inner if isinstance(item, dict)}
        files.discard("")
        text = f"Found {len(inner)} match{'es' if len(inner) != 1 else ''}"
        if len(files) > 1:
            text += f" in {len(files)} files"
        return [text], False

    if name == "glob_files" and isinstance(inner, list):
        return [f"Found {len(inner)} file{'s' if len(inner) != 1 else ''}"], False

    if name == "ask_user_question" and isinstance(inner, str):
        # The answers are the point: show every "· question → answer" line.
        rows = [_one_line(line, 200) for line in inner.strip().split("\n") if line.strip()]
        return (rows[:9] or ["No answer given"]), False

    if name in {"write_file", "create_file"}:
        content = args.get("content")
        path = str(args.get("path") or args.get("file_path") or "")
        if isinstance(content, str) and path:
            count = _count_lines(content)
            return [f"Wrote {count} line{'s' if count != 1 else ''} to {path}"], False

    if name in {"edit_file", "file_edit", "update_file"}:
        path = str(args.get("path") or args.get("file_path") or "")
        if path:
            return [f"Edited {path}"], False

    if name in {"web_fetch", "fetch_url", "fetch"} and isinstance(inner, dict):
        status = inner.get("status") or inner.get("status_code")
        size = inner.get("bytes") or inner.get("size")
        if status or size:
            size_text = ""
            if isinstance(size, (int, float)) and size:
                size_text = f"{size / 1000:.1f}KB" if size >= 1000 else f"{int(size)}B"
            parts = ["Received"] + ([size_text] if size_text else []) + ([f"({status} OK)"] if status else [])
            return [" ".join(parts)], False

    if name == "web_search" and isinstance(inner, list):
        return [f"Found {len(inner)} result{'s' if len(inner) != 1 else ''}"], False

    # Generic fallbacks: never leave a call without a result line.
    if isinstance(inner, str):
        text = _clean_status(inner)
        lines = _output_lines(text)
        return (lines or ["Done"]), False
    if isinstance(inner, list):
        return [f"{len(inner)} item{'s' if len(inner) != 1 else ''}"], False
    if isinstance(inner, dict):
        message = _clean_status(str(inner.get("message") or inner.get("content") or ""))
        if message:
            return _output_lines(message), False
    return ["Done"], False
