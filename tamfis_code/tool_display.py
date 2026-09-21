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
    "read_archive": "Read archive",
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
    "read_archive": ("Read", "archive", "archives"),
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
    "read_archive": "Read failed: ",
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

    if name in {"read_file", "read_archive"}:
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


# ---------------------------------------------------------------------------------------------
# Codex-style activity blocks
#
#   • Explored
#     └ Read README.md, index.ts (×2), campaign-engine.ts
#       Search campaign|scheduler in worker.ts
#       List src
#
#   • Ran systemctl status tamfisseo --no-pager -l 2>&1 | sed -n '1,100p'; systemctl cat
#     │ tamfisseo 2>&1 | sed -n '1,160p'; journalctl -u tamfisseo --since '2026-09-20'
#     │ … +1 lines
#     └ ● tamfisseo.service - TamfisSEO Pro v3 - Enterprise SEO Automation
#          Loaded: loaded (/etc/systemd/system/tamfisseo.service; enabled)
#       … +134 lines (Ctrl+T for full output)
#
# Pure functions again: they return plain text (with a role per line) and the renderer only
# decides colours. The "(Ctrl+T …)" hint is only ever emitted for output that was actually cut,
# and the full output is what Ctrl+T shows (see render.TOOL_TRANSCRIPT).
# ---------------------------------------------------------------------------------------------

TRANSCRIPT_HINT = "Ctrl+T for full output"
_MAX_COMMAND_LINES = 3
_PREVIEW_LINES_WHEN_CUT = 2
_SHOW_ALL_UP_TO = 4

_EDIT_VERBS = {
    "write_file": "Wrote", "create_file": "Wrote",
    "edit_file": "Edited", "file_edit": "Edited", "update_file": "Edited",
    "web_fetch": "Fetched", "fetch_url": "Fetched", "fetch": "Fetched",
    "web_search": "Searched the web for",
}
_SEARCH_TOOLS = {"search_code", "search_files", "grep_files"}
_COMMAND_TOOLS = {"execute_command", "run_command", "remote_exec"}


def block_verb(tool: str) -> str:
    name = normalized_name(tool)
    return _EDIT_VERBS.get(name) or display_name(tool)


def is_command_tool(tool: str) -> bool:
    return normalized_name(tool) in _COMMAND_TOOLS


def _wrap(text: str, width: int) -> list[str]:
    """Hard-wrap each physical line of `text` to `width` (long tokens are split)."""
    import textwrap

    width = max(12, width)
    rows: list[str] = []
    for physical in str(text or "").split("\n"):
        wrapped = textwrap.wrap(
            physical, width=width, break_long_words=True, break_on_hyphens=False,
            replace_whitespace=False, drop_whitespace=True,
        )
        rows.extend(wrapped or [""])
    while rows and rows[-1] == "":
        rows.pop()
    return rows or [""]


def _cut(text: str, width: int) -> str:
    text = str(text).rstrip()
    return text if len(text) <= width else text[: max(1, width - 1)].rstrip() + "…"


def _pluralise(count: int, word: str = "line") -> str:
    return f"{count} {word}{'s' if count != 1 else ''}"


def command_output(payload: dict[str, Any]) -> tuple[str, Optional[int], bool]:
    """(full output text, exit code, failed) for a finished command tool call."""
    envelope = _envelope(payload)
    inner = _inner(envelope)
    if isinstance(inner, dict):
        code = inner.get("return_code", inner.get("exit_code"))
        stdout = str(inner.get("stdout") or inner.get("content") or "")
        stderr = str(inner.get("stderr") or "")
        text = "\n".join(part for part in (stdout.rstrip("\n"), stderr.rstrip("\n")) if part)
        failed = code not in (None, 0) or inner.get("success") is False or envelope.get("success") is False
        return text, (code if isinstance(code, int) else None), bool(failed)
    if _is_failure(envelope, inner):
        return _failure_text(envelope, inner), None, True
    return (inner if isinstance(inner, str) else ""), None, False


def ran_block(
    command: str, output: str, *, width: int, exit_code: Optional[int] = None, failed: bool = False,
) -> list[tuple[str, str]]:
    """The "• Ran …" record as (role, text) rows.

    roles: head (first row; the renderer bolds its verb), cmd (wrapped command continuation),
    out (output preview), err (failure preview), more ("… +N lines" tails)."""
    body = max(20, width - 6)
    command_rows = _wrap(redact_secrets(str(command or "").strip()), body)
    rows: list[tuple[str, str]] = [("head", "Ran " + command_rows[0])]
    rows.extend(("cmd", row) for row in command_rows[1:_MAX_COMMAND_LINES])
    if len(command_rows) > _MAX_COMMAND_LINES:
        rows.append(("cmd", f"… +{_pluralise(len(command_rows) - _MAX_COMMAND_LINES)}"))

    role = "err" if failed else "out"
    lines = [ln.rstrip() for ln in str(output or "").strip("\n").split("\n")] if str(output or "").strip() else []
    if failed and exit_code not in (None, 0):
        rows.append(("err", f"Exit code {exit_code}"))
    if not lines:
        if not (failed and exit_code not in (None, 0)):
            rows.append(("out", "(no output)"))
        return rows
    if len(lines) <= _SHOW_ALL_UP_TO:
        rows.extend((role, _cut(line, body)) for line in lines)
    else:
        rows.extend((role, _cut(line, body)) for line in lines[:_PREVIEW_LINES_WHEN_CUT])
        hidden = len(lines) - _PREVIEW_LINES_WHEN_CUT
        rows.append(("more", f"… +{_pluralise(hidden)} ({TRANSCRIPT_HINT})"))
    return rows


def output_was_cut(output: str) -> bool:
    text = str(output or "").strip("\n")
    return bool(text) and text.count("\n") + 1 > _SHOW_ALL_UP_TO


_FAILED_VERBS = {"Wrote": "Failed to write", "Edited": "Failed to edit", "Fetched": "Failed to fetch"}


def tool_block(
    tool: str, arguments: Optional[dict[str, Any]], result_lines: list[str], *, width: int,
    failed: bool = False,
) -> list[tuple[str, str]]:
    """Any other finished tool call: "• Edited path" + a └ summary line. A FAILED call never wears
    the success verb ("Edited"): it reads "Failed to edit path"."""
    body = max(20, width - 6)
    target = display_target(tool, arguments, limit=body)
    verb = block_verb(tool)
    if failed:
        verb = _FAILED_VERBS.get(verb, verb)
    head = f"{verb} {target}".rstrip()
    rows: list[tuple[str, str]] = [("head", _cut(head, width - 4))]
    rows.extend(("out", _cut(line, body)) for line in (result_lines or ["Done"]))
    return rows


def explored_lines(items: list[dict[str, Any]], width: int) -> list[str]:
    """The body of the "• Explored" block: consecutive reads merged into one "Read a, b, c" line,
    each search / listing on its own line. A file read more than once shows "(×N)" -- a re-read loop is
    exactly what a reader wants to notice."""
    body = max(20, width - 4)
    entries: list[str] = []
    reads: dict[str, int] = {}

    def flush_reads() -> None:
        if reads:
            names = [f"{name} (×{count})" if count > 1 else name for name, count in reads.items()]
            entries.append("Read " + ", ".join(names))
            reads.clear()

    for item in items:
        tool = normalized_name(item.get("tool") or "")
        args = item.get("args") if isinstance(item.get("args"), dict) else {}
        if tool in {"read_file", "read_archive"}:
            path = str(args.get("path") or args.get("file_path") or item.get("short") or "file")
            name = path.rstrip("/").rsplit("/", 1)[-1] or path
            reads[name] = reads.get(name, 0) + 1
            continue
        flush_reads()
        path = str(args.get("path") or args.get("directory") or "")
        shown_path = "" if path in {"", ".", "./"} else path
        if tool == "list_directory":
            entries.append(f"List {shown_path or '.'}")
        elif tool in _SEARCH_TOOLS:
            pattern = _one_line(args.get("query") or args.get("pattern") or "", 60)
            entries.append(f"Search {pattern}" + (f" in {shown_path}" if shown_path else ""))
        elif tool == "find_references":
            entries.append(f"References {_one_line(args.get('symbol') or args.get('query') or args.get('name') or '', 60)}"
                           + (f" in {shown_path}" if shown_path else ""))
        elif tool == "glob_files":
            entries.append(f"Glob {_one_line(args.get('pattern') or args.get('query') or '', 60)}")
        elif tool == "get_git_info":
            entries.append("Git status")
        else:
            entries.append(f"{display_name(tool)} {display_target(tool, args, limit=body)}".rstrip())
    flush_reads()
    rows: list[str] = []
    for entry in entries:
        rows.extend(_wrap(entry, body))
    return rows
