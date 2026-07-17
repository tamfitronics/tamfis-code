"""Local risk classification and mutation-ledger recording for the
standalone agent loop.

Before this module existed, risk classification, approval gating, and the
file-mutation ledger all lived server-side in the TamfisGPT Remote Workspace
backend (tamgpt6) -- this CLI only ever *rendered* a server-supplied
`risk_level` and *responded to* a server-emitted `approval_required` event
(see runner.py's `resolve_approval_decision`, which is a pure policy-vs-risk
decision table and was never a classifier). Now that tamfis-code runs its
own agent loop with no remote backend behind it, something has to take over
both of those jobs locally -- that's what this module is for.
"""

from __future__ import annotations

import difflib
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from . import state as local_state

RISK_READ_ONLY = "read_only"
RISK_MEDIUM = "medium"
RISK_DANGEROUS = "dangerous"

READ_ONLY_TOOLS = {"read_file", "list_directory", "search_code", "get_git_info"}
MUTATING_FILE_TOOLS = {"write_file", "edit_file"}

MAX_MUTATION_HISTORY = 200

# Patterns that make a shell command dangerous regardless of approval_policy
# leniency -- destructive, irreversible, or credential-exposing. This is a
# heuristic allowlist-of-concerns, not a sandbox: it reduces the chance of an
# unreviewed catastrophic command slipping through "safe"/"accept-edits"
# policies, it does not replace real sandboxing (out of scope here, see the
# rebuild plan's explicit-non-goals section).
_DANGEROUS_COMMAND_PATTERNS = [
    re.compile(r"\brm\s+(-\w*r\w*f\w*|-\w*f\w*r\w*)\b"),  # rm -rf / rm -fr and letter-order variants
    re.compile(r"\bgit\s+push\b[^\n]*--force\b"),
    re.compile(r"\bgit\s+reset\s+--hard\b"),
    re.compile(r"\bgit\s+clean\s+-\w*f\w*d?\b"),
    re.compile(r"\bsudo\b"),
    re.compile(r"\bchmod\s+-R\s+777\b"),
    re.compile(r"\bdd\s+if="),
    re.compile(r"\bmkfs(\.\w+)?\b"),
    re.compile(r":\(\)\s*\{[^}]*\}\s*;\s*:"),  # classic fork bomb
    re.compile(r"\b(curl|wget)\b[^\n]*\|\s*(sudo\s+)?(ba)?sh\b"),
    re.compile(r">\s*/dev/sd[a-z]\b"),
    re.compile(r"\.ssh/(id_|authorized_keys)|\.aws/credentials|(^|\s)\.env\b"),
    re.compile(r"\bshutdown\b|\breboot\b|\bhalt\b"),
]


def classify_command_risk(command: str) -> str:
    """Heuristic risk tier for an `execute_command` tool call."""
    text = command or ""
    for pattern in _DANGEROUS_COMMAND_PATTERNS:
        if pattern.search(text):
            return RISK_DANGEROUS
    return RISK_MEDIUM


def classify_path_risk(path: str, workspace_root: str) -> str:
    """`dangerous` if the target resolves outside workspace_root, else `medium`.

    Mirrors the workspace-boundary check `mcp.py`'s `_write_file` never had
    (see the rebuild plan's Phase 2 goal) -- a path that escapes the
    workspace is exactly the case a local agent loop has no server-side
    backstop for anymore.
    """
    try:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = Path(workspace_root) / candidate
        resolved = candidate.resolve()
        root = Path(workspace_root).resolve()
    except (OSError, ValueError, RuntimeError):
        return RISK_DANGEROUS
    if resolved != root and root not in resolved.parents:
        return RISK_DANGEROUS
    return RISK_MEDIUM


def classify_tool_call_risk(name: str, arguments: dict[str, Any], *, workspace_root: str) -> str:
    """Single entry point the standalone loop consults before executing any
    tool call -- feeds `runner.py`'s existing `resolve_approval_decision`
    exactly the way a server-supplied `risk_level` used to."""
    if name in READ_ONLY_TOOLS:
        return RISK_READ_ONLY
    if name in MUTATING_FILE_TOOLS:
        path = str(arguments.get("path") or "")
        return classify_path_risk(path, workspace_root) if path else RISK_DANGEROUS
    if name == "execute_command":
        return classify_command_risk(str(arguments.get("command") or ""))
    if name == "browser":
        return RISK_MEDIUM
    return RISK_DANGEROUS  # unknown tool name -- fail safe, never default to permissive


def _unified_diff(path: str, original_content: Optional[str], new_content: Optional[str]) -> str:
    original_lines = (original_content or "").splitlines(keepends=True)
    new_lines = (new_content or "").splitlines(keepends=True)
    return "".join(difflib.unified_diff(original_lines, new_lines, fromfile=f"a/{path}", tofile=f"b/{path}"))


def record_mutation(
    session_id: int, *, path: str, operation: str,
    original_content: Optional[str], new_content: Optional[str],
) -> dict[str, Any]:
    """Append a local mutation-ledger entry to SessionState.modified_files.

    Replaces the remote backend's ledger, which this client used to only
    ever *observe* via `file_mutation` SSE events (see render.py/runner.py's
    prior handling) -- now the tool handler that performs the write is the
    one that must record it, since there's no server doing that anymore.
    `original_content=None` means this mutation created the file (revert ==
    delete); otherwise it's the exact pre-mutation bytes needed to restore.
    """
    diff_text = _unified_diff(path, original_content, new_content)
    added = sum(1 for line in diff_text.splitlines() if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in diff_text.splitlines() if line.startswith("-") and not line.startswith("---"))

    state = local_state.get_session_state(session_id)
    entry = {
        "mutation_id": f"mut_{uuid.uuid4().hex[:12]}",
        "path": path,
        "operation": operation,
        "lines_added": added,
        "lines_removed": removed,
        "unified_diff": diff_text,
        "original_content": original_content,
        "revert_status": "none",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    state.modified_files = (state.modified_files + [entry])[-MAX_MUTATION_HISTORY:]
    local_state.save_session_state(session_id, modified_files=state.modified_files)
    return entry


def revert_mutation(session_id: int, mutation_id: str) -> dict[str, Any]:
    """Restore a file to its content before the given mutation, using the
    ledger's own stored pre-mutation snapshot -- no server round-trip."""
    state = local_state.get_session_state(session_id)
    entry = next((m for m in state.modified_files if m.get("mutation_id") == mutation_id), None)
    if entry is None:
        raise ValueError(f"No recorded mutation with id {mutation_id!r} in this session")
    if entry.get("revert_status") == "reverted":
        return entry

    path = Path(entry["path"])
    original_content = entry.get("original_content")
    if original_content is None:
        if path.exists():
            path.unlink()
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(original_content, encoding="utf-8")

    entry["revert_status"] = "reverted"
    local_state.save_session_state(session_id, modified_files=state.modified_files)
    return entry
