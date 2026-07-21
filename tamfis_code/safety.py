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
import shlex
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from . import state as local_state

RISK_READ_ONLY = "read_only"
RISK_MEDIUM = "medium"
RISK_DANGEROUS = "dangerous"

READ_ONLY_TOOLS = {"read_file", "list_directory", "search_code", "find_references", "get_git_info", "ask_user_question"}
MUTATING_FILE_TOOLS = {"write_file", "edit_file", "extract_archive", "repackage_archive"}

MAX_MUTATION_HISTORY = 200

# Patterns that make a shell command dangerous regardless of approval_policy
# leniency -- destructive, irreversible, or credential-exposing. This is a
# heuristic allowlist-of-concerns, not a sandbox: it reduces the chance of an
# unreviewed catastrophic command slipping through "safe"/"accept-edits"
# policies, it does not replace real sandboxing (out of scope here, see the
# rebuild plan's explicit-non-goals section).
_DANGEROUS_COMMAND_PATTERNS = [
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


_RM_RE = re.compile(r"\brm\b")


def _is_dangerous_rm(text: str) -> bool:
    """True when an `rm` invocation combines a recursive flag with a force
    flag, in any spelling/order/combination -- `-rf`, `-fr`, `-r -f`,
    `-f -r`, `--recursive --force`, `-r --force`, etc.

    The original check was a single regex requiring the letters "r" and "f"
    adjacent inside one combined short flag cluster (`-rf`/`-fr` and
    letter-order variants of *that*), which missed the equally common
    separated-flag (`rm -r -f`) and long-flag (`rm --recursive --force`)
    spellings entirely -- confirmed live, both silently classified as only
    `medium` risk. Only the r+f *combination* is treated as dangerous here,
    matching the original threshold -- `rm -r somedir` or `rm -f file` alone
    stay `medium`, unchanged.
    """
    if not _RM_RE.search(text):
        return False
    has_recursive = False
    has_force = False
    for token in text.split():
        if token in ("-r", "-R", "--recursive"):
            has_recursive = True
        elif token in ("-f", "--force"):
            has_force = True
        elif token.startswith("-") and not token.startswith("--"):
            letters = token[1:]
            if "r" in letters or "R" in letters:
                has_recursive = True
            if "f" in letters:
                has_force = True
    return has_recursive and has_force


def classify_command_risk(command: str) -> str:
    """Heuristic risk tier for an `execute_command` tool call."""
    text = command or ""
    if _is_dangerous_rm(text):
        return RISK_DANGEROUS
    for pattern in _DANGEROUS_COMMAND_PATTERNS:
        if pattern.search(text):
            return RISK_DANGEROUS
    if _is_read_only_command(text):
        return RISK_READ_ONLY
    return RISK_MEDIUM


_READ_ONLY_COMMANDS = {
    "find", "rg", "grep", "ls", "pwd", "head", "tail", "sort",
    "uniq", "wc", "stat", "file", "du", "tree", "realpath", "readlink",
}
_READ_ONLY_GIT_SUBCOMMANDS = {"status", "diff", "log", "show", "rev-parse", "ls-files", "grep"}


def _is_read_only_command(command: str) -> bool:
    """Conservatively recognize shell commands that only inspect state."""
    # Discard only the common stderr-to-/dev/null suffix used by discovery
    # commands. Every other redirection/control/substitution construct is
    # treated as mutating/unknown and therefore not allowed in read-only mode.
    normalized = re.sub(r"(?:^|\s)2?>\s*/dev/null(?:\s|$)", " ", command).strip()
    if not normalized or re.search(r"[;&|<>`]", normalized):
        return False
    if "$(`" in normalized or "$(" in normalized or "${" in normalized:
        return False
    try:
        argv = shlex.split(normalized)
    except ValueError:
        return False
    if not argv:
        return False
    executable = Path(argv[0]).name
    if executable == "git":
        return (
            len(argv) > 1
            and argv[1] in _READ_ONLY_GIT_SUBCOMMANDS
            and not any(
                arg in {"--ext-diff", "--textconv", "--open-files-in-pager"}
                or arg.startswith("--open-files-in-pager=")
                for arg in argv[2:]
            )
        )
    if executable not in _READ_ONLY_COMMANDS:
        return False
    if executable == "find" and any(
        arg in {"-delete", "-exec", "-execdir", "-ok", "-okdir"}
        or arg.startswith(("-fprint", "-fprintf", "-fls"))
        for arg in argv[1:]
    ):
        return False
    if executable == "rg" and any(arg == "--pre" or arg.startswith("--pre=") for arg in argv[1:]):
        return False
    if executable in {"sort", "tree"} and any(
        arg in {"-o", "--output"} or arg.startswith("--output=")
        for arg in argv[1:]
    ):
        return False
    return True


# Confirmed live: a model that has just read wp-config.php (or any file with
# embedded DB credentials) can turn straight around and paste the live
# password into an inline `mysql -pSECRET ...` invocation -- and that raw
# command string was rendered verbatim, cleartext, in both the "Running
# command · ..." status line and the approval panel's "Command:" field.
# This is a display/logging concern, not a risk-tier concern (redacting a
# risk-tier-medium command doesn't make it dangerous, it just stops the
# secret from being echoed anywhere a human or a log file can see it).
# Scoped to the shapes actually seen -- mysql/mariadb/psql-family `-p`,
# `--password=`/`--password `, and URL-embedded `user:pass@host` credentials
# -- not a general-purpose secret scanner.
_SQL_CLIENT_RE = re.compile(r"\b(mysql|mariadb|mysqldump|mysqladmin|psql|pg_dump|pg_restore)\b")
_INLINE_DASH_P_PASSWORD_RE = re.compile(r"(?<![\w-])-p(?!assword\b)\S+")
_LONG_PASSWORD_FLAG_RE = re.compile(r"--password[= ]\S+")
_URL_CREDENTIALS_RE = re.compile(r"://[^/\s:@]+:[^/\s@]+@")


def redact_secrets(command: str) -> str:
    """Mask plaintext credentials in a shell command before it is rendered,
    logged, or persisted anywhere."""
    text = command or ""
    if _SQL_CLIENT_RE.search(text):
        text = _INLINE_DASH_P_PASSWORD_RE.sub("-p***", text)
    text = _LONG_PASSWORD_FLAG_RE.sub("--password=***", text)
    text = _URL_CREDENTIALS_RE.sub("://***:***@", text)
    return text


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
        path_keys = {
            "extract_archive": ("destination", "path"),
            "repackage_archive": ("output_path", "source_dir"),
        }.get(name, ("path",))
        paths = [str(arguments.get(key) or "") for key in path_keys]
        if not all(paths):
            # extract_archive may omit destination; its handler derives a
            # workspace-local sibling name from the validated source.
            if name != "extract_archive" or not paths[-1]:
                return RISK_DANGEROUS
            paths = [paths[-1]]
        risks = [classify_path_risk(path, workspace_root) for path in paths]
        return RISK_DANGEROUS if RISK_DANGEROUS in risks else RISK_MEDIUM
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
    transaction_id: Optional[str] = None,
) -> dict[str, Any]:
    """Append a local mutation-ledger entry to SessionState.modified_files.

    Replaces the remote backend's ledger, which this client used to only
    ever *observe* via `file_mutation` SSE events (see render.py/runner.py's
    prior handling) -- now the tool handler that performs the write is the
    one that must record it, since there's no server doing that anymore.
    `original_content=None` means this mutation created the file (revert ==
    delete); otherwise it's the exact pre-mutation bytes needed to restore.

    `transaction_id` groups every mutation made during one turn (see
    MCPServer's constructor -- runner_local.py mints one per turn) so they
    can later be reverted together via revert_transaction(), instead of a
    multi-file turn only ever being revertible one mutation_id at a time
    with no way to even discover which ids belonged together.
    """
    diff_text = _unified_diff(path, original_content, new_content)
    added = sum(1 for line in diff_text.splitlines() if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in diff_text.splitlines() if line.startswith("-") and not line.startswith("---"))

    state = local_state.get_session_state(session_id)
    entry = {
        "mutation_id": f"mut_{uuid.uuid4().hex[:12]}",
        "transaction_id": transaction_id,
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


def revert_transaction(session_id: int, transaction_id: str) -> dict[str, Any]:
    """Revert every not-yet-reverted mutation recorded under one turn's
    transaction_id, in reverse chronological order -- undoing the most
    recent change to a file before an earlier one to the same file, the
    only order guaranteed correct since a later mutation in the turn may
    have been made against the file state an earlier one in the same turn
    left behind.

    Stops at the first failure rather than pressing on into other files --
    the returned `remaining` list is exactly what still needs attention, so
    what happened is never ambiguous even though this cannot be a true
    atomic filesystem transaction (an already-reverted write cannot be
    un-reverted if a later one in the same batch then fails).

    Raises ValueError if no mutation in this session carries this
    transaction_id at all (as opposed to "all already reverted", which
    returns cleanly with an empty `reverted` list).
    """
    state = local_state.get_session_state(session_id)
    all_matches = [m for m in state.modified_files if m.get("transaction_id") == transaction_id]
    if not all_matches:
        raise ValueError(f"No recorded mutations with transaction id {transaction_id!r} in this session")
    pending = [m for m in all_matches if m.get("revert_status") != "reverted"]
    pending.sort(key=lambda m: m.get("created_at", ""), reverse=True)

    reverted: list[str] = []
    for entry in pending:
        mutation_id = str(entry["mutation_id"])
        try:
            revert_mutation(session_id, mutation_id)
        except (OSError, ValueError) as exc:
            remaining = [str(m["mutation_id"]) for m in pending if str(m["mutation_id"]) not in reverted]
            return {"transaction_id": transaction_id, "reverted": reverted, "remaining": remaining, "error": str(exc)}
        reverted.append(mutation_id)
    return {"transaction_id": transaction_id, "reverted": reverted, "remaining": [], "error": None}
