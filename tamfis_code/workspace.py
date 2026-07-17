"""Resolves the launch directory into a Remote session, idempotently.

Implements Phase 21's "CLI entry behaviour": running `tamfis-code` from a
directory treats that directory as workspace_root, never scans the wider
box, and reuses (rather than re-creates) the local server/session pair for
repeated runs from the same directory.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .api_client import RemoteAPIClient
from . import state as local_state

REUSABLE_STATUSES = {"idle", "active"}
INSTRUCTION_NAMES = {"AGENTS.md", "CLAUDE.md", "CODEX.md", "CONTRIBUTING.md", "README.md"}
REPORT_RE = re.compile(
    r"(report|audit|analysis|architecture|implementation|codex|claude|tamfis[-_ ]?code|"
    r"terminal|agent|roadmap|plan|findings|gaps|status|handover|review)", re.I,
)
REPORT_SUFFIXES = {".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".log", ".pdf", ".docx"}
IGNORED_PARTS = {".git", "node_modules", ".venv", "venv", "dist", "build", "__pycache__"}
DISPOSABLE_UNTRACKED_PARTS = {
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".tox", ".nox", ".coverage", "htmlcov", ".DS_Store",
}
MAX_INDEX_FILES = 20_000


@dataclass
class WorkspaceContext:
    session_id: int
    workspace_root: str
    # None for a purely local session (see resolve_local_workspace) -- only
    # meaningful for the legacy Remote-backend path (resolve_workspace),
    # which this field exists to support until that path is retired.
    server_id: Optional[int] = None


def _next_local_session_id() -> int:
    known = local_state.all_known_session_ids()
    return (max(known) + 1) if known else 1


def resolve_local_workspace(cwd: Optional[Path] = None, *, discover: bool = True) -> WorkspaceContext:
    """Resolve a launch directory into a purely local session -- no network
    calls, no RemoteAPIClient, no remote-assigned session/server id.

    Reuses a prior session for the same workspace_root (matched by
    `primary_workspace`, the same durable-root concept `resolve_workspace`
    already used) rather than minting a new one on every invocation; a
    fresh id is allocated via `_next_local_session_id()` (one past the
    highest session id state.py already knows about) when no match exists.
    """
    workspace_root = str((cwd or Path.cwd()).resolve())

    local_match = next((
        sid for sid in reversed(local_state.all_known_session_ids())
        if local_state.get_session_state(sid).primary_workspace == workspace_root
    ), None)
    session_id = local_match if local_match is not None else _next_local_session_id()

    local_state.save_session_state(session_id, workspace_root=workspace_root)
    if discover:
        discover_local_repository(session_id, Path(workspace_root))
    return WorkspaceContext(session_id=session_id, workspace_root=workspace_root)


async def _get_or_create_local_server(client: RemoteAPIClient) -> dict:
    servers = await client.list_servers()
    for server in servers:
        if server.get("transport_type") == "local":
            return server
    created = await client.register_local_server("local-vps")
    return created["server"]


async def resolve_workspace(
    client: RemoteAPIClient, cwd: Optional[Path] = None, *, discover: bool = True,
) -> WorkspaceContext:
    workspace_root = str((cwd or Path.cwd()).resolve())

    server = await _get_or_create_local_server(client)
    server_id = server["id"]

    # A session can move to an explicitly approved sibling workspace. Reuse
    # it by its durable primary root rather than creating a new conversation
    # just because its current working directory changed.
    local_match = next((
        sid for sid in reversed(local_state.all_known_session_ids())
        if local_state.get_session_state(sid).primary_workspace == workspace_root
    ), None)
    if local_match is not None:
        try:
            detail = await client.get_session(local_match)
        except Exception:
            detail = None
        if detail and str(detail.get("status", "")).lower() in REUSABLE_STATUSES:
            state = local_state.get_session_state(local_match)
            current = str(detail.get("working_directory") or state.current_working_directory or workspace_root)
            local_state.save_session_state(
                local_match, workspace_root=workspace_root,
                current_working_directory=current,
            )
            if discover:
                discover_local_repository(local_match, Path(current))
            return WorkspaceContext(session_id=local_match, server_id=server_id, workspace_root=current)

    sessions = await client.list_sessions()
    existing = next(
        (
            s for s in sessions
            if s.get("server_id") == server_id
            and s.get("working_directory") == workspace_root
            and str(s.get("status", "")).lower() in REUSABLE_STATUSES
        ),
        None,
    )
    if existing is not None:
        local_state.save_session_state(existing["id"], workspace_root=workspace_root)
        if discover:
            discover_local_repository(existing["id"], Path(workspace_root))
        return WorkspaceContext(session_id=existing["id"], server_id=server_id, workspace_root=workspace_root)

    created = await client.create_session(server_id, workspace_root)
    local_state.save_session_state(created["id"], workspace_root=workspace_root)
    if discover:
        discover_local_repository(created["id"], Path(workspace_root))
    return WorkspaceContext(session_id=created["id"], server_id=server_id, workspace_root=workspace_root)


async def context_from_session(client: RemoteAPIClient, session_id: int) -> WorkspaceContext:
    """Builds a WorkspaceContext from an existing session_id -- used by
    `/resume` and `tamfis-code resume`. Raises RemoteAPIError (404) if the
    session does not exist or is not owned by the authenticated user, same
    ownership check every other Remote endpoint already enforces."""
    detail = await client.get_session(session_id)
    workspace_root = detail.get("working_directory") or ""
    local_state.save_session_state(session_id, workspace_root=workspace_root)
    if workspace_root:
        discover_local_repository(session_id, Path(workspace_root))
    return WorkspaceContext(
        session_id=detail["id"],
        server_id=detail["server_id"],
        workspace_root=workspace_root,
    )


async def find_resumable_session(client: RemoteAPIClient, *, exclude_session_id: Optional[int] = None) -> Optional[dict]:
    """Most recent non-closed session other than the one already active --
    `list_sessions()` is ordered by created_at desc server-side."""
    sessions = await client.list_sessions()
    for session in sessions:
        if exclude_session_id is not None and session.get("id") == exclude_session_id:
            continue
        if str(session.get("status", "")).lower() in REUSABLE_STATUSES:
            return session
    return None


def _git(root: Path, *args: str) -> str:
    """Run a bounded, argument-safe Git query (never through a shell)."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True,
            timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _indexable_files(root: Path) -> list[Path]:
    output = _git(root, "ls-files", "-co", "--exclude-standard")
    if output:
        return [root / item for item in output.splitlines()[:MAX_INDEX_FILES]]
    found: list[Path] = []
    for path in root.rglob("*"):
        if any(part in IGNORED_PARTS for part in path.parts):
            continue
        if path.is_file():
            found.append(path)
            if len(found) >= MAX_INDEX_FILES:
                break
    return found


def blocking_dirty_files(status_lines: list[str]) -> list[str]:
    """Return entries that may represent user-authored work.

    Tracked changes always block execute mode. Only untracked, well-known
    disposable test/interpreter artefacts are ignored.
    """
    blocking: list[str] = []
    for raw in status_lines:
        line = str(raw).rstrip()
        if not line:
            continue
        status = line[:2]
        path_text = line[3:].strip() if len(line) > 3 else ""
        parts = {part for part in Path(path_text).parts if part not in {".", ""}}
        if status == "??" and parts.intersection(DISPOSABLE_UNTRACKED_PARTS):
            continue
        blocking.append(line)
    return blocking


def _report_title(path: Path) -> str:
    if path.suffix.lower() in {".md", ".txt"}:
        try:
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[:40]:
                if line.lstrip().startswith("#"):
                    return line.lstrip("# ").strip()
        except OSError:
            pass
    return path.stem.replace("_", " ").replace("-", " ")


def discover_local_repository(session_id: int, workspace_root: Path, *, force: bool = False) -> dict[str, Any]:
    """Cache local CLI context until Git HEAD or dirty-file count changes.

    The model-facing server snapshot remains authoritative. This lightweight
    index powers offline `/context` and `/reports`, and records the worktree
    state that existed before a task begins.
    """
    root_text = _git(workspace_root, "rev-parse", "--show-toplevel")
    root = Path(root_text).resolve() if root_text else workspace_root.resolve()
    branch = _git(root, "branch", "--show-current") or None
    head = _git(root, "rev-parse", "HEAD") or "no-head"
    dirty_lines = _git(root, "status", "--short").splitlines()
    fingerprint = hashlib.sha256(
        f"{root}|{head}|{'|'.join(dirty_lines)}".encode()
    ).hexdigest()
    current = local_state.get_session_state(session_id)
    if not force and current.discovery_fingerprint == fingerprint and current.repository_context:
        return current.repository_context

    files = _indexable_files(root)
    instructions: list[str] = []
    reports: list[dict[str, Any]] = []
    for path in files:
        if path.name in INSTRUCTION_NAMES:
            instructions.append(str(path))
        if path.suffix.lower() in REPORT_SUFFIXES and REPORT_RE.search(path.name):
            try:
                modified = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
            except OSError:
                modified = ""
            reports.append({
                "path": str(path), "title": _report_title(path), "modified_at": modified,
                "scope": "repository", "verification": "unverified",
            })

    context = {
        "repository_root": str(root), "working_directory": str(workspace_root.resolve()),
        "branch": branch, "head": None if head == "no-head" else head,
        "dirty": bool(dirty_lines), "dirty_files": dirty_lines[:100],
        "blocking_dirty_files": blocking_dirty_files(dirty_lines)[:100],
        "instruction_files": sorted(instructions), "indexed_file_count": len(files),
        "indexed_at": datetime.now(timezone.utc).isoformat(),
    }
    local_state.save_session_state(
        session_id, repository_root=str(root), current_working_directory=str(workspace_root.resolve()),
        active_branch=branch, repository_context=context,
        discovered_reports=sorted(reports, key=lambda item: item["modified_at"], reverse=True),
        discovery_fingerprint=fingerprint,
    )
    return context


# Total instruction-file content budget for the system prompt -- generous
# enough for a real AGENTS.md/CLAUDE.md, bounded so a huge README doesn't
# blow the context window before the actual objective is even sent.
MAX_INSTRUCTION_CHARS = 12_000


def build_system_prompt(session_id: int, workspace_root: Path, *, force_discovery: bool = False) -> str:
    """Build the standalone agent loop's system prompt from real local
    workspace awareness (git branch/dirty status, instruction file
    contents) -- replacing the context tamgpt6 used to assemble server-side.
    Reuses discover_local_repository's existing fingerprint-cached scan
    rather than re-walking the repo on every call.
    """
    context = discover_local_repository(session_id, workspace_root, force=force_discovery)
    lines = [
        "You are a coding agent working directly in a real local repository via tool calls. "
        "Verify with tools before claiming something is done or correct. Prefer minimal, "
        "targeted changes over broad rewrites.",
        "Never describe a fix, edit, or command in your written response without actually "
        "calling the corresponding tool (write_file/edit_file/execute_command/etc) in this "
        "same turn. A code block in your text is not a change -- if the task requires "
        "changing a file, call the tool that changes it before you say you've changed it. "
        "If you're unsure which file actually defines something, use read_file or "
        "search_code to find it first; do not guess a file's contents from its name.",
        "Before checking whether any local service is 'healthy' or 'running', you must "
        "first find its REAL configured port -- do not use 8080/3000/5000/8000 or any "
        "other common default unless you have actually confirmed that's the real one. "
        "Concrete required steps, in order: (1) search_code for \"port\" (or read "
        "config.yaml/.env/docker-compose.yml/package.json, whichever exists) to find the "
        "actual configured port; (2) only then curl/request that exact port. Getting ANY "
        "HTTP response back from a guessed port is NOT evidence the intended service is "
        "healthy -- an entirely different, unrelated process can easily be listening on a "
        "common default port instead. The same applies to any other environment-specific "
        "value (host, container/process ID, file path, env var): find the real one via a "
        "tool call before using it, never assume it from a common default.",
        "Before running any install/build/start command against a project (or one component "
        "of a multi-component stack), first find out what kind of project it actually is -- "
        "list_directory it and look for package.json (Node/npm), pyproject.toml/"
        "requirements.txt (Python), go.mod (Go), Cargo.toml (Rust), Dockerfile/"
        "docker-compose.yml, etc. Do not default to npm install/npm start just because a "
        "component is called a 'backend' or is mentioned alongside a Node frontend -- run "
        "the command that actually matches what's really there.",
        f"Workspace root: {context['working_directory']}",
    ]
    # discover_local_repository always sets repository_root (falling back to
    # workspace_root itself when `git rev-parse --show-toplevel` fails) --
    # `head` is the real "is this actually a Git repo" signal, since it's
    # only None when that git call failed (see its `"no-head"` sentinel).
    if context.get("head"):
        lines.append(f"Git repository root: {context['repository_root']}  branch: {context.get('branch') or '(detached HEAD)'}")
        if context.get("dirty"):
            lines.append(
                f"The working tree already has {len(context.get('dirty_files') or [])} uncommitted change(s) -- "
                "be careful not to conflate your own edits with pre-existing ones when reporting what changed."
            )
    else:
        lines.append("This directory is not a Git repository.")

    instruction_files = context.get("instruction_files") or []
    remaining_budget = MAX_INSTRUCTION_CHARS
    instruction_blocks = []
    for path_str in instruction_files:
        if remaining_budget <= 0:
            break
        try:
            content = Path(path_str).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        snippet = content[:remaining_budget]
        remaining_budget -= len(snippet)
        truncated = " (truncated)" if len(snippet) < len(content) else ""
        instruction_blocks.append(f"--- {path_str}{truncated} ---\n{snippet}")
    if instruction_blocks:
        lines.append("\nProject instructions found in this repository:\n" + "\n\n".join(instruction_blocks))

    return "\n".join(lines)
