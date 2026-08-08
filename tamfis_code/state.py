"""Durable local per-session CLI state: last received event id/task id per session,
so `attach`/`logs --follow` can resume a stream without a full replay-from-
zero, and so a bare `tamfis-code agents`/`status` can show "what was I last
doing" without another round trip.

This is client-side bookkeeping ONLY -- the server (RemoteEvent/RemoteTask
tables) remains the single source of truth for everything that must survive
a lost or wiped local state file; losing this file just means the next
`attach`/`logs --follow` replays from sequence 0 instead of resuming exactly
where it left off, not that any task/event data is lost.

Reuses the same CONFIG_DIR credentials.json/config.toml already live in
(Phase 18's "current canonical equivalent" allowance), rather than
introducing a second state directory.
"""

from __future__ import annotations

import json
import os
import re
import stat
import sys
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

from .config import CONFIG_DIR

STATE_PATH = CONFIG_DIR / "state.json"
_VOLATILE_STATE: dict[tuple[str, int], "SessionState"] = {}


def _volatile_key(session_id: int) -> tuple[str, int]:
    return (str(STATE_PATH), int(session_id))
MAX_ACTION_HISTORY = 250
MAX_CHECKPOINTS = 50
MAX_SAVED_PLANS = 50
MAX_CONVERSATION_MESSAGES = 60
MAX_TURN_CHECKPOINT_MESSAGES = 80
# Keep the on-disk recovery record useful without allowing a long thread or a
# large tool result to turn every checkpoint into a multi-megabyte write.
MAX_MEMORY_MESSAGE_CHARS = 12000
MAX_MEMORY_TOTAL_CHARS = 180000

# Defense-in-depth caps for the other per-session list/dict fields. Several
# call sites already truncate what they pass in (e.g. validation_results to
# the last 100), but that only helps if every call site remembers to -- a
# single site that appends without slicing (this shipped for a while with
# `modified_files`, see put_session_state below) turns one long-running,
# file-touching audit into a session that grows forever. These are enforced
# centrally in put_session_state so no call site can reintroduce the bug.
MAX_MODIFIED_FILES = 300
MAX_INSPECTED_FILES = 400
MAX_VALIDATION_RESULTS = 150
MAX_UNRESOLVED_ISSUES = 150
MAX_PENDING_ACTIONS = 150
MAX_DISCOVERED_SERVICES = 200
MAX_DISCOVERED_REPORTS = 200
MAX_DISCOVERED_SYMBOL_FILES = 400
MAX_QUEUED_INSTRUCTIONS = 200

# A long-lived install accumulates one state.json entry per session ever
# created (80 sessions measured live at 15.7MB total, several individual
# sessions multiple MB each) since nothing ever removed old ones. Every
# `put_session_state` call -- which fires many times per turn during an
# audit -- read-modify-wrote *all* of them, so the file kept growing and
# every single write got slower right when a long session most needed a
# responsive UI. Sessions untouched for this long are dropped from the hot
# file; their human-readable mirror in `.memory/session-<id>.json` and any
# `/compact` checkpoints are untouched, so nothing is actually lost -- it is
# just no longer paid for on every unrelated session's write.
STALE_SESSION_MAX_AGE = timedelta(days=21)

# How recently a "running" session's updated_at must have moved for it to
# still count as a live process actively working. start_action/finish_action
# and every save_session_state call refresh updated_at, so a genuinely busy
# session touches this well inside the window; a crashed/killed process
# leaves execution_status stuck at "running" with a stale updated_at, and
# should be treated as dead rather than block reuse forever.
SESSION_LIVENESS_WINDOW = timedelta(seconds=120)

_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(authorization\s*:\s*bearer\s+)([^\s]+)"),
    re.compile(r"(?i)\b(password|passwd|token|access_token|refresh_token|api[_-]?key|client_secret)\s*([=:])\s*([^\s&]+)"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def redact_secrets(text: str) -> str:
    value = text
    value = _SECRET_PATTERNS[0].sub(r"\1[REDACTED]", value)
    value = _SECRET_PATTERNS[1].sub(r"\1\2[REDACTED]", value)
    value = _SECRET_PATTERNS[2].sub("[REDACTED_JWT]", value)
    return value


def _sanitize(value: Any) -> Any:
    if isinstance(value, str):
        return redact_secrets(value)
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    return value


def _compact_memory_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Bound durable memory while retaining protocol shape and recent detail.

    This is deliberately a storage-only compaction pass. The live provider
    context is compacted by runner_local.py; memory compaction must never
    remove a tool-call/tool-result pair that the next process may need to
    close safely during resume.
    """
    bounded_reversed: list[dict[str, Any]] = []
    total = 0
    for message in reversed(messages[-MAX_TURN_CHECKPOINT_MESSAGES:]):
        item = dict(message)
        for key in ("content", "name"):
            value = item.get(key)
            if isinstance(value, str) and len(value) > MAX_MEMORY_MESSAGE_CHARS:
                item[key] = (
                    value[:MAX_MEMORY_MESSAGE_CHARS // 2]
                    + "\n… [memory compacted; full live context is unavailable in this snapshot] …\n"
                    + value[-MAX_MEMORY_MESSAGE_CHARS // 2:]
                )
        encoded_size = len(json.dumps(item, default=str))
        if total + encoded_size > MAX_MEMORY_TOTAL_CHARS and bounded_reversed:
            # Keep the latest complete messages; older context is represented
            # by conversation_summary/evidence rather than being silently
            # allowed to make future atomic writes stall.
            continue
        bounded_reversed.append(item)
        total += encoded_size
    return list(reversed(bounded_reversed))


@dataclass
class QueuedInstruction:
    id: str
    text: str
    classification: str = "append"
    priority: int = 100
    status: str = "queued"
    created_at: str = field(default_factory=_now)


@dataclass
class AgentAction:
    id: str
    type: str
    purpose: str
    status: str = "planned"
    risk: str = "read_only"
    detail: str = ""
    depends_on: list[str] = field(default_factory=list)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result_summary: str = ""
    attempts: int = 0
    last_error: str = ""


@dataclass
class CodePlan:
    id: str
    objective: str
    content: str
    status: str = "ready"
    source_task_id: Optional[str] = None
    execution_task_id: Optional[str] = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    # Structured mirror of the server's `plan_created` event payload (a list
    # of {"step": str, "status": str} items), plus a client-assigned `index`
    # since the server payload carries no stable step id of its own. `content`
    # remains the human-readable markdown fallback -- this is additive, not a
    # replacement, so nothing that reads `content` today needs to change.
    steps: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SessionState:
    session_id: int
    workspace_root: str = ""
    primary_workspace: str = ""
    allowed_workspaces: list[str] = field(default_factory=list)
    repository_root: Optional[str] = None
    current_working_directory: str = ""
    active_branch: Optional[str] = None
    last_event_id: int = 0
    last_task_id: Optional[str] = None
    active_task: Optional[dict[str, Any]] = None
    current_phase: str = "idle"
    execution_status: str = "idle"
    inspected_files: dict[str, dict[str, Any]] = field(default_factory=dict)
    discovered_symbols: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    discovered_services: list[dict[str, Any]] = field(default_factory=list)
    discovered_reports: list[dict[str, Any]] = field(default_factory=list)
    repository_context: dict[str, Any] = field(default_factory=dict)
    completed_actions: list[dict[str, Any]] = field(default_factory=list)
    pending_actions: list[dict[str, Any]] = field(default_factory=list)
    queued_user_instructions: list[dict[str, Any]] = field(default_factory=list)
    modified_files: list[dict[str, Any]] = field(default_factory=list)
    validation_results: list[dict[str, Any]] = field(default_factory=list)
    unresolved_issues: list[dict[str, Any]] = field(default_factory=list)
    running_action: Optional[dict[str, Any]] = None
    conversation_summary: str = ""
    # Durable standalone conversation context and the currently executing
    # provider/tool turn.  These are deliberately separate: completed turns
    # feed ordinary follow-ups, while turn_checkpoint lets a fresh process
    # continue after Ctrl+C, SSH loss, provider disconnect, or process death
    # without guessing what "proceed" refers to or repeating completed tools.
    conversation_history: list[dict[str, Any]] = field(default_factory=list)
    turn_checkpoint: Optional[dict[str, Any]] = None
    context_checkpoints: list[dict[str, Any]] = field(default_factory=list)
    saved_plans: list[dict[str, Any]] = field(default_factory=list)
    active_plan_id: Optional[str] = None
    discovery_fingerprint: str = ""
    selected_model: str = "auto"
    selected_provider: Optional[str] = None
    estimated_context_tokens: int = 0
    updated_at: str = ""
    # Set only for a swarm sub-task's own child session (see
    # workspace.resolve_swarm_subtask_workspace) -- None for every ordinary
    # session. Lets concurrent swarm sub-tasks over the same workspace_root
    # each get their own SessionState row instead of racing on single-value
    # fields (current_phase/running_action/active_task/...) of one shared
    # session the way resolve_local_workspace's same-workspace_root reuse
    # would otherwise cause. May itself be None (e.g. `agent-cmd delegate`
    # is a one-shot CLI invocation with no pre-existing session to record
    # as a parent) -- is_swarm_child below is the actual "hide this from
    # default listings" marker; parent_session_id is best-effort context,
    # not the tag itself. Live-caught bug: an earlier version used
    # `parent_session_id is not None` as the hide/show filter directly,
    # which silently failed to hide any child session minted with no real
    # parent to record (confirmed live via `agent-cmd delegate`).
    parent_session_id: Optional[int] = None
    is_swarm_child: bool = False
    swarm_label: str = ""
    # Set only for a mutating swarm child that actually got worktree
    # isolation (workspace.resolve_swarm_subtask_workspace's isolate=True
    # path) -- None for every read-only child and for any child where
    # isolation fell back to the shared root (non-git workspace, or worktree
    # creation failed). Recorded so a leftover isolated worktree is
    # discoverable/cleanable after the fact instead of only living in the
    # child process's own memory for the duration of the swarm run.
    swarm_worktree_path: Optional[str] = None
    swarm_worktree_branch: Optional[str] = None


# In-process cache of the parsed state.json, keyed by the file's mtime. A
# naive re-read-and-json.loads on every call cost ~150ms on a real 15.7MB/
# 80-session install (measured live) -- and get_session_state/put_session_state
# both call this, often several times per single turn event -- so a long
# audit spent a growing fraction of every UI tick just re-parsing state it
# had already parsed moments earlier. The mtime check still picks up writes
# from another process (e.g. `tamfis-code queue`, a swarm child, `attach`)
# since those change the file's mtime; only redundant same-process re-reads
# of an unchanged file are skipped.
_STATE_CACHE: Optional[dict[str, Any]] = None
# Keyed by (path, mtime) rather than mtime alone so a process (or test
# fixture) that repoints STATE_PATH at a different file mid-run can never
# match a stale cache entry left over from the previous path by mtime
# coincidence.
_STATE_CACHE_KEY: Optional[tuple[str, float]] = None


def _load_raw() -> dict[str, Any]:
    global _STATE_CACHE, _STATE_CACHE_KEY
    if not STATE_PATH.is_file():
        _STATE_CACHE, _STATE_CACHE_KEY = {}, None
        return {}
    try:
        mtime = STATE_PATH.stat().st_mtime
    except OSError:
        mtime = None
    cache_key = (str(STATE_PATH), mtime) if mtime is not None else None
    if _STATE_CACHE is not None and cache_key is not None and cache_key == _STATE_CACHE_KEY:
        return _STATE_CACHE
    try:
        payload = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        result = payload if isinstance(payload, dict) else {}
    except OSError as exc:
        # A permission/ownership mismatch here silently resets every session
        # to blank (no active_plan_id, no saved_plans, no conversation_summary)
        # -- the CLI then looks "amnesiac", re-proposing the same plan every
        # invocation with no memory of prior progress. Surface it instead of
        # swallowing it so that symptom is diagnosable.
        print(
            f"⚠ Could not read local session state at {STATE_PATH} ({exc}). "
            "Continuing with a blank session -- prior plan/task memory is unavailable "
            "until this is fixed (likely an ownership/permission mismatch on "
            f"{CONFIG_DIR}).",
            file=sys.stderr,
        )
        return _STATE_CACHE if _STATE_CACHE is not None else {}
    except json.JSONDecodeError:
        return _STATE_CACHE if _STATE_CACHE is not None else {}
    _STATE_CACHE, _STATE_CACHE_KEY = result, cache_key
    return result


def _save_raw(data: dict[str, Any]) -> None:
    """Atomically replace state so a killed CLI cannot leave invalid JSON."""
    global _STATE_CACHE, _STATE_CACHE_KEY
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    # Only chmod when it isn't already owner-only -- calling this unconditionally
    # on every save raises PermissionError (uncaught, all the way up) the moment
    # CONFIG_DIR is ever owned by a different user than the caller, which used to
    # crash the whole CLI on its very first state write.
    if stat.S_IMODE(os.stat(CONFIG_DIR).st_mode) != stat.S_IRWXU:
        os.chmod(CONFIG_DIR, stat.S_IRWXU)
    fd, temp_name = tempfile.mkstemp(prefix=".state-", suffix=".json", dir=CONFIG_DIR)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(_sanitize(data), handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_name, stat.S_IRUSR | stat.S_IWUSR)
        os.replace(temp_name, STATE_PATH)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
    try:
        # Record what we just wrote as the cache's ground truth, keyed by the
        # replaced file's fresh mtime, so the next _load_raw() in this or any
        # process sees it without a redundant re-parse of the file we just
        # produced ourselves.
        _STATE_CACHE, _STATE_CACHE_KEY = data, (str(STATE_PATH), STATE_PATH.stat().st_mtime)
    except OSError:
        pass


def _save_memory_snapshot(state: SessionState) -> None:
    """Write the canonical, human-readable realtime session memory mirror."""
    memory_dir = CONFIG_DIR / ".memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    if stat.S_IMODE(os.stat(memory_dir).st_mode) != stat.S_IRWXU:
        os.chmod(memory_dir, stat.S_IRWXU)
    payload = _sanitize({
        "schema_version": 1,
        "session_id": state.session_id,
        "workspace_root": state.workspace_root,
        "primary_workspace": state.primary_workspace,
        "updated_at": state.updated_at,
        "current_phase": state.current_phase,
        "execution_status": state.execution_status,
        "active_task": state.active_task,
        "running_action": state.running_action,
        "turn_checkpoint": state.turn_checkpoint,
        "conversation_history": state.conversation_history[-MAX_CONVERSATION_MESSAGES:],
        "conversation_summary": state.conversation_summary,
        "recent_completed_actions": state.completed_actions[-25:],
        "pending_actions": state.pending_actions[-25:],
        "recent_context_checkpoints": state.context_checkpoints[-10:],
        "modified_files": state.modified_files[-50:],
        "validation_results": state.validation_results[-20:],
        "unresolved_issues": state.unresolved_issues[-20:],
    })
    target = memory_dir / f"session-{state.session_id}.json"
    fd, temp_name = tempfile.mkstemp(prefix=f".session-{state.session_id}-", suffix=".json", dir=memory_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_name, stat.S_IRUSR | stat.S_IWUSR)
        os.replace(temp_name, target)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def get_session_state(session_id: int) -> SessionState:
    raw = _load_raw().get(str(session_id))
    if not raw:
        volatile = _VOLATILE_STATE.get(_volatile_key(session_id))
        if volatile is not None:
            return volatile
        return SessionState(session_id=session_id)
    allowed = set(SessionState.__dataclass_fields__)
    values = {key: value for key, value in raw.items() if key in allowed and key != "session_id"}
    values["last_event_id"] = int(values.get("last_event_id") or 0)
    return SessionState(session_id=session_id, **values)


def is_session_actively_running(state: SessionState) -> bool:
    """True if `state` looks like a live process working right now, as
    opposed to a session that merely finished (or crashed) with
    execution_status left at "running" from its last write.

    Used to decide whether a same-workspace_root session can safely be
    reused (see workspace.resolve_local_workspace) -- reusing the row of a
    session that is genuinely mid-task hands two concurrent processes the
    same state.json entry, so each one's current_phase/running_action/
    active_task/queued_user_instructions overwrite the other's -- visible
    live as one terminal's status flipping in sync with unrelated work
    happening in a second terminal opened in the same directory.
    """
    if state.execution_status != "running":
        return False
    if not state.updated_at:
        return False
    try:
        updated_at = datetime.fromisoformat(state.updated_at)
    except ValueError:
        return False
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - updated_at < SESSION_LIVENESS_WINDOW


def _enforce_state_caps(state: SessionState) -> None:
    """Trim every unbounded list/dict field to a hard cap, keeping the most
    recent entries. This is the safety net for the UI-freeze bug: a long
    audit that reads and edits many files grew `inspected_files` and
    `modified_files` without limit (no call site ever sliced them), so a
    session that ran for hours could make its own single entry in state.json
    multiple megabytes -- and every subsequent event in that same session
    then had to re-serialize and re-fsync that multi-megabyte blob on the
    live UI thread. Called on every put_session_state so no call site can
    reintroduce unbounded growth.
    """
    state.modified_files = state.modified_files[-MAX_MODIFIED_FILES:]
    state.validation_results = state.validation_results[-MAX_VALIDATION_RESULTS:]
    state.unresolved_issues = state.unresolved_issues[-MAX_UNRESOLVED_ISSUES:]
    state.pending_actions = state.pending_actions[-MAX_PENDING_ACTIONS:]
    state.completed_actions = state.completed_actions[-MAX_ACTION_HISTORY:]
    state.context_checkpoints = state.context_checkpoints[-MAX_CHECKPOINTS:]
    state.discovered_services = state.discovered_services[-MAX_DISCOVERED_SERVICES:]
    state.discovered_reports = state.discovered_reports[-MAX_DISCOVERED_REPORTS:]
    state.queued_user_instructions = state.queued_user_instructions[-MAX_QUEUED_INSTRUCTIONS:]
    if len(state.inspected_files) > MAX_INSPECTED_FILES:
        # dicts preserve insertion order; keep the most recently-touched
        # entries by dropping the oldest keys rather than an arbitrary set.
        overflow = len(state.inspected_files) - MAX_INSPECTED_FILES
        for key in list(state.inspected_files)[:overflow]:
            del state.inspected_files[key]
    if len(state.discovered_symbols) > MAX_DISCOVERED_SYMBOL_FILES:
        overflow = len(state.discovered_symbols) - MAX_DISCOVERED_SYMBOL_FILES
        for key in list(state.discovered_symbols)[:overflow]:
            del state.discovered_symbols[key]


def _prune_stale_sessions(data: dict[str, Any], *, keep_session_id: int) -> None:
    """Drop sessions untouched for STALE_SESSION_MAX_AGE from the hot,
    shared state.json so an install with months of history doesn't force
    every session's write to read-modify-write everyone else's data too (a
    live install measured at 15.7MB/80 sessions, several MB each, before
    this fix). The session being written right now is always kept regardless
    of its own `updated_at` (about to be refreshed anyway). Nothing is lost:
    each session's human-readable mirror in `.memory/session-<id>.json` and
    any `/compact` checkpoints are untouched by this -- only the copy in the
    single shared hot file is dropped.
    """
    cutoff = datetime.now(timezone.utc) - STALE_SESSION_MAX_AGE
    keep_key = str(keep_session_id)
    for key in list(data.keys()):
        if key == keep_key:
            continue
        entry = data.get(key)
        if not isinstance(entry, dict):
            continue
        updated_at = entry.get("updated_at")
        if not updated_at:
            continue
        try:
            when = datetime.fromisoformat(str(updated_at))
        except ValueError:
            continue
        if when < cutoff:
            del data[key]


def put_session_state(state: SessionState) -> None:
    data = _load_raw()
    latest = data.get(str(state.session_id), {})
    # A second `tamfis-code queue ...` process may add an instruction while
    # the streaming process is saving an event cursor. Merge by id so the
    # cursor write cannot erase that newly queued user input.
    if isinstance(latest, dict):
        merged_queue = {item.get("id"): item for item in latest.get("queued_user_instructions", []) if item.get("id")}
        merged_queue.update({item.get("id"): item for item in state.queued_user_instructions if item.get("id")})
        state.queued_user_instructions = sorted(
            merged_queue.values(), key=lambda item: (int(item.get("priority", 100)), item.get("created_at", ""))
        )
        # Event-cursor updates can race a foreground plan save in the same
        # way they race queued instructions. Preserve plans by id as well.
        merged_plans = {item.get("id"): item for item in latest.get("saved_plans", []) if item.get("id")}
        merged_plans.update({item.get("id"): item for item in state.saved_plans if item.get("id")})
        state.saved_plans = sorted(
            merged_plans.values(), key=lambda item: item.get("created_at", "")
        )[-MAX_SAVED_PLANS:]
    state.updated_at = _now()
    # Defense-in-depth caps + eviction of long-stale sessions, applied before
    # every write (see _enforce_state_caps/_prune_stale_sessions docstrings).
    # This is what actually keeps the write below fast: a live install
    # measured at 15.7MB/80 sessions (several individual sessions multiple
    # MB each, from fields like modified_files/inspected_files that no call
    # site ever bounded) turned every single event's write -- there can be
    # dozens per turn during a long audit -- into a multi-hundred-millisecond
    # synchronous json.dump+fsync on the same thread driving the live
    # terminal UI, which is what actually presented as "the UI freezes".
    _enforce_state_caps(state)
    data[str(state.session_id)] = asdict(state)
    _prune_stale_sessions(data, keep_session_id=state.session_id)
    # Keep the live session usable when a container, sandbox, or ownership
    # mismatch makes the configured state directory unwritable. This is not a
    # substitute for durable recovery: the warning makes that limitation
    # explicit, while the in-process ledger still lets mutation evidence,
    # approvals, and the current turn proceed consistently.
    _VOLATILE_STATE[_volatile_key(state.session_id)] = state
    try:
        _save_raw(data)
    except OSError as exc:
        print(
            f"⚠ Local session state is volatile ({CONFIG_DIR} is not writable: {exc}). "
            "The current task can continue, but resume data will not survive this process.",
            file=sys.stderr,
        )
        return
    try:
        _save_memory_snapshot(state)
    except OSError as exc:
        print(
            f"⚠ Could not update the session memory mirror ({exc}); state.json remains available.",
            file=sys.stderr,
        )


def save_session_state(
    session_id: int, *, workspace_root: Optional[str] = None,
    last_event_id: Optional[int] = None, last_task_id: Optional[str] = None,
    **updates: Any,
) -> None:
    state = get_session_state(session_id)
    if workspace_root is not None:
        state.workspace_root = workspace_root
        if not state.primary_workspace:
            state.primary_workspace = workspace_root
        if workspace_root not in state.allowed_workspaces:
            state.allowed_workspaces.append(workspace_root)
        if not state.current_working_directory:
            state.current_working_directory = workspace_root
    if last_event_id is not None:
        # Sequence ids only ever move forward for a given session -- never
        # regress state from a stale/out-of-order write (e.g. a slower
        # concurrent `logs --follow` process finishing after a newer one).
        state.last_event_id = max(int(last_event_id), int(state.last_event_id or 0))
    if last_task_id is not None:
        state.last_task_id = last_task_id
    for key, value in updates.items():
        if key in SessionState.__dataclass_fields__ and key != "session_id":
            setattr(state, key, value)
    put_session_state(state)


def save_turn_checkpoint(
    session_id: int, *, objective: str, mode: str,
    messages: list[dict[str, Any]], partial_assistant: str = "",
    status: str = "running", last_error: str = "",
) -> None:
    """Atomically persist the resumable portion of a local agent turn.

    The runner already compacts oversized tool results before a provider
    request.  Keeping only the newest bounded message window here prevents a
    long-running REPL from growing state.json forever while retaining native
    tool_call/tool-result pairs needed for protocol-correct continuation.
    """
    state = get_session_state(session_id)
    state.turn_checkpoint = {
        "objective": objective,
        "mode": mode,
        "status": status,
        "messages": _compact_memory_messages(messages),
        "partial_assistant": partial_assistant,
        "last_error": last_error,
        "updated_at": _now(),
    }
    put_session_state(state)


def mark_turn_checkpoint_interrupted(session_id: int, *, error: str) -> None:
    """Durably mark the latest live checkpoint after task cancellation.

    Cancellation can arrive while the runner is awaiting a provider or tool,
    so the runner's local working list is not available at that exact point.
    Updating the already-atomic snapshot is safer than overwriting it with
    the original pre-turn messages and still makes `continue` discoverable.
    """
    state = get_session_state(session_id)
    checkpoint = dict(state.turn_checkpoint or {})
    if not checkpoint:
        return
    checkpoint["status"] = "interrupted"
    checkpoint["last_error"] = error
    checkpoint["updated_at"] = _now()
    state.turn_checkpoint = checkpoint
    put_session_state(state)


def clear_turn_checkpoint(session_id: int) -> None:
    state = get_session_state(session_id)
    state.turn_checkpoint = None
    put_session_state(state)




def remember_conversation_turn(
    session_id: int, *, objective: str, answer: str, clear_checkpoint: bool = False,
) -> None:
    """Append a completed local turn to durable, bounded session memory."""
    state = get_session_state(session_id)
    history = [*state.conversation_history, {"role": "user", "content": objective}]
    if answer:
        history.append({"role": "assistant", "content": answer})
    # Keep a rolling, bounded digest of older turns so a long-lived REPL
    # retains continuity without replaying an ever-growing transcript.
    if len(history) > MAX_CONVERSATION_MESSAGES:
        older = history[:-MAX_CONVERSATION_MESSAGES]
        digest_parts = [
            f"{item.get('role', 'unknown')}: {str(item.get('content') or '')[:600]}"
            for item in older[-12:]
        ]
        digest = "\n".join(digest_parts)
        state.conversation_summary = (
            (state.conversation_summary + "\n" if state.conversation_summary else "")
            + digest
        )[-8000:]
    elif answer:
        state.conversation_summary = answer[-4000:]
    state.conversation_history = _compact_memory_messages(history[-MAX_CONVERSATION_MESSAGES:])
    if clear_checkpoint:
        state.turn_checkpoint = None
    put_session_state(state)


def start_action(session_id: int, *, action_type: str, purpose: str,
                 risk: str = "read_only", detail: str = "") -> AgentAction:
    state = get_session_state(session_id)
    action = AgentAction(
        id=f"action_{uuid.uuid4().hex[:12]}", type=action_type, purpose=purpose,
        risk=risk, detail=detail, status="running", started_at=_now(),
    )
    state.running_action = asdict(action)
    state.pending_actions.append(asdict(action))
    state.execution_status = "running"
    put_session_state(state)
    return action


# Number of consecutive failures for the same action `purpose` before it's
# escalated into `unresolved_issues` -- below this, a single failure is just
# noise (transient network blip, etc), not yet "the agent is stuck".
FAILURE_ESCALATION_THRESHOLD = 2


def finish_action(session_id: int, action_id: str, *, status: str, summary: str = "", error: str = "") -> None:
    state = get_session_state(session_id)
    finished = None
    remaining = []
    for action in state.pending_actions:
        if action.get("id") == action_id:
            action.update(status=status, completed_at=_now(), result_summary=summary)
            if status == "failed":
                action["last_error"] = error or summary
                purpose = action.get("purpose", "")
                prior_failures = sum(
                    1 for completed in state.completed_actions
                    if completed.get("purpose") == purpose and completed.get("status") == "failed"
                )
                action["attempts"] = prior_failures + 1
                if action["attempts"] >= FAILURE_ESCALATION_THRESHOLD and not any(
                    issue.get("type") == "repeated_action_failure" and issue.get("purpose") == purpose
                    for issue in state.unresolved_issues
                ):
                    state.unresolved_issues.append({
                        "type": "repeated_action_failure", "status": "needs_attention",
                        "purpose": purpose, "attempts": action["attempts"],
                        "detail": (
                            f"'{purpose}' has failed {action['attempts']} times in a row -- "
                            "consider a different approach instead of retrying as-is."
                        ),
                    })
            finished = action
        else:
            remaining.append(action)
    state.pending_actions = remaining
    if finished:
        state.completed_actions = (state.completed_actions + [finished])[-MAX_ACTION_HISTORY:]
    if state.running_action and state.running_action.get("id") == action_id:
        state.running_action = None
    state.execution_status = "idle" if state.running_action is None else "running"
    put_session_state(state)


def enqueue_instruction(session_id: int, text: str, *, classification: str = "append",
                        priority: int = 100) -> QueuedInstruction:
    state = get_session_state(session_id)
    item = QueuedInstruction(id=f"instruction_{uuid.uuid4().hex[:10]}", text=text,
                             classification=classification, priority=priority)
    state.queued_user_instructions.append(asdict(item))
    state.queued_user_instructions.sort(
        key=lambda value: (int(value.get("priority", 100)), value.get("created_at", ""))
    )
    put_session_state(state)
    return item


def update_instruction(session_id: int, instruction_id: str, status: str) -> bool:
    state = get_session_state(session_id)
    for item in state.queued_user_instructions:
        if item.get("id") == instruction_id:
            item["status"] = status
            put_session_state(state)
            return True
    return False


def checkpoint(session_id: int, *, reason: str, summary: str = "") -> None:
    state = get_session_state(session_id)
    state.context_checkpoints = (state.context_checkpoints + [{
        "created_at": _now(), "reason": reason, "phase": state.current_phase,
        "task_id": state.last_task_id, "last_event_id": state.last_event_id,
        "summary": summary,
    }])[-MAX_CHECKPOINTS:]
    put_session_state(state)


def save_plan(
    session_id: int, *, objective: str, content: str,
    source_task_id: Optional[str] = None,
    steps: Optional[list[dict[str, Any]]] = None,
) -> CodePlan:
    """Persist a completed planning result as an executable plan."""
    state = get_session_state(session_id)
    plan = CodePlan(
        id=f"plan_{uuid.uuid4().hex[:10]}",
        objective=objective.strip(), content=content.strip(),
        source_task_id=source_task_id,
        steps=[
            {**step, "index": index}
            for index, step in enumerate(steps or [])
            if isinstance(step, dict)
        ],
    )
    state.saved_plans = (state.saved_plans + [asdict(plan)])[-MAX_SAVED_PLANS:]
    state.active_plan_id = plan.id
    put_session_state(state)
    return plan


def get_plan(session_id: int, plan_id: Optional[str] = None) -> Optional[dict[str, Any]]:
    """Return the selected plan, accepting an exact id, unique prefix, or latest."""
    state = get_session_state(session_id)
    plans = state.saved_plans
    if not plans:
        return None
    wanted = (plan_id or state.active_plan_id or "").strip()
    if not wanted:
        return plans[-1]
    exact = next((item for item in plans if item.get("id") == wanted), None)
    if exact:
        return exact
    matches = [item for item in plans if str(item.get("id", "")).startswith(wanted)]
    return matches[0] if len(matches) == 1 else None


def update_plan(
    session_id: int, plan_id: str, *, status: str,
    execution_task_id: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    state = get_session_state(session_id)
    updated = None
    for item in state.saved_plans:
        if item.get("id") == plan_id:
            item["status"] = status
            item["updated_at"] = _now()
            if execution_task_id is not None:
                item["execution_task_id"] = execution_task_id
            updated = item
            break
    if updated is not None:
        state.active_plan_id = plan_id
        put_session_state(state)
    return updated


def update_plan_steps(session_id: int, plan_id: str, items: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Persist the server's `plan_created.items` onto the matching saved plan.

    Today the server only ever emits this list once per plan and never
    re-references individual steps afterward, so `steps` reflects whatever
    the most recent `plan_created` event said -- there's no per-step
    completion event to key off yet.
    """
    state = get_session_state(session_id)
    updated = None
    for item in state.saved_plans:
        if item.get("id") == plan_id:
            item["steps"] = [
                {**step, "index": index}
                for index, step in enumerate(items)
                if isinstance(step, dict)
            ]
            item["updated_at"] = _now()
            updated = item
            break
    if updated is not None:
        put_session_state(state)
    return updated


def plan_execution_objective(plan: dict[str, Any]) -> str:
    """Turn a reviewed saved plan into an unambiguous agent execution task."""
    return (
        "Execute the saved engineering plan below. Inspect the current workspace first and "
        "adapt only where repository drift requires it. Implement the work, run proportionate "
        "validation, and report concrete results. Do not merely restate the plan.\n\n"
        f"Original objective:\n{plan.get('objective', '')}\n\n"
        f"Saved plan ({plan.get('id', 'unknown')}):\n{plan.get('content', '')}"
    )


def all_known_session_ids() -> list[int]:
    ids = []
    for key in _load_raw().keys():
        try:
            ids.append(int(key))
        except ValueError:
            continue
    return sorted(ids)


# --- Thread compression / summarization -------------------------------------
#
# Claude Code's `/compact` and Codex's context-rollover both replace a long
# raw transcript with a structured, bounded summary the next turn can reason
# from without re-reading every prior message. tamfis-code already had a
# storage-only compaction pass (_compact_memory_messages) and a rolling
# conversation_summary digest, but nothing the user could invoke to actually
# *compress* the visible thread into a concise recap -- `/compact` only saved
# a checkpoint. This produces that recap deterministically (no extra provider
# call, no latency) from the durable session state already on disk, so a
# long REPL thread stops dominating the terminal and the next turn's context.

# How many recent turns to keep verbatim after compression; everything older
# is folded into the structured summary. Matches Claude Code's own behaviour
# of retaining the last few exchanges in full while summarizing the rest.
COMPACT_KEEP_RECENT_TURNS = 4
# Hard cap on the rendered summary so a pathological long thread can't make
# the compressed recap itself unreadable.
MAX_COMPACT_SUMMARY_CHARS = 4000


def _bounded_text(value: str, *, head: int, tail: int, label: str) -> str:
    """Return a compact, evidence-preserving representation of a large string.

    This is the same helper runner_local.py uses to bound oversized tool
    output. It is defined here (rather than imported from runner_local, which
    would create a circular import -- runner_local imports this module) so
    summarize_thread/compact_session_thread can bound a long prior
    conversation_summary without raising NameError. Confirmed live: a session
    that had been /compact-ed and then restarted in a fresh process had a long
    conversation_summary but an empty conversation_history, so summarize_thread
    took the ``prior_summary`` branch and crashed on the undefined name.
    """
    if len(value) <= head + tail:
        return value
    omitted = len(value) - head - tail
    return (
        f"{value[:head]}\n"
        f"...[{label}: {omitted} characters omitted; original={len(value)}]...\n"
        f"{value[-tail:] if tail else ''}"
    )


def _extract_turns(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Group a flat message list into (objective, answer) turns.

    A turn is a user objective followed by zero or more assistant messages
    (and the tool calls between them are summarized, not replayed). This is
    the same grouping render.py's print_recent_thread uses for /resume, kept
    here so the compression pass can reason about whole turns rather than
    individual messages.
    """
    turns: list[dict[str, str]] = []
    current: dict[str, str] = {"objective": "", "answer": ""}
    for message in messages:
        role = str(message.get("role") or "")
        content = str(message.get("content") or "").strip()
        if role == "user" and content:
            if current["objective"] or current["answer"]:
                turns.append(current)
            current = {"objective": content, "answer": ""}
        elif role == "assistant" and content:
            if current["objective"]:
                current["answer"] = (current["answer"] + "\n" + content).strip() if current["answer"] else content
    if current["objective"] or current["answer"]:
        turns.append(current)
    return turns


def summarize_thread(session_id: int, *, keep_recent: int = COMPACT_KEEP_RECENT_TURNS) -> str:
    """Produce a structured, bounded recap of this session's conversation.

    Draws only from durable SessionState (conversation_history,
    conversation_summary, completed_actions, modified_files, saved_plans) --
    no provider call, no network, no latency. Older turns are folded into a
    compact bullet recap; the most recent `keep_recent` turns are preserved
    verbatim so the user can still see exactly what just happened. This is
    what `/compact` and `/summary` render, and what the next turn's context
    can be seeded from instead of the full raw transcript.
    """
    state = get_session_state(session_id)
    turns = _extract_turns(state.conversation_history)
    if not turns:
        prior_summary = (state.conversation_summary or "").strip()
        if prior_summary:
            return _bounded_text(prior_summary, head=MAX_COMPACT_SUMMARY_CHARS, tail=0, label="summary")
        return "(No conversation recorded in this session yet.)"

    recent = turns[-keep_recent:] if keep_recent > 0 else []
    older = turns[:-keep_recent] if keep_recent > 0 else turns

    lines: list[str] = []
    if older:
        lines.append(f"Summary of {len(older)} earlier turn(s):")
        for index, turn in enumerate(older, start=1):
            objective = turn["objective"]
            answer = turn["answer"]
            obj_preview = objective if len(objective) <= 200 else objective[:197] + "..."
            if answer:
                ans_preview = answer if len(answer) <= 200 else answer[:197] + "..."
                lines.append(f"  {index}. You: {obj_preview}")
                lines.append(f"     Assistant: {ans_preview}")
            else:
                lines.append(f"  {index}. You: {obj_preview} (no recorded answer)")
        lines.append("")

    # Durable progress facts that survive even when individual messages were
    # compacted out of conversation_history by _compact_memory_messages.
    if state.modified_files:
        recent_paths = [str(m.get("path") or "") for m in state.modified_files[-10:] if m.get("path")]
        if recent_paths:
            lines.append("Files modified in this session: " + ", ".join(recent_paths))
    if state.unresolved_issues:
        lines.append(f"Unresolved issues: {len(state.unresolved_issues)} (run /doctor for detail)")
    if state.saved_plans:
        active = next((p for p in reversed(state.saved_plans) if p.get("id") == state.active_plan_id), None)
        if active:
            steps = active.get("steps") or []
            done = sum(1 for s in steps if s.get("status") == "completed")
            lines.append(f"Active plan: {active.get('id')} ({done}/{len(steps)} steps done)")

    if recent:
        lines.append("")
        lines.append(f"Recent {len(recent)} turn(s) (full text):")
        for turn in recent:
            lines.append(f"  You: {turn['objective']}")
            if turn["answer"]:
                answer = turn["answer"]
                if len(answer) > 1200:
                    answer = answer[:600] + "\n     ...[truncated]...\n     " + answer[-400:]
                lines.append(f"  Assistant: {answer}")
            lines.append("")

    summary = "\n".join(lines).strip()
    if len(summary) > MAX_COMPACT_SUMMARY_CHARS:
        summary = _bounded_text(summary, head=MAX_COMPACT_SUMMARY_CHARS // 2, tail=MAX_COMPACT_SUMMARY_CHARS // 2, label="thread summary")
    return summary


def compact_session_thread(session_id: int, *, keep_recent: int = COMPACT_KEEP_RECENT_TURNS) -> str:
    """Compress the durable thread in place: fold older turns into
    conversation_summary and keep only the recent `keep_recent` turns in
    conversation_history. Returns the structured recap (same as
    summarize_thread) so the caller can display it immediately.

    This is the real `/compact` action: after it, the next turn's context is
    seeded from the bounded summary + recent turns, not the full raw
    transcript -- matching how Claude Code/Codex keep a long REPL usable
    without the terminal UI/UX growing heavy.
    """
    state = get_session_state(session_id)
    turns = _extract_turns(state.conversation_history)
    if not turns or len(turns) <= keep_recent:
        # Nothing to fold -- still return the recap so /compact gives useful
        # feedback even on a short thread.
        return summarize_thread(session_id, keep_recent=keep_recent)

    older = turns[:-keep_recent] if keep_recent > 0 else turns
    digest_parts: list[str] = []
    for turn in older:
        objective = turn["objective"]
        answer = turn["answer"]
        obj_preview = objective if len(objective) <= 300 else objective[:297] + "..."
        if answer:
            ans_preview = answer if len(answer) <= 300 else answer[:297] + "..."
            digest_parts.append(f"- You: {obj_preview} -> Assistant: {ans_preview}")
        else:
            digest_parts.append(f"- You: {obj_preview} (no recorded answer)")
    digest = "\n".join(digest_parts)
    state.conversation_summary = (
        (state.conversation_summary + "\n" if state.conversation_summary else "") + digest
    )[-8000:]

    # Rebuild conversation_history from only the retained recent turns,
    # preserving role/content shape for the next turn's context seeding.
    recent_turns = turns[-keep_recent:] if keep_recent > 0 else []
    rebuilt: list[dict[str, Any]] = []
    for turn in recent_turns:
        rebuilt.append({"role": "user", "content": turn["objective"]})
        if turn["answer"]:
            rebuilt.append({"role": "assistant", "content": turn["answer"]})
    state.conversation_history = _compact_memory_messages(rebuilt)
    put_session_state(state)
    return summarize_thread(session_id, keep_recent=keep_recent)


def active_swarm_child_count(*, exclude_session_id: Optional[int] = None) -> int:
    """Count running delegated sessions with one state-file read.

    Terminal footer rendering used to call ``all_known_session_ids()`` and
    then ``get_session_state()`` once per id. Since each helper reparses the
    complete state file, one footer frame performed N+1 full JSON parses.
    Read the raw snapshot once and inspect only the two fields needed by the
    UI so rendering cost does not grow quadratically with session history.
    """
    count = 0
    for raw_session_id, raw in _load_raw().items():
        if not isinstance(raw, dict):
            continue
        try:
            session_id = int(raw_session_id)
        except (TypeError, ValueError):
            continue
        if exclude_session_id is not None and session_id == exclude_session_id:
            continue
        if raw.get("is_swarm_child") and raw.get("execution_status") == "running":
            count += 1
    return count
