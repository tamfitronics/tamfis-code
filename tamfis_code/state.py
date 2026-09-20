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

import asyncio
import json
import os
import re
import stat
import sys
import tempfile
import uuid
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Iterator, Optional

from .config import CONFIG_DIR

# Ceiling for the one best-effort LLM title-upgrade call per session. The
# old httpx path used 25s (a 6s timeout failed almost every live call);
# this bounds the providers path (which can try several providers in turn)
# while still never blocking task completion meaningfully -- callers await
# this only AFTER the turn's result is already persisted.
# Live-measured 2026-09-19: 150s was too generous for something the user is
# watching -- a slow route chain made /regenerate-title look hung and made an
# automatic retitle drag on after the answer was already delivered. The wall
# now fits the whole NIM chain's worst case (12 + 12 + 20 + 20 + 20 = 84s, see
# _TITLE_MODEL_TIMEOUT_SECONDS), but a healthy chain answers in seconds: the
# budget is only spent when the nemotrons stall and kimi-k3 / glm-5.3 must
# take over.
TITLE_UPGRADE_TIMEOUT_SECONDS = 90
# Per-attempt cap forwarded to the OpenAI SDK (chat_completion forwards
# **kwargs into create()): the manager's clients are built with a 120s
# timeout, so ONE cold provider attempt used to consume the entire title
# budget before the machinery could fall back to a healthy route.
_TITLE_ATTEMPT_TIMEOUT_SECONDS = 20
# Title routes: NVIDIA NIM ONLY, walked model by model, best first, failing
# open to the next. Owner ruling 2026-09-19: an auto-title is a background
# nicety and must never spend Ollama Cloud, Hugging Face, or any other metered
# credit -- NIM is the free, tool-calling tier. The call pins
# ProviderType.NVIDIA with allow_fallback=False, so the machinery's
# cross-provider fallback cannot wander off to a paid route; NIM key rotation
# (several free accounts) still applies inside that one provider.
#
# kimi-k3 and glm-5.3 are LAST (owner ruling 2026-09-19: they cause the latency;
# they stay in the pool as a final resort). They are the two that time out on
# NIM's free tier, so each gets a SHORT per-attempt cap (_TITLE_MODEL_TIMEOUT_SECONDS)
# and the loop skips any model whose per-model health circuit is open
# (ProviderManager.route_is_healthy): one failure parks that model for 30s
# (300s for a 404/410) without cooling NIM as a whole, so the next title goes
# straight to a model that answers instead of paying the timeout again.
#
# Live-probed 2026-09-19 against integrate.api.nvidia.com with the real title
# prompt, one request at a time (native tool_calls confirmed on the nemotrons):
#   nemotron-3-ultra-550b-a55b   1.2s  clean title
#   nemotron-3-super-120b-a12b   4.4s  answers (reasoning_effort=low keeps the
#                                      chain-of-thought out of the content)
#   nemotron-3.5-lightning-30b   17s   answers, slow -- last resort
#   kimi-k3, glm-5.3             timed out at 40-90s in every probe today
# Not usable right now, so deliberately absent: qwen3-coder-480b and
# qwen2.5-coder-32b (410 Gone, end-of-life), devstral / codestral-22b /
# nemotron-nano-3 / granite-34b-code (404, listed but not deployed),
# deepseek-v4-flash-0731 (~60s, longer than the whole title budget).
# 2026-09-20 re-ranked on a live benchmark (see providers.py's NVIDIA pool comment):
# super answers in ~1s, ultra in ~1-3s, muse-glimmer in ~1-4s; lightning (7-30s, timeouts)
# is dropped from the chain -- a title is a background nicety, not worth a 30s route.
_TITLE_NIM_MODELS = (
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/nemotron-3-ultra-550b-a55b",
    "meta/muse-glimmer-30b",
    "moonshotai/kimi-k3",
    "z-ai/glm-5.3",
)
# Bounded redraws: one attempt per model in the chain.
_TITLE_MAX_MACHINERY_ATTEMPTS = len(_TITLE_NIM_MODELS)
# Short cap for the two slow last-resort models so a stall costs seconds, not
# the whole title budget.
_TITLE_MODEL_TIMEOUT_SECONDS = {
    "moonshotai/kimi-k3": 12,
    "z-ai/glm-5.3": 12,
}

STATE_PATH = CONFIG_DIR / "state.json"
_VOLATILE_STATE: dict[tuple[str, int], "SessionState"] = {}

# Cross-process advisory lock file. state.json is read-modify-written by
# every CLI process, so concurrent launches in the same directory (or
# even unrelated commands like `tamfis-code queue`/`sessions`) can race and
# corrupt or duplicate session rows. A single per-user lock file protects
# all state.json operations without changing the on-disk format.
_LOCK_PATH = CONFIG_DIR / ".state.lock"
# Re-entrancy guard: flock is per open-file-description, so a nested
# state_lock() inside an already-held one would deadlock against itself.
# A depth counter makes nesting a no-op instead.
_LOCK_DEPTH = 0
_LOCK_FD: Optional[int] = None


def _release_state_lock() -> None:
    global _LOCK_FD
    if _LOCK_FD is None:
        return
    fd, _LOCK_FD = _LOCK_FD, None
    try:
        if sys.platform == "win32":
            import msvcrt

            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        os.close(fd)
    except OSError:
        pass


@contextmanager
def state_lock() -> Iterator[None]:
    """Advisory cross-process lock around state.json read-modify-write
    cycles (see put_session_state/clear_session_state and workspace.py's
    session-id selection).

    Blocking on POSIX (flock LOCK_EX); on Windows msvcrt.LK_LOCK, which
    retries for ~10s. The lock is released automatically if the holder
    dies, so a crashed process can never leave a stale lock behind. If
    locking is unavailable on the platform the body still runs (volatile
    state keeps the current process usable) but a warning is emitted so
    the degraded mode is diagnosable.
    """
    global _LOCK_DEPTH, _LOCK_FD
    if _LOCK_DEPTH > 0:
        # Already held by an outer state_lock() in this same process --
        # flock would deadlock against our own open file description.
        _LOCK_DEPTH += 1
        try:
            yield
        finally:
            _LOCK_DEPTH -= 1
        return
    acquired = False
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(_LOCK_PATH), os.O_CREAT | os.O_RDWR)
        if sys.platform == "win32":
            import msvcrt

            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX)
        _LOCK_FD = fd
        acquired = True
    except Exception as exc:
        print(
            f"\u26a0 Could not acquire the session-state lock ({exc}); concurrent "
            "tamfis-code processes may race on session bookkeeping.",
            file=sys.stderr,
        )
    _LOCK_DEPTH = 1
    try:
        yield
    finally:
        _LOCK_DEPTH = 0
        if acquired:
            _release_state_lock()


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


# Per-session-row sanitized-output cache, keyed by session id -> (id(raw_row),
# sanitized_row). _save_raw used to call _sanitize() on the *entire* on-disk
# store (every session, not just the one being written) on every single
# write. Confirmed live: a store with 14 sessions/21MB, several individual
# sessions multiple MB each, cost 11.5s of pure regex work per write --
# almost all of it re-redacting OTHER sessions' rows that were already
# redacted the previous time and have not changed since. _load_raw() returns
# the exact same cached dict object (same row objects, stable id()) across
# calls in one process as long as the on-disk mtime hasn't moved underneath
# it, so caching by id() is safe: an unrelated session's row is only ever
# re-sanitized when its own object identity changes (a fresh disk load, or
# that session actually being written).
_SANITIZED_ROW_CACHE: dict[str, tuple[int, Any]] = {}


def _sanitize_store(data: dict[str, Any]) -> dict[str, Any]:
    """Sanitize a full session-id -> row store, reusing cached output for
    rows whose object identity is unchanged since the last write."""
    result: dict[str, Any] = {}
    for key, row in data.items():
        cached = _SANITIZED_ROW_CACHE.get(key)
        if cached is not None and cached[0] == id(row):
            result[key] = cached[1]
            continue
        sanitized_row = _sanitize(row)
        _SANITIZED_ROW_CACHE[key] = (id(row), sanitized_row)
        result[key] = sanitized_row
    stale = _SANITIZED_ROW_CACHE.keys() - data.keys()
    for stale_key in stale:
        del _SANITIZED_ROW_CACHE[stale_key]
    return result


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
    # Stable, one-time-assigned display name for this conversation, set from
    # its first completed turn's objective (see remember_conversation_turn)
    # and never overwritten afterwards -- the
    # same "name a session after its first message" convention Codex/Claude
    # Code use, so `tamfis-code resume` and the persistent footer can show a
    # human-readable label instead of a bare session id. Empty until that
    # first turn completes; session_display_title() below supplies the
    # fallback for that window.
    session_title: str = ""
    # True once upgrade_session_title_with_ai has attempted (successfully or
    # not) to replace the mechanical title with an AI-written one. Set
    # *before* the network call, not after, so exactly one attempt is ever
    # made per session -- without this, every completed turn re-ran the
    # upgrade on that turn's own objective, silently overwriting an already
    # good title with one describing whatever the user typed most recently
    # instead of what the session as a whole was about.
    ai_title_attempted: bool = False
    # Title provenance (diagnostics): where the persisted session_title
    # came from -- "llm" (upgrade_session_title_with_ai), "user" (explicit
    # rename via /regenerate-title's rename form; NEVER overwritten
    # automatically) -- and, when the LLM failed, why. title_model/
    # title_generated_at record which model wrote it and when.
    title_source: str = ""
    title_fallback_reason: Optional[str] = None
    title_model: Optional[str] = None
    title_generated_at: Optional[str] = None
    # Set exactly once, the first time this session is ever persisted (see
    # put_session_state) -- distinct from updated_at, which changes on every
    # write. Backs the resume picker's "Sort: Created" option.
    created_at: str = ""
    # True once the user archives this session from the resume picker
    # (Ctrl+A there) -- a soft, reversible hide from the picker's default
    # "Active" view, never a deletion: archived sessions stay fully known,
    # resumable by id, and listed under "Status: Archived". Distinct from
    # `tamfis-code clear-session`, which is the only thing that actually
    # erases a session from active listings.
    archived: bool = False
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
    # Route provenance for THIS session, newest last: which provider/model a
    # turn actually ran on, and every time that changed mid-task (automatic
    # failover, a recovery, or a retry of a route that was cooling down).
    # Live-reported 2026-09-19: a task stopped on a 402 credit-exhaustion error
    # while other routes sat configured, and there was no way to see from
    # /status or the footer that the primary route was dead and nothing had
    # failed over -- a route switch was visible only as a scrolling-through
    # debug diagnostic. Bounded (see ROUTE_EVENT_LIMIT) because it is durable.
    route_events: list[dict] = field(default_factory=list)
    # Per-provider response-time samples for THIS session, newest last:
    # {provider: [seconds, ...]}. The in-process registry in providers.py
    # cannot answer "why is this session slow" after a restart, and it mixes
    # every session the process served, so the durable copy is what /routes
    # reports from (see route_latency_stats).
    route_latency: dict[str, list[float]] = field(default_factory=dict)
    # How many of those samples were failed attempts ({provider: count}): a
    # route failing instantly would otherwise average out looking FASTER than
    # a healthy one. See providers._PROVIDER_LATENCY_FAILURES.
    route_latency_failures: dict[str, int] = field(default_factory=dict)
    estimated_context_tokens: int = 0
    # Compact, schema-stable ledger for long-horizon execution.  This is
    # intentionally separate from the conversational turn checkpoint: the
    # latter is protocol context, while this record is the resumable task
    # state (plan progress, evidence and recovery history).
    task_state: dict[str, Any] = field(default_factory=dict)
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
    # A fork copies the durable conversation/repository context into a new
    # independent session while leaving the source untouched.  Keep explicit
    # lineage so session listings and future UIs can group branches without
    # overloading parent_session_id, which is reserved for swarm children.
    forked_from_session_id: Optional[int] = None


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
    except json.JSONDecodeError as exc:
        _quarantine_corrupt_state_file(exc)
        return _STATE_CACHE if _STATE_CACHE is not None else {}
    _STATE_CACHE, _STATE_CACHE_KEY = result, cache_key
    return result


def _quarantine_corrupt_state_file(exc: json.JSONDecodeError) -> Optional[Path]:
    """Self-healing for a corrupted state.json (partial write surviving a
    kill -9, disk-full, or a hand-edit gone wrong): move the unreadable file
    aside instead of silently discarding it in place, so a session that goes
    blank after a crash is diagnosable and forensically recoverable rather
    than just quietly amnesiac (the prior behavior -- see the OSError branch
    above for the same "surface it" reasoning applied to permission
    mismatches). The next _save_raw() call then writes a fresh, valid file
    at STATE_PATH on its own; nothing else needs to react to this.
    """
    quarantine_path = STATE_PATH.with_name(
        f"{STATE_PATH.name}.corrupted-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.bak"
    )
    try:
        STATE_PATH.replace(quarantine_path)
    except OSError:
        return None
    print(
        f"⚠ Local session state at {STATE_PATH} was corrupted ({exc}) and has been "
        f"quarantined to {quarantine_path} for recovery. Continuing with a blank "
        "session -- prior plan/task memory is unavailable until this is investigated.",
        file=sys.stderr,
    )
    try:
        from .runtime.journal import RuntimeEvent, append_event

        append_event(RuntimeEvent(
            event="state_self_healed",
            execution_id="state-corruption",
            mode="self_heal",
            session_id=0,
            timestamp=datetime.now(timezone.utc).isoformat(),
            status="healed",
            error=str(exc),
            summary=f"quarantined corrupt state.json to {quarantine_path.name}",
        ))
    except Exception:
        pass
    return quarantine_path


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
            json.dump(_sanitize_store(data), handle, indent=2, sort_keys=True)
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
        "session_title": state.session_title,
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
        "task_state": state.task_state,
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
    # Tolerant on read: a hand-edited or older row could carry anything here,
    # and a malformed value must never be able to break every state save.
    latency = state.route_latency if isinstance(state.route_latency, dict) else {}
    cleaned: dict[str, list[float]] = {}
    for provider, samples in latency.items():
        if not isinstance(samples, (list, tuple)):
            continue
        cleaned[str(provider)] = list(samples)[-ROUTE_LATENCY_SAMPLES_PER_PROVIDER:]
    if len(cleaned) > ROUTE_LATENCY_PROVIDER_LIMIT:
        kept = set(cleaned)[-ROUTE_LATENCY_PROVIDER_LIMIT:]
        cleaned = {name: values for name, values in cleaned.items() if name in kept}
    state.route_latency = cleaned
    failures = (
        state.route_latency_failures
        if isinstance(state.route_latency_failures, dict) else {}
    )
    state.route_latency_failures = {
        str(name): int(count)
        for name, count in failures.items()
        if isinstance(count, (int, float)) and not isinstance(count, bool)
        and str(name) in cleaned
    }


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
    # The whole read-modify-write cycle holds the cross-process state lock
    # (see state_lock): without it, two concurrent CLI processes -- e.g. a
    # second terminal opened in the same directory, or `tamfis-code queue`
    # racing a live turn -- could each load the same baseline and the
    # later save would silently erase the earlier one's row, which is one
    # way a session's recorded activity "leaked" into a blank duplicate.
    with state_lock():
        _put_session_state_locked(state)


def _put_session_state_locked(state: SessionState) -> None:
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
    if not state.created_at:
        # First-ever write for this session id. Prefer the existing row's
        # own created_at if one is already on disk (a concurrent writer may
        # have raced this one to the first write) so two processes creating
        # the same brand-new session id can never disagree about when it
        # was created.
        state.created_at = (
            latest.get("created_at") if isinstance(latest, dict) else None
        ) or _now()
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


_PATH_BEARING_TOOLS = {"read_file", "edit_file", "write_file", "apply_patch"}


def _extract_touched_paths(messages: list[dict[str, Any]]) -> set[str]:
    """Collect every file path a tool call in this turn read or wrote.

    Scans the same tool_calls[].function.{name,arguments} shape the runner
    itself uses (see runner_local.py's token-estimate/render helpers) -- a
    best-effort scrape, not a schema validation, so a malformed or unknown
    tool call is simply skipped rather than raised.
    """
    paths: set[str] = set()
    for message in messages:
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            if str(function.get("name") or "") not in _PATH_BEARING_TOOLS:
                continue
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except (TypeError, ValueError):
                continue
            path = str(arguments.get("path") or arguments.get("file_path") or "").strip()
            if path:
                paths.add(path)
    return paths


def _fingerprint_path(workspace_root: str, path: str) -> Optional[dict[str, float]]:
    resolved = Path(path)
    if not resolved.is_absolute() and workspace_root:
        resolved = Path(workspace_root) / path
    try:
        file_stat = resolved.stat()
    except OSError:
        return None
    return {"mtime": file_stat.st_mtime, "size": file_stat.st_size}


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

    Also fingerprints (mtime, size) every file this turn's tool calls
    touched, so a resumed run can tell whether another process or session
    modified one of them while this task was interrupted -- see
    diff_checkpoint_file_fingerprint, consumed by runner_local.py's
    resumed_from_checkpoint path.
    """
    state = get_session_state(session_id)
    touched = _extract_touched_paths(messages)
    fingerprint = {
        path: fp for path in sorted(touched)
        if (fp := _fingerprint_path(state.workspace_root, path)) is not None
    }
    state.turn_checkpoint = {
        "objective": objective,
        "mode": mode,
        "status": status,
        "messages": _compact_memory_messages(messages),
        "partial_assistant": partial_assistant,
        "last_error": last_error,
        "updated_at": _now(),
        "file_fingerprint": fingerprint,
    }
    put_session_state(state)


def diff_checkpoint_file_fingerprint(
    checkpoint: dict[str, Any], workspace_root: str,
) -> dict[str, list[str]]:
    """Compare a checkpoint's recorded (mtime, size) per touched file against
    the file's current state. Never raises: an unreadable path is reported
    as "missing" rather than aborting the comparison.

    Returns {"changed": [...], "missing": [...]} (both possibly empty).
    "changed" means content likely differs since the checkpoint was saved --
    another coder's edit, a deploy, or this task's own retry after a crash.
    "missing" means the path no longer exists at all.
    """
    fingerprint = checkpoint.get("file_fingerprint") or {}
    changed: list[str] = []
    missing: list[str] = []
    for path, recorded in fingerprint.items():
        current = _fingerprint_path(workspace_root, path)
        if current is None:
            missing.append(path)
        elif (
            current["mtime"] != recorded.get("mtime")
            or current["size"] != recorded.get("size")
        ):
            changed.append(path)
    return {"changed": sorted(changed), "missing": sorted(missing)}


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




def ensure_session_title(session_id: int, objective: str) -> None:
    """No-op title writer: the mechanical (first-N-words) title generator
    this used to call was removed entirely -- titles now come ONLY from
    the LLM (upgrade_session_title_with_ai), which every task-completion
    call site already awaits.

    Two deliberate exceptions, both confirmed live:
    - A user-set title (title_source == "user", via /regenerate-title's
      rename sibling or a future rename command) is never overwritten.
    - A previously generated LLM title is kept if the objective carries
      no new substantive content (a bare "continue" should not retitle
      the session).
    Until the LLM title lands, the session has NO persisted title and
    session_display_title() falls back to live activity snapshots -- an
    honest "nothing named yet" instead of a fake first-N-words label.
    """
    if not objective or not objective.strip():
        return
    state = get_session_state(session_id)
    if state.session_title:
        return
    if not _is_substantive_objective(objective):
        return
    put_session_state(state)


# A title candidate this short (after trimming) carries no task semantics --
# "continue", "fix it", "why?" -- so the LLM should not be asked to title a
# session from it alone (wait for the first substantive message instead).
_TITLE_MIN_OBJECTIVE_CHARS = 12
# Words that mark a message as pure conversational steering ("continue",
# "go on") rather than a task statement.
_TITLE_NONSUBSTANTIVE_WORDS = frozenset({
    "continue", "again", "go", "on", "keep", "going", "next", "ok", "okay",
    "yes", "no", "done", "proceed", "resume", "more", "further", "why?",
    "hello", "hi", "hey", "thanks", "thank", "you", "me", "help", "us", "we",
    "please", "kindly", "can", "could", "would", "i", "i'm", "i'd",
})


def _is_substantive_objective(text: str) -> bool:
    """True when an objective carries enough task semantics to title a
    session from -- a semantic-content test, not merely character count:
    a short message made of real task words ("fix login bug", 13 chars)
    is substantive; a long one made of filler ("please please please help
    me please") is not.
    """
    seed = " ".join((text or "").split()).strip()
    if not seed:
        return False
    if len(seed) < _TITLE_MIN_OBJECTIVE_CHARS:
        return False
    words = [w.strip(":,.!?;").lower() for w in seed.split()]
    content_words = [w for w in words if w and w not in _TITLE_NONSUBSTANTIVE_WORDS]
    return len(content_words) >= 2


# Session titles being written right now, keyed by session id (see
# start_session_title_in_background). Holding the task here also keeps it from
# being garbage-collected while it runs.
_TITLE_INFLIGHT: dict[int, "asyncio.Task[None]"] = {}


def start_session_title_in_background(session_id: int, objective: str) -> bool:
    """Begin the LLM session title NOW, alongside the turn, without blocking it.

    Live-reported 2026-09-19 ("Session 1380884488 not title"): with the
    mechanical title gone, a session only got its LLM title once a turn
    COMPLETED, so a long or stuck first task stayed a bare "Session N" for as
    long as it ran. The title needs only the user's request, which is known the
    moment the turn starts, so it is requested then. If it fails,
    upgrade_session_title_with_ai's end-of-turn call still retries.

    Returns True when a title task was started. Needs a running event loop; with
    none (a sync caller) it does nothing. Skips swarm children, user-named
    sessions, sessions that already have a title or an attempt in flight, and
    requests too thin to title ("continue").
    """
    if not objective or not _is_substantive_objective(objective):
        return False
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return False
    existing = _TITLE_INFLIGHT.get(session_id)
    if existing is not None and not existing.done():
        return False
    try:
        state = get_session_state(session_id)
    except Exception:
        return False
    if (
        getattr(state, "is_swarm_child", False)
        or state.title_source == "user"
        or state.session_title
        or state.ai_title_attempted
    ):
        return False
    task = loop.create_task(upgrade_session_title_with_ai(session_id, objective))
    _TITLE_INFLIGHT[session_id] = task

    def _done(finished: "asyncio.Task[None]") -> None:
        if _TITLE_INFLIGHT.get(session_id) is finished:
            _TITLE_INFLIGHT.pop(session_id, None)
        if not finished.cancelled():
            finished.exception()  # consume: a title must never surface as an unhandled task error

    task.add_done_callback(_done)
    return True


async def upgrade_session_title_with_ai(session_id: int, objective: str) -> None:
    """The ONLY title writer: generate a short, semantic session title
    from the objective with an LLM via the providers system (main +
    fallback providers). Never raises and never blocks a task's
    completion on this.

    The mechanical (first-N-words) generator was removed entirely -- the
    live-reported symptom was that sessions kept getting titles like
    "Fist you need to fix" because this path silently failed and a
    truncated-objective title stood in its place. Now a failed LLM
    attempt leaves the session with NO persisted title (display falls
    back to live activity), and the attempt is retried on the next turn.

    Every call site awaits this after each completed turn, not just the
    session's first. ai_title_attempted records the one real LLM attempt
    per session; a failed attempt does NOT burn that budget -- the next
    turn retries -- so a transient provider outage on the first turn
    doesn't leave a session untitled forever. A SUCCESSFUL upgrade does.
    Provenance (title_source, title_fallback_reason, title_model,
    title_generated_at) is persisted for diagnostics; the user can
    regenerate at any time via /regenerate-title.
    """
    if not objective or not objective.strip():
        return
    # A title already being written in the background (start_session_title_in_
    # background, kicked off when the turn STARTED): wait for THAT one instead
    # of returning at once. Returning would let a short-lived process exit and
    # cancel it mid-flight, and starting a second would spend a second NIM call.
    inflight = _TITLE_INFLIGHT.get(session_id)
    if inflight is not None and inflight is not asyncio.current_task() and not inflight.done():
        await asyncio.wait({inflight}, timeout=TITLE_UPGRADE_TIMEOUT_SECONDS + 5)
        return
    state = get_session_state(session_id)
    if state.ai_title_attempted:
        return
    # Never overwrite a user-named session; keep a prior LLM title when
    # this turn's objective adds nothing substantive to title from.
    if state.title_source == "user":
        return
    if state.session_title and not _is_substantive_objective(objective):
        return
    state.ai_title_attempted = True
    put_session_state(state)

    settled = False
    try:
        await _upgrade_session_title_attempt(session_id, objective)
        settled = True
    finally:
        if not settled:
            # Cancelled (Esc, process exit) or crashed mid-attempt: the one-shot
            # budget must not stay burned, or this session would never be
            # titled at all.
            try:
                state = get_session_state(session_id)
                if not state.session_title:
                    state.ai_title_attempted = False
                    put_session_state(state)
            except Exception:
                pass


async def _upgrade_session_title_attempt(session_id: int, objective: str) -> None:
    """The LLM call, validation and persistence half of
    upgrade_session_title_with_ai (which owns the guards and the budget flag)."""
    try:
        title, model_used, failure_reason = await asyncio.wait_for(
            _generate_and_validate_title(session_id, objective),
            timeout=TITLE_UPGRADE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        print(
            f"[title] title generation exceeded {TITLE_UPGRADE_TIMEOUT_SECONDS}s across all routes, title not generated",
            file=sys.stderr,
        )
        state = get_session_state(session_id)
        state.title_fallback_reason = "provider_timeout"
        state.ai_title_attempted = False
        put_session_state(state)
        return
    if not title:
        # A failed attempt must not burn the one-shot budget: un-mark so a
        # later turn can retry when the provider is healthy again, and
        # record WHY (reason-coded, never silent).
        state = get_session_state(session_id)
        state.ai_title_attempted = False
        state.title_fallback_reason = failure_reason or "provider_error"
        put_session_state(state)
        return
    state = get_session_state(session_id)
    if state.title_source == "user":
        return  # raced with a user rename while the LLM was running
    state.session_title = title
    state.title_source = "llm"
    state.title_fallback_reason = None
    state.title_model = model_used or None
    from datetime import datetime, timezone
    state.title_generated_at = datetime.now(timezone.utc).isoformat()
    put_session_state(state)
    print(f"[title] accepted {title!r}", file=sys.stderr)


# --------------------------------------------------------------------------
# Title generation: prompt, validation, and one bounded corrective retry
# --------------------------------------------------------------------------

# A title is worth one corrective retry, not a negotiation.
_TITLE_MAX_VALIDATION_ATTEMPTS = 2
# A title must never simply BE the opening words of the request. These are the
# openers that make a candidate obviously an echo of the prompt rather than a
# description of the task.
_TITLE_BAD_OPENERS = frozenset({
    "please", "kindly", "can", "could", "would", "will", "help", "i", "we",
    "you", "first", "fist", "now", "then", "just", "so", "ok", "okay", "the",
    "a", "an", "hi", "hello", "hey", "session", "task", "request",
    "conversation", "why", "what", "how", "when", "where", "need", "needs",
    "want", "wants", "lets", "let's", "make", "lets", "there", "here",
})
# Words that carry no task semantics on their own.
_TITLE_META_WORDS = frozenset({
    "session", "task", "request", "conversation", "chat", "prompt", "user",
    "assistant", "agent", "code", "coding", "work", "working", "thing",
    "things", "stuff", "help", "issue", "problem", "stuff", "misc",
})
# Function words that betray an arbitrary truncation when a candidate is
# simply the opening words of the request ("...why image and", "...so
# multiple"): a real title names a thing or an action at its end, not a
# conjunction or a determiner.
_TITLE_TRAILING_FUNCTION_WORDS = frozenset({
    "and", "or", "so", "the", "a", "an", "with", "to", "of", "for", "in", "on",
    "at", "by", "as", "from", "into", "about", "after", "before", "than", "that",
    "which", "while", "because", "if", "when", "my", "our", "your", "its", "it",
    "this", "these", "those", "be", "is", "are", "was", "were", "not", "no",
    "do", "does", "did", "then", "next", "also", "plus", "multiple", "some",
    "all", "any", "every", "both", "first", "please", "help", "can", "could",
    "would", "should", "need", "needs", "want", "wants", "i", "we", "you",
})

# Engineering verbs a good title normally leads with.
_TITLE_ACTION_VERBS = frozenset({
    "fix", "fixing", "build", "building", "add", "adding", "implement",
    "implementing", "refactor", "refactoring", "repair", "repairing",
    "investigate", "investigating", "debug", "debugging", "improve",
    "improving", "harden", "hardening", "reconcile", "reconciling", "train",
    "training", "migrate", "migrating", "rebalance", "rebalancing",
    "redesign", "redesigning", "remove", "removing", "upgrade", "upgrading",
    "document", "documenting", "test", "testing", "profile", "optimise",
    "optimize", "optimizing", "speed", "restore", "restoring", "review",
    "reviewing", "speed", "verify", "verifying", "wire", "wiring",
    "integrate", "integrating", "generate", "generating", "configure",
    "deploy", "deploying", "export", "exporting", "repair", "retitle",
    "write", "writing", "rewrite", "rewriting", "clean", "cleanup", "split",
    "merge", "enable", "disable", "guard", "parse", "stream", "plan",
    "spreadsheet", "pipeline", "support", "stabilise", "stabilize",
})

TITLE_SYSTEM_PROMPT = (
    "You name coding sessions for an agent CLI. You are shown a request and some "
    "context; you output ONE short title naming the ENGINEERING TASK."
    "\n\nHard rules:"
    "\n- 3 to 6 words, at most 60 characters."
    "\n- Title Case (capitalise important words; lowercase articles/prepositions)."
    "\n- Describe the actual task: what is being built, fixed, or investigated."
    "\n- Start with the engineering verb or the key subject (Fix, Add, Refactor, "
    "Investigate, Repair, Harden, Migrate, Train, Reconcile, Remove...)."
    "\n- NEVER repeat the opening words of the request, and never copy a sentence "
    "from it."
    "\n- Never lead with filler or meta words (please, can you, help me, first, "
    "session, task, request, thing)."
    "\n- No quotes, no markdown, no code fences, no trailing punctuation, no "
    "explanation."
    "\n- Output the title alone and nothing else."
    "\n\nExamples:"
    "\nRequest: \"Please investigate why image and video workspace generation keeps "
    "throwing errors after the routing changes\" -> Fix Image & Video Workspace"
    "\nRequest: \"First you need to fix tamfis-code to properly keep track of "
    "sessions so users can use multiple sessions\" -> Fix Tamfis-Code Sessions"
    "\nRequest: \"Build a production-grade spreadsheet engineering skill with "
    "workbook reconciliation and validation\" -> Spreadsheet Engineering Skill"
    "\nRequest: \"Please please can you kindly help me to fix streaming because "
    "responses keep repeating\" -> Fix Streaming Repetition"
    "\nRequest: \"Train TamGPT-3.0 on the new corpus and report loss curves\" -> "
    "Train TamGPT-3.0"
)

_TITLE_CORRECTION_PROMPT = (
    "That title was rejected: {reason}. Reply with a better title only -- 3 to 6 "
    "Title Case words, starting with the engineering verb or key subject, never "
    "the request's opening words, no filler, no quotes, no trailing punctuation."
)


def _title_normalized_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if w]


def validate_session_title(
    candidate: str,
    *,
    objective: str,
    previous_title: str = "",
) -> tuple[str, str]:
    """Accept or reject one title candidate. Returns (title, "") when it is
    usable, or ("", reason) when it is not -- the reason is fed back to the
    model for one corrective retry and recorded when it never recovers.

    This is the part that makes the titles tight: the model's own output is
    checked against what a title is NOT allowed to be (an echo of the
    request's opening words, filler, meta words, a fragment, a sentence).
    """
    cleaned = _sanitize_llm_title(candidate or "")
    if not cleaned:
        return "", "empty after sanitising"
    words = cleaned.split()
    if len(words) < 2:
        return "", "a single word is not a descriptive title"
    if len(words) > 7:
        return "", f"{len(words)} words is a sentence, not a title"
    if cleaned[-1:] in ".!?:;,":
        return "", "trailing punctuation"
    lowered = [w.strip("&,").lower() for w in words]
    if lowered[0] in _TITLE_BAD_OPENERS:
        return "", f"starts with filler/meta word {lowered[0]!r}"
    if all(word in _TITLE_META_WORDS or word in _TITLE_BAD_OPENERS for word in lowered):
        return "", "contains no task-specific words"
    objective_words = _title_normalized_words(objective)
    # Compare on the SAME tokenisation as the objective (punctuation/hyphens
    # split), or "Fix tamfis-code so multiple" would not be recognised as an
    # echo of "Fix tamfis-code so multiple sessions don't leak...".
    candidate_words = _title_normalized_words(cleaned)
    if objective_words and candidate_words:
        # Echo detection: the candidate is (a prefix of) how the request began.
        prefix_len = min(len(candidate_words), len(objective_words))
        is_prefix = (
            prefix_len >= 2
            and len(candidate_words) < len(objective_words)
            and candidate_words[:prefix_len] == objective_words[:prefix_len]
        )
        if is_prefix:
            # A genuine title may legitimately quote the start of the request
            # ("Train TamGPT-3.0" for "Train TamGPT-3.0 with the new corpus...").
            # Two shapes mark an arbitrary truncation instead: it STOPS on a
            # function word ("...why image and", "...so multiple"), or it stops
            # just before the words that actually carry the task
            # ("Build a production-grade spreadsheet" | "engineering skill...").
            if candidate_words[-1] in _TITLE_TRAILING_FUNCTION_WORDS:
                return "", "it stops mid-sentence in the request's own wording, not a description of the task"
            if objective_words[prefix_len] not in _TITLE_TRAILING_FUNCTION_WORDS:
                return "", "it is the first few words of the request with the actual task left out"
        if (
            not any(word in objective_words for word in candidate_words)
            and candidate_words[0] not in _TITLE_ACTION_VERBS
        ):
            return "", "it shares no words with the request and names no engineering action"
    if previous_title and cleaned.lower() == previous_title.strip().lower():
        return "", "identical to the existing title"
    # Validation judges the model's own words; the accepted title then gets the
    # deterministic tightening pass (imperative opener, Title Case, "&",
    # identifiers preserved) so every path -- automatic titling and
    # /regenerate-title alike -- persists the same tight shape.
    return _tighten_llm_title(cleaned), ""


def _session_title_seed(state: SessionState, objective: str) -> str:
    """Bounded, most-informative-first context for the title model.

    A bare objective is a poor titling input on its own: at turn 30, the last
    message may be "also add tests", which titles nothing. The request is
    always first (it is what the session is about), followed by the earlier
    substantive user turns, the workspace name, and the files actually touched.
    """
    lines = [f"Primary request: {_one_line_label(objective.strip())[:400]}"]
    # A self-sufficient request must not be outvoted by the session's older
    # history. Live-measured 2026-09-19: retitling an "add hello message"
    # session with the full spreadsheet-engineering brief produced "Add Hello
    # Message" -- the title described where the session CAME FROM because the
    # earlier requests and touched files were part of the seed. When the
    # request itself is long enough to name the task on its own, the seed is
    # the request alone; short follow-ups ("continue the migration") still get
    # the history they need to mean anything.
    if len(objective.split()) >= 20 or len(objective.strip()) >= 160:
        return lines[0]
    earlier: list[str] = []
    for entry in list(state.conversation_history or []):
        if entry.get("role") != "user":
            continue
        content = " ".join(str(entry.get("content") or "").split())
        if not content or not _is_substantive_objective(content):
            continue
        if content == " ".join(objective.split()):
            continue
        earlier.append(_one_line_label(content)[:200])
    if earlier:
        lines.append("Earlier substantive requests: " + " | ".join(earlier[-3:]))
    active = (state.active_task or {}).get("objective") if state.active_task else None
    if active and str(active).strip() and str(active).strip() != objective.strip():
        lines.append(f"Active task: {_one_line_label(str(active))[:200]}")
    touched = [str(path) for path in list(state.inspected_files or [])[-5:]]
    if touched:
        lines.append("Files involved: " + ", ".join(touched))
    if state.workspace_root:
        lines.append(f"Workspace: {state.workspace_root}")
    return "\n".join(lines)[:2000]


def build_session_title_messages(session_id: int, objective: str) -> list[dict[str, str]]:
    """The exact title request: a strict system contract plus the bounded
    session seed. Shared by the automatic upgrade and /regenerate-title, so
    both produce the same quality of title."""
    try:
        state = get_session_state(session_id)
    except Exception:  # pragma: no cover - defensive
        state = SessionState(session_id=session_id)
    return [
        {"role": "system", "content": TITLE_SYSTEM_PROMPT},
        {"role": "user", "content": _session_title_seed(state, objective)},
    ]


async def _generate_and_validate_title(session_id: int, objective: str) -> tuple[str, str, str]:
    """Generate a title, validate it, and allow ONE corrective retry.

    Returns (title, model_used, failure_reason); title is empty on failure
    with a reason-coded failure_reason. Never raises.
    """
    messages = build_session_title_messages(session_id, objective)
    previous_title = ""
    try:
        previous_title = get_session_state(session_id).session_title or ""
    except Exception:
        previous_title = ""
    model_used = ""
    for attempt in range(1, _TITLE_MAX_VALIDATION_ATTEMPTS + 1):
        title, model_used, failure_reason = await _generate_session_title(
            messages, route_offset=attempt - 1,
        )
        if not title:
            return "", model_used, failure_reason
        accepted, reason = validate_session_title(
            title, objective=objective, previous_title=previous_title,
        )
        if accepted:
            return accepted, model_used, ""
        print(
            f"[title] attempt {attempt}/{_TITLE_MAX_VALIDATION_ATTEMPTS} rejected "
            f"({reason}): {title!r}",
            file=sys.stderr,
        )
        if attempt >= _TITLE_MAX_VALIDATION_ATTEMPTS:
            return "", model_used, "invalid_response"
        messages = [
            *messages,
            {"role": "assistant", "content": title},
            {"role": "user", "content": _TITLE_CORRECTION_PROMPT.format(reason=reason)},
        ]
    return "", model_used, "invalid_response"


async def _generate_session_title(
    messages: list, route_offset: int = 0,
) -> tuple[str, str, str]:
    """Generate a session title via the providers system's OWN routing
    machinery (ProviderManager.chat_completion pinned to NVIDIA NIM, walking
    _TITLE_NIM_MODELS -- free routes only, no cross-provider fallback).

    The previous implementation hand-rolled a per-provider loop calling
    each client directly -- bypassing route-health circuits, key rotation,
    and fallback -- and live-confirmed (2026-09) it just hung on the first
    cold provider until the timeout while the main task on the same
    machine completed fine through the machinery. One call into the real
    router is the whole fix: it resolves AUTO to a healthy weighted route
    and falls back across providers on any retryable failure.

    `route_offset` rotates the preferred-route order, so a corrective retry
    after a REJECTED candidate (not merely an empty one) does not walk into the
    same weak model that produced the rejected title. Live-measured
    2026-09-19: two of three otherwise identical regenerations came back
    perfect ("Fix Tamfis-Code Sessions", "Fix Streaming Repetition") while one
    was rejected twice and left the session untitled -- a single route unable
    to follow the contract should not be the session's only chance.

    Returns (title, model_used, failure_reason) -- title empty on total
    failure with a reason-coded failure_reason. Diagnosed, not silent.
    """
    import sys

    try:
        from .providers import ProviderManager, ProviderType
    except Exception as exc:
        print(f"[title] providers unavailable, title not generated: {_redacted(str(exc))}", file=sys.stderr)
        return "", "", "routing_failure"

    try:
        manager = ProviderManager()
    except Exception as exc:
        print(f"[title] routing unavailable, title not generated: {_redacted(str(exc))}", file=sys.stderr)
        return "", "", "routing_failure"

    # NIM only, one model per attempt (see _TITLE_NIM_MODELS). Pinning
    # ProviderType.NVIDIA with allow_fallback=False is what keeps a title from
    # ever spending Ollama Cloud / Hugging Face / OpenRouter credit: a failed
    # attempt moves to the NEXT free NIM model here, never to another provider.
    # Key rotation across the configured NIM accounts still happens inside
    # chat_completion.
    provider = ProviderType.NVIDIA
    shift = max(0, route_offset) % len(_TITLE_NIM_MODELS)
    order = list(_TITLE_NIM_MODELS[shift:] + _TITLE_NIM_MODELS[:shift])

    def _healthy(model: str) -> bool:
        try:
            return bool(manager.route_is_healthy(provider, model))
        except Exception:
            return True

    # Skip models parked by their own health circuit (a model that just timed
    # out). If EVERY model is parked, try them all anyway: a circuit is a
    # latency heuristic, not proof the route is dead, and no title at all is
    # worse than a slow one.
    models = [model for model in order if _healthy(model)] or order
    total_attempts = len(models)

    last_reason = "empty_response"
    model_used = ""
    content = ""
    for attempt, model in enumerate(models, start=1):
        chunks: list[str] = []
        # Per-attempt try, not one try around the whole loop. Live-measured
        # 2026-09-19: a single route raising (dead socket, 429, SDK error)
        # aborted EVERY remaining route and returned provider_error with no
        # title, while the very next route answered fine -- exactly the "one
        # bad model decides the session stays unnamed" failure this rotation
        # exists to avoid.
        try:
            async for chunk in manager.chat_completion(
                provider,
                messages,
                model=model,
                stream=False,
                temperature=0.3,
                # Generous on purpose: reasoning models can burn tokens
                # thinking before writing a 4-word title. Live-confirmed
                # 2026-09: 120 tokens at the machinery's default HIGH
                # reasoning effort came back with EMPTY content (the budget
                # was spent reasoning, nothing left for the answer). A large
                # budget plus low effort lets reasoning models actually
                # emit the title.
                max_tokens=600,
                reasoning_effort="low",
                allow_fallback=False,
                # Cap EACH route attempt (shorter for the slow strong models)
                # so every NIM model gets a turn inside
                # TITLE_UPGRADE_TIMEOUT_SECONDS.
                timeout=_TITLE_MODEL_TIMEOUT_SECONDS.get(
                    model, _TITLE_ATTEMPT_TIMEOUT_SECONDS,
                ),
            ):
                chunks.append(str(chunk or ""))
        except Exception as exc:
            last_reason = "provider_error"
            print(
                f"[title] attempt {attempt}/{total_attempts} "
                f"failed ({type(exc).__name__}: {_redacted(str(exc))[:120]}), "
                "trying next preferred route",
                file=sys.stderr,
            )
            continue
        content = "".join(chunks).strip()
        if content:
            # Report the model that ACTUALLY answered: the provenance field is
            # what /regenerate-title and session_title_diagnostics show, and
            # "which model named this session" is exactly the question a bad
            # title raises.
            model_used = model
            break
        last_reason = "empty_response"
        print(
            f"[title] attempt {attempt}/{total_attempts}: "
            "empty response, trying next preferred route",
            file=sys.stderr,
        )

    if not content:
        print(
            f"[title] no title after {total_attempts} preferred "
            f"routes ({last_reason})",
            file=sys.stderr,
        )
        return "", "", last_reason
    stripped = content.strip("\"'\u201c\u201d \t")
    # Validate before accepting: a real title is short, multi-word enough
    # to be informative, and never a full sentence. A response that
    # answers the objective instead of titling it is rejected so a wrong
    # label never persists.
    word_count = len(stripped.split())
    if not stripped or len(stripped) > 80 or word_count > 8 or stripped[-1:] in ".!?":
        print("[title] response rejected by title validation", file=sys.stderr)
        return "", "", "invalid_response"
    title = _sanitize_llm_title(stripped)
    if not title:
        print("[title] sanitized title empty", file=sys.stderr)
        return "", "", "invalid_response"
    return title, model_used, ""


def _sanitize_llm_title(text: str) -> str:
    """Light-touch cleanup of an LLM title candidate -- NOT a fallback
    title generator. Trims quotes/markdown fencing and terminal
    punctuation, collapses whitespace, enforces the 60-char cap.
    Returns empty when nothing survives (caller rejects and moves on).
    """
    cleaned = (text or "").strip()
    # markdown fencing a model might wrap the title in
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("` \n\t")
        if cleaned.lower().startswith("text"):
            cleaned = cleaned[4:]
    cleaned = " ".join(cleaned.split())
    # trailing punctuation is noise in a title (keep ? ! -- real info)
    while cleaned and cleaned[-1] in ".:;," :
        cleaned = cleaned[:-1].rstrip()
    if len(cleaned) > 60:
        cleaned = cleaned[:60].rstrip()
        # don't cut mid-word
        if " " in cleaned:
            cleaned = cleaned[:cleaned.rfind(" ")].rstrip()
    return cleaned


# Gerund -> imperative. A title that opens "Investigating ..." reads like a
# sentence fragment; the reference titles all open with the imperative
# ("Investigate image video workspace errors" -> "Investigate Image & Video
# Workspace", "Fix Image & Video Workspace").
_TITLE_GERUND_TO_VERB = {
    "investigating": "Investigate", "fixing": "Fix", "building": "Build",
    "adding": "Add", "implementing": "Implement", "refactoring": "Refactor",
    "repairing": "Repair", "debugging": "Debug", "improving": "Improve",
    "hardening": "Harden", "reconciling": "Reconcile", "training": "Train",
    "migrating": "Migrate", "redesigning": "Redesign", "removing": "Remove",
    "upgrading": "Upgrade", "optimising": "Optimise", "optimizing": "Optimize",
    "documenting": "Document", "testing": "Test", "profiling": "Profile",
    "integrating": "Integrate", "generating": "Generate", "configuring": "Configure",
    "deploying": "Deploy", "exporting": "Export", "writing": "Write",
    "rewriting": "Rewrite", "reviewing": "Review", "verifying": "Verify",
    "wiring": "Wire", "merging": "Merge", "splitting": "Split",
    "streaming": "Stream", "restoring": "Restore", "rebalancing": "Rebalance",
    "stabilising": "Stabilise", "stabilizing": "Stabilize", "cleaning": "Clean",
    "enabling": "Enable", "disabling": "Disable", "guarding": "Guard",
    "parsing": "Parse", "planning": "Plan", "speeding": "Speed",
    "installing": "Install", "packaging": "Package", "uploading": "Upload",
    "downloading": "Download", "rendering": "Render", "compressing": "Compress",
    "switching": "Switch", "replacing": "Replace", "updating": "Update",
    "scaling": "Scale", "freeing": "Free", "summarising": "Summarise",
    "summarizing": "Summarize", "auditing": "Audit", "benchmarking": "Benchmark",
}

# Kept lowercase inside a title (never first or last) once it is title-cased.
_TITLE_SMALL_WORDS = frozenset({
    "a", "an", "and", "or", "of", "the", "to", "for", "in", "on", "with",
    "at", "by", "as", "per", "over", "under", "via", "from", "into",
})


def _title_word_is_identifier(core: str) -> bool:
    """True when a word is a name, not prose to sentence-case: any digit
    (TamGPT-3.0, MSC-2, v1.6.52), an all-caps token (API, HTTP, PTY), or an
    internal capital (TamfisGPT, CodeBuff). These are preserved exactly."""
    if not core:
        return False
    if any(ch.isdigit() for ch in core):
        return True
    if len(core) >= 2 and core.isupper():
        return True
    return any(ch.isupper() for ch in core[1:])


def _tighten_llm_title(title: str) -> str:
    """Deterministic quality pass over an ACCEPTED LLM title. Not a title
    generator: it only re-shapes words that are already in the model's own
    candidate, so a rejected title can never sneak back in through here.

    Live-reported titles were correct but loose ("Invesitgate image video
    workspace errors", "Add docstring and subtract function"). What makes a
    title read as a task name instead of a truncated sentence:
      * the imperative instead of a gerund opener,
      * Title Case with product/API names left exactly as written,
      * "&" between the two halves of a compound task,
      * no trailing punctuation.
    """
    cleaned = _sanitize_llm_title(title)
    if not cleaned:
        return ""
    words = cleaned.split()
    opener = words[0].strip("()[],;:&'\"")
    verb = _TITLE_GERUND_TO_VERB.get(opener.lower())
    if verb:
        words[0] = verb + words[0][len(opener):]
    # "Image and Video" -> "Image & Video": the ampersand form is tighter and
    # is what every reference title uses for a compound subject.
    if len(words) > 2:
        joined = " ".join(words)
        joined = re.sub(
            r"(\b[A-Za-z][A-Za-z0-9-]*)\s+and\s+([A-Za-z][A-Za-z0-9-]*\b)",
            r"\1 & \2", joined, flags=re.IGNORECASE,
        )
        words = joined.split()
    cased: list[str] = []
    last = len(words) - 1
    for index, word in enumerate(words):
        core = word.strip("()[],;:&'\"")
        if not core:
            cased.append(word)
            continue
        start = word.index(core)
        prefix, suffix = word[:start], word[start + len(core):]
        if _title_word_is_identifier(core):
            body = core
        elif (
            "-" in core
            and all(part.islower() for part in core.split("-") if part)
        ):
            # A hyphenated product name the model wrote in lowercase
            # ("tamfis-code", "kimi-code") is a NAME: capitalise each segment
            # rather than just the first letter, or the title reads as prose
            # ("Fix Tamfis-code Sessions").
            body = "-".join(
                part[:1].upper() + part[1:] for part in core.split("-")
            )
        elif core.lower() in _TITLE_SMALL_WORDS and 0 < index < last:
            body = core.lower()
        else:
            body = core[0].upper() + core[1:]
        cased.append(prefix + body + suffix)
    text = " ".join(" ".join(cased).split())
    # Same envelope the validator enforces: if the reshaping somehow pushed it
    # out of bounds, keep the validated (untightened) form rather than a
    # truncated, mangled one.
    if not text or len(text) > 60 or len(text.split()) > 7:
        return cleaned
    return text


def best_effort_session_label(state: SessionState) -> str:
    """One-line summary of what a session is doing RIGHT NOW, straight
    from already-recorded activity (active_task objective, then the most
    recent SUBSTANTIVE user turn, then conversation_summary) -- no persisted
    session_title required. Empty string if nothing usable is recorded yet
    (a session with no turns at all).

    Deliberately NOT a title generator: with the mechanical first-N-words
    generator removed, this is a transient DISPLAY fallback only (footer /
    resume picker "what is this session doing"), never persisted to
    session_title -- so nothing that looks like "first few words of the
    prompt" can ever become the session's name. Multi-line seeds are
    collapsed via _one_line_label so a raw newline can never reach the
    resume picker's or footer's single-line rendering.

    Shared by session_display_title's fallback below and workspace.py's
    `_describe_session_activity` (its "detail" line).
    """
    objective = str((state.active_task or {}).get("objective") or "").strip()
    if objective:
        return _one_line_label(objective)
    # The user's own most recent substantive request describes the session far
    # better than the assistant's last line: conversation_summary holds the
    # ASSISTANT's final answer, so preferring it labelled a real session
    # "Done." -- useless in the resume picker. Scan back past trivial
    # acknowledgements ("continue", "ok") to the last request that actually
    # says something.
    trivial_fallback = ""
    for entry in reversed(state.conversation_history or []):
        if entry.get("role") != "user":
            continue
        content = str(entry.get("content") or "").strip()
        if not content:
            continue
        candidate = _one_line_label(content.splitlines()[0])
        if not candidate:
            continue
        if candidate.lower().rstrip(".!") in _TRIVIAL_ACTIVITY_LABELS:
            trivial_fallback = trivial_fallback or candidate
            continue
        return candidate
    if state.conversation_summary:
        last_line = _one_line_label(state.conversation_summary.strip().splitlines()[-1])
        if last_line and last_line.lower().rstrip(".!") not in _TRIVIAL_ACTIVITY_LABELS:
            return last_line
    return trivial_fallback


# Labels that carry no information about what a session is about. Used ONLY
# by best_effort_session_label's display fallback -- never by title
# generation, which is LLM-only (see _generate_session_title).
_TRIVIAL_ACTIVITY_LABELS = frozenset({
    "done", "ok", "okay", "k", "yes", "y", "yep", "ya", "no", "n", "nope",
    "thanks", "thank you", "ty", "continue", "go on", "proceed", "sure",
    "hi", "hello", "hey", "test", "nice", "great", "cool", "wtf", "why",
})


def _redacted(text: str) -> str:
    """Backend names out of a diagnostic line that lands on a user's terminal."""
    from .public_identity import redact_routing_text

    return redact_routing_text(text)


def _one_line_label(text: str) -> str:
    """Collapse a multi-line seed onto one display line, hard-capped.
    Pure display hygiene -- no word selection, no title semantics."""
    line = " ".join((text or "").split()).strip()
    if not line:
        return ""
    return line[:60] + ("…" if len(line) > 60 else "")


def session_display_title(session_id: int) -> str:
    """Stable display name for a session, for the resume picker, the
    `sessions` listing, and the persistent footer.

    The name is the LLM-written session_title (or the user's own rename) and
    nothing else. Owner ruling 2026-09-19: NO mechanical title anywhere, not
    even as a display fallback -- the old fallback showed the first ~60
    characters of the prompt in the footer, which is exactly the "first few
    words of the request" label the LLM title exists to replace. Until the LLM
    title lands (it is generated when the first turn completes, and retried on
    every later turn if the routes were down) the session shows the neutral
    "Session N". What the session is doing right now is a different question,
    answered by best_effort_session_label for the resume picker's DETAIL line.
    """
    state = get_session_state(session_id)
    return state.session_title or f"Session {session_id}"


def session_has_recorded_activity(state: SessionState) -> bool:
    """True once a session has any real conversational activity -- exactly
    the same condition that gives session_display_title something better
    than a bare "Session N" to show.

    Live-reported: session ids kept appearing in the resume picker/
    `sessions` listing as permanently title-less "Session N" rows,
    crowding out real conversations by recency. Cause: resolve_local_
    workspace() (workspace.py) -- which mints/reuses a session id for a
    directory -- is called by read-only, non-conversational commands too
    (`doctor`, `sessions` itself), not just real turns, so simply running
    `tamfis-code doctor` in a fresh directory permanently registers an
    empty session with nothing to ever resume. Callers building a
    resumable-sessions listing should filter on this rather than show
    every known session id unconditionally.
    """
    return bool(state.session_title or best_effort_session_label(state))


def set_session_archived(session_id: int, archived: bool) -> None:
    """Soft-hide (or restore) a session from the resume picker's default
    "Active" view -- Ctrl+A there. Reversible and non-destructive: every
    other field, including conversation history, is untouched. Distinct
    from `tamfis-code clear-session`, the only operation that actually
    erases a session from active listings."""
    state = get_session_state(session_id)
    state.archived = archived
    put_session_state(state)


def rename_session_title(session_id: int, title: str) -> bool:
    """User-explicit rename. Persists with title_source="user"; automatic
    generation NEVER overwrites a user title afterwards. Returns True
    when applied."""
    cleaned = _sanitize_llm_title(title)
    if not cleaned:
        return False
    state = get_session_state(session_id)
    state.session_title = cleaned
    state.title_source = "user"
    state.title_fallback_reason = None
    from datetime import datetime, timezone
    state.title_generated_at = datetime.now(timezone.utc).isoformat()
    put_session_state(state)
    return True


def is_substantive_objective(text: str) -> bool:
    """Public form of the substantive-message test (see
    _is_substantive_objective) -- used by /regenerate-title to decide which
    recorded turn is worth titling the session from."""
    return _is_substantive_objective(text)


def title_route_preference() -> list[str]:
    """The NIM models a title attempt walks, in order. Internal provenance
    only: callers that show a user something must count them or brand them
    (see public_identity), never print these ids."""
    return list(_TITLE_NIM_MODELS)


def request_session_title_regeneration(session_id: int) -> bool:
    """Clear the current title so the next upgrade_session_title_with_ai
    (invoked with the session's own objective) regenerates it from the
    full conversation. A user-sourced title is never touched.
    """
    state = get_session_state(session_id)
    if state.title_source == "user":
        return False
    state.session_title = ""
    state.ai_title_attempted = False
    state.title_source = ""
    state.title_fallback_reason = None
    state.title_model = None
    state.title_generated_at = None
    put_session_state(state)
    return True


ROUTE_EVENT_LIMIT = 20

# Per-session response-time samples, so /routes (and `--json`) can still show
# which route was slow after a restart -- and can chart ONE session's timings
# instead of whatever this process happened to call. Bounded on both axes for
# the same reason every other durable list here is (see _enforce_state_caps):
# state.json is re-serialized on every event of a long session.
ROUTE_LATENCY_SAMPLES_PER_PROVIDER = 50
ROUTE_LATENCY_PROVIDER_LIMIT = 12


# Errors that mean "this ACCOUNT cannot serve the request" rather than "this
# request is bad": the route is dead until credits/quota/billing are fixed, so
# the only useful recovery is another provider. Matched alongside the numeric
# status (402 payment required, 401/403, 429) so a wrapper that lost the
# status attribute is still recognised by its wording.
_ROUTE_EXHAUSTION_MARKERS = (
    "depleted",
    "out of credits",
    "insufficient credits",
    "included credits",
    "pre-paid credits",
    "prepaid",
    "credit balance",
    "billing",
    "payment required",
    "quota",
    "rate limit",
    "402",
    "401",
    "403",
    "429",
)


def route_error_is_exhaustion(error: Any) -> bool:
    """True when a provider error is an account-level exhaustion (credits,
    quota, rate limit) rather than a bad request."""
    text = str(error or "").lower()
    return any(marker in text for marker in _ROUTE_EXHAUSTION_MARKERS)


def record_route_event(
    session_id: int,
    *,
    provider: str,
    model: str = "",
    previous_provider: str = "",
    previous_model: str = "",
    reason: str = "",
    kind: str = "select",
    exhausted: bool = False,
) -> None:
    """Persist one route change for /status and the footer. Never raises:
    route bookkeeping must not be able to fail a task."""
    try:
        state = get_session_state(session_id)
        events = [event for event in (state.route_events or [])]
        events.append({
            "at": _now(),
            "kind": kind,
            "provider": str(provider or ""),
            "model": str(model or ""),
            "from_provider": str(previous_provider or ""),
            "from_model": str(previous_model or ""),
            "reason": str(reason or "")[:400],
            "exhausted": bool(exhausted),
        })
        state.route_events = events[-ROUTE_EVENT_LIMIT:]
        put_session_state(state)
    except Exception:
        return


def record_route_error(
    session_id: int, *, provider: str, model: str = "", error: str = "",
) -> None:
    """Persist an ACCOUNT-LEVEL route failure (credits/quota exhausted).

    Kept separate from record_route_event's provider switches so /status can
    report "the route you were on is exhausted, and here is what it said" even
    when the task then fell back successfully (or could not fall back at all).
    """
    try:
        state = get_session_state(session_id)
        events = [event for event in (state.route_events or [])]
        events.append({
            "at": _now(),
            "kind": "exhaustion",
            "provider": str(provider or ""),
            "model": str(model or ""),
            "reason": str(error or "")[:400],
            "exhausted": True,
        })
        state.route_events = events[-ROUTE_EVENT_LIMIT:]
        put_session_state(state)
    except Exception:
        return


def route_diagnostics(session_id: int) -> dict:
    """What route this session is on, and how it got there.

    Returns current provider/model, the failovers seen (with the reason and
    whether the old route was EXHAUSTED rather than broken), and the list of
    routes currently cooling down after recent failures.
    """
    try:
        state = get_session_state(session_id)
    except Exception:
        return {"current": None, "failovers": [], "cooling": [], "events": []}
    events = list(state.route_events or [])
    # The current route is whichever route was established most recently -- a
    # selection or a failover. Preferring "select" here reported the route the
    # session STARTED on even after it had failed over to a healthy one.
    current = next(
        (
            event for event in reversed(events)
            if event.get("kind") in {"select", "failover", "recovery"}
        ),
        None,
    )
    failovers = [event for event in events if event.get("kind") in {"failover", "recovery"}]
    exhausted = [event for event in events if event.get("kind") == "exhaustion"]
    return {
        "current": current,
        "failovers": failovers,
        "exhausted": exhausted,
        "last_exhaustion": exhausted[-1] if exhausted else None,
        "cooling": cooling_route_names(),
        "events": events,
    }


def cooling_route_names() -> list[str]:
    """Routes whose health circuit is currently open (recent failures), from
    the providers layer. Empty when the providers module cannot be consulted."""
    try:
        import time as _time

        from .providers import _HEALTH_LOCK, _ROUTE_HEALTH  # noqa: PLC0415

        now = _time.monotonic()
        names: list[str] = []
        with _HEALTH_LOCK:
            for (provider_name, _model), health in _ROUTE_HEALTH.items():
                if health.circuit_open_until > now and provider_name not in names:
                    names.append(str(provider_name))
        return names
    except Exception:
        return []


def route_status_line(
    session_id: int, *, include_current: bool = True, max_chars: int = 0,
) -> str:
    """One compact route line in product vocabulary, e.g.
    "TamfisGPT · 1 failover · route exhausted (credits/quota)".

    Provider and model names are deployment details and never appear here
    (owner ruling 2026-09-19: a footer reading "nvidia→ollama_cloud · 2
    failovers" exposed the backends). What the user needs is that the route
    changed, how often, and whether one ran out of credit -- the per-route
    detail is in /routes, which brands each backend as "TamfisGPT (alt N)".

    The persistent footer already names the current model, so it passes
    `include_current=False` and gets ONLY the durable exceptions -- a
    failover or an exhausted route. (Transient provider circuits cooling down
    are /routes material, not footer material -- see route_status_compact.)
    """
    from .public_identity import PUBLIC_PROVIDER_NAME

    diag = route_diagnostics(session_id)
    current = diag.get("current") or {}
    count = len(diag.get("failovers") or [])
    last_exhaustion = diag.get("last_exhaustion") or {}
    if not current.get("provider") and not last_exhaustion:
        return ""
    if not include_current and not count and not last_exhaustion:
        return ""
    label = PUBLIC_PROVIDER_NAME
    if count:
        label += f" · {count} failover{'s' if count != 1 else ''}"
    if last_exhaustion:
        label += " · route exhausted (credits/quota)"
    # The footer shares one terminal line with the session title, mode, and
    # agents, and a long note wraps and squeezes them out (live-observed in a
    # pty at 80 columns). The footer asks for the short form; /status, which
    # owns a whole block, keeps the full text.
    if max_chars and len(label) > max_chars:
        label = label[: max(1, max_chars - 1)].rstrip(" ·→,") + "…"
    return label


def route_history(session_id: int) -> list[dict]:
    """This session's route timeline, oldest first, with how long each route
    actually HELD the task.

    Each returned entry is the stored event plus `held_seconds` -- the gap to
    the next route event (or to now, for the route still in effect). "How long
    did each provider hold the task" is the question a route-churn complaint
    actually asks, and the raw event timestamps alone do not answer it.
    """
    diag = route_diagnostics(session_id)
    events = [dict(event) for event in (diag.get("events") or [])]
    for index, event in enumerate(events):
        if event.get("kind") not in {"select", "failover", "recovery"}:
            # An exhaustion note describes the CURRENT route, it does not hold
            # the task for any length of time -- reporting a duration for it
            # would put a meaningless number in the /routes table.
            event["held_seconds"] = None
            continue
        start = _parse_route_timestamp(event.get("at"))
        following = next(
            (
                other for other in events[index + 1:]
                # Only a route CHANGE ends a route's hold; an exhaustion note is
                # about the same route, not a new one.
                if other.get("kind") in {"select", "failover", "recovery"}
            ),
            None,
        )
        end = (
            _parse_route_timestamp(following.get("at")) if following is not None
            else _parse_route_timestamp(_now())
        )
        if start is not None and end is not None and end >= start:
            event["held_seconds"] = round((end - start).total_seconds(), 1)
        else:
            event["held_seconds"] = None
    return events


def route_hold_totals(session_id: int) -> list[tuple[str, float, int]]:
    """(provider, seconds_held, turns) per provider, longest-held first."""
    totals: dict[str, list[float]] = {}
    for event in route_history(session_id):
        if event.get("kind") not in {"select", "failover", "recovery"}:
            continue
        provider = str(event.get("provider") or "")
        if not provider:
            continue
        held = event.get("held_seconds")
        entry = totals.setdefault(provider, [0.0, 0.0])
        if isinstance(held, (int, float)):
            entry[0] += float(held)
        entry[1] += 1
    return sorted(
        ((provider, values[0], int(values[1])) for provider, values in totals.items()),
        key=lambda item: item[1],
        reverse=True,
    )


def _parse_route_timestamp(value: Any) -> Optional[datetime]:
    """Parse a stored route timestamp; None when it is missing/unparseable."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def record_route_latency(
    session_id: int, provider: Any, seconds: float, *, ok: bool = True,
) -> None:
    """Persist one provider round-trip against THIS session. Never raises:
    latency bookkeeping must not be able to fail a task.

    A failed attempt is recorded too (its duration is often the symptom -- a
    route that hangs until the first-byte timeout contributes that full
    timeout), and counted in `route_latency_failures` so a fast p50 caused by
    instant failures can never be mistaken for a healthy route.
    """
    try:
        name = str(getattr(provider, "value", provider) or "")
        value = float(seconds)
    except (TypeError, ValueError):
        return
    if not name or value < 0:
        return
    try:
        state = get_session_state(session_id)
        samples = state.route_latency if isinstance(state.route_latency, dict) else {}
        state.route_latency = samples
        samples[name] = list(samples.get(name) or []) + [round(value, 3)]
        if len(samples[name]) > ROUTE_LATENCY_SAMPLES_PER_PROVIDER:
            samples[name] = samples[name][-ROUTE_LATENCY_SAMPLES_PER_PROVIDER:]
        if not ok:
            failures = (
                state.route_latency_failures
                if isinstance(state.route_latency_failures, dict) else {}
            )
            state.route_latency_failures = failures
            failures[name] = int(failures.get(name) or 0) + 1
        put_session_state(state)
    except Exception:
        return


def route_latency_stats(session_id: int) -> tuple[dict, str]:
    """(percentiles, source) for /routes and `--json`.

    Prefers this session's own persisted samples -- that is what "how slow has
    THIS session been, and on which route" means, and it survives a restart.
    Falls back to the current process's registry (source "process") for the
    calls that have no session to attribute themselves to, such as the plan /
    title / classifier requests, so the panel is not simply blank early on.
    """
    from .providers import latency_stats_from_samples, provider_latency_stats

    try:
        state = get_session_state(session_id)
        samples = state.route_latency if isinstance(state.route_latency, dict) else {}
        failures = (
            state.route_latency_failures
            if isinstance(state.route_latency_failures, dict) else {}
        )
    except Exception:
        samples, failures = {}, {}
    stats = latency_stats_from_samples(samples, failures)
    if stats:
        return stats, "session"
    try:
        process_stats = provider_latency_stats()
    except Exception:
        process_stats = {}
    return process_stats, ("process" if process_stats else "none")


def route_report(session_id: int) -> dict:
    """Everything known about this session's routing, as plain data.

    One structure for both the /routes table and `--json` (so a chart is drawn
    from the same numbers the table shows): what route is live, the timeline
    with how long each held the task, per-provider totals, per-provider latency
    percentiles, the last account-level failure, and any routes cooling down.
    """
    diag = route_diagnostics(session_id)
    history = route_history(session_id)
    latency, latency_source = route_latency_stats(session_id)
    return {
        "session_id": session_id,
        "current": diag.get("current"),
        "events": history,
        "failovers": diag.get("failovers") or [],
        "last_exhaustion": diag.get("last_exhaustion"),
        "cooling": diag.get("cooling") or [],
        "hold_totals": [
            {"provider": provider, "held_seconds": round(seconds, 1), "turns": turns}
            for provider, seconds, turns in route_hold_totals(session_id)
        ],
        # latency_source is "session" when these numbers come from the samples
        # this session recorded (the usual case), "process" when they are this
        # process's registry because the session has none yet, "none" when
        # neither exists -- so a chart never silently mixes the two.
        "latency": latency,
        "latency_source": latency_source,
    }


def public_route_event(event: dict, label: Any) -> dict:
    """One stored route event in product vocabulary: providers become
    RouteLabeler labels, model ids become public tier names, and free text
    (a provider's own error message) has backend names redacted."""
    from .public_identity import public_model_name, redact_routing_text

    branded = dict(event)
    # from_provider first: it is the EARLIER route, so labels are handed out in
    # the order the task actually moved through them.
    for key in ("from_provider", "provider"):
        if branded.get(key):
            branded[key] = label(branded[key])
    for key in ("model", "from_model"):
        if branded.get(key):
            branded[key] = public_model_name(branded[key])
    if branded.get("reason"):
        branded["reason"] = redact_routing_text(branded["reason"])
    return branded


def public_route_report(session_id: int, label: Any = None) -> dict:
    """route_report with every backend name branded -- what a user may see.

    Labels are assigned in timeline order (oldest event first) so the first
    route is "TamfisGPT" and each further distinct backend "TamfisGPT (alt N)",
    consistently across events, totals, latency and cooling.
    """
    from .public_identity import RouteLabeler

    label = label or RouteLabeler()
    report = route_report(session_id)
    events = [public_route_event(event, label) for event in report.get("events") or []]
    report["events"] = events
    report["failovers"] = [
        public_route_event(event, label) for event in report.get("failovers") or []
    ]
    for key in ("current", "last_exhaustion"):
        if report.get(key):
            report[key] = public_route_event(report[key], label)
    report["hold_totals"] = [
        {**entry, "provider": label(entry.get("provider"))}
        for entry in report.get("hold_totals") or []
    ]
    # Distinct raw providers can collapse onto one label only if they were the
    # same provider, so merging by label never mixes two backends.
    merged: dict[str, dict] = {}
    for entry in report["hold_totals"]:
        into = merged.setdefault(
            entry["provider"], {"provider": entry["provider"], "held_seconds": 0.0, "turns": 0},
        )
        into["held_seconds"] = round(into["held_seconds"] + float(entry.get("held_seconds") or 0), 1)
        into["turns"] += int(entry.get("turns") or 0)
    report["hold_totals"] = list(merged.values())
    report["latency"] = {
        label(provider): values for provider, values in (report.get("latency") or {}).items()
    }
    report["cooling"] = [label(name) for name in report.get("cooling") or []]
    return report


def route_report_json(session_id: int) -> str:
    """`/routes --json`: the route report as formatted JSON for charting.
    Branded (public_route_report) -- this is user-facing output."""
    return json.dumps(public_route_report(session_id), indent=2, sort_keys=True)


def public_route_status_lines(session_id: int) -> str:
    """The route section of /status, in product vocabulary (leading newline
    per line, ready to append to the status block; empty when nothing is
    recorded). Same persisted record as the footer and /routes."""
    report = public_route_report(session_id)
    lines = ""
    current = report.get("current") or {}
    if current.get("provider"):
        lines += f"\nroute={current['provider']}"
        if current.get("from_provider") and current["from_provider"] != current["provider"]:
            lines += f"  (was {current['from_provider']})"
    last_exhaustion = report.get("last_exhaustion") or {}
    if last_exhaustion:
        lines += (
            f"\nexhausted={last_exhaustion.get('provider')}"
            f"  at={str(last_exhaustion.get('at') or '')[:19]}"
            f"  reason={str(last_exhaustion.get('reason') or '')[:160]}"
        )
    for event in report.get("failovers") or []:
        lines += (
            f"\nfailover={event.get('from_provider') or '?'} -> {event.get('provider')}"
            f"  at={str(event.get('at') or '')[:19]}"
            f"  reason={str(event.get('reason') or '')[:120]}"
        )
    if report.get("cooling"):
        lines += f"\ncooling={', '.join(report['cooling'])}"
    return lines


def route_latency_lines(session_id: int, label: Any = None) -> list[str]:
    """Human lines for /routes: per-provider p50/p95 response time.

    Ordered slowest-p95 first, because the question this answers is "which
    route is making turns slow" -- the top line is the answer. Totals (held
    per route) can blame a route for time it spent waiting on a slow provider;
    these percentiles are what separate "this route is broken" from "this
    route is slow but working".
    """
    stats, source = route_latency_stats(session_id)
    lines = []
    for provider, values in sorted(
        stats.items(), key=lambda item: item[1].get("p95", 0), reverse=True,
    ):
        failures = int(values.get("failures", 0) or 0)
        failed_note = f", {failures} failed" if failures else ""
        lines.append(
            f"{label(provider) if label else provider}: "
            f"p50 {values.get('p50', 0):.1f}s  p95 {values.get('p95', 0):.1f}s  "
            f"max {values.get('max', 0):.1f}s  (n={int(values.get('samples', 0))}{failed_note})"
        )
    if lines and source == "process":
        # Say so: these are not this session's numbers, and pretending they are
        # would make a chart of "this session" quietly wrong.
        lines.append("(no samples recorded for this session yet -- showing this process's)")
    return lines


def route_status_compact(session_id: int, max_chars: int = 56) -> str:
    """The FOOTER form of the route note: only THAT the route changed and why,
    small enough to sit beside the session title and mode without being
    clipped by an 80-column terminal ("⟳ rerouted · credits · 2 failovers").

    Empty when this session has nothing exceptional to report, so a healthy
    turn's footer is unchanged. Never names a provider or model: those are
    deployment details (a footer reading "⟳ nvidia→ollama_cloud · 2
    failovers" was live-reported as exposing the backends).
    """
    diag = route_diagnostics(session_id)
    failovers = diag.get("failovers") or []
    last_exhaustion = diag.get("last_exhaustion") or {}
    # Cooling routes are deliberately NOT surfaced here. A failed request
    # briefly opens a provider's 30s health circuit as part of NORMAL
    # self-healing, so "cooling" fires constantly (e.g. nvidia + ollama_cloud
    # on a busy day), tells the user nothing they can act on, and sits in the
    # footer eating width from the title and mode -- live-reported as
    # "⟳ · cooling: nvidia,ollama_cloud⠇" on a session that was working fine.
    # The failure it describes is either transient (routing already healed) or
    # durable (an exhaustion/failover, which IS shown). Full cooling detail
    # stays in /routes, which owns a whole screen.
    if not failovers and not last_exhaustion:
        return ""
    # A failover means the task really did move to another route; an
    # exhaustion with no failover means the route is down and nothing took over.
    label = "rerouted" if failovers else "route down"
    # Budget-aware suffix order: the REASON outranks the count, because
    # "the route ran out of credits" is what explains a behaviour change and
    # the count is only trivia. Each extra is added only while it still fits,
    # so dropping the count never turns "· credits" into "· cre…" on an
    # 80-column terminal -- the one word that mattered, unreadable.
    extras = []
    if last_exhaustion:
        extras.append("credits")
    if failovers:
        extras.append(f"{len(failovers)} failover{'s' if len(failovers) != 1 else ''}")
    note = f"⟳ {label}"
    for extra in extras:
        candidate = f"{note} · {extra}"
        if not max_chars or len(candidate) <= max_chars:
            note = candidate
    if max_chars and len(note) > max_chars:
        note = note[: max(1, max_chars - 1)].rstrip(" ·→,") + "…"
    return note


def session_title_diagnostics(session_id: int) -> dict:
    """Inspectable title provenance for debugging -- the acceptance
    contract's diagnostic record (session_id, title, source, model,
    generated_at, fallback_reason)."""
    state = get_session_state(session_id)
    return {
        "session_id": session_id,
        "title": state.session_title or None,
        "title_source": state.title_source or None,
        "title_model": state.title_model,
        "title_generated_at": state.title_generated_at,
        "fallback_reason": state.title_fallback_reason,
    }


def remember_conversation_turn(
    session_id: int, *, objective: str, answer: str, clear_checkpoint: bool = False,
) -> None:
    """Append a completed local turn to durable, bounded session memory."""
    ensure_session_title(session_id, objective)
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


def edit_queued_instruction(session_id: int, instruction_id: str, text: str) -> bool:
    """Replace the text of an instruction that has not started yet."""
    replacement = text.strip()
    if not replacement:
        return False
    state = get_session_state(session_id)
    for item in state.queued_user_instructions:
        if item.get("id") == instruction_id and item.get("status") == "queued":
            item["text"] = replacement
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


def update_task_state(session_id: int, **updates: Any) -> dict[str, Any]:
    """Merge a bounded, JSON-safe long-horizon task ledger into session state."""
    state = get_session_state(session_id)
    ledger = dict(state.task_state or {})
    for key, value in updates.items():
        if isinstance(value, list):
            # Keep durable ledgers bounded; old detail remains available in
            # completed_actions/validation_results and external event logs.
            value = value[-300:]
        ledger[key] = _sanitize(value)
    ledger.setdefault("schema_version", 1)
    ledger["updated_at"] = _now()
    state.task_state = ledger
    put_session_state(state)
    return ledger


def task_checkpoint(session_id: int, *, reason: str, next_action: str = "",
                    **updates: Any) -> dict[str, Any]:
    """Persist task ledger and an indexed checkpoint atomically at a milestone."""
    ledger = update_task_state(session_id, **updates)
    state = get_session_state(session_id)
    entry = {
        "checkpoint_id": f"cp_{uuid.uuid4().hex[:12]}",
        "created_at": _now(), "reason": reason,
        "phase": state.current_phase, "next_action": next_action,
        "task_id": state.last_task_id,
        "task_state": {k: v for k, v in ledger.items() if k != "context_summary"},
    }
    state.context_checkpoints = (state.context_checkpoints + [entry])[-MAX_CHECKPOINTS:]
    put_session_state(state)
    return entry


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


def _task_compat_tokens(text: str) -> set[str]:
    """Content tokens for task-identity comparison (2026-09-17 session-leak fix).

    Stop words and generic scaffolding are dropped; surviving tokens are at
    least 3 characters so greetings/one-word objectives never match anything.
    """
    stop = {
        "a", "an", "the", "and", "or", "but", "if", "then", "else", "for", "to", "of",
        "in", "on", "at", "by", "with", "from", "into", "onto", "about", "as", "is",
        "are", "was", "were", "be", "been", "being", "do", "does", "did", "done",
        "please", "kindly", "just", "also", "again", "now", "this", "that", "these",
        "those", "it", "its", "he", "she", "they", "them", "their", "you", "your",
        "i", "me", "my", "we", "our", "us", "can", "could", "should", "would",
        "will", "shall", "may", "might", "must", "have", "has", "had", "not", "no",
        "yes", "ok", "okay", "fix", "add", "make", "work", "use", "using", "get",
        "all", "any", "some", "more", "new", "try", "run", "code", "file", "files",
    }
    tokens = re.findall(r"[a-z0-9_][a-z0-9_+-]{2,}", str(text or "").casefold())
    cleaned = {token.strip("_+-") for token in tokens}
    return {token for token in cleaned if len(token) >= 3 and token not in stop}


def task_objectives_compatible(incoming: str, stored: str) -> bool:
    """True when ``incoming`` and ``stored`` plausibly describe the same task.

    2026-09-17 (session-leak fix) helper. Two objectives are compatible when
    the incoming one is a short continuation-shaped message ("continue",
    "also add a totals column", "proceed") over an existing stored task, or
    when both texts share at least one distinctive content token. A greeting
    ("hello"), a distinct new task ("fix the payment bug" vs "write the api
    tests"), or an empty stored objective never matches.
    """
    incoming_text = str(incoming or "").strip()
    stored_text = str(stored or "").strip()
    if not incoming_text or not stored_text:
        return False
    incoming_lower = incoming_text.casefold()
    if len(incoming_lower.split()) <= 8 and re.search(
        r"(?:^|\s)(?:continue|resume|proceed|carry\s*on|go\s*on|keep\s*going|next|"
        r"step\s*\d+|also|additionally|plus|instead|rather|actually|retry|again|"
        r"ok(?:ay)?|sure|yes|no|stop|cancel)(?:\s|$|[,.;:!])",
        incoming_lower,
    ):
        return True
    incoming_tokens = _task_compat_tokens(incoming_lower)
    stored_tokens = _task_compat_tokens(stored_text.casefold())
    if not incoming_tokens or not stored_tokens:
        return False
    return bool(incoming_tokens & stored_tokens)


def mint_or_reuse_session_id(
    workspace_root: str, *, exclude_actively_running: bool = True,
    objective: Optional[str] = None,
) -> tuple[int, bool]:
    """Atomically pick the session id for a workspace launch: reuse the
    most recently updated idle session already rooted at ``workspace_root``
    (never a swarm child, never one actively running in another process),
    or mint the next fresh id. Returns ``(session_id, reused)``.

    The whole decision runs under the cross-process state lock so two
    concurrent launches in the same directory cannot both observe "no
    idle session exists" and mint two different ids for the same
    workspace -- the root cause behind one logical conversation showing
    up as several near-identical rows in `tamfis-code resume` (each new
    launch minted its own id, then recorded the same workspace and the
    same opening objective, so the picker listed the same conversation
    several times under different ids).

    2026-09-17 (session-leak fix): reuse is now **task-aware** when an
    ``objective`` is supplied. A stored session is only reused when its
    recorded task context is empty OR its objective is compatible with the
    incoming objective. Previously every launch for a workspace silently
    inherited the most-recent idle session's full task context -- 31 live
    sessions all carried primary_workspace=/home, each from an unrelated
    task, so a fresh launch landed on another conversation's plans,
    checkpoints, and history. Legacy callers that pass no objective keep
    the previous most-recent-match behaviour.
    """
    with state_lock():
        candidates = [
            (sid, get_session_state(sid)) for sid in all_known_session_ids()
        ]
        matching = [
            (sid, state) for sid, state in candidates
            if state.primary_workspace == workspace_root
            and not state.is_swarm_child
            and not (exclude_actively_running and is_session_actively_running(state))
        ]
        if matching:
            if objective is None:
                # Legacy call sites that have no objective yet (workspace
                # bookkeeping, read-only helpers): previous behaviour.
                matching.sort(key=lambda pair: pair[1].updated_at or "", reverse=True)
                return matching[0][0], True
            compatible = []
            for sid, state in matching:
                checkpoint_objective = ""
                if state.turn_checkpoint:
                    checkpoint_objective = str(state.turn_checkpoint.get("objective") or "")
                stored_objective = str(
                    (state.active_task or {}).get("objective")
                    or checkpoint_objective
                    or (state.saved_plans[-1].get("objective") if state.saved_plans and isinstance(state.saved_plans[-1], dict) else "")
                    or ""
                )
                has_task_context = bool(
                    stored_objective
                    or state.conversation_history
                    or state.turn_checkpoint
                    or state.active_plan_id
                )
                if not has_task_context or task_objectives_compatible(objective, stored_objective):
                    compatible.append((sid, state))
            if compatible:
                compatible.sort(key=lambda pair: pair[1].updated_at or "", reverse=True)
                return compatible[0][0], True
        known = [sid for sid, _ in candidates]
        return ((max(known) + 1) if known else 1), False


def reset_session_task_state(session_id: int) -> bool:
    """Clear inherited task context from a reused session row (2026-09-17
    session-leak fix): active plan, checkpoint, history, summary, in-flight
    instructions, and queued steps -- everything that made a fresh task look
    like a continuation of a previous conversation's work.

    Does NOT clear: identity/bookkeeping (workspace roots, allowed paths),
    the mutation ledger (modified_files/inspected_files -- the audit trail of
    real file changes stays with its session), or swarm metadata.
    """
    with state_lock():
        data = _load_raw()
        key = str(int(session_id))
        record = data.get(key)
        if not isinstance(record, dict):
            return False
        record.update({
            "active_task": None,
            "turn_checkpoint": None,
            "active_plan_id": None,
            "saved_plans": [],
            "conversation_history": [],
            "conversation_summary": None,
            "last_task_id": None,
            "current_phase": None,
            "running_action": None,
            "context_checkpoints": [],
            "completed_actions": [],
            "queued_user_instructions": [],
        })
        _save_raw(data)
    _VOLATILE_STATE.pop(_volatile_key(session_id), None)
    return True


def clear_session_state(session_id: int) -> bool:
    """Remove a session from the active local-session registry.

    The realtime ``state.json`` row and in-process copy are removed, which
    makes the session disappear from ``tamfis-code sessions`` and prevents
    workspace resolution from reusing it. Checkpoints, evidence, and the
    human-readable ``.memory`` snapshot are retained as a recovery archive.
    """
    with state_lock():
        data = _load_raw()
        key = str(int(session_id))
        existed = key in data or _volatile_key(session_id) in _VOLATILE_STATE
        if not existed:
            return False
        data.pop(key, None)
        _VOLATILE_STATE.pop(_volatile_key(session_id), None)
        _save_raw(data)
    return True


def mark_stale_session_superseded(session_id: int, *, replacement_session_id: int) -> bool:
    """Clear a dead task marker after the user starts a replacement session.

    A session whose timestamp is still inside the liveness window is owned by
    another process and is left untouched. Conversation and checkpoint data
    remain available for an explicit later resume.
    """
    state = get_session_state(session_id)
    if state.execution_status not in {"running", "backgrounded"}:
        return False
    if is_session_actively_running(state):
        return False
    state.execution_status = "superseded"
    state.current_phase = "idle"
    state.active_task = None
    state.running_action = None
    if state.turn_checkpoint:
        checkpoint = dict(state.turn_checkpoint)
        checkpoint["status"] = "superseded"
        checkpoint["superseded_by_session_id"] = int(replacement_session_id)
        checkpoint["updated_at"] = _now()
        state.turn_checkpoint = checkpoint
    put_session_state(state)
    return True


def fork_session_state(source_session_id: int, target_session_id: Optional[int] = None) -> SessionState:
    """Clone a durable local conversation into a fresh, idle session.

    Conversation, repository knowledge, plans, and mutation evidence are
    copied by value.  Process/task lifecycle state is deliberately cleared:
    a branch must not inherit an in-flight tool call, queued instruction,
    background task cursor, or swarm identity from its source.
    """
    known = all_known_session_ids()
    if source_session_id not in known:
        raise ValueError(f"No known local session {source_session_id}.")
    if target_session_id is None:
        target_session_id = (max(known) + 1) if known else 1
    if target_session_id in known:
        raise ValueError(f"Local session {target_session_id} already exists.")

    values = deepcopy(asdict(get_session_state(source_session_id)))
    values.update({
        "session_id": target_session_id,
        "last_event_id": 0,
        "last_task_id": None,
        "active_task": None,
        "current_phase": "idle",
        "execution_status": "idle",
        "pending_actions": [],
        "queued_user_instructions": [],
        "running_action": None,
        "turn_checkpoint": None,
        "parent_session_id": None,
        "is_swarm_child": False,
        "swarm_label": "",
        "swarm_worktree_path": None,
        "swarm_worktree_branch": None,
        "forked_from_session_id": source_session_id,
        "updated_at": "",
    })
    forked = SessionState(**values)
    put_session_state(forked)
    return forked


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


@dataclass
class RecapTurn:
    objective: str
    answer: str = ""


@dataclass
class ThreadRecap:
    """Structured form of a session recap -- the same data summarize_thread
    folds into one plain-text blob, kept as typed fields instead so
    render.render_thread_recap can lay each kind of fact out as its own
    section (a bordered multi-section card) rather than one undifferentiated
    paragraph. summarize_thread's plain-text form still exists unchanged
    for /compact's char-count bookkeeping; this is purely the display path.
    """
    empty_reason: str = ""
    older_count: int = 0
    older_turns: list[RecapTurn] = field(default_factory=list)
    modified_files: list[str] = field(default_factory=list)
    unresolved_count: int = 0
    active_plan_id: str = ""
    active_plan_objective: str = ""
    active_plan_steps: list[dict[str, Any]] = field(default_factory=list)
    recent_turns: list[RecapTurn] = field(default_factory=list)


def build_thread_recap(session_id: int, *, keep_recent: int = COMPACT_KEEP_RECENT_TURNS) -> ThreadRecap:
    """Structured counterpart to summarize_thread, for a polished card-style
    recap display (see render.render_thread_recap) instead of a single
    plain-text panel. Draws from the exact same durable SessionState fields
    summarize_thread does -- this is a presentation-layer split, not a
    different data source."""
    state = get_session_state(session_id)
    turns = _extract_turns(state.conversation_history)
    if not turns:
        prior_summary = (state.conversation_summary or "").strip()
        if prior_summary:
            return ThreadRecap(empty_reason=_bounded_text(prior_summary, head=MAX_COMPACT_SUMMARY_CHARS, tail=0, label="summary"))
        return ThreadRecap(empty_reason="No conversation recorded in this session yet.")

    recent = turns[-keep_recent:] if keep_recent > 0 else []
    older = turns[:-keep_recent] if keep_recent > 0 else turns

    recap = ThreadRecap(
        older_count=len(older),
        older_turns=[RecapTurn(objective=t["objective"], answer=t["answer"]) for t in older],
        recent_turns=[RecapTurn(objective=t["objective"], answer=t["answer"]) for t in recent],
    )
    if state.modified_files:
        recap.modified_files = [str(m.get("path") or "") for m in state.modified_files[-10:] if m.get("path")]
    recap.unresolved_count = len(state.unresolved_issues)
    if state.saved_plans:
        active = next((p for p in reversed(state.saved_plans) if p.get("id") == state.active_plan_id), None)
        if active:
            recap.active_plan_id = str(active.get("id") or "")
            recap.active_plan_objective = str(active.get("objective") or "")
            recap.active_plan_steps = list(active.get("steps") or [])
    return recap


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


def compact_session_thread(
    session_id: int, *, keep_recent: int = COMPACT_KEEP_RECENT_TURNS, preserve_note: str = "",
) -> str:
    """Compress the durable thread in place: fold older turns into
    conversation_summary and keep only the recent `keep_recent` turns in
    conversation_history. Returns the structured recap (same as
    summarize_thread) so the caller can display it immediately.

    This is the real `/compact` action: after it, the next turn's context is
    seeded from the bounded summary + recent turns, not the full raw
    transcript -- matching how Claude Code/Codex keep a long REPL usable
    without the terminal UI/UX growing heavy.

    `preserve_note`, when given (interactive.py's PreCompact hook output --
    Claude-Code-parity addition), is folded into the digest as its own
    leading line so critical information a hook flagged survives the fold,
    matching Claude Code's PreCompact "preserve context" contract.
    """
    state = get_session_state(session_id)
    turns = _extract_turns(state.conversation_history)
    if not turns or len(turns) <= keep_recent:
        # Nothing to fold -- still return the recap so /compact gives useful
        # feedback even on a short thread.
        return summarize_thread(session_id, keep_recent=keep_recent)

    older = turns[:-keep_recent] if keep_recent > 0 else turns
    digest_parts: list[str] = []
    if preserve_note:
        digest_parts.append(f"- [Preserved by pre_compact hook] {preserve_note}")
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
