"""Durable structured task ledger for long-horizon coding tasks.

This is the canonical, compression-safe task state -- deliberately separate
from conversational prose (SessionState.conversation_history) and from the
per-turn protocol checkpoint (SessionState.turn_checkpoint).  The ledger is
the anchor that recursive compaction rebuilds from: every compaction
generation reconstructs the active context from this structured record plus
new events, never by re-summarising a previous prose summary.

Layer model (see TAMFIS_CONTEXT_COMPRESSION.md):
  Layer A -- permanent task facts: objective, constraints, decisions, plan,
            modified files, unresolved blockers, next action.  Never
            discarded until the task completes.
  Layer B -- active working context: recent tool results, current file
            excerpts, recent errors.  Refreshed frequently.
  Layer C -- cold history: verbose tool output, superseded analysis,
            duplicate searches.  Compressed/archived, never kept in the
            active model context.

The ledger is persisted atomically under CONFIG_DIR/ledgers/<task_id>.json
and is redacted for secrets at save time (same redact_secrets pass as the
runtime checkpoint store).
"""
from __future__ import annotations

import json
import os
import re
import stat
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..config import CONFIG_DIR

LEDGER_DIR = CONFIG_DIR / "ledgers"

# Statuses a continuation may legitimately resume from.  A ledger in any of
# these states is "unfinished work" -- a `continue`/`resume` message must
# attach to it rather than minting a new task.
RESUMABLE_STATUSES = frozenset({
    "running", "partial", "blocked", "recovering", "checkpointing",
})

# Fact provenance: the compactor must never promote an ASSUMPTION into a
# FACT, so every discovery record carries one of these explicitly.
FACT_STATUSES = frozenset({"fact", "decision", "assumption", "pending", "failed", "unknown"})


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass(slots=True)
class PlanStep:
    """One plan step with its own durable status."""
    index: int
    name: str
    status: str = "pending"  # pending | in_progress | completed | blocked
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PlanStep":
        return cls(
            index=int(payload.get("index") or 0),
            name=str(payload.get("name") or ""),
            status=str(payload.get("status") or "pending"),
            evidence=[str(e) for e in (payload.get("evidence") or [])],
        )


@dataclass(slots=True)
class LedgerEdit:
    """A semantic record of one applied file mutation (diff-aware)."""
    file: str
    description: str
    symbols_changed: list[str] = field(default_factory=list)
    diff_summary: str = ""
    applied: bool = False
    validated: bool = False
    mutation_id: str = ""
    committed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LedgerEdit":
        return cls(
            file=str(payload.get("file") or ""),
            description=str(payload.get("description") or ""),
            symbols_changed=[str(s) for s in (payload.get("symbols_changed") or [])],
            diff_summary=str(payload.get("diff_summary") or ""),
            applied=bool(payload.get("applied")),
            validated=bool(payload.get("validated")),
            mutation_id=str(payload.get("mutation_id") or ""),
            committed=bool(payload.get("committed")),
        )


@dataclass(slots=True)
class LedgerTest:
    """One validation command and its outcome (tool-output-aware)."""
    command: str
    status: str  # passed | failed | not_run
    summary: str = ""
    important_output: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LedgerTest":
        return cls(
            command=str(payload.get("command") or ""),
            status=str(payload.get("status") or "not_run"),
            summary=str(payload.get("summary") or ""),
            important_output=str(payload.get("important_output") or ""),
        )


@dataclass(slots=True)
class LedgerFailure:
    """A failed approach that must not be blindly retried."""
    operation: str
    cause: str
    resolution: str = ""
    should_not_retry_same_way: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LedgerFailure":
        return cls(
            operation=str(payload.get("operation") or ""),
            cause=str(payload.get("cause") or ""),
            resolution=str(payload.get("resolution") or ""),
            should_not_retry_same_way=bool(payload.get("should_not_retry_same_way", True)),
        )


@dataclass(slots=True)
class LedgerDiscovery:
    """A durable search/inspection finding (search memory).

    `status` is one of FACT_STATUSES so a later compaction can never
    silently promote an assumption into a verified fact.
    """
    query: str
    finding: str
    authoritative_file: str = ""
    status: str = "fact"
    irrelevant_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LedgerDiscovery":
        return cls(
            query=str(payload.get("query") or ""),
            finding=str(payload.get("finding") or ""),
            authoritative_file=str(payload.get("authoritative_file") or ""),
            status=str(payload.get("status") or "fact"),
            irrelevant_paths=[str(p) for p in (payload.get("irrelevant_paths") or [])],
        )


@dataclass(slots=True)
class TaskLedger:
    """Canonical structured task state -- the compaction anchor."""
    # Core identity (Layer A -- permanent)
    task_id: str
    session_id: int
    objective: str
    status: str = "running"
    repo_roots: list[str] = field(default_factory=list)
    cwd: str = ""
    user_constraints: list[str] = field(default_factory=list)
    user_decisions: list[str] = field(default_factory=list)

    # Plan (Layer A)
    plan_steps: list[PlanStep] = field(default_factory=list)
    current_step_index: int = 0

    # Working set (Layer A/B boundary)
    relevant_files: list[str] = field(default_factory=list)
    currently_open_files: list[str] = field(default_factory=list)
    relevant_symbols: list[str] = field(default_factory=list)
    relevant_configs: list[str] = field(default_factory=list)
    irrelevant_paths_to_skip: list[str] = field(default_factory=list)

    # Discoveries (Layer A)
    discoveries: list[LedgerDiscovery] = field(default_factory=list)

    # Edits (Layer A)
    edits: list[LedgerEdit] = field(default_factory=list)

    # Commands/tests (Layer A)
    important_commands: list[str] = field(default_factory=list)
    tests: list[LedgerTest] = field(default_factory=list)

    # Failure memory (Layer A)
    failures: list[LedgerFailure] = field(default_factory=list)

    # Provider state (Layer A)
    attempted_providers: list[str] = field(default_factory=list)
    current_provider: str = ""
    current_model: str = ""

    # Queued user messages (Layer A)
    queued_user_messages: list[str] = field(default_factory=list)

    # Continuation (Layer A)
    last_successful_action: str = ""
    current_action: str = ""
    next_action: str = ""

    # Compaction bookkeeping
    compact_generation: int = 0
    checkpoint_version: int = 1
    events_compacted_through: int = 0
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TaskLedger":
        ledger = cls(
            task_id=str(payload.get("task_id") or ""),
            session_id=int(payload.get("session_id") or 0),
            objective=str(payload.get("objective") or ""),
            status=str(payload.get("status") or "running"),
            repo_roots=[str(r) for r in (payload.get("repo_roots") or [])],
            cwd=str(payload.get("cwd") or ""),
            user_constraints=[str(c) for c in (payload.get("user_constraints") or [])],
            user_decisions=[str(d) for d in (payload.get("user_decisions") or [])],
            current_step_index=int(payload.get("current_step_index") or 0),
            relevant_files=[str(f) for f in (payload.get("relevant_files") or [])],
            currently_open_files=[str(f) for f in (payload.get("currently_open_files") or [])],
            relevant_symbols=[str(s) for s in (payload.get("relevant_symbols") or [])],
            relevant_configs=[str(c) for c in (payload.get("relevant_configs") or [])],
            irrelevant_paths_to_skip=[str(p) for p in (payload.get("irrelevant_paths_to_skip") or [])],
            attempted_providers=[str(p) for p in (payload.get("attempted_providers") or [])],
            current_provider=str(payload.get("current_provider") or ""),
            current_model=str(payload.get("current_model") or ""),
            queued_user_messages=[str(m) for m in (payload.get("queued_user_messages") or [])],
            last_successful_action=str(payload.get("last_successful_action") or ""),
            current_action=str(payload.get("current_action") or ""),
            next_action=str(payload.get("next_action") or ""),
            compact_generation=int(payload.get("compact_generation") or 0),
            checkpoint_version=int(payload.get("checkpoint_version") or 1),
            events_compacted_through=int(payload.get("events_compacted_through") or 0),
            created_at=str(payload.get("created_at") or _now()),
            updated_at=str(payload.get("updated_at") or _now()),
        )
        ledger.plan_steps = [PlanStep.from_dict(p) for p in (payload.get("plan_steps") or [])]
        ledger.discoveries = [LedgerDiscovery.from_dict(p) for p in (payload.get("discoveries") or [])]
        ledger.edits = [LedgerEdit.from_dict(p) for p in (payload.get("edits") or [])]
        ledger.tests = [LedgerTest.from_dict(p) for p in (payload.get("tests") or [])]
        ledger.failures = [LedgerFailure.from_dict(p) for p in (payload.get("failures") or [])]
        return ledger

    # ------------------------------------------------------------------
    # Integrity
    # ------------------------------------------------------------------

    def integrity_errors(self) -> list[str]:
        """Return the list of missing critical fields (Layer A facts).

        Used after every compaction: if this returns non-empty, the
        compaction result is rejected and the previous valid compact state
        is kept.
        """
        errors: list[str] = []
        if not self.task_id:
            errors.append("task_id")
        if not self.objective:
            errors.append("objective")
        if not self.repo_roots:
            errors.append("repo_roots")
        if self.current_step_index < 0 or self.current_step_index >= max(1, len(self.plan_steps)):
            errors.append("current_step_index")
        if not self.next_action and self.status in RESUMABLE_STATUSES:
            errors.append("next_action")
        return errors


def ledger_path(task_id: str) -> Path:
    return LEDGER_DIR / f"{task_id}.json"


def save_ledger(ledger: TaskLedger) -> Path:
    """Atomically persist the ledger, redacting secrets first."""
    from ..state import redact_secrets

    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    ledger.updated_at = _now()
    payload = json.loads(redact_secrets(json.dumps(ledger.to_dict(), default=str)))
    target = ledger_path(ledger.task_id)
    fd, temp_name = tempfile.mkstemp(prefix=f".{ledger.task_id}-", suffix=".json", dir=LEDGER_DIR)
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
    return target


def load_ledger(task_id: str) -> TaskLedger | None:
    path = ledger_path(task_id)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return TaskLedger.from_dict(data)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def latest_resumable_ledger(session_id: int | None = None) -> TaskLedger | None:
    """Most recent unfinished ledger, optionally scoped to a session."""
    if not LEDGER_DIR.is_dir():
        return None
    candidates = sorted(LEDGER_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        ledger = load_ledger(path.stem)
        if ledger is None or ledger.status not in RESUMABLE_STATUSES:
            continue
        if session_id is None or ledger.session_id == session_id:
            return ledger
    return None