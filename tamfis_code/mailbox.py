"""Coordinator/worker approval mailbox.

Parallel sub-agents must not be able to approve themselves. A swarm worker that
needs a destructive action (``rm -rf`` outside the workspace, ``git push
--force``, a schema migration) files a request into a mailbox; the coordinator
-- the root agent, which owns the user's policy and the only console -- claims
and answers it. Centralized safety, in parallel, without a global lock over
everything else the workers are doing.

Atomicity is the whole point of using SQLite here rather than a dict:

* **open** -- ``INSERT OR IGNORE`` on a UUID primary key, so a retried open can
  never create a duplicate request;
* **claim** -- ``BEGIN IMMEDIATE`` + a single ``UPDATE ... WHERE id = (SELECT
  ... WHERE status='pending')``; SQLite serializes the write transaction, so
  with two coordinators polling, exactly one observes ``rowcount == 1`` and the
  other sees no pending row. Two workers can never both claim one request;
* **resolve** -- a conditional ``UPDATE ... WHERE status IN ('claimed',
  'pending')``, so concurrent answers (a coordinator and a timeout, or two
  coordinators) resolve exactly once, and the winner is whoever's UPDATE
  changed a row.

The module is synchronous inside (sqlite3 is), with thin async wrappers so a
worker can await an answer without blocking the event loop for other workers.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import sqlite3
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

STATUS_PENDING = "pending"
STATUS_CLAIMED = "claimed"
STATUS_APPROVED = "approved"
STATUS_DENIED = "denied"
STATUS_EXPIRED = "expired"

DECISION_APPROVE = "approve_once"
DECISION_DENY = "deny"
DECISION_TIMEOUT = "timeout"

DEFAULT_COORDINATOR = "coordinator"
DEFAULT_WAIT_SECONDS = 120.0
POLL_INTERVAL_SECONDS = 0.1
DEFAULT_DB_FILENAME = "swarm_mailbox.sqlite3"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS approval_requests (
    id            TEXT PRIMARY KEY,
    session_id    INTEGER,
    worker        TEXT NOT NULL,
    tool          TEXT NOT NULL,
    command       TEXT NOT NULL DEFAULT '',
    arguments     TEXT NOT NULL DEFAULT '{}',
    risk          TEXT NOT NULL DEFAULT 'medium',
    status        TEXT NOT NULL DEFAULT 'pending',
    decision      TEXT,
    coordinator   TEXT,
    note          TEXT NOT NULL DEFAULT '',
    created_at    REAL NOT NULL,
    claimed_at    REAL,
    resolved_at   REAL
);
CREATE INDEX IF NOT EXISTS idx_approval_status ON approval_requests(status, created_at);
CREATE INDEX IF NOT EXISTS idx_approval_session ON approval_requests(session_id, created_at);
"""


def default_mailbox_path() -> Path:
    """One mailbox per user config dir -- shared by every process (coordinator
    REPL, background swarm, `tamfis-code mailbox`), which is what makes cross-
    process approval possible at all."""
    try:
        from . import state as local_state

        base = Path(getattr(local_state, "CONFIG_DIR"))
    except Exception:  # pragma: no cover - config dir is always importable in practice
        base = Path.home() / ".config" / "tamfis-code"
    return base / DEFAULT_DB_FILENAME


class Mailbox:
    """SQLite-backed approval mailbox. Safe for concurrent processes: every
    mutating operation runs inside an IMMEDIATE transaction (write lock) and
    reports whether *this* caller was the one that changed the row."""

    def __init__(self, path: Path | str | None = None, *, busy_timeout: float = 5.0) -> None:
        self.path = Path(path) if path is not None else default_mailbox_path()
        self.busy_timeout = busy_timeout
        self._initialized = False

    # -- plumbing --------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.path), timeout=self.busy_timeout, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(f"PRAGMA busy_timeout={int(self.busy_timeout * 1000)}")
        except sqlite3.Error:
            pass
        if not self._initialized:
            connection.executescript(_SCHEMA)
            self._initialized = True
        return connection

    @contextlib.contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        """One write transaction. ``BEGIN IMMEDIATE`` takes the write lock up
        front, so two processes cannot interleave a read-then-write claim."""
        connection = self._connect()
        connection.execute("BEGIN IMMEDIATE")
        try:
            yield connection
        except Exception:
            connection.execute("ROLLBACK")
            raise
        else:
            connection.execute("COMMIT")
        finally:
            connection.close()

    @contextlib.contextmanager
    def _read(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        try:
            data["arguments"] = json.loads(data.get("arguments") or "{}")
        except (json.JSONDecodeError, TypeError):
            data["arguments"] = {}
        return data

    # -- worker side -----------------------------------------------------
    def open_request(
        self,
        *,
        worker: str,
        tool: str,
        arguments: Any = None,
        risk: str = "medium",
        session_id: Optional[int] = None,
        command: str = "",
    ) -> str:
        """File a request and return its id. ``INSERT OR IGNORE`` on a fresh
        UUID: idempotent by construction, never a duplicate row."""
        request_id = f"mreq_{uuid.uuid4().hex[:16]}"
        payload = json.dumps(arguments if isinstance(arguments, dict) else {"value": arguments}, default=str)
        with self._write() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO approval_requests "
                "(id, session_id, worker, tool, command, arguments, risk, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (request_id, session_id, worker, tool, command[:4000], payload, risk, STATUS_PENDING, time.time()),
            )
        return request_id

    def get(self, request_id: str) -> Optional[dict[str, Any]]:
        with self._read() as connection:
            row = connection.execute(
                "SELECT * FROM approval_requests WHERE id = ?", (request_id,),
            ).fetchone()
        return self._row_to_dict(row) if row is not None else None

    def wait_for_decision(self, request_id: str, *, timeout: float = DEFAULT_WAIT_SECONDS) -> str:
        """Block (polling) until the request is resolved, or time out.

        A timeout is reported as its own value rather than as a denial: the
        caller decides, and every caller here fails closed.
        """
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            record = self.get(request_id)
            if record is None:
                return DECISION_DENY
            status = record.get("status")
            if status == STATUS_APPROVED:
                return DECISION_APPROVE
            if status == STATUS_DENIED:
                return DECISION_DENY
            if status == STATUS_EXPIRED:
                return DECISION_TIMEOUT
            if time.monotonic() >= deadline:
                return DECISION_TIMEOUT
            time.sleep(POLL_INTERVAL_SECONDS)

    async def await_decision(self, request_id: str, *, timeout: float = DEFAULT_WAIT_SECONDS) -> str:
        """Async wrapper for the worker loop -- the sqlite polling happens off
        the event loop so other workers keep running."""
        return await asyncio.to_thread(self.wait_for_decision, request_id, timeout=timeout)

    # -- coordinator side ------------------------------------------------
    def claim_next(self, coordinator: str = DEFAULT_COORDINATOR) -> Optional[dict[str, Any]]:
        """Atomically claim the oldest pending request, or None.

        The subquery + conditional UPDATE run inside one IMMEDIATE
        transaction: exactly one concurrent caller can win a given row.
        """
        now = time.time()
        with self._write() as connection:
            cursor = connection.execute(
                "UPDATE approval_requests "
                "SET status = ?, coordinator = ?, claimed_at = ? "
                "WHERE id = (SELECT id FROM approval_requests WHERE status = ? "
                "            ORDER BY created_at LIMIT 1) "
                "  AND status = ?",
                (STATUS_CLAIMED, coordinator, now, STATUS_PENDING, STATUS_PENDING),
            )
            if cursor.rowcount != 1:
                return None
            row = connection.execute(
                "SELECT * FROM approval_requests WHERE coordinator = ? AND status = ? "
                "ORDER BY claimed_at DESC LIMIT 1",
                (coordinator, STATUS_CLAIMED),
            ).fetchone()
        return self._row_to_dict(row) if row is not None else None

    def resolve(
        self,
        request_id: str,
        decision: str,
        *,
        coordinator: Optional[str] = None,
        note: str = "",
    ) -> bool:
        """Answer a request. Returns True only if THIS call resolved it.

        Refuses self-approval (a worker answering its own request) and refuses
        to re-resolve an already-resolved request -- both are the properties
        that make the mailbox a real gate instead of a suggestion.
        """
        normalized = DECISION_APPROVE if decision in (DECISION_APPROVE, "approve", "allow", "yes") else DECISION_DENY
        status = STATUS_APPROVED if normalized == DECISION_APPROVE else STATUS_DENIED
        with self._write() as connection:
            row = connection.execute(
                "SELECT worker, coordinator, status FROM approval_requests WHERE id = ?", (request_id,),
            ).fetchone()
            if row is None:
                return False
            if coordinator is not None and str(row["worker"]) == str(coordinator):
                return False  # a worker may never approve itself
            if str(row["status"]) in (STATUS_APPROVED, STATUS_DENIED, STATUS_EXPIRED):
                return False  # already resolved -- first answer wins
            cursor = connection.execute(
                "UPDATE approval_requests SET status = ?, decision = ?, coordinator = ?, note = ?, resolved_at = ? "
                "WHERE id = ? AND status IN (?, ?)",
                (status, normalized, coordinator, note[:2000], time.time(), request_id, STATUS_PENDING, STATUS_CLAIMED),
            )
            return cursor.rowcount == 1

    def expire_stale(self, *, max_age_seconds: float = 900.0) -> int:
        """Mark long-abandoned requests expired so the coordinator never has to
        reason about a dead worker's request (and a worker that went away can't
        leave a forever-pending row)."""
        cutoff = time.time() - max(0.0, max_age_seconds)
        with self._write() as connection:
            cursor = connection.execute(
                "UPDATE approval_requests SET status = ?, decision = ?, resolved_at = ? "
                "WHERE status IN (?, ?) AND created_at < ?",
                (STATUS_EXPIRED, DECISION_TIMEOUT, time.time(), STATUS_PENDING, STATUS_CLAIMED, cutoff),
            )
            return cursor.rowcount

    # -- reporting -------------------------------------------------------
    def pending(self, *, session_id: Optional[int] = None, limit: int = 50) -> list[dict[str, Any]]:
        query = "SELECT * FROM approval_requests WHERE status IN (?, ?)"
        params: list[Any] = [STATUS_PENDING, STATUS_CLAIMED]
        if session_id is not None:
            query += " AND session_id = ?"
            params.append(session_id)
        query += " ORDER BY created_at LIMIT ?"
        params.append(limit)
        with self._read() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def history(self, *, session_id: Optional[int] = None, limit: int = 50) -> list[dict[str, Any]]:
        query = "SELECT * FROM approval_requests"
        params: list[Any] = []
        if session_id is not None:
            query += " WHERE session_id = ?"
            params.append(session_id)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._read() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def stats(self) -> dict[str, int]:
        with self._read() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS total FROM approval_requests GROUP BY status",
            ).fetchall()
        counts = {str(row["status"]): int(row["total"]) for row in rows}
        counts["total"] = sum(counts.values())
        return counts


# --------------------------------------------------------------------------
# Worker context: what makes a sub-agent know it must ask the coordinator
# --------------------------------------------------------------------------

_CURRENT_WORKER: ContextVar[Optional["WorkerContext"]] = ContextVar("tamfis_current_worker", default=None)


@dataclass
class WorkerContext:
    """Set for the duration of one swarm sub-task. Any mutating tool call the
    sub-task attempts is routed to the coordinator's mailbox instead of being
    answered locally (a sub-task runs non-interactively, so "locally" would
    just mean the default deny)."""

    worker_id: str
    mailbox: Mailbox
    session_id: Optional[int] = None
    timeout_seconds: float = DEFAULT_WAIT_SECONDS
    request_ids: list[str] = field(default_factory=list)

    def file_request(self, *, tool: str, arguments: Any, risk: str, command: str = "") -> str:
        request_id = self.mailbox.open_request(
            worker=self.worker_id, tool=tool, arguments=arguments, risk=risk,
            session_id=self.session_id, command=command,
        )
        self.request_ids.append(request_id)
        return request_id

    async def request_approval(
        self, *, tool: str, arguments: Any, risk: str = "medium", command: str = "",
        timeout: Optional[float] = None,
    ) -> str:
        """File + await. Returns DECISION_APPROVE / DECISION_DENY / DECISION_TIMEOUT."""
        request_id = self.file_request(tool=tool, arguments=arguments, risk=risk, command=command)
        return await self.mailbox.await_decision(
            request_id, timeout=self.timeout_seconds if timeout is None else timeout,
        )


@contextlib.contextmanager
def worker_context(context: WorkerContext) -> Iterator[WorkerContext]:
    """Bind a worker context to the current asyncio task/thread. Sub-tasks run
    concurrently in one process, and each agent executes inside its own task,
    so the binding is per-task and can never leak between workers."""
    token = _CURRENT_WORKER.set(context)
    try:
        yield context
    finally:
        _CURRENT_WORKER.reset(token)


def current_worker() -> Optional[WorkerContext]:
    return _CURRENT_WORKER.get()


def mailbox_for_swarm(session_id: Optional[int] = None) -> Mailbox:
    """The mailbox a swarm's coordinator and workers share."""
    return Mailbox()
