"""Durable, redacted execution journal for the unified runtime.

Every public execution adapter writes lifecycle events here.  The journal is
append-only JSONL so a killed process cannot corrupt prior records and doctor
can report concrete recent failures instead of only aggregate percentages.
"""
from __future__ import annotations

import json
import os
import stat
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from tamfis_code.config import CONFIG_DIR
from tamfis_code.state import redact_secrets

JOURNAL_PATH = CONFIG_DIR / "runtime-events.jsonl"
MAX_JOURNAL_BYTES = 5 * 1024 * 1024
KEEP_ROTATED_BYTES = 2 * 1024 * 1024


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize(value: Any) -> Any:
    if isinstance(value, str):
        return redact_secrets(value)
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    return value


@dataclass(slots=True)
class RuntimeEvent:
    event: str
    execution_id: str
    mode: str
    session_id: int
    timestamp: str
    status: str = ""
    objective: str = ""
    workspace_root: str = ""
    summary: str = ""
    error: str = ""
    duration_ms: int = 0
    metadata: dict[str, Any] | None = None


def _rotate_if_needed() -> None:
    if not JOURNAL_PATH.exists() or JOURNAL_PATH.stat().st_size <= MAX_JOURNAL_BYTES:
        return
    data = JOURNAL_PATH.read_bytes()[-KEEP_ROTATED_BYTES:]
    first_newline = data.find(b"\n")
    if first_newline >= 0:
        data = data[first_newline + 1 :]
    fd, temp_name = tempfile.mkstemp(prefix=".runtime-events-", suffix=".jsonl", dir=CONFIG_DIR)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_name, stat.S_IRUSR | stat.S_IWUSR)
        os.replace(temp_name, JOURNAL_PATH)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def append_event(event: RuntimeEvent) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _rotate_if_needed()
    payload = _sanitize(asdict(event))
    line = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str) + "\n"
    fd = os.open(JOURNAL_PATH, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, line.encode("utf-8", "replace"))
        os.fsync(fd)
    finally:
        os.close(fd)


def read_recent_events(limit: int = 100) -> list[dict[str, Any]]:
    if limit <= 0 or not JOURNAL_PATH.is_file():
        return []
    try:
        lines = JOURNAL_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    events: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def recent_failures(limit: int = 5) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for event in reversed(read_recent_events(max(limit * 20, 100))):
        if event.get("event") != "execution_finished":
            continue
        if event.get("status") not in {"failed", "cancelled", "partial", "blocked"}:
            continue
        failures.append(event)
        if len(failures) >= limit:
            break
    return failures
