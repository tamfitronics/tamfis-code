"""Durable, off-context evidence storage for internal context rollover.

When a working segment's provider context can no longer hold the full
tool-calling history for a task even after runner_local.py's in-place
compaction (_trim_tool_outputs), the segment is persisted here -- OUTSIDE
the provider prompt -- before the working context is reset to a compact
continuation package. This is what makes a rollover a checkpoint rather
than data loss: the model can call the retrieve_evidence tool to pull
exact prior tool output/file content back on demand, and a human can
inspect `tamfis-code diagnostics`-style tooling against the same file
later if needed.

One append-only JSONL file per session (mirrors state.py's existing
per-session storage granularity), stored under CONFIG_DIR rather than
inside SessionState.context_checkpoints -- full message histories can be
large and are not something every state.py read/write should have to
carry, and JSONL append is naturally crash-safe (a killed process leaves
at most one incomplete trailing line, never corrupts prior segments).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .config import CONFIG_DIR

EVIDENCE_DIR = CONFIG_DIR / "evidence"


def _evidence_path(session_id: int) -> Path:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    return EVIDENCE_DIR / f"session_{session_id}.jsonl"


def store_segment(
    session_id: int, *, objective: str, messages: list[dict[str, Any]], summary: str,
) -> str:
    """Append a full working-message segment to durable storage.

    Returns an evidence_id the continuation package / retrieve_evidence
    tool can reference to pull it back later.
    """
    evidence_id = f"evidence_{uuid.uuid4().hex[:12]}"
    record = {
        "evidence_id": evidence_id,
        "session_id": session_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": objective,
        "summary": summary,
        "message_count": len(messages),
        "messages": messages,
    }
    path = _evidence_path(session_id)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return evidence_id


def load_segment(session_id: int, evidence_id: str) -> Optional[dict[str, Any]]:
    """Return the full persisted segment for `evidence_id`, or None if this
    session has no evidence file or no segment with that id."""
    path = _evidence_path(session_id)
    if not path.is_file() or not evidence_id:
        return None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("evidence_id") == evidence_id:
                return record
    return None


def list_segments(session_id: int) -> list[dict[str, Any]]:
    """Lightweight index of every segment recorded for this session (no
    message bodies) -- for `tamfis-code` diagnostics/status surfaces."""
    path = _evidence_path(session_id)
    if not path.is_file():
        return []
    segments = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            segments.append({
                "evidence_id": record.get("evidence_id"),
                "created_at": record.get("created_at"),
                "objective": record.get("objective"),
                "summary": record.get("summary"),
                "message_count": record.get("message_count"),
            })
    return segments
