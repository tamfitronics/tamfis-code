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

# Evidence is deliberately retrieved in bounded windows.  Returning a whole
# multi-megabyte objective from a tool call would simply put the context
# overflow back into the next provider request.
DEFAULT_OBJECTIVE_CHUNK_CHARS = 12_000
MAX_OBJECTIVE_CHUNK_CHARS = 50_000


def _evidence_path(session_id: int) -> Path:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.chmod(0o700)
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
    path.chmod(0o600)
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


def objective_chunk(
    segment: dict[str, Any], *, offset: int = 0,
    max_chars: int = DEFAULT_OBJECTIVE_CHUNK_CHARS, query: str = "",
) -> dict[str, Any]:
    """Return a bounded, pageable window into an archived objective.

    ``query`` is an optional convenience for large logs/documents: the first
    case-insensitive match becomes the centre of the returned window, avoiding
    thousands of sequential paging calls when the relevant wording is known.
    """
    objective = str(segment.get("objective") or "")
    try:
        requested_size = int(max_chars)
    except (TypeError, ValueError):
        requested_size = DEFAULT_OBJECTIVE_CHUNK_CHARS
    try:
        requested_offset = int(offset)
    except (TypeError, ValueError):
        requested_offset = 0
    size = min(max(1, requested_size), MAX_OBJECTIVE_CHUNK_CHARS)
    start = min(max(0, requested_offset), len(objective))
    matched_at: Optional[int] = None
    if query:
        matched_at = objective.casefold().find(query.casefold())
        if matched_at >= 0:
            start = max(0, matched_at - size // 3)
        else:
            matched_at = None
    end = min(len(objective), start + size)
    return {
        "objective": objective[start:end],
        "objective_offset": start,
        "objective_end": end,
        "objective_total_chars": len(objective),
        "has_more": end < len(objective),
        "next_offset": end if end < len(objective) else None,
        "query_matched_at": matched_at,
    }


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
                "objective": (
                    str(record.get("objective") or "")[:400]
                    + (
                        "...[preview]"
                        if len(str(record.get("objective") or "")) > 400
                        else ""
                    )
                ),
                "objective_chars": len(str(record.get("objective") or "")),
                "summary": record.get("summary"),
                "message_count": record.get("message_count"),
            })
    return segments
