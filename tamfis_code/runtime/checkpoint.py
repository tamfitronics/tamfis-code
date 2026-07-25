"""Durable resumable execution checkpoints."""
from __future__ import annotations

import json
import os
import stat
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tamfis_code.config import CONFIG_DIR
from tamfis_code.state import redact_secrets

CHECKPOINT_DIR = CONFIG_DIR / "checkpoints"


@dataclass(slots=True)
class ExecutionCheckpoint:
    execution_id: str
    session_id: int
    mode: str
    objective: str
    workspace_root: str
    status: str
    phase: str = "understand"
    plan: dict[str, Any] | None = None
    changed_files: list[str] = field(default_factory=list)
    validations: list[dict[str, Any]] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def checkpoint_path(execution_id: str) -> Path:
    return CHECKPOINT_DIR / f"{execution_id}.json"


def save_checkpoint(checkpoint: ExecutionCheckpoint) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.loads(redact_secrets(json.dumps(asdict(checkpoint), default=str)))
    target = checkpoint_path(checkpoint.execution_id)
    fd, temp_name = tempfile.mkstemp(prefix=f".{checkpoint.execution_id}-", suffix=".json", dir=CHECKPOINT_DIR)
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


def load_checkpoint(execution_id: str) -> ExecutionCheckpoint | None:
    path = checkpoint_path(execution_id)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return ExecutionCheckpoint(**data)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def latest_resumable_checkpoint(session_id: int | None = None) -> ExecutionCheckpoint | None:
    if not CHECKPOINT_DIR.is_dir():
        return None
    candidates = sorted(CHECKPOINT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        checkpoint = load_checkpoint(path.stem)
        if checkpoint is None or checkpoint.status not in {"running", "partial", "blocked", "cancelled", "failed"}:
            continue
        if session_id is None or checkpoint.session_id == session_id:
            return checkpoint
    return None
