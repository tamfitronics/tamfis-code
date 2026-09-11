"""Durable resumable execution checkpoints."""

from __future__ import annotations

import json
import os
import stat
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tamfis_code.config import CONFIG_DIR
from tamfis_code.state import redact_secrets

CHECKPOINT_DIR = CONFIG_DIR / "checkpoints"


@dataclass(slots=True)
class ExecutionCheckpoint:
    """Comprehensive checkpoint state for durable task resumption."""
    # Core identity
    execution_id: str
    session_id: int
    mode: str
    objective: str
    workspace_root: str
    status: str
    phase: str = "understand"
    
    # Plan state
    plan: dict[str, Any] | None = None
    plan_steps: list[dict[str, Any]] = field(default_factory=list)
    current_step_index: int = 0
    completed_steps: list[int] = field(default_factory=list)
    remaining_steps: list[int] = field(default_factory=list)
    
    # Discovery state
    files_already_examined: list[str] = field(default_factory=list)
    directories_already_examined: list[str] = field(default_factory=list)
    symbols_already_traced: list[str] = field(default_factory=list)
    relevant_files: list[str] = field(default_factory=list)
    irrelevant_paths_to_skip: list[str] = field(default_factory=list)
    
    # Edit state
    changed_files: list[str] = field(default_factory=list)
    files_modified: list[dict[str, Any]] = field(default_factory=list)
    mutation_ids: list[str] = field(default_factory=list)
    patches_applied: list[str] = field(default_factory=list)
    patches_pending: list[str] = field(default_factory=list)
    validation_pending: list[str] = field(default_factory=list)
    # Validation results recorded by the unified runtime (backward compat).
    validations: list[dict[str, Any]] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    
    # Execution state
    last_successful_action: str = ""
    last_tool_call: dict[str, Any] | None = None
    current_operation: str = ""
    next_intended_operation: str = ""
    attempted_providers: list[str] = field(default_factory=list)
    current_provider: str = ""
    current_model: str = ""
    recovery_count: int = 0
    
    # Repository state
    repo_root: str = ""
    git_branch: str = ""
    starting_commit: str = ""
    git_status: str = ""
    modified_file_list: list[str] = field(default_factory=list)
    
    # Validation state
    tests_run: list[dict[str, Any]] = field(default_factory=list)
    tests_passed: list[str] = field(default_factory=list)
    tests_failed: list[str] = field(default_factory=list)
    tests_not_yet_run: list[str] = field(default_factory=list)
    
    # Compact context for quick resume
    compact_context: str = ""

    # Timestamps
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    updated_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    checkpoint_version: int = 1
    checkpoint_sequence: int = 0

    # Next action for seamless continuation
    next_action: str = ""
    last_completed_action: str = ""

    # Free-form metadata (task contract, evidence graph, review results) --
    # kept for backward compatibility with unified.py's checkpoint writes.
    metadata: dict[str, Any] = field(default_factory=dict)


def checkpoint_path(execution_id: str) -> Path:
    return CHECKPOINT_DIR / f"{execution_id}.json"


def save_checkpoint(checkpoint: ExecutionCheckpoint) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
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
        if checkpoint is None or checkpoint.status not in {"running", "partial", "blocked", "cancelled", "failed", "checkpointing", "recovering"}:
            continue
        if session_id is None or checkpoint.session_id == session_id:
            return checkpoint
    return None


def load_latest_checkpoint_for_session(session_id: int) -> ExecutionCheckpoint | None:
    """Load the most recent checkpoint for a specific session."""
    return latest_resumable_checkpoint(session_id)