"""Authoritative execution runtime for every Tamfis-Code invocation.

Phase 2 consolidates interactive, non-interactive, local, remote, lightweight
chat, delegated-agent and OpenHands entry points behind this module. Existing
engines remain internal implementation details while callers use one stable
controller contract.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

from .controller import ExecutionController
from .journal import RuntimeEvent, append_event
from .checkpoint import ExecutionCheckpoint, save_checkpoint
from .cognitive import EvidenceGraph, EvidenceNode, TaskContract
from .memory import relevant_memories
from .reviewer import IndependentReviewer
from .worktree import create_worktree
from .. import state as local_state


class ExecutionMode(str, Enum):
    LOCAL_AGENT = "local_agent"
    REMOTE_AGENT = "remote_agent"
    LOCAL_CHAT = "local_chat"
    LOCAL_STREAM = "local_stream"


@dataclass(slots=True)
class ExecutionRequest:
    mode: ExecutionMode
    session_id: int = 0
    objective: str = ""
    workspace_root: str = "."
    interactive: bool = True
    approval_policy: str = "ask"
    read_only: bool = False
    isolation: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionRecord:
    mode: ExecutionMode
    status: str
    session_id: int
    summary: str = ""
    error: str | None = None
    execution_id: str = ""
    duration_ms: int = 0


class UnifiedAgentRuntime:
    """Single authority for dispatch, lifecycle tracking and cancellation.

    Thin interface adapters call this class. The large historical engines are
    deliberately invoked only through private implementation functions so new
    policy, telemetry and cancellation behaviour has one insertion point.
    """

    def __init__(self) -> None:
        self.controller = ExecutionController()
        self._active_task: asyncio.Task[Any] | None = None
        # Reservation task used during synchronous lifecycle setup.  Without
        # this, a second invocation (or cancellation) could observe an idle
        # runtime while journaling/checkpoint creation was still in progress.
        self._runner_task: asyncio.Task[Any] | None = None
        self._active_request: ExecutionRequest | None = None
        self.history: list[ExecutionRecord] = []
        self.persistence_warnings: list[str] = []
        self._lock = asyncio.Lock()
        self.task_contract: TaskContract | None = None
        self.evidence_graph = EvidenceGraph()
        self.last_review = None

    @property
    def active(self) -> bool:
        return any(
            task is not None and not task.done()
            for task in (self._runner_task, self._active_task)
        )

    def _append_event(self, event: RuntimeEvent) -> None:
        """Persist telemetry without making telemetry a task dependency."""
        try:
            append_event(event)
        except Exception as exc:
            self.persistence_warnings.append(f"journal unavailable: {exc}")

    def _save_checkpoint(self, checkpoint: ExecutionCheckpoint) -> None:
        """Persist recovery state without turning a read-only config home
        into an execution failure. The runner's live state remains the source
        of truth while this process is running; the warning is retained for
        diagnostics and the next invocation can repair storage."""
        try:
            save_checkpoint(checkpoint)
        except Exception as exc:
            self.persistence_warnings.append(f"checkpoint unavailable: {exc}")

    async def _run_exclusive(self, request: ExecutionRequest, operation: Callable[[], Awaitable[Any]]) -> Any:
        # Fail fast instead of queueing a second agent behind the first. A queued
        # call can capture stale terminal/session state and was one source of the
        # overlapping-runtime behaviour Phase 2 is designed to eliminate.
        if self.active or self._lock.locked():
            raise RuntimeError("an agent execution is already active in this runtime")
        async with self._lock:
            if self.active:
                raise RuntimeError("an agent execution is already active in this runtime")
            self._runner_task = asyncio.current_task()
            self.persistence_warnings = []
            self.controller = ExecutionController()
            self._active_request = request
            memories = relevant_memories(request.objective)
            if memories:
                request.metadata["relevant_memory"] = [
                    {"name": m.name, "type": m.type.value, "description": m.description}
                    for m in memories
                ]
            self.task_contract = TaskContract.derive(
                request.objective,
                read_only=request.read_only,
                approval_policy=request.approval_policy,
                metadata=request.metadata,
            )
            self.evidence_graph = EvidenceGraph()
            self.last_review = None
            baseline_mutation_ids: set[str] = set()
            baseline_validation_count = 0
            if request.session_id:
                try:
                    baseline_state = local_state.get_session_state(request.session_id)
                    baseline_mutation_ids = {
                        str(item.get("mutation_id"))
                        for item in baseline_state.modified_files
                        if item.get("mutation_id")
                    }
                    baseline_validation_count = len(baseline_state.validation_results)
                except OSError:
                    # Evidence enrichment is best-effort; the actual runner
                    # still owns execution and reports its own result.
                    pass
            execution_id = uuid.uuid4().hex
            if request.isolation == "worktree":
                handle = create_worktree(request.workspace_root, branch=f"tamfis/{execution_id[:12]}")
                request.metadata["worktree_path"] = str(handle.path)
                request.metadata["worktree_branch"] = handle.branch
                # The operation runs against the isolated checkout, not the launch
                # workspace -- this is the one place isolation actually changes
                # where execution happens.
                request.workspace_root = str(handle.path)
            started = time.monotonic()
            self._append_event(RuntimeEvent(
                event="execution_started", execution_id=execution_id, mode=request.mode.value,
                session_id=request.session_id, timestamp=_utc_now(), status="running",
                objective=request.objective, workspace_root=request.workspace_root, metadata=request.metadata,
            ))
            self._save_checkpoint(ExecutionCheckpoint(
                execution_id=execution_id, session_id=request.session_id, mode=request.mode.value,
                objective=request.objective, workspace_root=request.workspace_root, status="running",
                metadata={**request.metadata, "task_contract": self.task_contract.to_dict()},
            ))
            self._active_task = asyncio.create_task(operation(), name=f"tamfis:{request.mode.value}:{request.session_id}")
            try:
                result = await self._active_task
            except asyncio.CancelledError:
                self.controller.fail("Execution cancelled by user.")
                if request.mode == ExecutionMode.LOCAL_AGENT and request.session_id:
                    local_state.mark_turn_checkpoint_interrupted(
                        request.session_id, error="Execution cancelled by user.",
                    )
                duration_ms = int((time.monotonic() - started) * 1000)
                self.history.append(ExecutionRecord(request.mode, "cancelled", request.session_id, error="cancelled", execution_id=execution_id, duration_ms=duration_ms))
                self._append_event(RuntimeEvent(event="execution_finished", execution_id=execution_id, mode=request.mode.value, session_id=request.session_id, timestamp=_utc_now(), status="cancelled", objective=request.objective, workspace_root=request.workspace_root, error="cancelled", duration_ms=duration_ms))
                self._save_checkpoint(ExecutionCheckpoint(execution_id=execution_id, session_id=request.session_id, mode=request.mode.value, objective=request.objective, workspace_root=request.workspace_root, status="cancelled", unresolved=["Execution cancelled by user."], metadata=request.metadata))
                raise
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                self.controller.fail(message)
                if request.mode == ExecutionMode.LOCAL_AGENT and request.session_id:
                    # Preserve the last atomically written local turn instead
                    # of allowing a provider disconnect or unexpected worker
                    # failure to look like an unstarted turn on resume.
                    local_state.mark_turn_checkpoint_interrupted(
                        request.session_id, error=message,
                    )
                duration_ms = int((time.monotonic() - started) * 1000)
                self.history.append(ExecutionRecord(request.mode, "failed", request.session_id, error=message, execution_id=execution_id, duration_ms=duration_ms))
                self._append_event(RuntimeEvent(event="execution_finished", execution_id=execution_id, mode=request.mode.value, session_id=request.session_id, timestamp=_utc_now(), status="failed", objective=request.objective, workspace_root=request.workspace_root, error=message, duration_ms=duration_ms))
                self._save_checkpoint(ExecutionCheckpoint(execution_id=execution_id, session_id=request.session_id, mode=request.mode.value, objective=request.objective, workspace_root=request.workspace_root, status="failed", unresolved=[message], metadata=request.metadata))
                raise
            else:
                status = str(getattr(result, "status", "completed"))
                summary = str(getattr(result, "summary", "") or "")
                error = getattr(result, "error", None)
                changed_files = list(getattr(result, "changed_files", []) or [])
                validations = list(getattr(result, "validations", []) or [])
                # Adapters should provide these fields, but derive them from
                # the canonical session ledger as a compatibility bridge for
                # both the legacy remote stream and older local outcomes.
                if request.session_id and (not changed_files or not validations):
                    try:
                        current_state = local_state.get_session_state(request.session_id)
                        if not changed_files:
                            changed_files = list(dict.fromkeys(
                                str(item.get("path"))
                                for item in current_state.modified_files
                                if item.get("path") and str(item.get("mutation_id")) not in baseline_mutation_ids
                            ))
                        if not validations:
                            validations = list(current_state.validation_results[baseline_validation_count:])
                    except OSError:
                        pass
                if hasattr(result, "changed_files"):
                    result.changed_files = changed_files
                if hasattr(result, "validations"):
                    result.validations = validations
                self.evidence_graph.add(EvidenceNode(
                    "completion", "completion_summary", summary or status, "execution_result", ["objective"]
                ))
                self.evidence_graph.add(EvidenceNode(
                    "observation", "tool_observation", "Execution produced a result", "unified_runtime", ["evidence"]
                ))
                for index, path in enumerate(changed_files):
                    self.evidence_graph.add(EvidenceNode(
                        f"mutation-{index}", "file_mutation", str(path), "mutation_ledger", ["mutation"]
                    ))
                for index, validation in enumerate(validations):
                    passed = bool(validation.get("passed", validation.get("success", False))) if isinstance(validation, dict) else False
                    if passed:
                        self.evidence_graph.add(EvidenceNode(
                            f"validation-{index}", "validation_result", str(validation), "validator", ["validation"]
                        ))
                if self.task_contract is not None:
                    self.last_review = IndependentReviewer().review(self.task_contract, self.evidence_graph)
                    if status == "completed" and not self.last_review.approved:
                        status = "partial"
                        missing = self.last_review.missing_requirements + self.last_review.warnings
                        error = "Independent review blocked completion: " + "; ".join(missing)
                if status == "completed":
                    self.controller.complete()
                elif status not in {"cancelled", "partial", "blocked", "no_changes_required"}:
                    self.controller.fail(str(error or status))
                duration_ms = int((time.monotonic() - started) * 1000)
                self.history.append(ExecutionRecord(request.mode, status, request.session_id, summary, error, execution_id, duration_ms))
                self._append_event(RuntimeEvent(event="execution_finished", execution_id=execution_id, mode=request.mode.value, session_id=request.session_id, timestamp=_utc_now(), status=status, objective=request.objective, workspace_root=request.workspace_root, summary=summary, error=str(error or ""), duration_ms=duration_ms))
                self._save_checkpoint(ExecutionCheckpoint(execution_id=execution_id, session_id=request.session_id, mode=request.mode.value, objective=request.objective, workspace_root=request.workspace_root, status=status, changed_files=changed_files, validations=validations, unresolved=[str(error)] if error else [], metadata={**request.metadata, "task_contract": self.task_contract.to_dict() if self.task_contract else None, "evidence_graph": self.evidence_graph.to_dict(), "independent_review": {"approved": self.last_review.approved, "missing_requirements": self.last_review.missing_requirements, "warnings": self.last_review.warnings} if self.last_review else None}))
                return result
            finally:
                self._active_task = None
                self._active_request = None
                self._runner_task = None

    def cancel(self) -> bool:
        task = (
            self._active_task
            if self._active_task is not None and not self._active_task.done()
            else self._runner_task
        )
        if task is None or task.done():
            return False
        task.cancel()
        return True

    async def execute_local(self, **kwargs: Any) -> Any:
        from tamfis_code.runner_local import _run_local_agent_turn_impl

        request = ExecutionRequest(
            mode=ExecutionMode.LOCAL_AGENT,
            session_id=int(kwargs.get("session_id", 0)),
            objective=_latest_user_text(kwargs.get("messages") or []),
            workspace_root=str(kwargs.get("workspace_root") or "."),
            interactive=bool(kwargs.get("interactive", True)),
            approval_policy=str(kwargs.get("approval_policy") or "ask"),
            read_only=bool(kwargs.get("read_only", False)),
        )
        return await self._run_exclusive(request, lambda: _run_local_agent_turn_impl(**kwargs))

    async def execute_remote(self, **kwargs: Any) -> Any:
        from tamfis_code.runner import _run_ai_task_and_stream_impl

        request = ExecutionRequest(
            mode=ExecutionMode.REMOTE_AGENT,
            session_id=int(kwargs.get("session_id", 0)),
            objective=str(kwargs.get("objective") or ""),
            interactive=bool(kwargs.get("interactive", True)),
            approval_policy=str(kwargs.get("approval_policy") or "ask"),
            read_only=str(kwargs.get("mode") or "") in {"chat", "audit", "plan"},
        )
        return await self._run_exclusive(request, lambda: _run_ai_task_and_stream_impl(**kwargs))

    async def execute_local_chat(self, **kwargs: Any) -> str:
        from tamfis_code.local_chat import _run_local_turn_impl

        request = ExecutionRequest(
            mode=ExecutionMode.LOCAL_CHAT,
            objective=_latest_user_text(kwargs.get("messages") or []),
            interactive=False,
            read_only=True,
        )
        return await self._run_exclusive(request, lambda: _run_local_turn_impl(**kwargs))

    async def stream_local_chat(self, **kwargs: Any) -> AsyncIterator[str]:
        """Stream through the unified authority while preserving async iteration.

        A lock is held for the complete stream so another adapter cannot seize
        the same runtime midway through a provider response.
        """
        from tamfis_code.local_chat import _stream_local_turn_impl

        request = ExecutionRequest(
            mode=ExecutionMode.LOCAL_STREAM,
            objective=_latest_user_text(kwargs.get("messages") or []),
            interactive=False,
            read_only=True,
        )
        async with self._lock:
            if self.active:
                raise RuntimeError("an agent execution is already active in this runtime")
            self.controller = ExecutionController()
            self._active_request = request
            self.task_contract = TaskContract.derive(
                request.objective,
                read_only=request.read_only,
                approval_policy=request.approval_policy,
                metadata=request.metadata,
            )
            self.evidence_graph = EvidenceGraph()
            self.last_review = None
            current = asyncio.current_task()
            self._active_task = current
            try:
                async for chunk in _stream_local_turn_impl(**kwargs):
                    yield chunk
                self.controller.complete()
                self.history.append(ExecutionRecord(request.mode, "completed", 0))
            except asyncio.CancelledError:
                self.controller.fail("Execution cancelled by user.")
                self.history.append(ExecutionRecord(request.mode, "cancelled", 0, error="cancelled"))
                raise
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                self.controller.fail(message)
                self.history.append(ExecutionRecord(request.mode, "failed", 0, error=message))
                raise
            finally:
                self._active_task = None
                self._active_request = None


def _latest_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if str(message.get("role") or "") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [str(item.get("text") or "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
            return "\n".join(part for part in parts if part)
    return ""


_DEFAULT_RUNTIME: UnifiedAgentRuntime | None = None


def get_unified_runtime() -> UnifiedAgentRuntime:
    global _DEFAULT_RUNTIME
    if _DEFAULT_RUNTIME is None:
        _DEFAULT_RUNTIME = UnifiedAgentRuntime()
    return _DEFAULT_RUNTIME


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
