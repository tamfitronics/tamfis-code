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
from .reviewer import IndependentReviewer


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
        self._active_request: ExecutionRequest | None = None
        self.history: list[ExecutionRecord] = []
        self._lock = asyncio.Lock()
        self.task_contract: TaskContract | None = None
        self.evidence_graph = EvidenceGraph()
        self.last_review = None

    @property
    def active(self) -> bool:
        return self._active_task is not None and not self._active_task.done()

    async def _run_exclusive(self, request: ExecutionRequest, operation: Callable[[], Awaitable[Any]]) -> Any:
        # Fail fast instead of queueing a second agent behind the first. A queued
        # call can capture stale terminal/session state and was one source of the
        # overlapping-runtime behaviour Phase 2 is designed to eliminate.
        if self.active or self._lock.locked():
            raise RuntimeError("an agent execution is already active in this runtime")
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
            execution_id = uuid.uuid4().hex
            started = time.monotonic()
            append_event(RuntimeEvent(
                event="execution_started", execution_id=execution_id, mode=request.mode.value,
                session_id=request.session_id, timestamp=_utc_now(), status="running",
                objective=request.objective, workspace_root=request.workspace_root, metadata=request.metadata,
            ))
            save_checkpoint(ExecutionCheckpoint(
                execution_id=execution_id, session_id=request.session_id, mode=request.mode.value,
                objective=request.objective, workspace_root=request.workspace_root, status="running",
                metadata={**request.metadata, "task_contract": self.task_contract.to_dict()},
            ))
            self._active_task = asyncio.create_task(operation(), name=f"tamfis:{request.mode.value}:{request.session_id}")
            try:
                result = await self._active_task
            except asyncio.CancelledError:
                self.controller.fail("Execution cancelled by user.")
                duration_ms = int((time.monotonic() - started) * 1000)
                self.history.append(ExecutionRecord(request.mode, "cancelled", request.session_id, error="cancelled", execution_id=execution_id, duration_ms=duration_ms))
                append_event(RuntimeEvent(event="execution_finished", execution_id=execution_id, mode=request.mode.value, session_id=request.session_id, timestamp=_utc_now(), status="cancelled", objective=request.objective, workspace_root=request.workspace_root, error="cancelled", duration_ms=duration_ms))
                save_checkpoint(ExecutionCheckpoint(execution_id=execution_id, session_id=request.session_id, mode=request.mode.value, objective=request.objective, workspace_root=request.workspace_root, status="cancelled", unresolved=["Execution cancelled by user."], metadata=request.metadata))
                raise
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                self.controller.fail(message)
                duration_ms = int((time.monotonic() - started) * 1000)
                self.history.append(ExecutionRecord(request.mode, "failed", request.session_id, error=message, execution_id=execution_id, duration_ms=duration_ms))
                append_event(RuntimeEvent(event="execution_finished", execution_id=execution_id, mode=request.mode.value, session_id=request.session_id, timestamp=_utc_now(), status="failed", objective=request.objective, workspace_root=request.workspace_root, error=message, duration_ms=duration_ms))
                save_checkpoint(ExecutionCheckpoint(execution_id=execution_id, session_id=request.session_id, mode=request.mode.value, objective=request.objective, workspace_root=request.workspace_root, status="failed", unresolved=[message], metadata=request.metadata))
                raise
            else:
                status = str(getattr(result, "status", "completed"))
                summary = str(getattr(result, "summary", "") or "")
                error = getattr(result, "error", None)
                changed_files = list(getattr(result, "changed_files", []) or [])
                validations = list(getattr(result, "validations", []) or [])
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
                append_event(RuntimeEvent(event="execution_finished", execution_id=execution_id, mode=request.mode.value, session_id=request.session_id, timestamp=_utc_now(), status=status, objective=request.objective, workspace_root=request.workspace_root, summary=summary, error=str(error or ""), duration_ms=duration_ms))
                save_checkpoint(ExecutionCheckpoint(execution_id=execution_id, session_id=request.session_id, mode=request.mode.value, objective=request.objective, workspace_root=request.workspace_root, status=status, changed_files=changed_files, validations=validations, unresolved=[str(error)] if error else [], metadata={**request.metadata, "task_contract": self.task_contract.to_dict() if self.task_contract else None, "evidence_graph": self.evidence_graph.to_dict(), "independent_review": {"approved": self.last_review.approved, "missing_requirements": self.last_review.missing_requirements, "warnings": self.last_review.warnings} if self.last_review else None}))
                return result
            finally:
                self._active_task = None
                self._active_request = None

    def cancel(self) -> bool:
        task = self._active_task
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
