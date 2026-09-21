"""Execution-progress tracking and the stall watchdog's state machine.

A spinner redrawing proves nothing: it kept animating for 27 minutes over a
dead provider request. ``last_progress_at`` therefore moves only when the run
does something real -- a model token, a tool starting/finishing, a state
transition, a follow-up arriving -- and never from a UI redraw.

The tracker is pure (no I/O, injectable clock) so the state machine can be
tested without a terminal:

    RUNNING            tokens are arriving / nothing is being awaited
    WAITING_PROVIDER   a model request is in flight, no output yet
    WAITING_TOOL       one or more tool calls have not reported back
    WAITING_USER       an approval prompt is open
    RETRYING           the last provider attempt failed; another is starting
    RATE_LIMITED       the provider answered 429 / asked us to slow down
    QUOTA_EXHAUSTED    the provider's quota/credit is spent
    STALLED            waiting on the provider past the warning threshold
    CANCELLING         an interrupt was requested and is being honoured
    FAILED / COMPLETED terminal
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional


class ExecState(str, Enum):
    RUNNING = "running"
    WAITING_PROVIDER = "waiting_provider"
    WAITING_TOOL = "waiting_tool"
    WAITING_USER = "waiting_user"
    RETRYING = "retrying"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXHAUSTED = "quota_exhausted"
    STALLED = "stalled"
    CANCELLING = "cancelling"
    FAILED = "failed"
    COMPLETED = "completed"


# Short, honest labels for the status line. "Waiting for the model's next step"
# is only right for WAITING_PROVIDER -- every other quiet state says what it is.
STATE_LABELS: dict[ExecState, str] = {
    ExecState.RUNNING: "Working",
    ExecState.WAITING_PROVIDER: "Waiting for the model",
    ExecState.WAITING_TOOL: "Waiting for a tool",
    ExecState.WAITING_USER: "Waiting for your approval",
    ExecState.RETRYING: "Retrying the request",
    ExecState.RATE_LIMITED: "Rate limited — backing off",
    ExecState.QUOTA_EXHAUSTED: "Provider quota exhausted",
    ExecState.STALLED: "Provider not responding",
    ExecState.CANCELLING: "Cancelling",
    ExecState.FAILED: "Failed",
    ExecState.COMPLETED: "Completed",
}


def _env_seconds(name: str, default: float, *, minimum: float = 1.0) -> float:
    try:
        return max(minimum, float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class StallPolicy:
    """Per-operation stall thresholds (seconds). Not one global timeout: a
    build may legitimately be silent for many minutes, a model request may not.

    ``provider_warn``   waiting on the model with no output -> STALLED + notice.
    ``provider_abort``  last resort: the request-level timeouts (first-byte /
                        idle / total, in runner_local) should have fired long
                        before this. If they did not, the watchdog stops the
                        stuck request itself instead of waiting silently.
    Tools and shell commands carry their own timeouts and are never aborted by
    the watchdog; they only report as WAITING_TOOL.
    """

    provider_warn: float = 45.0
    provider_abort: float = 240.0
    # Nothing at all (no token, no tool start/finish, no event) for this long while NOT waiting on
    # the model, a tool or the user. Every legitimate operation in that gap (context compaction,
    # planning, hooks) is bounded by its own timeouts well under this; a run silent for longer than
    # that is wedged. A 260-minute "Reviewing the tool result..." was exactly this case.
    silent_abort: float = 600.0

    @classmethod
    def from_env(cls) -> "StallPolicy":
        warn = _env_seconds("TAMFIS_CODE_STALL_WARN_SECONDS", cls.provider_warn, minimum=5.0)
        abort = _env_seconds("TAMFIS_CODE_STALL_ABORT_SECONDS", cls.provider_abort, minimum=10.0)
        silent = _env_seconds("TAMFIS_CODE_SILENT_ABORT_SECONDS", cls.silent_abort, minimum=30.0)
        return cls(provider_warn=warn, provider_abort=max(abort, warn + 5.0), silent_abort=max(silent, abort))


# Events that are real execution activity. Anything not listed (diagnostics,
# UI-only events) must not refresh last_progress_at.
_PROGRESS_EVENTS = frozenset({
    "task_started", "context_loading", "context_reused", "context_rescanned",
    "routing_started", "model_selected", "provider_request_started",
    "tool_call_requested", "tool_output", "file_mutation", "plan_created",
    "plan_updated", "plan_step_started", "plan_step_completed", "user_message",
    "approval_required", "approval_auto", "context_rollover",
})
_RESUMING_EVENTS = frozenset({
    "tool_call_requested", "tool_output", "file_mutation", "provider_request_started",
})


_QUOTA_RE = re.compile(r"quota|insufficient[_ ]?(?:funds|credit|balance)|payment required|billing|\b402\b", re.I)
_RATE_RE = re.compile(r"rate[- ]?limit|too many requests|\b429\b|overloaded|slow down", re.I)


def classify_provider_failure(error: object) -> ExecState:
    """Map a provider failure to RATE_LIMITED / QUOTA_EXHAUSTED / RETRYING."""
    status = getattr(error, "status_code", None)
    text = f"{status or ''} {error}"
    if status == 402 or _QUOTA_RE.search(text):
        return ExecState.QUOTA_EXHAUSTED
    if status == 429 or _RATE_RE.search(text):
        return ExecState.RATE_LIMITED
    return ExecState.RETRYING


@dataclass(frozen=True)
class ProgressSnapshot:
    state: ExecState
    label: str
    idle_seconds: float
    pending_tools: int
    last_event: str


class ProgressTracker:
    """Tracks meaningful progress and derives the execution state."""

    def __init__(
        self,
        policy: Optional[StallPolicy] = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.policy = policy or StallPolicy.from_env()
        self._clock = clock
        self.last_progress_at = clock()
        self.last_event = "start"
        self.pending_tools = 0
        self._awaiting: Optional[str] = None  # "provider" | "user" | None
        self._provider_state: Optional[ExecState] = None
        self._terminal: Optional[ExecState] = None
        self._cancelling = False

    # ---- inputs -----------------------------------------------------------
    def touch(self, kind: str) -> None:
        self.last_progress_at = self._clock()
        self.last_event = kind

    def observe(self, event_type: str, payload: Optional[dict] = None) -> None:
        payload = payload or {}
        if event_type == "task_started":
            self.reset()
            return
        if event_type in {"assistant_delta", "reasoning_delta"}:
            if str(payload.get("content") or ""):
                self._awaiting = None
                self._provider_state = None
                self.touch(event_type)
            return
        if event_type not in _PROGRESS_EVENTS:
            return
        if event_type == "provider_request_started":
            self._awaiting = "provider"
            self._provider_state = None
        elif event_type == "approval_required":
            self._awaiting = "user"
        elif event_type in _RESUMING_EVENTS and self._awaiting == "user":
            self._awaiting = None
        if event_type == "tool_call_requested":
            self.pending_tools += 1
            self._awaiting = None if self._awaiting == "provider" else self._awaiting
        elif event_type == "tool_output":
            self.pending_tools = max(0, self.pending_tools - 1)
        self.touch(event_type)

    def note_provider_failure(self, error: object) -> ExecState:
        """A provider attempt failed; record why (state persists until output)."""
        self._provider_state = classify_provider_failure(error)
        self._awaiting = "provider"
        self.touch("provider_failure")
        return self._provider_state

    def request_cancel(self) -> None:
        self._cancelling = True
        self.touch("cancel_requested")

    def finish(self, status: str) -> None:
        ok = str(status or "").lower() in {"completed", "complete", "success", "done"}
        self._terminal = ExecState.COMPLETED if ok else ExecState.FAILED
        self.touch("finished")

    def reset(self) -> None:
        self.last_progress_at = self._clock()
        self.last_event = "task_started"
        self.pending_tools = 0
        self._awaiting = None
        self._provider_state = None
        self._terminal = None
        self._cancelling = False

    # ---- derived ----------------------------------------------------------
    def idle_seconds(self) -> float:
        return max(0.0, self._clock() - self.last_progress_at)

    def state(self) -> ExecState:
        if self._terminal is not None:
            return self._terminal
        if self._cancelling:
            return ExecState.CANCELLING
        if self._awaiting == "user":
            return ExecState.WAITING_USER
        if self.pending_tools > 0:
            return ExecState.WAITING_TOOL
        if self._awaiting == "provider":
            if self._provider_state is not None:
                return self._provider_state
            if self.idle_seconds() >= self.policy.provider_warn:
                return ExecState.STALLED
            return ExecState.WAITING_PROVIDER
        return ExecState.RUNNING

    def snapshot(self) -> ProgressSnapshot:
        state = self.state()
        return ProgressSnapshot(
            state=state,
            label=STATE_LABELS[state],
            idle_seconds=self.idle_seconds(),
            pending_tools=self.pending_tools,
            last_event=self.last_event,
        )

    def should_abort_silent(self) -> bool:
        """True when the run has produced no activity for ``silent_abort`` seconds while it is not
        waiting on the provider (covered above), a tool (own timeouts) or the user."""
        return (
            self._terminal is None
            and not self._cancelling
            and self._awaiting is None
            and self.pending_tools == 0
            and self.idle_seconds() >= self.policy.silent_abort
        )

    def should_abort_provider_wait(self) -> bool:
        """True when a model request has produced nothing for so long that the
        request-level timeouts evidently failed to fire."""
        return (
            self._terminal is None
            and not self._cancelling
            and self._awaiting == "provider"
            and self.pending_tools == 0
            and self.idle_seconds() >= self.policy.provider_abort
        )


_LOG_CONFIGURED = False


def configure_execution_log() -> None:
    """Send `tamfis_code.*` diagnostics (state transitions, tool batches,
    follow-ups, watchdog) to ``<config>/logs/execution.log`` -- never to the
    terminal, where a stray log line would be painted over the composer. No
    secrets are logged: tool names, counts, ids and timings only."""
    global _LOG_CONFIGURED
    import logging
    import logging.handlers

    root = logging.getLogger("tamfis_code")
    if not any(isinstance(h, logging.NullHandler) for h in root.handlers):
        root.addHandler(logging.NullHandler())  # suppresses logging.lastResort -> stderr
    if _LOG_CONFIGURED:
        return
    try:
        from .. import state as local_state

        directory = local_state.CONFIG_DIR / "logs"
        directory.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            directory / "execution.log", maxBytes=1_000_000, backupCount=2, encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        root.addHandler(handler)
        if root.level == logging.NOTSET or root.level > logging.INFO:
            root.setLevel(logging.INFO)
        _LOG_CONFIGURED = True
    except Exception:
        pass
