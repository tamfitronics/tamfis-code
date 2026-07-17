"""Durable local per-session CLI state: last received event id/task id per session,
so `attach`/`logs --follow` can resume a stream without a full replay-from-
zero, and so a bare `tamfis-code agents`/`status` can show "what was I last
doing" without another round trip.

This is client-side bookkeeping ONLY -- the server (RemoteEvent/RemoteTask
tables) remains the single source of truth for everything that must survive
a lost or wiped local state file; losing this file just means the next
`attach`/`logs --follow` replays from sequence 0 instead of resuming exactly
where it left off, not that any task/event data is lost.

Reuses the same CONFIG_DIR credentials.json/config.toml already live in
(Phase 18's "current canonical equivalent" allowance), rather than
introducing a second state directory.
"""

from __future__ import annotations

import json
import os
import re
import stat
import sys
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from .config import CONFIG_DIR

STATE_PATH = CONFIG_DIR / "state.json"
MAX_ACTION_HISTORY = 250
MAX_CHECKPOINTS = 50
MAX_SAVED_PLANS = 50

_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(authorization\s*:\s*bearer\s+)([^\s]+)"),
    re.compile(r"(?i)\b(password|passwd|token|access_token|refresh_token|api[_-]?key|client_secret)\s*([=:])\s*([^\s&]+)"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def redact_secrets(text: str) -> str:
    value = text
    value = _SECRET_PATTERNS[0].sub(r"\1[REDACTED]", value)
    value = _SECRET_PATTERNS[1].sub(r"\1\2[REDACTED]", value)
    value = _SECRET_PATTERNS[2].sub("[REDACTED_JWT]", value)
    return value


def _sanitize(value: Any) -> Any:
    if isinstance(value, str):
        return redact_secrets(value)
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    return value


@dataclass
class QueuedInstruction:
    id: str
    text: str
    classification: str = "append"
    priority: int = 100
    status: str = "queued"
    created_at: str = field(default_factory=_now)


@dataclass
class AgentAction:
    id: str
    type: str
    purpose: str
    status: str = "planned"
    risk: str = "read_only"
    detail: str = ""
    depends_on: list[str] = field(default_factory=list)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result_summary: str = ""
    attempts: int = 0
    last_error: str = ""


@dataclass
class CodePlan:
    id: str
    objective: str
    content: str
    status: str = "ready"
    source_task_id: Optional[str] = None
    execution_task_id: Optional[str] = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    # Structured mirror of the server's `plan_created` event payload (a list
    # of {"step": str, "status": str} items), plus a client-assigned `index`
    # since the server payload carries no stable step id of its own. `content`
    # remains the human-readable markdown fallback -- this is additive, not a
    # replacement, so nothing that reads `content` today needs to change.
    steps: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SessionState:
    session_id: int
    workspace_root: str = ""
    primary_workspace: str = ""
    allowed_workspaces: list[str] = field(default_factory=list)
    repository_root: Optional[str] = None
    current_working_directory: str = ""
    active_branch: Optional[str] = None
    last_event_id: int = 0
    last_task_id: Optional[str] = None
    active_task: Optional[dict[str, Any]] = None
    current_phase: str = "idle"
    execution_status: str = "idle"
    inspected_files: dict[str, dict[str, Any]] = field(default_factory=dict)
    discovered_symbols: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    discovered_services: list[dict[str, Any]] = field(default_factory=list)
    discovered_reports: list[dict[str, Any]] = field(default_factory=list)
    repository_context: dict[str, Any] = field(default_factory=dict)
    completed_actions: list[dict[str, Any]] = field(default_factory=list)
    pending_actions: list[dict[str, Any]] = field(default_factory=list)
    queued_user_instructions: list[dict[str, Any]] = field(default_factory=list)
    modified_files: list[dict[str, Any]] = field(default_factory=list)
    validation_results: list[dict[str, Any]] = field(default_factory=list)
    unresolved_issues: list[dict[str, Any]] = field(default_factory=list)
    running_action: Optional[dict[str, Any]] = None
    conversation_summary: str = ""
    context_checkpoints: list[dict[str, Any]] = field(default_factory=list)
    saved_plans: list[dict[str, Any]] = field(default_factory=list)
    active_plan_id: Optional[str] = None
    discovery_fingerprint: str = ""
    selected_model: str = "auto"
    selected_provider: Optional[str] = None
    updated_at: str = ""


def _load_raw() -> dict[str, Any]:
    if not STATE_PATH.is_file():
        return {}
    try:
        payload = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except OSError as exc:
        # A permission/ownership mismatch here silently resets every session
        # to blank (no active_plan_id, no saved_plans, no conversation_summary)
        # -- the CLI then looks "amnesiac", re-proposing the same plan every
        # invocation with no memory of prior progress. Surface it instead of
        # swallowing it so that symptom is diagnosable.
        print(
            f"⚠ Could not read local session state at {STATE_PATH} ({exc}). "
            "Continuing with a blank session -- prior plan/task memory is unavailable "
            "until this is fixed (likely an ownership/permission mismatch on "
            f"{CONFIG_DIR}).",
            file=sys.stderr,
        )
        return {}
    except json.JSONDecodeError:
        return {}


def _save_raw(data: dict[str, Any]) -> None:
    """Atomically replace state so a killed CLI cannot leave invalid JSON."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    # Only chmod when it isn't already owner-only -- calling this unconditionally
    # on every save raises PermissionError (uncaught, all the way up) the moment
    # CONFIG_DIR is ever owned by a different user than the caller, which used to
    # crash the whole CLI on its very first state write.
    if stat.S_IMODE(os.stat(CONFIG_DIR).st_mode) != stat.S_IRWXU:
        os.chmod(CONFIG_DIR, stat.S_IRWXU)
    fd, temp_name = tempfile.mkstemp(prefix=".state-", suffix=".json", dir=CONFIG_DIR)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(_sanitize(data), handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_name, stat.S_IRUSR | stat.S_IWUSR)
        os.replace(temp_name, STATE_PATH)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def get_session_state(session_id: int) -> SessionState:
    raw = _load_raw().get(str(session_id))
    if not raw:
        return SessionState(session_id=session_id)
    allowed = set(SessionState.__dataclass_fields__)
    values = {key: value for key, value in raw.items() if key in allowed and key != "session_id"}
    values["last_event_id"] = int(values.get("last_event_id") or 0)
    return SessionState(session_id=session_id, **values)


def put_session_state(state: SessionState) -> None:
    data = _load_raw()
    latest = data.get(str(state.session_id), {})
    # A second `tamfis-code queue ...` process may add an instruction while
    # the streaming process is saving an event cursor. Merge by id so the
    # cursor write cannot erase that newly queued user input.
    if isinstance(latest, dict):
        merged_queue = {item.get("id"): item for item in latest.get("queued_user_instructions", []) if item.get("id")}
        merged_queue.update({item.get("id"): item for item in state.queued_user_instructions if item.get("id")})
        state.queued_user_instructions = sorted(
            merged_queue.values(), key=lambda item: (int(item.get("priority", 100)), item.get("created_at", ""))
        )
        # Event-cursor updates can race a foreground plan save in the same
        # way they race queued instructions. Preserve plans by id as well.
        merged_plans = {item.get("id"): item for item in latest.get("saved_plans", []) if item.get("id")}
        merged_plans.update({item.get("id"): item for item in state.saved_plans if item.get("id")})
        state.saved_plans = sorted(
            merged_plans.values(), key=lambda item: item.get("created_at", "")
        )[-MAX_SAVED_PLANS:]
    state.updated_at = _now()
    data[str(state.session_id)] = asdict(state)
    _save_raw(data)


def save_session_state(
    session_id: int, *, workspace_root: Optional[str] = None,
    last_event_id: Optional[int] = None, last_task_id: Optional[str] = None,
    **updates: Any,
) -> None:
    state = get_session_state(session_id)
    if workspace_root is not None:
        state.workspace_root = workspace_root
        if not state.primary_workspace:
            state.primary_workspace = workspace_root
        if workspace_root not in state.allowed_workspaces:
            state.allowed_workspaces.append(workspace_root)
        if not state.current_working_directory:
            state.current_working_directory = workspace_root
    if last_event_id is not None:
        # Sequence ids only ever move forward for a given session -- never
        # regress state from a stale/out-of-order write (e.g. a slower
        # concurrent `logs --follow` process finishing after a newer one).
        state.last_event_id = max(int(last_event_id), int(state.last_event_id or 0))
    if last_task_id is not None:
        state.last_task_id = last_task_id
    for key, value in updates.items():
        if key in SessionState.__dataclass_fields__ and key != "session_id":
            setattr(state, key, value)
    put_session_state(state)


def start_action(session_id: int, *, action_type: str, purpose: str,
                 risk: str = "read_only", detail: str = "") -> AgentAction:
    state = get_session_state(session_id)
    action = AgentAction(
        id=f"action_{uuid.uuid4().hex[:12]}", type=action_type, purpose=purpose,
        risk=risk, detail=detail, status="running", started_at=_now(),
    )
    state.running_action = asdict(action)
    state.pending_actions.append(asdict(action))
    state.execution_status = "running"
    put_session_state(state)
    return action


# Number of consecutive failures for the same action `purpose` before it's
# escalated into `unresolved_issues` -- below this, a single failure is just
# noise (transient network blip, etc), not yet "the agent is stuck".
FAILURE_ESCALATION_THRESHOLD = 2


def finish_action(session_id: int, action_id: str, *, status: str, summary: str = "", error: str = "") -> None:
    state = get_session_state(session_id)
    finished = None
    remaining = []
    for action in state.pending_actions:
        if action.get("id") == action_id:
            action.update(status=status, completed_at=_now(), result_summary=summary)
            if status == "failed":
                action["last_error"] = error or summary
                purpose = action.get("purpose", "")
                prior_failures = sum(
                    1 for completed in state.completed_actions
                    if completed.get("purpose") == purpose and completed.get("status") == "failed"
                )
                action["attempts"] = prior_failures + 1
                if action["attempts"] >= FAILURE_ESCALATION_THRESHOLD and not any(
                    issue.get("type") == "repeated_action_failure" and issue.get("purpose") == purpose
                    for issue in state.unresolved_issues
                ):
                    state.unresolved_issues.append({
                        "type": "repeated_action_failure", "status": "needs_attention",
                        "purpose": purpose, "attempts": action["attempts"],
                        "detail": (
                            f"'{purpose}' has failed {action['attempts']} times in a row -- "
                            "consider a different approach instead of retrying as-is."
                        ),
                    })
            finished = action
        else:
            remaining.append(action)
    state.pending_actions = remaining
    if finished:
        state.completed_actions = (state.completed_actions + [finished])[-MAX_ACTION_HISTORY:]
    if state.running_action and state.running_action.get("id") == action_id:
        state.running_action = None
    state.execution_status = "idle" if state.running_action is None else "running"
    put_session_state(state)


def enqueue_instruction(session_id: int, text: str, *, classification: str = "append",
                        priority: int = 100) -> QueuedInstruction:
    state = get_session_state(session_id)
    item = QueuedInstruction(id=f"instruction_{uuid.uuid4().hex[:10]}", text=text,
                             classification=classification, priority=priority)
    state.queued_user_instructions.append(asdict(item))
    state.queued_user_instructions.sort(
        key=lambda value: (int(value.get("priority", 100)), value.get("created_at", ""))
    )
    put_session_state(state)
    return item


def update_instruction(session_id: int, instruction_id: str, status: str) -> bool:
    state = get_session_state(session_id)
    for item in state.queued_user_instructions:
        if item.get("id") == instruction_id:
            item["status"] = status
            put_session_state(state)
            return True
    return False


def checkpoint(session_id: int, *, reason: str, summary: str = "") -> None:
    state = get_session_state(session_id)
    state.context_checkpoints = (state.context_checkpoints + [{
        "created_at": _now(), "reason": reason, "phase": state.current_phase,
        "task_id": state.last_task_id, "last_event_id": state.last_event_id,
        "summary": summary,
    }])[-MAX_CHECKPOINTS:]
    put_session_state(state)


def save_plan(
    session_id: int, *, objective: str, content: str,
    source_task_id: Optional[str] = None,
    steps: Optional[list[dict[str, Any]]] = None,
) -> CodePlan:
    """Persist a completed planning result as an executable plan."""
    state = get_session_state(session_id)
    plan = CodePlan(
        id=f"plan_{uuid.uuid4().hex[:10]}",
        objective=objective.strip(), content=content.strip(),
        source_task_id=source_task_id,
        steps=[
            {**step, "index": index}
            for index, step in enumerate(steps or [])
            if isinstance(step, dict)
        ],
    )
    state.saved_plans = (state.saved_plans + [asdict(plan)])[-MAX_SAVED_PLANS:]
    state.active_plan_id = plan.id
    put_session_state(state)
    return plan


def get_plan(session_id: int, plan_id: Optional[str] = None) -> Optional[dict[str, Any]]:
    """Return the selected plan, accepting an exact id, unique prefix, or latest."""
    state = get_session_state(session_id)
    plans = state.saved_plans
    if not plans:
        return None
    wanted = (plan_id or state.active_plan_id or "").strip()
    if not wanted:
        return plans[-1]
    exact = next((item for item in plans if item.get("id") == wanted), None)
    if exact:
        return exact
    matches = [item for item in plans if str(item.get("id", "")).startswith(wanted)]
    return matches[0] if len(matches) == 1 else None


def update_plan(
    session_id: int, plan_id: str, *, status: str,
    execution_task_id: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    state = get_session_state(session_id)
    updated = None
    for item in state.saved_plans:
        if item.get("id") == plan_id:
            item["status"] = status
            item["updated_at"] = _now()
            if execution_task_id is not None:
                item["execution_task_id"] = execution_task_id
            updated = item
            break
    if updated is not None:
        state.active_plan_id = plan_id
        put_session_state(state)
    return updated


def update_plan_steps(session_id: int, plan_id: str, items: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Persist the server's `plan_created.items` onto the matching saved plan.

    Today the server only ever emits this list once per plan and never
    re-references individual steps afterward, so `steps` reflects whatever
    the most recent `plan_created` event said -- there's no per-step
    completion event to key off yet.
    """
    state = get_session_state(session_id)
    updated = None
    for item in state.saved_plans:
        if item.get("id") == plan_id:
            item["steps"] = [
                {**step, "index": index}
                for index, step in enumerate(items)
                if isinstance(step, dict)
            ]
            item["updated_at"] = _now()
            updated = item
            break
    if updated is not None:
        put_session_state(state)
    return updated


def plan_execution_objective(plan: dict[str, Any]) -> str:
    """Turn a reviewed saved plan into an unambiguous agent execution task."""
    return (
        "Execute the saved engineering plan below. Inspect the current workspace first and "
        "adapt only where repository drift requires it. Implement the work, run proportionate "
        "validation, and report concrete results. Do not merely restate the plan.\n\n"
        f"Original objective:\n{plan.get('objective', '')}\n\n"
        f"Saved plan ({plan.get('id', 'unknown')}):\n{plan.get('content', '')}"
    )


def all_known_session_ids() -> list[int]:
    ids = []
    for key in _load_raw().keys():
        try:
            ids.append(int(key))
        except ValueError:
            continue
    return sorted(ids)
