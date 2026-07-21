"""Canonical provider/tool events used by the Tamfis-Code orchestrator."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AgentPhase(str, Enum):
    UNDERSTAND = "understand"
    INSPECT = "inspect"
    ROUTE = "route"
    PLAN = "plan"
    EXECUTE = "execute"
    OBSERVE = "observe"
    REPAIR = "repair"
    VALIDATE = "validate"
    REPORT = "report"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    COMPLETED = "completed"
    FAILED = "failed"


class EventType(str, Enum):
    STATUS = "status"
    REASONING_DELTA = "reasoning_delta"
    ASSISTANT_DELTA = "assistant_delta"
    ASSISTANT_COMPLETED = "assistant_completed"
    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_CALL_DELTA = "tool_call_delta"
    TOOL_CALL_COMPLETED = "tool_call_completed"
    TOOL_RESULT = "tool_result"
    ARTIFACT_GENERATED = "artifact_generated"
    FILE_GENERATED = "file_generated"
    IMAGE_GENERATED = "image_generated"
    VIDEO_GENERATED = "video_generated"
    DIFF_AVAILABLE = "diff_available"
    USAGE = "usage"
    ERROR = "error"
    DONE = "done"


@dataclass
class CanonicalEvent:
    event_type: EventType
    payload: dict[str, Any] = field(default_factory=dict)
    provider: str | None = None
    model: str | None = None
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["event_type"] = self.event_type.value
        return data


@dataclass
class ToolEnvelope:
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    purpose: str
    risk: str = "read_only"
    requires_approval: bool = False
    cwd: str | None = None
    started_at: str = field(default_factory=utc_now)
    completed_at: str | None = None
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    success: bool | None = None
    files_changed: list[str] = field(default_factory=list)
    retry_of: str | None = None

    def finish(self, *, result: dict[str, Any], success: bool) -> None:
        self.completed_at = utc_now()
        self.success = success
        actual = result.get("result") if isinstance(result.get("result"), dict) else result
        self.exit_code = actual.get("return_code", actual.get("exit_code"))
        self.stdout = str(actual.get("stdout") or "")
        self.stderr = str(actual.get("stderr") or actual.get("error") or "")
        path = (
            actual.get("path")
            or actual.get("destination")
            or self.arguments.get("output_path")
            or self.arguments.get("destination")
            or self.arguments.get("path")
        )
        if success and self.tool_name in {
            "write_file", "edit_file", "create_file", "patch_file",
            "extract_archive", "repackage_archive",
        } and path:
            self.files_changed.append(str(path))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
