from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator


class EventKind(str, Enum):
    USER_MESSAGE = "user_message"
    AGENT_MESSAGE = "agent_message"
    STATE_CHANGED = "state_changed"
    PLAN_CREATED = "plan_created"
    PLAN_UPDATED = "plan_updated"
    ACTION = "action"
    OBSERVATION = "observation"
    TOOL_STARTED = "tool_started"
    TOOL_FINISHED = "tool_finished"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_RESOLVED = "approval_resolved"
    FILE_CHANGED = "file_changed"
    SNAPSHOT_CREATED = "snapshot_created"
    VALIDATION = "validation"
    DELEGATION_STARTED = "delegation_started"
    DELEGATION_FINISHED = "delegation_finished"
    ARTIFACT_CREATED = "artifact_created"
    ERROR = "error"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class Event:
    conversation_id: str
    kind: EventKind
    payload: dict[str, Any]
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    sequence: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    actor: str = "system"
    parent_id: str | None = None
    correlation_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        clean = dict(data)
        clean["kind"] = EventKind(clean["kind"])
        return cls(**clean)


class EventStore:
    """Append-only JSONL event store with bounded streaming-write latency.

    Agent deltas can arrive many times per second.  They remain immediately
    visible to readers, while their durable flush is bounded to 50ms instead
    of blocking the agent event loop on an fsync for every token.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._subscribers: dict[str, list[Callable[[Event], None]]] = {}
        self._sequences: dict[str, int] = {}
        self._last_fsync: dict[str, float] = {}

    def _path(self, conversation_id: str) -> Path:
        safe = "".join(ch for ch in conversation_id if ch.isalnum() or ch in "-_")
        if not safe:
            raise ValueError("invalid conversation id")
        return self.root / f"{safe}.jsonl"

    def append(self, event: Event) -> Event:
        with self._lock:
            conversation_id = event.conversation_id
            if conversation_id not in self._sequences:
                self._sequences[conversation_id] = self.count(conversation_id)
            sequence = self._sequences[conversation_id] + 1
            self._sequences[conversation_id] = sequence
            stored = Event(
                conversation_id=event.conversation_id, kind=event.kind,
                payload=event.payload, id=event.id, sequence=sequence,
                created_at=event.created_at, actor=event.actor,
                parent_id=event.parent_id, correlation_id=event.correlation_id,
            )
            path = self._path(event.conversation_id)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(stored.as_dict(), ensure_ascii=False, default=str) + "\n")
                handle.flush()
                now = time.monotonic()
                # Preserve immediate durability for lifecycle/control events.
                # Streaming deltas are flushed at least every 50ms, enough to
                # survive normal process failures without stalling rendering.
                should_sync = (
                    stored.kind != EventKind.AGENT_MESSAGE
                    or now - self._last_fsync.get(conversation_id, 0.0) >= 0.05
                )
                if should_sync:
                    os.fsync(handle.fileno())
                    self._last_fsync[conversation_id] = now
            for callback in list(self._subscribers.get(event.conversation_id, [])):
                callback(stored)
            return stored

    def emit(self, conversation_id: str, kind: EventKind, payload: dict[str, Any], **kwargs: Any) -> Event:
        return self.append(Event(conversation_id=conversation_id, kind=kind, payload=payload, **kwargs))

    def read(self, conversation_id: str, *, after: int = 0) -> list[Event]:
        path = self._path(conversation_id)
        if not path.exists():
            return []
        events: list[Event] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                event = Event.from_dict(json.loads(line))
                if event.sequence > after:
                    events.append(event)
        return events

    def stream(self, conversation_id: str, *, after: int = 0) -> Iterator[Event]:
        yield from self.read(conversation_id, after=after)

    def count(self, conversation_id: str) -> int:
        path = self._path(conversation_id)
        if not path.exists():
            return 0
        with path.open("rb") as handle:
            return sum(1 for line in handle if line.strip())

    def list_conversations(self) -> list[str]:
        return sorted(path.stem for path in self.root.glob("*.jsonl"))

    def subscribe(self, conversation_id: str, callback: Callable[[Event], None]) -> Callable[[], None]:
        self._subscribers.setdefault(conversation_id, []).append(callback)
        def unsubscribe() -> None:
            callbacks = self._subscribers.get(conversation_id, [])
            if callback in callbacks:
                callbacks.remove(callback)
        return unsubscribe
