"""Durable scheduled automations for the local Tamfis Code runtime."""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Iterable


@dataclass(slots=True)
class Automation:
    name: str
    objective: str
    workspace: str
    interval_seconds: float
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    enabled: bool = True
    last_run: float | None = None
    next_run: float | None = None
    approval_policy: str = "accept-edits"
    last_status: str | None = None
    last_error: str | None = None

    def __post_init__(self) -> None:
        self.name = self.name.strip()
        self.objective = self.objective.strip()
        self.workspace = str(Path(self.workspace).expanduser().resolve())
        self.interval_seconds = float(self.interval_seconds)
        if not self.name:
            raise ValueError("automation name cannot be empty")
        if not self.objective:
            raise ValueError("automation objective cannot be empty")
        if self.interval_seconds < 60:
            raise ValueError("automation interval must be at least 60 seconds")


class AutomationStore:
    """Atomic JSON persistence with name uniqueness."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[Automation]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid automation store {self.path}: {exc}") from exc
        if not isinstance(payload, list):
            raise ValueError(f"invalid automation store {self.path}: expected a JSON list")
        return [Automation(**item) for item in payload]

    def save(self, items: Iterable[Automation]) -> None:
        materialized = list(items)
        names = [item.name for item in materialized]
        if len(names) != len(set(names)):
            raise ValueError("automation names must be unique")
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps([asdict(item) for item in materialized], indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def get(self, name: str) -> Automation:
        for item in self.load():
            if item.name == name or item.id == name:
                return item
        raise KeyError(name)

    def upsert(self, item: Automation, *, replace: bool = False) -> Automation:
        items = self.load()
        existing = next((index for index, value in enumerate(items) if value.name == item.name), None)
        if existing is not None and not replace:
            raise ValueError(f"automation already exists: {item.name}")
        if existing is None:
            items.append(item)
        else:
            item.id = items[existing].id
            items[existing] = item
        self.save(items)
        return item

    def update_enabled(self, name: str, enabled: bool) -> Automation:
        items = self.load()
        for item in items:
            if item.name == name or item.id == name:
                item.enabled = enabled
                self.save(items)
                return item
        raise KeyError(name)

    def remove(self, name: str) -> Automation:
        items = self.load()
        for index, item in enumerate(items):
            if item.name == name or item.id == name:
                removed = items.pop(index)
                self.save(items)
                return removed
        raise KeyError(name)


class AutomationScheduler:
    """Poll a store and persist run timestamps after each attempted run."""

    def __init__(
        self,
        store: AutomationStore,
        runner: Callable[[Automation], Awaitable[None]],
    ):
        self.store = store
        self.runner = runner
        self._stop = False

    async def run_due(self, *, now: float | None = None) -> list[str]:
        current = time.time() if now is None else now
        items = self.store.load()
        ran: list[str] = []
        for item in items:
            due = item.enabled and (item.next_run is None or item.next_run <= current)
            if not due:
                continue
            # Reserve the next slot before execution. A killed scheduler will not
            # immediately duplicate an expensive run when it restarts.
            item.last_run = current
            item.next_run = current + item.interval_seconds
            self.store.save(items)
            try:
                await self.runner(item)
            except Exception as exc:
                item.last_status = "failed"
                item.last_error = str(exc)[:2_000]
            else:
                item.last_status = "completed"
                item.last_error = None
            self.store.save(items)
            ran.append(item.name)
        return ran

    async def run_forever(self, poll: float = 1.0) -> None:
        self._stop = False
        while not self._stop:
            await self.run_due()
            await asyncio.sleep(max(0.1, poll))

    def stop(self) -> None:
        self._stop = True
