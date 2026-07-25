"""Durable, cross-session memory store.

Unlike ``state.py``'s per-session ``.memory/session-<id>.json`` snapshot
(which is scoped to one session and never recalled by a later one), this
module persists small, named, typed records that survive across sessions and
can be searched by keyword when a new objective comes in.
"""
from __future__ import annotations

import json
import os
import re
import stat
import tempfile
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable

from ..config import CONFIG_DIR

MEMORY_DIR = CONFIG_DIR / "memory"
INDEX_PATH = MEMORY_DIR / "index.json"

_SLUG_RE = re.compile(r"[^a-z0-9]+")


class MemoryType(str, Enum):
    USER = "user"
    FEEDBACK = "feedback"
    PROJECT = "project"
    REFERENCE = "reference"


class MemoryError(RuntimeError):
    """Raised for malformed memory records or store corruption callers must handle."""


@dataclass(slots=True)
class MemoryRecord:
    name: str
    type: MemoryType
    description: str
    content: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["type"] = self.type.value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "MemoryRecord":
        return cls(
            name=str(payload["name"]),
            type=MemoryType(str(payload["type"])),
            description=str(payload.get("description") or ""),
            content=str(payload.get("content") or ""),
            created_at=float(payload.get("created_at") or time.time()),
            updated_at=float(payload.get("updated_at") or time.time()),
        )


def slugify(name: str) -> str:
    slug = _SLUG_RE.sub("-", name.strip().lower()).strip("-")
    if not slug:
        raise MemoryError(f"Cannot derive a memory slug from {name!r}.")
    return slug


def _lock_down(path: Path) -> None:
    if stat.S_IMODE(os.stat(path).st_mode) != stat.S_IRWXU:
        os.chmod(path, stat.S_IRWXU)


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _lock_down(path.parent)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_name, stat.S_IRUSR | stat.S_IWUSR)
        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


class MemoryStore:
    """File-based memory persistence rooted at ``CONFIG_DIR/memory``."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or MEMORY_DIR
        self.index_path = self.root / "index.json"

    def _record_path(self, slug: str) -> Path:
        return self.root / f"{slug}.json"

    def _read_index(self) -> dict[str, dict[str, object]]:
        if not self.index_path.is_file():
            return {}
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _rebuild_index_entry(self, record: MemoryRecord) -> dict[str, object]:
        return {
            "type": record.type.value,
            "description": record.description,
            "updated_at": record.updated_at,
        }

    def save(self, record: MemoryRecord) -> MemoryRecord:
        slug = slugify(record.name)
        record = MemoryRecord(
            name=slug,
            type=record.type,
            description=record.description,
            content=record.content,
            created_at=record.created_at,
            updated_at=time.time(),
        )
        _atomic_write_json(self._record_path(slug), record.to_dict())
        index = self._read_index()
        index[slug] = self._rebuild_index_entry(record)
        _atomic_write_json(self.index_path, index)
        return record

    def load(self, name: str) -> MemoryRecord | None:
        slug = slugify(name)
        path = self._record_path(slug)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MemoryError(f"Corrupt memory record {slug!r}: {exc}") from exc
        return MemoryRecord.from_dict(payload)

    def delete(self, name: str) -> bool:
        slug = slugify(name)
        path = self._record_path(slug)
        existed = path.is_file()
        if existed:
            path.unlink()
        index = self._read_index()
        if slug in index:
            del index[slug]
            _atomic_write_json(self.index_path, index)
        return existed

    def list(self, type: MemoryType | None = None) -> list[MemoryRecord]:
        index = self._read_index()
        # The index is a cache, not the source of truth -- a record deleted
        # out-of-band (or an index that predates a record) must not desync
        # what `list`/`search` report, so reconcile against the directory.
        on_disk = {path.stem for path in self.root.glob("*.json") if path.name != "index.json"}
        stale = set(index) - on_disk
        missing = on_disk - set(index)
        if stale or missing:
            for slug in stale:
                del index[slug]
            for slug in missing:
                record = self.load(slug)
                if record is not None:
                    index[slug] = self._rebuild_index_entry(record)
            _atomic_write_json(self.index_path, index)
        records = [self.load(slug) for slug in index]
        results = [record for record in records if record is not None]
        if type is not None:
            results = [record for record in results if record.type == type]
        return sorted(results, key=lambda record: record.updated_at, reverse=True)

    def search(self, query: str, *, limit: int = 5) -> list[MemoryRecord]:
        terms = [term for term in re.split(r"\W+", query.lower()) if len(term) > 2]
        if not terms:
            return []
        scored: list[tuple[int, MemoryRecord]] = []
        for record in self.list():
            haystack = f"{record.name} {record.description} {record.content}".lower()
            score = sum(haystack.count(term) for term in terms)
            if score > 0:
                scored.append((score, record))
        scored.sort(key=lambda pair: (pair[0], pair[1].updated_at), reverse=True)
        return [record for _, record in scored[:limit]]


_DEFAULT_STORE: MemoryStore | None = None


def get_memory_store() -> MemoryStore:
    global _DEFAULT_STORE
    if _DEFAULT_STORE is None:
        _DEFAULT_STORE = MemoryStore()
    return _DEFAULT_STORE


def relevant_memories(objective: str, *, limit: int = 5) -> list[MemoryRecord]:
    if not objective:
        return []
    return get_memory_store().search(objective, limit=limit)
