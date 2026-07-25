"""Persistent repository intelligence for avoiding repeated reconnaissance."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class RepositorySnapshot:
    root: str
    fingerprint: str
    files: dict[str, dict[str, Any]] = field(default_factory=dict)
    manifests: list[str] = field(default_factory=list)
    entry_points: list[str] = field(default_factory=list)
    test_commands: list[str] = field(default_factory=list)


class RepositoryIndex:
    def __init__(self, root: str | Path, cache_path: str | Path | None = None) -> None:
        self.root = Path(root).resolve()
        self.cache_path = Path(cache_path) if cache_path else self.root / ".tamfis-code-index.json"

    @staticmethod
    def _hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def build(self, *, max_files: int = 10000) -> RepositorySnapshot:
        files: dict[str, dict[str, Any]] = {}
        manifests: list[str] = []
        ignored = {".git", "__pycache__", ".pytest_cache", "node_modules", "dist", "build"}
        for path in sorted(self.root.rglob("*")):
            if any(part in ignored for part in path.parts):
                continue
            if not path.is_file():
                continue
            rel = str(path.relative_to(self.root))
            stat = path.stat()
            files[rel] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            if path.name in {"pyproject.toml", "package.json", "Cargo.toml", "go.mod", "composer.json", "Makefile"}:
                manifests.append(rel)
            if len(files) >= max_files:
                break
        digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        snapshot = RepositorySnapshot(str(self.root), digest, files, manifests)
        self.save(snapshot)
        return snapshot

    def load(self) -> RepositorySnapshot | None:
        if not self.cache_path.is_file():
            return None
        try:
            return RepositorySnapshot(**json.loads(self.cache_path.read_text(encoding="utf-8")))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def save(self, snapshot: RepositorySnapshot) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=".tamfis-index-", suffix=".json", dir=self.cache_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(asdict(snapshot), handle, indent=2, sort_keys=True)
                handle.flush(); os.fsync(handle.fileno())
            os.replace(temp_name, self.cache_path)
        finally:
            try: os.unlink(temp_name)
            except FileNotFoundError: pass

    def unchanged(self) -> bool:
        previous = self.load()
        if previous is None:
            return False
        current = self.build()
        return current.fingerprint == previous.fingerprint
