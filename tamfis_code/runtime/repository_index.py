"""Persistent repository intelligence for avoiding repeated reconnaissance.

This index is intentionally metadata-first: it gives orchestration a cheap
repository fingerprint and manifest inventory without embedding or reading
source bodies on every turn. File content hashes are populated only for new
or changed files, which makes them useful as stable invalidation/deduplication
keys without turning a health check into a full reread of the repository.
"""
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
    """Incremental, metadata-first repository snapshot.

    ``build`` remains compatible with the original API, but unchanged files
    reuse their prior metadata/content hash and are not opened. ``unchanged``
    performs a pure filesystem metadata scan and never rewrites the cache;
    callers can safely use it on hot paths and in repeated turns.
    """

    _IGNORED_PARTS = frozenset({
        ".git", "__pycache__", ".pytest_cache", "node_modules", "dist", "build",
    })
    _MANIFEST_NAMES = frozenset({
        "pyproject.toml", "package.json", "Cargo.toml", "go.mod", "composer.json", "Makefile",
    })

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

    @classmethod
    def _metadata_fingerprint(cls, files: dict[str, dict[str, Any]]) -> str:
        metadata = {
            path: {key: value for key, value in entry.items() if key != "content_hash"}
            for path, entry in files.items()
        }
        return hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()

    def _scan_metadata(self, *, max_files: int = 10000) -> tuple[dict[str, dict[str, Any]], list[str]]:
        files: dict[str, dict[str, Any]] = {}
        manifests: list[str] = []
        try:
            paths = sorted(self.root.rglob("*"))
        except OSError:
            paths = []
        cache_resolved = self.cache_path.resolve()
        for path in paths:
            try:
                if path.resolve() == cache_resolved:
                    continue
            except OSError:
                continue
            if any(part in self._IGNORED_PARTS for part in path.parts):
                continue
            try:
                if not path.is_file():
                    continue
                stat = path.stat()
            except OSError:
                continue
            rel = str(path.relative_to(self.root))
            files[rel] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            if path.name in self._MANIFEST_NAMES:
                manifests.append(rel)
            if len(files) >= max_files:
                break
        return files, manifests

    def build(self, *, max_files: int = 10000) -> RepositorySnapshot:
        previous = self.load()
        scanned, manifests = self._scan_metadata(max_files=max_files)
        files: dict[str, dict[str, Any]] = {}
        for rel, metadata in scanned.items():
            old = (previous.files.get(rel) if previous else None) or {}
            if (
                old.get("size") == metadata["size"]
                and old.get("mtime_ns") == metadata["mtime_ns"]
                and old.get("content_hash")
            ):
                # The unchanged file is not opened. Keep its stable identity
                # so downstream embedding/search caches can deduplicate it.
                files[rel] = {**metadata, "content_hash": old["content_hash"]}
                continue
            try:
                content_hash = self._hash(self.root / rel)
            except OSError:
                content_hash = ""
            files[rel] = {**metadata, "content_hash": content_hash}

        snapshot = RepositorySnapshot(
            str(self.root), self._metadata_fingerprint(files), files, sorted(manifests),
            entry_points=list(previous.entry_points) if previous else [],
            test_commands=list(previous.test_commands) if previous else [],
        )
        # Keep build's historical persistence contract. Unlike unchanged(),
        # callers invoking build explicitly are asking for a refreshed cache.
        self.save(snapshot)
        return snapshot

    def load(self) -> RepositorySnapshot | None:
        if not self.cache_path.is_file():
            return None
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            # Older snapshots do not have content_hash; they remain readable
            # and receive hashes on the next explicit build.
            return RepositorySnapshot(**payload)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def save(self, snapshot: RepositorySnapshot) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=".tamfis-index-", suffix=".json", dir=self.cache_path.parent,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(asdict(snapshot), handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.cache_path)
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass

    def changed_files(self, snapshot: RepositorySnapshot | None = None) -> list[str]:
        """Return paths whose metadata differs from the cached snapshot."""
        previous = snapshot or self.load()
        if previous is None:
            current, _ = self._scan_metadata()
            return sorted(current)
        current, _ = self._scan_metadata()
        return sorted(
            path for path, metadata in current.items()
            if path not in previous.files or any(
                metadata.get(key) != previous.files[path].get(key)
                for key in ("size", "mtime_ns")
            )
        )

    def removed_files(self, snapshot: RepositorySnapshot | None = None) -> list[str]:
        """Return cached paths that no longer exist in the workspace."""
        previous = snapshot or self.load()
        if previous is None:
            return []
        current, _ = self._scan_metadata()
        return sorted(set(previous.files) - set(current))

    def unchanged(self) -> bool:
        """Check metadata only; do not rewrite or refresh the cache."""
        previous = self.load()
        if previous is None:
            return False
        current, manifests = self._scan_metadata()
        return (
            self._metadata_fingerprint(current) == previous.fingerprint
            and sorted(manifests) == sorted(previous.manifests)
        )
