from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class WorkspaceError(RuntimeError):
    pass


@dataclass(slots=True)
class CommandResult:
    command: str
    cwd: str
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def as_dict(self) -> dict[str, Any]:
        return {
            "command": self.command, "cwd": self.cwd, "returncode": self.returncode,
            "stdout": self.stdout, "stderr": self.stderr,
            "duration_seconds": self.duration_seconds, "timed_out": self.timed_out,
            "ok": self.ok,
        }


class BaseWorkspace:
    async def execute(self, command: str, *, cwd: str | None = None, timeout: float = 120.0) -> CommandResult:
        raise NotImplementedError


class LocalWorkspace(BaseWorkspace):
    """No-Docker workspace with strict path confinement and reversible snapshots."""

    def __init__(self, root: str | Path, *, state_dir: str | Path | None = None):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise WorkspaceError(f"workspace does not exist: {self.root}")
        self.state_dir = Path(state_dir or self.root / ".tamfis" / "runtime").expanduser().resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / "snapshots").mkdir(exist_ok=True)

    def resolve(self, path: str | Path = ".", *, must_exist: bool = False) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        candidate = candidate.resolve(strict=False)
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise WorkspaceError(f"path escapes workspace: {candidate}") from exc
        if must_exist and not candidate.exists():
            raise FileNotFoundError(str(candidate))
        return candidate

    async def execute(self, command: str, *, cwd: str | None = None, timeout: float = 120.0) -> CommandResult:
        working = self.resolve(cwd or ".", must_exist=True)
        if not working.is_dir():
            raise WorkspaceError(f"cwd is not a directory: {working}")
        started = time.monotonic()
        process = await asyncio.create_subprocess_shell(
            command, cwd=str(working), stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, start_new_session=True,
            env={**os.environ, "TAMFIS_WORKSPACE": str(self.root)},
        )
        timed_out = False
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            timed_out = True
            process.kill()
            stdout, stderr = await process.communicate()
        return CommandResult(
            command=command, cwd=str(working), returncode=124 if timed_out else int(process.returncode or 0),
            stdout=stdout.decode(errors="replace"), stderr=stderr.decode(errors="replace"),
            duration_seconds=time.monotonic() - started, timed_out=timed_out,
        )

    def read_text(self, path: str, *, limit: int = 4_000_000) -> str:
        target = self.resolve(path, must_exist=True)
        if target.stat().st_size > limit:
            raise WorkspaceError(f"file exceeds read limit: {target.stat().st_size} bytes")
        return target.read_text(encoding="utf-8", errors="replace")

    def write_text(self, path: str, content: str) -> dict[str, Any]:
        target = self.resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        before = target.read_bytes() if target.exists() else b""
        target.write_text(content, encoding="utf-8")
        return {"path": str(target), "created": not bool(before), "before_sha256": hashlib.sha256(before).hexdigest(), "after_sha256": hashlib.sha256(content.encode()).hexdigest()}

    def replace_text(self, path: str, old: str, new: str) -> dict[str, Any]:
        content = self.read_text(path)
        count = content.count(old)
        if count != 1:
            raise WorkspaceError(f"replacement must match exactly once; matched {count}")
        return self.write_text(path, content.replace(old, new, 1))

    def list_files(self, path: str = ".", *, max_entries: int = 2000) -> list[dict[str, Any]]:
        target = self.resolve(path, must_exist=True)
        entries = []
        for item in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))[:max_entries]:
            entries.append({"name": item.name, "path": str(item.relative_to(self.root)), "type": "directory" if item.is_dir() else "file", "size": item.stat().st_size if item.is_file() else None})
        return entries

    def snapshot(self, *, label: str = "checkpoint") -> dict[str, Any]:
        snapshot_id = f"{int(time.time())}-{uuid.uuid4().hex[:10]}"
        archive = self.state_dir / "snapshots" / f"{snapshot_id}.tar.gz"
        metadata = self.state_dir / "snapshots" / f"{snapshot_id}.json"
        state_dir_rel = None
        try:
            state_dir_rel = self.state_dir.relative_to(self.root)
        except ValueError:
            pass
        with tarfile.open(archive, "w:gz") as tar:
            for item in self.root.rglob("*"):
                if state_dir_rel and (item == self.root / state_dir_rel or self.root / state_dir_rel in item.parents):
                    continue
                if ".git" in item.parts:
                    continue
                tar.add(item, arcname=str(item.relative_to(self.root)), recursive=False)
        data = {"id": snapshot_id, "label": label, "archive": str(archive), "workspace": str(self.root), "created_at": time.time()}
        metadata.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return data

    def restore(self, snapshot_id: str) -> None:
        archive = self.resolve(str(self.state_dir / "snapshots" / f"{snapshot_id}.tar.gz")) if self.state_dir.is_relative_to(self.root) else self.state_dir / "snapshots" / f"{snapshot_id}.tar.gz"
        if not archive.exists():
            raise FileNotFoundError(str(archive))
        with tempfile.TemporaryDirectory(prefix="tamfis-restore-") as temp:
            temp_path = Path(temp)
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(temp_path, filter="data")
            for item in self.root.iterdir():
                if item == self.state_dir or item.name == ".git":
                    continue
                if item.is_dir(): shutil.rmtree(item)
                else: item.unlink()
            for item in temp_path.iterdir():
                shutil.move(str(item), str(self.root / item.name))


class SSHWorkspace(BaseWorkspace):
    def __init__(self, host: str, root: str, *, user: str | None = None, port: int = 22):
        self.host, self.root, self.user, self.port = host, root, user, port

    async def execute(self, command: str, *, cwd: str | None = None, timeout: float = 120.0) -> CommandResult:
        remote = f"{self.user}@{self.host}" if self.user else self.host
        workdir = cwd or self.root
        safe = workdir.replace("'", "'\''")
        process = await asyncio.create_subprocess_exec(
            "ssh", "-p", str(self.port), remote, f"cd '{safe}' && {command}",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        started = time.monotonic(); timed_out = False
        try: stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            timed_out = True; process.kill(); stdout, stderr = await process.communicate()
        return CommandResult(command, workdir, 124 if timed_out else int(process.returncode or 0), stdout.decode(errors="replace"), stderr.decode(errors="replace"), time.monotonic()-started, timed_out)


class RemoteWorkspace(BaseWorkspace):
    def __init__(self, base_url: str, workspace_id: str, *, token: str | None = None):
        self.base_url = base_url.rstrip("/"); self.workspace_id = workspace_id; self.token = token

    async def execute(self, command: str, *, cwd: str | None = None, timeout: float = 120.0) -> CommandResult:
        import httpx
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
            response = await client.post(f"{self.base_url}/v1/workspaces/{self.workspace_id}/execute", json={"command": command, "cwd": cwd, "timeout": timeout})
            response.raise_for_status(); data = response.json()
        return CommandResult(**data)
