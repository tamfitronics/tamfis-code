"""Detached background execution for standalone (non-Remote) agent tasks.

Before this, `--bg` only worked with `--remote` because the old Remote
Workspace backend kept a submitted task running server-side, independent of
this CLI process. Once execute_remote (runtime/unified.py) was changed to
always run turns through the local engine in-process, that safety net
disappeared for standalone tasks: closing the terminal killed the task with
it, and there was no way to "run until completed" without staying attached.

This spawns a real detached OS process (`start_new_session=True`, i.e.
setsid -- the same mechanism `nohup`/`disown` rely on) that re-invokes this
same CLI to run the task headlessly, with output redirected to a log file.
Once spawned, the parent process returns immediately; the child is no
longer part of the parent's session, so a SIGHUP from the terminal closing
never reaches it. Job bookkeeping is a small JSON file per job (mirrors
state.py's atomic-write pattern), not the full session state -- background
jobs are a thin, independent concept layered on top of an ordinary
standalone task.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

from .config import CONFIG_DIR

JOBS_DIR = CONFIG_DIR / "background"


@dataclass
class BackgroundJob:
    id: str
    pid: int
    session_id: int
    workspace_root: str
    mode: str
    objective_preview: str
    log_path: str
    prompt_path: str
    started_at: float
    status: str = "running"  # running | completed | failed | stopped
    finished_at: Optional[float] = None
    exit_code: Optional[int] = None


def _job_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def _write_job(job: BackgroundJob) -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    if os.stat(JOBS_DIR).st_mode & 0o777 != 0o700:
        os.chmod(JOBS_DIR, 0o700)
    # Atomic replace, same reasoning as state.py's _save_raw: a killed
    # process must never leave a half-written, unparseable job file behind.
    fd, temp_name = tempfile.mkstemp(prefix=".job-", suffix=".json", dir=JOBS_DIR)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(asdict(job), handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, _job_path(job.id))
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Owned by another user but still alive -- can't signal it, but it exists.
        return True
    return True


def read_job(job_id: str) -> Optional[dict[str, Any]]:
    """Read one job record, resolving a pid-alive/status mismatch: a process
    that died without calling update_job_status (killed, crashed, machine
    rebooted) would otherwise report "running" forever."""
    path = _job_path(job_id)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if record.get("status") == "running" and not is_pid_alive(int(record["pid"])):
        record["status"] = "stopped"
        record["finished_at"] = record.get("finished_at") or time.time()
        _write_job(BackgroundJob(**record))
    return record


def list_jobs() -> list[dict[str, Any]]:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for path in sorted(JOBS_DIR.glob("*.json")):
        job = read_job(path.stem)
        if job is not None:
            jobs.append(job)
    return jobs


def update_job_status(job_id: str, status: str, *, exit_code: Optional[int] = None) -> None:
    """Called by the background child itself right before it exits."""
    record = read_job(job_id)
    if record is None:
        return
    record["status"] = status
    record["finished_at"] = time.time()
    record["exit_code"] = exit_code
    _write_job(BackgroundJob(**record))


def stop_job(job_id: str) -> bool:
    import signal

    record = read_job(job_id)
    if record is None or record.get("status") != "running":
        return False
    pid = int(record["pid"])
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return False
    record["status"] = "stopped"
    record["finished_at"] = time.time()
    _write_job(BackgroundJob(**record))
    return True


def spawn_background_task(
    *,
    session_id: int,
    workspace_root: Path,
    mode: str,
    objective: str,
    model: str,
    provider: Optional[str],
    approval_policy: str,
    attachment_paths: tuple[str, ...] = (),
) -> BackgroundJob:
    """Launch objective as a detached standalone `ask` invocation and return
    immediately. The child is its own session leader (start_new_session),
    so it survives this process -- and the terminal that started it --
    exiting."""
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    job_id = f"bg-{uuid.uuid4().hex[:8]}"
    prompt_path = JOBS_DIR / f"{job_id}.prompt"
    prompt_path.write_text(objective, encoding="utf-8")
    log_path = JOBS_DIR / f"{job_id}.log"

    argv = [
        sys.executable, "-m", "tamfis_code",
        "--cwd", str(workspace_root),
        "--approval", approval_policy,
        "ask", "--prompt-file", str(prompt_path),
        "--mode", mode, "--model", model,
        "--_bg-job-id", job_id,
    ]
    if provider:
        argv += ["--provider", provider]
    for attachment in attachment_paths:
        argv += ["--attach", attachment]

    log_fh = open(log_path, "ab", buffering=0)
    try:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=log_fh,
            cwd=str(workspace_root),
            start_new_session=True,
            close_fds=True,
        )
    finally:
        log_fh.close()

    job = BackgroundJob(
        id=job_id, pid=proc.pid, session_id=session_id, workspace_root=str(workspace_root),
        mode=mode, objective_preview=objective[:200], log_path=str(log_path),
        prompt_path=str(prompt_path), started_at=time.time(),
    )
    _write_job(job)
    return job
