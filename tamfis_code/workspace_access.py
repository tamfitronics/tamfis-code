"""Host-side workspace ACL provisioning for Tamfis execution accounts."""
from __future__ import annotations

import asyncio
import contextlib
import os
from pathlib import Path


ACL_HELPER = Path("/usr/local/sbin/tamfis-workspace-acl")
MANAGED_ROOT = Path("/home")


def _required_access(*, read_only: bool) -> int:
    access = os.R_OK | os.X_OK
    if not read_only:
        access |= os.W_OK
    return access


async def ensure_workspace_access(path: str, *, read_only: bool) -> tuple[bool, str]:
    """Provision a denied managed workspace before any model is called.

    Normal inherited ACLs make this a no-op. The privileged helper is needed
    for roots created with an explicit restrictive mode such as ``0700``,
    which masks otherwise inherited named-user ACL entries.
    """
    workspace = Path(path).expanduser().resolve()
    access = _required_access(read_only=read_only)
    if os.access(workspace, access):
        return True, ""
    if workspace != MANAGED_ROOT and MANAGED_ROOT not in workspace.parents:
        return False, f"Workspace access requires host approval outside {MANAGED_ROOT}: {workspace}"
    if not ACL_HELPER.is_file():
        return False, f"Managed workspace ACL helper is unavailable: {ACL_HELPER}"

    command = (
        [str(ACL_HELPER), str(workspace)]
        if os.geteuid() == 0
        else ["sudo", "-n", str(ACL_HELPER), str(workspace)]
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            # See mcp.py's _execute_command for the full rationale -- same
            # fix, same reason: never give a spawned child a real TTY fd.
            stdin=asyncio.subprocess.DEVNULL,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
    except asyncio.TimeoutError:
        # FIX: this previously left the process running -- sudo -n fails
        # fast so this path is rare, but a genuine hang here leaked a
        # process forever with nothing ever killing it.
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        return False, f"Workspace ACL provisioning for {workspace} timed out after 120s"
    except OSError as exc:
        return False, f"Workspace ACL provisioning failed for {workspace}: {exc}"
    if proc.returncode != 0:
        detail = stderr.decode("utf-8", errors="ignore").strip()
        if not detail:
            detail = stdout.decode("utf-8", errors="ignore").strip()
        return False, f"Workspace ACL provisioning failed for {workspace}: {detail or 'unknown error'}"
    if not os.access(workspace, access):
        return False, f"Workspace remains inaccessible after ACL provisioning: {workspace}"
    return True, ""
