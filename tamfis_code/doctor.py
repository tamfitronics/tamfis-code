"""`tamfis-code doctor` -- connectivity/auth/workspace checks.

Reports PASS/WARNING/FAIL per Phase 21. Deliberately checks the things that
were the actual break points found during this project's Remote-workspace
audit (see docs/REMOTE_AGENT_MASTER_SPEC.md and the linked memory notes) --
API reachability, auth, and workspace-scope enforcement really working, not
just "is the process up."
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import httpx
from rich.console import Console

from .api_client import AuthRequiredError, RemoteAPIClient, RemoteAPIError
from .config import Config, load_credentials


@dataclass
class CheckResult:
    name: str
    status: str  # "PASS" | "WARNING" | "FAIL"
    detail: str = ""


_STATUS_STYLE = {"PASS": "green", "WARNING": "yellow", "FAIL": "red"}

_REUSABLE_SESSION_STATUSES = {"idle", "active"}


def check_event_sequence_integrity(events: list[dict[str, Any]]) -> CheckResult:
    """Pure check over a window of a session's RemoteEvent replay (as
    returned by GET .../thread): sequence numbers must be unique and, within
    the returned window, contiguous. A duplicate is a real replay-safety
    bug (the same event could be rendered/acted on twice by a client that
    resumes from `last_event_id`); a gap is either dropped events or just
    this window being truncated by the endpoint's own `limit` -- reported
    as a WARNING, not a FAIL, since it can't be told apart from here.
    """
    if not events:
        return CheckResult("Event replay integrity", "WARNING", "no events yet for this session")
    sequences = [e.get("sequence") for e in events if isinstance(e.get("sequence"), int)]
    if len(sequences) != len(events):
        return CheckResult("Event replay integrity", "FAIL", "one or more events are missing a sequence number")
    sequences.sort()
    duplicates = {s for s in sequences if sequences.count(s) > 1}
    if duplicates:
        return CheckResult("Event replay integrity", "FAIL", f"duplicate sequence number(s): {sorted(duplicates)[:5]}")
    gaps = [b for a, b in zip(sequences, sequences[1:]) if b != a + 1]
    if gaps:
        return CheckResult(
            "Event replay integrity", "WARNING",
            f"{len(gaps)} gap(s) in {len(sequences)} events checked (sequence {sequences[0]}-{sequences[-1]}) "
            "-- may just be this check's window, or a dropped event",
        )
    return CheckResult("Event replay integrity", "PASS", f"{len(sequences)} events, sequence {sequences[0]}-{sequences[-1]}, no gaps or duplicates")


async def _diagnose_session(client: RemoteAPIClient, session_id: int, workspace_root: Optional[Path]) -> list[CheckResult]:
    """Session/workspace-snapshot/event-replay health for the *active*
    session -- distinct from the generic connectivity checks above, this is
    the "am I actually in the state I think I'm in" self-diagnosis."""
    results: list[CheckResult] = []
    try:
        session = await client.get_session(session_id)
    except (AuthRequiredError, RemoteAPIError) as e:
        results.append(CheckResult("Active session", "FAIL", f"session {session_id}: {e}"))
        return results

    status = str(session.get("status") or "")
    if status in _REUSABLE_SESSION_STATUSES:
        results.append(CheckResult("Active session", "PASS", f"session {session_id} status={status}"))
    else:
        results.append(CheckResult("Active session", "WARNING", f"session {session_id} status={status} (not idle/active)"))

    server_wd = session.get("working_directory")
    if workspace_root is not None and server_wd:
        if str(workspace_root.resolve()) == server_wd:
            results.append(CheckResult("Session cwd matches local cwd", "PASS", server_wd))
        else:
            results.append(CheckResult(
                "Session cwd matches local cwd", "WARNING",
                f"server has {server_wd!r}, local cwd is {str(workspace_root.resolve())!r}",
            ))

    snapshot = session.get("workspace_snapshot")
    if snapshot is None:
        results.append(CheckResult("Workspace snapshot", "WARNING", "not scanned yet -- will populate on the first AI task"))
    else:
        detail = (
            f"v{snapshot.get('file_index_version')}, "
            f"repo={snapshot.get('repository_type')}, "
            f"branch={snapshot.get('git_branch') or '-'}, "
            f"last scan={snapshot.get('last_scan_at') or 'unknown'} "
            f"(reason: {snapshot.get('scan_reason') or 'unknown'})"
        )
        results.append(CheckResult("Workspace snapshot", "PASS", detail))

    try:
        thread = await client.get_thread(session_id, after_sequence=0)
        results.append(check_event_sequence_integrity(thread.get("events") or []))
    except (AuthRequiredError, RemoteAPIError) as e:
        results.append(CheckResult("Event replay integrity", "FAIL", str(e)))

    return results


async def run_doctor(
    config: Config,
    console: Console,
    workspace_root: Optional[Path] = None,
    *,
    session_id: Optional[int] = None,
) -> bool:
    results: list[CheckResult] = []

    results.append(CheckResult("Config", "PASS", f"api_base={config.api_base}"))

    creds = load_credentials()
    if creds is None:
        results.append(CheckResult("Authentication", "FAIL", "No credentials -- run `tamfis-code login`"))
    else:
        results.append(CheckResult("Authentication", "PASS", f"credentials present for {creds.email or creds.user_id or 'unknown user'}"))

    client = RemoteAPIClient(config, creds)
    try:
        try:
            servers = await client.list_servers()
            results.append(CheckResult("Remote API (Tier III, port 9500)", "PASS", f"{len(servers)} registered server(s)"))
        except AuthRequiredError as e:
            results.append(CheckResult("Remote API (Tier III, port 9500)", "FAIL", str(e)))
            servers = None
        except (RemoteAPIError, httpx.HTTPError) as e:
            results.append(CheckResult("Remote API (Tier III, port 9500)", "FAIL", str(e)))
            servers = None

        if servers is not None:
            local_server = next((s for s in servers if s.get("transport_type") == "local"), None)
            if local_server is not None:
                results.append(CheckResult("Local transport server", "PASS", f"server_id={local_server['id']}"))
            else:
                results.append(CheckResult("Local transport server", "WARNING", "none registered yet -- `tamfis-code init` will create one"))

        if workspace_root is not None:
            wr = str(workspace_root.resolve())
            if workspace_root.is_dir():
                results.append(CheckResult("Workspace directory", "PASS", wr))
            else:
                results.append(CheckResult("Workspace directory", "FAIL", f"{wr} is not a directory"))

        # Tier IV reachability is inferred, not probed directly -- there is
        # no public health endpoint on port 9555 to hit from here without
        # a session already existing; a real ai-task submission is what
        # actually proves the whole chain, which `doctor` deliberately does
        # not do (it would create session/task rows as a side effect of a
        # health check).
        results.append(CheckResult("Tier IV (agent runtime)", "WARNING", "not directly probed -- verified indirectly via a real `tamfis-code ask`"))

        if session_id is not None:
            results.extend(await _diagnose_session(client, session_id, workspace_root))

    finally:
        await client.aclose()

    for result in results:
        style = _STATUS_STYLE[result.status]
        console.print(f"[{style}]{result.status:8}[/{style}] {result.name}  [dim]{result.detail}[/dim]")

    return all(r.status != "FAIL" for r in results)
