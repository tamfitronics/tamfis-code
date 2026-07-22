"""Task discovery for `/resume` and `/retry` -- there is no dedicated "list
tasks for a session" endpoint (see docs/REMOTE_AGENT_MASTER_SPEC.md Phase 21;
not adding one this pass to keep the shared API surface unchanged), so the
most-recent-task lookup is derived from GET /sessions/{id}/thread's message
list the same way the web SessionView.tsx reconstructs AITurns on refresh --
distinct task_ids in reverse-chronological order, then GET /tasks/{id} for
status until a match is found.
"""

from __future__ import annotations

from typing import Optional

from .api_client import RemoteAPIClient

# Bounded lookback rather than scanning a session's entire history -- a
# session that has run hundreds of tasks should not make this many
# GET /tasks/{id} round trips just to find one recent failure.
DEFAULT_LOOKBACK = 15


async def find_recent_task(
    client: RemoteAPIClient, session_id: int, *, only_status: Optional[set[str]] = None,
    lookback: int = DEFAULT_LOOKBACK,
) -> Optional[dict]:
    thread = await client.get_thread(session_id, after_sequence=0)
    messages = thread.get("messages") or []

    candidate_ids: list[str] = []
    for message in reversed(messages):  # thread messages are oldest-first
        task_id = message.get("task_id")
        if not task_id or task_id in candidate_ids:
            continue
        candidate_ids.append(task_id)
        if len(candidate_ids) >= lookback:
            break

    for task_id in candidate_ids:
        task = await client.get_task(task_id)
        if only_status is None or str(task.get("status")) in only_status:
            return task
    return None
