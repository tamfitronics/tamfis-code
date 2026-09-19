# tamfis_code/swarm.py
"""Safe parallel "swarm" execution on top of AgentManager.execute_tasks --
does not reimplement the fan-out primitive (Semaphore + gather already
lives there), just makes it safe and reachable the way Claude Code/Codex/
Kimi expose parallel sub-agent work.

Three concrete gaps closed here:
  1. Terminal-rendering collision: N concurrent DelegatedCodingAgents each
     building their own StreamRenderer on the SAME shared Console would try
     to run N concurrent rich.live.Live regions -- BufferedSubagentRenderer
     never constructs a Live at all, by construction, not by locking.
  2. Non-silent mutation gate: swarm sub-tasks run non-interactively, and
     the default "ask" policy silently denies every mutating tool call for
     a non-interactive caller (see runner.py's _decision_for_policy) --
     mutation_policy_allows_swarm() lets a caller refuse up front with an
     actionable message instead of that silent per-call deny.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional

# Mirrors the exact policy groupings runner.py's _decision_for_policy
# already hard-codes for "would a non-interactive caller's mutating tool
# call ever be approved" -- kept here as an explicit, separately-testable
# list (see tests/test_swarm.py's drift-detection test) rather than
# re-deriving it, so any future change to that grouping is forced to
# consciously update this one too instead of silently diverging.
_AUTO_APPROVING_POLICIES = frozenset({
    "auto", "full-auto", "safe", "workspace", "accept-edits",
})


def mutation_policy_allows_swarm(approval_policy: str) -> bool:
    """True if a swarm sub-task running under this policy could actually
    get a mutating tool call approved (non-interactively). False for "ask"
    (and "never", "read-only", or anything unrecognized) -- those would
    silently deny every mutation, which is exactly the silent-failure mode
    this function exists to make loud instead."""
    return (approval_policy or "").strip().lower() in _AUTO_APPROVING_POLICIES


class BufferedSubagentRenderer:
    """A StreamRenderer substitute for one swarm sub-task: never
    constructs a rich.live.Live region (zero collision risk with a shared
    Console, or with any other concurrent sub-task's own instance), and
    never blocks on a terminal prompt (a swarm sub-task always runs
    interactive=False, so it can't legitimately reach one).

    Translates a small subset of events into calls to on_update(task_id,
    update_dict) -- enough for an aggregate status display (see
    run_swarm) to show what each sub-task is currently doing without
    reproducing its full transcript. Everything else is a no-op.

    Deliberately does NOT implement suspend_live/resume_live -- render.py's
    own suspend_live_if_active/resume_live_if_active helpers already
    tolerate a renderer that doesn't (getattr(..., None) + callable()
    check), so omitting them is the simplest correct choice, not a gap.
    """

    def __init__(self, task_id: str, description: str, on_update: Optional[Callable[[str, dict[str, Any]], None]] = None):
        self.task_id = task_id
        self.description = description
        self._on_update = on_update
        self._selected_provider: Optional[str] = None

    def _emit(self, **fields: Any) -> None:
        if self._on_update is not None:
            self._on_update(self.task_id, fields)

    def handle_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("event_type") or event.get("event") or event.get("type")
        payload = event.get("payload") or {}

        if event_type == "model_selected":
            self._selected_provider = payload.get("provider") or event.get("provider")
            self._emit(phase="running", detail=f"model: {payload.get('model') or event.get('model') or '?'}")
        elif event_type == "tool_call_requested":
            name = payload.get("name") or event.get("name") or "tool"
            self._emit(phase="running", detail=f"calling {name}")
        elif event_type == "file_mutation":
            path = payload.get("path") or event.get("path") or ""
            self._emit(phase="running", detail=f"edited {path}" if path else "edited a file")
        elif event_type == "ai_task_failed":
            reason = payload.get("message") or event.get("message") or "failed"
            self._emit(phase="failed", detail=reason)

    def finish(self) -> None:
        """No-op -- nothing was ever opened (no Live, no open assistant
        block) that would need closing."""


class SwarmCoordinator:
    """The root agent's half of the mailbox pattern.

    A swarm worker cannot approve its own destructive call, so it files a
    request and waits; this loop claims those requests (atomically -- see
    Mailbox.claim_next) and answers them the only way the coordinator can:
    through the user's own approval policy, and through the real console prompt
    when the policy says a human must decide. The coordinator runs as a
    background task alongside the workers, so answering one worker never stalls
    the others.
    """

    def __init__(
        self, mailbox, *, approval_policy: str, console, interactive: bool,
        config: Any = None, poll_interval: float = 0.25, stale_after_seconds: float = 900.0,
    ) -> None:
        self.mailbox = mailbox
        self.approval_policy = approval_policy
        self.console = console
        self.interactive = interactive
        self.config = config
        self.poll_interval = poll_interval
        self.stale_after_seconds = stale_after_seconds
        self.coordinator_id = f"coordinator_{id(self):x}"
        self.resolved = 0
        self.denied = 0

    async def _decide(self, request: dict[str, Any]) -> str:
        from .runner import _decision_for_policy, resolve_approval_decision_async

        risk = str(request.get("risk") or "medium")
        command = str(request.get("command") or request.get("tool") or "")
        try:
            decision = _decision_for_policy(self.approval_policy, risk, self.interactive)
        except Exception:
            decision = None
        if decision is None and self.interactive and self.console is not None:
            try:
                decision = await resolve_approval_decision_async(
                    self.console, command, risk, self.approval_policy, True,
                    config=self.config,
                )
            except Exception:
                decision = None
        if decision in ("approve_once", "approve_session"):
            return "approve_once"
        return "deny"

    async def handle(self, request: dict[str, Any]) -> None:
        decision = await self._decide(request)
        try:
            resolved = await asyncio.to_thread(
                self.mailbox.resolve, request["id"], decision,
                coordinator=self.coordinator_id,
                note=f"swarm coordinator ({self.approval_policy} policy)",
            )
        except Exception as exc:
            print(f"[mailbox] coordinator could not resolve {request.get('id')}: {exc}")
            return
        if not resolved:
            return  # already answered (or a worker tried to answer itself)
        self.resolved += 1
        if decision != "approve_once":
            self.denied += 1
        try:
            self.console.print(
                f"[dim]mailbox: {'approved' if decision == 'approve_once' else 'denied'} "
                f"{request.get('tool')} for worker {request.get('worker')}[/dim]"
            )
        except Exception:
            pass

    async def drain(self, stop: asyncio.Event) -> None:
        """Answer requests until `stop` is set. Never raises -- a coordinator
        that dies would leave every worker waiting for its timeout."""
        # Sweep first: a request left behind by a swarm that died must not be
        # answered now (its worker is long gone and it may describe work that
        # is no longer wanted) nor sit in front of this swarm's own requests.
        try:
            await asyncio.to_thread(self.mailbox.expire_stale, max_age_seconds=self.stale_after_seconds)
        except Exception:
            pass
        while not stop.is_set():
            try:
                request = await asyncio.to_thread(self.mailbox.claim_next, self.coordinator_id)
            except Exception:
                request = None
            if request is None:
                try:
                    await asyncio.wait_for(stop.wait(), timeout=self.poll_interval)
                except asyncio.TimeoutError:
                    pass
                continue
            await self.handle(request)


async def run_swarm(
    tasks: list[str], *,
    manager, provider, model, console, workspace_root, session_id: Optional[int] = None,
    approval_policy: str = "ask", mutate: bool = False, max_concurrency: int = 3,
    agent_types: Optional[list[Optional[str]]] = None,
) -> list[dict[str, Any]]:
    """Fan out N independent objectives concurrently via AgentManager.
    execute_tasks -- the entry point both the /swarm REPL command and the
    model-callable swarm tool call into. Defaults to read-only sub-tasks;
    mutate=True is checked against mutation_policy_allows_swarm up front
    and refuses to start (raising ValueError, with an actionable message)
    rather than silently letting every mutating call get denied one-by-one
    deep inside N concurrent sub-turns.

    max_concurrency defaults to 3 here (unlike AgentManager.execute_tasks's
    own bare default of 1) -- safe now that Phase 1's fixes actually landed:
    distinct child sessions per sub-task (workspace.
    resolve_swarm_subtask_workspace), no concurrent rich.live.Live regions
    (BufferedSubagentRenderer), and mutation only ever allowed under a
    policy that can actually approve it non-interactively. execute_tasks's
    own default and agent-cmd delegate's CLI default are deliberately left
    at 1 -- the higher default applies only to this hardened swarm surface.
    """
    from .agents import AgentManager

    if mutate and not mutation_policy_allows_swarm(approval_policy):
        raise ValueError(
            f"Swarm sub-tasks run non-interactively and cannot prompt for approval; policy "
            f"'{approval_policy}' would silently deny every mutating action. Switch to an "
            "auto-approving policy first (/mode auto, /mode accept-edits, --approval auto), "
            "or omit --mutate to run this swarm read-only."
        )

    status: dict[str, dict[str, Any]] = {
        f"pending_{i}": {"description": desc, "phase": "queued", "detail": ""}
        for i, desc in enumerate(tasks)
    }
    # Reassigned to the real task_id the moment execute_tasks mints one (see
    # renderer_factory below) -- the placeholder keys above only exist so
    # the status display has something to show before any sub-task has
    # actually started.
    order = list(status.keys())

    is_tty = bool(getattr(console, "is_terminal", False))
    live = None
    if is_tty:
        from rich.live import Live
        live = Live(_build_swarm_group(status, order), console=console, refresh_per_second=8, transient=True)
        live.start()

    def on_update(task_id: str, fields: dict[str, Any]) -> None:
        if task_id not in status:
            status[task_id] = {"description": "", "phase": "queued", "detail": ""}
            order.append(task_id)
        status[task_id].update(fields)
        if live is not None:
            live.update(_build_swarm_group(status, order))

    placeholder_iter = iter(order)

    def renderer_factory(task_id: str, description: str) -> BufferedSubagentRenderer:
        placeholder_key = next(placeholder_iter, None)
        if placeholder_key is not None:
            status[task_id] = status.pop(placeholder_key)
            order[order.index(placeholder_key)] = task_id
        else:
            status[task_id] = {"description": description, "phase": "queued", "detail": ""}
            order.append(task_id)
        status[task_id]["description"] = description
        status[task_id]["phase"] = "running"
        if live is not None:
            live.update(_build_swarm_group(status, order))
        return BufferedSubagentRenderer(task_id, description, on_update=on_update)

    # Mailbox + coordinator (only for a mutating swarm: a read-only swarm never
    # issues an approvable call, so giving it a mailbox would be pure overhead).
    mailbox = None
    coordinator: Optional[SwarmCoordinator] = None
    stop_event = asyncio.Event()
    drain_task: Optional[asyncio.Task[Any]] = None
    if mutate:
        from .mailbox import mailbox_for_swarm, worker_context

        mailbox = mailbox_for_swarm(session_id)
        coordinator = SwarmCoordinator(
            mailbox, approval_policy=approval_policy, console=console,
            interactive=bool(getattr(console, "is_terminal", False)),
        )
        drain_task = asyncio.ensure_future(coordinator.drain(stop_event))

    def _worker_context_factory(worker_id: str, worker_session_id: int, description: str):
        from .mailbox import WorkerContext

        return worker_context(WorkerContext(
            worker_id=worker_id, mailbox=mailbox, session_id=worker_session_id,
        ))

    try:
        agent_manager = AgentManager()
        results = await agent_manager.execute_tasks(
            tasks, manager=manager, provider=provider, model=model, console=console,
            workspace_root=workspace_root, approval_policy=approval_policy,
            mode="agent" if mutate else "chat",
            max_concurrency=max_concurrency, parent_session_id=session_id,
            renderer_factory=renderer_factory,
            agent_types=agent_types,
            worker_context_factory=_worker_context_factory if mailbox is not None else None,
        )
    finally:
        stop_event.set()
        if drain_task is not None:
            try:
                await asyncio.wait_for(drain_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                drain_task.cancel()
            except Exception:
                pass
        if live is not None:
            live.stop()

    return results


def _build_swarm_group(status: dict[str, dict[str, Any]], order: list[str]) -> Any:
    from rich.console import Group
    from rich.text import Text

    lines = []
    for task_id in order:
        entry = status.get(task_id) or {}
        phase = entry.get("phase", "queued")
        marker = {"completed": "✓", "failed": "✗", "running": "◉"}.get(phase, "○")
        desc = entry.get("description", "")
        detail = entry.get("detail", "")
        suffix = f" — {detail}" if detail else ""
        lines.append(Text.from_markup(f"  {marker} {desc}{suffix}"))
    return Group(*lines) if lines else Text("Starting swarm...")
