"""Task/command execution + streaming loop, shared by interactive and
non-interactive commands. Supports multiple providers: Ollama, HF, NVIDIA,
OpenRouter, and auto-fallback.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import traceback
from dataclasses import dataclass, field
from typing import Any, Optional, AsyncGenerator

from rich.console import Console
from rich.panel import Panel

from .api_client import AuthRequiredError, RemoteAPIClient, RemoteAPIError
from .render import StreamRenderer
from . import state as local_state
from .providers import ProviderManager, ProviderType

COMMAND_POLL_INTERVAL_SECONDS = 0.2
TERMINAL_COMMAND_STATUSES = {"completed", "failed", "denied", "cancelled"}
APPROVAL_PREVIEW_CHARS = 8_000
ACTIVE_TASK_STATUSES = {"running", "pending", "queued", "pending_approval"}
try:
    MAX_TASK_STREAM_RECONNECTS = min(
        30, max(3, int(os.getenv("TAMFIS_CODE_STREAM_RECONNECTS", "12")))
    )
except (TypeError, ValueError):
    MAX_TASK_STREAM_RECONNECTS = 12

# Allowed providers - expanded from just hf/openrouter
ALLOWED_PROVIDERS = ["hf", "huggingface", "or", "openrouter", "ollama", "nvidia", "nvidia_nim", "gemini", "apiframe", "auto", None]
PROVIDER_ALIASES = {"hf": "huggingface", "nvidia": "nvidia_nim", "or": "openrouter"}

# Provider name mapping
PROVIDER_NAME_MAP = {
    "hf": "Hugging Face",
    "huggingface": "Hugging Face",
    "openrouter": "OpenRouter",
    "ollama": "Ollama (Local)",
    "nvidia": "NVIDIA NIM",
    "nvidia_nim": "NVIDIA NIM",
    "gemini": "Google Gemini",
    "apiframe": "APIFRAME",
    "auto": "Auto (server-selected)",
}


def normalize_provider(provider: Optional[str]) -> str:
    if provider not in ALLOWED_PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}")
    return PROVIDER_ALIASES.get(provider, provider) or "auto"


@dataclass
class TaskOutcome:
    status: str
    summary: Optional[str] = None
    error: Optional[str] = None
    plan_id: Optional[str] = None
    plan_items: list[dict[str, Any]] = field(default_factory=list)


def resolve_approval_decision(
    console: Console, command_text: str, risk: str, approval_policy: str, interactive: bool,
    *, display_preview: bool = True,
) -> str:
    risk = (risk or "medium").lower()

    if approval_policy in {"auto", "full-auto"}:
        return "approve_once"
    if approval_policy in {"safe", "workspace", "accept-edits"}:
        return "approve_once" if risk != "dangerous" else ("deny" if not interactive else _prompt(console, command_text, risk, display_preview))
    if approval_policy in {"read-only", "plan-only", "suggest", "never"}:
        return "deny"
    if not interactive:
        return "deny"
    return _prompt(console, command_text, risk, display_preview)


def approval_command_preview(command_text: str, limit: int = APPROVAL_PREVIEW_CHARS) -> str:
    if len(command_text) <= limit:
        return command_text
    head = int(limit * 0.7)
    tail = limit - head
    omitted = len(command_text) - limit
    return (
        command_text[:head]
        + f"\n\n… {omitted:,} characters omitted from approval preview …\n\n"
        + command_text[-tail:]
    )


def _install_sigint_watcher() -> tuple[asyncio.Event, "Any"]:
    loop = asyncio.get_running_loop()
    event = asyncio.Event()
    installed = False
    try:
        loop.add_signal_handler(signal.SIGINT, event.set)
        installed = True
    except (NotImplementedError, RuntimeError):
        pass

    def uninstall() -> None:
        if installed:
            try:
                loop.remove_signal_handler(signal.SIGINT)
            except (NotImplementedError, RuntimeError):
                pass

    return event, uninstall


def _prompt(console: Console, command_text: str, risk: str, display_preview: bool = True) -> str:
    if display_preview:
        console.print(Panel(approval_command_preview(command_text), title=f"Approval required — risk: {risk}", border_style="magenta", expand=False))
    while True:
        answer = console.input("[bold]Approve? (y)es / (n)o / (a)lways this session: [/bold]").strip().lower()
        if answer in ("y", "yes"):
            return "approve_once"
        if answer in ("a", "always"):
            return "approve_session"
        if answer in ("n", "no", ""):
            return "deny"


async def submit_ai_task_background(
    client: RemoteAPIClient, *, session_id: int, objective: str, mode: str,
    model: str = "auto", provider: Optional[str] = None,
    attachments: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    provider = normalize_provider(provider)
    task = await client.run_ai_task(
        session_id, objective, mode, model=model, provider=provider, attachments=attachments,
    )
    task_id = str(task["task_id"])
    action = local_state.start_action(
        session_id, action_type="ai_task", purpose=objective,
        risk="read_only" if mode in {"chat", "audit", "plan"} else "workspace_write",
        detail=f"mode={mode}; background=true",
    )
    # finish_action() unconditionally resets execution_status to "idle"/
    # "running" as a side effect of closing out its own local bookkeeping --
    # it must run before the save_session_state() call below sets the real
    # status, not after, or "backgrounded" gets clobbered back to "idle"
    # immediately even though a real task is now running server-side.
    local_state.finish_action(session_id, action.id, status="completed", summary=f"submitted task {task_id}")
    local_state.save_session_state(
        session_id, last_task_id=task_id, active_task={"id": task_id, "objective": objective, "mode": mode},
        current_phase="queued", execution_status="backgrounded",
    )
    local_state.checkpoint(session_id, reason="background_task_submitted", summary=objective)
    return task


async def _print_command_budget_if_notable(client: RemoteAPIClient, console: Console, task_id: str) -> None:
    """Surface command-budget usage once a task finishes, but only when it's
    actually informative -- most short tasks use a handful of commands out
    of a budget in the hundreds, and printing that every time would just be
    noise. Silently does nothing if the task/budget fields aren't available
    (older server, or the request itself fails) -- this is purely
    informational and must never affect the task's real outcome."""
    try:
        task = await client.get_task(task_id)
    except (AuthRequiredError, RemoteAPIError):
        return
    budget = task.get("command_budget")
    count = task.get("command_count")
    if not budget or count is None:
        return
    if count >= budget * 0.8:
        style = "red" if count >= budget else "yellow"
        console.print(f"[{style}]Commands used: {count}/{budget} for this task.[/{style}]")


async def run_ai_task_and_stream(
    client: RemoteAPIClient,
    renderer: StreamRenderer,
    console: Console,
    *,
    session_id: int,
    objective: str,
    mode: str,
    approval_policy: str,
    interactive: bool,
    model: str = "auto",
    provider: Optional[str] = None,
    attachments: Optional[list[dict[str, Any]]] = None,
) -> TaskOutcome:
    # Normalize aliases at the boundary. Explicit routes are never silently
    # replaced with auto or a different provider.
    provider = normalize_provider(provider)

    provider_name = PROVIDER_NAME_MAP.get(provider or "auto", provider or "auto")
    console.print(f"[dim]Using provider: {provider_name}[/dim]")

    # Preserve auto selection for Tier IV. The CLI must not silently pin
    # automatic requests to Ollama or OpenRouter; the orchestration layer
    # has provider health, capability and workspace-tier information.
    if provider == "auto":
        provider = "auto"

    task = await client.run_ai_task(
        session_id, objective, mode, model=model, provider=provider, attachments=attachments,
    )
    task_id = str(task["task_id"])
    action = local_state.start_action(
        session_id, action_type="ai_task", purpose=objective,
        risk="read_only" if mode in {"chat", "audit", "plan"} else "workspace_write", detail=f"mode={mode}",
    )
    local_state.save_session_state(
        session_id, last_task_id=task_id, active_task={"id": task_id, "objective": objective, "mode": mode},
        current_phase="understand", execution_status="running",
    )
    try:
        # Resume from the durable session cursor. Replaying a long-lived
        # workspace from event 1 on every new task delays visible output and
        # can make a healthy task look hung while hundreds of irrelevant old
        # events are filtered client-side.
        stream_cursor = local_state.get_session_state(session_id).last_event_id
        outcome = await _stream_task(
            client, renderer, console, session_id=session_id, task_id=task_id,
            approval_policy=approval_policy, interactive=interactive,
            from_event_id=stream_cursor,
        )
    except BaseException:
        local_state.finish_action(session_id, action.id, status="failed", summary="stream failed")
        raise
    # A live-branched instruction may have continued this task under a new
    # task_id (see _stream_task's task_continued handling) -- last_task_id
    # tracks whichever one actually finished.
    final_task_id = local_state.get_session_state(session_id).last_task_id or task_id
    await _print_command_budget_if_notable(client, console, final_task_id)
    local_state.finish_action(session_id, action.id, status=outcome.status, summary=outcome.summary or outcome.error or "")
    if mode == "plan" and outcome.status == "completed" and outcome.summary:
        saved = local_state.save_plan(
            session_id, objective=objective, content=outcome.summary, source_task_id=task_id,
            steps=outcome.plan_items,
        )
        outcome.plan_id = saved.id
        console.print(
            f"[green]Plan saved[/green] · {saved.id} · run `/execute-plan {saved.id}` "
            f"in the REPL or `tamfis-code execute-plan {saved.id}` from the shell"
        )
    local_state.save_session_state(session_id, active_task=None, current_phase="report", execution_status=outcome.status)
    local_state.checkpoint(session_id, reason=f"task_{outcome.status}", summary=outcome.summary or outcome.error or "")
    return outcome


async def retry_task_and_stream(
    client: RemoteAPIClient,
    renderer: StreamRenderer,
    console: Console,
    *,
    session_id: int,
    task_id: str,
    mode: Optional[str],
    approval_policy: str,
    interactive: bool,
) -> TaskOutcome:
    retried = await client.retry_task(task_id, mode)
    new_task_id = str(retried["task_id"])
    console.print(f"[dim]Retrying as task {new_task_id}[/dim]")
    local_state.save_session_state(session_id, last_task_id=new_task_id)
    stream_cursor = local_state.get_session_state(session_id).last_event_id
    outcome = await _stream_task(
        client, renderer, console,
        session_id=session_id, task_id=new_task_id,
        approval_policy=approval_policy, interactive=interactive,
        from_event_id=stream_cursor,
    )
    await _print_command_budget_if_notable(client, console, new_task_id)
    return outcome


async def attach_and_stream(
    client: RemoteAPIClient,
    renderer: StreamRenderer,
    console: Console,
    *,
    session_id: int,
    task_id: str,
    approval_policy: str,
    interactive: bool,
) -> TaskOutcome:
    from_event_id = local_state.get_session_state(session_id).last_event_id
    return await _stream_task(
        client, renderer, console,
        session_id=session_id, task_id=task_id,
        approval_policy=approval_policy, interactive=interactive,
        from_event_id=from_event_id, on_interrupt="detach",
    )


async def _stream_task(
    client: RemoteAPIClient,
    renderer: StreamRenderer,
    console: Console,
    *,
    session_id: int,
    task_id: str,
    approval_policy: str,
    interactive: bool,
    from_event_id: int = 0,
    on_interrupt: str = "cancel",
    reconnect_attempt: int = 0,
) -> TaskOutcome:
    outcome: Optional[TaskOutcome] = None
    last_assistant_content: Optional[str] = None
    last_plan_items: list[dict[str, Any]] = []
    prompted_command_ids: set[int] = set()
    interrupted, uninstall_sigint = _install_sigint_watcher()
    queued_interrupt: Optional[dict[str, Any]] = None

    async def watch_instruction_queue() -> None:
        nonlocal queued_interrupt
        while not interrupted.is_set():
            state = local_state.get_session_state(session_id)
            queued_interrupt = next((
                item for item in state.queued_user_instructions
                if item.get("status") == "queued"
                and item.get("classification") in {"cancel", "replace", "reprioritise"}
            ), None)
            if queued_interrupt is not None:
                interrupted.set()
                return
            await asyncio.sleep(0.25)

    async def consume() -> None:
        nonlocal outcome, last_assistant_content, last_plan_items, task_id
        async for event in client.stream_session(session_id, from_event_id):
            sequence = event.get("stream_sequence")
            if sequence is not None:
                local_state.save_session_state(session_id, last_event_id=int(sequence))

            event_task_id = event.get("task_id")
            event_type = event.get("event_type") or event.get("event")
            payload = event.get("payload") or {}

            # A live-branched instruction (see LiveInstructionIn/
            # add_task_instruction) starts a real continuation task under a
            # NEW task_id once the current turn finishes at a safe boundary
            # -- this is the SAME session-level stream, so its events are
            # already arriving here. Follow it instead of stopping: from the
            # user's point of view this is still one running task that just
            # incorporated their new guidance and revised its plan.
            if str(event_task_id) == task_id and event_type == "task_continued":
                new_task_id = str(payload.get("continuation_task_id") or "")
                if new_task_id:
                    console.print(f"[cyan]◆ Continuing with your instruction[/cyan] (task {new_task_id})")
                    task_id = new_task_id
                    local_state.save_session_state(
                        session_id, last_task_id=task_id,
                        active_task={"id": task_id, "objective": str(payload.get("instruction") or ""), "mode": "coding"},
                    )
                continue

            if str(event_task_id) != task_id:
                continue

            phase_by_event = {
                "plan_created": "plan", "tool_call_requested": "execute",
                "command_started": "execute", "file_mutation": "execute",
                "approval_required": "waiting_for_approval", "task_diagnostics": "validate",
                "ai_task_completed": "report", "ai_task_failed": "report",
            }
            if event_type in phase_by_event:
                local_state.save_session_state(session_id, current_phase=phase_by_event[event_type])

            if event_type == "plan_created":
                items = payload.get("items") if isinstance(payload.get("items"), list) else []
                last_plan_items = items
                # If a saved plan is already active (e.g. re-planning mid
                # `execute-plan`), keep its persisted step list current too --
                # for a brand-new `plan` mode task there's no saved plan yet
                # (that only happens once the task completes, below), so
                # `last_plan_items` is what carries this forward in that case.
                state = local_state.get_session_state(session_id)
                if state.active_plan_id:
                    local_state.update_plan_steps(session_id, state.active_plan_id, items)

            if event_type == "file_mutation":
                state = local_state.get_session_state(session_id)
                mutation_id = str(payload.get("mutation_id") or "")
                existing = {str(item.get("mutation_id")): item for item in state.modified_files}
                existing[mutation_id] = {
                    "mutation_id": mutation_id, "path": payload.get("path"),
                    "original_hash": payload.get("hash_before"), "current_hash": payload.get("hash_after"),
                    "pre_existing_changes": bool(state.repository_context.get("dirty")),
                    "modified_by_action_ids": [state.running_action.get("id")] if state.running_action else [],
                    "validation_status": "pending", "revert_status": "none",
                }
                local_state.save_session_state(session_id, modified_files=list(existing.values()))

            if event_type == "file_mutation_reverted":
                state = local_state.get_session_state(session_id)
                for item in state.modified_files:
                    if str(item.get("mutation_id")) == str(payload.get("mutation_id")):
                        item["revert_status"] = "reverted"
                local_state.save_session_state(session_id, modified_files=state.modified_files)

            if event_type == "task_diagnostics":
                state = local_state.get_session_state(session_id)
                result = {
                    "task_id": task_id, "status": payload.get("completion_status"),
                    "provider": payload.get("provider"), "model": payload.get("model"),
                    "tool_calls": len(payload.get("tool_calls") or []),
                    "failed_tool_calls": sum(1 for call in payload.get("tool_calls") or [] if call.get("success") is False),
                }
                state.validation_results = (state.validation_results + [result])[-100:]
                local_state.save_session_state(session_id, validation_results=state.validation_results)

            if event_type == "tool_output":
                result_envelope = payload.get("result") if isinstance(payload.get("result"), dict) else payload
                evidence = payload.get("evidence") or result_envelope.get("evidence") or []
                file_reads = [
                    item for item in evidence
                    if isinstance(item, dict) and item.get("type") == "file_read" and item.get("path")
                ]
                if file_reads:
                    state = local_state.get_session_state(session_id)
                    inspected = dict(state.inspected_files)
                    for item in file_reads:
                        inspected[str(item["path"])] = {
                            "path": item["path"],
                            "sha256": item.get("sha256"),
                            "last_read_at": item.get("timestamp"),
                            "tool_call_id": item.get("tool_call_id"),
                        }
                    local_state.save_session_state(session_id, inspected_files=inspected)

            if event_type == "assistant_message":
                last_assistant_content = str(payload.get("visible_content", ""))

            if event_type == "approval_required":
                # Render the full command/cwd/reason/risk card before asking
                # for a decision. The prompt then stays compact and does not
                # duplicate a less-informative second approval panel.
                renderer.handle_event(event)
                command_id = payload.get("command_id")
                if command_id is not None and command_id not in prompted_command_ids:
                    prompted_command_ids.add(command_id)
                    decision = resolve_approval_decision(
                        console,
                        str(payload.get("command", "")),
                        str(payload.get("risk_level", "medium")),
                        approval_policy,
                        interactive,
                        display_preview=False,
                    )
                    try:
                        await client.approve_command(command_id, decision)
                    except RemoteAPIError:
                        pass

            if event_type != "approval_required":
                renderer.handle_event(event)

            if event_type == "ai_task_completed":
                outcome = TaskOutcome(status="completed", summary=last_assistant_content or "", plan_items=last_plan_items)
                if last_assistant_content:
                    local_state.save_session_state(
                        session_id, conversation_summary=last_assistant_content[-4000:],
                    )
                return
            if event_type == "ai_task_failed":
                outcome = TaskOutcome(status="failed", error=str(payload.get("error", "unknown error")))
                return

    consume_task = asyncio.ensure_future(consume())
    interrupt_task = asyncio.ensure_future(interrupted.wait())
    queue_task = asyncio.ensure_future(watch_instruction_queue())
    try:
        done, _pending = await asyncio.wait({consume_task, interrupt_task}, return_when=asyncio.FIRST_COMPLETED)
        if consume_task not in done:
            consume_task.cancel()
            try:
                await consume_task
            except asyncio.CancelledError:
                pass
            if on_interrupt == "detach":
                outcome = TaskOutcome(status="detached", summary=task_id)
            else:
                await client.cancel_task(task_id)
                if queued_interrupt and queued_interrupt.get("classification") == "cancel":
                    local_state.update_instruction(session_id, str(queued_interrupt.get("id")), "completed")
                reason = "Reprioritised by queued instruction" if queued_interrupt else "Interrupted by user"
                outcome = TaskOutcome(status="cancelled", error=reason)
        else:
            await consume_task
    finally:
        interrupt_task.cancel()
        queue_task.cancel()
        uninstall_sigint()
        renderer.finish()

    if outcome is None:
        task_status = await client.get_task(task_id)
        status = str(task_status.get("status", "failed"))
        if status == "completed":
            outcome = TaskOutcome(status="completed", summary=task_status.get("final_answer") or "")
        elif status in ACTIVE_TASK_STATUSES:
            if reconnect_attempt < MAX_TASK_STREAM_RECONNECTS:
                delay = min(8.0, 0.5 * (2 ** reconnect_attempt))
                console.print(
                    f"[dim]· Event stream closed; reconnecting from the last checkpoint "
                    f"({reconnect_attempt + 1}/{MAX_TASK_STREAM_RECONNECTS})...[/dim]"
                )
                await asyncio.sleep(delay)
                cursor = local_state.get_session_state(session_id).last_event_id
                return await _stream_task(
                    client, renderer, console,
                    session_id=session_id, task_id=task_id,
                    approval_policy=approval_policy, interactive=interactive,
                    from_event_id=cursor, on_interrupt=on_interrupt,
                    reconnect_attempt=reconnect_attempt + 1,
                )
            outcome = TaskOutcome(status="detached", summary=task_id)
        else:
            outcome = TaskOutcome(status=status, error=task_status.get("error"))
    return outcome


async def run_shell_command(
    client: RemoteAPIClient,
    console: Console,
    *,
    session_id: int,
    command: str,
    approval_policy: str,
    interactive: bool,
) -> TaskOutcome:
    action = local_state.start_action(
        session_id, action_type="shell_command", purpose="Run an explicit Remote command",
        risk="policy_classified", detail=command,
    )
    cmd = await client.submit_command(session_id, command)
    command_id = cmd["id"]
    console.print(f"[bold]$[/bold] {command}")

    prompted = False
    status = str(cmd.get("status", ""))
    interrupted, uninstall_sigint = _install_sigint_watcher()
    try:
        # Approval is durable server state, not a countdown. Keep polling
        # until the user decides or cancels with Ctrl+C; command execution
        # still has its own bounded server-side timeout after approval.
        while True:
            if interrupted.is_set():
                cmd = await client.cancel_command(command_id)
                status = str(cmd.get("status", "cancelled"))
                console.print("[dim]Cancelled.[/dim]")
                break
            cmd = await client.get_command(command_id)
            status = str(cmd.get("status", ""))
            if status == "pending_approval" and not prompted:
                prompted = True
                decision = resolve_approval_decision(
                    console, str(cmd.get("command_text", command)), str(cmd.get("safety_tier", "medium")),
                    approval_policy, interactive,
                )
                await client.approve_command(command_id, decision)
            if status in TERMINAL_COMMAND_STATUSES:
                break
            try:
                await asyncio.wait_for(interrupted.wait(), timeout=COMMAND_POLL_INTERVAL_SECONDS)
            except asyncio.TimeoutError:
                pass
    finally:
        uninstall_sigint()

    stdout = str(cmd.get("stdout") or "")
    stderr = str(cmd.get("stderr") or "")
    exit_code = cmd.get("exit_code")
    body = stdout.strip()
    if stderr.strip():
        body = (body + "\n" + stderr.strip()).strip()
    ok = status == "completed" and exit_code == 0
    if not body:
        if ok:
            body = "Command completed successfully with no output"
        elif status == "denied":
            body = "Command was rejected by the user"
        elif status == "cancelled":
            body = "Command was cancelled"
        else:
            body = f"Command failed with exit code {exit_code}"
    console.print(Panel(body, title=f"{status} · exit {exit_code}", border_style="green" if ok else "red"))

    if ok:
        outcome = TaskOutcome(status="completed", summary=stdout)
    else:
        outcome = TaskOutcome(status=status or "failed", error=stderr or f"exit code {exit_code}")
    local_state.finish_action(session_id, action.id, status=outcome.status, summary=f"exit={exit_code}")
    local_state.checkpoint(session_id, reason=f"command_{outcome.status}", summary=f"exit={exit_code}")
    return outcome


async def follow_session_logs(
    client: RemoteAPIClient,
    renderer: StreamRenderer,
    console: Console,
    *,
    session_id: int,
    from_event_id: int = 0,
) -> None:
    interrupted, uninstall_sigint = _install_sigint_watcher()

    async def consume() -> None:
        async for event in client.stream_session(session_id, from_event_id):
            sequence = event.get("stream_sequence")
            if sequence is not None:
                local_state.save_session_state(session_id, last_event_id=int(sequence))
            renderer.handle_event(event)

    consume_task = asyncio.ensure_future(consume())
    interrupt_task = asyncio.ensure_future(interrupted.wait())
    try:
        done, _pending = await asyncio.wait({consume_task, interrupt_task}, return_when=asyncio.FIRST_COMPLETED)
        if consume_task not in done:
            consume_task.cancel()
            try:
                await consume_task
            except asyncio.CancelledError:
                pass
        else:
            await consume_task
    finally:
        interrupt_task.cancel()
        uninstall_sigint()
        renderer.finish()
