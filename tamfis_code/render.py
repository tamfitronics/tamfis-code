"""Terminal rendering for the Remote event stream.

Mirrors the same event vocabulary and card concepts the web workspace's
RemoteMessageBubble/RemoteSuiteBubble cards render (plan/status lines, tool
call cards, command cards, file cards) -- see
tamfis-frontend/src/workspaces/remote/RemoteMessageBubble.tsx and
docs/REMOTE_AGENT_MASTER_SPEC.md Phase 7/9. Presentation only: no network
calls, no approval decisions -- see runner.py for that.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from .metrics import MetricsTracker

_TOOL_ANNOUNCE_RE = re.compile(r"Using tool:\s*(.+?)\.\.\.\s*$")

# Mirrors runner.py's own `phase_by_event` mapping (used there to persist
# SessionState.current_phase) purely for this renderer's live display -- kept
# as a separate copy rather than importing runner.py's dict so this module
# stays presentation-only and driven only by the events it already parses,
# per its module docstring.
_PHASE_BY_EVENT = {
    "plan_created": "plan", "tool_call_requested": "execute",
    "command_started": "execute", "file_mutation": "execute",
    "approval_required": "waiting_for_approval", "task_diagnostics": "validate",
    "ai_task_completed": "report", "ai_task_failed": "report",
}

# Chars-per-token is a rough English-text average, used only because
# assistant_delta payloads carry raw text, not a real token count -- the
# live panel labels this "~" to avoid presenting false precision.
_CHARS_PER_TOKEN_ESTIMATE = 4


def _tool_action_label(name: str, arguments: Optional[dict[str, Any]] = None, *, completed: bool = False) -> str:
    """Translate implementation identifiers into concise engineering actions."""
    arguments = arguments or {}
    normalized = (name or "tool").strip().lower().replace("-", "_").rsplit("/", 1)[-1]
    target = str(
        arguments.get("path") or arguments.get("file_path") or arguments.get("pattern")
        or arguments.get("query") or arguments.get("command") or ""
    ).strip()
    verbs = {
        "read_file": ("Reading", "Read"),
        "glob_files": ("Finding repository files", "Found repository files"),
        "search_files": ("Searching repository files", "Searched repository files"),
        "grep_files": ("Searching repository contents", "Searched repository contents"),
        "edit_file": ("Editing", "Edited"),
        "write_file": ("Writing", "Wrote"),
        "remote_exec": ("Running command", "Ran command"),
        "execute_command": ("Running command", "Ran command"),
        "run_command": ("Running command", "Ran command"),
        "web_search": ("Searching the web", "Searched the web"),
        "list_directory": ("Inspecting directory", "Inspected directory"),
        "create_directory": ("Creating directory", "Created directory"),
    }
    active, done = verbs.get(normalized, (normalized.replace("_", " ").strip().capitalize(), normalized.replace("_", " ").strip().capitalize()))
    label = done if completed else active
    if target:
        compact = target if len(target) <= 120 else target[:117] + "…"
        label += f" · {compact}"
    return label


def _tool_result_message(payload: dict[str, Any]) -> tuple[str, bool]:
    """Render structured tool results without asking the model to infer status.

    Backwards compatible with legacy payloads that only contain ``content``.
    """
    result = payload.get("result") if isinstance(payload.get("result"), dict) else payload
    success = result.get("ok")
    if success is None:
        success = result.get("success")
    status = str(result.get("status") or "").strip().lower()
    error_code = str(result.get("error_code") or "").strip()
    message = str(result.get("message") or result.get("error") or "").strip()
    content = str(result.get("content") or "").strip()
    stdout = str(result.get("stdout") or "").strip()
    stderr = str(result.get("stderr") or "").strip()
    exit_code = result.get("exit_code")
    path = str(result.get("resolved_path") or result.get("path") or result.get("requested_path") or "").strip()

    canonical = {
        "not_found": f"File not found: {path}" if path else "File not found",
        "permission_denied": f"Permission denied: {path}" if path else "Permission denied",
        "outside_allowed_scope": f"Path is outside the allowed workspace: {path}" if path else "Path is outside the allowed workspace",
        "path_rejected": f"Path rejected: {path}" if path else "Path rejected",
        "timed_out": message or "Command timed out",
        "cancelled": message or "Command was cancelled",
        "approval_rejected": message or "Command was rejected by the user",
        "tool_unavailable": message or "Tool unavailable",
        "provider_unavailable": message or "Provider unavailable",
        "model_unavailable": message or "Model unavailable",
    }
    if status in canonical:
        return canonical[status], True
    if error_code == "FILE_NOT_FOUND":
        return (message or (f"File not found: {path}" if path else "File not found")), True
    if error_code == "PERMISSION_DENIED":
        return (message or (f"Permission denied: {path}" if path else "Permission denied")), True

    failed = success is False or status in {
        "command_failed", "invalid_path", "not_a_file", "not_a_directory",
        "internal_error", "failed", "error",
    } or bool(error_code)
    if failed:
        if message:
            return message, True
        if stderr:
            return stderr, True
        if content:
            return content, True
        if exit_code is not None:
            return f"Command failed with exit code {exit_code}", True
        return "Tool operation failed", True

    if content:
        return content, False
    body = "\n".join(part for part in (stdout, stderr) if part).strip()
    if body:
        return body, False
    if status == "empty_success" or success is True or exit_code == 0:
        return message or "Command completed successfully with no output", False
    tool = str(payload.get("tool") or payload.get("name") or "tool")
    return message or f"{tool} returned an invalid result envelope", True


def _bounded_preview(text: str, limit: int = 8_000) -> str:
    if len(text) <= limit:
        return text
    head = int(limit * 0.7)
    tail = limit - head
    return f"{text[:head]}\n\n… {len(text) - limit:,} characters omitted …\n\n{text[-tail:]}"


def _format_diagnostics_line(payload: dict[str, Any]) -> str:
    """One-line summary of a task_diagnostics event (see PHASE 17 note in
    tier_ii_gateway/api/remote.py's _run_remote_ai_task_background) -- the
    self-diagnostic surface for a single turn: what context it reused/
    rescanned, which provider/model answered, how many tool calls it made
    and how many of those failed, how many artifacts it produced, and how
    it ended. Pulled out as a pure function so it's testable without a
    Console."""
    parts = []
    reused = payload.get("context_reused")
    if reused is True:
        parts.append("context reused")
    elif reused is False:
        parts.append(f"context rescanned ({payload.get('rescan_reason') or 'unknown'})")
    provider = payload.get("provider")
    model = payload.get("model")
    if provider or model:
        parts.append(f"{provider or '?'}/{model or '?'}")
    tool_calls = payload.get("tool_calls") or []
    if tool_calls:
        failed = sum(1 for tc in tool_calls if tc.get("success") is False)
        tool_text = f"{len(tool_calls)} tool call{'s' if len(tool_calls) != 1 else ''}"
        if failed:
            tool_text += f" ({failed} failed)"
        parts.append(tool_text)
    artifacts = payload.get("artifacts") or []
    if artifacts:
        parts.append(f"{len(artifacts)} artifact{'s' if len(artifacts) != 1 else ''}")
    parts.append(f"status={payload.get('completion_status') or 'unknown'}")
    return "Diagnostics: " + ", ".join(parts)


class StreamRenderer:
    def __init__(self, console: Console, objective: str = ""):
        self.console = console
        self._assistant_open = False
        self._tool_names_by_call_id: dict[str, str] = {}
        self._selected_provider: Optional[str] = None
        self.streamed_final_text = False  # True once any assistant_delta content is shown
        self.debug = os.environ.get("TAMFIS_CODE_DEBUG", "").lower() in {"1", "true", "yes"}

        # Live task-visibility panel -- gated on the console actually being a
        # TTY so redirected/piped output (`tamfis-code agent "..." > out.txt`)
        # keeps today's clean plain-text behaviour untouched.
        self._objective = objective
        self._phase = "idle"
        self._plan_steps: list[dict[str, Any]] = []
        self._task_start = time.monotonic()
        self._metrics = MetricsTracker()
        self._is_tty = bool(getattr(console, "is_terminal", False))
        self._live: Optional[Live] = None
        if self._is_tty:
            self._live = Live(self._build_panel(), console=self.console, refresh_per_second=8, transient=False)
            self._live.start()

    def _build_panel(self) -> Panel:
        lines = []
        if self._objective:
            lines.append(f"[bold]{self._objective}[/bold]")
        lines.append(f"[dim]phase:[/dim] {self._phase}")
        for step in self._plan_steps:
            status = step.get("status")
            marker = "[green]✓[/green]" if status == "completed" else (
                "[yellow]◉[/yellow]" if status == "in_progress" else "[dim]○[/dim]"
            )
            # No per-step completion event exists yet (see state.py's
            # update_plan_steps docstring) -- statuses beyond the initial
            # plan_created payload are a best-effort approximation, so this
            # is labelled "~" rather than presented as precise.
            lines.append(f"  {marker} {'~ ' if status == 'in_progress' else ''}{step.get('step') or ''}")
        elapsed = time.monotonic() - self._task_start
        tokens = self._metrics.metrics.tokens_used
        if tokens:
            lines.append(
                f"[dim]~{tokens:,} tok (est.) · {self._metrics.metrics.tokens_per_second:.1f} tok/s · "
                f"{elapsed:.1f}s[/dim]"
            )
        else:
            lines.append(f"[dim]{elapsed:.1f}s[/dim]")
        return Panel(Group(*(Text.from_markup(line) for line in lines)), border_style="cyan", expand=False)

    def _refresh_live(self) -> None:
        if self._live is not None:
            self._live.update(self._build_panel())

    def _record_tokens(self, content: str) -> None:
        if not content:
            return
        estimated_tokens = max(1, len(content) // _CHARS_PER_TOKEN_ESTIMATE)
        elapsed_ms = (time.monotonic() - self._task_start) * 1000
        self._metrics.record(estimated_tokens, elapsed_ms)

    def _close_assistant(self) -> None:
        if self._assistant_open:
            self.console.print()
            self._assistant_open = False

    def handle_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("event_type") or event.get("event") or event.get("type")
        payload = event.get("payload") or {}

        if event_type in _PHASE_BY_EVENT and self._phase != _PHASE_BY_EVENT[event_type]:
            self._phase = _PHASE_BY_EVENT[event_type]
            self._refresh_live()

        if event_type == "assistant_delta":
            content = str(payload.get("content", ""))
            if not self._assistant_open:
                self.console.print("[bold cyan]Assistant[/bold cyan]")
                self._assistant_open = True
            # Whitespace/reasoning-only provider frames are not a visible
            # final answer. Marking them as displayed suppresses cli.py's
            # authoritative persisted-summary fallback and leaves a blank
            # terminal even though the task completed with real text.
            if content.strip():
                self.streamed_final_text = True
            self._record_tokens(content)
            self._refresh_live()
            self.console.print(content, end="")
            return

        if event_type == "plan_created":
            self._close_assistant()
            stage = payload.get("stage")
            content = str(payload.get("content", ""))
            items = payload.get("items") if isinstance(payload.get("items"), list) else []
            if items:
                self._plan_steps = [item for item in items if isinstance(item, dict) and item.get("status") != "context"]
                self._refresh_live()
                # The live panel (when active) already shows this step list
                # in place, updating as later events refresh it -- printing
                # it again as a static scroll-log entry would just duplicate
                # it. Non-TTY output (no live panel) keeps today's one-shot
                # print, since that's the only place this ever showed up.
                if self._live is None:
                    self.console.print(f"[bold cyan]{payload.get('title') or 'Plan'}[/bold cyan]")
                    for item in self._plan_steps:
                        marker = "◉" if item.get("status") == "in_progress" else "○"
                        self.console.print(f"  [cyan]{marker}[/cyan] {item.get('step') or ''}")
                return
            if stage == "tool_execution":
                match = _TOOL_ANNOUNCE_RE.search(content)
                tool = str(payload.get("tool") or (match.group(1) if match else "tool"))
                call_id = payload.get("tool_call_id")
                if call_id:
                    self._tool_names_by_call_id[str(call_id)] = tool
                args = payload.get("arguments") or {}
                arg_text = ", ".join(f"{k}={v}" for k, v in args.items() if v not in (None, "")) if isinstance(args, dict) else ""
                label = f"[bold yellow]→ {_tool_action_label(tool, args)}[/bold yellow]"
                if self.debug and arg_text:
                    label += f"  [dim]{arg_text}[/dim]"
                self.console.print(label)
            else:
                # Tier V's compatibility router can describe its internal
                # adapter slot (for example "nim") even though Tier IV is
                # honoring an explicit end-provider such as Ollama.  Once an
                # authoritative model_selected event exists, never display a
                # contradictory provider claim in progress output.
                if self._selected_provider and content.lower().startswith("executing with "):
                    content = f"Executing with {self._selected_provider}..."
                self.console.print(f"[dim]· {content}[/dim]")
            return

        if event_type == "tool_call_requested":
            self._close_assistant()
            name = str(payload.get("name") or payload.get("tool") or "tool")
            args = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {}
            self.console.print(f"[bold yellow]→ {_tool_action_label(name, args)}[/bold yellow]")
            return

        if event_type == "tool_output":
            self._close_assistant()
            tool = str(payload.get("tool", "tool"))
            result_envelope = payload.get("result") if isinstance(payload.get("result"), dict) else payload
            # Command/file events already carry the useful result. Some
            # canonical tool-completion envelopes contain only a tool name
            # and success flag; rendering those produced the misleading,
            # repetitive "Tool completed without a structured result" card.
            if not any(
                result_envelope.get(key) not in (None, "", [], {})
                for key in (
                    "content", "stdout", "stderr", "message", "error",
                    "error_code", "status", "exit_code", "resolved_path",
                    "path", "requested_path",
                )
            ):
                return
            content, failed = _tool_result_message(payload)
            style = "red" if failed else "green"
            result = result_envelope
            args = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {}
            if not args and isinstance(result, dict):
                args = {
                    "path": result.get("resolved_path") or result.get("path"),
                    "command": result.get("command"),
                }
            title = (
                f"{_tool_action_label(tool, args, completed=True)} — failed"
                if failed else _tool_action_label(tool, args, completed=True)
            )
            self.console.print(Panel(content, title=title, border_style=style, expand=False))
            return

        if event_type in ("artifact_generated", "diff_available"):
            self._close_assistant()
            filename = payload.get("filename") or "generated file"
            size = payload.get("size_bytes")
            size_text = f" ({size} bytes)" if size else ""
            url = payload.get("download_url") or payload.get("file_url")
            body = f"{filename}{size_text}"
            if url:
                body += f"\n{url}"
            validated = payload.get("validated")
            title = "File generated" + (" · validated" if validated is True else "")
            self.console.print(Panel(body, title=title, border_style="blue", expand=False))
            return

        if event_type == "file_mutation":
            self._close_assistant()
            path = payload.get("path", "?")
            added, removed = payload.get("lines_added", 0), payload.get("lines_removed", 0)
            mutation_id = payload.get("mutation_id", "?")
            self.console.print(
                f"[bold blue]✎ {path}[/bold blue]  [dim]+{added}/-{removed}  "
                f"(revert with: /revert {mutation_id})[/dim]"
            )
            return

        if event_type == "file_mutation_reverted":
            self._close_assistant()
            self.console.print(f"[green]↺ Reverted {payload.get('path', '?')}[/green]")
            return

        if event_type == "command_started":
            self._close_assistant()
            self.console.print(f"[bold]$[/bold] {payload.get('command', '')}")
            return

        if event_type in ("command_completed", "command_failed"):
            self._close_assistant()
            stdout = str(payload.get("stdout", ""))
            stderr = str(payload.get("stderr", ""))
            exit_code = payload.get("exit_code")
            style = "green" if event_type == "command_completed" and exit_code == 0 else "red"
            body = stdout.strip()
            if stderr.strip():
                body = (body + "\n" + stderr.strip()).strip()
            if not body:
                body = (
                    "Command completed successfully with no output"
                    if event_type == "command_completed" and exit_code == 0
                    else f"Command failed with exit code {exit_code}"
                )
            self.console.print(Panel(body, title=f"exit {exit_code}", border_style=style, expand=False))
            return

        if event_type == "approval_required":
            self._close_assistant()
            # payload["command"] is the command TEXT (a plain string), not
            # an object -- see the matching comment in runner.py's approval
            # handling for why that distinction matters.
            text = payload.get("command")
            risk = payload.get("risk_level", "?")
            command_id = payload.get("command_id")
            cwd = str(payload.get('cwd') or payload.get('working_directory') or '?')
            reason = str(payload.get('reason') or 'The agent requested this command.')
            command_text = _bounded_preview(str(text or ''))
            body = (
                f"Command:\n{command_text}\n\n"
                f"Working directory:\n{cwd}\n\n"
                f"Reason:\n{reason}\n\n"
                f"Risk:\n{risk}"
            )
            # This event fires identically whether a human is actively
            # attached to answer it (attach's own interactive prompt handles
            # that case) or the task is backgrounded/being watched read-only
            # via `logs --follow` -- in the latter case there is no other
            # way to discover the id `tamfis-code approve`/`reject` need.
            # Before this fix, a backgrounded approval was fully opaque:
            # visible that SOMETHING needs approval, with no way to act on
            # it short of reading the database directly.
            if command_id is not None:
                body += f"\n\ntamfis-code approve {command_id}\ntamfis-code reject {command_id}"
            self.console.print(
                Panel(
                    body,
                    title=f"Approval required — risk: {risk}" + (f" (id {command_id})" if command_id is not None else ""),
                    border_style="magenta",
                    expand=False,
                )
            )
            return

        if event_type in ("context_reused", "context_rescanned"):
            self._close_assistant()
            reason = payload.get("reason", "unknown")
            if event_type == "context_reused":
                self.console.print("[dim]· Reusing workspace context — repository unchanged since last turn[/dim]")
            else:
                self.console.print(f"[dim]· Workspace rescanned (reason: {reason})[/dim]")
            return

        if event_type == "task_diagnostics":
            self._close_assistant()
            self.console.print(f"[dim]· {_format_diagnostics_line(payload)}[/dim]")
            return

        if event_type == "model_selected":
            self._close_assistant()
            provider = payload.get("provider") or "unknown"
            self._selected_provider = str(provider)
            model = payload.get("model") or "unknown"
            reason = payload.get("selection_reason") or "explicit selection or orchestration routing"
            self.console.print(f"[dim]· Provider: {provider} · Model: {model} · {reason}[/dim]")
            return

        if event_type == "ai_task_failed":
            self._close_assistant()
            self.console.print(f"[bold red]Task failed:[/bold red] {payload.get('error', 'unknown error')}")
            return

        if event_type in ("ai_task_completed", "assistant_message", "task_cancelled", "heartbeat", "stream_closed"):
            return  # runner.py owns lifecycle decisions for these; nothing new to print

        # Unrecognised event type: show it plainly rather than silently
        # dropping it -- a gap in this renderer should be visible, not hidden.
        self._close_assistant()
        content = payload.get("content") or payload.get("status") or ""
        if content:
            self.console.print(f"[dim]· {event_type}: {content}[/dim]")
        if self.debug:
            self.console.print(json.dumps(event, indent=2, default=str, ensure_ascii=False))

    def finish(self) -> None:
        self._close_assistant()
        if self._live is not None:
            self._live.stop()
            self._live = None


def print_banner(console: Console, *, host: str, workspace_root: str, mode: str, approval_policy: str) -> None:
    console.print(Text("TamfisGPT Code", style="bold cyan"))
    console.print(f"[dim]Workspace:[/dim] {workspace_root}")
    console.print(f"[dim]Mode:[/dim] {mode}   [dim]Approval:[/dim] {approval_policy}   [dim]Host:[/dim] {host}")


def print_error(console: Console, message: str) -> None:
    console.print(f"[bold red]Error:[/bold red] {message}")


def print_recent_thread(console: Console, messages: list[dict[str, Any]], limit: int = 6) -> None:
    """Prints the tail of GET /thread's message list -- used by `/resume`
    and `tamfis-code resume` so switching sessions doesn't drop the user
    into a blank prompt with no memory of what was being worked on."""
    if not messages:
        return

    turns: dict[str, dict[str, Optional[str]]] = {}
    order: list[str] = []
    for message in messages:  # oldest-first, per GET /thread's own ordering
        key = str(message.get("task_id") or message.get("id"))
        if key not in turns:
            turns[key] = {"objective": None, "answer": None}
            order.append(key)
        if message.get("role") == "user":
            turns[key]["objective"] = message.get("visible_content")
        else:
            turns[key]["answer"] = message.get("visible_content")

    shown = [key for key in order if turns[key]["objective"] or turns[key]["answer"]][-limit:]
    if not shown:
        return

    console.print("[bold]Recent history[/bold]")
    for key in shown:
        turn = turns[key]
        if turn["objective"]:
            console.print(f"[dim]›[/dim] {turn['objective']}")
        if turn["answer"]:
            console.print(f"  {turn['answer']}")
        console.print()


def print_unified_diff(console: Console, diff_text: str, *, title: str = "Changes") -> None:
    """Render a unified diff semantically; Console owns colour fallback."""
    no_color = bool(getattr(console, "no_color", False))
    console.print(title if no_color else f"[bold]{title}[/bold]")
    if not diff_text.strip():
        console.print("(empty diff)" if no_color else "[dim](empty diff)[/dim]")
        return
    for line in diff_text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            style = "bold"
        elif line.startswith("+"):
            style = "green"
        elif line.startswith("-"):
            style = "red"
        elif line.startswith("@@"):
            style = "cyan"
        else:
            style = "dim"
        console.print(Text(line, style=None if no_color else style), soft_wrap=True)
