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
from rich.table import Table
from rich.spinner import Spinner
from rich.text import Text

from .metrics import MetricsTracker
from .safety import redact_secrets

_TOOL_ANNOUNCE_RE = re.compile(r"Using tool:\s*(.+?)\.\.\.\s*$")

# Mirrors runner.py's own `phase_by_event` mapping (used there to persist
# SessionState.current_phase) purely for this renderer's live display -- kept
# as a separate copy rather than importing runner.py's dict so this module
# stays presentation-only and driven only by the events it already parses,
# per its module docstring.
_PHASE_BY_EVENT = {
    # Submission/context/model lifecycle.  These events are emitted by both
    # the local provider loop and the remote SSE runner so the live card never
    # sits at its constructor default while a network request is in flight.
    "task_submitting": "submitting",
    "task_submitted": "queued",
    "task_started": "understand",
    "context_loading": "understand",
    "context_reused": "understand",
    "context_rescanned": "understand",
    "routing_started": "route",
    "model_selected": "route",
    "provider_request_started": "respond",
    "reasoning_delta": "reasoning",
    "assistant_delta": "respond",
    "plan_created": "plan",
    "tool_call_requested": "execute",
    "tool_output": "execute",
    "command_started": "execute",
    "command_completed": "execute",
    "command_failed": "execute",
    "file_mutation": "execute",
    "approval_required": "waiting_for_approval",
    "task_diagnostics": "validate",
    "ai_task_completed": "report",
    "ai_task_failed": "report",
    "orchestrator_understand": "understand",
    "orchestrator_inspect": "inspect",
    "orchestrator_route": "route",
    "orchestrator_plan": "plan",
    "orchestrator_execute": "execute",
    "orchestrator_observe": "observe",
    "orchestrator_repair": "repair",
    "orchestrator_validate": "validate",
    "orchestrator_report": "report",
    "orchestrator_waiting_for_approval": "waiting_for_approval",
    "orchestrator_completed": "report",
    "orchestrator_failed": "report",
}

# Chars-per-token is a rough English-text average, used only because
# assistant_delta payloads carry raw text, not a real token count -- the
# live panel labels this "~" to avoid presenting false precision.
_CHARS_PER_TOKEN_ESTIMATE = 4

# One friendly present-participle per phase for the single-line live status
# (spinner + "Verb... (elapsed - tokens)"), grounded in what's actually
# happening rather than picked at random -- "idle" is the constructor
# default, never actually shown (task_started fires before the first
# network call, see run_local_agent_turn).
_VERB_BY_PHASE = {
    "idle": "Working",
    "submitting": "Submitting",
    "queued": "Queued",
    "understand": "Understanding",
    "inspect": "Inspecting",
    "route": "Routing",
    "reasoning": "Thinking",
    "respond": "Responding",
    "plan": "Planning",
    "execute": "Working",
    "observe": "Observing",
    "repair": "Repairing",
    "waiting_for_approval": "Waiting",
    "validate": "Checking",
    "report": "Wrapping up",
}


# Rotating hints shown under the live status line during longer-running
# turns -- every one names a real, working command/flag (verified against
# this session's own testing), not a guessed or aspirational one. Kept
# short: this is a passing hint, not documentation.
_TIPS = [
    "Tip: `tamfis-code diffs` lists recent file changes; `tamfis-code revert <id>` undoes one.",
    "Tip: `tamfis-code resume` picks up your last session where it left off.",
    "Tip: `/mode` in the REPL changes the approval policy without restarting.",
    "Tip: `tamfis-code plan \"...\"` saves a plan without touching any files.",
    "Tip: `tamfis-code index . -s <name>` searches this codebase by symbol name.",
    "Tip: `tamfis-code screenshot <url>` takes a real browser screenshot.",
    "Tip: `tamfis-code enforce` runs this workspace's own test suite.",
    "Tip: `tamfis-code providers` shows which AI providers are configured and healthy.",
    "Tip: Ctrl+C exits `tamfis-code` cleanly, mid-turn or not.",
    "Tip: `/compact` in the REPL saves a context checkpoint you can return to later.",
    "Tip: `tamfis-code tools list` shows every tool this agent knows how to call.",
    "Tip: `--approval` (or `/mode`) controls how much gets auto-approved: ask/safe/full-auto/never/...",
]

# Seconds of elapsed time before the first tip appears, and how long each
# stays up -- avoids tip noise on turns that finish almost instantly.
_TIP_START_AFTER_SECONDS = 4.0
_TIP_ROTATE_EVERY_SECONDS = 8.0

# Streaming output is coalesced into readable blocks. Re-rendering a complete
# Markdown document for every one-character provider delta is quadratic and
# was the direct cause of the painfully slow terminal typing experience.
_ASSISTANT_REFRESH_INTERVAL_SECONDS = 0.08
_ASSISTANT_REFRESH_MIN_CHARS = 96
# Virtualize the live terminal viewport. The complete response remains in
# the runner/checkpoint; Rich only reparses a bounded recent window so a very
# long answer cannot turn every refresh into an ever-larger Markdown parse.
_ASSISTANT_LIVE_MAX_CHARS = 12_000
_ASSISTANT_LIVE_WINDOW_CHARS = 8_000
_ASSISTANT_SENTENCE_BOUNDARY_RE = re.compile(r"(?:[.!?](?:[\"'’)]*)\s+|\n{2,}|```\s*$)")
_USER_MESSAGE_MAX_DISPLAY_CHARS = 20_000


def _current_tip(elapsed: float) -> Optional[str]:
    if elapsed < _TIP_START_AFTER_SECONDS or not _TIPS:
        return None
    index = int((elapsed - _TIP_START_AFTER_SECONDS) // _TIP_ROTATE_EVERY_SECONDS) % len(_TIPS)
    return _TIPS[index]


def _format_elapsed(seconds: float) -> str:
    total = int(seconds)
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    return f"{minutes}m {secs}s"


def _format_token_count(n: int) -> str:
    if n < 1000:
        return str(n)
    return f"{n / 1000:.1f}k"


def _tool_action_label(name: str, arguments: Optional[dict[str, Any]] = None, *, completed: bool = False) -> str:
    """Translate implementation identifiers into concise engineering actions."""
    arguments = arguments or {}
    normalized = (name or "tool").strip().lower().replace("-", "_").rsplit("/", 1)[-1]
    command_target = arguments.get("command")
    target = str(
        arguments.get("path") or arguments.get("file_path") or arguments.get("pattern")
        or arguments.get("query") or command_target or ""
    ).strip()
    if command_target and target == str(command_target).strip():
        target = redact_secrets(target)
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


_RESULT_BLOCK_MAX_LINES = 20


def _render_result_block(console: Console, *, ok: bool, label: str, content: str) -> None:
    """Minimal, unboxed rendering for a completed tool call or shell command:
    one status-glyph line plus dimmed, indented output -- not a bordered
    Panel. Every tool/command result getting its own box (as command_started/
    tool_call_requested's completions used to) produced exactly the
    box-per-line noise a real agent session showed: a wall of boxes for
    routine reads/greps/pip-freezes. Approval prompts still get a Panel
    (see approval_required below) since those genuinely need to interrupt."""
    glyph, style = ("✓", "green") if ok else ("✗", "red")
    console.print(f"[{style}]{glyph}[/{style}] {label}")
    body = content.strip("\n")
    if not body:
        return
    lines = _bounded_preview(body).split("\n")
    if len(lines) > _RESULT_BLOCK_MAX_LINES:
        omitted = len(lines) - _RESULT_BLOCK_MAX_LINES
        lines = lines[:_RESULT_BLOCK_MAX_LINES] + [f"… {omitted} more line{'s' if omitted != 1 else ''} …"]
    for line in lines:
        console.print(f"  [dim]{line}[/dim]")


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
    def __init__(self, console: Console, *, mode_label: Optional[str] = None):
        self.console = console
        # Live-reported: switching mode (Shift+Tab) while a task is
        # actively streaming only ever printed a one-time scrolling
        # "◆ Mode switched to X" line (live_input.py's _cycle_mode) that
        # the next few lines of streamed output push off-screen -- unlike
        # Claude Code, where the current mode is a persistent, always-
        # visible part of the UI, not a message you can miss. `_mode_label`
        # is folded into the persistent Live status line itself (see
        # _build_status) instead, and set_mode_label() below forces an
        # immediate refresh so a switch is visible the instant it happens,
        # not just on the status line's next natural per-token update.
        self._mode_label = mode_label
        self._assistant_open = False
        # Printed at most once per StreamRenderer instance (i.e. once per
        # overall user turn, however many tool-call rounds it takes) --
        # _assistant_open alone used to gate this, so every round after a
        # tool call reopened a brand new "Assistant" header, making a single
        # turn with two tool rounds read as two separate answers instead of
        # one continuous response with tool calls woven through it.
        self._assistant_header_shown = False
        self._tool_names_by_call_id: dict[str, str] = {}
        self._selected_provider: Optional[str] = None
        self.streamed_final_text = False  # True once any assistant_delta content is shown
        self.debug = os.environ.get("TAMFIS_CODE_DEBUG", "").lower() in {"1", "true", "yes"}

        # Live task-visibility status line -- gated on the console actually
        # being a TTY so redirected/piped output (`tamfis-code agent "..." >
        # out.txt`) keeps today's clean plain-text behaviour untouched.
        # Single line + spinner (verb..."(elapsed - tokens)"), not a bordered
        # panel: matches how Claude Code itself shows live progress, rather
        # than a boxed multi-line status card.
        self._phase = "idle"
        self._status_detail = "Preparing the task"
        self._plan_steps: list[dict[str, Any]] = []
        self._task_start = time.monotonic()
        self._metrics = MetricsTracker()
        # Real reasoning-phase timing (see provider_protocols.py's
        # reasoning_content extraction) -- None/None until a reasoning delta
        # actually arrives; _thought_seconds freezes once real answer
        # content starts, and stays visible in the status line for the rest
        # of the turn the way Claude Code's own "thought for Xs" does.
        self._reasoning_start: Optional[float] = None
        self._reasoning_last: Optional[float] = None
        self._thought_seconds: Optional[float] = None
        self._spinner = Spinner("dots", style="cyan")
        self._is_tty = bool(getattr(console, "is_terminal", False))
        self._live: Optional[Live] = None
        # Accumulated text for the assistant block currently streaming, and
        # the Live handle re-rendering it as Markdown on every delta (TTY
        # only -- see the assistant_delta branch below for why non-TTY output
        # stays raw). Reset in _close_assistant() so each block between tool
        # rounds gets its own buffer instead of concatenating onto the last.
        self._assistant_buffer = ""
        self._assistant_pending = ""
        self._assistant_rendered_length = 0
        self._assistant_last_refresh = 0.0
        self._assistant_live: Optional[Live] = None
        # Set by live_input.py's LiveInputListener.start() for the duration
        # of one interactive turn; None for every other caller (one-shot
        # CLI commands, tests, the --remote path) -- suspend_live/resume_live
        # above only touch it when it's actually present.
        self.live_input_listener: Optional[Any] = None
        if self._is_tty and self.live_input_listener is None:
            self._live = Live(self._build_status(), console=self.console, refresh_per_second=8, transient=True)
            self._live.start()

    def set_mode_label(self, label: str) -> None:
        """Update the persistent status line's mode tag and refresh
        immediately -- called by live_input.py's Shift+Tab handler so a
        mid-task mode switch is visible the instant it happens, not just
        via a scrolling diagnostic line that later output pushes away."""
        self._mode_label = label
        self._refresh_live()

    def live_input_status(self) -> str:
        """Compact status text for prompt-toolkit's persistent footer.

        Rich's Live region is deliberately suspended while prompt-toolkit
        owns the terminal input line. Keep the same phase information visible
        in prompt-toolkit's bottom toolbar so opening follow-up input never
        hides what the agent is doing.
        """
        elapsed = _format_elapsed(time.monotonic() - self._task_start)
        tokens = self._metrics.metrics.tokens_used
        details = elapsed
        if tokens:
            details += f" · {_format_token_count(tokens)} tokens"
        verb = _VERB_BY_PHASE.get(self._phase, self._phase).capitalize()
        return f"{verb}… · {self._status_detail} ({details})"

    def _update_status_detail(self, event_type: str, payload: dict[str, Any]) -> None:
        """Turn structured stream events into human-readable footer detail."""
        if event_type in {"task_submitting", "task_submitted"}:
            self._status_detail = "Submitting the task"
        elif event_type in {"task_started", "context_loading"}:
            self._status_detail = "Loading workspace context"
        elif event_type in {"context_reused", "context_rescanned"}:
            self._status_detail = "Preparing repository context"
        elif event_type in {"routing_started", "model_selected"}:
            self._status_detail = "Selecting the best available model"
        elif event_type == "provider_request_started":
            self._status_detail = "Waiting for the model's next step"
        elif event_type == "reasoning_delta":
            self._status_detail = "Thinking through the next step"
        elif event_type == "tool_call_requested":
            name = str(payload.get("name") or payload.get("tool_name") or "tool")
            arguments = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {}
            self._status_detail = _tool_action_label(name, arguments)
        elif event_type == "tool_output":
            self._status_detail = "Reviewing the tool result"
        elif event_type in {"command_started", "command_completed", "command_failed"}:
            command = str(payload.get("command") or "the requested command")
            self._status_detail = f"Running {command[:120]}"
        elif event_type == "file_mutation":
            path = str(payload.get("path") or payload.get("resolved_path") or "the requested file")
            self._status_detail = f"Applying the change to {path[:120]}"
        elif event_type == "approval_required":
            self._status_detail = "Waiting for your approval"
        elif event_type in {"task_diagnostics", "context_rollover"}:
            self._status_detail = "Checking the work and updating context"
        elif event_type in {"ai_task_completed", "ai_task_failed"}:
            self._status_detail = "Finishing the response"

    def _build_status(self) -> Any:
        elapsed = time.monotonic() - self._task_start
        tokens = self._metrics.metrics.tokens_used
        detail_parts = [_format_elapsed(elapsed)]
        if tokens:
            detail_parts.append(f"↓ {_format_token_count(tokens)} tokens")
        if self._thought_seconds is not None:
            detail_parts.append(f"thought for {_format_elapsed(self._thought_seconds)}")
        elif self._reasoning_start is not None:
            # Still actively reasoning -- live-incrementing, not yet frozen.
            detail_parts.append(f"thought for {_format_elapsed(time.monotonic() - self._reasoning_start)}")
        verb = _VERB_BY_PHASE.get(self._phase, self._phase)
        # The literal brackets are escaped (\[...]) because Text.from_markup
        # below would otherwise parse "[accept-edits]" itself as an
        # (invalid, silently-dropped) markup tag rather than visible text --
        # confirmed by a test failure where the whole tag vanished.
        mode_tag = f"[cyan]\\[{self._mode_label}][/cyan] " if self._mode_label else ""
        label = f"{mode_tag}[bold]{verb}…[/bold] [dim]({' · '.join(detail_parts)})[/dim]"
        self._spinner.update(text=Text.from_markup(label))
        tip = _current_tip(elapsed)
        if not self._plan_steps and not tip and self.live_input_listener is None:
            return self._spinner
        lines = []
        if self.live_input_listener is not None:
            # Keep a real, persistent input box in the live task display. The
            # ordinary REPL editor is suspended while the agent owns the
            # terminal, with the editable follow-up line owned by
            # prompt_toolkit rather than a special control key.
            input_box = Panel(
                Text(
                    "Type a message and press Enter · Esc stops the task · Ctrl+C/Ctrl+D exits"
                ),
                title="Input",
                border_style="cyan",
                padding=(0, 1),
            )
            lines.append(input_box)
        if tip:
            lines.append(Text.from_markup(f"  [dim]{tip}[/dim]"))
        for step in self._plan_steps:
            status = step.get("status")
            marker = "[green]✓[/green]" if status == "completed" else (
                "[yellow]◉[/yellow]" if status == "in_progress" else "[dim]○[/dim]"
            )
            # No per-step completion event exists yet (see state.py's
            # update_plan_steps docstring) -- statuses beyond the initial
            # plan_created payload are a best-effort approximation, so this
            # is labelled "~" rather than presented as precise.
            lines.append(Text.from_markup(
                f"  {marker} {'~ ' if status == 'in_progress' else ''}{step.get('step') or ''}"
            ))
        return Group(self._spinner, *lines)

    def _refresh_live(self) -> None:
        if self._live is not None:
            self._live.update(self._build_status())

    def _print_plan_snapshot(
        self, items: list[dict[str, Any]], *, title: str,
        assumptions: Optional[list[Any]] = None, risks: Optional[list[Any]] = None,
    ) -> None:
        """Print an authoritative plan snapshot to scrollback.

        Rich Live is intentionally transient for the spinner, so it cannot
        be the source of truth for a plan users need to follow. Every plan
        creation and every status transition gets a durable snapshot here.
        """
        table = Table.grid(padding=(0, 1))
        table.add_column(width=4, justify="right", style="cyan")
        table.add_column(ratio=1)
        for index, item in enumerate(items, start=1):
            status = str(item.get("status") or "pending")
            marker = "✓" if status == "completed" else (
                "✗" if status == "failed" else ("▶" if status == "in_progress" else "○")
            )
            table.add_row(f"{index}. {marker}", f"{item.get('step') or ''} · {status}")
        body: list[Any] = [table]
        if assumptions:
            body.extend([Text("Assumptions", style="bold"), Text(" • " + "\n • ".join(map(str, assumptions)))])
        if risks:
            body.extend([Text("Risks", style="bold yellow"), Text(" • " + "\n • ".join(map(str, risks)))])
        self.console.print(Panel(Group(*body), title=title, border_style="cyan", expand=False))

    def _stop_live(self) -> None:
        """End the progress display before ordinary streamed output begins."""
        if self._live is not None:
            self._live.stop()
            self._live = None

    def suspend_live(self) -> None:
        """Stop the live status line before a blocking interactive prompt.

        Rich's Live redraws the terminal on its own refresh timer
        (refresh_per_second=8) independent of whatever else is writing to
        the console; a blocking `console.input()` call for an approval
        decision has no way to coordinate with that redraw, so the prompt
        can be silently overwritten/garbled while a human is trying to
        answer it. Approval gates must be visible and durable, not raced
        by a spinner -- call this before prompting, then resume_live()
        after the decision is made. Safe to call when already suspended or
        when no live line exists at all (non-TTY output).

        Also stops the streaming-assistant Markdown Live (if one is open --
        see the assistant_delta handler) and pauses live_input_listener (if
        attached -- see live_input.py): both would otherwise race a blocking
        prompt for the same terminal/fd exactly like the status line does.
        """
        self._stop_live()
        if self._assistant_live is not None:
            self._assistant_live.stop()
            self._assistant_live = None
        if self.live_input_listener is not None:
            self.live_input_listener.pause()

    def resume_live(self) -> None:
        """Restore the live status line after suspend_live(); a no-op on
        non-TTY output or if a live line is already active. Does not
        restart the assistant Live -- the next assistant_delta lazily
        recreates it from the still-intact buffer if a block was open."""
        if self.live_input_listener is not None:
            self.live_input_listener.resume()
        if self._is_tty and self._live is None and not self._assistant_open:
            self._live = Live(self._build_status(), console=self.console, refresh_per_second=8, transient=True)
            self._live.start()

    def _record_tokens(self, content: str) -> None:
        if not content:
            return
        estimated_tokens = max(1, len(content) // _CHARS_PER_TOKEN_ESTIMATE)
        elapsed_ms = (time.monotonic() - self._task_start) * 1000
        self._metrics.record(estimated_tokens, elapsed_ms)

    def _flush_assistant(self, *, force: bool = False) -> None:
        """Flush buffered assistant text in coherent blocks.

        Providers may emit one character per network frame. Rendering every
        frame independently makes Rich repeatedly parse the full Markdown
        document and produces the impression of one-character-per-second
        typing. We retain true streaming, but refresh only at sentence/block
        boundaries, after a useful amount of text, or on finalisation.
        """
        if not self._assistant_pending:
            return
        # Redirected output has no Live refresh loop to flush the final small
        # fragment later. Emit each non-TTY delta promptly; TTY output keeps
        # the coalesced Markdown refresh path for smooth interactive rendering.
        force = force or not self._is_tty
        now = time.monotonic()
        boundary = bool(_ASSISTANT_SENTENCE_BOUNDARY_RE.search(self._assistant_pending))
        enough_text = len(self._assistant_pending) >= _ASSISTANT_REFRESH_MIN_CHARS
        interval_elapsed = now - self._assistant_last_refresh >= _ASSISTANT_REFRESH_INTERVAL_SECONDS
        if not force and not boundary and not (enough_text and interval_elapsed):
            return
        self._assistant_buffer += self._assistant_pending
        self._assistant_pending = ""
        self._assistant_last_refresh = now
        if self._is_tty and self.live_input_listener is None:
            render_buffer = self._assistant_buffer
            if len(render_buffer) > _ASSISTANT_LIVE_MAX_CHARS:
                render_buffer = (
                    "[Earlier response content virtualized; full text is retained.]\n\n"
                    + render_buffer[-_ASSISTANT_LIVE_WINDOW_CHARS:]
                )
            # During streaming, plain Text avoids reparsing the complete
            # Markdown document on every update. This is deliberately a
            # terminal viewport optimization; the canonical response text
            # and checkpoint remain unchanged.
            document = Text(render_buffer)
            if self._assistant_live is None:
                self._assistant_live = Live(document, console=self.console, refresh_per_second=12)
                self._assistant_live.start()
            else:
                self._assistant_live.update(document, refresh=True)
        else:
            delta = self._assistant_buffer[self._assistant_rendered_length:]
            if delta:
                self.console.print(delta, end="")
                self._assistant_rendered_length = len(self._assistant_buffer)

    def _close_assistant(self) -> None:
        if self._assistant_open:
            self._flush_assistant(force=True)
            if self._assistant_live is not None:
                self._assistant_live.stop()
                self._assistant_live = None
            self.console.print()
            self._assistant_open = False
            self._assistant_buffer = ""
            self._assistant_pending = ""
            self._assistant_rendered_length = 0
            self._assistant_last_refresh = 0.0

    def handle_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("event_type") or event.get("event") or event.get("type")
        payload = event.get("payload") or {}

        self._update_status_detail(event_type, payload)

        if event_type in _PHASE_BY_EVENT and self._phase != _PHASE_BY_EVENT[event_type]:
            self._phase = _PHASE_BY_EVENT[event_type]
            self._refresh_live()
            if self.live_input_listener is not None and hasattr(self.live_input_listener, "invalidate"):
                self.live_input_listener.invalidate()
        elif self.live_input_listener is not None and hasattr(self.live_input_listener, "invalidate"):
            self.live_input_listener.invalidate()

        if event_type == "user_message":
            content = str(payload.get("content", ""))
            if content:
                self._stop_live()
                self.console.print("[bold green]You[/bold green]")
                if len(content) > _USER_MESSAGE_MAX_DISPLAY_CHARS:
                    shown = content[:_USER_MESSAGE_MAX_DISPLAY_CHARS]
                    self.console.print(Text(shown), end="")
                    self.console.print(
                        f"\n[dim]… pasted message truncated in display "
                        f"({len(content):,} characters; sent in full)[/dim]"
                    )
                else:
                    self.console.print(Text(content), end="")
                self.console.print()
            return

        if event_type == "reasoning_delta":
            content = str(payload.get("content", ""))
            if content:
                now = time.monotonic()
                if self._reasoning_start is None:
                    self._reasoning_start = now
                self._reasoning_last = now
                self._refresh_live()
                if self.debug:
                    self.console.print(f"[dim italic]{content}[/dim italic]", end="")
            return

        if event_type == "assistant_delta":
            content = str(payload.get("content", ""))
            if content and self._reasoning_start is not None and self._thought_seconds is None:
                self._thought_seconds = (self._reasoning_last or self._reasoning_start) - self._reasoning_start
            if not self._assistant_open:
                self._stop_live()
                if not self._assistant_header_shown:
                    self.console.print("[bold cyan]Assistant[/bold cyan]")
                    self._assistant_header_shown = True
                self._assistant_open = True
            # Whitespace/reasoning-only provider frames are not a visible
            # final answer. Marking them as displayed suppresses cli.py's
            # authoritative persisted-summary fallback and leaves a blank
            # terminal even though the task completed with real text.
            if content.strip():
                self.streamed_final_text = True
            self._record_tokens(content)
            self._refresh_live()
            self._assistant_pending += content
            self._flush_assistant()
            return

        if event_type == "plan_step_progress":
            # Update both the live view and durable scrollback. The live view
            # may already have been stopped when assistant output began, so
            # refreshing it alone makes progress appear to disappear.
            self._close_assistant()
            items = payload.get("items") if isinstance(payload.get("items"), list) else []
            if items:
                self._plan_steps = [item for item in items if isinstance(item, dict) and item.get("status") != "context"]
                self._refresh_live()
                self._print_plan_snapshot(self._plan_steps, title=payload.get("title") or "Plan progress")
            return

        if event_type == "plan_created":
            self._close_assistant()
            stage = payload.get("stage")
            content = str(payload.get("content", ""))
            items = payload.get("items") if isinstance(payload.get("items"), list) else []
            if items:
                self._plan_steps = [item for item in items if isinstance(item, dict) and item.get("status") != "context"]
                self._refresh_live()
                # Rich's TTY Live region is transient and is stopped when
                # assistant output begins. Always print a durable snapshot;
                # otherwise the plan disappears at execution start.
                assumptions = payload.get("assumptions") or []
                risks = payload.get("risks") or []
                self._print_plan_snapshot(
                    self._plan_steps, title=payload.get("title") or "Execution plan",
                    assumptions=assumptions, risks=risks,
                )
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
            result = result_envelope
            args = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {}
            if not args and isinstance(result, dict):
                args = {
                    "path": result.get("resolved_path") or result.get("path"),
                    "command": result.get("command"),
                }
            label = _tool_action_label(tool, args, completed=True)
            _render_result_block(self.console, ok=not failed, label=label, content=content)
            return

        if event_type in (
            "artifact_generated", "file_generated", "image_generated",
            "video_generated", "diff_available",
        ):
            self._close_assistant()
            filename = payload.get("filename") or payload.get("name") or "generated file"
            size = payload.get("size_bytes")
            size_text = f" ({size} bytes)" if size else ""
            url = (
                payload.get("download_url") or payload.get("file_url")
                or payload.get("image_url") or payload.get("video_url") or payload.get("url")
            )
            body = f"{filename}{size_text}"
            if url:
                body += f"\n{url}"
            validated = payload.get("validated")
            kind_title = {
                "image_generated": "Image generated",
                "video_generated": "Video generated",
            }.get(event_type, "File generated")
            title = kind_title + (" · validated" if validated is True else "")
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
            ok = event_type == "command_completed" and exit_code == 0
            body = stdout.strip()
            if stderr.strip():
                body = (body + "\n" + stderr.strip()).strip()
            _render_result_block(self.console, ok=ok, label=f"exit {exit_code}", content=body)
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
            diff_text = payload.get("diff")
            if diff_text:
                print_unified_diff(self.console, str(diff_text), title="Proposed change")
            return

        if event_type == "context_rollover":
            self._close_assistant()
            before = payload.get("before_tokens")
            after = payload.get("after_tokens")
            self.console.print(
                f"[dim]· Internal context checkpointed and continued "
                f"(~{before} → ~{after} estimated tokens) -- same task, still running[/dim]"
            )
            return

        if event_type == "workspace_scope":
            # Fires on every single turn (workspace scope is always
            # computed) -- routine internal bookkeeping with nothing
            # actionable in it on the happy path, so (like context_reused/
            # context_rescanned/model_selected below) it's debug-only. This
            # was the biggest single contributor to feeling "bloated"
            # compared to Claude Code's own clean default: 3+ setup lines
            # printed before the model even starts, every turn, with no way
            # to turn them off.
            self._close_assistant()
            if self.debug:
                self.console.print(f"[dim]· {payload.get('content', '')}[/dim]")
            return

        if event_type in ("context_reused", "context_rescanned"):
            self._close_assistant()
            if self.debug:
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
            # FIX: an empty resolved model (Tier IV/NIM routes leave
            # config.default_model blank by design, letting the provider
            # pick its own default) previously showed as "Model: unknown"
            # in --debug output -- misleading, since nothing is actually
            # unknown here; the provider is just resolving its own default.
            model = payload.get("model") or "(provider default)"
            reason = payload.get("selection_reason") or "explicit selection or orchestration routing"
            if self.debug:
                self.console.print(f"[dim]· Provider: {provider} · Model: {model} · {reason}[/dim]")
            else:
                # Persist the authoritative route in the scrollback. The
                # previous debug-only display made "local:auto" in the
                # banner look like local Ollama even when AUTO had selected
                # NVIDIA/OpenRouter/HF; users could not tell which provider
                # was actually responsible for a slow or bad response.
                self.console.print(f"[dim]· Using {provider} · {model}[/dim]")
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


def suspend_live_if_active(renderer: Any) -> None:
    """Call renderer.suspend_live() if the renderer supports it.

    Renderer test doubles (recording stubs used across the test suite)
    don't implement the live-status protocol at all -- this lets callers
    (runner.py, runner_local.py) unconditionally suspend/resume around an
    approval prompt without every such double needing the method.
    """
    method = getattr(renderer, "suspend_live", None)
    if callable(method):
        method()


def resume_live_if_active(renderer: Any) -> None:
    method = getattr(renderer, "resume_live", None)
    if callable(method):
        method()


def print_banner(console: Console, *, host: str, workspace_root: str, mode: str, approval_policy: str) -> None:
    console.print(Text("TamfisGPT Code", style="bold cyan"))
    console.print(Text("by Tamfis Nig. Ltd", style="dim"))
    console.print(f"[dim]Workspace:[/dim] {workspace_root}")
    if host.startswith("local:"):
        route = host.split(":", 1)[1] or "auto"
        route_label = "auto (nvidia, openrouter, hf, in capability-ranked order)" if route == "auto" else route
        console.print(
            f"[dim]Mode:[/dim] {mode}   [dim]Approval:[/dim] {approval_policy}   "
            f"[dim]Runtime:[/dim] standalone   [dim]Provider:[/dim] {route_label}"
        )
    else:
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


def print_resume_plan_status(console: Console, state: Any) -> None:
    """Show the resumed session's active saved plan and its step progress.

    Before this, both `tamfis-code resume` and the REPL's `/resume` showed
    only a conversation summary -- a plan left mid-execution (some steps
    completed, one in_progress or failed, others still pending) became
    completely invisible the moment a session was resumed, even though
    state.py has carried this real, up-to-date step-status data (see
    orchestrator/engine.py's _advance_plan_step) since the fix for #11.
    A no-op if there's no saved plan, or the saved plan has nothing left
    outstanding (every step already completed).
    """
    if not state.saved_plans:
        return
    plan = next(
        (p for p in reversed(state.saved_plans) if p.get("id") == state.active_plan_id),
        state.saved_plans[-1],
    )
    steps = plan.get("steps") or []
    if not steps or all(step.get("status") == "completed" for step in steps):
        return
    no_color = bool(getattr(console, "no_color", False))
    objective = plan.get("objective") or "no objective recorded"
    console.print(f"Plan in progress ({objective}):" if no_color else f"[bold cyan]Plan in progress[/bold cyan] ({objective}):")
    markers = {"completed": "✓", "in_progress": "◉", "failed": "✗"}
    colors = {"completed": "green", "in_progress": "yellow", "failed": "red"}
    for step in steps:
        status = step.get("status", "pending")
        glyph = markers.get(status, "○")
        color = colors.get(status)
        marker = glyph if no_color or color is None else f"[{color}]{glyph}[/{color}]"
        console.print(f"  {marker} {step.get('step') or ''}")
