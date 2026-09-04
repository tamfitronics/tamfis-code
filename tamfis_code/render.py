"""Terminal rendering for the Remote event stream.

Mirrors the same event vocabulary and card concepts the web workspace's
RemoteMessageBubble/RemoteSuiteBubble cards render (plan/status lines, tool
call cards, command cards, file cards) -- see
tamfis-frontend/src/workspaces/remote/RemoteMessageBubble.tsx and
docs/REMOTE_AGENT_MASTER_SPEC.md Phase 7/9. Presentation only: no network
calls, no approval decisions -- see runner.py for that.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import re
import textwrap
import time
from typing import Any, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.markup import escape
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.spinner import Spinner
from rich.text import Text

from . import __version__
from .config import load_config
from .metrics import MetricsTracker
from .public_identity import (
    PUBLIC_PROVIDER_NAME,
    public_model_name,
    public_route_name,
    redact_routing_text,
    sanitize_public_event,
)
from .safety import redact_secrets

_TOOL_ANNOUNCE_RE = re.compile(r"Using tool:\s*(.+?)\.\.\.\s*$")

# Which backend actually served a Remote turn (Ollama Cloud, OpenRouter,
# NVIDIA NIM, a raw model id like "glm-5.2:cloud"...) is internal routing
# detail. TamfisGPT is the product; every place that would otherwise print
# a raw provider/model id to the user prints this branded label instead.
BRANDED_PROVIDER_LABEL = PUBLIC_PROVIDER_NAME

# Single source of truth for plan-step glyph/colour, shared by the transient
# live spinner (_build_status) and the durable scrollback snapshot
# (_print_plan_snapshot) -- these used to disagree (the live view showed
# ◉ for in_progress, the permanent snapshot showed ▶), and only the live
# view was coloured, so the same step looked like two different UIs
# depending on which one happened to be drawing it.
# (marker, marker_style, step-text style)
_PLAN_MARKER_BY_STATUS: dict[str, tuple[str, str, str]] = {
    "completed": ("✓", "green", "dim strike"),
    "failed": ("✗", "bold red", "red"),
    "in_progress": ("◉", "bold yellow", "bold"),
    "pending": ("○", "dim", "dim"),
}

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

_ACTIVITY_VARIANTS_BY_PHASE = {
    "idle": ("Working", "Razzmatazzing", "Smoothing"),
    "submitting": ("Submitting", "Packing", "Dispatching"),
    "queued": ("Queuing", "Lining up", "Preparing"),
    "understand": ("Analyzing", "Orienting", "Evaluating"),
    "inspect": ("Inspecting", "Tracing", "Rummaging"),
    "route": ("Routing", "Selecting", "Calibrating"),
    "reasoning": ("Reasoning", "Evaluating", "Synthesizing"),
    "respond": ("Composing", "Smoothing", "Summarizing"),
    "plan": ("Planning", "Sequencing", "Plotting"),
    "execute": ("Coding", "Wiring", "Polishing"),
    "observe": ("Observing", "Reviewing", "Measuring"),
    "repair": ("Repairing", "Untangling", "Mending"),
    "waiting_for_approval": ("Waiting", "Holding", "Standing by"),
    "validate": ("Evaluating", "Checking", "Verifying"),
    "report": ("Wrapping up", "Summarizing", "Finishing"),
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
    "Tip: `/compact` in the REPL compresses the thread (folds older turns into the summary, keeps recent ones) so a long session stays light.",
    "Tip: `/summary` in the REPL shows a structured recap of the conversation so far without compressing it.",
    "Tip: `/cd <path>` in the REPL re-orients the session to a different directory without restarting.",
    "Tip: `tamfis-code tools list` shows every tool this agent knows how to call.",
    "Tip: `--approval` (or `/mode`) controls how much gets auto-approved: ask/safe/full-auto/never/...",
    "Tip: While a task is running in `tamfis-code`, just type and press Enter -- it queues as a follow-up without interrupting the current work.",
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
# the runner/checkpoint; Rich only ever redraws a bounded recent tail in
# place, and everything before that tail is committed to real terminal
# scrollback (via Live.console.print) as soon as it's stable. A Live region
# redraws by moving the cursor up N lines and repainting every refresh --
# once N exceeds the terminal height that cursor math breaks (most
# terminals can't scroll-and-reposition mid-redraw), which read as the UI
# freezing and refusing to scroll on long answers. Keeping the live region
# itself short-and-bounded, with older text pushed into normal scrollback,
# is what lets the terminal auto-scroll the way Claude Code/Codex do.
_ASSISTANT_LIVE_TAIL_CHARS = 2_000
_ASSISTANT_SENTENCE_BOUNDARY_RE = re.compile(r"(?:[.!?](?:[\"'’)]*)\s+|\n{2,}|```\s*$)")
# Enclosing box width for the non-Live rendering path (see
# _print_box_top/_print_box_line): capped well below typical wide terminals
# so a long reply still reads as one message card, not a border stretched
# edge-to-edge.
_ASSISTANT_BOX_MAX_WIDTH = 100
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
        # search_code is the real, currently-registered local tool
        # (mcp.py) -- glob_files/search_files/grep_files/remote_exec/
        # run_command are kept here only because an external tool
        # reachable through the shared MCP bridge's own naming convention
        # (see mcp.py's call_tool fallback to _get_shared_mcp_bridge) might
        # still use them; they are not names this codebase's own local
        # engine emits.
        "search_code": ("Searching repository contents", "Searched repository contents"),
        "find_references": ("Finding references", "Found references"),
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


_READ_ONLY_TOOLS = {
    "read_file", "search_code", "find_references", "get_git_info", "read_background_job",
    "glob_files", "search_files", "grep_files", "list_directory",
}
_MUTATION_TOOLS = {"write_file", "edit_file", "file_edit", "create_file", "update_file"}

_DOCUMENTATION_SUFFIXES = {".md", ".mdx", ".rst", ".adoc", ".txt"}
_PROMPT_NAME_MARKERS = {"prompt", "prompts", "instruction", "instructions", "system_message"}


def _normalized_tool_name(name: str) -> str:
    return (name or "tool").strip().lower().replace("-", "_").rsplit("/", 1)[-1]


def _is_read_only_tool(name: str) -> bool:
    return _normalized_tool_name(name) in _READ_ONLY_TOOLS


def _is_mutation_tool(name: str) -> bool:
    return _normalized_tool_name(name) in _MUTATION_TOOLS


def _change_kind(path: Any) -> str:
    """Human label for a default-collapsed file mutation card."""
    value = str(path or "")
    lowered = value.lower()
    filename = lowered.rsplit("/", 1)[-1]
    stem = filename.rsplit(".", 1)[0]
    if any(marker in stem for marker in _PROMPT_NAME_MARKERS):
        return "Prompt updated"
    suffix = "." + filename.rsplit(".", 1)[-1] if "." in filename else ""
    if suffix in _DOCUMENTATION_SUFFIXES or filename in {"readme", "license", "changelog"}:
        return "Documentation updated"
    return "Code updated"


# Category label (gerund, singular noun, plural noun) for the round-summary
# line -- "Searching for 1 pattern, reading 2 files…" -- keyed by the same
# normalized tool names as everything else in this module. Tools with no
# entry here (retrieve_evidence, ask_user_question, ...) are simply not
# counted; they're not the kind of visible repository action this line is
# for.
_TOOL_CATEGORY = {
    "search_code": ("Searching for", "pattern", "patterns"),
    "grep_files": ("Searching for", "pattern", "patterns"),
    "search_files": ("Searching for", "pattern", "patterns"),
    "find_references": ("Finding references for", "symbol", "symbols"),
    "read_file": ("Reading", "file", "files"),
    "list_directory": ("Listing", "directory", "directories"),
    "glob_files": ("Finding", "file", "files"),
    "execute_command": ("Running", "shell command", "shell commands"),
    "remote_exec": ("Running", "shell command", "shell commands"),
    "run_command": ("Running", "shell command", "shell commands"),
    "read_background_job": ("Checking on", "background job", "background jobs"),
    "write_file": ("Writing", "file", "files"),
    "edit_file": ("Editing", "file", "files"),
    "extract_archive": ("Extracting", "archive", "archives"),
    "repackage_archive": ("Repackaging", "archive", "archives"),
    "create_artifact": ("Creating", "artifact", "artifacts"),
    "inspect_artifact": ("Inspecting", "artifact", "artifacts"),
    "get_git_info": ("Reading", "Git repository", "Git repositories"),
    "web_search": ("Searching", "web query", "web queries"),
}


def _round_activity_summary(counts: dict[str, int]) -> str:
    """The aggregated "Searching for 1 pattern, reading 1 file, running 1
    shell command…" line, built from a {normalized_tool_name: count} tally
    -- see StreamRenderer._round_tool_counts. Renders in first-used order
    (dict insertion order), not alphabetically, so the summary reads in the
    same sequence the actions actually happened."""
    parts = []
    for name, count in counts.items():
        if count <= 0:
            continue
        category = _TOOL_CATEGORY.get(name)
        if category is None:
            continue
        verb, singular, plural = category
        noun = singular if count == 1 else plural
        parts.append(f"{verb} {count} {noun}")
    if not parts:
        return ""
    # Lowercase every part after the first so they read as one sentence
    # ("Searching for 1 pattern, reading 1 file…") rather than a list of
    # separately-capitalized clauses.
    parts = [parts[0], *(p[0].lower() + p[1:] for p in parts[1:])]
    return ", ".join(parts) + "…"


def _read_target(arguments: Optional[dict[str, Any]]) -> str:
    arguments = arguments or {}
    return str(
        arguments.get("path") or arguments.get("file_path")
        or arguments.get("pattern") or arguments.get("query") or "workspace"
    ).strip()


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
    # `label` embeds raw tool arguments (grep patterns, paths, commands) --
    # e.g. a pattern like "[0-9]+" -- and `content` is raw tool/command
    # output. Both are untrusted-for-markup: printed through an f-string
    # console.print (which parses Rich markup/highlights by default) they
    # get silently mangled (bracket runs read as style tags, ReprHighlighter
    # recolours "words" inside them and can drop text). Building explicit
    # Text objects and disabling the console's syntax highlighter for them
    # keeps arbitrary tool output showing exactly what it is.
    header = Text(f"{glyph} ", style=style)
    header.append(label)
    console.print(header, highlight=False)
    body = content.strip("\n")
    if not body:
        return
    lines = _bounded_preview(body).split("\n")
    if len(lines) > _RESULT_BLOCK_MAX_LINES:
        omitted = len(lines) - _RESULT_BLOCK_MAX_LINES
        lines = lines[:_RESULT_BLOCK_MAX_LINES] + [f"… {omitted} more line{'s' if omitted != 1 else ''} …"]
    for line in lines:
        console.print(Text(f"  {line}", style="dim"), highlight=False)


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
    if payload.get("provider") or payload.get("model"):
        parts.append(public_route_name(payload.get("provider"), payload.get("model")))
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
        # Enclosing box state for the manual (non-Rich-Live) rendering path:
        # a real TTY with live_input_listener active, where Rich Live can't
        # be used (see _flush_assistant) so box lines are printed directly.
        self._box_open = False
        self._assistant_line_buffer = ""
        self._tool_names_by_call_id: dict[str, str] = {}
        self._selected_provider: Optional[str] = None
        self._announced_route: Optional[tuple[str, str]] = None
        # Model displayed in the persistent PTY/TTY footer. It must be
        # initialised even before routing emits a model_selected event.
        self._model: Optional[str] = None
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
        # A shell command currently executing, shown as its own live line
        # (see _build_status) with its own elapsed timer -- distinct from
        # _status_detail (set but, before this, never actually rendered
        # anywhere) and from the overall task elapsed time, since a single
        # long-running command inside a multi-step turn needs its own clock.
        self._running_command: Optional[str] = None
        self._running_command_started: Optional[float] = None
        # Tally of tool calls issued so far THIS TASK, for the aggregated
        # "Searching for 1 pattern, reading 1 file…" summary line -- a
        # cumulative running total across every round of the current turn,
        # not just the latest round (a model that issues one tool call per
        # round, the common case, would otherwise almost always show a
        # single-item summary, defeating the point of aggregating at all).
        # Reset on task_submitting/task_started (see _update_status_detail),
        # not on completion, so it stays visible through _finalize's summary
        # line rather than disappearing right before the task's own status.
        # dict, not Counter, to preserve first-used insertion order for
        # _round_activity_summary's rendering order.
        self._round_tool_counts: dict[str, int] = {}
        # Set by live_input.py's Ctrl+B keybinding, read by runner_local.py's
        # tool-dispatch loop (see mcp.py's _execute_command background_signal
        # param) -- an asyncio.Event rather than a plain bool since
        # _execute_command actually awaits it (asyncio.wait alongside the
        # command's own completion), not just polls it.
        self.background_requested = asyncio.Event()
        # The live composer increments this whenever the user submits a
        # steering message. Provider streaming watches the signal so the
        # update reaches the active turn without waiting for a long response
        # to finish first.
        self.steering_requested = asyncio.Event()
        self._steering_revision = 0
        self._steering_handled_revision = 0
        self._plan_steps: list[dict[str, Any]] = []
        self._task_start = time.monotonic()
        try:
            cost_cap = load_config().session_cost_cap_usd
        except Exception:
            # Config loading must never be able to break rendering --
            # falls back to the dataclass default rather than disabling
            # the warning outright, so a config-load hiccup doesn't
            # silently turn off a safety rail either.
            cost_cap = 5.0
        self._metrics = MetricsTracker(cost_cap_usd=cost_cap)
        # Real reasoning-phase timing (see provider_protocols.py's
        # reasoning_content extraction) -- None/None until a reasoning delta
        # actually arrives; _thought_seconds freezes once real answer
        # content starts, and stays visible in the status line for the rest
        # of the turn the way Claude Code's own "thought for Xs" does.
        self._reasoning_start: Optional[float] = None
        self._reasoning_last: Optional[float] = None
        self._thought_seconds: Optional[float] = None
        self._terminal_status: Optional[str] = None
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
        # How much of _assistant_buffer has already been committed to real
        # scrollback (printed through the Live's own console, which pauses
        # the live redraw, prints a permanent line, then resumes). The Live
        # region itself only ever shows the remainder -- see _flush_assistant.
        self._assistant_scrolled_length = 0
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

    def request_steering(self) -> None:
        """Wake the active provider stream for a new user direction."""
        self._steering_revision += 1
        self.steering_requested.set()

    def steering_revision(self) -> int:
        return self._steering_revision

    def acknowledge_steering(self, revision: Optional[int] = None) -> None:
        """Mark all messages claimed at the current safe boundary."""
        handled = self._steering_revision if revision is None else revision
        self._steering_handled_revision = max(self._steering_handled_revision, handled)
        if not self.has_pending_steering():
            self.steering_requested.clear()

    def has_pending_steering(self) -> bool:
        return self._steering_revision > self._steering_handled_revision

    async def wait_for_steering(self) -> None:
        while not self.has_pending_steering():
            await self.steering_requested.wait()

    def live_input_activity_line(self) -> Optional[str]:
        """Plain-text "Searching for N patterns, reading M files…" summary,
        for prompt-toolkit's bottom toolbar (see live_input.py's
        _bottom_toolbar). This is the same tally _build_status renders above
        its Rich spinner, but that Live region is suspended for the whole
        interactive REPL (Rich Live and prompt-toolkit fight over the same
        rows -- see _flush_assistant's box-drawing fallback), so without this
        the activity tally never reaches the ordinary interactive session at
        all, only the no-listener path (`tamfis-code agent`/CI-style runs).
        """
        activity_summary = _round_activity_summary(self._round_tool_counts)
        if self._running_command:
            command_elapsed = _format_elapsed(
                time.monotonic() - (self._running_command_started or time.monotonic())
            )
            preview = self._running_command[:120]
            command_line = f"⎿  $ {preview} ({command_elapsed})"
            return f"{activity_summary}  {command_line}" if activity_summary else command_line
        return activity_summary or None

    def live_input_status(self, spinner_frame: str = "") -> str:
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
        if self._terminal_status is not None:
            model = self._model or "auto"
            return f"{self._terminal_status} · {model} · {details}"
        activities = _ACTIVITY_VARIANTS_BY_PHASE.get(
            self._phase,
            (_VERB_BY_PHASE.get(self._phase, self._phase).capitalize(),),
        )
        activity_index = int((time.monotonic() - self._task_start) // 2) % len(activities)
        verb = activities[activity_index]
        model = self._model or "auto"
        prefix = f"{spinner_frame} " if spinner_frame else ""
        return f"{prefix}{verb}… · {model} · {details}"

    def print_work_summary(self, status: str = "completed") -> None:
        """Leave one Claude-style durable timing line after live UI exits."""
        elapsed = _format_elapsed(time.monotonic() - self._task_start)
        model = self._model or "auto"
        if status == "completed":
            marker, label, style = "✻", "Worked for", "dim"
        elif status in {"cancelled", "exited"}:
            marker, label, style = "■", "Stopped after", "yellow"
        else:
            marker, label, style = "✗", "Stopped after", "red"
        self.console.print(
            Text(f"{marker} {label} {elapsed} · {model}", style=style),
            highlight=False,
        )

    def conclude(self, status: str) -> None:
        """Clear transient activity before the terminal returns to the REPL."""
        self._round_tool_counts = {}
        self._running_command = None
        self._running_command_started = None
        self._terminal_status = (
            "Completed" if status == "completed"
            else "Stopped" if status in {"cancelled", "exited"}
            else "Failed"
        )
        self._refresh_live()
        if self.live_input_listener is not None and hasattr(self.live_input_listener, "invalidate"):
            self.live_input_listener.invalidate()

    def _update_status_detail(self, event_type: str, payload: dict[str, Any]) -> None:
        """Turn structured stream events into human-readable footer detail."""
        if event_type in {"task_submitting", "task_submitted"}:
            self._status_detail = "Submitting the task"
        elif event_type in {"task_started", "context_loading"}:
            self._status_detail = "Loading workspace context"
            # task_started fires once at the very start of a turn (see
            # runner_local.py) -- task_submitting/task_submitted, checked
            # above, are dead: only the retired Remote engine ever sent
            # them. This is the real point at which a fresh tool-usage
            # tally should begin.
            self._round_tool_counts = {}
            self._announced_route = None
            self._terminal_status = None
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
            normalized_name = _normalized_tool_name(name)
            if normalized_name in _TOOL_CATEGORY:
                self._round_tool_counts[normalized_name] = self._round_tool_counts.get(normalized_name, 0) + 1
            # execute_command gets its own live line with its own elapsed
            # timer (see _build_status) -- the local engine's actual event
            # vocabulary (this and tool_output below) never emits the
            # separate command_started/command_completed/command_failed
            # events this used to key off of, which only the retired Remote
            # engine ever sent.
            if _normalized_tool_name(name) == "execute_command":
                self._running_command = str(arguments.get("command") or "the requested command")
                self._running_command_started = time.monotonic()
        elif event_type == "tool_output":
            self._status_detail = "Reviewing the tool result"
            self._running_command = None
            self._running_command_started = None
        elif event_type == "file_mutation":
            path = str(payload.get("path") or payload.get("resolved_path") or "the requested file")
            self._status_detail = f"Applying the change to {path[:120]}"
        elif event_type == "approval_required":
            self._status_detail = "Waiting for your approval"
        elif event_type in {"task_diagnostics", "context_rollover"}:
            self._status_detail = "Checking the work and updating context"
        elif event_type in {"ai_task_completed", "ai_task_failed"}:
            self._status_detail = "Finishing the response"
            self.conclude("completed" if event_type == "ai_task_completed" else "failed")

    def _live_input_title(self) -> str:
        """"Input", or "Input — session N" once other agents are running
        concurrently (e.g. /delegate'd swarm children) -- otherwise a user
        watching several panes has no way to tell which one this box
        belongs to."""
        listener = self.live_input_listener
        if listener is not None and getattr(listener, "_active_agents", 0):
            return f"Input — session {listener.session_id}"
        return "Input"

    def _live_input_hint(self) -> Text:
        """Instructions for the in-task follow-up box, reflecting what Esc
        and Enter actually do right now rather than a fixed caption --
        before this, the text claimed "Esc stops the task" even after a
        stop had already been requested (Esc/Ctrl+C do nothing once
        `interrupt_classification` is set -- see live_input.py's
        `_request_interrupt`), and never mentioned follow-ups already
        queued behind the running turn."""
        listener = self.live_input_listener
        default = Text(
            "Type a message and press Enter · Esc stops the task · Ctrl+C/Ctrl+D exits"
        )
        if listener is None:
            return default
        if getattr(listener, "interrupt_classification", None) is not None:
            return Text("Stopping the task… · Ctrl+C/Ctrl+D exits")
        session_id = getattr(listener, "session_id", None)
        if session_id is None:
            # Test doubles and other non-real listeners: fall back to the
            # static caption rather than guessing at session state.
            return default
        from . import state as local_state

        try:
            session_state = local_state.get_session_state(session_id)
            queued = sum(
                1
                for item in session_state.queued_user_instructions
                if item.get("classification") == "follow_up"
                and item.get("status") == "queued"
            )
        except Exception:
            queued = 0
        parts = ["Type a message and press Enter", "Esc stops the task", "Ctrl+C/Ctrl+D exits"]
        if queued:
            parts.insert(1, f"{queued} steering update{'s' if queued != 1 else ''} pending")
        return Text(" · ".join(parts))

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
        activity_summary = _round_activity_summary(self._round_tool_counts)
        if (
            not self._plan_steps and not tip and self.live_input_listener is None
            and not self._running_command and not activity_summary
        ):
            return self._spinner
        # Above the spinner, not in `lines` below -- matches Claude Code's
        # own layout: the aggregated tally of what's happened so far this
        # turn, and the specific command currently running, both read as
        # a heading above the live verb/elapsed spinner underneath them.
        top_lines = []
        if activity_summary:
            top_lines.append(Text.from_markup(f"[bold]{escape(activity_summary)}[/bold]"))
        if self._running_command:
            command_elapsed = _format_elapsed(
                time.monotonic() - (self._running_command_started or time.monotonic())
            )
            preview = self._running_command[:200]
            top_lines.append(Text.from_markup(
                f"  [dim]⎿  $ {escape(preview)} ({command_elapsed})[/dim]"
            ))
            # Only shown while the live REPL editor (live_input.py) actually
            # owns a Ctrl+B binding that does something -- see
            # LiveInputListener._input_loop -- never a hint for a keypress
            # that would silently do nothing (e.g. a non-interactive `agent`
            # run with no live_input_listener at all).
            if self.live_input_listener is not None:
                top_lines.append(Text.from_markup("     [dim](ctrl+b to run in background)[/dim]"))
        lines = []
        if self.live_input_listener is not None:
            # Keep a real, persistent input box in the live task display. The
            # ordinary REPL editor is suspended while the agent owns the
            # terminal, with the editable follow-up line owned by
            # prompt_toolkit rather than a special control key.
            input_box = Panel(
                self._live_input_hint(),
                title=self._live_input_title(),
                border_style="cyan",
                padding=(0, 1),
            )
            lines.append(input_box)
        if tip:
            lines.append(Text.from_markup(f"  [dim]{tip}[/dim]"))
        for step in self._plan_steps:
            status = str(step.get("status") or "pending")
            marker, marker_style, text_style = _PLAN_MARKER_BY_STATUS.get(status, _PLAN_MARKER_BY_STATUS["pending"])
            # Built with Text.append (styled spans), not Text.from_markup --
            # step text comes from the model's own plan and from_markup
            # would parse any literal "[...]" inside it as a (possibly
            # unintended, possibly broken) style tag instead of visible text.
            line = Text("  ")
            line.append(marker, style=marker_style)
            line.append(" ")
            # No per-step completion event exists yet (see state.py's
            # update_plan_steps docstring) -- statuses beyond the initial
            # plan_created payload are a best-effort approximation, so this
            # is labelled "~" rather than presented as precise.
            if status == "in_progress":
                line.append("~ ", style=marker_style)
            line.append(str(step.get("step") or ""), style=text_style)
            lines.append(line)
        return Group(*top_lines, self._spinner, *lines)

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
        table.add_column(width=4, justify="right")
        table.add_column(ratio=1)
        for index, item in enumerate(items, start=1):
            status = str(item.get("status") or "pending")
            marker, marker_style, text_style = _PLAN_MARKER_BY_STATUS.get(status, _PLAN_MARKER_BY_STATUS["pending"])
            step_text = str(item.get("step") or "")
            table.add_row(
                Text(f"{index}. {marker}", style=marker_style),
                Text(step_text, style=text_style),
            )
        body: list[Any] = [table]
        if assumptions:
            body.extend([Text("Assumptions", style="bold"), Text(" • " + "\n • ".join(map(str, assumptions)))])
        if risks:
            body.extend([Text("Risks", style="bold yellow"), Text(" • " + "\n • ".join(map(str, risks)))])
        self.console.print(Panel(Group(*body), title=Text(title), border_style="cyan", expand=False))

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

    async def suspend_live_async(self) -> None:
        """Like `suspend_live`, but awaits the live-input listener's prompt
        actually releasing the terminal before returning.

        Every approval-gate call site suspends the live UI and then, a few
        synchronous statements later, opens its own PromptSession for the
        y/n decision. `suspend_live`'s `pause()` only *requests* the old
        prompt exit (fire-and-forget) -- if the new PromptSession starts
        before that request is actually processed by the event loop, two
        prompt_toolkit Applications race for the same stdin fd and the new
        one can be starved of keystrokes entirely (the gate renders but
        never responds). Callers that will immediately open a new prompt
        (i.e. every approval gate) must use this instead of `suspend_live`.
        """
        self._stop_live()
        if self._assistant_live is not None:
            self._assistant_live.stop()
            self._assistant_live = None
        if self.live_input_listener is not None:
            await self.live_input_listener.pause_async()

    def resume_live(self) -> None:
        """Restore the active terminal status owner after suspension.

        A running LiveInputListener owns the terminal footer through
        prompt-toolkit's bottom toolbar. Rich Live must not also be started,
        otherwise both systems repaint the same terminal rows and the footer,
        input prompt and spinner overwrite one another.
        """
        if self.live_input_listener is not None:
            self.live_input_listener.resume()
            return

        if self._is_tty and self._live is None and not self._assistant_open:
            self._live = Live(
                self._build_status(),
                console=self.console,
                refresh_per_second=8,
                transient=True,
            )
            self._live.start()

    def _box_content_width(self) -> int:
        try:
            width = int(self.console.width)
        except Exception:
            width = 80
        return max(20, min(width, _ASSISTANT_BOX_MAX_WIDTH))

    def _print_box_top(self, *, title: Optional[str] = None) -> None:
        width = self._box_content_width()
        line = Text()
        if title:
            label = f" {title} "
            right = max(1, width - 3 - len(label))
            line.append("╭─", style="cyan")
            line.append(label, style="bold cyan")
            line.append("─" * right + "╮", style="cyan")
        else:
            line.append("╭" + "─" * (width - 2) + "╮", style="cyan")
        self.console.print(line)

    def _print_box_bottom(self) -> None:
        width = self._box_content_width()
        self.console.print(Text("╰" + "─" * (width - 2) + "╯", style="cyan"))

    def _print_box_line(self, line: str) -> None:
        width = self._box_content_width()
        inner = width - 4
        pieces = textwrap.wrap(line, inner, break_long_words=True, break_on_hyphens=False) if line.strip() else [""]
        for piece in pieces:
            row = Text()
            row.append("│ ", style="cyan")
            row.append(piece.ljust(inner))
            row.append(" │", style="cyan")
            self.console.print(row)

    def _record_tokens(self, content: str) -> None:
        if not content:
            return
        estimated_tokens = max(1, len(content) // _CHARS_PER_TOKEN_ESTIMATE)
        elapsed_ms = (time.monotonic() - self._task_start) * 1000
        self._metrics.record(estimated_tokens, elapsed_ms, model=self._model or "default")
        cost_warning = self._metrics.check_cost_cap()
        if cost_warning:
            self.console.print(f"\n[yellow]{escape(cost_warning)}[/yellow]")

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
            if self._assistant_live is None:
                self._assistant_live = Live(Text(""), console=self.console, refresh_per_second=12)
                self._assistant_live.start()
            unscrolled = self._assistant_buffer[self._assistant_scrolled_length:]
            while len(unscrolled) > _ASSISTANT_LIVE_TAIL_CHARS:
                # Commit everything but a bounded tail to real scrollback.
                # Looped (not a single trim) so one large burst -- force=True
                # flushing a big pending chunk in one go, not just steady
                # per-delta growth -- still converges to a bounded tail
                # instead of leaving a live region taller than the terminal.
                # The newline search only looks in a small window right
                # before the target cut point: searching further back (like
                # the previous "any earlier newline" version did) could
                # under-cut by however far away that newline was, which is
                # exactly how this overshot before.
                target_cut = len(unscrolled) - _ASSISTANT_LIVE_TAIL_CHARS
                nl_idx = unscrolled.rfind("\n", max(0, target_cut - 500), target_cut)
                cut = nl_idx + 1 if nl_idx != -1 else target_cut
                self._assistant_live.console.print(Text(unscrolled[:cut]), end="")
                self._assistant_scrolled_length += cut
                unscrolled = unscrolled[cut:]
            # During streaming, plain Text avoids reparsing the complete
            # Markdown document on every update. This is deliberately a
            # terminal viewport optimization; the canonical response text
            # and checkpoint remain unchanged.
            self._assistant_live.update(Text(unscrolled), refresh=True)
        elif self._is_tty:
            # prompt-toolkit owns the live input rows, so a Rich Live
            # Markdown renderer would corrupt the composer. Keep this block
            # buffered and render one proper Markdown card at its natural
            # boundary instead. Previously we printed each source line in a
            # box; headings, lists, and Markdown tables then appeared as raw
            # `##`, `-`, and `|` characters and wrapped unreadably.
            self._assistant_rendered_length = len(self._assistant_buffer)
        else:
            # Redirected/non-terminal output: keep the original unboxed
            # plain-text stream -- box-drawing characters are noise in a
            # log file or piped output, and there is no terminal to enclose.
            delta = self._assistant_buffer[self._assistant_rendered_length:]
            if delta:
                self.console.print(Text(delta), end="")
                self._assistant_rendered_length = len(self._assistant_buffer)

    def _close_assistant(self) -> None:
        if self._assistant_open:
            self._flush_assistant(force=True)
            rendered_markdown = self._assistant_buffer.strip()
            if self._box_open:
                if self._assistant_line_buffer:
                    self._print_box_line(self._assistant_line_buffer)
                    self._assistant_line_buffer = ""
                self._print_box_bottom()
                self._box_open = False
            if self._is_tty and self.live_input_listener is not None and rendered_markdown:
                self.console.print(Panel(
                    Markdown(rendered_markdown),
                    title="Assistant",
                    border_style="cyan",
                    expand=False,
                    padding=(0, 1),
                ))
            if self._assistant_live is not None:
                self._assistant_live.stop()
                self._assistant_live = None
            self.console.print()
            self._assistant_open = False
            self._assistant_buffer = ""
            self._assistant_pending = ""
            self._assistant_rendered_length = 0
            self._assistant_last_refresh = 0.0
            self._assistant_scrolled_length = 0

    def _print_task_failure(self, payload: dict[str, Any]) -> None:
        """Render ai_task_failed as a structured card instead of one long
        wrapped red line -- a comma-joined path list (e.g. workspace-grant
        denials) reads as noise packed into a sentence at terminal width,
        not a message a user can act on at a glance."""
        error = str(payload.get("error") or "unknown error")
        denied = [str(item) for item in payload.get("denied_targets") or []]
        allowed = [str(item) for item in payload.get("allowed_roots") or []]
        if not denied and not allowed:
            self.console.print(f"[bold red]Task failed:[/bold red] {escape(error)}")
            return
        # The denied/allowed paths are rendered as real bullet lists below,
        # so the headline is deliberately just the sentence explaining what
        # happened -- repeating the paths inline here too would duplicate
        # exactly what the bullets already say.
        headline = "Requested target is outside the active workspace grant." if denied and allowed else error.split(". ", 1)[0]
        body: list[Any] = [Text(headline)]
        if denied:
            body.append(Text(""))
            body.append(Text("Requested targets (outside the grant):"))
            body.extend(Text(f"  · {path}") for path in denied)
        if allowed:
            body.append(Text(""))
            body.append(Text("Active workspace roots:"))
            body.extend(Text(f"  · {path}") for path in allowed)
        body.append(Text(""))
        body.append(Text("Restart from the target directory, or run `tamfis-code workspace add PATH`.", style="dim"))
        self.console.print(Panel(Group(*body), title="Task failed", border_style="red", expand=False))

    def handle_event(self, event: dict[str, Any]) -> None:
        event = sanitize_public_event(event)
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
                    self.console.print(f"[dim italic]{escape(content)}[/dim italic]", end="")
            return

        if event_type == "assistant_delta":
            content = str(payload.get("content", ""))
            if content and self._reasoning_start is not None and self._thought_seconds is None:
                self._thought_seconds = (self._reasoning_last or self._reasoning_start) - self._reasoning_start
            if not self._assistant_open:
                self._stop_live()
                use_box = self._is_tty and self.live_input_listener is not None
                if use_box:
                    self._print_box_top(title="Assistant" if not self._assistant_header_shown else None)
                    self._box_open = True
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
            # Step markers update in place inside the live view, matching
            # engine.py's _sync_plan_progress contract: "no banner reprint,
            # no spinner phase change, every round" -- this fires on every
            # advanced/edited/added step, so printing a fresh durable panel
            # here (like plan_created does) would spam scrollback with a
            # near-duplicate of the same plan on every round.
            # Only fall back to a durable print when the live view is not
            # currently on screen (self._live is None -- e.g. assistant text
            # already stopped it), since refreshing a stopped Live is a
            # no-op and the update would otherwise be silently lost.
            self._close_assistant()
            items = payload.get("items") if isinstance(payload.get("items"), list) else []
            if items:
                self._plan_steps = [item for item in items if isinstance(item, dict) and item.get("status") != "context"]
                if self._live is not None:
                    self._refresh_live()
                else:
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
                if _is_read_only_tool(tool):
                    if self._is_tty and self.live_input_listener is not None:
                        return
                    self.console.print(f"[dim]Reading {escape(_read_target(args))}[/dim]")
                    return
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
                    content = f"Executing with {BRANDED_PROVIDER_LABEL}..."
                self.console.print(f"[dim]· {escape(content)}[/dim]")
            return

        if event_type == "tool_call_requested":
            self._close_assistant()
            name = str(payload.get("name") or payload.get("tool") or "tool")
            args = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {}
            if _is_read_only_tool(name):
                # The live footer already aggregates routine reads/searches.
                # Avoid leaving one permanent scrollback line per file while
                # preserving explicit lines in redirected/non-interactive
                # logs and preserving durable read failures below.
                if self._is_tty and self.live_input_listener is not None:
                    return
                self.console.print(f"[dim]Reading {escape(_read_target(args))}[/dim]")
                return
            # On an interactive terminal the persistent live status line is
            # the progress indicator.  Printing a second arrow line for an
            # edit/write leaves duplicate history behind; the durable
            # file_mutation event below is the single completed summary.
            if _is_mutation_tool(name) and self._is_tty:
                return
            self.console.print(f"[bold yellow]→ {escape(_tool_action_label(name, args))}[/bold yellow]")
            return

        if event_type == "tool_output":
            self._close_assistant()
            tool = str(payload.get("tool", "tool"))
            result_envelope = payload.get("result") if isinstance(payload.get("result"), dict) else payload
            if _is_read_only_tool(tool):
                success = result_envelope.get("success")
                if success is False or result_envelope.get("status") in {"failed", "error"}:
                    args = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else result_envelope
                    reason, _failed = _tool_result_message(payload)
                    self.console.print(
                        f"[red]Read failed[/red] {escape(_read_target(args))}: {escape(reason)}"
                    )
                return
            # A successful edit/write is represented by the compact
            # file_mutation card. Rendering the tool envelope too can expose
            # the entire inserted file and makes one change appear twice.
            # Failures still render in full because they are actionable.
            if _is_mutation_tool(tool) and self._is_tty:
                _content, mutation_failed = _tool_result_message(payload)
                if not mutation_failed:
                    return
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
            if failed and _normalized_tool_name(tool) in {"execute_command", "run_command", "remote_exec"}:
                label = "Command failed"
            if failed and _normalized_tool_name(tool) == "edit_file":
                label = "Edit failed"
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
            body = f"{escape(str(filename))}{size_text}"
            if url:
                body += f"\n{escape(str(url))}"
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
            label = _change_kind(path)
            self.console.print(
                f"[bold blue]▸ {escape(label)}[/bold blue] · {escape(str(path))}  "
                f"[dim]+{added}/-{removed} · /diff {escape(str(mutation_id))} to expand · "
                f"/revert {escape(str(mutation_id))}[/dim]"
            )
            return

        if event_type == "file_mutation_reverted":
            self._close_assistant()
            self.console.print(f"[green]↺ Reverted {escape(str(payload.get('path', '?')))}[/green]")
            return

        if event_type == "command_started":
            self._close_assistant()
            self.console.print(f"[bold]$[/bold] {escape(str(payload.get('command', '')))}")
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
            # Final secrecy boundary: remote and grouped approval events may
            # arrive without having passed through the local runner's copy.
            command_text = _bounded_preview(redact_secrets(str(text or '')))
            # This is the one screen where display accuracy is a safety
            # property, not just cosmetics: a human approves/rejects based
            # on what's shown here. Panel's body/title are markup-parsed by
            # default, so a real shell command containing brackets --
            # `grep -E '[a-z]+' file`, `echo ${arr[0]}` -- could previously
            # render mangled or truncated, meaning the text the user
            # approved was NOT what actually ran. escape() keeps it literal.
            body = (
                f"Command:\n{escape(command_text)}\n\n"
                f"Working directory:\n{escape(cwd)}\n\n"
                f"Reason:\n{escape(reason)}\n\n"
                f"Risk:\n{escape(str(risk))}"
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
                    title=f"Approval required — risk: {escape(str(risk))}" + (f" (id {command_id})" if command_id is not None else ""),
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
                self.console.print(f"[dim]· {escape(str(payload.get('content', '')))}[/dim]")
            return

        if event_type in ("context_reused", "context_rescanned"):
            self._close_assistant()
            if self.debug:
                reason = payload.get("reason", "unknown")
                if event_type == "context_reused":
                    self.console.print("[dim]· Reusing workspace context — repository unchanged since last turn[/dim]")
                else:
                    self.console.print(f"[dim]· Workspace rescanned (reason: {escape(str(reason))})[/dim]")
            return

        if event_type == "task_diagnostics":
            self._close_assistant()
            self.console.print(f"[dim]· {escape(_format_diagnostics_line(payload))}[/dim]")
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
            self._model = public_model_name(model)
            route = (str(provider), str(model))
            if self._announced_route == route:
                return
            self._announced_route = route
            reason = payload.get("selection_reason") or "explicit selection or orchestration routing"
            if self.debug:
                self.console.print(
                    f"[dim]· Model: {escape(public_model_name(model))} · "
                    f"{escape(redact_routing_text(reason))}[/dim]"
                )
            else:
                # Persist the authoritative route in the scrollback -- users
                # need to know a route was resolved -- but never the raw
                # backend/model id behind it; TamfisGPT owns that identity.
                self.console.print(f"[dim]· Using {BRANDED_PROVIDER_LABEL}[/dim]")
            return

        if event_type == "ai_task_failed":
            self._close_assistant()
            self._print_task_failure(payload)
            return

        if event_type in ("ai_task_completed", "assistant_message", "task_cancelled", "heartbeat", "stream_closed"):
            return  # runner.py owns lifecycle decisions for these; nothing new to print

        # Unrecognised event type: show it plainly rather than silently
        # dropping it -- a gap in this renderer should be visible, not hidden.
        self._close_assistant()
        content = str(payload.get("content") or payload.get("status") or "")
        if content:
            self.console.print(f"[dim]· {escape(str(event_type))}: {escape(content)}[/dim]")
        if self.debug:
            self.console.print(json.dumps(event, indent=2, default=str, ensure_ascii=False))

    def finish(self) -> None:
        self._close_assistant()
        if self._live is not None:
            self._live.stop()
            self._live = None


class StructuredRenderer:
    """Machine-readable event renderer for CI, editors, and orchestration."""

    def __init__(self, *, mode: str = "jsonl", stream: Any = None):
        self.mode = mode
        self.stream = stream or sys.stdout
        self.events: list[dict[str, Any]] = []
        self.background_requested = asyncio.Event()
        self.streamed_final_text = False

    def handle_event(self, event: dict[str, Any]) -> None:
        event = sanitize_public_event(event)
        event_type = event.get("event_type") or event.get("event") or event.get("type")
        if event_type == "assistant_delta" and (event.get("payload") or {}).get("content"):
            self.streamed_final_text = True
        clean = json.loads(json.dumps(event, default=str, ensure_ascii=False))
        if self.mode == "jsonl":
            self.stream.write(json.dumps(clean, ensure_ascii=False) + "\n")
            self.stream.flush()
        else:
            self.events.append(clean)

    def record_outcome(self, outcome: Any) -> None:
        self.handle_event({
            "event_type": "outcome",
            "payload": {
                "status": getattr(outcome, "status", "unknown"),
                "summary": getattr(outcome, "summary", None),
                "error": getattr(outcome, "error", None),
            },
        })

    def finish(self) -> None:
        if self.mode == "json":
            self.stream.write(json.dumps({"events": self.events}, ensure_ascii=False) + "\n")
            self.stream.flush()


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


async def suspend_live_async_if_active(renderer: Any) -> None:
    """Async counterpart of `suspend_live_if_active` -- use this at every
    approval-gate call site (a new PromptSession opens right after), so the
    just-paused live-input listener has actually released the terminal
    before that new prompt starts reading stdin. See
    `StreamRenderer.suspend_live_async` for why this matters."""
    method = getattr(renderer, "suspend_live_async", None)
    if callable(method):
        await method()
        return
    # Test doubles / renderers without the async protocol: fall back to the
    # sync one so approval flow still works, just without the stronger
    # ordering guarantee (matches suspend_live_if_active's existing
    # best-effort behaviour for such doubles).
    suspend_live_if_active(renderer)


def print_banner(console: Console, *, host: str, workspace_root: str, mode: str, approval_policy: str) -> None:
    console.print(Text(f"TamfisGPT Code v{__version__}", style="bold cyan"))
    console.print(Text("by Tamfis Nig. Ltd", style="dim"))
    console.print(f"[dim]Workspace:[/dim] {escape(workspace_root)}")
    if host.startswith("local:"):
        route = host.split(":", 1)[1] or "auto"
        route_label = public_route_name(route)
        console.print(
            f"[dim]Mode:[/dim] {mode}   [dim]Approval:[/dim] {approval_policy}   "
            f"[dim]Runtime:[/dim] standalone   [dim]Model:[/dim] {route_label}"
        )
    else:
        console.print(f"[dim]Mode:[/dim] {escape(mode)}   [dim]Approval:[/dim] {escape(approval_policy)}   [dim]Host:[/dim] {escape(host)}")


def print_error(console: Console, message: str) -> None:
    console.print(f"[bold red]Error:[/bold red] {escape(redact_routing_text(message))}")


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
            console.print(f"[dim]›[/dim] {escape(str(turn['objective']))}")
        if turn["answer"]:
            console.print(f"  {escape(str(turn['answer']))}")
        console.print()


def print_unified_diff(console: Console, diff_text: str, *, title: str = "Changes") -> None:
    """Render a unified diff as a bordered card (file path as the panel
    title, +/-/@@ lines coloured), matching how plan/approval/failure
    output is already boxed elsewhere in this renderer -- a diff printed
    as bare scrolling lines with no border was the one card type in this
    file that didn't visually read as "a card" the way Claude Code's own
    file-edit diffs do."""
    no_color = bool(getattr(console, "no_color", False))
    if not diff_text.strip():
        body: Any = Text("(empty diff)", style=None if no_color else "dim")
    else:
        lines: list[Text] = []
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
            lines.append(Text(line, style=None if no_color else style))
        body = Group(*lines)
    # Text(title), not the raw string -- Panel's title is markup-parsed by
    # default, and a bracketed filename (Next.js-style "[id].tsx" routes are
    # common in exactly the kind of repo this diffs) would otherwise be
    # misread as a style tag instead of shown literally.
    console.print(Panel(body, title=Text(title), border_style="blue", expand=False))


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
    objective = escape(str(plan.get("objective") or "no objective recorded"))
    console.print(f"Plan in progress ({objective}):" if no_color else f"[bold cyan]Plan in progress[/bold cyan] ({objective}):")
    markers = {"completed": "✓", "in_progress": "◉", "failed": "✗"}
    colors = {"completed": "green", "in_progress": "yellow", "failed": "red"}
    for step in steps:
        status = step.get("status", "pending")
        glyph = markers.get(status, "○")
        color = colors.get(status)
        marker = glyph if no_color or color is None else f"[{color}]{glyph}[/{color}]"
        console.print(f"  {marker} {escape(str(step.get('step') or ''))}")
