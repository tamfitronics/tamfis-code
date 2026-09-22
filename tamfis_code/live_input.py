"""Non-blocking follow-up input for a running standalone task.

The old implementation put stdin into cbreak mode, discarded ordinary
characters, and required Ctrl+Y to open a second editor. That made the
terminal feel frozen and made mouse selection/scrolling fight Rich's live
redraw. A running task now owns a normal prompt-toolkit line editor instead:
the user can type at any time, press Enter to queue a follow-up, and keep
typing the next one while the model continues streaming.
"""
from __future__ import annotations

import asyncio
import contextlib
import difflib
import inspect
import os
import re
import sys
import time
from typing import Any, Awaitable, Callable, Optional

from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI, FormattedText, HTML, to_formatted_text
from prompt_toolkit.styles import Style

from . import state as local_state
import logging
from .terminal_guard import TerminalGuard, disable_focus_reporting
from .config import Config, mode_label_for_policy, next_mode_in_cycle
from .render import StreamRenderer

_log = logging.getLogger("tamfis_code.live_input")
_SHIFT_TAB = b"\x1b[Z"
# Retained only for backwards-compatible imports. Ctrl+Y is no longer read
# specially by the live listener; it is ordinary editable prompt input.
_CTRL_T = b"\x14"
_CTRL_Y = b"\x19"
_STATUS_SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
# The running status above the composer cycles Claude Code's star glyph
# ("✢ Mustering… (1m 58s · ↓ 7.7k tokens · thinking)").
_HEADLINE_GLYPHS = ("✢", "✳", "✶", "✻", "✽", "✻", "✶", "✳")
# One tick counter drives both animations (10 braille frames, 8 star glyphs);
# wrapping at their least common multiple keeps each cycle smooth.
_STATUS_TICK_PERIOD = 40
_STATUS_REFRESH_INTERVAL_SECONDS = 0.25
_STDOUT_BATCH_INTERVAL_SECONDS = 0.01
# The composer and Assistant message container deliberately share the full
# terminal width, matching Claude Code's message layout.
_COMPOSER_MAX_WIDTH = None

# Claude Code's own bottom-toolbar phrasing for the three MODE_CYCLE stops
# that actually change what gets auto-approved -- "manual" (/mode's "ask")
# is the quiet default and gets no banner, matching how Claude Code only
# announces the modes that suppress a prompt.
_MODE_ON_LABEL = {
    "accept-edits": "auto-accept edits on",
    "auto": "auto mode on",
    "plan-only": "plan mode on",
}

# Right-aligned corner chip, same layout as Claude Code's own bottom bar
# (left-aligned status, one short hint flush right). Rotates on a slow clock
# rather than every render -- the toolbar redraws several times a second
# while a task streams, and a hint that changes that fast is just noise.
#
# Every tip below is gated on the session state that would make it actually
# actionable -- a fixed rotation advertised "/diff to review pending
# changes" on a session with nothing modified and "/retry to rerun the last
# turn" on a brand-new thread with no prior turn, which is just noise dressed
# up as a hint. Each entry is (predicate, text); predicate takes the
# SessionState plus the caller's already-known active-agent count so this
# doesn't need its own duplicate agent-counting logic.
_ALWAYS = lambda state, agents: True
_ROTATING_TIPS: tuple[tuple[Callable[[Any, int], bool], str], ...] = (
    (_ALWAYS, "Tip: Use /btw for a quick side question without interrupting the current task"),
    (lambda state, agents: not state.active_plan_id, "/plan to think before executing"),
    (_ALWAYS, "/model to switch models"),
    (lambda state, agents: bool(state.modified_files), "/diff to review pending changes"),
    (lambda state, agents: agents > 0, "/agents to see what's running"),
    (lambda state, agents: bool(state.conversation_history), "/retry to rerun the last turn"),
    (lambda state, agents: bool(state.unresolved_issues), "/doctor to check unresolved issues"),
    (_ALWAYS, "/status for session, task, cwd"),
    (
        lambda state, agents: any(
            item.get("status") == "queued" for item in (state.queued_user_instructions or [])
        ),
        "↑ to edit your queued message",
    ),
)
_TIP_ROTATE_SECONDS = 8.0


def _tip_text(session_id: Optional[int] = None, active_agents: int = 0) -> str:
    """The plain text of the currently rotating tip."""
    if session_id is None:
        applicable = [text for predicate, text in _ROTATING_TIPS if predicate is _ALWAYS]
    else:
        state = local_state.get_session_state(session_id)
        applicable = [
            text for predicate, text in _ROTATING_TIPS if predicate(state, active_agents)
        ]
    if not applicable:
        applicable = [text for _, text in _ROTATING_TIPS]
    return applicable[int(time.monotonic() // _TIP_ROTATE_SECONDS) % len(applicable)]


_TERMINAL_NOISE_RE = re.compile(r"\x1b\[[0-9;?<>=]*[ -/]*[@-~]|\x1b[O@-Z\\-_]|\x1b|[\x00-\x08\x0b\x0c\x0e-\x1a\x1c-\x1f\x7f]")


def strip_terminal_noise(text: str) -> str:
    """Remove terminal control sequences/bytes that leaked into typed text.

    Defence in depth only -- the primary fix is that the key decoder consumes
    them (focus events, Shift+Tab, arrows) and that a prompt always owns the
    tty. A control sequence must never reach the model or the scrollback as
    ordinary text. Newlines/tabs are kept.
    """
    return _TERMINAL_NOISE_RE.sub("", text or "")


def _format_ago(seconds: float) -> str:
    total = int(seconds)
    if total < 60:
        return f"{total}s ago"
    return f"{total // 60}m {total % 60}s ago"


def _truncate(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: max(1, limit - 1)].rstrip() + "…"


def composer_rule_html() -> str:
    """A bounded horizontal rule for the top/bottom edge of the composer.

    Claude Code and Codex draw the input between two plain rules with the
    running status ABOVE it and the mode line BELOW. Capping the rule at the
    assistant card width keeps ultrawide terminals readable instead of
    producing two wall-to-wall separators.
    """
    import shutil

    terminal_width = shutil.get_terminal_size(fallback=(80, 24)).columns
    width = max(20, terminal_width) if _COMPOSER_MAX_WIDTH is None else min(
        _COMPOSER_MAX_WIDTH, max(20, terminal_width)
    )
    return f"<ansigray>{'─' * width}</ansigray>"


def _right_chip(session_id: Optional[int] = None, active_agents: int = 0) -> str:
    # ansibrightblack (the ghost-text/auto-suggestion color -- deliberately
    # dim) renders as unreadable-to-invisible against some terminal themes'
    # backgrounds for ordinary body text. ansigray is the same tone the rest
    # of this toolbar's left-side status already uses, so the chip stays
    # visually secondary without disappearing.
    return f"<ansigray>{_tip_text(session_id, active_agents)}</ansigray>"


def _right_align(left_html: str, right_html: str, *, min_gap: int = 2) -> str:
    """Pad `left_html` with spaces so `right_html` lands flush against the
    terminal's right edge -- prompt-toolkit's bottom_toolbar has no builtin
    two-zone layout, so this measures visible width by stripping the HTML
    tags (there is no markup inside the tag text itself, so a plain regex
    is exact here) rather than pulling in prompt_toolkit's heavier
    formatted-text width calculation for what is just plain-ASCII content.
    """
    import re
    import shutil

    def visible_len(fragment: str) -> int:
        return len(re.sub(r"<[^>]+>", "", fragment))

    width = shutil.get_terminal_size(fallback=(80, 24)).columns
    gap = max(min_gap, width - visible_len(left_html) - visible_len(right_html))
    return f"{left_html}{' ' * gap}{right_html}"


class _CompletedAwaitable:
    """A no-op awaitable that is also safe to ignore for non-TTY callers."""

    def __await__(self):
        if False:
            yield None
        return None


def _active_agent_count(exclude_session_id: int) -> int:
    """Count other known sessions currently mid-task (e.g. swarm children
    delegated via /delegate), for the "N agents" toolbar suffix."""
    # Ordinary sessions can remain marked "running" after a killed SSH
    # process. Only real delegated swarm children belong in the agent
    # counter; counting every stale top-level session produced nonsense
    # footers such as "← 16 agents" on a fresh prompt.
    return local_state.active_swarm_child_count(
        exclude_session_id=exclude_session_id,
    )


def _route_note_html(session_id: int) -> str:
    """Footer fragment for a route exception (failover / exhausted / cooling).

    Empty for a healthy session, so nothing changes on the common path. Short
    by construction (route_status_compact) and designed to sit immediately
    after the session title so an 80-column terminal cannot clip it away.
    """
    try:
        from .state import route_status_compact

        note = route_status_compact(session_id)
    except Exception:
        return ""
    if not note:
        return ""
    from xml.sax.saxutils import escape as _xml_escape

    return f" <ansiyellow>{_xml_escape(note)}</ansiyellow>"


def _mode_html(cli_config: Config) -> str:
    mode = mode_label_for_policy(cli_config.approval_policy)
    mode_on = _MODE_ON_LABEL.get(mode)
    mode_line = (
        f"<ansiyellow>⏵⏵ {mode_on.replace(' mode on', '').replace(' on', '')} · shift+tab</ansiyellow>"
        if mode_on
        else f"<ansigray>⏵⏵ {mode} · shift+tab</ansigray>"
    )
    return mode_line


def _agents_suffix_html(agents: int) -> str:
    return f" <ansigray>· ← {agents} agent{'s' if agents != 1 else ''}</ansigray>" if agents else ""


def _mode_and_agents_html(
    cli_config: Config,
    session_id: int,
    *,
    active_agents: Optional[int] = None,
) -> str:
    agents = (
        _active_agent_count(session_id)
        if active_agents is None
        else active_agents
    )
    return f"{_mode_html(cli_config)}{_agents_suffix_html(agents)}"


_FOOTER_TITLE_MAX_CHARS = 28


def _session_title_prefix(session_id: Optional[int]) -> str:
    """Persistent "which conversation is this" label for the far left of
    the footer -- shown at all times (idle prompt and mid-task), the same
    way Codex/Claude Code keep a session's name pinned in their status bar
    instead of only showing it once at startup."""
    if session_id is None:
        return ""
    from xml.sax.saxutils import escape as _xml_escape

    title = local_state.session_display_title(session_id)
    if len(title) > _FOOTER_TITLE_MAX_CHARS:
        title = title[: _FOOTER_TITLE_MAX_CHARS - 1] + "…"
    return f"<ansicyan>{_xml_escape(title)}</ansicyan> <ansigray>·</ansigray> "


def idle_bottom_toolbar(
    cli_config: Config,
    session_id: int,
    *,
    provider: str = "auto",
    model: Optional[str] = None,
    has_suggestion: bool = False,
    active_agents: Optional[int] = None,
    update_version: Optional[str] = None,
    update_handler: Optional[Callable[..., Any]] = None,
) -> HTML | FormattedText:
    """Bottom-toolbar content for the plain REPL prompt (no task running) --
    same mode/agents banner as the live in-task footer below, so the bar
    doesn't disappear the moment a turn finishes."""
    suggestion_hint = (
        " <ansigray>· tab to use next suggestion</ansigray>"
        if has_suggestion else ""
    )
    from .public_identity import public_model_name

    resolved_agents = (
        _active_agent_count(session_id) if active_agents is None else active_agents
    )
    # Same route-exception note as the live in-task footer (see
    # LiveInputListener._bottom_toolbar), so the information survives the
    # moment the turn ends instead of disappearing with the in-task bar.
    route_html = _route_note_html(session_id)
    left = (
        f" {_session_title_prefix(session_id)}{route_html}"
        f"<ansigray>ready · {public_model_name(model)} ·</ansigray> "
        f"{_mode_and_agents_html(cli_config, session_id, active_agents=resolved_agents)}"
        f"{suggestion_hint}"
    )
    chip = (
        f"<update-action>↑ Install v{update_version} · Ctrl+U or /update</update-action>"
        if update_version else _right_chip(session_id, resolved_agents)
    )
    if not update_version:
        # A rotating tip is a nicety: when the footer is already full it used to
        # be sliced mid-word at the terminal edge ("Tip: Use /btw for a q").
        # Show it only when it fits whole. The install chip is important and
        # is always kept.
        import shutil

        def _visible(fragment: str) -> int:
            return len(re.sub(r"<[^>]+>", "", fragment))

        columns = shutil.get_terminal_size(fallback=(80, 24)).columns
        if _visible(left) + _visible(chip) + 3 > columns:
            chip = ""
    # The composer's bottom edge: a plain rule, then the footer line -- the
    # same Claude/Codex split the running composer uses.
    rendered = HTML(f"{composer_rule_html()}\n{_right_align(left, chip + ' ')}")
    if not update_version or update_handler is None:
        return rendered
    # prompt_toolkit supports a mouse handler as the optional third item in
    # a formatted-text fragment. HTML gives us the layout and escaping; tag
    # only the update chip's fragments as clickable after conversion.
    fragments = []
    for style, text in to_formatted_text(rendered):
        if "class:update-action" in style:
            fragments.append((style, text, update_handler))
        else:
            fragments.append((style, text))
    return FormattedText(fragments)


@contextlib.contextmanager
def responsive_patch_stdout(*, raw: bool = False):
    """Keep prompt-safe concurrent output without the default 200 ms lag."""
    from prompt_toolkit.patch_stdout import StdoutProxy

    with StdoutProxy(
        sleep_between_writes=_STDOUT_BATCH_INTERVAL_SECONDS,
        raw=raw,
    ) as proxy:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = proxy
        sys.stderr = proxy
        try:
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def composer_style() -> Style:
    """Claude-like footer styling: plain/dim text, never a reverse block."""
    return Style.from_dict({
        "bottom-toolbar": "noreverse",
        "bottom-toolbar.text": "noreverse",
        "update-action": "ansiyellow bold underline",
        # Do not rely on prompt-toolkit's palette-dependent #888 default:
        # several SSH themes render it indistinguishably from the composer
        # background. Bright-black is the portable ANSI "ghost text" color.
        "auto-suggestion": "fg:ansibrightblack italic",
    })


def live_next_message_suggestion(renderer: Any) -> Optional[str]:
    """Suggest one useful steering message from live execution progress.

    This stays deterministic and renderer-backed: the composer may reflect
    the current plan step or phase, but it must not invent repository facts
    while the model is still working.
    """
    has_pending = getattr(renderer, "has_pending_steering", None)
    if callable(has_pending) and has_pending():
        return None

    steps = [
        item for item in (getattr(renderer, "_plan_steps", None) or [])
        if isinstance(item, dict) and item.get("step")
    ]
    for status, prefix in (
        ("failed", "Repair and revalidate the failed plan step"),
        ("in_progress", "Finish and verify the active plan step"),
        ("pending", "Continue with the next plan step"),
    ):
        step = next(
            (item for item in steps if str(item.get("status") or "pending") == status),
            None,
        )
        if step is not None:
            compact = " ".join(str(step["step"]).split())
            return f"{prefix}: {compact}"[:240]

    if getattr(renderer, "_running_command", None):
        return "After the command finishes, inspect any failure and adapt the implementation"

    phase = str(getattr(renderer, "_phase", "") or "").lower()
    tool_counts = getattr(renderer, "_round_tool_counts", None) or {}
    if tool_counts:
        # Make the ghost action reflect the evidence currently on screen,
        # rather than offering the same generic phase slogan after every
        # tool call.  Counts are insertion-ordered by the renderer, so the
        # last entry is the most recently observed operation.
        latest_tool = str(next(reversed(tool_counts))).replace("_", " ")
        if phase in {"observe", "validate", "repair", "execute"}:
            return f"Review the latest {latest_tool} result, update the plan if needed, then verify the change"
    by_phase = {
        "understand": "Inspect related tests and call sites before making the change",
        "inspect": "Inspect related tests and call sites before making the change",
        "execute": "Validate the current change before moving to the next step",
        "observe": "Use the latest tool result to verify assumptions before continuing",
        "repair": "Confirm the repair addresses the root cause, then rerun validation",
        "validate": "Fix any validation failure before declaring the task complete",
    }
    return by_phase.get(phase)


class _LiveProgressAutoSuggest(AutoSuggest):
    def __init__(self, renderer: Any) -> None:
        self._renderer = renderer

    def get_suggestion(self, buffer, document: Document) -> Optional[Suggestion]:
        if document.text:
            return None
        value = live_next_message_suggestion(self._renderer)
        return Suggestion(value) if value else None


def force_bottom_toolbar_visible(session: Any) -> None:
    """Render status on terminals where prompt-toolkit disables CPR.

    PromptSession normally hides its bottom toolbar until the terminal
    confirms cursor-position-report support. Several SSH/web terminals never
    answer that query, leaving Tamfis-Code's mode/model/status permanently
    absent. The toolbar is essential UI here, so remove only that capability
    gate while retaining prompt-toolkit's own toolbar window.
    """
    root = getattr(getattr(session, "layout", None), "container", None)
    children = getattr(root, "children", None)
    if not children:
        return
    toolbar = children[-1]
    if hasattr(toolbar, "filter"):
        toolbar.filter = Condition(
            lambda: getattr(session, "bottom_toolbar", None) is not None
        )


class LiveInputListener:
    """Run a persistent, asynchronous follow-up editor during a task."""

    def __init__(
        self,
        *,
        session_id: int,
        renderer: StreamRenderer,
        cli_config: Config,
        interrupt_callback: Optional[Callable[[str], None]] = None,
        side_question_callback: Optional[Callable[[str], Awaitable[str]]] = None,
        command_completer: Any = None,
    ) -> None:
        self.session_id = session_id
        self.renderer = renderer
        self.cli_config = cli_config
        self._interrupt_callback = interrupt_callback
        self._side_question_callback = side_question_callback
        self._command_completer = command_completer
        self._interrupt_classification: Optional[str] = None
        self._is_tty = bool(getattr(sys.stdin, "isatty", lambda: False)())
        self._input_task: Optional[asyncio.Task] = None
        self._interject_task: Optional[asyncio.Task] = None
        self._prompt_session = None
        self._previous_loop_exception_handler: Optional[Callable[[asyncio.AbstractEventLoop, dict[str, Any]], None]] = None
        self._paused = False
        self._active = False
        self._last_invalidate = 0.0
        self._status_tick = 0
        self._ticker_task: Optional[asyncio.Task] = None
        # Side questions run independently from the active agent turn. Keep
        # strong references so asyncio cannot collect an in-flight answer;
        # they deliberately are not cancelled when the main turn finishes.
        self._btw_tasks: set[asyncio.Task] = set()
        self._outcome_status: Optional[str] = None
        self._editing_instruction_id: Optional[str] = None
        # Distinguish a programmatic prompt shutdown (approval/tool UI,
        # turn completion) from a user pressing Enter.  Without this marker,
        # prompt_toolkit returns the current buffer through ``prompt_async``
        # and a fast pause/resume race can enqueue half-typed text.
        self._prompt_exit_requested = False
        self._draft_text = ""
        # Keyboard-ownership bookkeeping (see _supervised_input_loop): the prompt
        # is restarted if it ends while the task is still running, and a hard
        # terminal hang-up is told apart from a stray Ctrl+D.
        self._terminal = TerminalGuard()
        self._eof_times: list[float] = []
        self._input_restarts: list[float] = []
        self._stall_reported = False
        self._input_exit_expected = False
        self._loop_lag_reported = 0.0
        # A footer callback runs for every redraw and keystroke. Snapshot the
        # count once per listener instead of touching the multi-megabyte
        # session-state file from prompt-toolkit's latency-sensitive path.
        self._active_agents = (
            _active_agent_count(session_id)
            if self._is_tty
            else 0
        )

    def start(self) -> None:
        if not self._is_tty:
            return
        # Stop Rich's repainting while the prompt owns the terminal. Streamed
        # assistant text is intentionally rendered as scrollback in this
        # mode, so the input line and mouse scrolling never compete.
        self.renderer.suspend_live()
        self.renderer.live_input_listener = self
        self._active = True
        self._terminal.snapshot()
        disable_focus_reporting()
        from .runtime.progress import configure_execution_log

        configure_execution_log()
        self._ticker_task = asyncio.create_task(self._status_ticker())
        self._schedule_prompt()

    def stop(self):
        """Stop input ownership, returning an awaitable only when needed.

        Non-TTY callers commonly use the listener as a capability probe and
        call lifecycle methods synchronously. Returning immediately in that
        case avoids an un-awaited coroutine warning while preserving the
        awaited shutdown path for a real prompt-toolkit session.
        """
        if not self._is_tty and not self._active and self._input_task is None:
            self._active = False
            return _CompletedAwaitable()
        return self._stop_async()

    async def _stop_async(self) -> None:
        """Stop input ownership and wait until prompt-toolkit releases stdin."""
        self._active = False
        if self._outcome_status and hasattr(self.renderer, "conclude"):
            # Clear the accumulated "Reading N files..." activity before
            # prompt-toolkit draws its last teardown frame. Cancellation
            # bypasses the normal task-completed/failed renderer events, so
            # this is the only common finalization point for every outcome.
            self.renderer.conclude(self._outcome_status)
        ticker = self._ticker_task
        self._ticker_task = None
        if ticker is not None and not ticker.done():
            ticker.cancel()
        if ticker is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await ticker
        await self._shutdown_prompt()
        self._terminal.restore()
        try:
            from .message_viewer import VIEWER

            VIEWER.close()
        except Exception:
            pass
        if self.renderer.live_input_listener is self:
            self.renderer.live_input_listener = None
        # Whatever the outcome, commit the pinned plan's final state to scrollback once
        # (a no-op when nothing was pinned): with the composer gone the plan would
        # otherwise vanish. Idempotent -- print_work_summary below calls it too.
        try:
            self.renderer._print_final_plan()
        except Exception:
            pass
        if self._outcome_status:
            # The turn is over: do not restart Rich's transient live spinner
            # for the few milliseconds before renderer.finish(). Replace the
            # animated footer directly with one durable timing summary.
            if hasattr(self.renderer, "print_work_summary"):
                self.renderer.print_work_summary(self._outcome_status)
            if hasattr(self.renderer, "print_recap"):
                self.renderer.print_recap(self.session_id)
        else:
            self.renderer.resume_live()

    def set_outcome_status(self, status: str) -> None:
        self._outcome_status = status

    async def _status_ticker(self) -> None:
        """Animate and update elapsed time even during silent provider waits."""
        try:
            last_wake = time.monotonic()
            while self._active:
                await asyncio.sleep(_STATUS_REFRESH_INTERVAL_SECONDS)
                now = time.monotonic()
                # If this 0.25s sleep took seconds, something ran synchronously on
                # the event loop -- and the keyboard was frozen for that long.
                lag = now - last_wake - _STATUS_REFRESH_INTERVAL_SECONDS
                last_wake = now
                if lag > 2.0 and now - self._loop_lag_reported > 30.0:
                    self._loop_lag_reported = now
                    _log.warning("event loop blocked for %.1fs (keyboard unresponsive meanwhile)", lag)
                self._status_tick = (self._status_tick + 1) % _STATUS_TICK_PERIOD
                self._trace_tty_mode()
                self._watchdog_check()
                self.invalidate()
        except asyncio.CancelledError:
            raise

    def _trace_tty_mode(self) -> None:
        """Log (on change only) whether the tty is in raw/no-echo mode. A composer that believes it
        is running while the tty is cooked is the "dead Enter / ^[ painted as text" failure; this
        makes the moment it flips visible in the execution log."""
        try:
            import termios

            fd = sys.stdin.fileno()
            echo = bool(termios.tcgetattr(fd)[3] & termios.ECHO)
        except Exception:
            return
        if echo != getattr(self, "_last_tty_echo", None):
            self._last_tty_echo = echo
            _log.info("tty echo=%s (composer_running=%s paused=%s)", echo,
                      bool(self._prompt_session is not None), self._paused)

    def _watchdog_check(self) -> None:
        """Diagnose a run that has stopped making meaningful progress.

        The request-level timeouts (first byte / idle / total, runner_local) are
        the primary defence and switch route on their own. This is the backstop
        for the case they miss: a model request that has produced nothing at all
        for ``provider_abort`` seconds is stopped explicitly -- with a checkpoint
        and a `continue` -- instead of leaving a spinner running over a dead
        request. Long tools/commands are never touched: only WAITING_PROVIDER.
        """
        progress = getattr(self.renderer, "progress", None)
        if progress is None or self._interrupt_classification is not None:
            return
        snap = progress.snapshot()
        if snap.state.value == "stalled" and not self._stall_reported:
            self._stall_reported = True
            _log.warning("watchdog: provider silent for %.0fs; awaiting route replacement", snap.idle_seconds)
            self.renderer.handle_event({
                "event_type": "diagnostics",
                "payload": {"content": (
                    f"⚠ The model has not responded for {int(snap.idle_seconds)}s — "
                    "replacing the request automatically…"
                )},
            })
        elif snap.state.value not in {"stalled"} and snap.idle_seconds < 5:
            self._stall_reported = False
        if progress.should_abort_silent():
            _log.error("watchdog: no activity for %.0fs; stopping the wedged run", snap.idle_seconds)
            self.renderer.handle_event({
                "event_type": "diagnostics",
                "payload": {"content": (
                    f"⚠ No activity for {int(snap.idle_seconds) // 60} minutes — the run looks stuck, so it "
                    "was stopped. Your work is checkpointed; type `continue` to resume."
                )},
            })
            self._request_interrupt("stall")
            return
        if progress.should_abort_provider_wait():
            _log.error("watchdog: aborting provider wait after %.0fs without progress", snap.idle_seconds)
            self.renderer.handle_event({
                "event_type": "diagnostics",
                "payload": {"content": (
                    f"⚠ Provider stopped responding ({int(snap.idle_seconds)}s with no output) — "
                    "stopping the stuck request. Your work is checkpointed; type `continue` to resume."
                )},
            })
            self._request_interrupt("stall")

    def pause(self) -> None:
        """Synchronously request prompt shutdown before another UI reads stdin.

        This only *requests* the exit -- ``app.exit()``/``task.cancel()`` are
        both fire-and-forget, so the underlying prompt_toolkit Application
        may not have actually released the terminal by the time this
        returns. A caller that is about to open its own PromptSession right
        away (an approval gate, most notably -- see resolve_approval_decision_async
        called from a live in-task context) must use `pause_async` instead,
        or risk two Applications racing for the same stdin fd: the new
        prompt renders but silently never receives keystrokes, reported live
        as the y/n approval gate "not responding"/"freezing" in manual mode,
        the mode that actually stops to ask.
        """
        _log.info("composer pause() requested")
        self._paused = True
        self._request_prompt_exit()
        self._cancel_prompt()

    async def pause_async(self) -> None:
        """Like `pause`, but waits until the old prompt has actually released
        the terminal before returning. Use this (not `pause`) whenever a new
        PromptSession/Application is about to be opened right after -- e.g.
        immediately before an approval gate prompt."""
        _log.info("composer pause_async() begin")
        self._paused = True
        self._request_prompt_exit()
        self._cancel_prompt()
        task = self._input_task
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        _log.info("composer pause_async() released the terminal")

    def resume(self) -> None:
        _log.info("composer resume() active=%s task_alive=%s", self._active,
                  bool(self._input_task is not None and not self._input_task.done()))
        self._paused = False
        self._prompt_exit_requested = False
        if not self._active:
            return
        task = self._input_task
        if task is not None and not task.done():
            task.add_done_callback(lambda _task: self._schedule_prompt() if self._active and not self._paused else None)
            return
        self._schedule_prompt()

    def _dispatch(self) -> None:
        """Compatibility hook for older embedders; start() no longer uses it."""
        buf = bytes(getattr(self, "_buf", b""))
        if _SHIFT_TAB in buf:
            self._buf = bytearray()
            self._cycle_mode()
        elif buf in {b"\x1b", b"\x1b["}:
            return
        elif _CTRL_Y in buf:
            self._buf = bytearray()
            if self._interject_task is None or self._interject_task.done():
                self._interject_task = asyncio.create_task(self._interject())
        elif buf:
            self._buf = bytearray()

    def _cycle_mode(self) -> str:
        """Cycle mode without letting BackTab's ESC prefix cancel the task."""
        self.cli_config.approval_policy = next_mode_in_cycle(
            self.cli_config.approval_policy
        )
        label = mode_label_for_policy(self.cli_config.approval_policy)
        if hasattr(self.renderer, "set_mode_label"):
            self.renderer.set_mode_label(label)
        self.invalidate()
        return label

    def _schedule_prompt(self) -> None:
        if self._input_task is None or self._input_task.done():
            _log.info("composer starting a new input loop")
            self._input_task = asyncio.create_task(self._supervised_input_loop())

    def _request_prompt_exit(self) -> None:
        session = self._prompt_session
        app = getattr(session, "app", None)
        if app is not None and not getattr(app, "is_done", False):
            buffer = getattr(session, "default_buffer", None)
            draft = str(getattr(buffer, "text", "") or "")
            if draft.strip():
                self._draft_text = draft
            self._prompt_exit_requested = True
            with contextlib.suppress(Exception):
                app.exit(result="")

    def _cancel_prompt(self) -> None:
        task = self._input_task
        if task is not None and not task.done():
            task.cancel()

    async def _shutdown_prompt(self) -> None:
        self._request_prompt_exit()
        task = self._input_task
        self._input_task = None
        if task is not None and not task.done():
            # Give app.exit()'s own coroutine resumption (prompt_toolkit's
            # render finalization -- erasing the framed composer/bottom
            # toolbar, since result="" looks like a real submission rather
            # than a forced teardown) a bounded chance to actually run on
            # the event loop before force-cancelling it. Cancelling
            # immediately after app.exit() -- the previous behaviour --
            # interrupts that finalization mid-flight: the input task still
            # tears down fine (caught below), but the terminal is left with
            # the last busy-toolbar frame stuck as static scrollback.
            # Live-reported as the status line staying frozen forever after
            # a turn completes (cosmetic only -- a new prompt still works
            # right underneath it). `shield` keeps the task itself running
            # past the wait_for's own timeout so the explicit cancel below
            # remains the one and only cancellation path.
            with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError, Exception):
                await asyncio.wait_for(asyncio.shield(task), timeout=0.5)
            if not task.done():
                task.cancel()
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._prompt_session = None

    def invalidate(self) -> None:
        """Refresh the prompt footer after a streamed phase/status update."""
        app = getattr(self._prompt_session, "app", None)
        if app is not None:
            # Reasoning models can emit dozens of tiny deltas per second.
            # Invalidating prompt-toolkit for every delta makes the input UI
            # compete with Rich and makes the whole terminal feel sluggish.
            # The footer is status-only, so a bounded 4 Hz refresh is enough
            # while keeping phase/model/elapsed-time changes responsive.
            now = time.monotonic()
            if now - self._last_invalidate < _STATUS_REFRESH_INTERVAL_SECONDS:
                return
            self._last_invalidate = now
            app.invalidate()

    def _composer_message(self):
        """Everything ABOVE the input, then the input's own top rule and the
        `❯` prompt -- Claude Code / Codex layout.

            ⠴ Musing… (5m 45s · ↓ 12.9k tokens)
            Tip: Use /btw for a quick side question…
            ──────────────────────────────────────
            ❯ <input>

        The running status and the tip used to sit in the bottom toolbar
        squeezed in with the mode, title and route note, under a framed box.
        """
        from xml.sax.saxutils import escape as _xml_escape

        spinner = _HEADLINE_GLYPHS[self._status_tick % len(_HEADLINE_GLYPHS)]
        lines = []
        # The plan, pinned and redrawn in place as steps complete -- not a fresh panel
        # printed into the scrollback per step. A blank line on each side so it does not
        # crowd the tool records above or the status line below.
        try:
            import shutil

            size = shutil.get_terminal_size(fallback=(80, 24))
            plan_lines = self.renderer.live_input_plan_lines(size.columns, size.lines)
        except Exception:
            plan_lines = []
        from .message_viewer import VIEWER, panel_ansi

        if plan_lines and not VIEWER.is_open:  # the viewer takes the room while it is open
            lines.append("")
            lines.extend(plan_lines)
            lines.append("")
        activity = self.renderer.live_input_activity_line()
        if activity:
            # "Reading 3 files…  ⎿  $ pytest -q (12s)" is two facts: what the round is
            # doing, and the command running right now. Claude Code shows the
            # command on its own "⎿" line under the activity title.
            import shutil

            head, marker, running = activity.partition("  ⎿  ")
            room = max(20, shutil.get_terminal_size(fallback=(80, 24)).columns - 6)
            if head.strip():
                lines.append(f" <ansigray>{_xml_escape(_truncate(head.strip(), room))}</ansigray>")
            if marker:
                # Cut the COMMAND to fit, never its elapsed-time suffix "(12s)".
                body, elapsed = running, ""
                match = re.match(r"^(.*?)(\s\(\d[^)]*\))$", running)
                if match:
                    body, elapsed = match.group(1), match.group(2)
                shown = _truncate(body, max(10, room - 5 - len(elapsed))) + elapsed
                lines.append(f" <ansigray>  ⎿  {_xml_escape(shown)}</ansigray>")
        import shutil as _shutil

        columns = _shutil.get_terminal_size(fallback=(80, 24)).columns
        # Leave room for the right-aligned route note (when there is one) and the margin.
        try:
            from .state import route_status_compact as _note_for_width

            reserved = len(_note_for_width(self.session_id) or "")
        except Exception:
            reserved = 0
        # Keep the activity block separate from the headline/tip block.
        if lines:
            lines.append("")
        headline = self.renderer.live_input_headline(
            spinner, width=max(30, columns - 2 - (reserved + 2 if reserved else 0)),
        )
        headline_html = f" <ansicyan>{_xml_escape(headline)}</ansicyan>"
        # A route exception (failover / exhausted route) rides on the SAME line,
        # right-aligned: this line is short, so it cannot be clipped away the
        # way it was when it shared the crowded footer. Product vocabulary only
        # (route_status_compact) -- never a provider name.
        try:
            from .state import route_status_compact

            note = route_status_compact(self.session_id)
        except Exception:
            note = ""
        if note:
            headline_html = _right_align(
                headline_html, f"<ansiyellow>{_xml_escape(note)}</ansiyellow> ",
            )
        lines.append(headline_html)
        try:
            queued = max(0, self.renderer.steering_revision() - self.renderer._steering_handled_revision)
        except Exception:
            queued = 0
        if queued:
            lines.append(f" <ansigray>↳ {queued} follow-up{'s' if queued != 1 else ''} queued</ansigray>")
        pending_update = getattr(self.renderer, "pending_update_version", None)
        if pending_update:
            # Informational only: self_update.py's apply_update()/reexec()
            # deliberately never run mid-task (re-exec would abandon the
            # in-flight turn), so this has no click/Ctrl+U handler -- that
            # action stays on the idle toolbar (idle_bottom_toolbar).
            lines.append(
                f" <ansiyellow>↑ v{_xml_escape(str(pending_update))} available · update when idle · Ctrl+U or /update queues a safe resume</ansiyellow>"
            )
        else:
            tip = _tip_text(self.session_id, self._active_agents)
            if not tip.startswith("Tip"):
                tip = f"Tip: {tip}"
            # Cut to the terminal width like the activity lines above: an 83-character
            # tip wrapped onto a second row on a 72-column terminal, pushing the
            # composer down (and made test_a_long_command_is_cut_to_the_terminal_width
            # fail whenever that tip happened to be the one rotating in).
            import shutil

            tip_room = max(20, shutil.get_terminal_size(fallback=(80, 24)).columns - 2)
            lines.append(f" <ansigray>{_xml_escape(_truncate(tip, tip_room))}</ansigray>")
        lines.append(composer_rule_html())
        composer = HTML("\n".join(lines) + "\n<ansicyan><b>❯</b></ansicyan> ")
        viewer = panel_ansi()
        if viewer:
            # The full message (Ctrl+E) sits above the status; Ctrl+E/Esc closes it.
            return FormattedText(list(to_formatted_text(ANSI(viewer))) + list(to_formatted_text(composer)))
        return composer

    def _bottom_toolbar(self):
        """Everything BELOW the input: the bottom rule, then ONE footer line --
        the pinned session title, then mode and shortcuts. The running status,
        tip and route note are above the input (see _composer_message)."""
        import shutil
        from xml.sax.saxutils import escape as _xml_escape

        # FIX (2026-08-21): a multi-line function call inside an f-string
        # expression (the {...} spanning several lines) is PEP 701 syntax,
        # Python 3.12+ only -- pyproject.toml declares requires-python
        # ">=3.10" and this was a real SyntaxError on 3.10/3.11, invisible
        # locally on this host's 3.13 interpreter and never caught by CI
        # because CI never actually ran pytest until today's fix wired that
        # in (see the CI-gating commit history). Computing the value first
        # is unambiguous across every supported Python version.
        agents_suffix = _agents_suffix_html(self._active_agents)
        rest = (
            f"{_mode_html(self.cli_config)}"
            f"<ansigray> · esc to interrupt</ansigray>{agents_suffix}"
        )
        width = shutil.get_terminal_size(fallback=(80, 24)).columns
        # The session title is pinned at the far left of the footer at all
        # times (idle and mid-task), but never at the cost of clipping the mode
        # and shortcuts: it takes what is left, truncated, or is dropped.
        rest_len = len(re.sub(r"<[^>]+>", "", rest))
        title_room = min(60, width - rest_len - 6)
        title_html = ""
        if title_room >= 8:
            title = local_state.session_display_title(self.session_id)
            if len(title) > title_room:
                title = title[: title_room - 1] + "…"
            title_html = f"<ansicyan>{_xml_escape(title)}</ansicyan> <ansigray>·</ansigray> "
        return HTML(f"{composer_rule_html()}\n {title_html}{rest}")

    async def _input_loop(self) -> None:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.key_binding import KeyBindings

        bindings = KeyBindings()

        # Some SSH/terminal clients emit focus-in/focus-out as CSI sequences
        # (ESC [ I / ESC [ O).  Because ``escape`` is also the intentional
        # cancel binding below, prompt_toolkit can otherwise dispatch the
        # first byte as a cancellation and leak the trailing ``I``/``O`` into
        # the prompt.  Consume those complete sequences before the bare-Esc
        # binding gets a chance to act.
        @bindings.add("escape", "[", "I")
        def _ignore_focus_in(event) -> None:
            return

        @bindings.add("escape", "[", "O")
        def _ignore_focus_out(event) -> None:
            return

        @bindings.add("c-d")
        def _ctrl_d_is_not_exit(event) -> None:
            # prompt_toolkit's default Ctrl+D on an empty line ends the prompt with
            # EOFError. Mid-task that used to END THE LIVE COMPOSER silently: the
            # task kept running, the tty fell back to cooked/echo mode and every
            # later key (Esc, focus events, Enter) was painted as "^[" / "^[[O"
            # text while nothing read it. Ctrl+D now never exits the prompt; Esc /
            # Ctrl+C are the (advertised) ways to stop a running task.
            buffer = event.current_buffer
            if buffer.text:
                buffer.delete(1)
                return
            self.renderer.handle_event({
                "event_type": "diagnostics",
                "payload": {"content": "◆ Ctrl+D does not exit while a task is running — press Esc or Ctrl+C to stop it."},
            })

        @bindings.add("s-tab")
        def _cycle_running_mode(event) -> None:
            # BackTab is encoded by terminals as ESC [ Z. Registering the
            # complete key explicitly makes prompt-toolkit wait for and
            # consume that sequence instead of dispatching its ESC prefix as
            # "cancel" and echoing the remaining bytes as ^[ / [Z.
            self._cycle_mode()
            event.app.invalidate()

        @bindings.add("enter")
        def _submit_nonempty(event) -> None:
            # Claude/Codex keep an empty composer open. Do not exit and
            # recreate the prompt (or enqueue a blank follow-up) on Enter.
            if event.current_buffer.text.strip():
                event.current_buffer.validate_and_handle()

        @bindings.add("tab")
        def _accept_progress_suggestion(event) -> None:
            buffer = event.current_buffer
            suggestion = live_next_message_suggestion(self.renderer)
            if not buffer.text and suggestion:
                buffer.insert_text(suggestion)

        @bindings.add("up")
        def _edit_latest_queued_instruction(event) -> None:
            # With an empty live composer, Up recalls the newest follow-up
            # that is still queued so Enter can replace it instead of adding
            # a duplicate. Once text is present, retain prompt-toolkit's
            # normal multiline/history navigation.
            if not event.current_buffer.text and self._recall_latest_queued(event.current_buffer):
                return
            event.current_buffer.auto_up(count=1)

        @bindings.add("c-u")
        def _queue_available_update(event) -> None:
            # During a live task, updating immediately would abandon the
            # current provider/tool boundary. Queue the command instead; the
            # durable checkpoint completes normally, then the idle loop
            # installs the release and re-execs into the same session.
            if getattr(self.renderer, "pending_update_version", None):
                self._enqueue("/update")
                event.app.invalidate()

        @bindings.add("c-b")
        def _background_running_command(event) -> None:
            # Only meaningful while execute_command actually has a command
            # in flight (see render.py's _running_command / the "ctrl+b to
            # run in background" hint, which is only shown under the same
            # condition) -- an idle prompt or a non-command tool call (a
            # file edit, a search) has nothing to detach, so this is a
            # deliberate no-op rather than an error, matching Ctrl+B being
            # otherwise unbound everywhere else in this REPL.
            if not self.renderer._running_command:
                return
            self.renderer.background_requested.set()

        @bindings.add("c-e")
        def _toggle_full_message(event) -> None:
            # Ctrl+E shows the newest collapsed long message (user or assistant) in
            # full in a viewer above the input -- render.py's collapse hint ("N more
            # chars -- press Ctrl+E to show full message") points here -- and Ctrl+E
            # again (or Esc) shows less: nothing is printed into scrollback, so
            # closing restores the screen exactly. With nothing collapsed it is a
            # quiet no-op.
            try:
                from .message_viewer import VIEWER

                VIEWER.toggle()
            except Exception:
                # A rendering hiccup in the viewer must never take down the input loop.
                pass
            event.app.invalidate()

        @bindings.add("c-o")
        @bindings.add("c-t")   # quiet alias: Termius and some terminals keep Ctrl+T for themselves
        def _toggle_tool_transcript(event) -> None:
            # Ctrl+O pages through the FULL output of finished tool calls -- what a block's
            # "… +N lines (Ctrl+O for full output)" refers to. Ctrl+E (long messages) is untouched.
            try:
                from .message_viewer import VIEWER
                from .render import TOOL_TRANSCRIPT

                VIEWER.toggle(TOOL_TRANSCRIPT, key="Ctrl+O")
            except Exception:
                pass
            event.app.invalidate()

        @bindings.add("escape")
        def _cancel_running_turn(event) -> None:
            # Escape cancels the active turn immediately and returns control
            # to the ordinary REPL without terminating Tamfis-Code.
            if self._paused or not self._active:
                return

            self._request_interrupt("cancel")

            if not event.app.is_done:
                event.app.exit(result="")

        # Registered LAST: for the same key prompt_toolkit takes the last matching
        # binding, and while the full-message viewer is open Esc/Up/Down must scroll or
        # close it, not cancel the turn or recall a queued instruction.
        from .message_viewer import install_bindings as _install_viewer_bindings

        _install_viewer_bindings(bindings)

        # The running composer is the same as the idle one: the input between
        # two plain rules, the running status and tip ABOVE it, the mode line
        # BELOW (see _composer_message / _bottom_toolbar) -- Claude Code and
        # Codex's layout, not a framed box with everything under it.
        session = PromptSession(
            key_bindings=bindings,
            completer=self._command_completer,
            show_frame=False,
            reserve_space_for_menu=0,
            style=composer_style(),
            auto_suggest=_LiveProgressAutoSuggest(self.renderer),
            # The running composer is a live status area, not part of the conversation: without this its last
            # frame (rules, tip, and the "esc to interrupt" footer) stayed in the scrollback after the task
            # ended and after every follow-up Enter, so a finished session still read "esc to interrupt"
            # and stale composer copies piled up between messages (owner paste 2026-09-21).
            erase_when_done=True,
        )
        force_bottom_toolbar_visible(session)
        self._prompt_session = session

        # prompt_toolkit's own built-in Ctrl+C key binding calls
        # Application.exit() unconditionally, with no way for it to know this
        # loop already exited the same Application for a different reason a
        # moment earlier (e.g. a cancel queued from another terminal, handled
        # via _cancel_running_turn / _shutdown_prompt). When both land in the
        # same input cycle, the second exit() raises "Return value already
        # set" -- but critically, that raise happens *inside a key-processor
        # callback*, not inside the coroutine `await session.prompt_async()`
        # is suspended on, so an ordinary try/except around that await never
        # sees it. `Application.run_async()` instead hands it to
        # `loop.set_exception_handler(self._handle_exception)` (installed for
        # the duration of the run, see prompt_toolkit's application.py), whose
        # handler prints the raw traceback and blocks on "Press ENTER to
        # continue" -- live-reported as the live UI appearing to freeze mid-
        # audit. Passing `set_exception_handler=False` below stops
        # prompt_toolkit installing that handler and leaves our own in place
        # for the duration of the prompt instead, so this specific known race
        # is resolved as an ordinary interrupt with no blocking prompt; any
        # other exception still falls through to asyncio's own default
        # handling (a plain logged error, not a blocking one) rather than
        # being silently hidden.
        loop = asyncio.get_running_loop()
        self._previous_loop_exception_handler = loop.get_exception_handler()
        loop.set_exception_handler(self._handle_prompt_loop_exception)

        def _prepare_prompt() -> None:
            if self._draft_text and not session.default_buffer.text:
                session.default_buffer.text = self._draft_text
                session.default_buffer.cursor_position = len(self._draft_text)
                self._draft_text = ""
            value = live_next_message_suggestion(self.renderer)
            session.default_buffer.suggestion = Suggestion(value) if value else None

        try:
            while self._active and not self._paused:
                # The marker belongs to ONE prompt instance. pause() sets it and then cancels the
                # task, which pre-empts the line below that would clear it -- so it used to survive
                # into the NEXT composer (the one restarted after an approval gate), which then
                # treated the user's first Enter as a teardown: the message was dropped, the composer
                # exited and the tty was left in echo mode. Every new prompt starts clean.
                self._prompt_exit_requested = False
                try:
                    with responsive_patch_stdout(raw=True):
                        text = await session.prompt_async(
                            self._composer_message,
                            bottom_toolbar=self._bottom_toolbar,
                            show_frame=False,
                            set_exception_handler=False,
                            pre_run=_prepare_prompt,
                        )
                except asyncio.CancelledError:
                    raise
                except KeyboardInterrupt:
                    self._input_exit_expected = True
                    self._request_interrupt("cancel")
                    return
                except EOFError:
                    if self._handle_input_eof():
                        self._input_exit_expected = True
                        return
                    continue
                except Exception as exc:
                    # Defense in depth: if a future prompt_toolkit version
                    # ever does propagate this race through the coroutine
                    # instead of the loop exception handler above, still
                    # treat it as an ordinary interrupt rather than crashing.
                    if "Return value already set" in str(exc) or "Application.exit()" in str(exc):
                        self._input_exit_expected = True
                        self._request_interrupt("cancel")
                        return
                    raise

                # app.exit(result="") is an internal teardown signal, not a
                # submission.  Check the marker before _paused because a
                # rapid resume may have flipped that flag by this point.
                forced_exit = self._prompt_exit_requested
                self._prompt_exit_requested = False
                if forced_exit or self._paused:
                    self._input_exit_expected = True
                    break
                if self._draft_text and not text.strip():
                    text, self._draft_text = self._draft_text, ""
                self._enqueue(text)
                if not self._active:
                    self._input_exit_expected = True
                    break
            else:
                # while-condition went false: stopped/paused from outside.
                self._input_exit_expected = True
        finally:
            loop.set_exception_handler(self._previous_loop_exception_handler)
            self._previous_loop_exception_handler = None
            if self._prompt_session is session:
                self._prompt_session = None

    async def _supervised_input_loop(self) -> None:
        """Keep a working composer for as long as the task runs.

        ``_input_loop`` used to be the whole story: any way of leaving it (EOF,
        an exception inside prompt_toolkit, a rendering error) ended keyboard
        ownership permanently while the task went on -- a frozen frame, cooked
        tty, Enter and Esc dead. Any exit the listener did not itself ask for is
        now logged, surfaced once, and the composer is restarted; if it cannot be
        kept alive, the tty echo is muted (so keys are never painted as ``^[``)
        and the user is told plainly instead of being shown a dead prompt.
        """
        try:
            while self._active and not self._paused and self._interrupt_classification is None:
                self._input_exit_expected = False
                _log.info("composer input loop begin")
                try:
                    await self._input_loop()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    _log.error("live composer failed: %s: %s", type(exc).__name__, exc)
                _log.info("composer input loop ended expected=%s active=%s paused=%s interrupt=%s",
                          self._input_exit_expected, self._active, self._paused, self._interrupt_classification)
                if (
                    not self._active
                    or self._paused
                    or self._interrupt_classification is not None
                ):
                    break
                if self._input_exit_expected:
                    # The loop ended for a reason it considered deliberate, yet the task is still
                    # running and nobody paused it: there would be no composer for the rest of the
                    # turn. Restart it (a deliberate stop always pauses or deactivates first).
                    _log.warning("composer ended 'expectedly' while the task is still active; restarting it")
                    await asyncio.sleep(0.05)
                    continue
                now = time.monotonic()
                self._input_restarts = [t for t in self._input_restarts if now - t < 10.0] + [now]
                if len(self._input_restarts) > 5:
                    _log.error("live composer could not be kept alive; giving up input for this turn")
                    self._terminal.mute_echo()
                    self.renderer.handle_event({
                        "event_type": "diagnostics",
                        "payload": {"content": (
                            "⚠ The message box stopped working — keys are not being read. The task is still "
                            "running; Ctrl+C stops it. Type your follow-up at the next prompt."
                        )},
                    })
                    break
                _log.warning("live composer ended unexpectedly; restarting it")
                await asyncio.sleep(0.1)
        finally:
            current = asyncio.current_task()
            if self._input_task is current:
                self._input_task = None

    def _stdin_closed(self) -> bool:
        try:
            if getattr(sys.stdin, "closed", False):
                return True
            return not os.isatty(sys.stdin.fileno())
        except Exception:
            return True

    def _handle_input_eof(self) -> bool:
        """True when EOF means the terminal is really gone (stop the task)."""
        now = time.monotonic()
        self._eof_times = [t for t in self._eof_times if now - t < 3.0] + [now]
        if self._stdin_closed() or len(self._eof_times) >= 3:
            _log.error("terminal input closed (EOF); stopping the running task")
            self.renderer.handle_event({
                "event_type": "diagnostics",
                "payload": {"content": "◆ Terminal input closed — stopping the running task."},
            })
            self._request_interrupt("exit")
            return True
        return False

    def _handle_prompt_loop_exception(self, loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        """Loop-level exception handler installed for the duration of each
        `session.prompt_async(set_exception_handler=False, ...)` call in
        `_input_loop` (see the comment there for why this has to be a loop
        exception handler rather than an ordinary try/except).

        Recognizes the one specific, benign race this exists for -- a queued
        cancel from another terminal and prompt_toolkit's own built-in
        Ctrl+C handler both calling `Application.exit()` in the same input
        cycle, where the second call raises "Return value already set" -- and
        resolves it as an ordinary interrupt. Anything else is handed to
        whatever handler was on the loop before this one (or asyncio's own
        default) so a genuine bug stays visible instead of being swallowed.
        """
        exc = context.get("exception")
        message = str(exc) if exc is not None else str(context.get("message") or "")
        if "Return value already set" in message or "Application.exit()" in message:
            self._request_interrupt("cancel")
            app = getattr(self._prompt_session, "app", None)
            if app is not None and not getattr(app, "is_done", False):
                with contextlib.suppress(Exception):
                    app.exit(result="")
            return
        previous_handler = self._previous_loop_exception_handler
        if previous_handler is not None:
            previous_handler(loop, context)
        else:
            loop.default_exception_handler(context)

    @property
    def interrupt_classification(self) -> Optional[str]:
        return self._interrupt_classification

    def _request_interrupt(self, classification: str) -> None:
        """Record and immediately propagate one terminal interrupt.

        The durable queue item remains useful for history and recovery, but
        immediate cancellation is delivered through interrupt_callback so a
        blocked provider request or long-running tool does not need to reach
        another orchestration boundary first.
        """
        if self._interrupt_classification is not None:
            return

        self._interrupt_classification = classification
        self._paused = True
        with contextlib.suppress(Exception):
            self.renderer.progress.request_cancel()
        if classification != "stall":
            # "stall" is the watchdog's own stop, not a user instruction: there is
            # nothing to persist for another terminal or the next turn to replay.
            self._enqueue_control(classification)

        if self._interrupt_callback is not None:
            self._interrupt_callback(classification)

    def _enqueue_control(self, classification: str) -> None:
        # A rapid second/third Ctrl+C while this write is still in flight
        # can raise a raw KeyboardInterrupt mid-call (confirmed live,
        # inside state.py's redact_secrets) -- this call site is itself
        # already reacting to the first interrupt, so a further one here
        # just reconfirms "stop", not a reason to crash the interrupt
        # handler with an ugly traceback. state.py's writes are tempfile+
        # os.replace atomic, so an interrupted write here cannot corrupt
        # state.json.
        with contextlib.suppress(KeyboardInterrupt):
            item = local_state.enqueue_instruction(
                self.session_id, "", classification=classification,
            )
            self.renderer.handle_event({
                "event_type": "diagnostics",
                "payload": {"content": f"◆ Queued {classification} for the running task ({item.id})."},
            })

    def _handle_live_model_command(self, text: str) -> bool:
        """Handle `/model ...` typed into the live in-task follow-up prompt.

        Previously every line typed here -- including `/model kimi-k3:cloud`
        -- was queued as an ordinary chat follow-up instruction, since only
        the top-level REPL loop (interactive.py) special-cased `/model`.
        That meant switching models required waiting for the running task
        to finish and returning to the plain prompt, unlike Shift+Tab's
        already-live approval-mode cycling. This mirrors interactive.py's
        standalone-runtime /model handler so the switch takes effect
        immediately (from the next turn/queued follow-up onward) without
        leaving the running task. Returns True if `text` was a /model
        command (handled or reported as a usage error either way), so the
        caller must not also enqueue it as a chat message.
        """
        if not (text == "/model" or text.startswith("/model ")):
            return False

        def _report(message: str) -> None:
            self.renderer.handle_event({
                "event_type": "diagnostics",
                "payload": {"content": f"◆ {message}"},
            })

        arg = text[len("/model"):].strip()
        state = local_state.get_session_state(self.session_id)
        if not arg:
            from .public_identity import public_model_name
            _report(f"model={public_model_name(state.selected_model)}")
            return True

        parts = arg.split()
        if parts[0].lower() == "auto":
            local_state.save_session_state(self.session_id, selected_model="auto", selected_provider=None)
            _report("Model set to TamfisGPT-Auto -- takes effect on the next turn.")
            return True

        if parts[0].lower() == "list":
            _report("Use the top-level `/model list` (outside a running task) to view TamfisGPT models.")
            return True

        from .local_chat import resolve_provider_type as _resolve_provider_type

        try:
            provider_type = _resolve_provider_type(parts[0])
        except ValueError as exc:
            _report("Unknown model. Use /model list to view TamfisGPT models.")
            return True
        del provider_type  # only validated here; resolve_route re-resolves it per call

        if len(parts) > 2:
            _report("Model ids cannot contain spaces.")
            return True

        model_id = parts[1] if len(parts) > 1 else "auto"
        local_state.save_session_state(
            self.session_id, selected_model=model_id, selected_provider=parts[0].lower(),
        )
        from .public_identity import public_model_name
        _report(f"Model set to {public_model_name(model_id)} -- takes effect on the next turn.")
        return True

    def _enqueue(self, text: str) -> None:
        text = strip_terminal_noise(text).strip()
        if text.lower() == "/update":
            pending = getattr(self.renderer, "pending_update_version", None)
            if not pending:
                self.renderer.handle_event({
                    "event_type": "diagnostics",
                    "payload": {"content": "◆ No Tamfis-Code update is currently available."},
                })
                if self._active and not self._paused:
                    self._schedule_prompt()
                return
            item = local_state.enqueue_instruction(
                self.session_id, "/update", classification="update", priority=0,
            )
            self.renderer.handle_event({
                "event_type": "diagnostics",
                "payload": {
                    "content": (
                        f"◆ Update {pending} queued ({item.id}). The current task will finish its "
                        "checkpoint, then Tamfis-Code will update and resume automatically."
                    ),
                },
            })
            if self._active and not self._paused:
                self._schedule_prompt()
            return
        if not text:
            if self._active and not self._paused:
                self._schedule_prompt()
            return
        if self._handle_btw_command(text):
            if self._active and not self._paused:
                self._schedule_prompt()
            return
        if self._handle_live_model_command(text):
            if self._active and not self._paused:
                self._schedule_prompt()
            return
        if self._handle_status_command(text):
            if self._active and not self._paused:
                self._schedule_prompt()
            return
        if self._handle_live_slash_command(text):
            if self._active and not self._paused:
                self._schedule_prompt()
            return
        if self._handle_live_scope_correction(text):
            return
        editing_id = self._editing_instruction_id
        self._editing_instruction_id = None
        if editing_id and local_state.edit_queued_instruction(self.session_id, editing_id, text):
            self.renderer.handle_event({
                "event_type": "diagnostics",
                "payload": {
                    "content": (
                        f"◆ Updated queued instruction {editing_id}: {text} "
                        "-- applied at the next safe round boundary."
                    ),
                },
            })
            if self._active and not self._paused:
                self._schedule_prompt()
            return
        item = local_state.enqueue_instruction(
            self.session_id, text, classification="follow_up",
        )
        request_steering = getattr(self.renderer, "request_steering", None)
        if callable(request_steering):
            request_steering()
        self.renderer.handle_event({
            "event_type": "user_message",
            "payload": {"content": text},
        })
        _log.info("follow-up queued id=%s session=%s chars=%d", item.id, self.session_id, len(text))
        self.renderer.handle_event({
            "event_type": "diagnostics",
            "payload": {
                "content": (
                    f"↳ Follow-up queued ({item.id}): {_truncate(text, 120)} "
                    "-- the running task picks it up at the next safe step."
                ),
            },
        })
        self._start_followup_ack(text)
        if self._active and not self._paused:
            self._schedule_prompt()

    def live_status_lines(self) -> list[str]:
        """Facts about the running task, for `/status` typed mid-task.

        Deliberately mechanical: state, step, provider (public product name only),
        time since the last meaningful progress, pending tools and follow-ups. No
        keys, no internal route identifiers, no model reasoning.
        """
        from .public_identity import public_model_name

        progress = self.renderer.progress
        snap = progress.snapshot()
        steps = [
            item for item in (getattr(self.renderer, "_plan_steps", None) or [])
            if isinstance(item, dict) and item.get("step")
        ]
        active = next((i for i in steps if str(i.get("status")) == "in_progress"), None) \
            or next((i for i in steps if str(i.get("status") or "pending") == "pending"), None)
        step_text = _truncate(active["step"], 100) if active else _truncate(
            self.renderer.current_activity(include_command=False), 100,
        )
        done = sum(1 for i in steps if str(i.get("status")) == "completed")
        model = getattr(self.renderer, "_model", None)
        pending = 0
        with contextlib.suppress(Exception):
            pending = max(0, self.renderer.steering_revision() - self.renderer._steering_handled_revision)
        queued_total = pending
        with contextlib.suppress(Exception):
            queued_total = max(pending, sum(
                1 for item in local_state.get_session_state(self.session_id).queued_user_instructions
                if item.get("status") == "queued"
            ))
        lines = [
            f"Current step: {step_text}",
            f"State: {snap.label}",
            f"Provider: {public_model_name(model) if model else 'TamfisGPT'}",
            f"Last progress: {_format_ago(snap.idle_seconds)} ({snap.last_event})",
            f"Pending tools: {snap.pending_tools}",
            f"Queued follow-ups: {queued_total}",
        ]
        if steps:
            lines.insert(1, f"Plan: {done}/{len(steps)} steps done")
        return lines

    def _handle_status_command(self, text: str) -> bool:
        """`/status` typed into the live composer answers locally, at once.

        It used to be queued as a steering message ("You: /status") and handed to
        the model, which could not answer it while blocked on a provider call.
        """
        if text.lower() != "/status":
            return False
        for line in self.live_status_lines():
            self.renderer.handle_event({
                "event_type": "diagnostics", "payload": {"content": f"◆ {line}"},
            })
        return True

    # Commands the live composer can honour at once. Anything else that is shaped like a slash command is
    # kept OUT of the model's follow-up stream (the model used to receive the literal text "/diff" as
    # prose, so the command silently did nothing) and is deferred to run as a command when the task ends.
    _LIVE_STOP_COMMANDS = {
        "/stop": "cancel", "/cancel": "cancel", "/pause": "pause",
        "/interrupt": "cancel", "/exit": "exit", "/quit": "exit",
    }

    def _handle_live_slash_command(self, text: str) -> bool:
        """A `/command` typed mid-task is never silent and never becomes model prose.

        Runs at once when it can (stop/pause, /help, /queue), otherwise says so and queues it as a
        *command* for the moment the task finishes. Text that merely looks like a path or a sentence
        (``/home/x/y.py``, ``/ what about``) is not a command and flows through as a follow-up.
        """
        parts = text.split(None, 1)
        head = parts[0].lower() if parts else ""
        if not re.fullmatch(r"/[a-z][a-z0-9_-]*", head):
            return False

        def note(content: str) -> None:
            self.renderer.handle_event({"event_type": "diagnostics", "payload": {"content": f"◆ {content}"}})

        if head in self._LIVE_STOP_COMMANDS:
            classification = self._LIVE_STOP_COMMANDS[head]
            note(
                "Stopping the task now…" if classification in {"cancel", "exit"}
                else "Pausing the task at the next safe step…"
            )
            self._request_interrupt(classification)
            return True
        if head == "/help":
            note("While a task runs: /status, /recap, /btw <question>, /model, /queue, /stop, /pause, /update run at once; "
                 "any other /command is queued and runs when the task finishes. Plain messages steer the task.")
            return True
        if head == "/recap":
            from .return_recap import build_return_recap

            recap = build_return_recap(self.session_id)
            if recap is None:
                note("Nothing to recap yet: this session has no recorded conversation.")
            else:
                note(f"Objective: {recap.objective}")
                note(f"Where it stands: {recap.standing}")
                note(f"Next: {recap.next_step}")
            return True
        if head == "/queue":
            items = [
                i for i in local_state.get_session_state(self.session_id).queued_user_instructions
                if i.get("status") == "queued" and str(i.get("text") or "").strip()
            ]
            if not items:
                note("Nothing is queued.")
            for item in items[:10]:
                note(f"queued {item.get('id')} ({item.get('classification')}): {_truncate(str(item.get('text')), 100)}")
            return True

        from .interactive import SLASH_COMMANDS

        known = {name.lower() for name, _ in SLASH_COMMANDS}
        if head not in known:
            close = difflib.get_close_matches(head, sorted(known), n=1)
            note(f"Unknown command {head}" + (f" -- did you mean {close[0]}?" if close else ". Type /help for what works mid-task."))
            return True
        item = local_state.enqueue_instruction(self.session_id, text, classification="command", priority=50)
        note(f"{head} can't run inside a running task. Queued ({item.id}): it runs as soon as the task finishes.")
        return True

    def _handle_live_scope_correction(self, text: str) -> bool:
        """Stop before the next tool call when the user narrows the workspace.

        Scope is fixed for a running turn; a steering message cannot safely
        shrink already-created tool authority. A clear ``only /path``
        correction is queued for the next turn and interrupts this one first.
        """
        if not self._active or self._paused:
            return False
        if not re.search(r"\b(?:only|just|confine|restrict|limit)\s+/(?:\S+)", text, re.IGNORECASE):
            return False
        try:
            from .runtime.workspace_authority import explicit_absolute_targets

            targets = explicit_absolute_targets(text)
        except Exception:
            return False
        if len(targets) != 1 or not targets[0].is_dir():
            return False
        item = local_state.enqueue_instruction(
            self.session_id, text, classification="follow_up", priority=150,
        )
        self.renderer.handle_event({
            "event_type": "diagnostics",
            "payload": {
                "content": (
                    f"◆ Workspace scope narrowed to {targets[0]}. "
                    f"Stopping this turn ({item.id}); the correction will restart in the narrowed scope."
                )
            },
        })
        self._request_interrupt("cancel")
        return True

    # ---- immediate acknowledgement of a mid-task follow-up (a concurrent branch, not a wait) ----

    _FOLLOWUP_ACK_TIMEOUT_SECONDS = 45.0

    def _start_followup_ack(self, text: str) -> None:
        """Answer a mid-task follow-up straight away from a side branch.

        The main task only sees the follow-up at its next safe step, which can be minutes away; until
        then the user got a one-line "queued" and silence. This branch reads the follow-up beside the
        running task's state and says what it understood, how it will be applied and whether it
        changes the plan -- asking one question when it is ambiguous -- while the task keeps going.
        Best effort: any failure leaves just the "queued" line.
        """
        if self._side_question_callback is None or os.environ.get("TAMFIS_CODE_FOLLOWUP_ACK", "1").strip().lower() in {"0", "false", "no", "off"}:
            return
        task = asyncio.create_task(self._answer_followup(text))
        self._btw_tasks.add(task)
        task.add_done_callback(self._btw_tasks.discard)

    async def _call_side_callback(self, question: str, *, followup: bool) -> str:
        callback = self._side_question_callback
        assert callback is not None
        try:
            accepts = "followup" in inspect.signature(callback).parameters
        except (TypeError, ValueError):
            accepts = False
        result = callback(question, followup=True) if (followup and accepts) else callback(question)  # type: ignore[call-arg]
        return await result

    async def _answer_followup(self, text: str) -> None:
        status = "\n".join(self.live_status_lines()[:8])
        question = f"The user's follow-up message:\n{text}\n\nWhat the running task is doing right now:\n{status}"
        try:
            answer = (await asyncio.wait_for(
                self._call_side_callback(question, followup=True), timeout=self._FOLLOWUP_ACK_TIMEOUT_SECONDS,
            )).strip()
            if answer:
                self.renderer.handle_event({
                    "event_type": "side_question_answer",
                    "payload": {"question": text, "content": answer, "kind": "followup"},
                })
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _log.info("follow-up acknowledgement unavailable: %s", exc)

    def _handle_btw_command(self, text: str) -> bool:
        """Dispatch `/btw` outside the active turn's steering queue.

        The callback uses an independent, read-only model request. This is
        the key behavioural guarantee: a side question cannot cancel,
        replace, reprioritise, or append instructions to the running task.
        """
        if not (text.lower() == "/btw" or text.lower().startswith("/btw ")):
            return False

        question = text[len("/btw"):].strip()
        if not question:
            self.renderer.handle_event({
                "event_type": "diagnostics",
                "payload": {"content": "◆ Usage: /btw <quick side question>"},
            })
            return True
        if self._side_question_callback is None:
            self.renderer.handle_event({
                "event_type": "diagnostics",
                "payload": {"content": "◆ /btw is unavailable for this task connection."},
            })
            return True

        task = asyncio.create_task(self._answer_btw(question))
        self._btw_tasks.add(task)
        task.add_done_callback(self._btw_tasks.discard)
        self.renderer.handle_event({
            "event_type": "diagnostics",
            "payload": {"content": "◆ Answering /btw separately; the active task is still running."},
        })
        return True

    async def _answer_btw(self, question: str) -> None:
        try:
            answer = (await self._side_question_callback(question)).strip()  # type: ignore[misc]
            if not answer:
                raise RuntimeError("The side-question model returned no visible answer.")
            self.renderer.handle_event({
                "event_type": "side_question_answer",
                "payload": {"question": question, "content": answer},
            })
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.renderer.handle_event({
                "event_type": "diagnostics",
                "payload": {"content": f"◆ /btw failed: {exc}"},
            })

    def _recall_latest_queued(self, buffer) -> bool:
        """Load the newest editable queue item into an empty live composer."""
        items = local_state.get_session_state(self.session_id).queued_user_instructions
        editable = next((
            item for item in reversed(items)
            if item.get("status") == "queued"
            and item.get("classification") in {"append", "follow_up"}
            and str(item.get("text") or "").strip()
        ), None)
        if editable is None:
            return False
        text = str(editable["text"])
        buffer.text = text
        buffer.cursor_position = len(text)
        self._editing_instruction_id = str(editable["id"])
        return True

    async def _interject(self) -> None:
        """Compatibility helper for callers/tests that submit one line."""
        from prompt_toolkit import PromptSession

        try:
            with responsive_patch_stdout(raw=True):
                text = await PromptSession().prompt_async("message> ")
        except KeyboardInterrupt:
            self._enqueue_control("exit")
            return
        except EOFError:
            text = ""
        self._enqueue(text)
