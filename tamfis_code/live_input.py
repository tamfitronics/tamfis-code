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
import sys
import time
from typing import Any, Callable, Optional

from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style

from . import state as local_state
from .config import Config, mode_label_for_policy, next_mode_in_cycle
from .render import StreamRenderer

_SHIFT_TAB = b"\x1b[Z"
# Retained only for backwards-compatible imports. Ctrl+Y is no longer read
# specially by the live listener; it is ordinary editable prompt input.
_CTRL_T = b"\x14"
_CTRL_Y = b"\x19"
_STATUS_SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
_STATUS_REFRESH_INTERVAL_SECONDS = 0.25
_STDOUT_BATCH_INTERVAL_SECONDS = 0.01

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
    (lambda state, agents: not state.active_plan_id, "/plan to think before executing"),
    (_ALWAYS, "/model to switch models"),
    (lambda state, agents: bool(state.modified_files), "/diff to review pending changes"),
    (lambda state, agents: agents > 0, "/agents to see what's running"),
    (lambda state, agents: bool(state.conversation_history), "/retry to rerun the last turn"),
    (lambda state, agents: bool(state.unresolved_issues), "/doctor to check unresolved issues"),
    (_ALWAYS, "/status for session, task, cwd"),
)
_TIP_ROTATE_SECONDS = 8.0


def _right_chip(session_id: Optional[int] = None, active_agents: int = 0) -> str:
    # ansibrightblack (the ghost-text/auto-suggestion color -- deliberately
    # dim) renders as unreadable-to-invisible against some terminal themes'
    # backgrounds for ordinary body text. ansigray is the same tone the rest
    # of this toolbar's left-side status already uses, so the chip stays
    # visually secondary without disappearing.
    if session_id is None:
        applicable = [text for predicate, text in _ROTATING_TIPS if predicate is _ALWAYS]
    else:
        state = local_state.get_session_state(session_id)
        applicable = [
            text for predicate, text in _ROTATING_TIPS if predicate(state, active_agents)
        ]
    if not applicable:
        applicable = [text for _, text in _ROTATING_TIPS]
    index = int(time.monotonic() // _TIP_ROTATE_SECONDS) % len(applicable)
    return f"<ansigray>{applicable[index]}</ansigray>"


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


def _mode_and_agents_html(
    cli_config: Config,
    session_id: int,
    *,
    active_agents: Optional[int] = None,
) -> str:
    mode = mode_label_for_policy(cli_config.approval_policy)
    mode_on = _MODE_ON_LABEL.get(mode)
    mode_line = (
        f"<ansiyellow>⏵⏵ {mode_on.replace(' mode on', '').replace(' on', '')} · shift+tab</ansiyellow>"
        if mode_on
        else f"<ansigray>⏵⏵ {mode} · shift+tab</ansigray>"
    )
    agents = (
        _active_agent_count(session_id)
        if active_agents is None
        else active_agents
    )
    agents_suffix = f" <ansigray>· ← {agents} agent{'s' if agents != 1 else ''}</ansigray>" if agents else ""
    return f"{mode_line}{agents_suffix}"


def idle_bottom_toolbar(
    cli_config: Config,
    session_id: int,
    *,
    provider: str = "auto",
    model: Optional[str] = None,
    has_suggestion: bool = False,
    active_agents: Optional[int] = None,
) -> HTML:
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
    left = (
        f" <ansigray>ready · {public_model_name(model)} ·</ansigray> "
        f"{_mode_and_agents_html(cli_config, session_id, active_agents=resolved_agents)}"
        f"{suggestion_hint}"
    )
    chip = _right_chip(session_id, resolved_agents)
    return HTML(_right_align(left, chip + " "))


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
    ) -> None:
        self.session_id = session_id
        self.renderer = renderer
        self.cli_config = cli_config
        self._interrupt_callback = interrupt_callback
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
        self._outcome_status: Optional[str] = None
        self._editing_instruction_id: Optional[str] = None
        # Distinguish a programmatic prompt shutdown (approval/tool UI,
        # turn completion) from a user pressing Enter.  Without this marker,
        # prompt_toolkit returns the current buffer through ``prompt_async``
        # and a fast pause/resume race can enqueue half-typed text.
        self._prompt_exit_requested = False
        self._draft_text = ""
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
        if self.renderer.live_input_listener is self:
            self.renderer.live_input_listener = None
        if self._outcome_status:
            # The turn is over: do not restart Rich's transient live spinner
            # for the few milliseconds before renderer.finish(). Replace the
            # animated footer directly with one durable timing summary.
            if hasattr(self.renderer, "print_work_summary"):
                self.renderer.print_work_summary(self._outcome_status)
        else:
            self.renderer.resume_live()

    def set_outcome_status(self, status: str) -> None:
        self._outcome_status = status

    async def _status_ticker(self) -> None:
        """Animate and update elapsed time even during silent provider waits."""
        try:
            while self._active:
                await asyncio.sleep(_STATUS_REFRESH_INTERVAL_SECONDS)
                self._status_tick = (self._status_tick + 1) % len(_STATUS_SPINNER_FRAMES)
                self.invalidate()
        except asyncio.CancelledError:
            raise

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
        self._paused = True
        self._request_prompt_exit()
        self._cancel_prompt()

    async def pause_async(self) -> None:
        """Like `pause`, but waits until the old prompt has actually released
        the terminal before returning. Use this (not `pause`) whenever a new
        PromptSession/Application is about to be opened right after -- e.g.
        immediately before an approval gate prompt."""
        self._paused = True
        self._request_prompt_exit()
        self._cancel_prompt()
        task = self._input_task
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    def resume(self) -> None:
        self._paused = False
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
            self._input_task = asyncio.create_task(self._input_loop())

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

    def _bottom_toolbar(self):
        spinner = _STATUS_SPINNER_FRAMES[self._status_tick]
        status = self.renderer.live_input_status(spinner)
        # FIX (2026-08-21): a multi-line function call inside an f-string
        # expression (the {...} spanning several lines) is PEP 701 syntax,
        # Python 3.12+ only -- pyproject.toml declares requires-python
        # ">=3.10" and this was a real SyntaxError on 3.10/3.11, invisible
        # locally on this host's 3.13 interpreter and never caught by CI
        # because CI never actually ran pytest until today's fix wired that
        # in (see the CI-gating commit history). Computing the value first
        # is unambiguous across every supported Python version.
        mode_and_agents_html = _mode_and_agents_html(
            self.cli_config,
            self.session_id,
            active_agents=self._active_agents,
        )
        left = (
            f" <ansigray>{status} · ↑ edit queued · esc to interrupt ·</ansigray> "
            f"{mode_and_agents_html}"
        )
        chip = _right_chip(self.session_id, self._active_agents)
        bottom_line = _right_align(left, chip + " ")
        activity = self.renderer.live_input_activity_line()
        if not activity:
            return HTML(bottom_line)
        from xml.sax.saxutils import escape as _xml_escape
        activity_line = f" <ansigray>{_xml_escape(activity)}</ansigray>"
        return HTML(f"{activity_line}\n{bottom_line}")

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

        @bindings.add("escape")
        def _cancel_running_turn(event) -> None:
            # Escape cancels the active turn immediately and returns control
            # to the ordinary REPL without terminating Tamfis-Code.
            if self._paused or not self._active:
                return

            self._request_interrupt("cancel")

            if not event.app.is_done:
                event.app.exit(result="")

        # Keep the same boxed composer while a task is running. Replacing the
        # idle editor with a bare "message>" line made the interface jump
        # between two unrelated layouts and hid the footer hierarchy.
        session = PromptSession(
            key_bindings=bindings,
            show_frame=True,
            reserve_space_for_menu=0,
            style=composer_style(),
            auto_suggest=_LiveProgressAutoSuggest(self.renderer),
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
                try:
                    with responsive_patch_stdout(raw=True):
                        text = await session.prompt_async(
                            "message› ",
                            bottom_toolbar=self._bottom_toolbar,
                            show_frame=True,
                            set_exception_handler=False,
                            pre_run=_prepare_prompt,
                        )
                except asyncio.CancelledError:
                    raise
                except KeyboardInterrupt:
                    self._request_interrupt("cancel")
                    return
                except EOFError:
                    return
                except Exception as exc:
                    # Defense in depth: if a future prompt_toolkit version
                    # ever does propagate this race through the coroutine
                    # instead of the loop exception handler above, still
                    # treat it as an ordinary interrupt rather than crashing.
                    if "Return value already set" in str(exc) or "Application.exit()" in str(exc):
                        self._request_interrupt("cancel")
                        return
                    raise

                # app.exit(result="") is an internal teardown signal, not a
                # submission.  Check the marker before _paused because a
                # rapid resume may have flipped that flag by this point.
                forced_exit = self._prompt_exit_requested
                self._prompt_exit_requested = False
                if forced_exit or self._paused:
                    break
                if self._draft_text and not text.strip():
                    text, self._draft_text = self._draft_text, ""
                self._enqueue(text)
                if not self._active:
                    break
        finally:
            loop.set_exception_handler(self._previous_loop_exception_handler)
            self._previous_loop_exception_handler = None
            if self._prompt_session is session:
                self._prompt_session = None
            current = asyncio.current_task()
            if self._input_task is current:
                self._input_task = None

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
        self._enqueue_control(classification)

        if self._interrupt_callback is not None:
            self._interrupt_callback(classification)

    def _enqueue_control(self, classification: str) -> None:
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
            _report("Model set to TamfisGPT Auto -- takes effect on the next turn.")
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
        text = text.strip()
        if not text:
            if self._active and not self._paused:
                self._schedule_prompt()
            return
        if self._handle_live_model_command(text):
            if self._active and not self._paused:
                self._schedule_prompt()
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
        self.renderer.handle_event({
            "event_type": "diagnostics",
            "payload": {
                "content": (
                    f"◆ Steering update sent {item.id}: {text} "
                    "-- applying it to the active task now."
                ),
            },
        })
        if self._active and not self._paused:
            self._schedule_prompt()

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
