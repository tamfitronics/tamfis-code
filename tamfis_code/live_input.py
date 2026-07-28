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
from typing import Callable, Optional

from prompt_toolkit.formatted_text import HTML

from . import state as local_state
from .config import Config, mode_label_for_policy, next_mode_in_cycle
from .render import StreamRenderer

_SHIFT_TAB = b"\x1b[Z"
# Retained only for backwards-compatible imports. Ctrl+Y is no longer read
# specially by the live listener; it is ordinary editable prompt input.
_CTRL_T = b"\x14"
_CTRL_Y = b"\x19"

# Claude Code's own bottom-toolbar phrasing for the three MODE_CYCLE stops
# that actually change what gets auto-approved -- "manual" (/mode's "ask")
# is the quiet default and gets no banner, matching how Claude Code only
# announces the modes that suppress a prompt.
_MODE_ON_LABEL = {
    "accept-edits": "auto-accept edits on",
    "auto": "auto mode on",
    "plan-only": "plan mode on",
}


class _CompletedAwaitable:
    """A no-op awaitable that is also safe to ignore for non-TTY callers."""

    def __await__(self):
        if False:
            yield None
        return None


def _active_agent_count(exclude_session_id: int) -> int:
    """Count other known sessions currently mid-task (e.g. swarm children
    delegated via /delegate), for the "N agents" toolbar suffix."""
    count = 0
    for sid in local_state.all_known_session_ids():
        if sid == exclude_session_id:
            continue
        if local_state.get_session_state(sid).execution_status == "running":
            count += 1
    return count


def _mode_and_agents_html(cli_config: Config, session_id: int) -> str:
    mode = mode_label_for_policy(cli_config.approval_policy)
    mode_on = _MODE_ON_LABEL.get(mode)
    mode_line = (
        f"<ansiyellow>⏵⏵ {mode_on} (shift+tab to cycle)</ansiyellow>"
        if mode_on else "<ansigray>shift+tab to cycle mode</ansigray>"
    )
    agents = _active_agent_count(session_id)
    agents_suffix = f" <ansigray>· ← {agents} agent{'s' if agents != 1 else ''}</ansigray>" if agents else ""
    return f"{mode_line}{agents_suffix}"


def idle_bottom_toolbar(cli_config: Config, session_id: int) -> HTML:
    """Bottom-toolbar content for the plain REPL prompt (no task running) --
    same mode/agents banner as the live in-task footer below, so the bar
    doesn't disappear the moment a turn finishes."""
    return HTML(_mode_and_agents_html(cli_config, session_id))


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
        self._paused = False
        self._active = False

    def start(self) -> None:
        if not self._is_tty:
            return
        # Stop Rich's repainting while the prompt owns the terminal. Streamed
        # assistant text is intentionally rendered as scrollback in this
        # mode, so the input line and mouse scrolling never compete.
        self.renderer.suspend_live()
        self.renderer.live_input_listener = self
        self._active = True
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
        await self._shutdown_prompt()
        if self.renderer.live_input_listener is self:
            self.renderer.live_input_listener = None
        self.renderer.resume_live()

    def pause(self) -> None:
        """Synchronously request prompt shutdown before another UI reads stdin."""
        self._paused = True
        self._request_prompt_exit()
        self._cancel_prompt()

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
            self.cli_config.approval_policy = next_mode_in_cycle(self.cli_config.approval_policy)
            self.renderer.handle_event({
                "event_type": "diagnostics",
                "payload": {"content": f"◆ Mode switched to {mode_label_for_policy(self.cli_config.approval_policy)}."},
            })
            if hasattr(self.renderer, "set_mode_label"):
                self.renderer.set_mode_label(mode_label_for_policy(self.cli_config.approval_policy))
        elif buf in {b"\x1b", b"\x1b["}:
            return
        elif _CTRL_Y in buf:
            self._buf = bytearray()
            if self._interject_task is None or self._interject_task.done():
                self._interject_task = asyncio.create_task(self._interject())
        elif buf:
            self._buf = bytearray()

    def _schedule_prompt(self) -> None:
        if self._input_task is None or self._input_task.done():
            self._input_task = asyncio.create_task(self._input_loop())

    def _request_prompt_exit(self) -> None:
        session = self._prompt_session
        app = getattr(session, "app", None)
        if app is not None and not getattr(app, "is_done", False):
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
            task.cancel()
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._prompt_session = None

    def invalidate(self) -> None:
        """Refresh the prompt footer after a streamed phase/status update."""
        app = getattr(self._prompt_session, "app", None)
        if app is not None:
            app.invalidate()

    def _bottom_toolbar(self):
        status = self.renderer.live_input_status()
        return HTML(
            f"<b><ansicyan>◆</ansicyan></b> <b>{status}</b>  "
            f"<ansigray>│ esc interrupt │</ansigray> {_mode_and_agents_html(self.cli_config, self.session_id)}"
        )

    async def _input_loop(self) -> None:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.patch_stdout import patch_stdout

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

        @bindings.add("escape")
        def _cancel_running_turn(event) -> None:
            # Escape cancels the active turn immediately and returns control
            # to the ordinary REPL without terminating Tamfis-Code.
            if self._paused or not self._active:
                return

            self._request_interrupt("cancel")

            if not event.app.is_done:
                event.app.exit(result="")

        session = PromptSession(key_bindings=bindings)
        self._prompt_session = session
        try:
            while self._active and not self._paused:
                try:
                    with patch_stdout(raw=True):
                        text = await session.prompt_async(
                            "message> ", bottom_toolbar=self._bottom_toolbar,
                        )
                except asyncio.CancelledError:
                    raise
                except KeyboardInterrupt:
                    self._request_interrupt("cancel")
                    return
                except EOFError:
                    return

                if self._paused:
                    break
                self._enqueue(text)
                if not self._active:
                    break
        finally:
            if self._prompt_session is session:
                self._prompt_session = None
            current = asyncio.current_task()
            if self._input_task is current:
                self._input_task = None

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
            _report(f"model={state.selected_model}  provider={state.selected_provider or 'auto'}")
            return True

        parts = arg.split()
        if parts[0].lower() == "auto":
            local_state.save_session_state(self.session_id, selected_model="auto", selected_provider=None)
            _report("Provider routing set to automatic -- takes effect on the next turn.")
            return True

        if parts[0].lower() == "list":
            _report("Use the top-level `/model list` (outside a running task) to browse full model tables.")
            return True

        from .local_chat import resolve_provider_type as _resolve_provider_type

        try:
            provider_type = _resolve_provider_type(parts[0])
        except ValueError as exc:
            _report(f"{exc} Usage: /model auto | /model <tamfis|hf|nvidia|openrouter> [model-id]")
            return True
        del provider_type  # only validated here; resolve_route re-resolves it per call

        if len(parts) > 2:
            _report("Model ids cannot contain spaces.")
            return True

        model_id = parts[1] if len(parts) > 1 else "auto"
        local_state.save_session_state(
            self.session_id, selected_model=model_id, selected_provider=parts[0].lower(),
        )
        _report(f"Pinned {parts[0].lower()} route, model={model_id} -- takes effect on the next turn.")
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
        item = local_state.enqueue_instruction(
            self.session_id, text, classification="follow_up",
        )
        self.renderer.handle_event({
            "event_type": "user_message",
            "payload": {"content": text},
        })
        self.renderer.handle_event({
            "event_type": "diagnostics",
            "payload": {
                "content": (
                    f"◆ Queued next instruction {item.id}: {text} "
                    "-- applied at the next safe round boundary."
                ),
            },
        })
        if self._active and not self._paused:
            self._schedule_prompt()

    async def _interject(self) -> None:
        """Compatibility helper for callers/tests that submit one line."""
        from prompt_toolkit import PromptSession
        from prompt_toolkit.patch_stdout import patch_stdout

        try:
            with patch_stdout(raw=True):
                text = await PromptSession().prompt_async("message> ")
        except KeyboardInterrupt:
            self._enqueue_control("exit")
            return
        except EOFError:
            text = ""
        self._enqueue(text)
