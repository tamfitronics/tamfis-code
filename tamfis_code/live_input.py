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

    def stop(self) -> None:
        self._active = False
        self._cancel_prompt()
        if self.renderer.live_input_listener is self:
            self.renderer.live_input_listener = None
            self.renderer.resume_live()

    def pause(self) -> None:
        self._paused = True
        self._cancel_prompt()

    def resume(self) -> None:
        self._paused = False
        if self._active:
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

    def _cancel_prompt(self) -> None:
        task = self._input_task
        self._input_task = None
        if task is not None and not task.done():
            task.cancel()

    def invalidate(self) -> None:
        """Refresh the prompt footer after a streamed phase/status update."""
        app = getattr(self._prompt_session, "app", None)
        if app is not None:
            app.invalidate()

    def _bottom_toolbar(self):
        status = self.renderer.live_input_status()
        mode = mode_label_for_policy(self.cli_config.approval_policy)
        return HTML(
            f"<b><ansicyan>◆</ansicyan></b> <b>{status}</b>  "
            f"<ansigray>│ {mode} │ esc interrupt │ shift+tab mode</ansigray>"
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
        while self._active and not self._paused:
            try:
                # patch_stdout keeps concurrent tool/assistant output above
                # the current line and redraws the line beneath it.
                with patch_stdout(raw=True):
                    text = await session.prompt_async(
                        "message> ", bottom_toolbar=self._bottom_toolbar,
                    )
            except asyncio.CancelledError:
                return
            except KeyboardInterrupt:
                # Ctrl+C is the process/REPL exit affordance. The active
                # runner's signal path remains reserved for true process
                # interrupts; queue an explicit exit so the local runner
                # can finish its current safe boundary and the REPL exits.
                self._enqueue_control("exit")
                return
            except EOFError:
                return

            # Escape sets _paused before exiting the prompt, so its empty
            # result must not be queued or cause another editor to start.
            if self._paused:
                break

            # A genuine line already returned by prompt_async remains valid
            # even when the listener was deactivated during that prompt.
            # Queue it first, then let the loop terminate cleanly.
            self._enqueue(text)

            if not self._active:
                break

        self._prompt_session = None

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

    def _enqueue(self, text: str) -> None:
        text = text.strip()
        if not text:
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
