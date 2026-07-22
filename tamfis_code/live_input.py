"""Same-terminal, same-process live input while a standalone agent turn is
actively streaming.

Two recognised keys, both usable at any moment during a running turn, not
just when an approval prompt happens to be showing:

- Shift+Tab cycles `cli_config.approval_policy` immediately -- the exact
  same live Config object interactive.py's prompt indicator and
  runner.py's approval-gate Shift+Tab handler (`_prompt`'s own binding)
  already read from and write to, so this is not a second, competing mode
  concept, just another writer of the same value.
- Ctrl+Y opens a real line editor for the next instruction and queues it
  through the on-disk queue a second
  terminal's `tamfis-code queue "..."` already used
  (runner_local.py's `_claim_live_queued_instructions` polls it at the top
  of every round) -- this only adds an in-process producer for
  infrastructure that already existed; no new queue plumbing. The prompt
  itself runs as a real `PromptSession.prompt_async()` wrapped in
  prompt_toolkit's own `patch_stdout(raw=True)`, not a blocking `input()`
  -- so any concurrent console output the still-running turn produces
  while the human is mid-line gets safely inserted above the prompt and
  redrawn beneath it, instead of interleaving with/corrupting the typed
  text. Confirmed against a real pty, not just a mock.

Deliberately narrow: exactly these two keys. Every other keypress while a
turn is streaming is silently dropped -- there is no line editor to hand it
to while Rich's Live display owns the terminal, and before this listener
existed that input was equally unusable, just buffered invisibly in the
kernel tty queue instead of being read at all.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import sys
from typing import Optional

from . import state as local_state
from .config import Config, mode_label_for_policy, next_mode_in_cycle
from .render import StreamRenderer

try:
    import termios
    import tty
    _TTY_AVAILABLE = True
except ImportError:  # Windows -- no termios/tty; this feature just disables itself.
    termios = None  # type: ignore
    tty = None  # type: ignore
    _TTY_AVAILABLE = False

_SHIFT_TAB = b"\x1b[Z"
# Ctrl+T is reserved by Termius for opening a new terminal. Keep the queue
# shortcut on Ctrl+Y by default, while allowing an operator to choose a local
# key explicitly when another terminal client owns it.
_CTRL_T = b"\x14"  # legacy/exported test constant; no longer the default
_CTRL_Y = b"\x19"
_QUEUE_KEY_ENV = "TAMFIS_CODE_QUEUE_KEY"


def queue_key_bytes() -> tuple[bytes, str]:
    value = os.environ.get(_QUEUE_KEY_ENV, "ctrl-y").strip().lower().replace(" ", "")
    if value in {"ctrl-t", "^t"}:
        return _CTRL_T, "Ctrl+T"
    if value in {"ctrl-y", "^y", ""}:
        return _CTRL_Y, "Ctrl+Y"
    # Single control letters are convenient for terminal clients without
    # reliable multi-key escape support: ctrl-a through ctrl-z.
    if value.startswith("ctrl-") and len(value) == 6 and "a" <= value[-1] <= "z":
        return bytes([ord(value[-1]) - ord("a") + 1]), f"Ctrl+{value[-1].upper()}"
    return _CTRL_Y, "Ctrl+Y"
_ESCAPE_PREFIXES = (b"\x1b", b"\x1b[")


class LiveInputListener:
    """Started once per interactive turn (see interactive.py's call sites);
    a clean no-op off a real TTY or on a platform without termios."""

    def __init__(self, *, session_id: int, renderer: StreamRenderer, cli_config: Config) -> None:
        self.session_id = session_id
        self.renderer = renderer
        self.cli_config = cli_config
        self._is_tty = _TTY_AVAILABLE and sys.stdin.isatty()
        self._fd: Optional[int] = None
        self._old_termios: Optional[list] = None
        self._buf = bytearray()
        self._active = False  # currently reading raw bytes (vs paused)
        self._interject_task: Optional[asyncio.Task] = None

    def start(self) -> None:
        if not self._is_tty:
            return
        self._fd = sys.stdin.fileno()
        self._old_termios = termios.tcgetattr(self._fd)
        self._enable()
        self.renderer.live_input_listener = self

    def stop(self) -> None:
        if not self._is_tty or self._fd is None:
            return
        self._disable()
        self.renderer.live_input_listener = None
        self._fd = None

    def _enable(self) -> None:
        # cbreak (not raw): drops ICANON/ECHO so a single keypress is
        # readable immediately and doesn't echo into Rich's Live redraw,
        # but keeps ISIG so Ctrl+C still raises SIGINT exactly as it does
        # today (runner.py's own _install_sigint_watcher handles that).
        tty.setcbreak(self._fd)
        asyncio.get_running_loop().add_reader(self._fd, self._on_readable)
        self._active = True

    def _disable(self) -> None:
        if not self._active:
            return
        with contextlib.suppress(Exception):
            asyncio.get_running_loop().remove_reader(self._fd)
        with contextlib.suppress(termios.error):
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_termios)
        self._active = False

    # Called by render.py's suspend_live_if_active/resume_live_if_active so
    # this listener never competes with a blocking console.input() prompt
    # (an approval gate, or our own inject-a-line sub-prompt below) for the
    # same fd -- both would otherwise try to read the same bytes.
    def pause(self) -> None:
        self._disable()

    def resume(self) -> None:
        if self._fd is not None and not self._active:
            self._enable()

    def _on_readable(self) -> None:
        try:
            data = os.read(self._fd, 8)
        except OSError:
            return
        if not data:
            return
        self._buf.extend(data)
        self._dispatch()

    def _dispatch(self) -> None:
        buf = bytes(self._buf)
        # os.read() is allowed to return more than one byte. A keypress can
        # therefore arrive together with terminal noise or a paste prefix;
        # equality-only matching silently discarded the queue control byte in that case and
        # made the in-task editor appear broken. Consume the control byte
        # wherever it occurs, while keeping the editor single-flight.
        queue_key, _ = queue_key_bytes()
        if queue_key in buf:
            self._buf.clear()
            if self._interject_task is None or self._interject_task.done():
                self._interject_task = asyncio.ensure_future(self._interject())
            return
        if _SHIFT_TAB in buf:
            self._buf.clear()
            self._cycle_mode()
            return
        if buf in _ESCAPE_PREFIXES:
            return  # incomplete escape sequence -- wait for the rest
        self._buf.clear()

    def _cycle_mode(self) -> None:
        self.cli_config.approval_policy = next_mode_in_cycle(self.cli_config.approval_policy)
        label = mode_label_for_policy(self.cli_config.approval_policy)
        # Two signals, not one: a durable scrolling line for anyone reading
        # the transcript back later, AND an immediate update to the
        # persistent status line (set_mode_label) so the switch is visible
        # right now even if it's about to scroll off in a busy stream --
        # see StreamRenderer.__init__'s docstring for why the diagnostic
        # line alone wasn't enough.
        self.renderer.handle_event({
            "event_type": "diagnostics",
            "payload": {
                "content": f"◆ Mode switched to {label} -- takes effect on the next approval decision.",
            },
        })
        if hasattr(self.renderer, "set_mode_label"):
            self.renderer.set_mode_label(label)

    async def _interject(self) -> None:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.patch_stdout import patch_stdout

        from .render import resume_live_if_active, suspend_live_if_active

        suspend_live_if_active(self.renderer)
        try:
            # patch_stdout is prompt_toolkit's own, battle-tested answer to
            # "something else may print while this prompt is being typed":
            # anything the still-running turn writes to stdout/stderr while
            # this Application is active (a tool-call announcement, a tool
            # result) gets inserted cleanly above the input line and the
            # line is redrawn below it, instead of corrupting/interleaving
            # with it. raw=True keeps Rich's own ANSI colour codes intact
            # rather than having patch_stdout escape them as literal text.
            # prompt_async (not the blocking input()/executor-thread
            # approach this replaced) also means the model's response
            # keeps streaming on this same event loop while the human
            # types, without a second thread.
            with patch_stdout(raw=True):
                text = await PromptSession().prompt_async("queue next> ")
        except (EOFError, KeyboardInterrupt):
            text = ""
        finally:
            resume_live_if_active(self.renderer)
        text = text.strip()
        if text:
            item = local_state.enqueue_instruction(self.session_id, text, classification="follow_up")
            self.renderer.handle_event({
                "event_type": "diagnostics",
                "payload": {
                    "content": (
                        f"◆ Queued next instruction {item.id}: {text} "
                        "-- accepted; applied at the next safe round boundary."
                    ),
                },
            })
