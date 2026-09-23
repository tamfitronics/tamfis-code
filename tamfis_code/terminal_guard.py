"""Terminal-mode hygiene for the live composer.

Two things went wrong together in a live report ("Iam still waiting…" then
``^[^[^[^[[O^[[I``): the prompt application that owns the keyboard was gone
while the task kept running, so the tty fell back to cooked mode and started
ECHOING every key -- Esc as ``^[``, focus-in/out as ``^[[I`` / ``^[[O`` --
straight into the scrollback, and Enter went nowhere.

* ``disable_focus_reporting`` turns off xterm focus events (mode 1004). Nothing
  in Tamfis-Code enables it; a previous full-screen program (or a multiplexer)
  can leave it on, and then every window switch injects ``ESC [ I`` / ``ESC [ O``
  into whatever is reading the tty.
* ``TerminalGuard`` remembers the termios state before the prompt takes over and
  can (a) mute echo while no prompt owns the keyboard, so stray input is never
  painted as text, and (b) put the original state back on every exit path.
"""
from __future__ import annotations

import contextlib
import os
import sys
from typing import Any, Optional

FOCUS_REPORTING_OFF = "\x1b[?1004l"
# DEC mouse modes that prompt-toolkit/terminal clients may enable.  A prompt
# can be cancelled while a redraw is in progress, leaving one of these modes
# active in the user's shell; the terminal then sends mouse events to the
# application instead of allowing native drag-selection/copy.
MOUSE_REPORTING_OFF = (
    "\x1b[?1000l\x1b[?1002l\x1b[?1003l"
    "\x1b[?1005l\x1b[?1006l\x1b[?1015l"
)


def disable_focus_reporting(stream: Any = None) -> None:
    """Best-effort: tell the terminal to stop sending focus-in/out sequences."""
    stream = stream if stream is not None else sys.__stdout__
    try:
        if stream is None or not stream.isatty():
            return
        stream.write(FOCUS_REPORTING_OFF)
        stream.flush()
    except Exception:
        pass


def disable_mouse_reporting(stream: Any = None) -> None:
    """Reset every mouse-reporting mode Tamfis-Code may have enabled.

    This is intentionally safe to call at both startup and teardown.  Startup
    repairs a terminal left dirty by a killed/crashed prompt; teardown makes
    normal exits and clarification/approval prompts return native terminal
    selection to the user.
    """
    stream = stream if stream is not None else sys.__stdout__
    try:
        if stream is None or not stream.isatty():
            return
        stream.write(MOUSE_REPORTING_OFF)
        stream.flush()
    except Exception:
        pass


class TerminalGuard:
    """Snapshot/mute/restore for a tty file descriptor (no-op without a tty)."""

    def __init__(self, fd: Optional[int] = None) -> None:
        self._fd = fd
        self._saved: Any = None
        self._muted = False

    def _resolve_fd(self) -> Optional[int]:
        if self._fd is not None:
            return self._fd
        try:
            fd = sys.stdin.fileno()
            return fd if os.isatty(fd) else None
        except Exception:
            return None

    def snapshot(self) -> None:
        fd = self._resolve_fd()
        if fd is None or self._saved is not None:
            return
        try:
            import termios

            self._saved = termios.tcgetattr(fd)
        except Exception:
            self._saved = None

    @property
    def muted(self) -> bool:
        return self._muted

    def mute_echo(self) -> bool:
        """Stop the tty painting typed/injected bytes while nothing reads them.

        ISIG stays on so Ctrl+C still interrupts. Returns True when applied.
        """
        fd = self._resolve_fd()
        if fd is None:
            return False
        try:
            import termios

            attrs = termios.tcgetattr(fd)
            if self._saved is None:
                self._saved = list(attrs)
            lflag = attrs[3]
            lflag &= ~(termios.ECHO | termios.ECHOE | termios.ECHOK | termios.ECHONL | termios.ICANON)
            if hasattr(termios, "ECHOCTL"):
                lflag &= ~termios.ECHOCTL
            attrs[3] = lflag
            termios.tcsetattr(fd, termios.TCSADRAIN, attrs)
            self._muted = True
            return True
        except Exception:
            return False

    def restore(self) -> None:
        # Do this even when no termios snapshot exists: a prompt can enable
        # mouse reporting before snapshot/restore reaches its normal path.
        disable_mouse_reporting()
        fd = self._resolve_fd()
        saved, self._saved = self._saved, None
        was_muted, self._muted = self._muted, False
        if fd is None or saved is None or not was_muted:
            return
        with contextlib.suppress(Exception):
            import termios

            termios.tcsetattr(fd, termios.TCSADRAIN, saved)
