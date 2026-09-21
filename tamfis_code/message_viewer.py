"""Show more / show less for a long message: a viewer above the composer, not scrollback.

A long reply is shown collapsed ("N more chars -- press Ctrl+E to show full message").
Ctrl+E used to PRINT the full text into the scrollback, which cannot be taken back: the
"show less" it announced was just a dead line, and there was no key that did it.

Now Ctrl+E opens the full message in a viewer drawn in the composer's own area (the same
place the pinned plan lives), and Ctrl+E -- or Esc -- closes it again, leaving the
scrollback exactly as it was:

    ── Assistant · full message ── lines 1-24 of 87 ──────────────
    <the message, scrollable>
      ↑/↓ scroll · PgUp/PgDn page · ←/→ other messages · Ctrl+E or Esc to show less

Process-wide state (`VIEWER`), like COLLAPSED_MESSAGES: the mid-task composer
(live_input.py) and the idle prompt (interactive.py) are different prompt_toolkit
sessions, and the collapsed message and its hint outlive the turn that printed them.
"""
from __future__ import annotations

import shutil
from io import StringIO
from typing import Any, Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

MIN_ROWS = 6
MAX_ROWS = 40
_RESET = "\x1b[0m"
_CYAN = "\x1b[36m"
_DIM = "\x1b[90m"


def viewer_rows(terminal_rows: int) -> int:
    """Body rows for the viewer: the composer (status, tip, rules, input, footer) needs
    about a dozen lines of the terminal, and the viewer must never push the input off."""
    return max(MIN_ROWS, min(MAX_ROWS, terminal_rows - 12))


def _render_lines(kind: str, content: str, width: int) -> list[str]:
    """The message rendered to ANSI-styled lines wrapped at `width`."""
    buffer = StringIO()
    console = Console(
        file=buffer, force_terminal=True, color_system="standard", width=max(20, width),
        legacy_windows=False, highlight=False,
    )
    console.print(Markdown(content) if kind == "assistant" else Text(content))
    return buffer.getvalue().rstrip("\n").split("\n")


class MessageViewer:
    def __init__(self) -> None:
        self.is_open = False
        self._index = 0
        self._offset = 0
        self._entries: list[tuple[str, str]] = []
        self._key = "Ctrl+E"
        self._cache_key: Optional[tuple] = None
        self._cache_lines: list[str] = []
        self._last_body_rows = MIN_ROWS

    # -- state -------------------------------------------------------------------
    def toggle(self, store: Any = None, *, key: str = "Ctrl+E") -> bool:
        """Open the newest collapsed message (Ctrl+E) or tool output (Ctrl+O), or close the viewer if
        it is open. False (and nothing changes) when there is nothing to show."""
        if self.is_open:
            self.close()
            return True
        if store is None:
            from .render import COLLAPSED_MESSAGES as store
        entries = store.entries()
        if not entries:
            return False
        self._key = key
        self._entries = entries
        self._index = len(entries) - 1
        self._offset = 0
        self.is_open = True
        return True

    def close(self) -> None:
        self.is_open = False
        self._entries = []
        self._cache_key = None
        self._cache_lines = []

    def other(self, delta: int) -> None:
        """Move to an older (-1) or newer (+1) collapsed message."""
        if not self.is_open:
            return
        target = max(0, min(len(self._entries) - 1, self._index + delta))
        if target != self._index:
            self._index = target
            self._offset = 0

    def scroll(self, delta: int) -> None:
        self._offset = max(0, min(self._max_offset(), self._offset + delta))

    def page(self, direction: int) -> None:
        self.scroll(direction * max(1, self._last_body_rows - 1))

    def home(self) -> None:
        self._offset = 0

    def end(self) -> None:
        self._offset = self._max_offset()

    def _max_offset(self) -> int:
        return max(0, len(self._cache_lines) - self._last_body_rows)

    # -- drawing -------------------------------------------------------------------
    def panel_lines(self, columns: int, terminal_rows: int) -> list[str]:
        """The viewer as ANSI lines (no trailing newline); [] when closed."""
        if not self.is_open or not self._entries:
            return []
        kind, content = self._entries[self._index]
        width = max(20, min(columns - 2, 120))
        key = (kind, hash(content), width)
        if key != self._cache_key:
            self._cache_key = key
            self._cache_lines = _render_lines(kind, content, width)
        rows = viewer_rows(terminal_rows)
        self._last_body_rows = rows
        self._offset = max(0, min(self._offset, self._max_offset()))
        total = len(self._cache_lines)
        body = self._cache_lines[self._offset:self._offset + rows]
        first = self._offset + 1
        last = self._offset + len(body)
        label = {"assistant": "Assistant", "tool": "Tool output"}.get(kind, "You")
        noun = "output" if kind == "tool" else "message"
        parts = [f"{label} · full {'transcript' if kind == 'tool' else 'message'}", f"lines {first}-{last} of {total}"]
        if len(self._entries) > 1:
            parts.append(f"{noun} {self._index + 1} of {len(self._entries)}")
        head_text = "── " + " ── ".join(parts) + " "
        head = f"{_CYAN}{head_text}{'─' * max(0, columns - len(head_text))}{_RESET}"
        hints = ["↑/↓ scroll"]
        if total > rows:
            hints.append("PgUp/PgDn page")
        if len(self._entries) > 1:
            hints.append(f"←/→ other {noun}s")
        hints.append(f"{self._key} or Esc to show less")
        foot = f"{_DIM}  {' · '.join(hints)}{_RESET}"
        return ["", head, *[f"{line}{_RESET}" for line in body], foot, ""]


VIEWER = MessageViewer()


def panel_ansi(columns: Optional[int] = None, terminal_rows: Optional[int] = None) -> str:
    """The open viewer as one ANSI string ending in a newline, or "" when closed."""
    if not VIEWER.is_open:
        return ""
    size = shutil.get_terminal_size(fallback=(80, 24))
    lines = VIEWER.panel_lines(columns or size.columns, terminal_rows or size.lines)
    return ("\n".join(lines) + "\n") if lines else ""


def install_bindings(bindings: Any) -> None:
    """Scroll/close keys, active ONLY while the viewer is open (so ordinary editing is
    untouched otherwise). Call after every other binding is registered: for the same
    key prompt_toolkit uses the LAST matching binding, and Esc/Up/Down here must beat the
    composer's own (cancel-turn / queued-instruction recall / history)."""
    from prompt_toolkit.filters import Condition

    is_open = Condition(lambda: VIEWER.is_open)

    def bind(keys: tuple, action: Any, *, eager: bool = False) -> None:
        @bindings.add(*keys, filter=is_open, eager=eager)
        def _handler(event) -> None:
            action()
            event.app.invalidate()

    bind(("up",), lambda: VIEWER.scroll(-1))
    bind(("down",), lambda: VIEWER.scroll(1))
    bind(("pageup",), lambda: VIEWER.page(-1))
    bind(("pagedown",), lambda: VIEWER.page(1))
    bind(("home",), VIEWER.home)
    bind(("end",), VIEWER.end)
    bind(("left",), lambda: VIEWER.other(-1))
    bind(("right",), lambda: VIEWER.other(1))
    # Esc closes the viewer instead of cancelling the running turn.
    bind(("escape",), VIEWER.close, eager=True)
