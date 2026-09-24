"""The live "think card": what the model is thinking about right now.

    ╭─ Thinking ─────────────────────────────────╮
    │ Weighing whether resume.py should trust    │
    │ … the checkpoint's step statuses or re-    │
    ╰────────────────────────────────────────────╯

Some provider routes (NVIDIA NIM, DeepSeek-style adapters) stream a real
``reasoning_content`` delta ahead of the answer (see provider_protocols.py's
extraction). Until now that text was only visible with ``--debug``; the
user just saw a spinner. This card surfaces the reasoning in the pinned
composer view while it streams -- the same way the plan panel is pinned --
and the renderer prints a durable ``✻ Thought for 12s`` line once real
answer content starts.

Pure functions of the buffered reasoning text -- no terminal or
prompt_toolkit dependency -- so the layout is unit-testable (mirrors
plan_panel.py).
"""
from __future__ import annotations

from xml.sax.saxutils import escape as _xml_escape

_TITLE = "Thinking"
MIN_CHARS = 24          # below this the card is noise, not signal
MAX_CHARS = 400         # bounded window of the LATEST reasoning, always the tail
MIN_ROWS = 2
MAX_ROWS = 6
MAX_WIDTH = 100         # content-width like the plan panel, never wall-to-wall


def _plain_lines(text: str, *, width: int, rows: int) -> list[str]:
    """The last `rows` wrapped lines of `text` fitted to `width`.

    Hard-wraps on words (long words are cut -- model identifiers and paths
    in reasoning prose routinely exceed the card width), then keeps the
    tail so the card always shows the most recent thought.
    """
    import textwrap

    words = " ".join(str(text or "").split())
    if not words:
        return []
    wrap_width = max(16, width)
    wrapped: list[str] = []
    for word in words.split(" "):
        if not wrapped or len(wrapped[-1]) + 1 + len(word) > wrap_width:
            while len(word) > wrap_width:  # a single token longer than the card
                head, word = word[: wrap_width - 1] + "-", word[wrap_width - 1:]
                wrapped.append(head)
            if word:
                wrapped.append(word)
        else:
            wrapped[-1] = f"{wrapped[-1]} {word}"
    return wrapped[-rows:]


def think_card_html(text: str, *, width: int, active: bool = True) -> list[str]:
    """The card as prompt_toolkit HTML lines, or [] when there is nothing
    worth showing (below MIN_CHARS of accumulated reasoning). Content-width
    like plan_panel_html; `width` is the terminal width."""
    terminal = max(20, int(width or 0))
    inner = max(MIN_CHARS, min(MAX_WIDTH, terminal) - 6)  # borders + padding
    body = _plain_lines(text, width=inner, rows=MAX_ROWS)
    if sum(len(line) for line in body) < MIN_CHARS:
        return []
    title_text = f" {_TITLE} "
    left = (inner + 2 - len(title_text)) // 2
    right = inner + 2 - len(title_text) - left
    top = f"<ansicyan>╭{'─' * left}{_xml_escape(title_text)}{'─' * right}╮</ansicyan>"
    lines = [top]
    for line in body:
        pad = " " * max(0, inner - len(line))
        lines.append(
            f"<ansicyan>│</ansicyan> <ansibrightblack>{_xml_escape(line)}</ansibrightblack>{pad} <ansicyan>│</ansicyan>"
        )
    lines.append(f"<ansicyan>╰{'─' * (inner + 2)}╯</ansicyan>")
    return lines


def thought_summary_line(seconds: float) -> str:
    """The durable one-liner for scrollback once thinking ends."""
    try:
        seconds = max(0.0, float(seconds))
    except (TypeError, ValueError):
        seconds = 0.0
    if seconds >= 60:
        minutes, rest = divmod(int(seconds), 60)
        return f"✻ Thought for {minutes}m {rest:02d}s"
    return f"✻ Thought for {seconds:.0f}s"
