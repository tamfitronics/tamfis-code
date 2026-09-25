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

TITLE = "Thinking"
MIN_CHARS = 24          # below this the card is noise, not signal
MAX_CHARS = 400         # bounded window of the LATEST reasoning, always the tail
MIN_ROWS = 2
MAX_ROWS = 8            # was 6: six rows forced constant scrolling of the tail,
                        # which read as a flickering, hard-to-follow strip
DEFAULT_MAX_WIDTH = 100  # content-width like the plan panel, never wall-to-wall
# Body text style. `ansibrightblack` (colour 8, dark grey) was nearly
# invisible on the dark terminals this card actually runs on -- users read
# it as "too dark to read". `ansigray` (colour 7, light grey) keeps the
# text visually secondary to the answer while staying comfortably legible,
# and matches the composer's existing convention (live_input.py renders the
# activity line in <ansigray> too). Configurable via think_card_style.
DEFAULT_BODY_STYLE = "ansigray"
# Kept as an alias because tests and config.py's loader referenced the old
# module-level constant name.
BODY_STYLE = DEFAULT_BODY_STYLE


def _safe_style(style: str | None) -> str:
    """Validate a configured prompt_toolkit style tag before interpolating it.

    The value reaches the composer as `<{style}>...</{style}>` markup; a
    malicious/typo'd config value containing markup characters (e.g.
    `<script>` or a stray `<`) must fall back to the default rather than
    produce broken prompt_toolkit HTML.
    """
    candidate = str(style or "").strip()
    if candidate and all(ch.isalnum() or ch == "_" for ch in candidate):
        return candidate
    return DEFAULT_BODY_STYLE


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


def think_card_html(
    text: str, *, width: int, active: bool = True,
    style: str | None = None, max_width: int | None = None,
) -> list[str]:
    """The card as prompt_toolkit HTML lines, or [] when there is nothing
    worth showing (below MIN_CHARS of accumulated reasoning). Content-width
    like plan_panel_html; `width` is the terminal width.

    `style` overrides the body text colour (a validated prompt_toolkit ANSI
    tag -- see _safe_style); `max_width` raises the card's widest content
    width, so a user on a very wide terminal can let the card use more of
    the screen instead of wrapping at the 100-column default.
    """
    terminal = max(20, int(width or 0))
    cap = DEFAULT_MAX_WIDTH
    try:
        configured = int(max_width) if max_width is not None else DEFAULT_MAX_WIDTH
    except (TypeError, ValueError):
        configured = DEFAULT_MAX_WIDTH
    if configured >= 20:
        cap = configured
    inner = max(MIN_CHARS, min(cap, terminal) - 6)  # borders + padding
    body = _plain_lines(text, width=inner, rows=MAX_ROWS)
    if sum(len(line) for line in body) < MIN_CHARS:
        return []
    body_style = _safe_style(style)
    title_text = f" {TITLE} "
    left = (inner + 2 - len(title_text)) // 2
    right = inner + 2 - len(title_text) - left
    top = f"<ansicyan>╭{'─' * left}{_xml_escape(title_text)}{'─' * right}╮</ansicyan>"
    lines = [top]
    for line in body:
        pad = " " * max(0, inner - len(line))
        lines.append(
            f"<ansicyan>│</ansicyan> <{body_style}>{_xml_escape(line)}</{body_style}>{pad} <ansicyan>│</ansicyan>"
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
