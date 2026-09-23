"""The pinned plan panel: one box, kept above the composer and updated in place.

    ╭────────────── Plan progress ───────────────╮
    │ 1. ✓ List directory contents               │
    │ 2. ◉ Read package.json manifest            │
    │ 3. ○ List directory contents               │
    ╰────────────────────────────────────────────╯

Before this, every step transition printed a FRESH "Plan progress" panel into the
scrollback (interactive mode cannot use Rich's Live -- it fights prompt_toolkit for the
same rows), so a long plan left a wall of near-identical boxes between the tool
records. The interactive composer now draws this panel as part of its own
above-the-input area, which prompt_toolkit redraws in place; one durable snapshot of
the final state is printed when the turn ends.

Pure functions of the plan items -- no terminal or prompt_toolkit dependency -- so the
layout is unit-testable.
"""
from __future__ import annotations

from typing import Any, Sequence
from xml.sax.saxutils import escape as _xml_escape

# (marker, prompt_toolkit colour tag for the marker, tag for the step text)
_MARKERS: dict[str, tuple[str, str, str]] = {
    # Standard ANSI green is low-luminance on many dark terminal palettes;
    # use the bright variant for completed work so the status remains legible.
    "completed": ("✓", "ansibrightgreen", "ansigray"),
    "failed": ("✗", "ansired", "ansired"),
    "in_progress": ("◉", "ansiyellow", "ansiwhite"),
    "awaiting_approval": ("?", "ansimagenta", "ansimagenta"),
    "blocked": ("!", "ansired", "ansired"),
    "cancelled": ("×", "ansigray", "ansigray"),
    "pending": ("○", "ansigray", "ansigray"),
}

_TITLE = "Plan progress"
MIN_ROWS = 4
MAX_ROWS = 12


def plan_progress(items: Sequence[Any]) -> tuple[int, int, int]:
    """Return ``(completed, total, percent)`` from persisted step state.

    Only steps explicitly marked ``completed`` count toward the percentage.
    An active, failed, blocked, or pending step never inflates progress, so
    the number is an honest completion measure rather than an estimate.
    """
    steps = visible_steps(items)
    total = len(steps)
    completed = sum(1 for item in steps if str(item.get("status") or "pending") == "completed")
    percent = round((completed / total) * 100) if total else 0
    return completed, total, percent


def plan_progress_label(items: Sequence[Any]) -> str:
    completed, total, percent = plan_progress(items)
    return f"{percent}% ({completed}/{total} complete)"


def visible_steps(items: Sequence[Any]) -> list[dict[str, Any]]:
    """Plan items that are actual steps (the engine also sends `context` rows)."""
    return [
        item for item in (items or [])
        if isinstance(item, dict) and item.get("status") != "context" and str(item.get("step") or "").strip()
    ]


def rows_that_fit(terminal_rows: int) -> int:
    """How many step rows may be shown: the composer, status, tip and footer need
    roughly a dozen of the terminal's lines, and the panel must never push the
    input off screen."""
    return max(MIN_ROWS, min(MAX_ROWS, terminal_rows - 14))


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: max(1, limit - 1)].rstrip() + "…"


def _window(steps: list[dict[str, Any]], max_rows: int) -> list[tuple[str, Any]]:
    """The rows to draw: ("step", (number, item)) or ("more", "… N done|more").

    A plan longer than `max_rows` scrolls with the work: finished steps fold into one
    "… N done" row and steps not reached yet into "… N more", keeping the current step
    (and one before it) on screen."""
    total = len(steps)
    if total <= max_rows:
        return [("step", (i, item)) for i, item in enumerate(steps, start=1)]
    current = next(
        (i for i, item in enumerate(steps) if item.get("status") == "in_progress"),
        next((i for i, item in enumerate(steps) if item.get("status") in (None, "pending")), total - 1),
    )
    start = max(0, current - 1)
    # A "… N done" row is needed whenever steps precede the window and a "… N more" row
    # whenever steps follow it; each takes one of the `max_rows` lines.
    shown = max_rows - (1 if start > 0 else 0) - 1
    end = start + shown
    if end >= total - 1:
        # The window reaches the tail: no "more" row, so refill from the end.
        end = total
        start = total - (max_rows - 1)  # total > max_rows, so a "done" row is always needed
    rows: list[tuple[str, Any]] = []
    if start > 0:
        rows.append(("more", f"… {start} done"))
    rows.extend(("step", (i + 1, steps[i])) for i in range(start, end))
    if end < total:
        rows.append(("more", f"… {total - end} more"))
    return rows


def plan_panel_html(
    items: Sequence[Any], *, width: int, terminal_rows: int = 30, title: str = _TITLE,
) -> list[str]:
    """The panel as prompt_toolkit HTML lines (no trailing newline), or [] when there
    is no plan. Content-width like the Rich panel it replaces, capped at `width`."""
    steps = visible_steps(items)
    if not steps:
        return []
    rows = _window(steps, rows_that_fit(terminal_rows))
    number_width = len(str(len(steps)))
    prefix_width = number_width + 4  # "N. ✓ "
    cap = max(24, width - 4)  # inside the two border cells and their padding
    entries: list[tuple[str, str, str, str]] = []  # (prefix_plain, marker_tag, text_tag, text)
    for kind, payload in rows:
        if kind == "more":
            entries.append((" " * prefix_width, "ansigray", "ansigray", str(payload)))
            continue
        number, item = payload
        marker, marker_tag, text_tag = _MARKERS.get(str(item.get("status") or "pending"), _MARKERS["pending"])
        entries.append((f"{number:>{number_width}}. ", marker_tag, text_tag, f"{marker}\x00{item.get('step')}"))
    body: list[tuple[str, str]] = []  # (markup, plain) per row
    plain_widths = []
    for prefix, marker_tag, text_tag, text in entries:
        if "\x00" in text:
            marker, step = text.split("\x00", 1)
            step = _clip(step, max(8, cap - len(prefix) - 2))
            markup = (
                f"<ansigray>{_xml_escape(prefix)}</ansigray><{marker_tag}>{marker}</{marker_tag}> "
                f"<{text_tag}>{_xml_escape(step)}</{text_tag}>"
            )
            plain = f"{prefix}{marker} {step}"
        else:
            text = _clip(text, max(8, cap - len(prefix)))
            markup = f"<ansigray>{_xml_escape(prefix)}{_xml_escape(text)}</ansigray>"
            plain = f"{prefix}{text}"
        body.append((markup, plain))
        plain_widths.append(len(plain))
    title_text = f" {title} "
    inner = min(cap, max(max(plain_widths), len(title_text) + 6))
    left = (inner + 2 - len(title_text)) // 2
    right = inner + 2 - len(title_text) - left
    top = f"<ansicyan>╭{'─' * left}{_xml_escape(title_text)}{'─' * right}╮</ansicyan>"
    lines = [top]
    for markup, plain in body:
        pad = " " * max(0, inner - len(plain))
        lines.append(f"<ansicyan>│</ansicyan> {markup}{pad} <ansicyan>│</ansicyan>")
    lines.append(f"<ansicyan>╰{'─' * (inner + 2)}╯</ansicyan>")
    return lines
