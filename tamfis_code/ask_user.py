"""Structured questions for the human at the terminal -- "probing".

The agent can stop and ask the person a real decision, with concrete options,
instead of guessing or stalling: which of two conventions to follow, whether to
take a hard-to-reverse or outward-facing action (restart a service that drops
connections, delete data, publish), what scope they meant. The person answers
with arrow keys (or a number) and the agent continues with the answer:

    ● Ask(3 questions)
      ⎿  User answered the agent's questions:
         · Restart Caddy now? → Restart now (Recommended)
         · What happens to the 4.6 GB of site files? → Keep offline for now

Up to MAX_QUESTIONS per call, each with 2+ options that carry a one-line
description of what choosing them does, an optional multi-select, the
recommended option FIRST and marked "(Recommended)", and an "Other" free-text
choice that is always available.

The selection logic (`SelectorState`) and the wording (normalize / format) are
pure so they are unit-testable; only `ask_questions` touches the terminal.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from prompt_toolkit.keys import Keys
from prompt_toolkit.mouse_events import MouseEventType

from .terminal_guard import disable_mouse_reporting

MAX_QUESTIONS = 4
MAX_OPTIONS = 6
_MAX_LABEL_CHARS = 60
_MAX_TEXT_CHARS = 400
_RECOMMENDED_RE = re.compile(r"\s*\(\s*recommended\s*\)\s*$", re.IGNORECASE)
OTHER_LABEL = "Other…"


@dataclass
class Option:
    label: str
    description: str = ""

    @property
    def recommended(self) -> bool:
        return bool(_RECOMMENDED_RE.search(self.label))


@dataclass
class Question:
    question: str
    header: str = ""
    options: list[Option] = field(default_factory=list)
    multi_select: bool = False


@dataclass
class Answer:
    """One question's outcome. `values` holds every chosen label (one for a
    single-select); `skipped` means the person dismissed the question."""

    values: list[str] = field(default_factory=list)
    other: bool = False
    skipped: bool = False

    def text(self) -> str:
        if self.skipped or not self.values:
            return "(no answer given)"
        return ", ".join(self.values)


def _clip(text: Any, limit: int) -> str:
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def _coerce_option(raw: Any) -> Optional[Option]:
    if isinstance(raw, Option):
        return raw
    if isinstance(raw, str):
        label = _clip(raw, _MAX_LABEL_CHARS)
        return Option(label) if label else None
    if isinstance(raw, dict):
        label = _clip(raw.get("label") or raw.get("name") or raw.get("value") or "", _MAX_LABEL_CHARS)
        if not label:
            return None
        return Option(label, _clip(raw.get("description") or raw.get("detail") or "", _MAX_TEXT_CHARS))
    return None


def _option_items(raw: Any) -> list[Any]:
    """Normalize provider/MCP argument variants without splitting strings.

    Some tool-call adapters preserve an array argument as the JSON text
    ``["yes", "no"]``. Iterating that text directly creates one menu row per
    character, which is especially confusing in the approval gate. A plain
    string remains one option; only a valid JSON array is expanded.
    """
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                decoded = json.loads(text)
            except (TypeError, ValueError, json.JSONDecodeError):
                decoded = None
            if isinstance(decoded, list):
                return decoded
        return [raw]
    if isinstance(raw, (list, tuple)):
        return list(raw)
    return []


def normalize_questions(
    questions: Any = None, question: Any = None, options: Any = None,
) -> list[Question]:
    """The tool's arguments as validated Question objects.

    Accepts the structured form (`questions=[{question, header, options, multiSelect}]`)
    and the original single form (`question` + optional string `options`). Raises
    ValueError with a message the model can act on when nothing usable was asked.
    """
    raw_list: list[Any]
    if isinstance(questions, list) and questions:
        raw_list = questions
    elif question:
        raw_list = [{"question": question, "options": options or []}]
    else:
        raise ValueError(
            "ask_user_question needs `questions` (a list of {question, options}) or a "
            "single `question`."
        )
    if len(raw_list) > MAX_QUESTIONS:
        raise ValueError(f"ask at most {MAX_QUESTIONS} questions per call (got {len(raw_list)}); batch the most important ones.")
    result: list[Question] = []
    for item in raw_list:
        if isinstance(item, str):
            item = {"question": item}
        if not isinstance(item, dict):
            continue
        text = _clip(item.get("question") or item.get("text") or "", _MAX_TEXT_CHARS)
        if not text:
            raise ValueError("every question needs non-empty `question` text.")
        opts: list[Option] = []
        seen: set[str] = set()
        for raw in _option_items(item.get("options") or item.get("choices") or []):
            option = _coerce_option(raw)
            if option is None or option.label.casefold() in seen:
                continue
            seen.add(option.label.casefold())
            opts.append(option)
        # Recommended first: the person's eye and Enter key land on the model's pick.
        opts.sort(key=lambda o: 0 if o.recommended else 1)
        result.append(Question(
            question=text,
            header=_clip(item.get("header") or "", 24),
            options=opts[:MAX_OPTIONS],
            multi_select=bool(item.get("multiSelect") or item.get("multi_select")),
        ))
    if not result:
        raise ValueError("no valid question was provided.")
    return result


ANSWER_HEADING = "User answered the agent's questions:"


def answers_to_text(questions: Sequence[Question], answers: Sequence[Answer]) -> str:
    """The tool result: the heading, then one "· question → answer" line each.
    The same text is what the model reads AND what the terminal shows under the
    `● Ask(...)` header, so the person sees exactly what the agent was told."""
    lines = [ANSWER_HEADING]
    for question, answer in zip(questions, answers):
        lines.append(f"· {question.question} → {answer.text()}")
    return "\n".join(lines)


def unavailable_message() -> str:
    return (
        "ask_user_question is unavailable in this session (no attached interactive terminal). "
        "Do not guess on a decision that is destructive, hard to reverse, or outward-facing "
        "(deleting data, restarting or taking a service offline, pushing or publishing): take "
        "the reversible/safest option, or stop and report exactly which decision you need. "
        "For anything routine, proceed on your best judgement and state what you assumed."
    )


# ---------------------------------------------------------------------------
# Selection logic (pure)
# ---------------------------------------------------------------------------


class SelectorState:
    """Cursor / selection state for one question. Rows are the options followed
    by an always-present "Other…" row."""

    def __init__(self, question: Question, *, position: int = 1, total: int = 1) -> None:
        self.question = question
        self.position = position
        self.total = total
        self.cursor = 0
        self.selected: set[int] = set()

    @property
    def row_count(self) -> int:
        return len(self.question.options) + 1

    @property
    def other_index(self) -> int:
        return len(self.question.options)

    def move(self, delta: int) -> None:
        self.cursor = (self.cursor + delta) % self.row_count

    def jump(self, number: int) -> bool:
        """Jump to row `number` (1-based). True when it exists."""
        if 1 <= number <= self.row_count:
            self.cursor = number - 1
            return True
        return False

    def toggle(self) -> None:
        if not self.question.multi_select or self.cursor == self.other_index:
            return
        self.selected.symmetric_difference_update({self.cursor})

    def confirm(self) -> tuple[str, list[str]]:
        """("other", []) when the person wants to type their own answer;
        ("done", labels) otherwise."""
        options = self.question.options
        if self.question.multi_select:
            chosen = sorted(self.selected)
            if self.cursor == self.other_index:
                return "other", [options[i].label for i in chosen]
            if not chosen:
                chosen = [self.cursor]
            return "done", [options[i].label for i in chosen]
        if self.cursor == self.other_index:
            return "other", []
        return "done", [options[self.cursor].label]

    # -- rendering -----------------------------------------------------
    def render(self) -> list[tuple[str, str]]:
        """prompt_toolkit formatted-text fragments for the current state."""
        question = self.question
        out: list[tuple[str, str]] = []
        badge = f"{question.header}  " if question.header else ""
        counter = f" ({self.position}/{self.total})" if self.total > 1 else ""
        out.append(("bold ansicyan", f"\n {badge}{question.question}{counter}\n"))
        rows = [*(o for o in question.options), Option(OTHER_LABEL, "Type your own answer")]
        for index, option in enumerate(rows):
            here = index == self.cursor
            marker = "❯" if here else " "
            if question.multi_select and index != self.other_index:
                box = "[x]" if index in self.selected else "[ ]"
                label = f"{box} {option.label}"
            else:
                label = option.label
            style = "bold ansicyan" if here else ""
            out.append((style, f"  {marker} {index + 1}. {label}\n"))
            if option.description:
                out.append(("ansigray", f"       {option.description}\n"))
        hint = "↑/↓ move · Enter select · Esc skip"
        if question.multi_select:
            hint = "↑/↓ move · Space toggle · Enter confirm · Esc skip"
        out.append(("ansigray", f"\n  {hint}\n"))
        return out


# ---------------------------------------------------------------------------
# Terminal I/O
# ---------------------------------------------------------------------------


async def _run_selector(state: SelectorState, *, input: Any = None, output: Any = None) -> tuple[str, list[str]]:
    """Run one inline selector. Returns ("done", labels) / ("other", labels) /
    ("skip", [])."""
    from prompt_toolkit.application import Application
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import FormattedTextControl, Layout, Window

    bindings = KeyBindings()

    @bindings.add("up")
    @bindings.add("k")
    def _up(event) -> None:
        state.move(-1)

    @bindings.add("down")
    @bindings.add("j")
    @bindings.add("tab")
    def _down(event) -> None:
        state.move(1)

    @bindings.add(" ")
    def _space(event) -> None:
        state.toggle()

    for digit in "123456789":
        @bindings.add(digit)
        def _digit(event, _digit=digit) -> None:
            if state.jump(int(_digit)) and not state.question.multi_select:
                event.app.exit(result=state.confirm())

    @bindings.add("enter")
    def _enter(event) -> None:
        event.app.exit(result=state.confirm())

    @bindings.add("escape", eager=True)
    @bindings.add("c-c")
    def _skip(event) -> None:
        event.app.exit(result=("skip", []))

    @bindings.add(Keys.Vt100MouseEvent)
    def _mouse(event) -> None:
        # prompt_toolkit reports coordinates relative to the selector window.
        # The first rendered row starts after the question/header lines; use
        # the nearest option row rather than requiring arrow-key navigation.
        # Some prompt_toolkit/input backends deliver a normal KeyPressEvent
        # to this handler when a VT100 mouse sequence is incomplete or the
        # terminal switches modes during redraw. Treat that as a no-op; an
        # approval/question screen must never crash the event loop merely
        # because keyboard and mouse input crossed at the same tick.
        mouse = getattr(event, "mouse_event", None)
        if mouse is None:
            return
        if mouse.event_type != MouseEventType.MOUSE_UP:
            return
        clicked_y = int(mouse.position.y)
        row = None
        render_y = 2  # leading blank line + question line
        options = [*state.question.options, Option(OTHER_LABEL, "Type your own answer")]
        for index, option in enumerate(options):
            if clicked_y == render_y:
                row = index
                break
            render_y += 1
            if option.description:
                render_y += 1
        if row is not None:
            state.cursor = row
            if not state.question.multi_select:
                event.app.exit(result=state.confirm())
            else:
                state.toggle()
                event.app.invalidate()

    control = FormattedTextControl(lambda: FormattedText(state.render()), focusable=True, show_cursor=False)
    application = Application(
        layout=Layout(Window(control, wrap_lines=True, dont_extend_height=True)),
        key_bindings=bindings, full_screen=False, mouse_support=True,
        input=input, output=output,
    )
    try:
        result = await application.run_async()
    finally:
        # Approval/clarification uses mouse capture intentionally, but an
        # interrupted Application must not leave the parent terminal unable
        # to drag-select and copy text.
        disable_mouse_reporting()
    return result if result else ("skip", [])


async def _prompt_free_text(prompt: str, *, input: Any = None, output: Any = None) -> str:
    from prompt_toolkit import PromptSession

    session = PromptSession(input=input, output=output)
    try:
        return (await session.prompt_async(prompt)).strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def _has_terminal() -> bool:
    import sys

    try:
        return bool(sys.stdin.isatty() and sys.stdout.isatty())
    except Exception:  # pragma: no cover - exotic stdin replacements
        return False


def _console_answer(console: Any, question: Question, position: int, total: int) -> Answer:
    """The plain numbered prompt: for a session with no real terminal (piped
    stdin, CI, some SSH wrappers) or when the arrow-key selector cannot start."""
    from rich.panel import Panel

    counter = f" ({position}/{total})" if total > 1 else ""
    console.print(Panel(question.question, title=(question.header or "Question from the agent") + counter,
                        border_style="cyan", expand=False))
    for index, option in enumerate(question.options, start=1):
        console.print(f"  {index}. {option.label}" + (f" -- {option.description}" if option.description else ""))
    if not question.options:
        raw = str(console.input("Your answer: ")).strip()
        return Answer([raw]) if raw else Answer(skipped=True)
    hint = "numbers separated by commas, or free text" if question.multi_select else "a number above, or free text"
    raw = str(console.input(f"Your answer ({hint}): ")).strip()
    if not raw:
        return Answer(skipped=True)
    parts = [part.strip() for part in raw.split(",")] if question.multi_select else [raw]
    if all(part.isdigit() and 1 <= int(part) <= len(question.options) for part in parts):
        picked = []
        for part in parts:
            label = question.options[int(part) - 1].label
            if label not in picked:
                picked.append(label)
        return Answer(picked)
    return Answer([raw], other=True)  # free text is always accepted


async def ask_questions(
    console: Any,
    questions: Sequence[Question],
    *,
    input: Any = None,
    output: Any = None,
    selector: Callable[..., Any] = None,
) -> list[Answer]:
    """Ask each question in turn and return the answers (same order).

    On a real terminal each choice question is an arrow-key selector; with no
    terminal (or if the selector cannot start) it is the numbered console prompt.
    """
    use_terminal_ui = selector is not None or input is not None or _has_terminal()
    run = selector or _run_selector
    answers: list[Answer] = []
    for position, question in enumerate(questions, start=1):
        total = len(questions)
        if not use_terminal_ui:
            answers.append(_console_answer(console, question, position, total))
            continue
        if not question.options:
            # A free-text question: no choices to select from.
            text = await _prompt_free_text(f"\n {question.question}\n  Your answer: ", input=input, output=output)
            answers.append(Answer([text]) if text else Answer(skipped=True))
            continue
        state = SelectorState(question, position=position, total=total)
        try:
            kind, labels = await run(state, input=input, output=output)
        except Exception:
            answers.append(_console_answer(console, question, position, total))
            continue
        if kind == "skip":
            answers.append(Answer(skipped=True))
        elif kind == "other":
            text = await _prompt_free_text("  Your answer: ", input=input, output=output)
            values = [*labels, text] if text else labels
            answers.append(Answer(values, other=bool(text)) if values else Answer(skipped=True))
        else:
            answers.append(Answer(labels))
    return answers
