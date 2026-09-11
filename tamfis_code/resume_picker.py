"""Full-screen `tamfis-code resume` picker.

Replaces the old numbered console prompt with the same kind of resume
screen Codex/Claude Code offer: type-to-search, a Cwd/All filter, an
Active/Archived status filter, a sort toggle, and a scrollable list of
named sessions with relative timestamps.

Split into pure, directly-unit-testable state and rendering
(PickerState, render_picker) and a thin prompt_toolkit Application that
wires them to the keyboard and the terminal -- the same split
live_input.py uses for its own footer (pure _mode_and_agents_html/
_right_chip helpers vs the actual PromptSession plumbing) -- so the
filtering/sorting/focus logic can be exercised in tests without a real
terminal or event loop.

Deliberately not replicated: a transcript viewer and a "comfortable" row
density toggle. Those need real UI surfaces of their own (rendering a full
conversation, a second row layout) rather than a few lines of glue, and
weren't asked for beyond "make it look like this screen" -- search,
filtering, sorting, archiving, and resuming are the parts that actually
change what `resume` can do.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from .workspace import ResumableSessionInfo

CwdFilter = Literal["cwd", "all"]
StatusFilter = Literal["active", "archived"]
SortKey = Literal["updated", "created"]
Focus = Literal["list", "cwd", "status", "sort"]
Action = Literal["resume", "new", "quit"]

_FOCUS_ORDER: tuple[Focus, ...] = ("list", "cwd", "status", "sort")


def relative_time(value: str, *, now: Optional[datetime] = None) -> str:
    """Compact "41m ago"/"11h ago"/"3d ago" label for an ISO timestamp."""
    if not value:
        return "--"
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError:
        return "--"
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    reference = now or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    seconds = max(0, int((reference - timestamp).total_seconds()))
    if seconds < 60:
        return f"{seconds}s ago"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    if days < 30:
        return f"{days}d ago"
    months = days // 30
    if months < 12:
        return f"{months}mo ago"
    return f"{days // 365}y ago"


@dataclass
class PickerState:
    """All picker state, plus the pure transitions the keyboard bindings
    in run_resume_picker call. Kept independent of prompt_toolkit so the
    filtering/sorting/focus behavior is directly testable."""

    all_rows: list[ResumableSessionInfo]
    query: str = ""
    cwd_filter: CwdFilter = "cwd"
    status_filter: StatusFilter = "active"
    sort_key: SortKey = "updated"
    focus: Focus = "list"
    selected_index: int = 0
    expanded: bool = False

    def visible_rows(self) -> list[ResumableSessionInfo]:
        rows = [
            row for row in self.all_rows
            if (self.cwd_filter == "all" or row.in_current_workspace)
            and (row.archived if self.status_filter == "archived" else not row.archived)
        ]
        query = self.query.strip().lower()
        if query:
            rows = [row for row in rows if query in row.title.lower()]
        sort_field = "created_at" if self.sort_key == "created" else "updated_at"
        rows.sort(key=lambda row: getattr(row, sort_field) or "", reverse=True)
        return rows

    def _clamp(self, count: int) -> None:
        self.selected_index = 0 if count == 0 else max(0, min(self.selected_index, count - 1))

    def selected_row(self) -> Optional[ResumableSessionInfo]:
        rows = self.visible_rows()
        self._clamp(len(rows))
        return rows[self.selected_index] if rows else None

    def type_char(self, char: str) -> None:
        self.query += char
        self.selected_index = 0

    def backspace(self) -> None:
        self.query = self.query[:-1]
        self.selected_index = 0

    def move_selection(self, delta: int) -> None:
        rows = self.visible_rows()
        if rows:
            self.selected_index = (self.selected_index + delta) % len(rows)

    def cycle_focus(self) -> None:
        self.focus = _FOCUS_ORDER[(_FOCUS_ORDER.index(self.focus) + 1) % len(_FOCUS_ORDER)]

    def toggle_focused_filter(self) -> bool:
        """Flip the option of whichever filter/sort group has focus.
        Returns False (no-op) when the list itself, not a filter group, has
        focus -- arrow keys there move the selection instead."""
        if self.focus == "cwd":
            self.cwd_filter = "all" if self.cwd_filter == "cwd" else "cwd"
        elif self.focus == "status":
            self.status_filter = "archived" if self.status_filter == "active" else "active"
        elif self.focus == "sort":
            self.sort_key = "created" if self.sort_key == "updated" else "updated"
        else:
            return False
        self.selected_index = 0
        return True

    def toggle_archive_selected(self) -> Optional[ResumableSessionInfo]:
        """Flip `archived` on the selected row in place and return it so
        the caller can persist the change -- or None if nothing is
        selected."""
        row = self.selected_row()
        if row is None:
            return None
        row.archived = not row.archived
        self._clamp(len(self.visible_rows()))
        return row

    def toggle_expanded(self) -> None:
        self.expanded = not self.expanded


def _render_group(label: str, options: tuple[str, str], selected: str, *, focused: bool) -> list[tuple[str, str]]:
    style = "ansicyan bold" if focused else "ansigray"
    parts = [f"{label}:"]
    for option in options:
        shown = option.capitalize()
        parts.append(f"[{shown}]" if option == selected else shown)
    return [(style, " ".join(parts))]


def render_picker(state: PickerState, *, width: int) -> list[tuple[str, str]]:
    """Render the whole screen as prompt_toolkit-style (style, text)
    fragments. Pure function of `state` and `width` -- no terminal or
    event loop required, so this is exercised directly in tests."""
    width = max(width, 20)
    fragments: list[tuple[str, str]] = []

    search_text = state.query if state.query else "Type to search"
    search_style = "" if state.query else "ansigray italic"
    left = f" {search_text}"

    right_fragments: list[tuple[str, str]] = []
    groups = (
        _render_group("Filter", ("cwd", "all"), state.cwd_filter, focused=state.focus == "cwd"),
        _render_group("Status", ("active", "archived"), state.status_filter, focused=state.focus == "status"),
        _render_group("Sort", ("updated", "created"), state.sort_key, focused=state.focus == "sort"),
    )
    for index, group_fragments in enumerate(groups):
        if index:
            right_fragments.append(("", "    "))
        right_fragments.extend(group_fragments)
    right_len = sum(len(text) for _style, text in right_fragments)
    gap = max(2, width - len(left) - right_len - 1)

    fragments.append((search_style, left))
    fragments.append(("", " " * gap))
    fragments.extend(right_fragments)
    fragments.append(("", " \n\n"))

    rows = state.visible_rows()
    if not rows:
        fragments.append(("ansigray", "  No sessions match the current search/filters.\n"))
    for index, row in enumerate(rows):
        is_selected = index == state.selected_index
        marker = "❯ " if is_selected and state.focus == "list" else "  "
        stamp = row.created_at if state.sort_key == "created" else row.updated_at
        time_field = f"{relative_time(stamp):<10}"
        line_style = "ansicyan bold" if is_selected else ""
        fragments.append((line_style, f"{marker}{time_field}  {row.title}\n"))
        if state.expanded and is_selected and row.description:
            fragments.append(("ansigray", f"          {row.description}\n"))

    fragments.append(("", "\n"))
    position = f"{state.selected_index + 1} / {len(rows)}" if rows else "0 / 0"
    fragments.append(("ansigray", f" {position}\n"))
    fragments.append((
        "ansigray",
        " enter resume   ctrl+a archive/unarchive   esc start new session   ctrl+c cancel\n",
    ))
    fragments.append((
        "ansigray",
        " tab focus filters   ←/→ change option   ctrl+e expand   ↑/↓ browse\n",
    ))
    return fragments


async def run_resume_picker(
    rows: list[ResumableSessionInfo], *, input: Any = None, output: Any = None,
) -> tuple[Action, Optional[int]]:
    """Run the full-screen picker to completion and return the chosen
    action ("resume"/"new"/"quit") plus a session id (only for "resume").
    Awaits until the user resumes a session, asks to start a new one
    (Esc), or cancels (Ctrl+C) -- thin prompt_toolkit glue over PickerState/
    render_picker above, not itself unit tested for the same reason
    interactive.py's own REPL loop isn't: the logic worth testing already
    lives in the pure functions it calls.

    Must be awaited, never driven via the synchronous `Application.run()`:
    the caller (cli.py's `resume` command) is itself a coroutine already
    running inside `asyncio.run()` (see async_command/_run_async), and
    `Application.run()` starts its own `asyncio.run()` internally -- which
    raises "asyncio.run() cannot be called from a running event loop" the
    moment it's invoked from inside one, confirmed live the first time
    `tamfis-code resume` was actually run after this was added.

    `input`/`output` are forwarded to prompt_toolkit's Application
    unchanged (its own testing hooks -- e.g. create_pipe_input()/
    DummyOutput()); left None for a real terminal.
    """
    from prompt_toolkit import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.containers import Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.styles import Style

    from . import state as local_state

    state = PickerState(all_rows=rows)
    result: dict[str, Any] = {"action": "quit", "session_id": None}

    def get_text() -> list[tuple[str, str]]:
        size = app.output.get_size()
        return render_picker(state, width=size.columns)

    control = FormattedTextControl(get_text, focusable=True, show_cursor=False)
    window = Window(content=control, always_hide_cursor=True, wrap_lines=False)
    layout = Layout(window)
    bindings = KeyBindings()

    @bindings.add("c-c")
    def _cancel(event: Any) -> None:
        result["action"] = "quit"
        event.app.exit()

    @bindings.add("escape")
    def _start_new(event: Any) -> None:
        result["action"] = "new"
        event.app.exit()

    @bindings.add("enter")
    def _resume(event: Any) -> None:
        row = state.selected_row()
        if row is not None:
            result["action"] = "resume"
            result["session_id"] = row.session_id
        else:
            result["action"] = "new"
        event.app.exit()

    @bindings.add("tab")
    def _cycle_focus(event: Any) -> None:
        state.cycle_focus()
        event.app.invalidate()

    @bindings.add("up")
    def _up(event: Any) -> None:
        if not state.toggle_focused_filter():
            state.move_selection(-1)
        event.app.invalidate()

    @bindings.add("down")
    def _down(event: Any) -> None:
        if not state.toggle_focused_filter():
            state.move_selection(1)
        event.app.invalidate()

    @bindings.add("left")
    def _left(event: Any) -> None:
        state.toggle_focused_filter()
        event.app.invalidate()

    @bindings.add("right")
    def _right(event: Any) -> None:
        state.toggle_focused_filter()
        event.app.invalidate()

    @bindings.add("c-a")
    def _archive(event: Any) -> None:
        row = state.toggle_archive_selected()
        if row is not None:
            local_state.set_session_archived(row.session_id, row.archived)
        event.app.invalidate()

    @bindings.add("c-e")
    def _expand(event: Any) -> None:
        state.toggle_expanded()
        event.app.invalidate()

    @bindings.add("backspace")
    @bindings.add("c-h")
    def _backspace(event: Any) -> None:
        if state.focus == "list":
            state.backspace()
        event.app.invalidate()

    @bindings.add(Keys.Any)
    def _type(event: Any) -> None:
        if state.focus == "list" and len(event.data) == 1 and event.data.isprintable():
            state.type_char(event.data)
        event.app.invalidate()

    style = Style.from_dict({"bottom-toolbar": "noreverse"})
    app: Application = Application(
        layout=layout, key_bindings=bindings, full_screen=True,
        mouse_support=False, style=style, input=input, output=output,
    )
    await app.run_async()
    return result["action"], result["session_id"]
