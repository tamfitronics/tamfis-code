"""Tests for the full-screen `tamfis-code resume` picker's pure logic.

resume_picker.py deliberately splits filtering/sorting/focus state
(PickerState) and rendering (render_picker) out from the actual
prompt_toolkit Application (run_resume_picker) so this behavior is
testable without a real terminal or event loop -- the same split
live_input.py uses for its own footer.
"""
import unittest
from datetime import datetime, timedelta, timezone

from tamfis_code.resume_picker import PickerState, relative_time, render_picker
from tamfis_code.workspace import ResumableSessionInfo


def _row(
    session_id: int, *, title: str = "", updated_at: str = "", created_at: str = "",
    archived: bool = False, in_current_workspace: bool = True, status: str = "idle",
    description: str = "",
) -> ResumableSessionInfo:
    return ResumableSessionInfo(
        session_id=session_id,
        workspace_root="/repo",
        title=title or f"Session {session_id}",
        description=description,
        status=status,
        updated_at=updated_at,
        created_at=created_at,
        archived=archived,
        in_current_workspace=in_current_workspace,
    )


class RelativeTimeTests(unittest.TestCase):
    def test_blank_value_is_a_placeholder(self):
        self.assertEqual(relative_time(""), "--")

    def test_unparseable_value_is_a_placeholder(self):
        self.assertEqual(relative_time("not-a-timestamp"), "--")

    def test_minutes_ago(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        value = (now - timedelta(minutes=41)).isoformat()
        self.assertEqual(relative_time(value, now=now), "41m ago")

    def test_hours_ago(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        value = (now - timedelta(hours=11)).isoformat()
        self.assertEqual(relative_time(value, now=now), "11h ago")

    def test_days_ago(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        value = (now - timedelta(days=3)).isoformat()
        self.assertEqual(relative_time(value, now=now), "3d ago")

    def test_naive_timestamps_are_treated_as_utc(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        value = (now - timedelta(minutes=5)).replace(tzinfo=None).isoformat()
        self.assertEqual(relative_time(value, now=now), "5m ago")


class PickerStateFilteringTests(unittest.TestCase):
    def test_defaults_to_current_workspace_and_active_only(self):
        state = PickerState(all_rows=[
            _row(1, in_current_workspace=True, archived=False),
            _row(2, in_current_workspace=False, archived=False),
            _row(3, in_current_workspace=True, archived=True),
        ])
        self.assertEqual([row.session_id for row in state.visible_rows()], [1])

    def test_cwd_filter_all_includes_other_workspaces(self):
        state = PickerState(
            all_rows=[
                _row(1, in_current_workspace=True),
                _row(2, in_current_workspace=False),
            ],
            cwd_filter="all",
        )
        self.assertEqual(
            {row.session_id for row in state.visible_rows()}, {1, 2},
        )

    def test_status_filter_archived_shows_only_archived(self):
        state = PickerState(
            all_rows=[_row(1, archived=False), _row(2, archived=True)],
            status_filter="archived",
        )
        self.assertEqual([row.session_id for row in state.visible_rows()], [2])

    def test_search_query_matches_title_case_insensitively(self):
        state = PickerState(
            all_rows=[_row(1, title="Fix the flaky test"), _row(2, title="Improve docs")],
            query="FLAKY",
        )
        self.assertEqual([row.session_id for row in state.visible_rows()], [1])

    def test_sort_by_updated_is_the_default(self):
        state = PickerState(all_rows=[
            _row(1, updated_at="2026-01-01T00:00:00+00:00", created_at="2026-01-03T00:00:00+00:00"),
            _row(2, updated_at="2026-01-02T00:00:00+00:00", created_at="2026-01-01T00:00:00+00:00"),
        ])
        self.assertEqual([row.session_id for row in state.visible_rows()], [2, 1])

    def test_sort_by_created_uses_created_at_instead(self):
        state = PickerState(
            all_rows=[
                _row(1, updated_at="2026-01-01T00:00:00+00:00", created_at="2026-01-03T00:00:00+00:00"),
                _row(2, updated_at="2026-01-02T00:00:00+00:00", created_at="2026-01-01T00:00:00+00:00"),
            ],
            sort_key="created",
        )
        self.assertEqual([row.session_id for row in state.visible_rows()], [1, 2])


class PickerStateTransitionTests(unittest.TestCase):
    def test_type_char_appends_and_resets_selection(self):
        state = PickerState(all_rows=[_row(1), _row(2)], selected_index=1)
        state.type_char("a")
        self.assertEqual(state.query, "a")
        self.assertEqual(state.selected_index, 0)

    def test_backspace_removes_last_character(self):
        state = PickerState(all_rows=[], query="abc")
        state.backspace()
        self.assertEqual(state.query, "ab")

    def test_move_selection_wraps_around(self):
        state = PickerState(all_rows=[_row(1), _row(2), _row(3)])
        state.move_selection(-1)
        self.assertEqual(state.selected_index, 2)
        state.move_selection(1)
        self.assertEqual(state.selected_index, 0)

    def test_move_selection_is_a_noop_with_no_rows(self):
        state = PickerState(all_rows=[])
        state.move_selection(1)
        self.assertEqual(state.selected_index, 0)

    def test_cycle_focus_visits_list_then_each_filter_group_then_back(self):
        state = PickerState(all_rows=[])
        seen = [state.focus]
        for _ in range(4):
            state.cycle_focus()
            seen.append(state.focus)
        self.assertEqual(seen, ["list", "cwd", "status", "sort", "list"])

    def test_toggle_focused_filter_is_a_noop_when_list_is_focused(self):
        state = PickerState(all_rows=[], focus="list")
        self.assertFalse(state.toggle_focused_filter())
        self.assertEqual(state.cwd_filter, "cwd")

    def test_toggle_focused_filter_flips_cwd(self):
        state = PickerState(all_rows=[], focus="cwd")
        self.assertTrue(state.toggle_focused_filter())
        self.assertEqual(state.cwd_filter, "all")
        state.toggle_focused_filter()
        self.assertEqual(state.cwd_filter, "cwd")

    def test_toggle_focused_filter_flips_status(self):
        state = PickerState(all_rows=[], focus="status")
        state.toggle_focused_filter()
        self.assertEqual(state.status_filter, "archived")

    def test_toggle_focused_filter_flips_sort(self):
        state = PickerState(all_rows=[], focus="sort")
        state.toggle_focused_filter()
        self.assertEqual(state.sort_key, "created")

    def test_toggle_archive_selected_flips_the_row_and_returns_it(self):
        row = _row(1, archived=False)
        state = PickerState(all_rows=[row])
        toggled = state.toggle_archive_selected()
        self.assertIs(toggled, row)
        self.assertTrue(row.archived)
        # Now archived and the default filter is "active" -- it drops out
        # of view and the selection is clamped safely.
        self.assertEqual(state.visible_rows(), [])
        self.assertEqual(state.selected_index, 0)

    def test_toggle_archive_selected_with_no_rows_returns_none(self):
        state = PickerState(all_rows=[])
        self.assertIsNone(state.toggle_archive_selected())

    def test_toggle_expanded_flips_the_flag(self):
        state = PickerState(all_rows=[])
        self.assertFalse(state.expanded)
        state.toggle_expanded()
        self.assertTrue(state.expanded)


class RenderPickerTests(unittest.TestCase):
    def test_shows_search_placeholder_when_query_is_empty(self):
        text = "".join(chunk for _style, chunk in render_picker(PickerState(all_rows=[]), width=80))
        self.assertIn("Type to search", text)

    def test_shows_typed_query_instead_of_the_placeholder(self):
        state = PickerState(all_rows=[], query="auth bug")
        text = "".join(chunk for _style, chunk in render_picker(state, width=80))
        self.assertIn("auth bug", text)
        self.assertNotIn("Type to search", text)

    def test_selected_row_is_marked_with_the_cursor_glyph(self):
        state = PickerState(all_rows=[_row(1, title="Fix the flaky test")])
        text = "".join(chunk for _style, chunk in render_picker(state, width=80))
        self.assertIn("❯", text)
        self.assertIn("Fix the flaky test", text)

    def test_marker_absent_when_focus_is_on_a_filter_group(self):
        state = PickerState(all_rows=[_row(1)], focus="cwd")
        text = "".join(chunk for _style, chunk in render_picker(state, width=80))
        self.assertNotIn("❯", text)

    def test_no_rows_shows_a_clear_empty_state_message(self):
        text = "".join(chunk for _style, chunk in render_picker(PickerState(all_rows=[]), width=80))
        self.assertIn("No sessions match", text)

    def test_footer_shows_position_and_key_hints(self):
        state = PickerState(all_rows=[_row(1), _row(2)])
        text = "".join(chunk for _style, chunk in render_picker(state, width=80))
        self.assertIn("1 / 2", text)
        self.assertIn("enter resume", text)
        self.assertIn("ctrl+a archive", text)
        self.assertIn("esc start new session", text)

    def test_filter_groups_show_the_selected_option_bracketed(self):
        state = PickerState(all_rows=[])
        text = "".join(chunk for _style, chunk in render_picker(state, width=80))
        self.assertIn("[Cwd]", text)
        self.assertIn("[Active]", text)
        self.assertIn("[Updated]", text)

    def test_expanded_shows_the_selected_rows_description(self):
        state = PickerState(
            all_rows=[_row(1, title="Fix bug", description="continue the auth refactor")],
            expanded=True,
        )
        text = "".join(chunk for _style, chunk in render_picker(state, width=80))
        self.assertIn("continue the auth refactor", text)

    def test_not_expanded_by_default_hides_the_description(self):
        state = PickerState(
            all_rows=[_row(1, title="Fix bug", description="continue the auth refactor")],
        )
        text = "".join(chunk for _style, chunk in render_picker(state, width=80))
        self.assertNotIn("continue the auth refactor", text)
