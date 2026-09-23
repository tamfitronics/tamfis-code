"""The plan is pinned above the composer and updated in place, not reprinted per step.

Owner request 2026-09-20: interactive tamfis-code printed a fresh "Plan progress" panel
into the scrollback after every step (a wall of near-identical boxes between the tool
records); it must be ONE panel, pinned, updated in situ, with one final snapshot.
"""
import os
import re
import unittest
from io import StringIO
from unittest.mock import patch

from rich.console import Console

from tamfis_code.live_input import LiveInputListener
from tamfis_code.plan_panel import MAX_ROWS, plan_panel_html, plan_progress, plan_progress_label, rows_that_fit, visible_steps
from tamfis_code.render import StreamRenderer

from test_live_input import _config, _console


def _plain(line: str) -> str:
    return re.sub(r"<[^>]+>", "", line).replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")


def _steps(*pairs):
    return [{"step": text, "status": status} for text, status in pairs]


FOUR = _steps(
    ("List directory contents", "completed"),
    ("Read package.json manifest", "in_progress"),
    ("List directory contents", "pending"),
    ("Read setup.cfg and setup.py manifests", "pending"),
)


class PanelLayoutTests(unittest.TestCase):
    def test_progress_is_based_only_on_explicitly_completed_steps(self):
        completed, total, percent = plan_progress(_steps(
            ("done", "completed"),
            ("active", "in_progress"),
            ("blocked", "blocked"),
            ("waiting", "pending"),
        ))
        self.assertEqual((completed, total, percent), (1, 4, 25))
        self.assertEqual(plan_progress_label(_steps(("done", "completed"))), "100% (1/1 complete)")

    def test_the_panel_matches_the_requested_layout(self):
        lines = [_plain(line) for line in plan_panel_html(FOUR, width=100)]
        self.assertEqual(lines, [
            "╭────────────── Plan progress ───────────────╮",
            "│ 1. ✓ List directory contents               │",
            "│ 2. ◉ Read package.json manifest            │",
            "│ 3. ○ List directory contents               │",
            "│ 4. ○ Read setup.cfg and setup.py manifests │",
            "╰────────────────────────────────────────────╯",
        ])

    def test_every_line_is_the_same_width(self):
        widths = {len(_plain(line)) for line in plan_panel_html(FOUR, width=100)}
        self.assertEqual(len(widths), 1, widths)

    def test_no_plan_no_panel(self):
        self.assertEqual(plan_panel_html([], width=80), [])
        self.assertEqual(plan_panel_html(_steps(("ctx", "context")), width=80), [])

    def test_a_completed_plan_shows_every_step_ticked(self):
        done = [dict(item, status="completed") for item in FOUR]
        body = [_plain(line) for line in plan_panel_html(done, width=100)][1:-1]
        self.assertTrue(all("✓" in row for row in body), body)

    def test_failed_steps_are_marked(self):
        rows = [_plain(line) for line in plan_panel_html(_steps(("Run tests", "failed")), width=80)]
        self.assertIn("✗", rows[1])

    def test_step_text_is_escaped_for_prompt_toolkit_markup(self):
        html = "".join(plan_panel_html(_steps(("Fix <b>& more", "pending")), width=80))
        self.assertIn("Fix &lt;b&gt;&amp; more", html)
        self.assertNotIn("<b>", html)

    def test_a_narrow_terminal_clips_long_steps_and_never_overflows(self):
        long = _steps(("Refactor the entire authentication and session handling subsystem " * 3, "in_progress"))
        for line in plan_panel_html(long, width=40):
            self.assertLessEqual(len(_plain(line)), 40, _plain(line))

    def test_a_long_plan_scrolls_with_the_work_and_fits_the_terminal(self):
        for current in (0, 1, 9, 20, 28, 29):
            plan = _steps(*[
                (f"step {i + 1}", "completed" if i < current else "in_progress" if i == current else "pending")
                for i in range(30)
            ])
            lines = [_plain(line) for line in plan_panel_html(plan, width=80, terminal_rows=30)]
            body = lines[1:-1]
            self.assertLessEqual(len(body), rows_that_fit(30))
            self.assertTrue(any(f"step {current + 1}" in row for row in body), (current, body))
            self.assertTrue(any("◉" in row for row in body))

    def test_the_panel_never_takes_more_than_the_cap(self):
        self.assertLessEqual(rows_that_fit(200), MAX_ROWS)
        self.assertGreaterEqual(rows_that_fit(5), 4)

    def test_context_rows_are_not_steps(self):
        self.assertEqual(len(visible_steps(_steps(("a", "context"), ("b", "pending")))), 1)


class PinnedRendererTests(unittest.TestCase):
    def _renderer(self, *, pinned: bool):
        stream = StringIO()
        renderer = StreamRenderer(Console(file=stream, no_color=True, width=100))
        renderer._is_tty = pinned
        renderer.live_input_listener = object() if pinned else None
        return renderer, stream

    def _plan_events(self, renderer):
        renderer.handle_event({"event_type": "plan_created", "payload": {
            "stage": "plan", "title": "Execution plan", "items": FOUR,
            "assumptions": ["a"], "risks": ["r"],
        }})
        for done in (2, 3, 4):
            items = [dict(item, status="completed" if i < done else "pending") for i, item in enumerate(FOUR)]
            renderer.handle_event({"event_type": "plan_step_progress", "payload": {"items": items}})

    def test_when_pinned_no_panel_is_printed_per_step(self):
        renderer, stream = self._renderer(pinned=True)
        self._plan_events(renderer)
        self.assertNotIn("Plan progress", stream.getvalue())
        self.assertNotIn("Execution plan", stream.getvalue())
        self.assertNotIn("Assumptions", stream.getvalue())

    def test_the_pinned_panel_tracks_the_current_state(self):
        renderer, _ = self._renderer(pinned=True)
        self._plan_events(renderer)
        body = [_plain(line) for line in renderer.live_input_plan_lines(100, 30)][1:-1]
        self.assertEqual([row[2:5] for row in body], ["1. ", "2. ", "3. ", "4. "])
        self.assertIn("✓", body[0])
        self.assertIn("✓", body[1])
        self.assertIn("✓", body[2])
        self.assertIn("✓", body[3])  # the last event completed every step
        # ...and an intermediate event is reflected in place too
        renderer.handle_event({"event_type": "plan_step_progress", "payload": {"items": [
            dict(item, status="completed" if i < 1 else "in_progress" if i == 1 else "pending")
            for i, item in enumerate(FOUR)
        ]}})
        body = [_plain(line) for line in renderer.live_input_plan_lines(100, 30)][1:-1]
        self.assertEqual([("✓" in r, "◉" in r, "○" in r) for r in body],
                         [(True, False, False), (False, True, False), (False, False, True), (False, False, True)])

    def test_exactly_one_final_snapshot_is_printed_when_the_turn_ends(self):
        renderer, stream = self._renderer(pinned=True)
        self._plan_events(renderer)
        renderer.live_input_listener = None  # the composer has detached by now
        renderer.print_work_summary("completed")
        renderer.print_work_summary("completed")
        out = stream.getvalue()
        self.assertEqual(out.count("Plan progress"), 1, out)
        self.assertIn("Worked for", out)
        self.assertLess(out.index("Plan progress"), out.index("Worked for"))

    def test_without_the_composer_plans_still_print_as_before(self):
        renderer, stream = self._renderer(pinned=False)
        self._plan_events(renderer)
        self.assertIn("Execution plan", stream.getvalue())
        before = stream.getvalue().count("Plan progress")
        renderer.print_work_summary("completed")
        # not pinned: the per-step panels already are the record; no extra final snapshot
        self.assertEqual(stream.getvalue().count("Plan progress"), before)

    def test_a_turn_with_no_plan_prints_no_snapshot(self):
        renderer, stream = self._renderer(pinned=True)
        renderer.print_work_summary("completed")
        self.assertNotIn("Plan progress", stream.getvalue())


class ComposerTests(unittest.TestCase):
    def _lines(self, plan, width=100):
        renderer = StreamRenderer(_console())
        renderer._plan_steps = plan
        listener = LiveInputListener(session_id=1, renderer=renderer, cli_config=_config("ask"))
        with patch("shutil.get_terminal_size", return_value=os.terminal_size((width, 40))):
            text = "".join(t for _s, t in listener._composer_message().__pt_formatted_text__())
        return text.split("\n")

    def test_the_composer_draws_the_plan_above_the_status_with_breathing_room(self):
        lines = self._lines(FOUR)
        top = next(i for i, line in enumerate(lines) if line.startswith("╭"))
        bottom = next(i for i, line in enumerate(lines) if line.startswith("╰"))
        self.assertEqual(lines[top - 1].strip(), "", "a blank line above the panel")
        self.assertEqual(lines[bottom + 1].strip(), "", "a blank line between the panel and the status")
        status = next(i for i, line in enumerate(lines) if "…" in line and i > bottom)
        self.assertGreater(status, bottom + 1)
        self.assertTrue(lines[-1].startswith("❯"))

    def test_without_a_plan_the_composer_is_unchanged(self):
        lines = self._lines([])
        self.assertFalse(any(line.startswith(("╭", "╰")) for line in lines))


if __name__ == "__main__":
    unittest.main()


class ListenerShutdownTests(unittest.TestCase):
    def test_stopping_the_composer_commits_the_plan_even_without_an_outcome_status(self):
        import asyncio

        stream = StringIO()
        renderer = StreamRenderer(Console(file=stream, no_color=True, width=100))
        renderer._is_tty = True
        listener = LiveInputListener(session_id=1, renderer=renderer, cli_config=_config("ask"))
        renderer.live_input_listener = listener
        renderer.handle_event({"event_type": "plan_created", "payload": {"stage": "plan", "items": FOUR}})
        self.assertNotIn("Plan progress", stream.getvalue())
        asyncio.run(listener._stop_async())
        self.assertEqual(stream.getvalue().count("Plan progress"), 1)
