"""Ctrl+E shows a long message in full and Ctrl+E / Esc shows less again.

Owner report 2026-09-20: "there is no way of showing less after Ctrl+E for showing more".
The full text used to be printed into scrollback (irreversible) with a dead "show less"
line. It now opens in a viewer above the composer that the same key closes.
"""
import os
import re
import unittest
from io import StringIO
from unittest.mock import MagicMock, patch

from rich.console import Console

from tamfis_code.live_input import LiveInputListener
from tamfis_code.message_viewer import VIEWER, install_bindings, panel_ansi, viewer_rows
from tamfis_code.render import COLLAPSED_MESSAGES, CollapsedMessageStore, StreamRenderer

from test_live_input import _config, _console


def _plain(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _long(tag="A", words=600):
    return f"{tag}-start " + " ".join(f"word{i}" for i in range(words)) + f" {tag}-END"


class _Base(unittest.TestCase):
    def setUp(self):
        VIEWER.close()
        COLLAPSED_MESSAGES.clear()
        self.addCleanup(VIEWER.close)
        self.addCleanup(COLLAPSED_MESSAGES.clear)


class ViewerStateTests(_Base):
    def test_toggle_opens_the_newest_collapsed_message_then_closes(self):
        store = CollapsedMessageStore()
        store.add("assistant", "old message " * 100)
        store.add("assistant", _long("NEW"))
        self.assertTrue(VIEWER.toggle(store))
        self.assertTrue(VIEWER.is_open)
        self.assertIn("NEW-start", _plain("\n".join(VIEWER.panel_lines(100, 40))))
        self.assertTrue(VIEWER.toggle(store))
        self.assertFalse(VIEWER.is_open)
        self.assertEqual(VIEWER.panel_lines(100, 40), [])

    def test_showing_less_leaves_the_message_collapsed_and_reopenable(self):
        store = CollapsedMessageStore()
        store.add("assistant", _long())
        VIEWER.toggle(store)
        VIEWER.toggle(store)
        self.assertEqual(store.pending(), 1)
        self.assertTrue(VIEWER.toggle(store))

    def test_nothing_collapsed_is_a_no_op(self):
        self.assertFalse(VIEWER.toggle(CollapsedMessageStore()))
        self.assertFalse(VIEWER.is_open)

    def test_scrolling_moves_the_window_and_is_clamped(self):
        store = CollapsedMessageStore()
        store.add("assistant", _long(words=1500))
        VIEWER.toggle(store)
        first = _plain("\n".join(VIEWER.panel_lines(100, 30)))
        self.assertIn("lines 1-", first)
        VIEWER.scroll(5)
        self.assertIn("lines 6-", _plain("\n".join(VIEWER.panel_lines(100, 30))))
        VIEWER.scroll(-999)
        self.assertIn("lines 1-", _plain("\n".join(VIEWER.panel_lines(100, 30))))
        VIEWER.end()
        last = _plain("\n".join(VIEWER.panel_lines(100, 30)))
        self.assertIn("END", last)
        VIEWER.scroll(999)  # past the end stays on the last page
        self.assertIn("END", _plain("\n".join(VIEWER.panel_lines(100, 30))))
        VIEWER.page(-1)
        self.assertNotIn("END", _plain("\n".join(VIEWER.panel_lines(100, 30))))
        VIEWER.home()
        self.assertIn("lines 1-", _plain("\n".join(VIEWER.panel_lines(100, 30))))

    def test_left_and_right_walk_between_collapsed_messages(self):
        store = CollapsedMessageStore()
        store.add("assistant", _long("ONE"))
        store.add("assistant", _long("TWO"))
        VIEWER.toggle(store)
        self.assertIn("TWO-start", _plain("\n".join(VIEWER.panel_lines(100, 40))))
        VIEWER.other(-1)
        text = _plain("\n".join(VIEWER.panel_lines(100, 40)))
        self.assertIn("ONE-start", text)
        self.assertIn("message 1 of 2", text)
        VIEWER.other(-1)  # clamped at the oldest
        self.assertIn("ONE-start", _plain("\n".join(VIEWER.panel_lines(100, 40))))
        VIEWER.other(1)
        self.assertIn("TWO-start", _plain("\n".join(VIEWER.panel_lines(100, 40))))

    def test_the_panel_says_how_to_show_less_and_fits_the_terminal(self):
        store = CollapsedMessageStore()
        store.add("assistant", _long(words=2000))
        VIEWER.toggle(store)
        lines = VIEWER.panel_lines(100, 30)
        self.assertIn("Ctrl+E or Esc to show less", _plain(lines[-2]))
        self.assertEqual(lines[0], "")
        self.assertEqual(lines[-1], "")
        self.assertLessEqual(len(lines) - 4, viewer_rows(30))
        for line in lines:
            self.assertLessEqual(len(_plain(line)), 100)

    def test_a_short_terminal_still_shows_a_usable_window(self):
        self.assertGreaterEqual(viewer_rows(10), 6)
        self.assertLessEqual(viewer_rows(500), 40)

    def test_a_user_message_is_shown_verbatim(self):
        store = CollapsedMessageStore()
        store.add("user", "please [keep] <these> literally " * 30)
        VIEWER.toggle(store)
        text = _plain("\n".join(VIEWER.panel_lines(100, 30)))
        self.assertIn("You · full message", text)
        self.assertIn("[keep] <these>", text)


class BindingTests(_Base):
    def _bindings(self):
        from prompt_toolkit.key_binding import KeyBindings

        bindings = KeyBindings()
        install_bindings(bindings)
        return bindings

    def _run(self, bindings, key):
        from prompt_toolkit.keys import Keys  # noqa: F401

        event = MagicMock()
        for binding in bindings.bindings:
            if [getattr(k, "value", k) for k in binding.keys] == [key] and binding.filter():
                binding.handler(event)
                return event
        return None

    def test_the_scroll_keys_only_act_while_the_viewer_is_open(self):
        bindings = self._bindings()
        self.assertIsNone(self._run(bindings, "escape"))
        store = CollapsedMessageStore()
        store.add("assistant", _long(words=1500))
        VIEWER.toggle(store)
        VIEWER.panel_lines(100, 30)
        self.assertIsNotNone(self._run(bindings, "down"))
        self.assertIn("lines 2-", _plain("\n".join(VIEWER.panel_lines(100, 30))))

    def test_escape_closes_the_viewer(self):
        bindings = self._bindings()
        store = CollapsedMessageStore()
        store.add("assistant", _long())
        VIEWER.toggle(store)
        event = self._run(bindings, "escape")
        self.assertIsNotNone(event)
        self.assertFalse(VIEWER.is_open)
        event.app.invalidate.assert_called()


class ComposerIntegrationTests(_Base):
    def _text(self, listener, width=100):
        with patch("shutil.get_terminal_size", return_value=os.terminal_size((width, 40))):
            return "".join(t for _s, t in listener._composer_message().__pt_formatted_text__())

    def test_the_running_composer_shows_the_full_message_and_then_less(self):
        renderer = StreamRenderer(_console())
        renderer._collapse_or_print_assistant(_long("LIVE", words=300))
        listener = LiveInputListener(session_id=1, renderer=renderer, cli_config=_config("ask"))
        closed = self._text(listener)
        self.assertNotIn("full message", closed)
        VIEWER.toggle()
        opened = self._text(listener)
        self.assertIn("LIVE-start", opened)
        self.assertIn("Ctrl+E or Esc to show less", opened)
        self.assertTrue(opened.rstrip().endswith("❯"))
        VIEWER.toggle()
        self.assertEqual(self._text(listener), closed)

    def test_the_viewer_takes_the_room_of_the_pinned_plan(self):
        renderer = StreamRenderer(_console())
        renderer._plan_steps = [{"step": "Read the file", "status": "in_progress"}]
        renderer._collapse_or_print_assistant(_long("LIVE", words=300))
        listener = LiveInputListener(session_id=1, renderer=renderer, cli_config=_config("ask"))
        self.assertIn("Plan progress", self._text(listener))
        VIEWER.toggle()
        self.assertNotIn("Plan progress", self._text(listener))


class LegacyPrintPathTests(_Base):
    def test_the_printing_expansion_no_longer_claims_a_show_less_it_cannot_do(self):
        console = _console()
        renderer = StreamRenderer(console)
        renderer._collapse_or_print_assistant(_long())
        self.assertTrue(renderer.expand_next_collapsed_message())
        self.assertNotIn("show less", console.file.getvalue())


if __name__ == "__main__":
    unittest.main()
