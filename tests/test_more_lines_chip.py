"""Tests for the clickable Show more/less lines chip (message_viewer.chip_fragments).

Contract: tool-output blocks print "… +N lines (Ctrl+O for full output)" into Rich
scrollback, which can never be clickable. The chip is the same affordance rendered
in the prompt_toolkit composer area with a real mouse handler: clicking it opens the
transcript viewer, clicking again (now "Show less lines") closes it. The chip is
absent when there is nothing to expand, and the mouse-capture filter tracks that so
the terminal's native wheel scrolling is preserved whenever nothing is clickable.
"""
import unittest

from tamfis_code import message_viewer as mv
from tamfis_code.render import COLLAPSED_MESSAGES, TOOL_TRANSCRIPT


class FakeMouseEvent:
    def __init__(self, event_type):
        self.event_type = event_type


def _click(fragment):
    handler = fragment[2]
    handler(FakeMouseEvent(_mouse_up()))
    return True


def _mouse_up():
    from prompt_toolkit.mouse_events import MouseEventType

    return MouseEventType.MOUSE_UP


class ChipStateTests(unittest.TestCase):
    def setUp(self):
        COLLAPSED_MESSAGES.clear()
        TOOL_TRANSCRIPT.clear()
        mv.VIEWER.close()

    def tearDown(self):
        COLLAPSED_MESSAGES.clear()
        TOOL_TRANSCRIPT.clear()
        mv.VIEWER.close()

    def test_no_chip_when_nothing_pending(self):
        self.assertEqual(mv.chip_fragments(), [])
        self.assertFalse(mv.mouse_capture_active())

    def test_chip_shows_pending_output_count(self):
        TOOL_TRANSCRIPT.add("tool", "x\ny\nz")
        COLLAPSED_MESSAGES.add("assistant", "long message")
        self.assertEqual(mv.pending_expansion_count(), 2)
        chip = mv.chip_fragments()
        self.assertTrue(chip)
        text = "".join(fragment[1] for fragment in chip)
        self.assertIn("Show more lines (2)", text)
        self.assertIn("Ctrl+O", text)
        self.assertTrue(mv.mouse_capture_active())

    def test_click_opens_tool_transcript_viewer(self):
        TOOL_TRANSCRIPT.add("tool", "$ ls\nfile.txt")
        chip = mv.chip_fragments()
        _click(chip[1])
        self.assertTrue(mv.VIEWER.is_open)

    def test_click_falls_back_to_messages_when_no_tool_output(self):
        COLLAPSED_MESSAGES.add("assistant", "a long message")
        chip = mv.chip_fragments()
        _click(chip[1])
        self.assertTrue(mv.VIEWER.is_open)

    def test_open_viewer_chip_offers_show_less_and_click_closes(self):
        TOOL_TRANSCRIPT.add("tool", "$ ls\nfile.txt")
        _click(mv.chip_fragments()[1])
        chip = mv.chip_fragments()
        text = "".join(fragment[1] for fragment in chip)
        self.assertIn("Show less lines", text)
        self.assertNotIn("Show more lines", text)
        _click(chip[1])
        self.assertFalse(mv.VIEWER.is_open)
        # Closed again and the transcript is still pending -> chip returns.
        self.assertTrue(mv.chip_fragments())

    def test_mouse_capture_tracks_viewer_and_pending(self):
        self.assertFalse(mv.mouse_capture_active())
        TOOL_TRANSCRIPT.add("tool", "a\nb")
        self.assertTrue(mv.mouse_capture_active())
        _click(mv.chip_fragments()[1])          # open
        self.assertTrue(mv.mouse_capture_active())
        mv.VIEWER.close()
        self.assertTrue(mv.mouse_capture_active())   # still pending
        TOOL_TRANSCRIPT.take_next()              # mark expanded
        self.assertFalse(mv.mouse_capture_active())

    def test_chip_ignores_non_mouse_up_events(self):
        from prompt_toolkit.mouse_events import MouseEventType

        TOOL_TRANSCRIPT.add("tool", "a\nb")
        chip = mv.chip_fragments()
        chip[1][2](FakeMouseEvent(MouseEventType.MOUSE_DOWN))
        self.assertFalse(mv.VIEWER.is_open)

    def test_wheel_scroll_moves_open_viewer_and_tolerates_closed(self):
        content = "\n".join(f"line {i}" for i in range(80))
        TOOL_TRANSCRIPT.add("tool", content)
        _click(mv.chip_fragments()[1])
        before = mv.VIEWER._offset
        mv.wheel_scroll(3, None)
        self.assertEqual(mv.VIEWER._offset, before + 3)
        mv.wheel_scroll(-1, None)
        self.assertEqual(mv.VIEWER._offset, before + 2)
        mv.VIEWER.close()
        # Closed viewer: inert, never raises.
        mv.wheel_scroll(5, None)


class ComposerIntegrationTests(unittest.TestCase):
    def setUp(self):
        COLLAPSED_MESSAGES.clear()
        TOOL_TRANSCRIPT.clear()
        mv.VIEWER.close()

    def tearDown(self):
        COLLAPSED_MESSAGES.clear()
        TOOL_TRANSCRIPT.clear()
        mv.VIEWER.close()

    def _listener(self):
        from tamfis_code.config import Config
        from tamfis_code.live_input import LiveInputListener

        class FakeRenderer:
            def __init__(self):
                self.progress = None
                self._running_command = None
                self._active_agents = 0

            def live_input_plan_lines(self, columns, rows):
                return []

            def live_input_activity_line(self):
                return None

            def live_input_headline(self, spinner, width):
                return "Working…"

            def steering_revision(self):
                return 0

            def current_activity(self, include_command=True):
                return ""

            def __getattr__(self, name):
                return lambda *a, **k: None

        listener = LiveInputListener.__new__(LiveInputListener)
        listener.renderer = FakeRenderer()
        listener._status_tick = 0
        listener._active_agents = 0
        listener.session_id = 1
        listener.cli_config = Config()
        listener._interrupt_classification = None
        return listener

    def test_live_composer_renders_chip_once_with_click_handlers(self):
        TOOL_TRANSCRIPT.add("tool", "$ cmd\nout1\nout2\nout3\nout4\nout5\n")
        formatted = self._listener()._composer_message()
        text = "".join(fragment[1] for fragment in formatted)
        self.assertEqual(text.count("Show more lines"), 1)
        clickable = [f for f in formatted if len(f) >= 3 and callable(f[2])]
        self.assertTrue(clickable)
        # The chip sits above the input prompt glyph.
        self.assertLess(text.find("Show more lines"), text.find("❯"))

    def test_live_composer_without_pending_output_has_no_chip(self):
        from prompt_toolkit.formatted_text import to_formatted_text

        message = self._listener()._composer_message()
        formatted = to_formatted_text(message)
        text = "".join(fragment[1] for fragment in formatted)
        self.assertNotIn("Show more lines", text)
        self.assertNotIn("Show less lines", text)
        self.assertFalse([f for f in formatted if len(f) >= 3 and callable(f[2])])


if __name__ == "__main__":
    unittest.main()
