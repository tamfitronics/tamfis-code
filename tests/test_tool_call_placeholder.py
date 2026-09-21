"""A model that echoes the request-side "[tool call]" placeholder must not show an empty Assistant panel.

Owner report 2026-09-21: after "Recovering with another TamfisGPT route", the log filled with boxed
Assistant panels reading only "[tool call]" between the tool blocks.
"""
import io
import unittest

from rich.console import Console

from tamfis_code.provider_protocols import TOOL_CALL_PLACEHOLDER, is_tool_call_placeholder
from tamfis_code.render import StreamRenderer


class PlaceholderTests(unittest.TestCase):
    def test_only_the_bare_placeholder_matches(self):
        for text in (TOOL_CALL_PLACEHOLDER, " [tool call] \n", "[Tool Call]", "[tool_calls]", "tool call"):
            self.assertTrue(is_tool_call_placeholder(text), text)
        for text in ("", "I made a tool call to read it", "[tool call] then more text", None, 3):
            self.assertFalse(is_tool_call_placeholder(text), text)

    def test_an_echoed_placeholder_prints_no_assistant_panel(self):
        console = Console(file=io.StringIO(), width=100, force_terminal=False)
        renderer = StreamRenderer(console)
        renderer._is_tty = True
        renderer.live_input_listener = object()
        renderer.handle_event({"event_type": "assistant_delta", "payload": {"content": "[tool call]"}})
        renderer._close_assistant()
        output = console.file.getvalue()
        self.assertNotIn("tool call", output)
        self.assertNotIn("Assistant", output)

    def test_real_text_still_gets_its_panel(self):
        console = Console(file=io.StringIO(), width=100, force_terminal=False)
        renderer = StreamRenderer(console)
        renderer._is_tty = True
        renderer.live_input_listener = object()
        renderer.handle_event({"event_type": "assistant_delta", "payload": {"content": "The lock file is stale."}})
        renderer._close_assistant()
        self.assertIn("The lock file is stale.", console.file.getvalue())


if __name__ == "__main__":
    unittest.main()
