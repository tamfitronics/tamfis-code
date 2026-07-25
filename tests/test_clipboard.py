import base64
import unittest
from io import StringIO

from rich.console import Console

from tamfis_code.clipboard import MAX_CLIPBOARD_CHARS, copy_to_clipboard


def _console(is_terminal: bool) -> Console:
    # Console.is_terminal is a read-only property derived from the
    # underlying file: force_terminal=True fakes a TTY for a StringIO,
    # which otherwise (correctly) never reports isatty() as true.
    return Console(file=StringIO(), no_color=True, width=200, force_terminal=is_terminal or None)


class CopyToClipboardTests(unittest.TestCase):
    def test_writes_osc52_sequence_when_terminal_attached(self):
        console = _console(is_terminal=True)
        result = copy_to_clipboard(console, "hello clipboard")

        self.assertTrue(result)
        written = console.file.getvalue()
        self.assertTrue(written.startswith("\x1b]52;c;"))
        self.assertTrue(written.endswith("\x07"))
        encoded = written[len("\x1b]52;c;"):-1]
        self.assertEqual(base64.b64decode(encoded).decode("utf-8"), "hello clipboard")

    def test_does_nothing_without_an_attached_terminal(self):
        console = _console(is_terminal=False)
        result = copy_to_clipboard(console, "hello clipboard")

        self.assertFalse(result)
        self.assertEqual(console.file.getvalue(), "")

    def test_very_long_text_is_truncated_not_dropped(self):
        console = _console(is_terminal=True)
        text = "x" * (MAX_CLIPBOARD_CHARS + 5_000)

        result = copy_to_clipboard(console, text)

        self.assertTrue(result)
        written = console.file.getvalue()
        encoded = written[len("\x1b]52;c;"):-1]
        decoded = base64.b64decode(encoded).decode("utf-8")
        self.assertEqual(len(decoded), MAX_CLIPBOARD_CHARS)


if __name__ == "__main__":
    unittest.main()
