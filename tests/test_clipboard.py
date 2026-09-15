import base64
import subprocess
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

from rich.console import Console

from tamfis_code.clipboard import MAX_CLIPBOARD_CHARS, copy_to_clipboard, read_clipboard_image


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


def _completed(stdout: bytes = b"", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=b"")


class ReadClipboardImageLinuxTests(unittest.TestCase):
    def setUp(self):
        self._platform_patch = patch("tamfis_code.clipboard.platform.system", return_value="Linux")
        self._platform_patch.start()
        self.addCleanup(self._platform_patch.stop)

    def test_wayland_session_uses_wl_paste_when_an_image_is_present(self):
        with patch.dict("os.environ", {"WAYLAND_DISPLAY": "wayland-0"}, clear=False), \
             patch("tamfis_code.clipboard.shutil.which", side_effect=lambda name: "/usr/bin/wl-paste" if name == "wl-paste" else None), \
             patch("tamfis_code.clipboard._run") as fake_run:
            fake_run.side_effect = [
                _completed(stdout=b"image/png\ntext/plain\n"),
                _completed(stdout=b"\x89PNG\r\n...fakebytes"),
            ]
            data, reason = read_clipboard_image()
        self.assertEqual(data, b"\x89PNG\r\n...fakebytes")
        self.assertEqual(reason, "")

    def test_wayland_session_with_no_image_type_reports_no_image(self):
        with patch.dict("os.environ", {"WAYLAND_DISPLAY": "wayland-0"}, clear=False), \
             patch("tamfis_code.clipboard.shutil.which", side_effect=lambda name: "/usr/bin/wl-paste" if name == "wl-paste" else None), \
             patch("tamfis_code.clipboard._run", return_value=_completed(stdout=b"text/plain\n")):
            data, reason = read_clipboard_image()
        self.assertIsNone(data)
        self.assertIn("no image", reason)

    def test_x11_session_uses_xclip_when_an_image_is_present(self):
        with patch.dict("os.environ", {}, clear=True), \
             patch("tamfis_code.clipboard.shutil.which", side_effect=lambda name: "/usr/bin/xclip" if name == "xclip" else None), \
             patch("tamfis_code.clipboard._run") as fake_run:
            fake_run.side_effect = [
                _completed(stdout=b"image/png\nSTRING\n"),
                _completed(stdout=b"\x89PNG\r\n...fakebytes"),
            ]
            data, reason = read_clipboard_image()
        self.assertEqual(data, b"\x89PNG\r\n...fakebytes")

    def test_no_clipboard_tool_installed_reports_actionable_reason(self):
        with patch.dict("os.environ", {}, clear=True), \
             patch("tamfis_code.clipboard.shutil.which", return_value=None):
            data, reason = read_clipboard_image()
        self.assertIsNone(data)
        self.assertIn("xclip", reason)

    def test_a_tool_failure_is_reported_not_raised(self):
        with patch.dict("os.environ", {}, clear=True), \
             patch("tamfis_code.clipboard.shutil.which", side_effect=lambda name: "/usr/bin/xclip" if name == "xclip" else None), \
             patch("tamfis_code.clipboard._run", side_effect=subprocess.TimeoutExpired(cmd="xclip", timeout=10)):
            data, reason = read_clipboard_image()
        self.assertIsNone(data)
        self.assertIn("xclip failed", reason)


class ReadClipboardImageMacosTests(unittest.TestCase):
    def test_a_real_image_on_the_clipboard_is_read_back(self):
        real_bytes = b"\x89PNG\r\n\x1a\ncertainly-not-a-real-png-but-real-bytes"

        def fake_run(args, input_bytes=None):
            # Simulate osascript writing the clipboard image to the temp
            # path the real implementation passes on its own command line.
            tmp_path = Path(args[-1].split('POSIX file "')[1].split('"')[0])
            tmp_path.write_bytes(real_bytes)
            return _completed(stdout=b"")

        with patch("tamfis_code.clipboard.platform.system", return_value="Darwin"), \
             patch("tamfis_code.clipboard._run", side_effect=fake_run):
            data, reason = read_clipboard_image()
        self.assertEqual(data, real_bytes)
        self.assertEqual(reason, "")

    def test_no_image_on_the_clipboard_is_reported_cleanly(self):
        def fake_run(args, input_bytes=None):
            return _completed(stdout=b"NO_IMAGE\n")

        with patch("tamfis_code.clipboard.platform.system", return_value="Darwin"), \
             patch("tamfis_code.clipboard._run", side_effect=fake_run):
            data, reason = read_clipboard_image()
        self.assertIsNone(data)
        self.assertIn("no image", reason)


class ReadClipboardImageWindowsTests(unittest.TestCase):
    def test_a_real_image_on_the_clipboard_is_read_back(self):
        real_bytes = b"\x89PNG\r\n\x1a\nfake-windows-clipboard-bytes"

        def fake_run(args, input_bytes=None):
            tmp_path = Path(args[-1].split("Save('")[1].split("'")[0])
            tmp_path.write_bytes(real_bytes)
            return _completed(stdout=b"OK\n")

        with patch("tamfis_code.clipboard.platform.system", return_value="Windows"), \
             patch("tamfis_code.clipboard._run", side_effect=fake_run):
            data, reason = read_clipboard_image()
        self.assertEqual(data, real_bytes)

    def test_no_image_on_the_clipboard_is_reported_cleanly(self):
        with patch("tamfis_code.clipboard.platform.system", return_value="Windows"), \
             patch("tamfis_code.clipboard._run", return_value=_completed(stdout=b"NO_IMAGE\n")):
            data, reason = read_clipboard_image()
        self.assertIsNone(data)
        self.assertIn("no image", reason)


if __name__ == "__main__":
    unittest.main()
