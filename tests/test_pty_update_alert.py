"""An installed app must tell the user a newer release exists -- instantly, offline, without capturing the mouse.

* The alert comes from the on-disk cache, so a launch never waits on the network.
  * It must NOT turn on terminal mouse tracking: with tracking on, the mouse wheel is delivered to the app
  instead of scrolling the terminal's scrollback (jumpy scrolling, wheel turned into Up/Down history keys),
  and native drag-select/copy of streamed errors stops working. The main composers are keyboard-only;
  only the deliberate clarification/approval prompt captures the mouse temporarily.
"""
import json
import os
import pty
import select
import struct
import sys
import tempfile
import time
import unittest
from pathlib import Path

try:
    import fcntl
    import termios
except ImportError:  # pragma: no cover
    fcntl = termios = None

REPO = Path(__file__).resolve().parents[1]
BASE = "https://gpt.tamfitronics.com/releases/tamfis-code"
# any DEC private mode that makes the terminal report mouse events to the app
MOUSE_ON = (b"\x1b[?1000h", b"\x1b[?1002h", b"\x1b[?1003h", b"\x1b[?1005h", b"\x1b[?1006h", b"\x1b[?1015h")


def _newer(version: str) -> str:
    major, minor, patch = (int(x) for x in version.split("."))
    return f"{major}.{minor}.{patch + 5}"


@unittest.skipIf(fcntl is None or not sys.platform.startswith("linux"), "needs a POSIX pty")
class UpdateAlertTerminalTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=str(REPO))
        base = Path(self.tmp.name)
        (base / "ws").mkdir()
        (base / "empty.env").write_text("")
        sys.path.insert(0, str(REPO))
        from tamfis_code import __version__

        self.available = _newer(__version__)
        config = base / "home" / ".config" / "tamfis-code"
        config.mkdir(parents=True)
        (config / "update_check.json").write_text(json.dumps({
            "release": {
                "version": self.available,
                "url": f"{BASE}/tamfis_code-{self.available}-py3-none-any.whl",
                "sha256": "a" * 64,
            },
            "checked_at": time.time(),          # fresh: the background refresh is skipped, no network at all
        }))
        env = {
            "PATH": os.environ["PATH"], "HOME": str(base / "home"), "TERM": "xterm-256color",
            "LANG": "C.UTF-8", "PYTHONPATH": str(REPO), "TAMFIS_CODE_ENV_FILE": str(base / "empty.env"),
            "TAMFIS_CODE_REPO": str(base / "no-checkout"),        # the checkout must not answer for "published"
        }
        self.out = bytearray()
        self.pid, self.fd = pty.fork()
        if self.pid == 0:
            os.chdir(base / "ws")
            os.execvpe(sys.executable, [sys.executable, "-m", "tamfis_code", "--new-session"], env)
        fcntl.ioctl(self.fd, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 0, 0))

    def tearDown(self):
        try:
            os.kill(self.pid, 9)
            os.waitpid(self.pid, 0)
        except OSError:
            pass
        try:
            os.close(self.fd)
        except OSError:
            pass
        self.tmp.cleanup()

    def pump(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            ready, _, _ = select.select([self.fd], [], [], 0.1)
            if ready:
                try:
                    data = os.read(self.fd, 65536)
                except OSError:
                    return
                if not data:
                    return
                self.out.extend(data)

    def test_update_is_announced_at_startup_and_the_mouse_is_never_captured(self):
        deadline = time.time() + 60
        while time.time() < deadline and "❯".encode() not in bytes(self.out):
            self.pump(0.3)
        self.pump(2)
        text = bytes(self.out).decode("utf-8", "replace")
        self.assertIn("Update available", text)
        self.assertIn(self.available, text)
        self.assertIn("Ctrl+U", text)
        for sequence in MOUSE_ON:
            self.assertNotIn(sequence, bytes(self.out), f"mouse tracking enabled ({sequence!r}): the wheel stops scrolling")


if __name__ == "__main__":
    unittest.main()
