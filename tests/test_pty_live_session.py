"""End-to-end: the real CLI in a real pseudo-terminal, against a fake provider.

Reproduces the incident: tools finish, the next model request hangs, and the
user keeps typing. Verifies Enter is processed and reaches the model as a
follow-up, `/status` answers locally, Ctrl+D / focus events / Esc never turn
into painted ``^[`` text, the tty stays in raw (no-echo) mode, and Esc
cancels cleanly.
"""
import asyncio
import fcntl
import json
import os
import select
import struct
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

try:
    import pty
    import termios
except ImportError:  # pragma: no cover - non-POSIX
    pty = termios = None

REPO = Path(__file__).resolve().parents[1]


class FakeProvider:
    """OpenAI-compatible SSE server: round 1 = one tool call, then it hangs."""

    def __init__(self, first_tool=None):
        self.first_tool = first_tool          # (name, arguments dict) instead of the default list_directory
        self.requests = []
        self.port = 0
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        self._ready.wait(10)

    def stop(self):
        def _shutdown():
            for task in asyncio.all_tasks(self._loop):
                task.cancel()
            self._loop.call_later(0.2, self._loop.stop)

        self._loop.call_soon_threadsafe(_shutdown)
        self._thread.join(5)

    def _run(self):
        asyncio.set_event_loop(self._loop)
        server = self._loop.run_until_complete(asyncio.start_server(self._handle, "127.0.0.1", 0))
        self.port = server.sockets[0].getsockname()[1]
        self._ready.set()
        self._loop.run_forever()

    async def _handle(self, reader, writer):
        head = await reader.readuntil(b"\r\n\r\n")
        lines = head.decode().split("\r\n")
        headers = {l.split(":", 1)[0].lower(): l.split(":", 1)[1].strip() for l in lines[1:] if ":" in l}
        body = await reader.readexactly(int(headers.get("content-length", 0))) if headers.get("content-length") else b""
        try:
            req = json.loads(body or b"{}")
        except ValueError:
            req = {}
        msgs = req.get("messages") or []
        if not req.get("stream"):
            out = json.dumps({"id": "x", "object": "chat.completion", "model": "m", "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "4"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}).encode()
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % len(out) + out)
            await writer.drain()
            writer.close()
            return
        self.requests.append(msgs)
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n")

        async def send(obj):
            data = f"data: {json.dumps(obj)}\n\n".encode()
            writer.write(b"%x\r\n" % len(data) + data + b"\r\n")
            await writer.drain()

        base = {"id": "c", "object": "chat.completion.chunk", "model": "m"}
        if not any(m.get("role") == "tool" for m in msgs):
            await send({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]})
            tool_name, tool_args = self.first_tool or ("list_directory", {"path": "."})
            await send({**base, "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": tool_name, "arguments": json.dumps(tool_args)}}]}, "finish_reason": None}]})
            await send({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
            writer.write(b"11\r\ndata: [DONE]\n\n\r\n0\r\n\r\n")
            await writer.drain()
            writer.close()
            return
        await send({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]})
        await asyncio.sleep(3600)  # provider hangs after its first event


@unittest.skipIf(pty is None or not sys.platform.startswith("linux"), "needs a POSIX pty")
class RealTerminalIncidentTest(unittest.TestCase):
    FIRST_TOOL = None

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=str(REPO))  # not /tmp: see release notes
        self.provider = FakeProvider(first_tool=self.first_tool())
        self.provider.start()
        base = Path(self.tmp.name)
        (base / "home").mkdir()
        (base / "ws").mkdir()
        (base / "ws" / "a.txt").write_text("hi\n")
        (base / "empty.env").write_text("")
        env = {
            "PATH": os.environ["PATH"], "HOME": str(base / "home"), "TERM": "xterm-256color",
            "LANG": "C.UTF-8", "PYTHONPATH": str(REPO),
            "OLLAMA_BASE_URL": f"http://127.0.0.1:{self.provider.port}/v1", "OLLAMA_API_KEY": "dummy",
            "TAMFIS_CODE_ENV_FILE": str(base / "empty.env"),
        }
        self.out = bytearray()
        self.pid, self.fd = pty.fork()
        if self.pid == 0:
            os.chdir(base / "ws")
            os.execvpe(sys.executable, [sys.executable, "-m", "tamfis_code", "--new-session", "--approval", "auto"], env)
        fcntl.ioctl(self.fd, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 0, 0))

    def first_tool(self):
        return self.FIRST_TOOL

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
        self.provider.stop()
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

    def expect(self, needle, timeout):
        end = time.time() + timeout
        while time.time() < end:
            if needle in bytes(self.out):
                return True
            self.pump(0.2)
        return needle in bytes(self.out)

    def echo_on(self):
        return bool(termios.tcgetattr(self.fd)[3] & termios.ECHO)

    def test_incident_sequence_keeps_the_composer_alive(self):
        self.assertTrue(self.expect("❯".encode(), 60), "no idle prompt")
        self.pump(1)
        self.out.clear()
        os.write(self.fd, b"list the files in this directory\r")

        # Tools complete, then the model request hangs -> "Waiting for the model".
        self.assertTrue(self.expect(b"Waiting for the model", 60), "never reached the wait state")
        self.pump(2)
        self.assertEqual(len(self.provider.requests), 2, "continuation request not dispatched after tools")
        self.assertFalse(self.echo_on(), "tty in echo mode while the composer should own it")

        # The incident: Ctrl+D, focus out/in, several Esc-free keystrokes.
        os.write(self.fd, b"\x04")
        self.pump(1.5)
        os.write(self.fd, b"\x1b[O\x1b[I\x1b[O")
        self.pump(1.5)
        self.assertFalse(self.echo_on(), "Ctrl+D / focus events ended the composer (tty went cooked)")
        self.assertNotIn(b"^[", bytes(self.out), "raw escape sequence painted as text")

        # /status answers locally, mid-wait.
        os.write(self.fd, b"/status\r")
        self.assertTrue(self.expect(b"Pending tools", 15), "/status did not answer")
        self.assertTrue(self.expect(b"Queued follow-ups", 5))

        # A follow-up: Enter is acknowledged immediately and reaches the model.
        before = len(self.provider.requests)
        os.write(self.fd, b"Also inspect the gateway.")
        self.pump(0.5)
        os.write(self.fd, b"\r")
        self.assertTrue(self.expect(b"Follow-up queued", 10), "no immediate acknowledgement")
        deadline = time.time() + 30
        while time.time() < deadline and len(self.provider.requests) <= before:
            self.pump(0.5)
        self.assertGreater(len(self.provider.requests), before, "follow-up never reached orchestration")
        last = json.dumps(self.provider.requests[-1])
        self.assertIn("Also inspect the gateway.", last)

        # Esc cancels; the terminal stays usable (idle prompt returns, raw mode).
        self.out.clear()
        os.write(self.fd, b"\x1b")
        self.assertTrue(self.expect(b"Stopped", 30), "Esc did not stop the task")
        self.pump(2)
        self.assertNotIn(b"^[", bytes(self.out))
        self.assertFalse(self.echo_on())
        self.assertIn("❯".encode(), bytes(self.out))


class ComposerSurvivesAnApprovalGateTest(RealTerminalIncidentTest):
    """Live report 2026-09-21: after an approval gate the composer was dead -- Enter did nothing, keys
    were echoed as text and follow-ups were never sent. Cause: the gate pauses the composer, and a stale
    "programmatic exit" marker made the restarted composer treat the user's FIRST Enter as a teardown."""

    def first_tool(self):
        # Outside the workspace and outside the tamfis-code source tree (writes into a source tree are
        # rated dangerous and would wait on a human prompt), named by the user in the request.
        self._outside = Path(tempfile.mkdtemp(prefix="tamfis-approval-"))
        self.target = self._outside / "target.txt"
        return ("write_file", {"path": str(self.target), "content": "x\n"})

    def tearDown(self):
        import shutil

        shutil.rmtree(getattr(self, "_outside", ""), ignore_errors=True)
        super().tearDown()

    def test_incident_sequence_keeps_the_composer_alive(self):  # replaced below
        self.skipTest("covered by test_follow_up_after_an_approval_gate_reaches_the_model")

    def test_follow_up_after_an_approval_gate_reaches_the_model(self):
        self.assertTrue(self.expect("❯".encode(), 60), "no idle prompt")
        self.pump(1)
        self.out.clear()
        os.write(self.fd, f"create the file {self.target} with one line\r".encode())
        deadline = time.time() + 60
        while time.time() < deadline and len(self.provider.requests) < 2:
            self.pump(0.5)
        self.assertEqual(len(self.provider.requests), 2, "the tool call never completed / no continuation request")
        self.pump(2)   # the composer has been paused and resumed around the tool's approval by now
        self.assertFalse(self.echo_on(), "tty in echo mode right after the approval gate")

        before = len(self.provider.requests)
        os.write(self.fd, b"Also inspect the gateway.")
        self.pump(0.5)
        os.write(self.fd, b"\r")
        self.assertTrue(self.expect(b"Follow-up queued", 10), "Enter was not acknowledged after the approval")
        deadline = time.time() + 30
        while time.time() < deadline and len(self.provider.requests) <= before:
            self.pump(0.5)
        self.assertGreater(len(self.provider.requests), before, "the follow-up never reached the model")
        self.assertIn("Also inspect the gateway.", json.dumps(self.provider.requests[-1]))
        self.assertFalse(self.echo_on(), "the tty went cooked after the user's first Enter")
        self.assertNotIn(b"^[", bytes(self.out))


if __name__ == "__main__":
    unittest.main()
