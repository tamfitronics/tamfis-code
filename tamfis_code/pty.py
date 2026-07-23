"""Small local PTY broker used by standalone interactive sessions.

The broker drains the master fd continuously, so a noisy child cannot block
because nobody has issued ``/pty read`` yet.  Output is a bounded byte ring;
the terminal remains responsive even when a process emits unbounded logs.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import os
import pty
import signal
import struct
import termios
import uuid
from dataclasses import dataclass, field


@dataclass
class LocalPty:
    id: str
    pid: int
    fd: int
    command: str
    output: bytearray = field(default_factory=bytearray)
    total: int = 0
    status: str = "running"


class LocalPtyBroker:
    MAX_BUFFER = 2 * 1024 * 1024

    def __init__(self, *, cwd: str, loop: asyncio.AbstractEventLoop | None = None):
        self.cwd = cwd
        self.loop = loop or asyncio.get_running_loop()
        self.sessions: dict[str, LocalPty] = {}

    def start(self, command: str = "bash", *, cols: int = 80, rows: int = 24) -> LocalPty:
        pid, fd = pty.fork()
        if pid == 0:  # pragma: no cover - executed in the forked child
            os.chdir(self.cwd)
            os.execvpe("/bin/bash", ["bash", "-lc", command], os.environ.copy())
        os.set_blocking(fd, False)
        session = LocalPty(uuid.uuid4().hex, pid, fd, command)
        self.sessions[session.id] = session
        self._resize_fd(fd, cols, rows)
        self.loop.add_reader(fd, self._drain, session.id)
        return session

    @staticmethod
    def _resize_fd(fd: int, cols: int, rows: int) -> None:
        winsize = struct.pack("HHHH", max(1, rows), max(1, cols), 0, 0)
        with contextlib.suppress(OSError):
            fcntl.ioctl(fd, termios.TIOCSWINSZ, winsize)

    def _drain(self, session_id: str) -> None:
        session = self.sessions.get(session_id)
        if not session:
            return
        try:
            while True:
                chunk = os.read(session.fd, 65536)
                if not chunk:
                    break
                session.output.extend(chunk)
                session.total += len(chunk)
                if len(session.output) > self.MAX_BUFFER:
                    del session.output[: len(session.output) - self.MAX_BUFFER]
        except (BlockingIOError, OSError):
            pass
        self._refresh_status(session)

    @staticmethod
    def _refresh_status(session: LocalPty) -> None:
        with contextlib.suppress(ChildProcessError):
            pid, status = os.waitpid(session.pid, os.WNOHANG)
            if pid:
                session.status = "exited" if os.WIFEXITED(status) else "killed"

    def get(self, prefix: str) -> LocalPty:
        matches = [s for sid, s in self.sessions.items() if sid == prefix or sid.startswith(prefix)]
        if len(matches) != 1:
            raise KeyError("terminal id is missing or ambiguous")
        return matches[0]

    def write(self, prefix: str, data: str) -> LocalPty:
        session = self.get(prefix)
        if session.status != "running":
            raise RuntimeError("terminal is no longer running")
        os.write(session.fd, data.encode())
        return session

    def read(self, prefix: str, since: int = 0) -> tuple[LocalPty, str, int]:
        session = self.get(prefix)
        self._drain(prefix)
        # total is monotonic; retain only the newest bounded window.
        first = max(0, session.total - len(session.output))
        start = max(0, since - first)
        data = bytes(session.output[start:]).decode(errors="replace")
        return session, data, session.total

    def kill(self, prefix: str) -> LocalPty:
        session = self.get(prefix)
        if session.status == "running":
            with contextlib.suppress(ProcessLookupError):
                os.killpg(session.pid, signal.SIGTERM)
            session.status = "killed"
        return session

    def close(self) -> None:
        for session in list(self.sessions.values()):
            self.kill(session.id)
            self.loop.remove_reader(session.fd)
            with contextlib.suppress(OSError):
                os.close(session.fd)
