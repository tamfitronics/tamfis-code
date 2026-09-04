"""Persistent TamfisGPT Remote Workspace bridge for a local Tamfis-Code CLI."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import socket
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit

import websockets

from .api_client import AuthRequiredError
from .config import CONFIG_DIR
from .mcp import MCPServer

DEVICE_PATH = CONFIG_DIR / "device.json"
OUTBOX_PATH = CONFIG_DIR / "agent-outbox.json"
RECONNECT_DELAY_SECONDS = 1.0
MAX_RECONNECT_DELAY_SECONDS = 30.0


def load_or_create_device_identity() -> dict[str, str]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        data = json.loads(DEVICE_PATH.read_text(encoding="utf-8"))
        if data.get("device_id"):
            return data
    except (OSError, ValueError):
        pass
    data = {
        "device_id": uuid.uuid4().hex,
        "name": socket.gethostname() or "Tamfis-Code device",
    }
    DEVICE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.chmod(DEVICE_PATH, 0o600)
    return data


def workspace_identity(device_id: str, workspace_root: str) -> str:
    canonical = str(Path(workspace_root).expanduser().resolve())
    return hashlib.sha256(f"{device_id}\0{canonical}".encode()).hexdigest()


def _websocket_url(api_base: str, device_id: str) -> str:
    parts = urlsplit(api_base)
    scheme = "wss" if parts.scheme == "https" else "ws"
    base_path = parts.path.rstrip("/")
    return urlunsplit((
        scheme, parts.netloc,
        f"{base_path}/api/v1/remote/agent/ws/{device_id}", "", "",
    ))


class RemoteAgentBridge:
    """Reconnects forever while the CLI is active and executes server RPCs."""

    def __init__(self, client, workspace_root: str, session_id: Optional[int] = None):
        self.client = client
        self.workspace_root = str(Path(workspace_root).resolve())
        self.session_id = session_id
        self.device = load_or_create_device_identity()
        self.workspace_id = workspace_identity(
            self.device["device_id"], self.workspace_root,
        )
        self.server: Optional[dict[str, Any]] = None
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._started = asyncio.Event()
        self._start_error: Optional[Exception] = None
        self._connected = asyncio.Event()
        self._outbox: dict[str, dict[str, Any]] = self._load_outbox()
        self._running: set[str] = set()
        self._send_lock = asyncio.Lock()

    def _load_outbox(self) -> dict[str, dict[str, Any]]:
        try:
            data = json.loads(OUTBOX_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save_outbox(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        # An RPC result must survive a process or power interruption between
        # execution and its server acknowledgement. Replace atomically so a
        # partial JSON write cannot discard every unacknowledged result.
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=CONFIG_DIR, delete=False,
            prefix=f".{OUTBOX_PATH.name}.", suffix=".tmp",
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(self._outbox, handle, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, OUTBOX_PATH)
        os.chmod(OUTBOX_PATH, 0o600)

    async def start(self) -> dict[str, Any]:
        self._task = asyncio.create_task(self._run())
        # A bridge is explicitly a persistent connection. Waiting only 15
        # seconds converted a recoverable server/network outage into a failed
        # CLI command, which immediately closed the reconnecting task.
        try:
            await self._started.wait()
        except asyncio.CancelledError:
            # ensure_agent_bridge only stores the bridge after start() returns,
            # so it cannot clean up a bridge interrupted during initial
            # connection. Own that cancellation path here.
            await self.stop()
            raise
        if self._start_error is not None:
            raise self._start_error
        assert self.server is not None
        return self.server

    async def stop(self) -> None:
        self._stop.set()
        self._started.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *_exc):
        await self.stop()

    async def _run(self) -> None:
        delay = RECONNECT_DELAY_SECONDS
        while not self._stop.is_set():
            try:
                if self.server is None:
                    self.server = await self.client.register_agent_device(
                        name=self.device["name"],
                        device_id=self.device["device_id"],
                        os_family=platform.system().lower(),
                    )
                    self._started.set()
                # Refresh an expired access token through the normal
                # authenticated API path before opening a new socket.
                await self.client.me()
                credentials = self.client.credentials
                if credentials is None:
                    self._start_error = AuthRequiredError(
                        401, "Not authenticated -- run `tamfis-code login`",
                    )
                    self._started.set()
                    return
                url = _websocket_url(
                    self.client.config.api_base,
                    self.device["device_id"],
                )
                async with websockets.connect(
                    url, ping_interval=20, ping_timeout=20, max_size=8 * 1024 * 1024,
                    additional_headers={
                        "Authorization": f"Bearer {credentials.access_token}",
                    },
                ) as websocket:
                    self._connected.set()
                    delay = RECONNECT_DELAY_SECONDS
                    for message in list(self._outbox.values()):
                        await self._send(websocket, message)
                    await self._send_workspace_sync(websocket)
                    async for raw in websocket:
                        message = json.loads(raw)
                        if message.get("type") == "rpc_request":
                            request_id = str(message.get("id") or "")
                            if request_id in self._outbox:
                                await self._send(websocket, self._outbox[request_id])
                            elif request_id not in self._running:
                                self._running.add(request_id)
                                asyncio.create_task(self._handle_rpc(websocket, message))
                        elif message.get("type") == "rpc_ack":
                            self._outbox.pop(str(message.get("id") or ""), None)
                            self._save_outbox()
            except asyncio.CancelledError:
                raise
            except AuthRequiredError as exc:
                self._connected.clear()
                self._start_error = exc
                self._started.set()
                return
            except Exception:
                self._connected.clear()
                await asyncio.sleep(delay)
                delay = min(delay * 2, MAX_RECONNECT_DELAY_SECONDS)
            finally:
                # A clean server close is still a disconnection. Without this
                # clear, callers can mistake a dead socket for a live bridge.
                self._connected.clear()

    async def _handle_rpc(self, websocket, message: dict[str, Any]) -> None:
        request_id = str(message.get("id") or "")
        method = str(message.get("method") or "")
        params = message.get("params") or {}
        started = time.monotonic()
        try:
            if method == "execute":
                result = await self._execute(params)
            elif method == "write_text_file":
                result = await self._write_text_file(params)
            else:
                result = {"error": f"Unsupported local-agent method: {method}"}
        except Exception as exc:
            result = {"error": str(exc)}
        result.setdefault("duration_ms", (time.monotonic() - started) * 1000)
        envelope = {"type": "rpc_result", "id": request_id, "result": result}
        self._outbox[request_id] = envelope
        self._save_outbox()
        try:
            if result.get("stdout"):
                await self._send(websocket, {
                    "type": "rpc_output", "id": request_id,
                    "stream": "stdout", "data": result["stdout"],
                })
            if result.get("stderr"):
                await self._send(websocket, {
                    "type": "rpc_output", "id": request_id,
                    "stream": "stderr", "data": result["stderr"],
                })
            await self._send(websocket, envelope)
            await self._send_workspace_sync(websocket)
        except Exception:
            # Durable outbox is replayed after reconnect.
            return
        finally:
            self._running.discard(request_id)

    async def _send(self, websocket, message: dict[str, Any]) -> None:
        async with self._send_lock:
            await websocket.send(json.dumps(message))

    async def _execute(self, params: dict[str, Any]) -> dict[str, Any]:
        server = MCPServer(workspace_root=self.workspace_root)
        result = await server.call_tool("execute_command", {
            "command": str(params.get("command") or ""),
            "timeout": int(params.get("timeout_seconds") or 120),
        })
        payload = result.get("result") if isinstance(result.get("result"), dict) else result
        return {
            "stdout": str(payload.get("stdout") or ""),
            "stderr": str(payload.get("stderr") or payload.get("error") or ""),
            "exit_code": int(payload.get("return_code", 0 if result.get("success") else 1)),
            "truncated": False,
        }

    async def _write_text_file(self, params: dict[str, Any]) -> dict[str, Any]:
        server = MCPServer(workspace_root=self.workspace_root)
        result = await server.call_tool("write_file", {
            "path": str(params.get("path") or ""),
            "content": str(params.get("content") or ""),
        })
        output = str(result.get("result") or result.get("error") or "")
        if not result.get("success") or output.startswith("❌"):
            return {"error": output or "File write failed"}
        return {"stdout": str(params.get("path") or ""), "exit_code": 0}

    async def _send_workspace_sync(self, websocket) -> None:
        if self.session_id is None:
            return
        server = MCPServer(workspace_root=self.workspace_root)
        result = await server.call_tool("execute_command", {
            "command": "git rev-parse HEAD 2>/dev/null; git status --porcelain=v1 2>/dev/null",
            "timeout": 10,
        })
        payload = result.get("result") if isinstance(result.get("result"), dict) else result
        stdout = str(payload.get("stdout") or "")
        lines = stdout.splitlines()
        await self._send(websocket, {
            "type": "workspace_sync", "sync_id": uuid.uuid4().hex,
            "session_id": self.session_id, "workspace_id": self.workspace_id,
            "workspace_root": self.workspace_root,
            "head": lines[0] if lines else None,
            "dirty": bool(lines[1:]),
            "status": lines[1:200],
        })
