"""Portable MCP client owned by Tamfis Code (no TamfisGPT monorepo).

Supports two transports:
  - stdio: a locally-spawned server process (Claude-Code-compatible
    .mcp.json ``{"command": ..., "args": [...]}`` entries).
  - Streamable HTTP: a remote server reachable over HTTP, per the current
    MCP spec transport (supersedes the deprecated HTTP+SSE transport --
    see modelcontextprotocol.io's transport docs). Claude-Code-compatible
    ``{"type": "http", "url": ..., "headers": {...}}`` entries.

Both are exposed through the same StandaloneMCPBridge surface
(initialize/list_tools/call_tool/shutdown) so callers never need to know
which transport a given server uses.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from . import __version__
from .config import CONFIG_DIR

_HTTP_REQUEST_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    # Exactly one of (command, url) is set -- stdio servers spawn a local
    # process, http servers speak Streamable HTTP to a remote endpoint.
    command: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] | None = None
    cwd: str | None = None
    url: str | None = None
    headers: dict[str, str] | None = None

    @property
    def transport(self) -> str:
        return "http" if self.url else "stdio"


def _read_servers(path: Path) -> dict[str, MCPServerConfig]:
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    items = raw.get("mcpServers", raw.get("servers", {})) if isinstance(raw, dict) else {}
    result: dict[str, MCPServerConfig] = {}
    for name, spec in items.items() if isinstance(items, dict) else ():
        if not isinstance(spec, dict) or spec.get("enabled", True) is False:
            continue
        command = spec.get("command")
        url = spec.get("url")
        # A server needs a way to be reached: either a local command to
        # spawn (stdio) or a remote url (Streamable HTTP). Neither present
        # means this entry is unusable -- skip it rather than error, same
        # tolerance the old command-only check already had for malformed
        # entries elsewhere in a shared .mcp.json.
        if not command and not url:
            continue
        env = spec.get("env")
        headers = spec.get("headers")
        result[str(name)] = MCPServerConfig(
            name=str(name),
            command=str(command) if command else None,
            args=tuple(str(arg) for arg in (spec.get("args") or [])),
            env={str(k): str(v) for k, v in env.items()} if isinstance(env, dict) else None,
            cwd=str(spec.get("cwd")) if spec.get("cwd") else None,
            url=str(url) if url else None,
            headers={str(k): str(v) for k, v in headers.items()} if isinstance(headers, dict) else None,
        )
    return result


def load_mcp_servers(workspace_root: str | Path | None = None) -> dict[str, MCPServerConfig]:
    """Layer personal, Claude-compatible, and Tamfis project MCP configs."""
    root = Path(workspace_root or Path.cwd()).expanduser().resolve()
    servers: dict[str, MCPServerConfig] = {}
    for path in (CONFIG_DIR / "mcp.json", root / ".mcp.json", root / ".tamfis" / "mcp.json"):
        servers.update(_read_servers(path))
    return servers


def _public_name(server: str, tool: str) -> str:
    safe_server = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in server)
    safe_tool = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in tool)
    return f"mcp__{safe_server}__{safe_tool}"


@dataclass
class _HTTPConnection:
    """State for one Streamable HTTP MCP server: a shared client plus
    whatever session id the server assigned during initialize (if any --
    session ids are optional per spec, some servers are stateless)."""
    client: httpx.AsyncClient
    url: str
    session_id: str | None = None


def _parse_streamable_http_body(body: str, content_type: str) -> list[dict[str, Any]]:
    """Decode JSON or SSE-framed JSON-RPC messages from a response body.

    A Streamable HTTP response is either a plain JSON object/array, or an
    SSE event stream carrying one or more `data: <json>` lines -- the
    transport lets a server stream several messages (e.g. progress
    notifications) before the final response to one request.
    """
    if not body.strip():
        return []
    if "text/event-stream" not in content_type.lower():
        value = json.loads(body)
        return value if isinstance(value, list) else [value]

    messages: list[dict[str, Any]] = []
    data_lines: list[str] = []
    for line in body.splitlines() + [""]:
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
        elif not line.strip() and data_lines:
            value = json.loads("\n".join(data_lines))
            if isinstance(value, list):
                messages.extend(item for item in value if isinstance(item, dict))
            elif isinstance(value, dict):
                messages.append(value)
            data_lines = []
    return messages


class StandaloneMCPBridge:
    def __init__(self, workspace_root: str | Path | None = None):
        self.workspace_root = str(Path(workspace_root or Path.cwd()).resolve())
        self.servers = load_mcp_servers(self.workspace_root)
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._http: dict[str, _HTTPConnection] = {}
        self._request_id = 0
        self._tool_map: dict[str, tuple[str, str]] = {}
        self._tools: list[dict[str, Any]] = []

    @property
    def available(self) -> bool:
        return bool(self._processes or self._http)

    # ── stdio transport ──────────────────────────────────────────────
    async def _request_stdio(self, process: asyncio.subprocess.Process, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._request_id += 1
        request_id = self._request_id
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write((json.dumps({
            "jsonrpc": "2.0", "id": request_id, "method": method, "params": params,
        }) + "\n").encode())
        await process.stdin.drain()
        while True:
            raw = await asyncio.wait_for(process.stdout.readline(), timeout=15)
            if not raw:
                stderr = ""
                if process.stderr is not None:
                    with contextlib.suppress(asyncio.TimeoutError):
                        stderr = (await asyncio.wait_for(process.stderr.read(), timeout=.2)).decode(errors="replace")
                raise RuntimeError(f"MCP server exited during {method}: {stderr.strip()}")
            message = json.loads(raw)
            if message.get("id") != request_id:
                continue
            if "error" in message:
                raise RuntimeError(str(message["error"]))
            return message.get("result") or {}

    async def _notify_stdio(self, process: asyncio.subprocess.Process, method: str, params: dict[str, Any] | None = None) -> None:
        assert process.stdin is not None
        process.stdin.write((json.dumps({
            "jsonrpc": "2.0", "method": method, **({"params": params} if params is not None else {}),
        }) + "\n").encode())
        await process.stdin.drain()

    # ── Streamable HTTP transport ────────────────────────────────────
    def _http_request_headers(self, conn: _HTTPConnection) -> dict[str, str]:
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        if conn.session_id:
            headers["Mcp-Session-Id"] = conn.session_id
        return headers

    async def _request_http(self, name: str, method: str, params: dict[str, Any]) -> dict[str, Any]:
        conn = self._http[name]
        self._request_id += 1
        request_id = self._request_id
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        response = await conn.client.post(
            conn.url, json=payload, headers=self._http_request_headers(conn),
            timeout=_HTTP_REQUEST_TIMEOUT_SECONDS,
        )
        # Per spec, a server MAY assign a session id on its response to
        # `initialize` -- every later request/notification on this
        # connection must then carry it back.
        session_id = response.headers.get("Mcp-Session-Id")
        if session_id:
            conn.session_id = session_id
        if response.status_code >= 400:
            raise RuntimeError(f"MCP server '{name}' returned HTTP {response.status_code} during {method}")
        messages = _parse_streamable_http_body(response.text, response.headers.get("content-type", ""))
        for message in messages:
            if message.get("id") != request_id:
                continue
            if "error" in message:
                raise RuntimeError(str(message["error"]))
            return message.get("result") or {}
        raise RuntimeError(f"MCP server '{name}' sent no response to {method}")

    async def _notify_http(self, name: str, method: str, params: dict[str, Any] | None = None) -> None:
        conn = self._http[name]
        payload = {"jsonrpc": "2.0", "method": method, **({"params": params} if params is not None else {})}
        # A notification carries no id and expects no JSON-RPC response --
        # per the Streamable HTTP spec, the server replies 202 Accepted
        # with an empty body. The response is still awaited (not
        # fire-and-forget) so a transport-level failure surfaces here
        # rather than silently vanishing.
        await conn.client.post(
            conn.url, json=payload, headers=self._http_request_headers(conn),
            timeout=_HTTP_REQUEST_TIMEOUT_SECONDS,
        )

    # ── transport-agnostic surface ───────────────────────────────────
    async def _request(self, name: str, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if name in self._processes:
            return await self._request_stdio(self._processes[name], method, params)
        return await self._request_http(name, method, params)

    async def _notify(self, name: str, method: str, params: dict[str, Any] | None = None) -> None:
        if name in self._processes:
            await self._notify_stdio(self._processes[name], method, params)
        else:
            await self._notify_http(name, method, params)

    async def _register_tools(self, name: str) -> None:
        listed = await self._request(name, "tools/list", {})
        for tool in listed.get("tools", []):
            public = _public_name(name, str(tool.get("name") or "tool"))
            self._tool_map[public] = (name, str(tool["name"]))
            self._tools.append({
                "name": public,
                "description": f"[{name} MCP] {tool.get('description') or tool.get('name')}",
                "parameters": tool.get("inputSchema") or {"type": "object", "properties": {}},
                "server": name,
            })

    async def initialize(self, background: bool = False) -> bool:
        del background
        if self._processes or self._http:
            return self.available
        for name, config in self.servers.items():
            if config.transport == "http":
                await self._initialize_http_server(name, config)
            else:
                await self._initialize_stdio_server(name, config)
        return self.available

    async def _initialize_stdio_server(self, name: str, config: MCPServerConfig) -> None:
        process: asyncio.subprocess.Process | None = None
        try:
            environment = os.environ.copy()
            if config.env:
                environment.update(config.env)
            process = await asyncio.create_subprocess_exec(
                config.command, *config.args, cwd=config.cwd or self.workspace_root,
                env=environment, stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            await self._request_stdio(process, "initialize", {
                "protocolVersion": "2025-06-18", "capabilities": {},
                "clientInfo": {"name": "tamfis-code", "version": __version__},
            })
            await self._notify_stdio(process, "notifications/initialized")
            self._processes[name] = process
            await self._register_tools(name)
        except Exception:
            if process is not None and process.returncode is None:
                process.terminate()

    async def _initialize_http_server(self, name: str, config: MCPServerConfig) -> None:
        assert config.url is not None
        client = httpx.AsyncClient(headers=config.headers or {})
        conn = _HTTPConnection(client=client, url=config.url)
        self._http[name] = conn
        try:
            await self._request_http(name, "initialize", {
                "protocolVersion": "2025-06-18", "capabilities": {},
                "clientInfo": {"name": "tamfis-code", "version": __version__},
            })
            await self._notify_http(name, "notifications/initialized")
            await self._register_tools(name)
        except Exception:
            del self._http[name]
            await client.aclose()

    async def list_tools(self) -> list[dict[str, Any]]:
        return list(self._tools)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in self._tool_map:
            return {"success": False, "is_error": True, "error": f"Unknown MCP tool: {name}"}
        server, original = self._tool_map[name]
        result = await self._request(server, "tools/call", {
            "name": original, "arguments": arguments,
        })
        is_error = bool(result.get("isError"))
        return {"success": not is_error, "is_error": is_error, "content": result.get("content") or []}

    async def shutdown(self) -> None:
        for process in self._processes.values():
            if process.stdin is not None:
                process.stdin.close()
            if process.returncode is None:
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=3)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        self._processes.clear()
        for conn in self._http.values():
            await conn.client.aclose()
        self._http.clear()
