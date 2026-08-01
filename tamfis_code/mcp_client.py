"""Portable MCP client owned by Tamfis Code (no TamfisGPT monorepo)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import __version__
from .config import CONFIG_DIR


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    command: str
    args: tuple[str, ...] = ()
    env: dict[str, str] | None = None
    cwd: str | None = None


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
        if not isinstance(spec, dict) or spec.get("enabled", True) is False or not spec.get("command"):
            continue
        env = spec.get("env")
        result[str(name)] = MCPServerConfig(
            name=str(name), command=str(spec["command"]),
            args=tuple(str(arg) for arg in (spec.get("args") or [])),
            env={str(k): str(v) for k, v in env.items()} if isinstance(env, dict) else None,
            cwd=str(spec.get("cwd")) if spec.get("cwd") else None,
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


class StandaloneMCPBridge:
    def __init__(self, workspace_root: str | Path | None = None):
        self.workspace_root = str(Path(workspace_root or Path.cwd()).resolve())
        self.servers = load_mcp_servers(self.workspace_root)
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._request_id = 0
        self._tool_map: dict[str, tuple[str, str]] = {}
        self._tools: list[dict[str, Any]] = []

    @property
    def available(self) -> bool:
        return bool(self._processes)

    async def _request(self, process: asyncio.subprocess.Process, method: str, params: dict[str, Any]) -> dict[str, Any]:
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

    async def _notify(self, process: asyncio.subprocess.Process, method: str, params: dict[str, Any] | None = None) -> None:
        assert process.stdin is not None
        process.stdin.write((json.dumps({
            "jsonrpc": "2.0", "method": method, **({"params": params} if params is not None else {}),
        }) + "\n").encode())
        await process.stdin.drain()

    async def initialize(self, background: bool = False) -> bool:
        del background
        if self._processes:
            return self.available
        for name, config in self.servers.items():
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
                await self._request(process, "initialize", {
                    "protocolVersion": "2025-06-18", "capabilities": {},
                    "clientInfo": {"name": "tamfis-code", "version": __version__},
                })
                await self._notify(process, "notifications/initialized")
                self._processes[name] = process
                listed = await self._request(process, "tools/list", {})
                for tool in listed.get("tools", []):
                    public = _public_name(name, str(tool.get("name") or "tool"))
                    self._tool_map[public] = (name, str(tool["name"]))
                    self._tools.append({
                        "name": public,
                        "description": f"[{name} MCP] {tool.get('description') or tool.get('name')}",
                        "parameters": tool.get("inputSchema") or {"type": "object", "properties": {}},
                        "server": name,
                    })
            except Exception:
                if process is not None and process.returncode is None:
                    process.terminate()
                continue
        return self.available

    async def list_tools(self) -> list[dict[str, Any]]:
        return list(self._tools)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in self._tool_map:
            return {"success": False, "is_error": True, "error": f"Unknown MCP tool: {name}"}
        server, original = self._tool_map[name]
        result = await self._request(self._processes[server], "tools/call", {
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
