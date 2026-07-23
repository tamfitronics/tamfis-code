"""Exposes tamfis-code's own tools as a real MCP server over stdio, so
external agents/IDEs (Claude Code, Cursor, any MCP-speaking client) can
drive tamfis-code the same way tamfis-code itself already consumes
external MCP servers (mcp.py's MCPServer is a client-side facade for
that direction). Before this, that direction didn't exist under any name
-- no stdio/JSON-RPC listener, no `mcp` SDK dependency.

Only a fixed, read-only tool subset is exposed (see DEFAULT_EXPOSED_TOOLS)
-- an external MCP client has no equivalent of runner_local.py's
approval-gated agent loop, so anything that could mutate the workspace or
run arbitrary commands would bypass every local safety mechanism this
project otherwise enforces (safety.py's risk classification/approval
gating, the mutation ledger). This is deliberately conservative rather
than configurable-to-unsafe by default.
"""
from __future__ import annotations

import json
from typing import Any, Optional

import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from .mcp import MCPServer

DEFAULT_EXPOSED_TOOLS = frozenset({
    "read_file", "list_directory", "search_code", "find_references", "get_git_info",
})


async def list_exposed_tools(tool_server: MCPServer, exposed: frozenset[str]) -> list[types.Tool]:
    return [
        types.Tool(name=name, description=definition.description, inputSchema=definition.parameters)
        for name, definition in tool_server.tools.items()
        if name in exposed
    ]


async def call_exposed_tool(
    tool_server: MCPServer, exposed: frozenset[str], name: str, arguments: dict[str, Any],
) -> list[types.TextContent]:
    if name not in exposed:
        payload = {"error": f"Tool '{name}' is not exposed over MCP (read-only tools only).", "success": False}
        return [types.TextContent(type="text", text=json.dumps(payload))]
    result = await tool_server.call_tool(name, arguments)
    return [types.TextContent(type="text", text=json.dumps(result, default=str))]


def build_server(workspace_root: str, *, expose_tools: Optional[frozenset[str]] = None) -> Server:
    """Build (but do not run) the MCP Server wrapping a real, workspace-
    scoped MCPServer instance. Split out from run_stdio_server so
    list_exposed_tools/call_exposed_tool can be tested directly without a
    real stdio transport."""
    exposed = expose_tools if expose_tools is not None else DEFAULT_EXPOSED_TOOLS
    tool_server = MCPServer(workspace_root=workspace_root)
    server: Server = Server("tamfis-code")

    @server.list_tools()
    async def _list_tools() -> list[types.Tool]:
        return await list_exposed_tools(tool_server, exposed)

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
        return await call_exposed_tool(tool_server, exposed, name, arguments)

    return server


async def run_stdio_server(workspace_root: str, *, expose_tools: Optional[frozenset[str]] = None) -> None:
    """Entry point for `tamfis-code mcp-server` -- serves over stdio until
    the client disconnects (stdin closes)."""
    server = build_server(workspace_root, expose_tools=expose_tools)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
