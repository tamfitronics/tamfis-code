import json
import sys
import asyncio
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from tamfis_code.mcp import MCPServer
from tamfis_code.mcp_client import StandaloneMCPBridge, load_mcp_servers


def _configure(root: Path):
    config = root / ".tamfis" / "mcp.json"
    config.parent.mkdir(parents=True)
    script = Path(__file__).parent / "fixtures" / "mcp_echo_server.py"
    config.write_text(json.dumps({
        "mcpServers": {
            "demo-server": {"command": sys.executable, "args": [str(script)]},
        },
    }))


def test_loads_project_owned_mcp_configuration(tmp_path: Path):
    _configure(tmp_path)
    servers = load_mcp_servers(tmp_path)
    assert servers["demo-server"].command == sys.executable


@pytest.mark.asyncio
async def test_discovers_and_calls_external_mcp_tool_without_monorepo(tmp_path: Path):
    _configure(tmp_path)
    server = MCPServer(workspace_root=str(tmp_path))
    schemas = await asyncio.wait_for(server.external_tool_schemas_openai(), timeout=10)
    assert [item["function"]["name"] for item in schemas] == ["mcp__demo_server__echo"]
    result = await asyncio.wait_for(server.call_tool("mcp__demo_server__echo", {"message": "portable"}), timeout=10)
    assert result["success"] is True
    assert result["result"]["content"][0]["text"] == "portable"
    await asyncio.wait_for(server.shutdown(), timeout=10)


def _configure_http(root: Path, url: str = "https://mcp.example.com/mcp", headers: dict | None = None):
    config = root / ".tamfis" / "mcp.json"
    config.parent.mkdir(parents=True)
    spec = {"type": "http", "url": url}
    if headers:
        spec["headers"] = headers
    config.write_text(json.dumps({"mcpServers": {"remote-server": spec}}))


def test_loads_remote_http_server_configuration(tmp_path: Path):
    _configure_http(tmp_path, headers={"Authorization": "Bearer secret"})
    servers = load_mcp_servers(tmp_path)
    server = servers["remote-server"]
    assert server.transport == "http"
    assert server.url == "https://mcp.example.com/mcp"
    assert server.headers == {"Authorization": "Bearer secret"}
    assert server.command is None


class _FakeRemoteMCPServer:
    """In-process Streamable HTTP MCP server: real JSON-RPC framing over a
    real httpx transport, no actual socket -- see httpx.MockTransport."""

    def __init__(self):
        self.session_id = "sess-abc123"
        self.seen_session_ids: list[str | None] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.seen_session_ids.append(request.headers.get("mcp-session-id"))
        body = json.loads(request.content)
        method = body.get("method")
        if method == "initialize":
            result = {
                "protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                "serverInfo": {"name": "remote-echo", "version": "1"},
            }
            return httpx.Response(
                200, json={"jsonrpc": "2.0", "id": body["id"], "result": result},
                headers={"Mcp-Session-Id": self.session_id},
            )
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "tools/list":
            result = {"tools": [{
                "name": "remote_echo", "description": "Echo over HTTP",
                "inputSchema": {"type": "object", "properties": {"message": {"type": "string"}}},
            }]}
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})
        if method == "tools/call":
            message = body["params"]["arguments"]["message"]
            result = {"content": [{"type": "text", "text": message}], "isError": False}
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body.get("id"), "result": {}})


@pytest.mark.asyncio
async def test_discovers_and_calls_remote_http_mcp_tool(tmp_path: Path):
    _configure_http(tmp_path)
    fake_server = _FakeRemoteMCPServer()
    transport = httpx.MockTransport(fake_server.handler)

    real_async_client = httpx.AsyncClient

    def _client(headers=None, **_kwargs):
        return real_async_client(headers=headers, transport=transport)

    bridge = StandaloneMCPBridge(workspace_root=str(tmp_path))
    with patch("tamfis_code.mcp_client.httpx.AsyncClient", side_effect=_client):
        await asyncio.wait_for(bridge.initialize(), timeout=10)
        tools = await bridge.list_tools()
        assert [t["name"] for t in tools] == ["mcp__remote_server__remote_echo"]

        result = await asyncio.wait_for(
            bridge.call_tool("mcp__remote_server__remote_echo", {"message": "hi"}), timeout=10,
        )
        assert result["success"] is True
        assert result["content"][0]["text"] == "hi"

        await bridge.shutdown()

    # The session id the server assigned on `initialize` must be echoed
    # back on every later request on this connection (tools/list,
    # notifications/initialized, tools/call) -- not just remembered.
    assert fake_server.seen_session_ids[0] is None  # initialize itself carries none yet
    assert all(sid == fake_server.session_id for sid in fake_server.seen_session_ids[1:])


@pytest.mark.asyncio
async def test_mixed_stdio_and_http_servers_both_become_available(tmp_path: Path):
    # A user's .mcp.json can legitimately mix a local stdio server with a
    # remote http one -- both must initialize and both sets of tools must
    # be reachable through the same bridge.
    config = tmp_path / ".tamfis" / "mcp.json"
    config.parent.mkdir(parents=True)
    script = Path(__file__).parent / "fixtures" / "mcp_echo_server.py"
    config.write_text(json.dumps({
        "mcpServers": {
            "local-server": {"command": sys.executable, "args": [str(script)]},
            "remote-server": {"type": "http", "url": "https://mcp.example.com/mcp"},
        },
    }))
    fake_server = _FakeRemoteMCPServer()
    transport = httpx.MockTransport(fake_server.handler)

    real_async_client = httpx.AsyncClient

    def _client(headers=None, **_kwargs):
        return real_async_client(headers=headers, transport=transport)

    bridge = StandaloneMCPBridge(workspace_root=str(tmp_path))
    with patch("tamfis_code.mcp_client.httpx.AsyncClient", side_effect=_client):
        await asyncio.wait_for(bridge.initialize(), timeout=10)
        tool_names = {t["name"] for t in await bridge.list_tools()}
        assert tool_names == {"mcp__local_server__echo", "mcp__remote_server__remote_echo"}
        await bridge.shutdown()
