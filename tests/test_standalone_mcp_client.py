import json
import sys
import asyncio
from pathlib import Path

import pytest

from tamfis_code.mcp import MCPServer
from tamfis_code.mcp_client import load_mcp_servers


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
