from pathlib import Path
from unittest.mock import patch

import pytest

from tamfis_code.mcp import MCPServer
from tamfis_code.plugins import load_plugins


async def _hello(name: str):
    return {"message": f"hello {name}"}


class _Dist:
    version = "2.0"


class _Entry:
    name = "sample"
    value = "sample:factory"
    dist = _Dist()

    def load(self):
        return lambda: {
            "name": "sample-plugin",
            "tools": [{
                "name": "plugin_hello", "description": "Say hello",
                "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
                "handler": _hello,
            }],
            "skill_roots": ["/tmp/sample-skills"],
        }


def test_loads_entrypoint_plugin_metadata():
    with patch("tamfis_code.plugins._entry_points", return_value=[_Entry()]):
        plugins = load_plugins()
    assert plugins[0].name == "sample-plugin"
    assert plugins[0].version == "2.0"


@pytest.mark.asyncio
async def test_plugin_tool_is_registered_and_callable(tmp_path: Path):
    with patch("tamfis_code.plugins._entry_points", return_value=[_Entry()]):
        server = MCPServer(workspace_root=str(tmp_path))
    result = await server.call_tool("plugin_hello", {"name": "Tamfis"})
    assert result["success"] is True
    assert result["result"]["message"] == "hello Tamfis"
