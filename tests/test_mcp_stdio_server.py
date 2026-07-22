"""tamfis-code as an MCP *server* (outbound direction) -- before this,
tamfis-code could only consume external MCP tools (mcp.py's MCPServer is
a client-side facade for that), never be driven by another agent/IDE as
an MCP server itself. Covers the read-only tool-exposure boundary and
real delegation to MCPServer.call_tool, without spinning up a real stdio
transport (build_server/list_exposed_tools/call_exposed_tool are all
directly testable without one).
"""
from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from tamfis_code.mcp import MCPServer
from tamfis_code.mcp_stdio_server import (
    DEFAULT_EXPOSED_TOOLS,
    build_server,
    call_exposed_tool,
    list_exposed_tools,
)


def _run(coro):
    return asyncio.run(coro)


class ListExposedToolsTests(unittest.TestCase):
    def test_only_the_default_read_only_subset_is_listed(self):
        with tempfile.TemporaryDirectory() as ws:
            tool_server = MCPServer(workspace_root=ws)
            tools = _run(list_exposed_tools(tool_server, DEFAULT_EXPOSED_TOOLS))
            names = {t.name for t in tools}
            self.assertEqual(names, set(DEFAULT_EXPOSED_TOOLS))
            self.assertNotIn("write_file", names)
            self.assertNotIn("edit_file", names)
            self.assertNotIn("execute_command", names)

    def test_each_listed_tool_carries_a_real_json_schema(self):
        with tempfile.TemporaryDirectory() as ws:
            tool_server = MCPServer(workspace_root=ws)
            tools = _run(list_exposed_tools(tool_server, DEFAULT_EXPOSED_TOOLS))
            by_name = {t.name: t for t in tools}
            self.assertEqual(by_name["read_file"].inputSchema["required"], ["path"])


class CallExposedToolTests(unittest.TestCase):
    def test_read_file_delegates_to_the_real_workspace_scoped_tool(self):
        with tempfile.TemporaryDirectory() as ws:
            target = Path(ws) / "hello.py"
            target.write_text("print('hi')\n")
            tool_server = MCPServer(workspace_root=ws)

            content = _run(call_exposed_tool(tool_server, DEFAULT_EXPOSED_TOOLS, "read_file", {"path": str(target)}))

            payload = json.loads(content[0].text)
            self.assertTrue(payload.get("success"))
            self.assertIn("print('hi')", payload.get("result", ""))

    def test_write_file_is_refused_even_if_requested(self):
        # The exposure boundary is enforced at call time too, not just in
        # the tool listing -- a client that ignores list_tools and calls a
        # non-exposed tool name directly must still be refused.
        with tempfile.TemporaryDirectory() as ws:
            tool_server = MCPServer(workspace_root=ws)
            content = _run(call_exposed_tool(
                tool_server, DEFAULT_EXPOSED_TOOLS, "write_file",
                {"path": str(Path(ws) / "x.py"), "content": "malicious = True\n"},
            ))
            payload = json.loads(content[0].text)
            self.assertFalse(payload.get("success"))
            self.assertIn("not exposed", payload.get("error", ""))
            self.assertFalse((Path(ws) / "x.py").exists())

    def test_execute_command_is_refused(self):
        with tempfile.TemporaryDirectory() as ws:
            tool_server = MCPServer(workspace_root=ws)
            content = _run(call_exposed_tool(
                tool_server, DEFAULT_EXPOSED_TOOLS, "execute_command", {"command": "echo hi"},
            ))
            payload = json.loads(content[0].text)
            self.assertFalse(payload.get("success"))

    def test_find_references_works_over_the_mcp_boundary(self):
        with tempfile.TemporaryDirectory() as ws:
            (Path(ws) / "core.py").write_text("def widen():\n    pass\n")
            tool_server = MCPServer(workspace_root=ws)
            content = _run(call_exposed_tool(tool_server, DEFAULT_EXPOSED_TOOLS, "find_references", {"symbol": "widen"}))
            payload = json.loads(content[0].text)
            self.assertTrue(payload.get("success"))  # transport-level success
            self.assertTrue(payload["result"]["success"])  # find_references' own result
            self.assertEqual(len(payload["result"]["definitions"]), 1)


class BuildServerTests(unittest.TestCase):
    def test_builds_a_real_mcp_server_scoped_to_the_workspace(self):
        with tempfile.TemporaryDirectory() as ws:
            from mcp.server.lowlevel import Server
            server = build_server(ws)
            self.assertIsInstance(server, Server)
            self.assertEqual(server.name, "tamfis-code")

    def test_custom_expose_tools_overrides_the_default_subset(self):
        with tempfile.TemporaryDirectory() as ws:
            tool_server = MCPServer(workspace_root=ws)
            tools = _run(list_exposed_tools(tool_server, frozenset({"read_file"})))
            self.assertEqual({t.name for t in tools}, {"read_file"})


if __name__ == "__main__":
    unittest.main()
