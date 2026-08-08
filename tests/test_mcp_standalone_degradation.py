"""Regression tests for tamfis-code's standalone independence from tamgpt6.

_get_shared_mcp_bridge()/get_browser_tool_class() used to hard-require a
co-located monorepo checkout (raising ModuleNotFoundError otherwise), which
contradicted tamfis-code being an independently installable package. They now
return None when the monorepo isn't present, and every call site must turn
that into a clear "unavailable outside a monorepo checkout" error instead of
crashing with an unrelated AttributeError/TypeError.
"""
import unittest
from unittest.mock import patch

from tamfis_code.mcp import MCPServer, _import_monorepo_attr


class ImportMonorepoAttrTests(unittest.TestCase):
    def test_returns_none_for_a_module_that_does_not_exist_anywhere(self):
        self.assertIsNone(_import_monorepo_attr("definitely_not_a_real_package.submodule", "thing"))


class StandaloneDegradationTests(unittest.IsolatedAsyncioTestCase):
    @patch("tamfis_code.mcp._get_shared_mcp_bridge", return_value=None)
    async def test_call_tool_reports_unavailable_bridge_clearly(self, _mock_bridge):
        server = MCPServer()
        result = await server.call_tool("some_shared_tool", {})
        self.assertFalse(result["success"])
        self.assertIn("External MCP tool unavailable", result["error"])

    @patch("tamfis_code.mcp._get_shared_mcp_bridge", return_value=None)
    async def test_list_tools_async_reports_unavailable_bridge_clearly(self, _mock_bridge):
        server = MCPServer()
        tools = await server.list_tools_async()
        shared_entry = next(t for t in tools if t["name"] == "shared_mcp")
        self.assertFalse(shared_entry["available"])
        self.assertIn("External MCP registry unavailable", shared_entry["description"])

    @patch("tamfis_code.mcp.get_browser_tool_class", return_value=None)
    async def test_browser_facade_reports_unavailable_class_clearly(self, _mock_cls):
        server = MCPServer()
        result = await server.call_tool("browser", {"url": "https://example.com", "action": "navigate"})
        self.assertFalse(result["success"])
        self.assertIn("Portable browser support is unavailable", result["error"])

    def test_portable_browser_is_owned_by_tamfis_code(self):
        from tamfis_code.browser import PortableBrowserTool
        from tamfis_code.mcp import get_browser_tool_class

        self.assertIs(get_browser_tool_class(), PortableBrowserTool)


if __name__ == "__main__":
    unittest.main()
