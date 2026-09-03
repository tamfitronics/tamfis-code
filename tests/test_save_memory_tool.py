"""Regression coverage for the agent-callable save_memory MCP tool --
Phase 1's "auto memory the agent can append during a session," distinct
from the pre-existing `tamfis-code memory save` CLI command (user-only).
"""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from tamfis_code.mcp import MCPServer
from tamfis_code.runtime import memory as memory_module
from tamfis_code.runtime.memory import MemoryStore
from tamfis_code.safety import RISK_MEDIUM, classify_tool_call_risk


def _run(coro):
    return asyncio.run(coro)


class SaveMemoryToolTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._tmp = Path(self._tmpdir.name)
        self._original_store = memory_module._DEFAULT_STORE
        memory_module._DEFAULT_STORE = MemoryStore(root=self._tmp / "memory")

    def tearDown(self):
        memory_module._DEFAULT_STORE = self._original_store
        self._tmpdir.cleanup()

    def test_tool_is_registered_with_a_handler(self):
        server = MCPServer()
        self.assertIn("save_memory", server.tools)
        self.assertTrue(callable(server.tools["save_memory"].handler))

    def test_saving_persists_a_retrievable_record(self):
        server = MCPServer()
        result = _run(server._save_memory(
            name="deploy-command",
            type="project",
            description="how to deploy this project",
            content="npm run build && ./deploy.sh",
        ))
        self.assertIn("✅", result)
        loaded = memory_module.get_memory_store().load("deploy-command")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.content, "npm run build && ./deploy.sh")

    def test_invalid_type_returns_an_error_without_raising(self):
        server = MCPServer()
        result = _run(server._save_memory(
            name="x", type="not-a-real-type", description="d", content="c",
        ))
        self.assertIn("❌", result)
        self.assertIn("must be one of", result.lower())

    def test_saving_twice_with_same_name_overwrites_not_duplicates(self):
        server = MCPServer()
        _run(server._save_memory(name="note", type="project", description="d", content="first"))
        _run(server._save_memory(name="note", type="project", description="d", content="second"))
        records = memory_module.get_memory_store().list()
        self.assertEqual(len([r for r in records if r.name == "note"]), 1)
        self.assertEqual(memory_module.get_memory_store().load("note").content, "second")

    def test_risk_classification_is_medium_not_bypassed(self):
        risk = classify_tool_call_risk(
            "save_memory",
            {"name": "x", "type": "project", "description": "d", "content": "c"},
            workspace_root=str(self._tmp),
        )
        self.assertEqual(risk, RISK_MEDIUM)


if __name__ == "__main__":
    unittest.main()
