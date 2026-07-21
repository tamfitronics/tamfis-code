"""Real cross-file reference resolution: MCPServer.find_references combines
CodeIndexer's symbol table (go-to-definition) with a whole-word codebase
search (find-all-usages) -- previously references.py's ReferenceResolver
was the only thing called "references" in this codebase, and it's an
unrelated feature (@file/@folder mention expansion in prompt text), not
find-usages/go-to-definition. No tool under any name let the model call
this mid-turn before this.
"""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tamfis_code.mcp import MCPServer
from tamfis_code.safety import READ_ONLY_TOOLS, classify_tool_call_risk
from tamfis_code.tool_policy import READ_TOOLS


def _run(coro):
    return asyncio.run(coro)


class FindReferencesTests(unittest.TestCase):
    """find_references's default CodeIndexer (no explicit index_path)
    persists to ~/.tamfis/index/<hash of root>/ -- patch Path.home() to a
    throwaway temp dir for the duration of each test so these never write
    into the real user's home directory (the same class of test-isolation
    bug fixed for state.json elsewhere this session, caught proactively
    here before it could repeat)."""

    def setUp(self):
        self._home_tmp = tempfile.TemporaryDirectory()
        self._home_patch = patch("pathlib.Path.home", return_value=Path(self._home_tmp.name))
        self._home_patch.start()

    def tearDown(self):
        self._home_patch.stop()
        self._home_tmp.cleanup()

    def test_finds_the_definition_and_every_usage_across_files(self):
        with tempfile.TemporaryDirectory() as ws:
            root = Path(ws)
            (root / "core.py").write_text("def widen_theme():\n    pass\n")
            (root / "caller_a.py").write_text("from core import widen_theme\nwiden_theme()\n")
            (root / "caller_b.py").write_text("import core\ncore.widen_theme()\n")

            server = MCPServer(workspace_root=str(root))
            result = _run(server._find_references("widen_theme", path=str(root)))

            self.assertTrue(result["success"])
            self.assertEqual(len(result["definitions"]), 1)
            self.assertEqual(result["definitions"][0]["kind"], "function")
            self.assertTrue(result["definitions"][0]["file"].endswith("core.py"))
            files_referenced = {Path(r["file"]).name for r in result["references"]}
            self.assertEqual(files_referenced, {"core.py", "caller_a.py", "caller_b.py"})
            self.assertGreaterEqual(result["reference_count"], 4)

    def test_whole_word_match_does_not_match_a_longer_identifier(self):
        with tempfile.TemporaryDirectory() as ws:
            root = Path(ws)
            (root / "a.py").write_text("def widen():\n    pass\ndef widen_more():\n    pass\n")

            server = MCPServer(workspace_root=str(root))
            result = _run(server._find_references("widen", path=str(root)))

            contents = [r["content"] for r in result["references"]]
            self.assertTrue(any("def widen():" in c for c in contents))
            self.assertFalse(any("widen_more" in c for c in contents))

    def test_symbol_with_no_definition_still_returns_references(self):
        # A local variable, or a symbol the regex-based indexer's parsers
        # don't recognise (e.g. a plain assignment) -- definitions can be
        # legitimately empty while references are still real and useful.
        with tempfile.TemporaryDirectory() as ws:
            root = Path(ws)
            (root / "a.py").write_text("SOME_CONSTANT = 1\nprint(SOME_CONSTANT)\n")

            server = MCPServer(workspace_root=str(root))
            result = _run(server._find_references("SOME_CONSTANT", path=str(root)))

            self.assertTrue(result["success"])
            self.assertEqual(result["definitions"], [])
            self.assertGreaterEqual(result["reference_count"], 2)

    def test_empty_symbol_is_a_clear_error_not_a_crash(self):
        server = MCPServer()
        result = _run(server._find_references("   "))
        self.assertFalse(result["success"])
        self.assertIn("error", result)

    def test_registered_as_a_read_only_tool_available_by_default(self):
        server = MCPServer()
        self.assertIn("find_references", server.tools)
        self.assertIn("find_references", READ_ONLY_TOOLS)
        self.assertIn("find_references", READ_TOOLS)
        self.assertEqual(
            classify_tool_call_risk("find_references", {"symbol": "x"}, workspace_root="/tmp"),
            "read_only",
        )

    def test_incremental_reindex_makes_a_second_call_fast_not_a_full_reparse(self):
        # Synergy with the indexer's incremental re-indexing fix: a second
        # find_references call in the same directory should not re-parse
        # files that have not changed.
        with tempfile.TemporaryDirectory() as ws:
            root = Path(ws)
            (root / "a.py").write_text("def widen():\n    pass\n")
            server = MCPServer(workspace_root=str(root))

            _run(server._find_references("widen", path=str(root)))
            from unittest.mock import patch
            from tamfis_code.indexer import CodeIndexer
            with patch.object(CodeIndexer, "_parse_python", wraps=CodeIndexer._parse_python) as spy:
                _run(server._find_references("widen", path=str(root)))
            spy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
