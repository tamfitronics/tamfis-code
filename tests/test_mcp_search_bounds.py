"""Tool-execution-layer bounding/exclusion for MCPServer's search_code and
list_directory -- these must never depend on the caller (runner_local.py's
workspace scoping) to keep output small; a directly-invoked tool call
against a real, unscoped directory must still be bounded on its own.

Regression coverage for: thousands of search results (spec test #3),
exclusion of generated/dependency directories from search tools (spec test
#9), and a directory listing that would otherwise return an unbounded
"3858 item(s)"-style result.
"""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from tamfis_code.mcp import (
    EXCLUDED_DIR_NAMES,
    MAX_LIST_DIRECTORY_ENTRIES,
    MAX_SEARCH_RESULTS,
    MCPServer,
)


def _run(coro):
    return asyncio.run(coro)


class ListDirectoryBoundsTests(unittest.TestCase):
    def test_excludes_known_generated_and_dependency_directories(self):
        with tempfile.TemporaryDirectory() as ws:
            root = Path(ws)
            (root / "src").mkdir()
            (root / "node_modules").mkdir()
            (root / ".git").mkdir()
            (root / "dist").mkdir()
            (root / "__pycache__").mkdir()

            server = MCPServer()
            results = _run(server._list_directory(str(root)))

            names = {item["name"] for item in results if "name" in item}
            self.assertIn("src", names)
            self.assertNotIn("node_modules", names)
            self.assertNotIn(".git", names)
            self.assertNotIn("dist", names)
            self.assertNotIn("__pycache__", names)

            excluded_marker = next((item for item in results if item.get("excluded")), None)
            self.assertIsNotNone(excluded_marker)

    def test_caps_entry_count_with_truncation_marker(self):
        with tempfile.TemporaryDirectory() as ws:
            root = Path(ws)
            total = MAX_LIST_DIRECTORY_ENTRIES + 50
            for i in range(total):
                (root / f"file_{i:05d}.txt").write_text("x")

            server = MCPServer()
            results = _run(server._list_directory(str(root)))

            truncated_marker = next((item for item in results if item.get("truncated")), None)
            self.assertIsNotNone(truncated_marker)
            self.assertIn(str(total - MAX_LIST_DIRECTORY_ENTRIES), truncated_marker["note"])
            real_entries = [item for item in results if "name" in item]
            self.assertEqual(len(real_entries), MAX_LIST_DIRECTORY_ENTRIES)


class SearchCodeBoundsTests(unittest.TestCase):
    def test_caps_total_matches_with_truncation_marker(self):
        with tempfile.TemporaryDirectory() as ws:
            root = Path(ws)
            total = MAX_SEARCH_RESULTS + 40
            for i in range(total):
                (root / f"module_{i:05d}.py").write_text("NEEDLE_TOKEN = 1\n")

            server = MCPServer()
            results = _run(server._search_code("NEEDLE_TOKEN", path=str(root)))

            self.assertLessEqual(len(results), MAX_SEARCH_RESULTS + 1)
            truncated_marker = next((item for item in results if item.get("truncated")), None)
            self.assertIsNotNone(truncated_marker)
            real_matches = [item for item in results if "file" in item]
            self.assertEqual(len(real_matches), MAX_SEARCH_RESULTS)

    def test_excludes_matches_inside_generated_or_dependency_directories(self):
        with tempfile.TemporaryDirectory() as ws:
            root = Path(ws)
            (root / "src").mkdir()
            (root / "src" / "app.py").write_text("NEEDLE_TOKEN = 'real'\n")
            for excluded_name in ("node_modules", "dist", "__pycache__", ".git"):
                nested = root / excluded_name
                nested.mkdir()
                (nested / "noise.py").write_text("NEEDLE_TOKEN = 'noise'\n")

            server = MCPServer()
            results = _run(server._search_code("NEEDLE_TOKEN", path=str(root)))

            files_matched = {item["file"] for item in results if "file" in item}
            self.assertTrue(any("src/app.py" in f or f.endswith("app.py") for f in files_matched))
            for excluded_name in ("node_modules", "dist", "__pycache__", ".git"):
                self.assertFalse(
                    any(f"/{excluded_name}/" in f for f in files_matched),
                    f"search_code returned a match from excluded directory {excluded_name!r}",
                )

    def test_bounds_a_single_extremely_long_match_line(self):
        with tempfile.TemporaryDirectory() as ws:
            root = Path(ws)
            huge_line = "NEEDLE_TOKEN " + ("x" * 5000)
            (root / "minified.js").write_text(huge_line + "\n")

            server = MCPServer()
            results = _run(server._search_code("NEEDLE_TOKEN", path=str(root)))

            real_matches = [item for item in results if "content" in item]
            self.assertEqual(len(real_matches), 1)
            self.assertLess(len(real_matches[0]["content"]), 1000)
            self.assertIn("chars omitted", real_matches[0]["content"])

    def test_excluded_dir_names_cover_spec_required_set(self):
        required = {
            ".git", "node_modules", "dist", "build", "coverage", ".pytest_cache",
            "__pycache__", ".venv", "venv", "vendor", "target", "logs", "archives",
        }
        self.assertTrue(required.issubset(EXCLUDED_DIR_NAMES))


if __name__ == "__main__":
    unittest.main()
