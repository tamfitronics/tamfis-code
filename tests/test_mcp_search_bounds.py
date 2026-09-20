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
    MAX_LIST_DIRECTORY_DEPTH,
    MAX_LIST_DIRECTORY_ENTRIES,
    MAX_SEARCH_RESULTS,
    SEARCH_PAGE_RESULTS,
    MCPServer,
    _page_search_matches,
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

    def test_depth_argument_lists_bounded_nested_children(self):
        with tempfile.TemporaryDirectory() as ws:
            root = Path(ws)
            nested = root / "src" / "pkg"
            nested.mkdir(parents=True)
            (nested / "module.py").write_text("x = 1")

            server = MCPServer()
            results = _run(server._list_directory(str(root), depth=3))
            paths = {item["path"] for item in results if "path" in item}
            self.assertIn(str(root / "src"), paths)
            self.assertIn(str(nested), paths)
            self.assertIn(str(nested / "module.py"), paths)

    def test_depth_is_rejected_above_the_hard_bound(self):
        with tempfile.TemporaryDirectory() as ws:
            result = _run(MCPServer()._list_directory(ws, depth=MAX_LIST_DIRECTORY_DEPTH + 1))
            self.assertIn("depth must be between", result[0]["error"])

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


    def test_read_file_accepts_line_start_and_line_end_aliases(self):
        with tempfile.TemporaryDirectory() as ws:
            root = Path(ws)
            target = root / "sample.py"
            target.write_text("one\ntwo\nthree\nfour\n")
            server = MCPServer(workspace_root=str(root))
            result = _run(server._read_file("sample.py", line_start=2, line_end=3))
            self.assertIn("Showing lines 2-3", result)
            self.assertIn("2: two", result)
            self.assertIn("3: three", result)
            self.assertNotIn("1: one", result)
            self.assertNotIn("4: four", result)


class SearchCodePagingTests(unittest.TestCase):
    """A broad query must not dump every match into one tool result, and must
    not throw away the rest either: it returns ONE page plus the offset that
    continues it (read_file's contract), so "the answer is on match 120" is
    reachable instead of only being told to narrow the query."""

    def test_one_page_plus_a_continuation_pointer(self):
        with tempfile.TemporaryDirectory() as ws:
            root = Path(ws)
            total = SEARCH_PAGE_RESULTS + 40
            for i in range(total):
                (root / f"module_{i:05d}.py").write_text("NEEDLE_TOKEN = 1\n")

            server = MCPServer()
            results = _run(server._search_code("NEEDLE_TOKEN", path=str(root)))

            real_matches = [item for item in results if "file" in item]
            self.assertEqual(len(real_matches), SEARCH_PAGE_RESULTS)
            self.assertLessEqual(len(results), SEARCH_PAGE_RESULTS + 1)
            pagination = results[-1]["pagination"]
            self.assertEqual(pagination["total"], total)
            self.assertEqual(pagination["next_offset"], SEARCH_PAGE_RESULTS + 1)
            self.assertIn("offset=", pagination["note"])

            # The continuation really continues: no repeats, and the rest of
            # the matches arrive with their own end-of-results marker.
            following = _run(server._search_code(
                "NEEDLE_TOKEN", path=str(root), offset=pagination["next_offset"],
            ))
            following_matches = [item for item in following if "file" in item]
            self.assertEqual(len(following_matches), total - SEARCH_PAGE_RESULTS)
            self.assertIsNone(following[-1]["pagination"]["next_offset"])
            first_page_files = {item["file"] for item in real_matches}
            self.assertFalse(first_page_files & {item["file"] for item in following_matches})

    def test_an_unpaged_small_result_is_returned_whole(self):
        """Paging must be invisible for the common case: no pagination entry,
        no offset needed, exactly what the old behaviour returned."""
        with tempfile.TemporaryDirectory() as ws:
            root = Path(ws)
            (root / "only.py").write_text("NEEDLE_TOKEN = 1\n")
            results = _run(MCPServer()._search_code("NEEDLE_TOKEN", path=str(root)))
            self.assertEqual(len(results), 1)
            self.assertIn("file", results[0])

    def test_offset_past_the_end_says_so_instead_of_returning_nothing(self):
        page = _page_search_matches([{"file": "a.py", "line": 1}], offset=9)
        self.assertEqual(len(page), 1)
        self.assertIn("past the end", page[0]["note"])

    def test_an_explicit_max_results_is_honoured_and_capped(self):
        matches = [{"file": f"f{i}.py", "line": 1} for i in range(30)]
        page = _page_search_matches(matches, max_results=5)
        self.assertEqual(len([m for m in page if "file" in m]), 5)
        self.assertEqual(page[-1]["pagination"]["next_offset"], 6)
        capped = _page_search_matches(matches, max_results=10_000)
        self.assertEqual(capped[-1]["pagination"]["total"], 30)
        self.assertLessEqual(len([m for m in capped if "file" in m]), MAX_SEARCH_RESULTS)


class SearchCodeBoundsTests(unittest.TestCase):
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
