"""Repository searches must be grounded in a directory-tree inspection."""
from __future__ import annotations

import asyncio
from pathlib import Path

from tamfis_code.mcp import MCPServer


def _run(coro):
    return asyncio.run(coro)


def test_real_agent_search_is_blocked_until_target_tree_is_listed(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("NEEDLE = True\n")
    server = MCPServer(workspace_root=str(tmp_path), session_id=17)

    blocked = _run(server._search_code("NEEDLE", path=str(tmp_path)))
    assert blocked == [{
        "error": (
            "Repository orientation required before searching '.'. "
            "Call list_directory on '.' first (use a bounded depth such as 2), "
            "inspect the returned structure, then retry this search in the most relevant scope."
        )
    }]

    listing = _run(server._list_directory(str(tmp_path), depth=2))
    assert any(item.get("path", "").endswith("src/app.py") for item in listing)
    matches = _run(server._search_code("NEEDLE", path=str(tmp_path / "src")))
    assert any(item.get("file", "").endswith("src/app.py") for item in matches)


def test_low_level_server_without_session_keeps_direct_search_api(tmp_path: Path):
    (tmp_path / "app.py").write_text("DIRECT_NEEDLE = True\n")
    server = MCPServer(workspace_root=str(tmp_path))
    matches = _run(server._search_code("DIRECT_NEEDLE", path=str(tmp_path)))
    assert any(item.get("file", "").endswith("app.py") for item in matches)
