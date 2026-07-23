"""Workspace-root (cwd) boundary enforcement across every read-only tool.

Regression coverage for a real gap found while auditing tool confinement:
_read_file/_write_file/_edit_file/_execute_command all routed through
MCPServer._resolve_in_workspace, but _list_directory/_search_code/
_get_git_info did not -- a model could list, grep, or read git info
anywhere on disk regardless of workspace_root. Also covers local_tools.py's
LocalReadOnlyTools, which used to construct MCPServer() with no
workspace_root at all, leaving offline/local chat mode's read-only tools
completely unconfined despite the module's "read-only boundary" docstring.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from tamfis_code.local_tools import LocalReadOnlyTools
from tamfis_code.mcp import MCPServer


def _run(coro):
    import asyncio
    return asyncio.run(coro)


class TestListDirectoryConfinement:
    def setup_method(self):
        self.workspace = tempfile.mkdtemp()
        self.outside = tempfile.mkdtemp()
        self.server = MCPServer(workspace_root=self.workspace)

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.workspace)
        shutil.rmtree(self.outside)

    def test_list_directory_inside_workspace_succeeds(self):
        (Path(self.workspace) / "sub").mkdir()
        result = _run(self.server._list_directory(self.workspace))
        assert not any("error" in item for item in result)

    def test_list_directory_outside_workspace_is_blocked(self):
        result = _run(self.server._list_directory(self.outside))
        assert len(result) == 1
        assert "outside the workspace" in result[0]["error"]

    def test_list_directory_relative_traversal_is_blocked(self):
        result = _run(self.server._list_directory("../"))
        assert len(result) == 1
        assert "outside the workspace" in result[0]["error"]


class TestSearchCodeConfinement:
    def setup_method(self):
        self.workspace = tempfile.mkdtemp()
        self.outside = tempfile.mkdtemp()
        (Path(self.outside) / "secret.txt").write_text("needle")
        self.server = MCPServer(workspace_root=self.workspace)

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.workspace)
        shutil.rmtree(self.outside)

    def test_search_code_outside_workspace_is_blocked(self):
        result = _run(self.server._search_code("needle", path=self.outside))
        assert len(result) == 1
        assert "outside the workspace" in result[0]["error"]

    def test_search_code_relative_traversal_is_blocked(self):
        result = _run(self.server._search_code("needle", path="../"))
        assert len(result) == 1
        assert "outside the workspace" in result[0]["error"]


class TestGetGitInfoConfinement:
    def setup_method(self):
        self.workspace = tempfile.mkdtemp()
        self.outside = tempfile.mkdtemp()
        self.server = MCPServer(workspace_root=self.workspace)

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.workspace)
        shutil.rmtree(self.outside)

    def test_get_git_info_inside_workspace_succeeds(self):
        result = _run(self.server._get_git_info(self.workspace))
        assert "error" not in result

    def test_get_git_info_outside_workspace_is_blocked(self):
        result = _run(self.server._get_git_info(self.outside))
        assert "outside the workspace" in result["error"]

    def test_get_git_info_relative_traversal_is_blocked(self):
        result = _run(self.server._get_git_info("../"))
        assert "outside the workspace" in result["error"]


class TestNoWorkspaceRootRemainsUnrestricted:
    """MCPServer() with no workspace_root (the `tools`/`screenshot` debug
    commands) must keep today's unrestricted behaviour for these three
    tools too -- confining only the standalone/local-chat-scoped case."""

    def setup_method(self):
        self.server = MCPServer()
        self.outside = tempfile.mkdtemp()

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.outside)

    def test_list_directory_anywhere_is_not_blocked(self):
        result = _run(self.server._list_directory(self.outside))
        assert not any("error" in item for item in result)

    def test_get_git_info_anywhere_is_not_blocked(self):
        result = _run(self.server._get_git_info(self.outside))
        assert "error" not in result


class TestLocalReadOnlyToolsConfinement:
    """local_chat.py's offline tool surface: previously built MCPServer()
    with no workspace_root at all, so read_file/list_directory/search_code/
    get_git_info had zero path-escape protection in local/offline chat
    mode even though they're the only tools a model can call there."""

    def setup_method(self):
        self.workspace = tempfile.mkdtemp()
        self.outside = tempfile.mkdtemp()
        (Path(self.outside) / "secret.txt").write_text("do-not-read-me")
        self.tools = LocalReadOnlyTools(workspace_root=self.workspace)

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.workspace)
        shutil.rmtree(self.outside)

    def test_defaults_to_cwd_when_no_workspace_root_given(self):
        tools = LocalReadOnlyTools()
        assert tools._server.workspace_root == str(Path.cwd())

    def test_read_file_outside_workspace_is_blocked(self):
        # read_file raises rather than returning an error string (see
        # mcp.py's _resolve_readable_input) -- local_chat.py's tool-call
        # loop is what turns this into a {"error": ...} message for the
        # model, same as any other tool exception.
        target = Path(self.outside) / "secret.txt"
        with pytest.raises(PermissionError, match="outside the workspace"):
            _run(self.tools.call("read_file", {"path": str(target)}))

    def test_list_directory_outside_workspace_is_blocked(self):
        result = _run(self.tools.call("list_directory", {"path": self.outside}))
        assert len(result) == 1
        assert "outside the workspace" in result[0]["error"]

    def test_search_code_outside_workspace_is_blocked(self):
        result = _run(self.tools.call("search_code", {"query": "do-not-read-me", "path": self.outside}))
        assert len(result) == 1
        assert "outside the workspace" in result[0]["error"]

    def test_get_git_info_outside_workspace_is_blocked(self):
        result = _run(self.tools.call("get_git_info", {"path": self.outside}))
        assert "outside the workspace" in result["error"]

    def test_reads_inside_workspace_still_work(self):
        (Path(self.workspace) / "readme.txt").write_text("hello")
        result = _run(self.tools.call("read_file", {"path": str(Path(self.workspace) / "readme.txt")}))
        assert result == "hello"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
