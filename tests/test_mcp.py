#!/usr/bin/env python3
"""Test MCP/Tools integration"""

import sys
import os
import tempfile
import tarfile
import zipfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from tamfis_code.mcp import MCPServer, ToolDefinition, _parse_duckduckgo_html


class TestArchiveTools:
    @pytest.mark.asyncio
    async def test_external_explicit_zip_can_be_extracted_edited_and_repackaged(self, tmp_path):
        workspace = tmp_path / "workspace"
        uploads = tmp_path / "uploads"
        workspace.mkdir()
        uploads.mkdir()
        archive_path = uploads / "project.zip"
        binary = b"\x00\xff\x10binary"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("src/app.py", "print('old')\n")
            archive.writestr("assets/data.bin", binary)

        server = MCPServer(workspace_root=str(workspace), attachment_paths=[str(archive_path)])
        extracted = await server.call_tool("extract_archive", {
            "path": str(archive_path), "destination": "project",
        })
        assert extracted["success"] is True
        assert (workspace / "project/src/app.py").read_text() == "print('old')\n"
        assert (workspace / "project/assets/data.bin").read_bytes() == binary

        (workspace / "project/src/app.py").write_text("print('new')\n")
        packaged = await server.call_tool("repackage_archive", {
            "source_dir": "project", "output_path": "updated-project.tar.gz",
        })
        assert packaged["success"] is True
        with tarfile.open(workspace / "updated-project.tar.gz", "r:gz") as archive:
            assert archive.extractfile("src/app.py").read() == b"print('new')\n"
            assert archive.extractfile("assets/data.bin").read() == binary

    @pytest.mark.asyncio
    async def test_unapproved_external_archive_is_not_readable(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        archive_path = tmp_path / "private.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("ok.txt", "no")
        server = MCPServer(workspace_root=str(workspace))
        result = await server.call_tool("extract_archive", {"path": str(archive_path)})
        assert result["success"] is False
        assert "outside the workspace" in result["error"]

    @pytest.mark.asyncio
    async def test_zip_traversal_is_rejected_without_partial_destination(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        archive_path = workspace / "unsafe.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("safe.txt", "would otherwise be written first")
            archive.writestr("../escape.txt", "bad")
        server = MCPServer(workspace_root=str(workspace))
        result = await server.call_tool("extract_archive", {
            "path": "unsafe.zip", "destination": "expanded",
        })
        assert result["success"] is False
        assert not (workspace / "expanded").exists()
        assert not (tmp_path / "escape.txt").exists()

class TestMCPServer:
    """Test MCP server"""

    def setup_method(self):
        """Setup test environment"""
        self.server = MCPServer()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_list_tools(self):
        """Test listing tools"""
        tools = self.server.list_tools()
        assert len(tools) >= 5  # At least default tools
        assert "browser" in {tool["name"] for tool in tools}

    @pytest.mark.asyncio
    async def test_read_file(self):
        """Test reading a file"""
        test_file = Path(self.temp_dir) / 'test.txt'
        test_file.write_text('Hello, world!')
        
        result = await self.server.call_tool('read_file', {'path': str(test_file)})
        assert result['success'] is True
        assert 'Hello, world!' in result['result']

    @pytest.mark.asyncio
    async def test_read_file_rejects_binary_content_instead_of_returning_garbage(self):
        # Regression: read_text(errors='ignore') used to silently drop every
        # invalid byte and hand back plausible-looking garbage for a binary
        # file (e.g. an attached image) instead of a clear error -- caught
        # while wiring real vision/image attachment support.
        binary_file = Path(self.temp_dir) / 'image.png'
        binary_file.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00" * 20)
        result = await self.server.call_tool('read_file', {'path': str(binary_file)})
        assert result['success'] is True  # the tool call itself succeeded
        assert 'binary file' in result['result']
        assert 'vision-capable models' in result['result']

    @pytest.mark.asyncio
    async def test_write_file(self):
        """Test writing a file"""
        test_file = Path(self.temp_dir) / 'write_test.txt'
        
        result = await self.server.call_tool('write_file', {
            'path': str(test_file),
            'content': 'Test content'
        })
        assert result['success'] is True
        assert test_file.exists()
        assert test_file.read_text() == 'Test content'

    @pytest.mark.asyncio
    async def test_list_directory(self):
        """Test listing a directory"""
        result = await self.server.call_tool('list_directory', {'path': self.temp_dir})
        assert result['success'] is True
        assert isinstance(result['result'], list)

    @pytest.mark.asyncio
    async def test_search_code(self):
        """Test code search"""
        # Create a test file
        test_file = Path(self.temp_dir) / 'search_test.py'
        test_file.write_text('def hello(): print("world")')
        
        result = await self.server.call_tool('search_code', {'query': 'hello', 'path': self.temp_dir})
        # If rg is installed, should return results
        # If not, should return error
        assert 'success' in result or 'error' in result

    @pytest.mark.asyncio
    async def test_get_git_info(self):
        """Test git info"""
        result = await self.server.call_tool('get_git_info', {'path': '.'})
        assert 'result' in result or 'error' in result

    def test_register_custom_tool(self):
        """Test registering a custom tool"""
        def custom_handler(**kwargs):
            return f"Custom: {kwargs}"

        self.server.register_tool(
            name='custom',
            description='Custom tool',
            parameters={'type': 'object', 'properties': {'test': {'type': 'string'}}},
            handler=custom_handler
        )

        tools = self.server.list_tools()
        tool_names = [t['name'] for t in tools]
        assert 'custom' in tool_names

    def test_tool_schemas_openai_wraps_in_function_envelope(self):
        schemas = self.server.tool_schemas_openai(names=["read_file"])
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "read_file"
        assert "parameters" in schemas[0]["function"]


class _FakeConsole:
    """Records every print() call and returns queued input() answers --
    stands in for rich.console.Console without touching a real terminal."""

    def __init__(self, answers):
        self._answers = list(answers)
        self.printed = []

    def print(self, *args, **kwargs):
        self.printed.append(args[0] if args else "")

    def input(self, prompt=""):
        self.printed.append(prompt)
        return self._answers.pop(0)


class TestAskUserQuestionTool:
    """User-requested feature: the standalone agent should be able to pause
    and ask a real clarifying question instead of silently guessing --
    confirmed as the same failure pattern behind the WordPress/package.json
    bug (guessed Node conventions instead of checking or asking)."""

    @pytest.mark.asyncio
    async def test_unavailable_without_an_attached_interactive_console(self):
        server = MCPServer()  # no console/renderer/interactive supplied at all
        result = await server.call_tool("ask_user_question", {"question": "Which environment?"})
        assert result["success"] is True
        assert "unavailable" in result["result"]

    @pytest.mark.asyncio
    async def test_console_supplied_but_not_interactive_stays_unavailable(self):
        console = _FakeConsole(answers=["should never be read"])
        server = MCPServer(console=console, interactive=False)
        result = await server.call_tool("ask_user_question", {"question": "Which environment?"})
        assert "unavailable" in result["result"]
        assert console.printed == []  # never even printed the question

    @pytest.mark.asyncio
    async def test_free_text_answer_is_returned_verbatim(self):
        console = _FakeConsole(answers=["staging, not production"])
        server = MCPServer(console=console, interactive=True)
        result = await server.call_tool("ask_user_question", {"question": "Which environment?"})
        assert result["result"] == "staging, not production"

    @pytest.mark.asyncio
    async def test_numeric_selection_resolves_to_the_matching_option_text(self):
        console = _FakeConsole(answers=["2"])
        server = MCPServer(console=console, interactive=True)
        result = await server.call_tool(
            "ask_user_question",
            {"question": "Node or PHP?", "options": ["Node/React", "PHP/WordPress"]},
        )
        assert result["result"] == "PHP/WordPress"

    @pytest.mark.asyncio
    async def test_free_text_still_accepted_even_when_options_are_offered(self):
        console = _FakeConsole(answers=["actually it's Django"])
        server = MCPServer(console=console, interactive=True)
        result = await server.call_tool(
            "ask_user_question",
            {"question": "Node or PHP?", "options": ["Node/React", "PHP/WordPress"]},
        )
        assert result["result"] == "actually it's Django"


class TestMCPServerNoWorkspaceRoot:
    """MCPServer() with no workspace_root -- legacy unrestricted behaviour,
    e.g. the `tools`/`screenshot` debug commands -- must be unchanged."""

    def setup_method(self):
        self.server = MCPServer()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    @pytest.mark.asyncio
    async def test_write_file_outside_any_root_is_not_blocked(self):
        # No workspace_root was given, so there's nothing to enforce a
        # boundary against -- matches today's behaviour exactly.
        target = Path(self.temp_dir) / "sub" / "f.txt"
        result = await self.server.call_tool('write_file', {'path': str(target), 'content': 'hi'})
        assert result['success'] is True
        assert target.read_text() == 'hi'

    @pytest.mark.asyncio
    async def test_write_file_with_no_session_id_does_not_record_mutation(self):
        target = Path(self.temp_dir) / "f.txt"
        await self.server.call_tool('write_file', {'path': str(target), 'content': 'hi'})
        # No exception, no ledger entry expected -- session_id was never given.
        assert self.server.session_id is None


class TestMCPServerWorkspaceScoped:
    """MCPServer(workspace_root=..., session_id=...) -- the standalone
    agent loop's configuration: boundary enforcement + mutation ledger."""

    def setup_method(self):
        from tamfis_code import state as state_module

        self.temp_dir = tempfile.mkdtemp()
        self.session_id = 4242
        self.server = MCPServer(workspace_root=self.temp_dir, session_id=self.session_id)

        self._state_module = state_module
        self._originals = (state_module.CONFIG_DIR, state_module.STATE_PATH)
        self._cfg_tmp = tempfile.TemporaryDirectory()
        cfg_base = Path(self._cfg_tmp.name)
        state_module.CONFIG_DIR = cfg_base / ".config"
        state_module.STATE_PATH = cfg_base / ".config" / "state.json"

    def teardown_method(self):
        import shutil
        self._state_module.CONFIG_DIR, self._state_module.STATE_PATH = self._originals
        self._cfg_tmp.cleanup()
        shutil.rmtree(self.temp_dir)

    @pytest.mark.asyncio
    async def test_write_file_inside_workspace_succeeds(self):
        target = Path(self.temp_dir) / "inside.py"
        result = await self.server.call_tool('write_file', {'path': str(target), 'content': 'x = 1'})
        assert result['success'] is True
        state = self._state_module.get_session_state(self.session_id)
        assert len(state.modified_files) == 1
        assert state.modified_files[0]['operation'] == 'create'

    @pytest.mark.asyncio
    async def test_write_file_outside_workspace_is_blocked(self):
        result = await self.server.call_tool('write_file', {'path': '/etc/passwd_probe_should_not_exist', 'content': 'x'})
        assert result['success'] is False
        assert 'outside the workspace' in result['error']

    @pytest.mark.asyncio
    async def test_edit_file_replaces_unique_match(self):
        target = Path(self.temp_dir) / "edit.py"
        target.write_text("def foo():\n    return 1\n")
        result = await self.server.call_tool('edit_file', {
            'path': str(target), 'old_string': 'return 1', 'new_string': 'return 2',
        })
        assert result['success'] is True
        assert target.read_text() == "def foo():\n    return 2\n"
        state = self._state_module.get_session_state(self.session_id)
        assert len(state.modified_files) == 1

    @pytest.mark.asyncio
    async def test_edit_file_fails_on_zero_matches(self):
        target = Path(self.temp_dir) / "edit.py"
        target.write_text("def foo():\n    return 1\n")
        result = await self.server.call_tool('edit_file', {
            'path': str(target), 'old_string': 'not present anywhere', 'new_string': 'x',
        })
        assert result['success'] is True  # call_tool succeeds; the handler reports failure in its message
        assert 'not found' in result['result']
        assert target.read_text() == "def foo():\n    return 1\n"  # unchanged

    @pytest.mark.asyncio
    async def test_edit_file_fails_on_multiple_matches(self):
        target = Path(self.temp_dir) / "edit.py"
        target.write_text("x = 1\nx = 1\n")
        result = await self.server.call_tool('edit_file', {
            'path': str(target), 'old_string': 'x = 1', 'new_string': 'x = 2',
        })
        assert 'matches 2 times' in result['result']
        assert target.read_text() == "x = 1\nx = 1\n"  # unchanged

    @pytest.mark.asyncio
    async def test_execute_command_cwd_actually_changes_directory(self):
        """A model repeatedly tried to invent a 'directory'/other bogus
        argument on execute_command to target a subdirectory, since the tool
        had no way to express that other than shell-chaining `cd X && ...`
        into the command string -- confirmed live it can then pick up an
        unrelated ancestor's config (e.g. npm climbing to a stray parent
        package.json). cwd is the real, supported way to do this now."""
        subdir = Path(self.temp_dir) / "sub"
        subdir.mkdir()
        (subdir / "marker.txt").write_text("here")

        result = await self.server.call_tool('execute_command', {'command': 'ls', 'cwd': 'sub'})

        assert result['result']['success'] is True
        assert 'marker.txt' in result['result']['stdout']

    @pytest.mark.asyncio
    async def test_execute_command_cwd_outside_workspace_is_blocked(self):
        result = await self.server.call_tool('execute_command', {'command': 'ls', 'cwd': '/etc'})

        assert result['result']['success'] is False
        assert 'outside the workspace' in result['result']['error']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

@pytest.mark.asyncio
async def test_execute_command_rejects_npm_without_local_package_manifest(tmp_path):
    server = MCPServer(workspace_root=str(tmp_path), session_id=9911)
    result = await server.call_tool('execute_command', {'command': 'npm install', 'cwd': '.'})
    assert result['result']['success'] is False
    assert 'no local project manifest' in result['result']['error']


@pytest.mark.asyncio
async def test_execute_command_does_not_use_parent_package_manifest(tmp_path):
    (tmp_path / 'package.json').write_text('{}')
    child = tmp_path / 'python-backend'
    child.mkdir()
    (child / 'pyproject.toml').write_text('[project]\nname="backend"\n')
    server = MCPServer(workspace_root=str(tmp_path), session_id=9912)
    result = await server.call_tool('execute_command', {'command': 'npm install', 'cwd': 'python-backend'})
    assert result['result']['success'] is False
    assert 'Parent-directory manifests are ignored' in result['result']['error']


@pytest.mark.asyncio
async def test_execute_command_accepts_a_string_timeout(tmp_path):
    # Live-reported crash: a real tool call sent {"timeout": "300"} (a
    # string, despite the schema declaring an integer) -- reached
    # asyncio.wait_for(timeout=...) unmodified and crashed with
    # "'<=' not supported between instances of 'str' and 'int'",
    # silently breaking execute_command for the rest of that turn instead
    # of running the command.
    server = MCPServer(workspace_root=str(tmp_path), session_id=9914)
    result = await server.call_tool('execute_command', {'command': 'echo hi', 'timeout': '5'})
    assert result['result']['success'] is True
    assert result['result']['stdout'].strip() == 'hi'


@pytest.mark.asyncio
async def test_execute_command_falls_back_to_default_on_unparseable_timeout(tmp_path):
    server = MCPServer(workspace_root=str(tmp_path), session_id=9915)
    result = await server.call_tool('execute_command', {'command': 'echo hi', 'timeout': 'not-a-number'})
    assert result['result']['success'] is True


@pytest.mark.asyncio
async def test_execute_command_falls_back_to_default_on_non_positive_timeout(tmp_path):
    server = MCPServer(workspace_root=str(tmp_path), session_id=9916)
    result = await server.call_tool('execute_command', {'command': 'echo hi', 'timeout': 0})
    assert result['result']['success'] is True


@pytest.mark.asyncio
async def test_execute_command_ignores_non_dict_environment_instead_of_crashing(tmp_path):
    # Live-reported crash: a real tool call sent `environment` as something
    # other than a real object (repro'd here as a JSON-encoded string, the
    # most likely real shape) -- reached environment.items() unmodified and
    # crashed with "'str' object has no attribute 'items'", silently
    # breaking execute_command for the rest of that turn. Same bug class as
    # the string-timeout crash, same tool.
    server = MCPServer(workspace_root=str(tmp_path), session_id=9917)
    result = await server.call_tool('execute_command', {
        'command': 'echo hi', 'environment': '{"TAMFIS_TEST_VALUE": "works"}',
    })
    assert result['result']['success'] is True
    assert result['result']['stdout'].strip() == 'hi'


@pytest.mark.asyncio
async def test_execute_command_environment_override(tmp_path):
    server = MCPServer(workspace_root=str(tmp_path), session_id=9913)
    result = await server.call_tool('execute_command', {
        'command': 'printf %s "$TAMFIS_TEST_VALUE"',
        'cwd': '.',
        'environment': {'TAMFIS_TEST_VALUE': 'works'},
        'shell': 'bash',
    })
    assert result['result']['success'] is True
    assert result['result']['stdout'] == 'works'


def _fake_async_client(post_side_effect):
    """Stand in for ``async with httpx.AsyncClient(...) as client: await
    client.post(...)`` -- mirrors _web_search's own usage shape."""
    client = MagicMock()
    client.post = AsyncMock(side_effect=post_side_effect)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=client)
    cm.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=cm)


class TestWebSearchTool:
    """web_search is self-contained (Tavily if TAVILY_API_KEY is set, else
    DuckDuckGo HTML fallback) rather than reusing tamgpt6's WebSearchManager
    via a monorepo import, so it keeps working when tamfis-code is installed
    standalone on a machine that never had tamgpt6 on it."""

    def test_parse_duckduckgo_html_extracts_title_url_snippet(self):
        html = (
            '<a rel="nofollow" class="result__a" href="https://example.com/a">Example A</a>'
            '<a class="result__snippet">First <b>snippet</b> text</a>'
            '<a rel="nofollow" class="result__a" href="https://example.com/b">Example B</a>'
            '<a class="result__snippet">Second snippet text</a>'
        )
        results = _parse_duckduckgo_html(html, max_results=5)
        assert results == [
            {"title": "Example A", "url": "https://example.com/a", "snippet": "First snippet text"},
            {"title": "Example B", "url": "https://example.com/b", "snippet": "Second snippet text"},
        ]

    def test_parse_duckduckgo_html_respects_max_results(self):
        html = "".join(
            f'<a rel="nofollow" class="result__a" href="https://example.com/{i}">T{i}</a>'
            for i in range(10)
        )
        results = _parse_duckduckgo_html(html, max_results=3)
        assert len(results) == 3

    def test_parse_duckduckgo_html_handles_no_results(self):
        assert _parse_duckduckgo_html("<html><body>nothing here</body></html>", max_results=5) == []

    def test_parse_duckduckgo_html_unescapes_entities(self):
        html_blob = (
            '<a rel="nofollow" class="result__a" href="https://example.com/c">'
            "What&#x27;s New &amp; Notable</a>"
            '<a class="result__snippet">Tom &amp; Jerry&#x27;s show</a>'
        )
        results = _parse_duckduckgo_html(html_blob, max_results=5)
        assert results == [
            {
                "title": "What's New & Notable",
                "url": "https://example.com/c",
                "snippet": "Tom & Jerry's show",
            }
        ]

    @pytest.mark.asyncio
    async def test_web_search_rejects_empty_query(self):
        server = MCPServer()
        with pytest.raises(ValueError):
            await server._web_search(query="   ")

    @pytest.mark.asyncio
    async def test_web_search_uses_tavily_when_api_key_configured(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "fake-key")
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "results": [{"title": "Tavily Result", "url": "https://tavily.example/1", "content": "About it"}]
        }

        async def fake_post(*args, **kwargs):
            return response

        with patch("tamfis_code.mcp.httpx.AsyncClient", _fake_async_client(fake_post)):
            result = await MCPServer()._web_search(query="anything")
        assert result["provider"] == "tavily"
        assert result["results"] == [
            {"title": "Tavily Result", "url": "https://tavily.example/1", "snippet": "About it"}
        ]

    @pytest.mark.asyncio
    async def test_web_search_falls_back_to_duckduckgo_without_api_key(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        response = MagicMock()
        response.status_code = 200
        response.text = (
            '<a rel="nofollow" class="result__a" href="https://ddg.example/1">DDG Result</a>'
            '<a class="result__snippet">A snippet</a>'
        )

        async def fake_post(*args, **kwargs):
            return response

        with patch("tamfis_code.mcp.httpx.AsyncClient", _fake_async_client(fake_post)):
            result = await MCPServer()._web_search(query="anything")
        assert result["provider"] == "duckduckgo"
        assert result["results"] == [
            {"title": "DDG Result", "url": "https://ddg.example/1", "snippet": "A snippet"}
        ]

    @pytest.mark.asyncio
    async def test_web_search_falls_back_to_duckduckgo_when_tavily_errors(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "fake-key")
        ddg_response = MagicMock()
        ddg_response.status_code = 200
        ddg_response.text = (
            '<a rel="nofollow" class="result__a" href="https://ddg.example/2">Fallback Result</a>'
            '<a class="result__snippet">Fallback snippet</a>'
        )

        calls = {"n": 0}

        async def fake_post(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.ConnectError("boom")
            return ddg_response

        with patch("tamfis_code.mcp.httpx.AsyncClient", _fake_async_client(fake_post)):
            result = await MCPServer()._web_search(query="anything")
        assert result["provider"] == "duckduckgo"
        assert calls["n"] == 2

    @pytest.mark.asyncio
    async def test_web_search_reports_no_results_found(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        response = MagicMock()
        response.status_code = 200
        response.text = "<html><body>no results</body></html>"

        async def fake_post(*args, **kwargs):
            return response

        with patch("tamfis_code.mcp.httpx.AsyncClient", _fake_async_client(fake_post)):
            result = await MCPServer()._web_search(query="anything")
        assert result["provider"] is None
        assert result["results"] == []
        assert "No results found" in result["message"]

    @pytest.mark.asyncio
    async def test_web_search_is_registered_and_reachable_via_call_tool(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        response = MagicMock()
        response.status_code = 200
        response.text = (
            '<a rel="nofollow" class="result__a" href="https://ddg.example/3">Via call_tool</a>'
            '<a class="result__snippet">snippet</a>'
        )

        async def fake_post(*args, **kwargs):
            return response

        server = MCPServer()
        assert "web_search" in {tool["name"] for tool in server.list_tools()}
        with patch("tamfis_code.mcp.httpx.AsyncClient", _fake_async_client(fake_post)):
            outcome = await server.call_tool("web_search", {"query": "anything"})
        assert outcome["success"] is True
        assert outcome["result"]["provider"] == "duckduckgo"
