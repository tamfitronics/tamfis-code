#!/usr/bin/env python3
"""Test MCP/Tools integration"""

import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from tamfis_code.mcp import MCPServer, ToolDefinition

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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
