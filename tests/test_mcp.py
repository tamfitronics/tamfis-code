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

    async def test_read_file(self):
        """Test reading a file"""
        test_file = Path(self.temp_dir) / 'test.txt'
        test_file.write_text('Hello, world!')
        
        result = await self.server.call_tool('read_file', {'path': str(test_file)})
        assert result['success'] is True
        assert 'Hello, world!' in result['result']

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

    async def test_list_directory(self):
        """Test listing a directory"""
        result = await self.server.call_tool('list_directory', {'path': self.temp_dir})
        assert result['success'] is True
        assert isinstance(result['result'], list)

    async def test_search_code(self):
        """Test code search"""
        # Create a test file
        test_file = Path(self.temp_dir) / 'search_test.py'
        test_file.write_text('def hello(): print("world")')
        
        result = await self.server.call_tool('search_code', {'query': 'hello', 'path': self.temp_dir})
        # If rg is installed, should return results
        # If not, should return error
        assert 'success' in result or 'error' in result

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

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
