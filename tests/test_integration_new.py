#!/usr/bin/env python3
"""Integration test for all new modules"""

import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

class TestNewModulesIntegration:
    """Test integration of all new modules"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace = Path(self.temp_dir)
        
        # Create a test file
        test_file = self.workspace / 'test.py'
        test_file.write_text('def test_func(): return "Hello"')
        
        # Create TAMFIS.md
        instruction_file = self.workspace / 'TAMFIS.md'
        instruction_file.write_text('# Test Instructions\nTest content')

    def teardown_method(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_references(self):
        """Test reference resolution"""
        from tamfis_code.references import process_references
        
        result = process_references('@test.py', self.workspace)
        assert 'references' in result
        assert len(result['references']['files']) >= 0

    def test_sessions(self):
        """Test session management"""
        from tamfis_code.sessions import SessionManager
        
        manager = SessionManager(self.workspace / 'sessions.db')
        session = manager.create_session('Integration Test')
        session.add_message('user', 'Hello')
        manager.save(session)
        
        loaded = manager.load(session.id)
        assert loaded is not None
        assert len(loaded.messages) == 1

    def test_metrics(self):
        """Test metrics tracking"""
        from tamfis_code.metrics import MetricsTracker
        
        tracker = MetricsTracker()
        tracker.record(100, 500)
        summary = tracker.get_summary()
        
        assert summary['tokens_used'] == 100

    def test_planreview(self):
        """Test plan and review"""
        from tamfis_code.planreview import PlanReviewer, FileChange, ChangeType
        
        reviewer = PlanReviewer()
        changes = [
            FileChange(
                path=str(self.workspace / 'new.py'),
                type=ChangeType.CREATE,
                content='print("Hello")'
            )
        ]
        reviewer.create_plan('Test Plan', changes)
        reviewer.approve()
        
        results = reviewer.apply()
        assert results[0]['success'] is True

    def test_agents(self):
        """Test agents"""
        import asyncio
        from tamfis_code.agents import AgentManager
        
        manager = AgentManager()
        agents = manager.list_agents()
        assert len(agents) >= 3

    def test_mcp(self):
        """Test MCP tools"""
        import asyncio
        from tamfis_code.mcp import MCPServer
        
        server = MCPServer()
        tools = server.list_tools()
        assert len(tools) >= 5

    def test_indexer(self):
        """Test code indexer"""
        from tamfis_code.indexer import CodeIndexer
        
        indexer = CodeIndexer(self.workspace, self.workspace / 'index')
        indexer.index([str(self.workspace)])
        stats = indexer.get_stats()
        
        assert stats['files'] >= 1

    def test_completion(self):
        """Test completion"""
        from tamfis_code.completion import ShellCompleter
        
        bash = ShellCompleter.generate_bash()
        assert '_tamfis_code_completion' in bash

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
