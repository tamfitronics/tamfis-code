#!/usr/bin/env python3
"""Test subagent system"""

import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from tamfis_code.agents import AgentManager, CodeAnalyzer, TestGenerator, DocGenerator, SubAgent, AgentTask, AgentStatus

class TestSubAgent:
    """Test subagent base class"""

    def test_init(self):
        """Test agent initialization"""
        agent = SubAgent('test', 'Test agent', ['test', 'demo'])
        assert agent.name == 'test'
        assert agent.description == 'Test agent'
        assert len(agent.capabilities) == 2

    def test_can_handle(self):
        """Test capability matching"""
        agent = SubAgent('test', 'Test', ['analyze', 'inspect'])
        assert agent.can_handle('Please analyze this code') is True
        assert agent.can_handle('This is unrelated') is False

class TestCodeAnalyzer:
    """Test code analyzer"""

    def setup_method(self):
        """Setup test environment"""
        self.analyzer = CodeAnalyzer()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_can_handle(self):
        """Test capability matching"""
        assert self.analyzer.can_handle('Analyze this code') is True
        assert self.analyzer.can_handle('Check complexity') is True

    @pytest.mark.asyncio
    async def test_execute(self):
        """Test executing analysis"""
        test_file = Path(self.temp_dir) / 'test.py'
        test_file.write_text('''
def hello():
    print("Hello")

class TestClass:
    def method(self):
        pass
''')
        
        task = AgentTask(
            id='test',
            description='Analyze test file',
            parameters={'file': str(test_file)}
        )
        
        result = await self.analyzer.execute(task)
        assert 'functions' in result
        assert result['functions'] >= 1
        assert 'classes' in result
        assert result['classes'] >= 1
        assert 'lines' in result
        assert 'issues' in result

class TestTestGenerator:
    """Test test generator"""

    def setup_method(self):
        """Setup test environment"""
        self.generator = TestGenerator()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_can_handle(self):
        """Test capability matching"""
        assert self.generator.can_handle('Generate unit tests') is True
        assert self.generator.can_handle('Test coverage') is True

    @pytest.mark.asyncio
    async def test_execute(self):
        """Test generating tests"""
        test_file = Path(self.temp_dir) / 'code.py'
        test_file.write_text('''
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b
''')
        
        task = AgentTask(
            id='test',
            description='Generate tests',
            parameters={'file': str(test_file)}
        )
        
        result = await self.generator.execute(task)
        assert 'file' in result
        assert 'functions_found' in result
        assert result['functions_found'] >= 2

class TestDocGenerator:
    """Test doc generator"""

    def setup_method(self):
        """Setup test environment"""
        self.generator = DocGenerator()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_can_handle(self):
        """Test capability matching"""
        assert self.generator.can_handle('Generate documentation') is True
        assert self.generator.can_handle('Add docstrings') is True

    @pytest.mark.asyncio
    async def test_execute(self):
        """Test generating documentation"""
        test_file = Path(self.temp_dir) / 'code.py'
        test_file.write_text('''
def hello():
    """Hello function"""
    return "Hello"

def world():
    return "World"
''')
        
        task = AgentTask(
            id='test',
            description='Generate docs',
            parameters={'file': str(test_file)}
        )
        
        result = await self.generator.execute(task)
        assert 'file' in result
        assert 'functions' in result
        assert 'missing_docs' in result
        # Should find that world() has no docstring
        assert len(result['missing_docs']) >= 1

class TestAgentManager:
    """Test agent manager"""

    def test_init(self):
        """Test manager initialization"""
        manager = AgentManager()
        agents = manager.list_agents()
        # Should have at least CodeAnalyzer, TestGenerator, DocGenerator
        assert len(agents) >= 3
        agent_names = [a['name'] for a in agents]
        assert 'code_analyzer' in agent_names
        assert 'test_generator' in agent_names
        assert 'doc_generator' in agent_names

    @pytest.mark.asyncio
    async def test_execute_task(self):
        """Test executing a task"""
        import tempfile
        temp_dir = tempfile.mkdtemp()
        try:
            test_file = Path(temp_dir) / 'test.py'
            test_file.write_text('def hello(): return "Hello"')
            
            manager = AgentManager()
            result = await manager.execute_task(
                'Analyze the code complexity',
                {'file': str(test_file)}
            )
            assert 'result' in result or 'error' in result
            if 'result' in result:
                assert 'functions' in result['result']
                assert result['result']['functions'] >= 1
        finally:
            import shutil
            shutil.rmtree(temp_dir)

    @pytest.mark.asyncio
    async def test_execute_with_no_agent(self):
        """Test executing a task with no matching agent"""
        manager = AgentManager()
        result = await manager.execute_task(
            'Do something unrelated',
            {}
        )
        assert 'error' in result
        assert 'No agent can handle this task' in result['error']

    @pytest.mark.asyncio
    async def test_get_agent(self):
        """Test finding an agent for a task"""
        manager = AgentManager()
        
        # Should find code_analyzer
        agent = manager.get_agent('Please analyze this code')
        assert agent is not None
        assert agent.name == 'code_analyzer'
        
        # Should find test_generator
        agent = manager.get_agent('Generate unit tests')
        assert agent is not None
        assert agent.name == 'test_generator'
        
        # Should find doc_generator
        agent = manager.get_agent('Add documentation')
        assert agent is not None
        assert agent.name == 'doc_generator'

    @pytest.mark.asyncio
    async def test_register_custom_agent(self):
        """Test registering a custom agent"""
        manager = AgentManager()
        
        class CustomAgent(SubAgent):
            async def execute(self, task):
                return {"custom": "result", "task": task.description}
        
        custom = CustomAgent('custom', 'Custom agent', ['custom', 'special'])
        manager.register(custom)
        
        agents = manager.list_agents()
        agent_names = [a['name'] for a in agents]
        assert 'custom' in agent_names
        
        result = await manager.execute_task('Do custom task', {})
        assert 'result' in result
        assert result['result'].get('custom') == 'result'

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
