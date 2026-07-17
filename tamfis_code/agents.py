"""Subagent system for autonomous task execution"""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import subprocess
import json
import re
import uuid
from pathlib import Path  # <-- Add this import

class AgentStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING = "waiting"

@dataclass
class AgentTask:
    """A task for a subagent"""
    id: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: AgentStatus = AgentStatus.IDLE
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class SubAgent:
    """Base class for subagents"""
    
    def __init__(self, name: str, description: str, capabilities: List[str]):
        self.name = name
        self.description = description
        self.capabilities = capabilities
        self.current_task: Optional[AgentTask] = None
    
    async def execute(self, task: AgentTask) -> Dict[str, Any]:
        """Execute a task - to be overridden"""
        raise NotImplementedError
    
    def can_handle(self, task_description: str) -> bool:
        """Check if agent can handle this task"""
        task_lower = task_description.lower()
        for cap in self.capabilities:
            if cap.lower() in task_lower:
                return True
        return False

class CodeAnalyzer(SubAgent):
    """Analyzes code structure and quality"""
    
    def __init__(self):
        super().__init__(
            name="code_analyzer",
            description="Analyzes code for patterns, issues, and complexity",
            capabilities=["analyze", "inspect", "complexity", "quality", "metrics"]
        )
    
    async def execute(self, task: AgentTask) -> Dict[str, Any]:
        """Analyze code based on task parameters"""
        file_path = task.parameters.get('file')
        if not file_path:
            return {"error": "No file specified"}
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            lines = content.split('\n')
            result = {
                "file": file_path,
                "lines": len(lines),
                "characters": len(content),
                "functions": self._count_functions(content),
                "classes": self._count_classes(content),
                "imports": self._count_imports(content),
                "complexity_score": self._calculate_complexity(content),
                "issues": self._find_issues(content),
            }
            return result
        except Exception as e:
            return {"error": str(e)}
    
    def _count_functions(self, content: str) -> int:
        return len(re.findall(r'^\s*def\s+\w+\s*\(', content, re.MULTILINE))
    
    def _count_classes(self, content: str) -> int:
        return len(re.findall(r'^\s*class\s+\w+', content, re.MULTILINE))
    
    def _count_imports(self, content: str) -> int:
        return len(re.findall(r'^\s*(?:from|import)\s+\w+', content, re.MULTILINE))
    
    def _calculate_complexity(self, content: str) -> float:
        lines = content.split('\n')
        complexity = 1
        keywords = ['if', 'elif', 'else', 'for', 'while', 'except', 'case', 'switch', '?']
        for line in lines:
            if any(k in line for k in keywords):
                complexity += 1
        return complexity
    
    def _find_issues(self, content: str) -> List[Dict[str, Any]]:
        issues = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            if len(line) > 120:
                issues.append({
                    "line": i,
                    "type": "line_too_long",
                    "message": f"Line {i} exceeds 120 characters ({len(line)})"
                })
            if line.strip().startswith('#') and 'TODO' in line:
                issues.append({
                    "line": i,
                    "type": "todo",
                    "message": f"TODO: {line.strip()}"
                })
            if line.strip().startswith('#') and 'FIXME' in line:
                issues.append({
                    "line": i,
                    "type": "fixme",
                    "message": f"FIXME: {line.strip()}"
                })
        
        return issues

class TestGenerator(SubAgent):
    """Generates tests for code"""
    
    def __init__(self):
        super().__init__(
            name="test_generator",
            description="Generates unit tests for code",
            capabilities=["test", "unit test", "coverage", "pytest"]
        )
    
    async def execute(self, task: AgentTask) -> Dict[str, Any]:
        """Generate tests for specified file"""
        file_path = task.parameters.get('file')
        if not file_path:
            return {"error": "No file specified"}
        
        # In production, this would use LLM to generate tests
        # For now, return a template
        funcs = self._extract_functions(file_path)
        return {
            "file": file_path,
            "functions_found": len(funcs),
            "functions": funcs,
            "test_file": f"test_{Path(file_path).name}",
            "status": "ready_for_generation"
        }
    
    def _extract_functions(self, file_path: str) -> List[str]:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            return re.findall(r'def\s+(\w+)\s*\(', content)
        except:
            return []

class DocGenerator(SubAgent):
    """Generates documentation for code"""
    
    def __init__(self):
        super().__init__(
            name="doc_generator",
            description="Generates documentation for code",
            capabilities=["doc", "documentation", "comment", "docstring"]
        )
    
    async def execute(self, task: AgentTask) -> Dict[str, Any]:
        """Generate documentation for specified file"""
        file_path = task.parameters.get('file')
        if not file_path:
            return {"error": "No file specified"}
        
        funcs = self._extract_with_docstrings(file_path)
        return {
            "file": file_path,
            "functions": funcs,
            "missing_docs": [f for f in funcs if not f.get('docstring')],
            "status": "ready_for_review"
        }
    
    def _extract_with_docstrings(self, file_path: str) -> List[Dict[str, Any]]:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            results = []
            lines = content.split('\n')
            i = 0
            while i < len(lines):
                line = lines[i]
                func_match = re.match(r'^\s*def\s+(\w+)\s*\(', line)
                if func_match:
                    func_name = func_match.group(1)
                    docstring = None
                    # Look ahead for docstring
                    j = i + 1
                    while j < len(lines) and j < i + 3:
                        if '"""' in lines[j] or "'''" in lines[j]:
                            docstring = lines[j].strip()
                            break
                        j += 1
                    results.append({
                        'name': func_name,
                        'line': i + 1,
                        'docstring': docstring
                    })
                i += 1
            return results
        except:
            return []

class DelegatedCodingAgent(SubAgent):
    """Delegates its task to the real standalone agent loop (runner_local.py --
    the same one `tamfis-code agent`/`exec`/`ask` and the interactive REPL
    drive), instead of a local heuristic.

    Unlike CodeAnalyzer/TestGenerator/DocGenerator, this one can actually act
    on an arbitrary objective via the model -- calling a provider directly
    and executing tools locally, with its own workspace/session. It's never
    selected by AgentManager's keyword-based `get_agent()` routing
    (capabilities=[]) since it's only meant to be dispatched explicitly via
    `execute_tasks`.
    """

    def __init__(
        self, *, manager, provider, model, console, workspace_root: str, session_id: int,
        approval_policy: str = "ask", mode: str = "agent",
    ):
        super().__init__(
            name="delegated_coding_agent",
            description="Delegates a sub-objective to the real standalone agent loop",
            capabilities=[],
        )
        self._manager = manager
        self._provider = provider
        self._model = model
        self._console = console
        self._workspace_root = workspace_root
        self._session_id = session_id
        self._approval_policy = approval_policy
        self._mode = mode

    async def execute(self, task: AgentTask) -> Dict[str, Any]:
        from .render import StreamRenderer
        from .runner_local import run_local_agent_turn

        renderer = StreamRenderer(self._console)
        outcome = await run_local_agent_turn(
            self._manager, self._provider, self._model, [{"role": "user", "content": task.description}],
            self._console, renderer,
            workspace_root=self._workspace_root, session_id=self._session_id,
            approval_policy=self._approval_policy, interactive=False,
            read_only=self._mode in {"chat", "audit", "plan"},
        )
        renderer.finish()
        return {"status": outcome.status, "summary": outcome.summary, "error": outcome.error}


class AgentManager:
    """Manages subagents and task execution"""
    
    def __init__(self):
        self.agents: Dict[str, SubAgent] = {}
        self.tasks: Dict[str, AgentTask] = {}
        self._register_default_agents()
    
    def _register_default_agents(self):
        """Register default agents"""
        self.register(CodeAnalyzer())
        self.register(TestGenerator())
        self.register(DocGenerator())
    
    def register(self, agent: SubAgent):
        """Register a subagent"""
        self.agents[agent.name] = agent
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all registered agents"""
        return [
            {
                "name": agent.name,
                "description": agent.description,
                "capabilities": agent.capabilities,
                "status": "ready"
            }
            for agent in self.agents.values()
        ]
    
    def get_agent(self, task_description: str) -> Optional[SubAgent]:
        """Find an agent that can handle the task"""
        for agent in self.agents.values():
            if agent.can_handle(task_description):
                return agent
        return None
    
    async def execute_task(self, description: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task using the appropriate agent"""
        import uuid
        
        task = AgentTask(
            id=str(uuid.uuid4())[:8],
            description=description,
            parameters=parameters,
            status=AgentStatus.RUNNING
        )
        
        agent = self.get_agent(description)
        if not agent:
            task.status = AgentStatus.FAILED
            task.error = "No agent can handle this task"
            return {"error": task.error}
        
        try:
            agent.current_task = task
            result = await agent.execute(task)
            task.result = result
            task.status = AgentStatus.COMPLETED
            return {"task_id": task.id, "agent": agent.name, "result": result}
        except Exception as e:
            task.status = AgentStatus.FAILED
            task.error = str(e)
            return {"error": str(e), "task_id": task.id}

    async def execute_tasks(
        self, descriptions: List[str], *,
        manager, provider, model, console, workspace_root, approval_policy: str = "ask",
        mode: str = "agent", max_concurrency: int = 1,
    ) -> List[Dict[str, Any]]:
        """Fan out N sub-objectives concurrently (bounded by max_concurrency),
        each delegated to the standalone agent loop in its own local session.

        Defaults to sequential (max_concurrency=1): whether concurrent tool
        execution against the same workspace is safe in every case (two
        sub-tasks editing overlapping files, concurrent state.json writers
        from separate local sessions) hasn't been stress-tested -- raise the
        cap only once you've validated that for your own workloads.
        """
        from .workspace import resolve_local_workspace

        semaphore = asyncio.Semaphore(max(1, max_concurrency))

        async def run_one(description: str) -> Dict[str, Any]:
            async with semaphore:
                task = AgentTask(id=f"delegated_{uuid.uuid4().hex[:8]}", description=description, status=AgentStatus.RUNNING)
                self.tasks[task.id] = task
                try:
                    workspace = resolve_local_workspace(Path(workspace_root), discover=False)
                    agent = DelegatedCodingAgent(
                        manager=manager, provider=provider, model=model, console=console,
                        workspace_root=workspace.workspace_root, session_id=workspace.session_id,
                        approval_policy=approval_policy, mode=mode,
                    )
                    result = await agent.execute(task)
                    task.result = result
                    task.status = AgentStatus.COMPLETED if result.get("status") == "completed" else AgentStatus.FAILED
                    task.error = result.get("error")
                except Exception as e:
                    result = {"error": str(e)}
                    task.status = AgentStatus.FAILED
                    task.error = str(e)
                return {"task_id": task.id, "description": description, "status": task.status.value, "result": result}

        return list(await asyncio.gather(*(run_one(description) for description in descriptions)))
