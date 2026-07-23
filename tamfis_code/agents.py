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
    # Application component, not a pytest test container.
    __test__ = False
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
        renderer_factory: Optional[Callable[[], Any]] = None,
        extra_system_prompt: Optional[str] = None,
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
        # None (the default) preserves every existing caller's exact prior
        # behavior: a real StreamRenderer on the shared console. Only a
        # concurrent swarm run (execute_tasks with max_concurrency > 1)
        # passes one, to avoid N concurrent rich.live.Live regions on one
        # Console -- see swarm.BufferedSubagentRenderer.
        self._renderer_factory = renderer_factory
        # A declarative subagent type's own prompt (agent_definitions.py) --
        # prepended as a real system message ahead of the objective, not
        # merged into it, so the underlying model still sees the actual
        # task description as a distinct user turn.
        self._extra_system_prompt = extra_system_prompt

    async def execute(self, task: AgentTask) -> Dict[str, Any]:
        from .runner_local import run_local_agent_turn

        if self._renderer_factory is not None:
            renderer = self._renderer_factory()
        else:
            from .render import StreamRenderer
            renderer = StreamRenderer(self._console)
        messages: list[dict[str, str]] = []
        if self._extra_system_prompt:
            messages.append({"role": "system", "content": self._extra_system_prompt})
        messages.append({"role": "user", "content": task.description})
        outcome = await run_local_agent_turn(
            self._manager, self._provider, self._model, messages,
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
        parent_session_id: Optional[int] = None,
        renderer_factory: Optional[Callable[[str, str], Any]] = None,
        agent_types: Optional[List[Optional[str]]] = None,
    ) -> List[Dict[str, Any]]:
        """Fan out N sub-objectives concurrently (bounded by max_concurrency),
        each delegated to the standalone agent loop in its own local session.

        Each sub-task gets its own child session (workspace.
        resolve_swarm_subtask_workspace) rather than resolve_local_workspace's
        usual same-workspace_root reuse -- concurrent sub-tasks sharing one
        session_id would race on state.json's single-value fields
        (current_phase/running_action/active_task/...), which aren't
        merge-safe the way queued_user_instructions/saved_plans are.
        parent_session_id (the caller's own session, when known -- may be
        None, e.g. a one-shot `agent-cmd delegate` invocation with no
        pre-existing session) is recorded as best-effort context; every
        child is unconditionally tagged is_swarm_child=True regardless,
        which is the actual filter default session listings use.

        Defaults to sequential (max_concurrency=1): whether concurrent tool
        execution against the same workspace is safe in every case (two
        sub-tasks editing overlapping files) hasn't been stress-tested --
        raise the cap only once you've validated that for your own workloads.

        renderer_factory(task_id, description), when given, is called once
        per sub-task to build its renderer instead of the default real
        StreamRenderer on the shared console -- required at max_concurrency
        > 1 (see swarm.BufferedSubagentRenderer) to avoid N concurrent
        rich.live.Live regions colliding on one Console. None (the default)
        preserves today's exact behavior for every existing caller.

        agent_types, when given, is a list the same length as descriptions
        (None entries mean "no override" -- every existing caller omitting
        this parameter entirely is unaffected). A named entry resolves a
        declarative subagent type (agent_definitions.py: `.tamfis/agents/
        <name>.md` or the user config equivalent) for just that one
        sub-task -- its system_prompt is prepended ahead of the sub-task's
        own objective, and its model/provider (if the definition sets them)
        override this call's shared model/provider for that sub-task only.
        An unknown agent type name is a no-op (falls back to the shared
        model/provider/no extra prompt) rather than failing the whole
        fan-out over one bad name.
        """
        from .agent_definitions import load_agent_definitions
        from .local_chat import resolve_provider_type
        from .workspace import resolve_swarm_subtask_workspace

        definitions = load_agent_definitions(workspace_root) if agent_types else {}
        resolved_agent_types: List[Optional[str]] = (
            list(agent_types) if agent_types is not None else [None] * len(descriptions)
        )
        semaphore = asyncio.Semaphore(max(1, max_concurrency))

        async def run_one(description: str, agent_type: Optional[str]) -> Dict[str, Any]:
            async with semaphore:
                task = AgentTask(id=f"delegated_{uuid.uuid4().hex[:8]}", description=description, status=AgentStatus.RUNNING)
                self.tasks[task.id] = task
                try:
                    workspace = resolve_swarm_subtask_workspace(
                        Path(workspace_root), parent_session_id=parent_session_id, label=description[:80],
                    )
                    task_provider, task_model, extra_system_prompt = provider, model, None
                    definition = definitions.get(agent_type) if agent_type else None
                    if definition is not None:
                        extra_system_prompt = definition.system_prompt
                        if definition.model:
                            task_model = definition.model
                        if definition.provider:
                            try:
                                task_provider = resolve_provider_type(definition.provider)
                            except ValueError:
                                pass
                    agent = DelegatedCodingAgent(
                        manager=manager, provider=task_provider, model=task_model, console=console,
                        workspace_root=workspace.workspace_root, session_id=workspace.session_id,
                        approval_policy=approval_policy, mode=mode,
                        renderer_factory=(
                            (lambda tid=task.id, desc=description: renderer_factory(tid, desc))
                            if renderer_factory is not None else None
                        ),
                        extra_system_prompt=extra_system_prompt,
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

        return list(await asyncio.gather(*(
            run_one(description, resolved_agent_types[i] if i < len(resolved_agent_types) else None)
            for i, description in enumerate(descriptions)
        )))
