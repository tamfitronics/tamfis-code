"""Model Context Protocol (MCP) integration for tools"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
import json
import os
import subprocess
import asyncio
import sys
from pathlib import Path


def _import_monorepo_attr(module_path: str, attr: str):
    """Import `attr` from `module_path`, only if a monorepo (tamgpt6) checkout
    happens to be co-located next to this standalone package -- e.g. a dev
    running an editable install from inside tamgpt6/tamfis_code, or with
    tamgpt6 as the current working directory.

    Returns None (never raises) when the monorepo isn't present. tamfis-code
    is an independent package with no hard dependency on tamgpt6's backend
    modules; callers of this helper must treat None as "unavailable outside
    a monorepo checkout" and report that clearly rather than crash.
    """
    try:
        module = __import__(module_path, fromlist=[attr])
        return getattr(module, attr)
    except ModuleNotFoundError:
        pass
    top_level_package = module_path.split(".", 1)[0]
    candidates = [Path.cwd(), *Path(__file__).resolve().parents]
    for root in candidates:
        if (root / top_level_package).is_dir():
            root_text = str(root)
            if root_text not in sys.path:
                sys.path.insert(0, root_text)
            try:
                module = __import__(module_path, fromlist=[attr])
                return getattr(module, attr)
            except ModuleNotFoundError:
                continue
    return None


def _get_shared_mcp_bridge():
    """Load the monorepo MCP bridge, if a tamgpt6 checkout is co-located.

    Returns None otherwise -- see _import_monorepo_attr's docstring.
    """
    get_mcp_bridge = _import_monorepo_attr("tier_viii_infrastructure.mcp.orchestrator_bridge", "get_mcp_bridge")
    return get_mcp_bridge() if get_mcp_bridge is not None else None


def get_browser_tool_class():
    """Load BrowserTool, if a tamgpt6 checkout is co-located, else None."""
    return _import_monorepo_attr("tier_iv_orchestration.tools.browser_tool", "BrowserTool")

@dataclass
class ToolDefinition:
    """Definition of a tool for MCP"""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema
    handler: Optional[Callable] = None

class MCPServer:
    """MCP server for tool execution"""

    def __init__(self, *, workspace_root: Optional[str] = None, session_id: Optional[int] = None):
        # workspace_root/session_id are optional so existing callers that
        # construct MCPServer() with no arguments (tests, the `tools`/
        # `screenshot` debug commands) keep today's behaviour: no boundary
        # enforcement on write_file/edit_file, no mutation-ledger recording.
        # The standalone agent loop (runner_local.py) always supplies both.
        self.workspace_root = workspace_root
        self.session_id = session_id
        self.tools: Dict[str, ToolDefinition] = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register default tools"""
        
        self.register_tool(
            name="read_file",
            description="Read contents of a file",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"}
                },
                "required": ["path"]
            },
            handler=self._read_file
        )
        
        self.register_tool(
            name="write_file",
            description="Write content to a file",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "File content"}
                },
                "required": ["path", "content"]
            },
            handler=self._write_file
        )
        
        self.register_tool(
            name="edit_file",
            description=(
                "Replace an exact, unique occurrence of old_string with new_string in a file. "
                "Fails if old_string is not found, or is not unique -- include enough surrounding "
                "context in old_string to make the match unambiguous. Use write_file instead for "
                "creating a brand-new file or replacing one's entire contents."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "old_string": {"type": "string", "description": "Exact text to replace (must match exactly once)"},
                    "new_string": {"type": "string", "description": "Replacement text"},
                },
                "required": ["path", "old_string", "new_string"],
            },
            handler=self._edit_file,
        )

        self.register_tool(
            name="list_directory",
            description="List contents of a directory",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path"}
                },
                "required": ["path"]
            },
            handler=self._list_directory
        )
        
        self.register_tool(
            name="search_code",
            description="Search code using ripgrep",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search pattern"},
                    "path": {"type": "string", "description": "Search directory"},
                    "file_pattern": {"type": "string", "description": "File pattern to match"}
                },
                "required": ["query"]
            },
            handler=self._search_code
        )
        
        self.register_tool(
            name="execute_command",
            description=(
                "Execute a shell command. To run it in a subdirectory, pass cwd -- "
                "do not chain `cd <dir> && ...` into the command string."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command to execute"},
                    "cwd": {
                        "type": "string",
                        "description": (
                            "Directory to run the command in, relative to the workspace root "
                            "(or absolute). Defaults to the workspace root."
                        ),
                    },
                    "timeout": {"type": "integer", "description": "Timeout in seconds"}
                },
                "required": ["command"]
            },
            handler=self._execute_command
        )
        
        self.register_tool(
            name="get_git_info",
            description="Get git repository information",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Repository path"}
                }
            },
            handler=self._get_git_info
        )

        self.register_tool(
            name="browser",
            description=(
                "Use a clean headless Playwright session to navigate a public page, extract or interact "
                "with elements, test mobile scrolling, and capture a real PNG screenshot"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Absolute public http(s) URL"},
                    "action": {
                        "type": "string",
                        "enum": ["navigate", "extract", "click", "fill_form", "scroll", "screenshot"],
                    },
                    "selector": {"type": "string"},
                    "form_data": {"type": "object", "additionalProperties": {"type": "string"}},
                    "submit_selector": {"type": "string"},
                    "viewport_width": {"type": "integer", "minimum": 320, "maximum": 3840},
                    "viewport_height": {"type": "integer", "minimum": 480, "maximum": 2160},
                    "wait_for_selector": {"type": "string"},
                    "wait_after_load_ms": {"type": "integer", "minimum": 0, "maximum": 5000},
                    "scroll_y": {"type": "integer"},
                    "full_page": {"type": "boolean"},
                    "screenshot_selector": {"type": "string"},
                    "screenshot_name": {"type": "string"},
                },
                "required": ["url", "action"],
            },
            handler=self._browser,
        )
    
    def register_tool(self, name: str, description: str, 
                      parameters: Dict[str, Any], handler: Callable):
        """Register a tool"""
        self.tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler
        )
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools"""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            }
            for tool in self.tools.values()
        ]

    def tool_schemas_openai(self, names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Wrap registered tools in the `{"type":"function","function":{...}}`
        envelope a chat-completions `tools=[...]` payload needs. `names`
        restricts to a subset (e.g. read-only tools for a lower-trust mode);
        omit for the full registered set."""
        selected = names if names is not None else list(self.tools.keys())
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self.tools.values()
            if tool.name in selected
        ]

    async def list_tools_async(self, include_shared: bool = True) -> List[Dict[str, Any]]:
        """List native CLI tools and tools discovered by the shared MCP hub."""
        tools = self.list_tools()
        if not include_shared:
            return tools
        bridge = None
        owns_bridge = False
        try:
            bridge = _get_shared_mcp_bridge()
            if bridge is None:
                raise RuntimeError("Shared MCP bridge unavailable outside a monorepo checkout")
            if not bridge.available:
                owns_bridge = True
                await bridge.initialize(background=False)
            shared = await bridge.list_tools()
            tools.extend({**tool, "source": "shared_mcp"} for tool in shared)
        except Exception as exc:
            tools.append({
                "name": "shared_mcp",
                "description": f"Shared MCP registry unavailable: {exc}",
                "parameters": {},
                "available": False,
            })
        finally:
            # ``tamfis-code tools list`` runs in a short asyncio.run() loop.
            # Explicitly close any MCP processes this invocation opened; leaving
            # their transports for loop finalization caused the CLI to hang.
            if owns_bridge and bridge is not None:
                await bridge.shutdown()
        return tools
    
    async def call_tool(self, name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool by name"""
        if name not in self.tools:
            bridge = None
            owns_bridge = False
            try:
                bridge = _get_shared_mcp_bridge()
                if bridge is None:
                    raise RuntimeError("Shared MCP bridge unavailable outside a monorepo checkout")
                if not bridge.available:
                    owns_bridge = True
                    await bridge.initialize(background=False)
                result = await bridge.call_tool(name, parameters)
                success = bool(result.get("success")) and not result.get("is_error")
                return {
                    "result": result,
                    "tool": name,
                    "source": "shared_mcp",
                    "success": success,
                    **({"error": result.get("error_message") or result.get("error")}
                       if not success else {}),
                }
            except Exception as exc:
                return {
                    "error": f"Shared MCP tool unavailable: {exc}",
                    "tool": name,
                    "source": "shared_mcp",
                    "success": False,
                }
            finally:
                if owns_bridge and bridge is not None:
                    await bridge.shutdown()
        
        tool = self.tools[name]
        try:
            result = await tool.handler(**parameters)
            return {"result": result, "tool": name, "success": True}
        except Exception as e:
            return {"error": str(e), "tool": name, "success": False}
    
    async def _read_file(self, path: str) -> str:
        # Resolve path relative to current working directory
        p = Path(path)
        if not p.is_absolute():
            p = Path.cwd() / p
        p = p.resolve()
        if not p.exists():
            return f"Error: File '{path}' not found"
        if not p.is_file():
            return f"Error: '{path}' is not a file"
        return p.read_text(encoding='utf-8', errors='ignore')
    
    def _resolve_in_workspace(self, path: str) -> Path:
        """Resolve `path` against workspace_root (or cwd if none was given),
        raising if it escapes the workspace boundary. Only enforced when
        `self.workspace_root` is set -- see __init__'s docstring on why
        legacy no-arg callers get today's unrestricted behaviour instead."""
        base = Path(self.workspace_root) if self.workspace_root else Path.cwd()
        p = Path(path)
        if not p.is_absolute():
            p = base / p
        resolved = p.resolve()
        if self.workspace_root:
            root = base.resolve()
            if resolved != root and root not in resolved.parents:
                raise PermissionError(f"'{path}' resolves outside the workspace root ({root})")
        return resolved

    async def _write_file(self, path: str, content: str) -> str:
        p = self._resolve_in_workspace(path)
        original_content = p.read_text(encoding="utf-8", errors="ignore") if p.is_file() else None
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding='utf-8')
        if not p.exists():
            return f"❌ Failed to write to '{path}'"
        if self.session_id is not None:
            from .safety import record_mutation
            record_mutation(
                self.session_id, path=str(p), operation="create" if original_content is None else "update",
                original_content=original_content, new_content=content,
            )
        return f"✅ Successfully wrote {len(content)} bytes to '{path}'"

    async def _edit_file(self, path: str, old_string: str, new_string: str) -> str:
        p = self._resolve_in_workspace(path)
        if not p.is_file():
            return f"❌ Error: File '{path}' not found"
        original_content = p.read_text(encoding="utf-8", errors="ignore")
        occurrences = original_content.count(old_string)
        if occurrences == 0:
            return f"❌ Error: old_string not found in '{path}' -- no changes made"
        if occurrences > 1:
            return (
                f"❌ Error: old_string matches {occurrences} times in '{path}' -- it must be unique. "
                "Include more surrounding context to disambiguate."
            )
        new_content = original_content.replace(old_string, new_string, 1)
        p.write_text(new_content, encoding="utf-8")
        if self.session_id is not None:
            from .safety import record_mutation
            record_mutation(
                self.session_id, path=str(p), operation="update",
                original_content=original_content, new_content=new_content,
            )
        return f"✅ Edited '{path}'"
    
    async def _list_directory(self, path: str = ".") -> List[Dict[str, Any]]:
        p = Path(path)
        if not p.exists():
            return [{"error": f"Directory '{path}' not found"}]
        if not p.is_dir():
            return [{"error": f"'{path}' is not a directory"}]
        
        results = []
        for item in p.iterdir():
            results.append({
                "name": item.name,
                "path": str(item),
                "is_file": item.is_file(),
                "is_dir": item.is_dir(),
                "size": item.stat().st_size if item.exists() else 0,
                "modified": item.stat().st_mtime if item.exists() else 0,
            })
        return sorted(results, key=lambda x: x['name'])
    
    async def _search_code(self, query: str, path: str = ".", file_pattern: str = None) -> List[Dict[str, Any]]:
        try:
            cmd = ['rg', '--json', '--line-number', '--no-heading', query, path]
            if file_pattern:
                cmd.extend(['--glob', file_pattern])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            matches = []
            
            for line in result.stdout.split('\n'):
                if line.strip():
                    try:
                        data = json.loads(line)
                        if data.get('type') == 'match':
                            matches.append({
                                'file': data['data']['path']['text'],
                                'line': data['data']['line_number'],
                                'content': data['data']['lines']['text'].strip(),
                            })
                    except json.JSONDecodeError:
                        continue
            
            return matches
        except subprocess.TimeoutExpired:
            return [{"error": "Search timed out"}]
        except FileNotFoundError:
            return [{"error": "ripgrep (rg) not installed"}]
    
    async def _execute_command(self, command: str, cwd: Optional[str] = None, timeout: int = 60) -> Dict[str, Any]:
        try:
            run_dir = self._resolve_in_workspace(cwd) if cwd else None
        except PermissionError as e:
            return {"error": str(e), "success": False}
        if run_dir is not None and not run_dir.is_dir():
            return {"error": f"cwd '{cwd}' is not a directory", "success": False}
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(run_dir) if run_dir is not None else None,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return {
                "stdout": stdout.decode('utf-8', errors='ignore'),
                "stderr": stderr.decode('utf-8', errors='ignore'),
                "return_code": proc.returncode,
                "success": proc.returncode == 0
            }
        except asyncio.TimeoutError:
            return {"error": f"Command timed out after {timeout} seconds", "success": False}
        except Exception as e:
            return {"error": str(e), "success": False}
    
    async def _get_git_info(self, path: str = ".") -> Dict[str, Any]:
        p = Path(path)
        if not p.exists():
            return {"error": f"Path '{path}' not found"}
        
        info = {"path": str(p)}
        
        # Check if it's a git repo
        git_dir = p / ".git"
        if not git_dir.exists():
            info["is_git_repo"] = False
            return info
        
        info["is_git_repo"] = True
        
        try:
            # Get current branch
            result = subprocess.run(
                ['git', '-C', str(p), 'rev-parse', '--abbrev-ref', 'HEAD'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                info["branch"] = result.stdout.strip()
            
            # Get remote URL
            result = subprocess.run(
                ['git', '-C', str(p), 'config', '--get', 'remote.origin.url'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                info["remote_url"] = result.stdout.strip()
            
            # Get latest commit
            result = subprocess.run(
                ['git', '-C', str(p), 'log', '-1', '--format=%H%n%s%n%an%n%ae%n%ad'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                if len(lines) >= 5:
                    info["latest_commit"] = {
                        "hash": lines[0],
                        "message": lines[1],
                        "author": lines[2],
                        "email": lines[3],
                        "date": lines[4],
                    }
            
            # Get status
            result = subprocess.run(
                ['git', '-C', str(p), 'status', '--porcelain'],
                capture_output=True, text=True
            )
            info["has_changes"] = bool(result.stdout.strip())
            info["changed_files"] = len([l for l in result.stdout.split('\n') if l.strip()])
            
        except Exception as e:
            info["git_error"] = str(e)
        
        return info

    async def _browser(self, **parameters: Any) -> Dict[str, Any]:
        """Public-web browser facade for ``tamfis-code tools call``.

        The agentic Remote path injects trusted task context separately and
        can therefore test loopback development servers. This direct facade
        intentionally receives no trusted fields, so BrowserTool keeps its
        public-only SSRF boundary.
        """
        browser_tool = get_browser_tool_class()
        if browser_tool is None:
            raise RuntimeError("BrowserTool unavailable outside a monorepo checkout")
        result = await browser_tool().execute_async(**parameters)
        if not result.get("success"):
            raise RuntimeError(str(result.get("error") or "Browser action failed"))
        return result

# Convenience function for CLI use
async def call_tool(name: str, **kwargs):
    """Call a tool with given parameters"""
    server = MCPServer()
    return await server.call_tool(name, kwargs)
