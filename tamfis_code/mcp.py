"""Model Context Protocol (MCP) integration for tools"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
import html
import json
import os
import re
import subprocess
import fnmatch
import asyncio
import sys
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path

import httpx
from rich.panel import Panel

from .render import resume_live_if_active, suspend_live_if_active

# web_search (see MCPServer._web_search) is self-contained rather than
# reusing tamgpt6's WebSearchManager via _import_monorepo_attr, unlike
# browser (which needs Playwright's real headless browser binary and is
# only meaningful when co-located). A plain search-API HTTP call is cheap
# enough to implement natively, so tamfis-code keeps a working web_search
# tool when installed standalone on a machine that never had tamgpt6 on it
# at all -- confirmed as the right call by the user (portability over
# reuse), matching the same "worldwide-installable" bar already applied to
# config/state paths (see config.resolve_config_dir).
_TAVILY_SEARCH_ENDPOINT = "https://api.tavily.com/search"
_DUCKDUCKGO_HTML_ENDPOINT = "https://html.duckduckgo.com/html/"
_DUCKDUCKGO_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
}
_DDG_RESULT_RE = re.compile(
    r'<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL
)
_DDG_SNIPPET_RE = re.compile(r'<a class="result__snippet"[^>]*>(.*?)</a>', re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _parse_duckduckgo_html(html_text: str, max_results: int) -> List[Dict[str, str]]:
    """Parse DuckDuckGo's HTML-only search endpoint into structured results.

    No API key required -- this is the always-available fallback (and,
    absent TAVILY_API_KEY, the only provider) for MCPServer._web_search.
    """
    links = _DDG_RESULT_RE.findall(html_text)
    snippets = _DDG_SNIPPET_RE.findall(html_text)
    results: List[Dict[str, str]] = []
    for i in range(min(len(links), max_results)):
        url, title = links[i]
        snippet = snippets[i] if i < len(snippets) else ""
        snippet = html.unescape(_HTML_TAG_RE.sub("", snippet))
        snippet = re.sub(r"\s+", " ", snippet).strip()
        title = html.unescape(_HTML_TAG_RE.sub("", title)).strip()
        results.append({
            "title": title or "Untitled",
            "url": url.strip(),
            "snippet": snippet[:400],
        })
    return results


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
    ancestors = [Path.cwd(), *Path(__file__).resolve().parents]
    candidates = list(ancestors)
    # tamfis-code is commonly installed as a SIBLING of a tamgpt6 monorepo
    # checkout (.../tamgpt6 and .../tamfis-code side by side) rather than
    # nested inside it -- the walk-upward search above only ever finds a
    # monorepo tamfis-code happens to be running from inside of. Also check
    # each ancestor's "tamgpt6" child, and an explicit override, so the
    # common sibling-checkout layout (confirmed live: this environment's
    # own layout) is actually found instead of always reporting unavailable.
    env_root = os.environ.get("TAMFIS_MONOREPO_ROOT")
    if env_root:
        candidates.insert(0, Path(env_root))
    candidates.extend(ancestor / "tamgpt6" for ancestor in ancestors)
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


# Directory names never descended into or enumerated by list_directory/
# search_code, regardless of what path they're invoked against. This is
# tool-execution-layer enforcement, not just prompt guidance: a scoped path
# argument (see runner_local.py's _detect_workspace_scope) only controls
# WHICH directory a tool targets, not how much noise it returns once inside
# it -- a single unfiltered `rg`/iterdir() over a real project can still
# return thousands of node_modules/build/.git entries with no scope rule
# involved at all.
EXCLUDED_DIR_NAMES = {
    ".git", "node_modules", "dist", "build", "coverage", ".pytest_cache",
    "__pycache__", ".venv", "venv", "vendor", "target", "logs", "archives",
    ".mypy_cache", ".ruff_cache", ".tox", ".nox", "htmlcov", ".next",
    ".turbo", ".cache", "site-packages",
}
MAX_LIST_DIRECTORY_ENTRIES = 500
MAX_SEARCH_RESULTS = 200
# Files larger than this are skipped by search_code -- a single huge
# (often generated/minified) file can otherwise dominate the whole result
# set with one or two enormous match lines.
MAX_SEARCH_FILE_SIZE_BYTES = 2_000_000
MAX_SEARCH_MATCH_CHARS = 500

@dataclass
class ToolDefinition:
    """Definition of a tool for MCP"""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema
    handler: Optional[Callable] = None

class MCPServer:
    """MCP server for tool execution"""

    def __init__(
        self, *, workspace_root: Optional[str] = None, session_id: Optional[int] = None,
        console: Optional[Any] = None, renderer: Optional[Any] = None, interactive: bool = False,
        transaction_id: Optional[str] = None,
        attachment_paths: Optional[List[str]] = None,
    ):
        # workspace_root/session_id are optional so existing callers that
        # construct MCPServer() with no arguments (tests, the `tools`/
        # `screenshot` debug commands) keep today's behaviour: no boundary
        # enforcement on write_file/edit_file, no mutation-ledger recording.
        # The standalone agent loop (runner_local.py) always supplies both.
        self.workspace_root = workspace_root
        self.session_id = session_id
        # Explicit CLI attachments are readable inputs, not extra writable
        # workspaces. Only these exact files are admitted; their parent
        # directories never become browsable and every output still has to
        # resolve inside workspace_root.
        self.attachment_paths = {
            Path(item).expanduser().resolve() for item in (attachment_paths or [])
        }
        # One id per turn (runner_local.py mints it once per
        # run_local_agent_turn call) -- groups every mutation this server
        # instance records so a whole turn's file changes can later be
        # reverted together via safety.revert_transaction(), not just one
        # mutation_id at a time. None for any caller that doesn't pass one
        # (tests, debug commands) -- record_mutation still works, the
        # mutation just isn't part of any group.
        self.transaction_id = transaction_id
        # console/renderer/interactive back ask_user_question only -- optional
        # and default to "unavailable" so every other existing caller (tests,
        # `tools`/`screenshot` debug commands, anything constructing
        # MCPServer() bare) is unaffected. `interactive` defaults False (not
        # inherited from the real terminal) so a caller must opt in
        # explicitly, the same fail-safe-closed default `resolve_approval_decision`
        # already uses for `interactive`.
        self._console = console
        self._renderer = renderer
        self._interactive = interactive
        # Per-server, per-root temporary indexes keep find_references
        # incremental across repeated calls without writing cache files into
        # either the user's repository or home directory. TemporaryDirectory
        # owns cleanup when this MCPServer/turn is released.
        self._symbol_index_dirs: Dict[str, tempfile.TemporaryDirectory] = {}
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
            name="find_references",
            description=(
                "Find where a symbol (function/class/variable name) is DEFINED (via the code "
                "index) and every line across the codebase that references it (a whole-word "
                "search, not a substring match) -- use this instead of read_file/search_code "
                "guesswork to find all call sites and definitions of a symbol before renaming "
                "or changing it, or before assuming you already know where something is used."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Exact symbol name to look up (case-sensitive, whole word)"},
                    "path": {"type": "string", "description": "Directory to search (defaults to the whole workspace)"},
                },
                "required": ["symbol"],
            },
            handler=self._find_references,
        )

        self.register_tool(
            name="extract_archive",
            description=(
                "Safely extract a ZIP or TAR variant inside the workspace, preserving binary files. "
                "Use this before inspecting or editing an uploaded/archive project. Traversal paths, "
                "symlinks, archive bombs, and destinations outside the workspace are rejected."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "ZIP/TAR archive path inside the workspace"},
                    "destination": {"type": "string", "description": "Optional extraction directory inside the workspace"},
                },
                "required": ["path"],
            },
            handler=self._extract_archive,
        )

        self.register_tool(
            name="repackage_archive",
            description=(
                "Create a ZIP or TAR variant from a workspace directory after its files were analysed/updated. "
                "The output stays inside the workspace and is returned as a real artifact path."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "source_dir": {"type": "string", "description": "Directory to package inside the workspace"},
                    "output_path": {"type": "string", "description": "Output .zip/.tar/.tar.gz/.tgz/.tar.bz2/.tar.xz path inside the workspace"},
                },
                "required": ["source_dir", "output_path"],
            },
            handler=self._repackage_archive,
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
                    "timeout": {"type": "integer", "description": "Timeout in seconds"},
                    "environment": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "Environment variable overrides for this command only"
                    },
                    "shell": {
                        "type": "string",
                        "enum": ["bash", "sh"],
                        "description": "Shell used to execute the command"
                    },
                    "approval_metadata": {"type": "object", "description": "Caller approval/audit metadata"}
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

        self.register_tool(
            name="web_search",
            description=(
                "Search the public web for current information not available in this "
                "repository or from training data alone -- news, current prices/releases, "
                "documentation for a library, error messages, anything time-sensitive or "
                "external. Returns a short list of results with title, URL, and snippet. "
                "Read-only, no side effects. Uses Tavily if TAVILY_API_KEY is configured, "
                "else falls back to DuckDuckGo automatically -- no configuration required."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "Maximum number of results to return (default 5)",
                    },
                },
                "required": ["query"],
            },
            handler=self._web_search,
        )

        self.register_tool(
            name="ask_user_question",
            description=(
                "Pause and ask the human at the terminal a direct clarifying question when you "
                "are genuinely uncertain about something only they can resolve -- e.g. which of "
                "two conflicting conventions to follow, which of several ambiguous targets they "
                "mean, or confirming a stated fact you cannot verify with a tool (project type, "
                "intended scope, which environment). Do not use this for anything answerable by "
                "reading files or running a tool yourself -- investigate first. Only available in "
                "a real interactive terminal session; if unavailable, proceed on your best "
                "judgement and say what you assumed."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The question to ask, in plain language"},
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional short list of suggested answers -- the user may still type something else",
                    },
                },
                "required": ["question"],
            },
            handler=self._ask_user_question,
        )

    async def _ask_user_question(self, question: str, options: Optional[List[str]] = None) -> str:
        if self._console is None or not self._interactive:
            return (
                "ask_user_question is unavailable in this session (no attached interactive "
                "terminal) -- proceed using your best judgement from the evidence already "
                "gathered, and clearly state what you assumed in your final answer."
            )
        # Same ordering discipline as the approval-gate panel (see safety.py's
        # module docstring / STATUS.md's v0.4.5 fix): suspend the live status
        # line before the panel prints, not just before the blocking input
        # call, so a stray background redraw can never land between them.
        suspend_live_if_active(self._renderer)
        try:
            self._console.print(Panel(question, title="Question from the agent", border_style="cyan", expand=False))
            if options:
                for index, option in enumerate(options, start=1):
                    self._console.print(f"  {index}. {option}")
                raw = self._console.input(
                    "Your answer (type a number above, or free text): "
                ).strip()
                if raw.isdigit() and 1 <= int(raw) <= len(options):
                    return options[int(raw) - 1]
                return raw or "(no answer given)"
            return self._console.input("Your answer: ").strip() or "(no answer given)"
        finally:
            resume_live_if_active(self._renderer)

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
        p = self._resolve_readable_input(path)
        if not p.exists():
            return f"Error: File '{path}' not found"
        if not p.is_file():
            return f"Error: '{path}' is not a file"
        # A null byte anywhere in the first 8000 bytes is the same
        # binary-detection heuristic `file`/git use -- without this,
        # read_text(errors='ignore') silently drops every invalid byte and
        # hands back plausible-looking garbage instead of an error, which
        # is worse than failing loudly (confirmed while wiring real image
        # attachments: the old behaviour would have let a model call
        # read_file on an attached PNG and "read" mangled nonsense as if it
        # were the image's real content).
        try:
            with p.open("rb") as fh:
                prefix = fh.read(8000)
        except OSError as e:
            return f"Error: could not read '{path}' ({e})"
        if b"\x00" in prefix:
            return (
                f"Error: '{path}' looks like a binary file (a null byte was found in its first "
                "8000 bytes) -- read_file only supports text. If this is an attached image, its "
                "content is already included directly in this conversation for vision-capable "
                "models -- look at it there instead of calling read_file. For an archive, use "
                "extract_archive."
            )
        return p.read_text(encoding='utf-8', errors='ignore')

    def _resolve_readable_input(self, path: str) -> Path:
        """Resolve a workspace file or one exact, user-supplied attachment.

        This also closes a longstanding boundary gap where read_file used
        cwd directly and could read arbitrary absolute paths even while all
        write tools were workspace-confined.
        """
        try:
            return self._resolve_in_workspace(path)
        except PermissionError:
            candidate = Path(path).expanduser()
            if not candidate.is_absolute():
                raise
            resolved = candidate.resolve()
            if resolved in self.attachment_paths:
                return resolved
            raise
    
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
                transaction_id=self.transaction_id,
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
                transaction_id=self.transaction_id,
            )
        return f"✅ Edited '{path}'"
    
    async def _list_directory(self, path: str = ".") -> List[Dict[str, Any]]:
        try:
            p = self._resolve_in_workspace(path)
        except PermissionError as exc:
            return [{"error": str(exc)}]
        if not p.exists():
            return [{"error": f"Directory '{path}' not found"}]
        if not p.is_dir():
            return [{"error": f"'{path}' is not a directory"}]

        results = []
        excluded_count = 0
        for item in p.iterdir():
            if item.is_dir() and item.name in EXCLUDED_DIR_NAMES:
                excluded_count += 1
                continue
            results.append({
                "name": item.name,
                "path": str(item),
                "is_file": item.is_file(),
                "is_dir": item.is_dir(),
                "size": item.stat().st_size if item.exists() else 0,
                "modified": item.stat().st_mtime if item.exists() else 0,
            })
        results = sorted(results, key=lambda x: x['name'])
        total = len(results)
        if total > MAX_LIST_DIRECTORY_ENTRIES:
            results = results[:MAX_LIST_DIRECTORY_ENTRIES]
            results.append({
                "truncated": True,
                "note": f"{total - MAX_LIST_DIRECTORY_ENTRIES} more entrie(s) omitted "
                        f"(showing first {MAX_LIST_DIRECTORY_ENTRIES} of {total}). "
                        "Narrow the path or use search_code for a targeted query.",
            })
        if excluded_count:
            results.append({
                "excluded": True,
                "note": f"{excluded_count} ignored subdirectory name(s) not listed "
                        f"({', '.join(sorted(EXCLUDED_DIR_NAMES))}, when present).",
            })
        return results

    async def _search_code(self, query: str, path: str = ".", file_pattern: str = None) -> List[Dict[str, Any]]:
        try:
            resolved_path = self._resolve_in_workspace(path)
        except PermissionError as exc:
            return [{"error": str(exc)}]
        try:
            cmd = [
                'rg', '--json', '--line-number', '--no-heading',
                '--max-filesize', str(MAX_SEARCH_FILE_SIZE_BYTES),
                # Per-file match cap keeps one pathological file (e.g. a huge
                # generated table) from consuming the whole result budget by
                # itself; the total cap below still applies across all files.
                '--max-count', str(MAX_SEARCH_RESULTS),
            ]
            for name in sorted(EXCLUDED_DIR_NAMES):
                cmd.extend(['--glob', f'!**/{name}/**'])
            cmd.extend([query, str(resolved_path)])
            if file_pattern:
                cmd.extend(['--glob', file_pattern])

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            matches = []

            for line in result.stdout.split('\n'):
                if not line.strip():
                    continue
                if len(matches) >= MAX_SEARCH_RESULTS:
                    break
                try:
                    data = json.loads(line)
                    if data.get('type') == 'match':
                        content = data['data']['lines']['text'].strip()
                        if len(content) > MAX_SEARCH_MATCH_CHARS:
                            content = content[:MAX_SEARCH_MATCH_CHARS] + f"...[{len(content) - MAX_SEARCH_MATCH_CHARS} chars omitted]"
                        matches.append({
                            'file': data['data']['path']['text'],
                            'line': data['data']['line_number'],
                            'content': content,
                        })
                except json.JSONDecodeError:
                    continue

            if len(matches) >= MAX_SEARCH_RESULTS:
                matches.append({
                    "truncated": True,
                    "note": f"Showing the first {MAX_SEARCH_RESULTS} matches; the search "
                            "produced more. Narrow the query (more specific pattern, a "
                            "file_pattern glob, or a deeper path) instead of relying on "
                            "the full result set.",
                })

            return matches
        except subprocess.TimeoutExpired:
            return [{"error": "Search timed out"}]
        except FileNotFoundError:
            # `rg` is fast and preferred, but it is not part of Python and is
            # absent from some minimal servers and hosted CI images. A
            # portable install must retain search functionality without a
            # host-specific binary, so use the same bounds and exclusions in
            # a small standard-library fallback.
            return self._search_code_python(query, resolved_path, file_pattern)

    @staticmethod
    def _search_code_python(query: str, root: Path, file_pattern: Optional[str]) -> List[Dict[str, Any]]:
        try:
            matcher = re.compile(query)
        except re.error as exc:
            return [{"error": f"Invalid search pattern: {exc}"}]
        matches: List[Dict[str, Any]] = []
        try:
            paths = [root] if root.is_file() else sorted(root.rglob("*"))
            for candidate in paths:
                if len(matches) >= MAX_SEARCH_RESULTS:
                    break
                if not candidate.is_file() or candidate.stat().st_size > MAX_SEARCH_FILE_SIZE_BYTES:
                    continue
                relative_parts = candidate.relative_to(root if root.is_dir() else root.parent).parts
                if any(part in EXCLUDED_DIR_NAMES for part in relative_parts[:-1]):
                    continue
                if file_pattern and not fnmatch.fnmatch(candidate.name, file_pattern) and not fnmatch.fnmatch(str(candidate), file_pattern):
                    continue
                try:
                    with candidate.open("r", encoding="utf-8", errors="replace") as handle:
                        for line_number, line in enumerate(handle, 1):
                            if not matcher.search(line):
                                continue
                            content = line.strip()
                            if len(content) > MAX_SEARCH_MATCH_CHARS:
                                content = content[:MAX_SEARCH_MATCH_CHARS] + f"...[{len(content) - MAX_SEARCH_MATCH_CHARS} chars omitted]"
                            matches.append({"file": str(candidate), "line": line_number, "content": content})
                            if len(matches) >= MAX_SEARCH_RESULTS:
                                break
                except (OSError, UnicodeError):
                    continue
        except OSError as exc:
            return [{"error": str(exc)}]
        if len(matches) >= MAX_SEARCH_RESULTS:
            matches.append({
                "truncated": True,
                "note": f"Showing the first {MAX_SEARCH_RESULTS} matches; narrow the query or path.",
            })
        return matches

    async def _find_references(self, symbol: str, path: str = ".") -> Dict[str, Any]:
        """Real cross-file reference resolution: where `symbol` is defined
        (via CodeIndexer's symbol table) plus every line across the
        codebase that mentions it as a whole word (via _search_code, reused
        rather than reimplemented). Distinct from references.py's
        ReferenceResolver, an unrelated older feature that inlines @file/
        @folder mentions typed directly into a prompt -- this is the
        find-usages/go-to-definition tool the model can call mid-turn that
        was previously missing under any name."""
        symbol = (symbol or "").strip()
        if not symbol:
            return {"error": "symbol is required", "success": False}

        definitions: List[Dict[str, Any]] = []
        try:
            root = self._resolve_in_workspace(path)
        except (PermissionError, OSError):
            root = None
        if root is not None and root.is_dir():
            try:
                from .indexer import CodeIndexer
                # find_references is read-only. Reuse a turn-local temporary
                # index for this root so unchanged files are not reparsed on
                # every tool call, while keeping the cache out of both the
                # workspace and ~/.tamfis.
                root_key = str(root.resolve())
                temp_index = self._symbol_index_dirs.get(root_key)
                if temp_index is None:
                    temp_index = tempfile.TemporaryDirectory(prefix="tamfis-symbol-index-")
                    self._symbol_index_dirs[root_key] = temp_index
                indexer = CodeIndexer(root, index_path=Path(temp_index.name))
                indexer.index()
                definitions = [
                    {"name": sym.name, "kind": sym.kind, "file": sym.file_path, "line": sym.line_start}
                    for sym in indexer.search_symbol(symbol)
                    if sym.name == symbol  # search_symbol matches substrings; only exact names are real definitions
                ]
            except Exception:
                pass  # indexing is best-effort -- the reference search below still works standalone

        references = await self._search_code(rf"\b{re.escape(symbol)}\b", path=path)
        clean_references = [r for r in references if isinstance(r, dict) and "error" not in r and not r.get("truncated")]
        truncated = any(isinstance(r, dict) and r.get("truncated") for r in references)
        return {
            "symbol": symbol,
            "definitions": definitions,
            "references": clean_references,
            "reference_count": len(clean_references),
            "truncated": truncated,
            "success": True,
        }

    @staticmethod
    def _archive_suffix(path: str) -> Optional[str]:
        lower = str(path or "").lower()
        suffixes = (".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz", ".tar", ".zip")
        return next((suffix for suffix in suffixes if lower.endswith(suffix)), None)

    @staticmethod
    def _safe_archive_member(name: str) -> Optional[str]:
        import posixpath

        normalized = posixpath.normpath(str(name or "").replace("\\", "/"))
        if not normalized or normalized in {".", ".."}:
            return None
        if normalized.startswith("/") or normalized.startswith("../"):
            return None
        return normalized

    async def _extract_archive(self, path: str, destination: Optional[str] = None) -> Dict[str, Any]:
        source = self._resolve_readable_input(path)
        if not source.is_file():
            raise FileNotFoundError(f"Archive not found: {path}")
        suffix = self._archive_suffix(source.name)
        if suffix is None:
            raise ValueError("Only ZIP and TAR archive variants are supported")
        default_name = source.name[:-len(suffix)] + "_extracted"
        target_root = self._resolve_in_workspace(destination or default_name)
        if target_root.exists() and (not target_root.is_dir() or any(target_root.iterdir())):
            raise FileExistsError(f"Extraction destination must be absent or empty: {target_root}")
        max_files = 5000
        max_bytes = 250 * 1024 * 1024
        written: List[str] = []
        total = 0
        target_root.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".tamfis-extract-", dir=target_root.parent))
        try:
            if suffix == ".zip":
                with zipfile.ZipFile(source, "r") as archive:
                    all_members = archive.infolist()
                    for item in all_members:
                        relative = self._safe_archive_member(item.filename)
                        is_symlink = ((item.external_attr >> 16) & 0o170000) == 0o120000
                        if relative is None or is_symlink:
                            raise ValueError(f"Unsafe archive member rejected: {item.filename}")
                    members = [item for item in all_members if not item.is_dir()]
                    if len(members) > max_files or sum(item.file_size for item in members) > max_bytes:
                        raise ValueError("Archive exceeds the 5,000-file or 250 MB expanded-size limit")
                    for item in members:
                        relative = self._safe_archive_member(item.filename)
                        assert relative is not None
                        output = staging.joinpath(*relative.split("/"))
                        output.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(item, "r") as incoming, output.open("wb") as outgoing:
                            shutil.copyfileobj(incoming, outgoing)
                        total += output.stat().st_size
                        written.append(relative)
            else:
                with tarfile.open(source, "r:*") as archive:
                    all_members = archive.getmembers()
                    for item in all_members:
                        relative = self._safe_archive_member(item.name)
                        if item.isdir() and str(item.name or "").replace("\\", "/").rstrip("/") in {"", "."}:
                            continue
                        if relative is None or item.issym() or item.islnk() or not (item.isfile() or item.isdir()):
                            raise ValueError(f"Unsafe archive member rejected: {item.name}")
                    members = [item for item in all_members if item.isfile()]
                    if len(members) > max_files or sum(item.size for item in members) > max_bytes:
                        raise ValueError("Archive exceeds the 5,000-file or 250 MB expanded-size limit")
                    for item in members:
                        relative = self._safe_archive_member(item.name)
                        assert relative is not None
                        incoming = archive.extractfile(item)
                        if incoming is None:
                            raise ValueError(f"Could not read archive member: {item.name}")
                        output = staging.joinpath(*relative.split("/"))
                        output.parent.mkdir(parents=True, exist_ok=True)
                        with incoming, output.open("wb") as outgoing:
                            shutil.copyfileobj(incoming, outgoing)
                        total += output.stat().st_size
                        written.append(relative)
            if target_root.exists():
                target_root.rmdir()  # already verified empty above
            os.replace(staging, target_root)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return {
            "success": True, "operation": "extract_archive", "source": str(source),
            "destination": str(target_root), "file_count": len(written),
            "expanded_bytes": total, "files": written[:500], "truncated": len(written) > 500,
        }

    async def _repackage_archive(self, source_dir: str, output_path: str) -> Dict[str, Any]:
        source = self._resolve_in_workspace(source_dir)
        output = self._resolve_in_workspace(output_path)
        if not source.is_dir():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")
        suffix = self._archive_suffix(output.name)
        if suffix is None:
            raise ValueError("Output must use a ZIP or TAR archive suffix")
        if output == source or source in output.parents:
            raise ValueError("Output archive must be outside the source directory to avoid packaging itself")
        entries = list(source.rglob("*"))
        symlinks = [item for item in entries if item.is_symlink()]
        if symlinks:
            raise ValueError(f"Refusing to package symlink: {symlinks[0].relative_to(source)}")
        files = sorted(item for item in entries if item.is_file())
        if len(files) > 5000 or sum(item.stat().st_size for item in files) > 250 * 1024 * 1024:
            raise ValueError("Package exceeds the 5,000-file or 250 MB input limit")
        output.parent.mkdir(parents=True, exist_ok=True)
        temp_handle = tempfile.NamedTemporaryFile(prefix=".tamfis-package-", dir=output.parent, delete=False)
        temp_handle.close()
        temp_output = Path(temp_handle.name)
        try:
            if suffix == ".zip":
                with zipfile.ZipFile(temp_output, "w", zipfile.ZIP_DEFLATED) as archive:
                    for item in files:
                        archive.write(item, item.relative_to(source).as_posix())
            else:
                mode = {
                    ".tar.gz": "w:gz", ".tgz": "w:gz", ".tar.bz2": "w:bz2", ".tbz2": "w:bz2",
                    ".tar.xz": "w:xz", ".txz": "w:xz", ".tar": "w",
                }[suffix]
                with tarfile.open(temp_output, mode) as archive:
                    for item in files:
                        archive.add(item, arcname=item.relative_to(source).as_posix(), recursive=False)
            os.replace(temp_output, output)
        finally:
            temp_output.unlink(missing_ok=True)
        return {
            "success": True, "operation": "repackage_archive", "source_dir": str(source),
            "path": str(output), "filename": output.name, "size_bytes": output.stat().st_size,
            "file_count": len(files), "artifact_type": "archive",
        }
    
    async def _execute_command(
        self, command: str, cwd: Optional[str] = None, timeout: int = 60,
        environment: Optional[Dict[str, str]] = None, shell: str = "bash",
        approval_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # `timeout: int` above is only a type hint -- the tool schema
        # declares it as an integer, but nothing coerces a model's actual
        # tool-call arguments to match it. Confirmed live: a real turn sent
        # `"timeout": "300"` (a string) in the approval panel, which reached
        # asyncio.wait_for(timeout=...) unmodified and crashed with
        # "'<=' not supported between instances of 'str' and 'int'"
        # (asyncio's own internal timeout<=0 check) -- silently breaking
        # every execute_command call for the rest of that turn instead of
        # running the command. A model outputting a numeric field as a
        # string is a common tool-calling failure mode, not exotic.
        try:
            timeout = int(timeout)
        except (TypeError, ValueError):
            timeout = 60
        if timeout <= 0:
            timeout = 60
        try:
            # Commands always execute inside the approved workspace. Omitting
            # cwd means workspace_root, never the caller process's accidental
            # current directory (which may contain unrelated manifests).
            run_dir = self._resolve_in_workspace(cwd or ".")
        except PermissionError as e:
            return {"error": str(e), "success": False}
        if not run_dir.is_dir():
            return {"error": f"cwd '{cwd}' is not a directory", "success": False}

        first = command.strip().split(None, 1)[0] if command.strip() else ""
        first = Path(first).name
        manifest_rules = {
            "npm": ("package.json",), "npx": ("package.json",),
            "pnpm": ("package.json",), "yarn": ("package.json",),
            "cargo": ("Cargo.toml",), "go": ("go.mod",),
            "mvn": ("pom.xml",), "gradle": ("build.gradle", "build.gradle.kts"),
            "pip": ("pyproject.toml", "requirements.txt", "setup.py", "setup.cfg"),
            "pip3": ("pyproject.toml", "requirements.txt", "setup.py", "setup.cfg"),
        }
        required = manifest_rules.get(first)
        if required and not any((run_dir / name).is_file() for name in required):
            return {
                "error": (
                    f"Refusing to run '{first}' in {run_dir}: no local project manifest "
                    f"found ({', '.join(required)}). Parent-directory manifests are ignored."
                ),
                "success": False,
            }
        if shell not in {"bash", "sh"}:
            return {"error": f"Unsupported shell: {shell}", "success": False}
        env = os.environ.copy()
        # Same bug class as the timeout fix above: `environment: Optional[
        # Dict[str, str]]` is only a type hint. Live-reported crash --
        # `'str' object has no attribute 'items'` -- from a real tool call
        # that sent `environment` as something other than a real object
        # (e.g. a JSON-encoded string instead of an actual dict). Anything
        # that isn't actually a dict is treated as "no override" rather
        # than crashing the whole command.
        if isinstance(environment, dict):
            env.update({str(k): str(v) for k, v in environment.items()})
        try:
            proc = await asyncio.create_subprocess_exec(
                shell, "-lc", command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(run_dir), env=env,
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
        try:
            p = self._resolve_in_workspace(path)
        except PermissionError as exc:
            return {"error": str(exc)}
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

    async def _web_search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Self-contained public web search: Tavily primary if TAVILY_API_KEY
        is set, DuckDuckGo HTML fallback otherwise (no key required, always
        available). See the module-level comment above _parse_duckduckgo_html
        for why this doesn't reuse tamgpt6's WebSearchManager.
        """
        query = (query or "").strip()
        if not query:
            raise ValueError("web_search requires a non-empty query")
        try:
            max_results = int(max_results)
        except (TypeError, ValueError):
            max_results = 5
        max_results = max(1, min(max_results, 10))

        provider: Optional[str] = None
        results: List[Dict[str, str]] = []
        tavily_key = os.environ.get("TAVILY_API_KEY", "").strip()
        if tavily_key:
            try:
                async with httpx.AsyncClient(timeout=20.0) as client:
                    response = await client.post(
                        _TAVILY_SEARCH_ENDPOINT,
                        json={
                            "api_key": tavily_key,
                            "query": query,
                            "search_depth": "basic",
                            "include_answer": False,
                            "include_raw_content": False,
                            "include_images": False,
                            "max_results": max_results,
                        },
                    )
                if response.status_code == 200:
                    raw_results = (response.json() or {}).get("results") or []
                    if raw_results:
                        provider = "tavily"
                        results = [
                            {
                                "title": str(r.get("title") or "Untitled"),
                                "url": str(r.get("url") or ""),
                                "snippet": str(r.get("content") or "")[:500],
                            }
                            for r in raw_results[:max_results]
                        ]
            except (httpx.HTTPError, ValueError):
                pass  # falls through to DuckDuckGo below

        if not results:
            try:
                async with httpx.AsyncClient(timeout=20.0, headers=_DUCKDUCKGO_HEADERS) as client:
                    response = await client.post(
                        _DUCKDUCKGO_HTML_ENDPOINT, data={"q": query, "kl": "us-en"}
                    )
                if response.status_code == 200:
                    parsed = _parse_duckduckgo_html(response.text, max_results)
                    if parsed:
                        provider = "duckduckgo"
                        results = parsed
            except httpx.HTTPError:
                pass

        if not results:
            return {"query": query, "provider": None, "results": [], "message": "No results found."}
        return {"query": query, "provider": provider, "results": results}

# Convenience function for CLI use
async def call_tool(name: str, **kwargs):
    """Call a tool with given parameters"""
    server = MCPServer()
    return await server.call_tool(name, kwargs)
