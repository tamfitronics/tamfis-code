"""Model Context Protocol (MCP) integration for tools"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
import contextlib
import html
import json
import os
import re
import signal
import subprocess
import fnmatch
import asyncio
import sys
import shutil
import tarfile
import tempfile
import time
import uuid
import zipfile
from pathlib import Path

import httpx
from rich.panel import Panel

from .render import resume_live_if_active, suspend_live_if_active
from .sandbox import SandboxPolicy, build_sandbox_command
# MCP commands can be invoked without constructing a ProviderManager (for
# example, `tamfis-code tools list`). Reuse the canonical project `.env`
# loader here so TAMGPT_MCP_CONFIG and TAMFIS_MONOREPO_ROOT are available in
# that path too, while preserving already-exported environment variables.
from .providers import _load_project_env

_load_project_env()

# web_search (see MCPServer._web_search) is self-contained rather than
# reusing tamgpt6's WebSearchManager via _import_monorepo_attr, unlike
# browser. Both capabilities are implemented in this standalone package.
# A plain search-API HTTP call is cheap
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


def _sandbox_result(command: Any) -> Dict[str, Any]:
    if command is None:
        return {"active": False, "backend": "not-configured"}
    result: Dict[str, Any] = {"active": command.active, "backend": command.backend}
    if command.warning:
        result["warning"] = command.warning
    return result


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


def _get_shared_mcp_bridge(workspace_root: str | None = None):
    """Return Tamfis Code's standalone MCP client bridge."""
    from .mcp_client import StandaloneMCPBridge
    return StandaloneMCPBridge(workspace_root)


def get_browser_tool_class():
    """Return tamfis-code's portable Playwright browser implementation."""
    from .browser import PortableBrowserTool

    return PortableBrowserTool


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

# How much of a still-running (or already-finished) background job's own
# output read_background_job returns per call -- same bounded-tail idea as
# the rest of this module's output caps, so polling a chatty long-running
# command repeatedly can't blow the context budget.
MAX_BACKGROUND_OUTPUT_CHARS = 20_000


@dataclass
class BackgroundJob:
    """A command detached from execute_command's normal blocking wait (see
    _execute_command's background_signal) -- the real asyncio.subprocess.
    Process keeps running exactly as it was, not restarted under a
    different mechanism; only who's waiting on it changes.

    Registered at module level, not per-MCPServer-instance: MCPServer is
    recreated fresh every turn (see runner_local.py), but a backgrounded
    job legitimately needs to survive past the turn that started it -- the
    whole point is "keep working, check on this later," possibly several
    turns later.
    """
    job_id: str
    command: str
    cwd: str
    started_at: float
    proc: "asyncio.subprocess.Process"
    # The SAME communicate() call _execute_command already had in flight
    # when it detached -- must be awaited here, not re-issued. A second,
    # concurrent proc.communicate() call on top of the first would race it
    # for the same stdout/stderr pipes, which asyncio explicitly forbids.
    communicate_task: "asyncio.Task"
    stdout: str = ""
    stderr: str = ""
    return_code: Optional[int] = None
    finished: bool = False
    error: str = ""


_BACKGROUND_JOBS: Dict[str, BackgroundJob] = {}


async def _watch_background_job(job: BackgroundJob) -> None:
    """Keeps draining the detached process's already-in-flight communicate()
    after _execute_command has returned -- if nothing awaited it at all, an
    exited process becomes a zombie and stdout/stderr pipes can fill and
    deadlock the child. Fills in the job record for read_background_job to
    report once this completes."""
    try:
        stdout, stderr = await job.communicate_task
        job.stdout = stdout.decode("utf-8", errors="ignore")
        job.stderr = stderr.decode("utf-8", errors="ignore")
        job.return_code = job.proc.returncode
    except Exception as exc:
        job.error = str(exc)
    finally:
        job.finished = True


def read_background_job_status(job_id: str) -> Dict[str, Any]:
    """Module-level (not an MCPServer method -- see BackgroundJob's
    docstring on why the registry itself is module-level) lookup used by
    runner_local.py's read_background_job tool dispatch."""
    job = _BACKGROUND_JOBS.get(job_id)
    if job is None:
        return {"success": False, "error": f"No background job found with id {job_id!r}."}
    elapsed = time.monotonic() - job.started_at
    if not job.finished:
        return {
            "success": True, "job_id": job_id, "command": job.command,
            "status": "running", "elapsed_seconds": round(elapsed, 1),
        }
    result: Dict[str, Any] = {
        "success": True, "job_id": job_id, "command": job.command,
        "status": "failed" if job.error else "finished",
        "elapsed_seconds": round(elapsed, 1), "return_code": job.return_code,
        "stdout": job.stdout[-MAX_BACKGROUND_OUTPUT_CHARS:],
        "stderr": job.stderr[-MAX_BACKGROUND_OUTPUT_CHARS:],
    }
    if job.error:
        result["error"] = job.error
    return result


@dataclass
class ToolDefinition:
    """Definition of a tool for MCP"""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema
    handler: Optional[Callable] = None


_TOOL_PARAMETER_ALIASES: Dict[str, tuple[str, ...]] = {
    "path": ("file_path", "filepath", "target_path", "filename", "file"),
    "content": ("text", "new_content", "file_content"),
    "old_string": ("old_text", "old_content"),
    "new_string": ("new_text", "replacement"),
    "command": ("cmd", "shell_command"),
}

class MCPServer:
    """MCP server for tool execution"""

    def __init__(
        self, *, workspace_root: Optional[str] = None, session_id: Optional[int] = None,
        console: Optional[Any] = None, renderer: Optional[Any] = None, interactive: bool = False,
        transaction_id: Optional[str] = None,
        attachment_paths: Optional[List[str]] = None,
        allowed_workspace_roots: Optional[List[str]] = None,
        sandbox_policy: Optional[SandboxPolicy] = None,
    ):
        # workspace_root/session_id are optional so existing callers that
        # construct MCPServer() with no arguments (tests, the `tools`/
        # `screenshot` debug commands) keep today's behaviour: no boundary
        # enforcement on write_file/edit_file, no mutation-ledger recording.
        # The standalone agent loop (runner_local.py) always supplies both.
        self.workspace_root = workspace_root
        self.allowed_workspace_roots = {
            Path(item).expanduser().resolve()
            for item in (allowed_workspace_roots or [])
            if item
        }
        if workspace_root:
            self.allowed_workspace_roots.add(Path(workspace_root).expanduser().resolve())
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
        # None preserves the low-level MCPServer test/debug API. The real
        # agent runtime always supplies the configured policy.
        self.sandbox_policy = sandbox_policy
        self._external_mcp = _get_shared_mcp_bridge(workspace_root)
        # Per-server, per-root temporary indexes keep find_references
        # incremental across repeated calls without writing cache files into
        # either the user's repository or home directory. TemporaryDirectory
        # owns cleanup when this MCPServer/turn is released.
        self._symbol_index_dirs: Dict[str, tempfile.TemporaryDirectory] = {}
        self.tools: Dict[str, ToolDefinition] = {}
        self._register_default_tools()
        from .plugins import register_plugin_tools
        self.plugins = register_plugin_tools(self)
    
    def _register_default_tools(self):
        """Register default tools"""
        
        self.register_tool(
            name="read_file",
            description=(
                "Read text from one file. For large files, use 1-based offset and limit to read "
                "only the relevant line range; an unpaged large read returns the first page with "
                "an explicit continuation offset. Prefer search_code (or find_references for a "
                "known symbol) to locate the relevant region first. Fails with a "
                "clear error on a binary file (detected by a null byte in the first 8000 bytes) "
                "instead of returning corrupted text -- do not call this on an attached image; "
                "its content is already visible directly in this conversation for vision-capable "
                "models. Never guess a file's contents from its name or path; call this (or "
                "search_code) before describing what a file contains."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "offset": {
                        "type": "integer", "minimum": 1,
                        "description": "Optional 1-based first line to return",
                    },
                    "limit": {
                        "type": "integer", "minimum": 1, "maximum": 2000,
                        "description": "Optional maximum number of lines to return",
                    },
                },
                "required": ["path"]
            },
            handler=self._read_file
        )

        self.register_tool(
            name="write_file",
            description=(
                "Create a new file, or replace an existing file's ENTIRE contents. This is not "
                "an append or partial update -- any existing content at `path` not included in "
                "`content` is gone. To change only part of an existing file, use edit_file "
                "instead so the rest of the file (and any concurrent, unrelated edits) survives."
            ),
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
            name="save_memory",
            description=(
                "Save a durable, cross-session note the agent itself has learned during this "
                "session -- a build/test command that worked, a gotcha hit, a correction the "
                "user gave, a project fact worth remembering next time. Distinct from write_file: "
                "this does not touch any workspace file. Saving with an existing `name` overwrites "
                "that record (use this to correct or refresh a note, not to duplicate it). Only "
                "save something genuinely reusable across a future session -- not routine task "
                "narration."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Short, stable identifier for this note (e.g. \"deploy-command\")"},
                    "type": {
                        "type": "string", "enum": ["user", "feedback", "project", "reference"],
                        "description": (
                            "user: facts about the user/their role. feedback: guidance the user gave "
                            "about how to work. project: facts about this project/task. reference: "
                            "pointers to external systems/docs."
                        ),
                    },
                    "description": {"type": "string", "description": "One-line summary of what this note is, for future search/listing"},
                    "content": {"type": "string", "description": "The note itself"},
                },
                "required": ["name", "type", "description", "content"],
            },
            handler=self._save_memory,
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
            description=(
                "List the immediate children of one directory (not recursive -- subdirectory "
                "contents are not included; call this again on a specific subdirectory to go "
                "deeper). Common noise directories (.git, node_modules, __pycache__, and "
                "similar) are always excluded. For a broad, unfocused request, list the top "
                "level once and then act on what it actually returns -- read_file a specific "
                "file it named, list_directory a specific subdirectory, or use search_code for "
                "a concrete pattern -- rather than repeatedly listing while deciding what to do."
            ),
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
            description=(
                "Search file contents recursively under `path` using ripgrep (regex, not a "
                "literal substring match -- escape regex metacharacters if you want a literal "
                "string). This is the fast way to find where something is used or defined across "
                "many files; prefer it over read_file-ing files speculatively to look for a "
                "pattern, and prefer find_references instead when you already have an exact "
                "symbol name and want every definition and call site. `file_pattern` is a glob "
                "(e.g. '*.py') to narrow which files are searched. Common noise directories "
                "(.git, node_modules, __pycache__, and similar) are always excluded."
            ),
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
            name="create_artifact",
            description=(
                "Create a real DOCX, XLSX, PPTX, or PDF file inside the workspace. "
                "Use this for reports, spreadsheets, presentations, proposals, manuals, "
                "and other deliverables instead of writing fake text with an Office extension."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Output path ending in .docx, .xlsx, .pptx, or .pdf"},
                    "format": {"type": "string", "enum": ["docx", "xlsx", "pptx", "pdf"]},
                    "content": {
                        "type": "object",
                        "description": (
                            "Structured content. DOCX/PDF: title + sections[{heading,content}]. "
                            "XLSX: sheets[{name,rows,header,freeze_panes}]. PPTX: title/subtitle + "
                            "slides[{title,body or bullets}]."
                        ),
                        "additionalProperties": True,
                    },
                },
                "required": ["path", "format", "content"],
            },
            handler=self._create_artifact,
        )

        self.register_tool(
            name="inspect_artifact",
            description="Extract structured text and metadata from a DOCX, XLSX, PPTX, or PDF artifact.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Artifact path in the workspace or an exact attachment path"},
                    "max_chars": {"type": "integer", "description": "Maximum extracted text characters (default 30000)"},
                },
                "required": ["path"],
            },
            handler=self._inspect_artifact,
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
                    "sandbox_permissions": {
                        "type": "string",
                        "enum": ["use_default", "require_escalated"],
                        "description": (
                            "Use the configured sandbox, or request explicit human approval "
                            "to run without it. Never escalate silently."
                        ),
                    },
                    "approval_metadata": {"type": "object", "description": "Caller approval/audit metadata"}
                },
                "required": ["command"]
            },
            handler=self._execute_command
        )
        
        self.register_tool(
            name="get_git_info",
            description=(
                "Get a quick snapshot of a git repository's current state: current branch, "
                "remote.origin.url, the latest commit (hash/message/author/email/date), and "
                "whether the working tree is dirty (has_changes plus a count of changed files "
                "from `git status --porcelain`). Returns {\"is_git_repo\": false} if the path "
                "has no .git directory, and only the fields a given git command actually "
                "succeeded on -- a fresh repo with no commits yet, for example, still returns "
                "branch/remote without a latest_commit. This is a fixed read-only snapshot, not "
                "a general git command runner -- for anything else (diff, log history, blame, "
                "specific file status), use execute_command with the real git subcommand."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Repository path, relative to the workspace root (or absolute). "
                            "Defaults to the workspace root."
                        ),
                    }
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
            bridge = self._external_mcp
            if not bridge.available:
                owns_bridge = True
                await bridge.initialize(background=False)
            shared = await bridge.list_tools()
            tools.extend({**tool, "source": "shared_mcp"} for tool in shared)
        except Exception as exc:
            tools.append({
                "name": "shared_mcp",
                "description": f"External MCP registry unavailable: {exc}",
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

    async def external_tool_schemas_openai(self) -> List[Dict[str, Any]]:
        """Discover configured external MCP tools and keep sessions alive for this turn."""
        if not self._external_mcp.servers:
            return []
        if not self._external_mcp.available:
            await self._external_mcp.initialize(background=False)
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                },
            }
            for tool in await self._external_mcp.list_tools()
        ]

    async def shutdown(self) -> None:
        await self._external_mcp.shutdown()
    
    async def call_tool(
        self, name: str, parameters: Dict[str, Any], *, extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Call a tool under the active task trace without recording arguments."""
        from .runtime.telemetry import span

        with span("tool.invoke", tool_name=name):
            return await self._call_tool_impl(name, parameters, extra_kwargs=extra_kwargs)

    async def _call_tool_impl(
        self, name: str, parameters: Dict[str, Any], *, extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Call a tool by name.

        `extra_kwargs` (e.g. execute_command's background_signal) is passed
        straight to the handler alongside `parameters` but is never merged
        into it -- `parameters` is the model's own tool-call arguments,
        which get echoed back into working_messages and persisted (see
        state.py's completed_actions); a live object like an asyncio.Event
        in there would break json.dumps on the very next round.
        """
        if name not in self.tools:
            bridge = None
            owns_bridge = False
            try:
                bridge = self._external_mcp
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
                    "error": f"External MCP tool unavailable: {exc}",
                    "tool": name,
                    "source": "shared_mcp",
                    "success": False,
                }
            finally:
                if owns_bridge and bridge is not None:
                    await bridge.shutdown()
        
        tool = self.tools[name]
        if not isinstance(parameters, dict):
            return {
                "error": f"{name} requires an object of named arguments",
                "tool": name,
                "success": False,
            }
        parameters = self._normalise_tool_parameters(name, tool, parameters)
        missing = self._missing_tool_parameters(name, tool, parameters)
        if missing:
            rendered = ", ".join(missing)
            return {
                "error": f"{name} requires {rendered}; retry with the missing argument(s)",
                "tool": name,
                "success": False,
            }
        try:
            result = await tool.handler(**parameters, **(extra_kwargs or {}))
            return {"result": result, "tool": name, "success": True}
        except Exception as e:
            return {"error": str(e), "tool": name, "success": False}

    @staticmethod
    def _normalise_tool_parameters(
        name: str, tool: ToolDefinition, parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Canonicalise common model-generated argument aliases.

        Providers occasionally emit ``file_path`` instead of the schema's
        ``path`` (or ``text`` instead of ``content``). Normalising at the
        dispatch edge keeps every handler and every runner path consistent.
        Unknown arguments are preserved so handlers with compatibility
        ``**aliases`` continue to work.
        """
        result = dict(parameters)
        properties = set((tool.parameters.get("properties") or {}).keys())
        supported = set(properties)
        if name == "edit_file":
            supported.add("content")
        for canonical, aliases in _TOOL_PARAMETER_ALIASES.items():
            if canonical not in supported:
                continue
            for alias in aliases:
                if alias not in result:
                    continue
                if result.get(canonical) in (None, ""):
                    result[canonical] = result[alias]
                result.pop(alias, None)
        return result

    @staticmethod
    def _missing_tool_parameters(
        name: str, tool: ToolDefinition, parameters: Dict[str, Any],
    ) -> List[str]:
        def absent(key: str) -> bool:
            return key not in parameters or parameters[key] is None or parameters[key] == ""

        if name == "edit_file":
            missing = ["path"] if absent("path") else []
            if absent("content") and (absent("old_string") or absent("new_string")):
                missing.append("old_string and new_string (or content)")
            return missing
        return [
            str(key) for key in (tool.parameters.get("required") or [])
            if absent(str(key))
        ]
    
    async def _read_file(
        self, path: str, offset: Optional[int] = None, limit: Optional[int] = None,
    ) -> str:
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
        content = p.read_text(encoding='utf-8', errors='ignore')
        lines = content.splitlines(keepends=True)
        default_page_lines = 800
        requested_page = offset is not None or limit is not None
        if not requested_page and len(lines) <= default_page_lines:
            return content
        try:
            start = max(1, int(offset or 1))
            page_size = min(2000, max(1, int(limit or default_page_lines)))
        except (TypeError, ValueError):
            return "Error: read_file offset and limit must be positive integers"
        if start > len(lines) and lines:
            return f"[Offset {start} is beyond the end of {path} ({len(lines)} lines).]"
        selected = lines[start - 1:start - 1 + page_size]
        end = start + len(selected) - 1
        numbered = "".join(
            f"{line_number}: {line}" for line_number, line in enumerate(selected, start=start)
        )
        if selected and not selected[-1].endswith(("\n", "\r")):
            numbered += "\n"
        continuation = (
            f" Continue with offset={end + 1}, limit={page_size}."
            if end < len(lines) else " End of file."
        )
        return f"[Showing lines {start}-{max(end, start)} of {len(lines)}.{continuation}]\n{numbered}"

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
            roots = self.allowed_workspace_roots or {base.resolve()}
            if not any(resolved == root or root in resolved.parents for root in roots):
                rendered = ", ".join(str(root) for root in sorted(roots, key=str))
                raise PermissionError(
                    f"'{path}' resolves outside the workspace; approved roots: ({rendered})"
                )
        return resolved

    def _atomic_write_text(self, target: Path, content: str) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            # FIX: os.replace() swaps inodes -- without this, an edited
            # file silently lost its original mode/owner and inherited
            # mkstemp's restrictive 0600 + the running process's uid/gid
            # (confirmed live: reported as files ending up 0600 owned by
            # "nobody:nobody" after write_file/edit_file).
            from .fs_atomic import preserve_existing_metadata
            preserve_existing_metadata(temp_name, target)
            os.replace(temp_name, target)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(temp_name)
            raise

    async def _write_file(self, path: str, content: str | None = None, **aliases: Any) -> str:
        content = content if content is not None else aliases.pop("text", None)
        content = content if content is not None else aliases.pop("new_content", None)
        content = content if content is not None else aliases.pop("file_content", None)
        if content is None:
            return "❌ Error: write_file requires content"
        p = self._resolve_in_workspace(path)
        original_content = p.read_text(encoding="utf-8", errors="ignore") if p.is_file() else None
        self._atomic_write_text(p, content)
        if p.read_text(encoding="utf-8", errors="strict") != content:
            return f"❌ Failed to verify write to '{path}'"
        if self.session_id is not None:
            from .safety import record_mutation
            record_mutation(
                self.session_id, path=str(p), operation="create" if original_content is None else "update",
                original_content=original_content, new_content=content,
                transaction_id=self.transaction_id,
            )
        return f"✅ Successfully wrote {len(content)} bytes to '{path}'"

    async def _save_memory(
        self, name: str, type: str, description: str, content: str, **_aliases: Any
    ) -> str:
        """Append/update a durable, cross-session memory record (runtime/memory.py's
        MemoryStore) -- distinct from write_file/edit_file, which only ever touch
        workspace files. Saving overwrites any existing record of the same name
        (same as `tamfis-code memory save`, which this mirrors); size and
        record-count are capped automatically by the store, oldest evicted first."""
        from .runtime.memory import MemoryError as _MemoryError, MemoryRecord, MemoryType, get_memory_store

        try:
            memory_type = MemoryType(type)
        except ValueError:
            valid = ", ".join(t.value for t in MemoryType)
            return f"❌ Error: invalid memory type {type!r}. Must be one of: {valid}"
        try:
            record = get_memory_store().save(
                MemoryRecord(name=name, type=memory_type, description=description, content=content)
            )
        except _MemoryError as exc:
            return f"❌ Error: {exc}"
        return f"✅ Saved memory '{record.name}' ({memory_type.value})"

    async def _edit_file(
        self, path: str, old_string: str | None = None, new_string: str | None = None, **aliases: Any
    ) -> str:
        old_string = old_string if old_string is not None else aliases.pop("old_text", None)
        new_string = new_string if new_string is not None else aliases.pop("new_text", None)
        new_string = new_string if new_string is not None else aliases.pop("replacement", None)
        full_content = aliases.pop("content", None)
        full_content = full_content if full_content is not None else aliases.pop("new_content", None)
        if full_content is not None and old_string is None:
            return await self._write_file(path, content=full_content)
        if old_string is None or new_string is None:
            return "❌ Error: edit_file requires old_string and new_string, or content for full replacement"
        p = self._resolve_in_workspace(path)
        if not p.is_file():
            return f"❌ Error: File '{path}' not found"
        original_content = p.read_text(encoding="utf-8", errors="ignore")
        occurrences = original_content.count(old_string)
        if occurrences == 0:
            hint = ""
            # The model just re-read this exact file yet still produced a
            # non-matching old_string three rounds running (the transcript
            # that prompted this fix showed identical retries) -- the most
            # common real cause is whitespace/line-ending drift (CRLF vs LF,
            # or reformatted indentation) rather than the text being truly
            # absent. Normalize both sides and say so explicitly so the
            # model stops re-issuing the identical failing call and instead
            # copies whitespace verbatim from a fresh read.
            def _normalize(text: str) -> str:
                return "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").split("\n"))
            if _normalize(old_string) in _normalize(original_content):
                hint = (
                    " (a whitespace/line-ending-normalized version of old_string DOES match -- "
                    "the mismatch is likely trailing whitespace, indentation, or CRLF vs LF; "
                    "re-read the file and copy old_string verbatim from that fresh content instead "
                    "of reusing this same old_string again)"
                )
            return f"❌ Error: old_string not found in '{path}' -- no changes made{hint}"
        if occurrences > 1:
            return (
                f"❌ Error: old_string matches {occurrences} times in '{path}' -- it must be unique. "
                "Include more surrounding context to disambiguate."
            )
        new_content = original_content.replace(old_string, new_string, 1)
        self._atomic_write_text(p, new_content)
        if p.read_text(encoding="utf-8", errors="strict") != new_content:
            return f"❌ Failed to verify edit to '{path}'"
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

            # Use asyncio's subprocess transport directly. Running
            # subprocess.run inside asyncio.to_thread can deadlock during
            # process creation on some Python/runtime combinations, leaving
            # both search_code and find_references stuck until their caller
            # is killed. A separate process group also lets timeout cleanup
            # reach any unexpected descendants.
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            try:
                stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
            except asyncio.TimeoutError:
                await self._kill_process_group(proc)
                return [{"error": "Search timed out"}]
            stdout = stdout_bytes.decode("utf-8", errors="replace")
            matches = []

            for line in stdout.split('\n'):
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

    async def _create_artifact(self, path: str, format: str, content: Dict[str, Any]) -> Dict[str, Any]:
        from .artifacts import create_artifact
        target = self._resolve_in_workspace(path)
        existed = target.exists()
        result = create_artifact(target, format, content if isinstance(content, dict) else {})
        if self.session_id is not None:
            from .safety import record_mutation
            record_mutation(
                self.session_id, path=str(target), operation="update" if existed else "create",
                original_content=None, new_content=None, transaction_id=self.transaction_id,
            )
        return result

    async def _inspect_artifact(self, path: str, max_chars: int = 30_000) -> Dict[str, Any]:
        from .artifacts import inspect_artifact
        source = self._resolve_readable_input(path)
        if not source.is_file():
            return {"success": False, "error": f"Artifact not found: {path}"}
        try:
            limit = min(max(int(max_chars), 1_000), 100_000)
        except (TypeError, ValueError):
            limit = 30_000
        return inspect_artifact(source, max_chars=limit)
    
    async def _kill_process_group(self, proc: "asyncio.subprocess.Process") -> None:
        # Kill the whole process group (the shell was started with
        # start_new_session=True) rather than just the immediate `sh -lc`
        # process, so children the command spawned die too. Bound the
        # follow-up wait() -- if the process is stuck (e.g. uninterruptible
        # I/O) it must not block the caller forever; a prior version of this
        # code awaited proc.wait() with no timeout at all and could hang a
        # turn indefinitely once a command's own timeout had already fired.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(proc.wait(), timeout=10)

    async def _execute_command(
        self, command: str, cwd: Optional[str] = None, timeout: int = 60,
        environment: Optional[Dict[str, str]] = None, shell: str = "bash",
        sandbox_permissions: str = "use_default",
        approval_metadata: Optional[Dict[str, Any]] = None,
        background_signal: Optional[asyncio.Event] = None,
    ) -> Dict[str, Any]:
        # `background_signal` is never part of this tool's schema and the
        # model never sets it -- runner_local.py injects it into arguments
        # right before dispatch, sourced from the live REPL's Ctrl+B
        # keybinding (see live_input.py), so it can only ever be set by the
        # human actually watching this specific command run.
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
        if sandbox_permissions not in {"use_default", "require_escalated"}:
            return {"error": f"Unsupported sandbox permission: {sandbox_permissions}", "success": False}
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
            sandbox_command = None
            argv = (shell, "-lc", command)
            if self.sandbox_policy is not None and self.workspace_root:
                sandbox_command = build_sandbox_command(
                    command=command, shell=shell, cwd=run_dir,
                    workspace_root=Path(self.workspace_root).expanduser().resolve(),
                    policy=self.sandbox_policy,
                    require_escalated=sandbox_permissions == "require_escalated",
                )
                argv = sandbox_command.argv
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                # FIX: no stdin= here meant the child inherited the real
                # terminal's stdin fd unmodified. Nothing in this tool can
                # ever supply interactive input to a running command (the
                # model has no channel to answer a prompt), so any command
                # that waits on stdin -- a credential prompt, a pager, a
                # confirmation, an interactive subcommand invoked by
                # mistake -- blocked forever on a real TTY read that would
                # never be satisfied, while tamfis-code's own prompt_toolkit
                # input loop was concurrently trying to read raw bytes from
                # that same terminal. Live-reported: total input freeze
                # ("no response to input until I close the terminal") with
                # no way to Ctrl+C past it, since the hang was in the child
                # process's own blocking read, not anywhere this process's
                # asyncio loop could intercept. DEVNULL gives any such
                # prompt an immediate EOF instead of an indefinite wait, so
                # it fails fast (or the command handles EOF gracefully) and
                # this tool's own `timeout`/kill-on-timeout path (below)
                # actually gets a chance to run.
                stdin=asyncio.subprocess.DEVNULL,
                cwd=str(run_dir), env=env,
                # New session/process group so a kill on timeout can reach
                # any children the command spawns (e.g. `npm run dev`),
                # not just the immediate shell -- see the kill/wait paths
                # below, which target the group via os.killpg.
                start_new_session=True,
            )
            if background_signal is None:
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                    return {
                        "stdout": stdout.decode('utf-8', errors='ignore'),
                        "stderr": stderr.decode('utf-8', errors='ignore'),
                        "return_code": proc.returncode,
                        "success": proc.returncode == 0,
                        "sandbox": _sandbox_result(sandbox_command),
                    }
                except asyncio.TimeoutError:
                    # asyncio.wait_for only cancels the communicate() task on
                    # timeout, it never touches the subprocess -- without an
                    # explicit kill here the process (and any children) leak
                    # and keep running forever in the background.
                    await self._kill_process_group(proc)
                    return {"error": f"Command timed out after {timeout} seconds", "success": False}
            # Race the ordinary completion wait against a possible mid-flight
            # background request -- the SAME already-running proc either way;
            # detaching never restarts it under a different mechanism, only
            # who is waiting on it changes.
            communicate_task = asyncio.ensure_future(proc.communicate())
            background_wait = asyncio.ensure_future(background_signal.wait())
            try:
                done, _pending = await asyncio.wait(
                    {communicate_task, background_wait},
                    timeout=timeout, return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                if not background_wait.done():
                    background_wait.cancel()
            if communicate_task in done:
                stdout, stderr = communicate_task.result()
                return {
                    "stdout": stdout.decode('utf-8', errors='ignore'),
                    "stderr": stderr.decode('utf-8', errors='ignore'),
                    "return_code": proc.returncode,
                    "success": proc.returncode == 0,
                    "sandbox": _sandbox_result(sandbox_command),
                }
            if background_wait in done:
                job_id = uuid.uuid4().hex[:12]
                job = BackgroundJob(
                    job_id=job_id, command=command, cwd=str(run_dir),
                    started_at=time.monotonic(), proc=proc,
                    communicate_task=communicate_task,
                )
                _BACKGROUND_JOBS[job_id] = job
                asyncio.ensure_future(_watch_background_job(job))
                return {
                    "success": True, "backgrounded": True, "job_id": job_id,
                    "sandbox": _sandbox_result(sandbox_command),
                    "message": (
                        f"Moved to the background as job {job_id} -- it keeps running. "
                        "Continue with other work now; call read_background_job with this "
                        "job_id later to check on it or collect its output."
                    ),
                }
            # Neither finished in time: a genuine timeout, not a background
            # request. Same outcome as the no-signal path above.
            communicate_task.cancel()
            await self._kill_process_group(proc)
            return {"error": f"Command timed out after {timeout} seconds", "success": False}
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
        
        # Run Git natively through asyncio. asyncio.to_thread kept a default-
        # executor worker alive after this coroutine returned under Python
        # 3.13/strict event-loop teardown, which could leave `get_git_info`
        # callers hanging indefinitely. A bounded async subprocess also
        # prevents a hook/filesystem-stalled Git command from freezing the
        # live input loop.
        async def _git(*args: str) -> subprocess.CompletedProcess[str]:
            command = ['git', '-C', str(p), *args]
            proc = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10.0)
            except asyncio.TimeoutError:
                proc.kill()
                stdout, stderr = await proc.communicate()
                return subprocess.CompletedProcess(
                    command, 124,
                    stdout.decode("utf-8", errors="replace"),
                    (stderr.decode("utf-8", errors="replace") + "\nGit command timed out").strip(),
                )
            return subprocess.CompletedProcess(
                command, proc.returncode,
                stdout.decode("utf-8", errors="replace"),
                stderr.decode("utf-8", errors="replace"),
            )

        try:
            # Get current branch
            result = await _git('rev-parse', '--abbrev-ref', 'HEAD')
            if result.returncode == 0:
                info["branch"] = result.stdout.strip()

            # Get remote URL
            result = await _git('config', '--get', 'remote.origin.url')
            if result.returncode == 0:
                info["remote_url"] = result.stdout.strip()

            # Get latest commit
            result = await _git('log', '-1', '--format=%H%n%s%n%an%n%ae%n%ad')
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
            result = await _git('status', '--porcelain')
            info["has_changes"] = bool(result.stdout.strip())
            info["changed_files"] = len([line for line in result.stdout.split('\n') if line.strip()])
            
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
            raise RuntimeError("Portable browser support is unavailable")
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
