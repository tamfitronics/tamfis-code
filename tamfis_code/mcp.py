"""Model Context Protocol (MCP) integration for tools"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
import contextlib
import hashlib
import html
import json
import os
import re
import shlex
import signal
import subprocess
import fnmatch
import asyncio
import sys
import shutil
import tarfile
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path

import httpx
from rich.panel import Panel

from .render import resume_live_if_active, suspend_live_if_active
from .sandbox import SandboxPolicy, build_sandbox_command
from .timeouts import adaptive_command_timeout
from .safety import _is_read_only_command
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
# TamfisGPT's internal Tier IV service (tier_iv_orchestration/tamgpt_api.py),
# 127.0.0.1-only, no auth needed -- only reachable/useful on a host that also
# runs TamfisGPT. Overridable for anyone running that service on a different
# port/host.
_TAMGPT_TIER_IV_BASE = os.environ.get("TAMGPT_TIER_IV_URL", "http://127.0.0.1:9555").rstrip("/")
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

_WP_CLI_ROOT_GUARD_RE = re.compile(
    r"(?:running this as root|meant to run this as the user|use the --allow-root flag)",
    re.IGNORECASE,
)


def _wp_cli_root_retry_command(command: str, output: str) -> Optional[str]:
    """Return a safe WP-CLI retry after its root guard rejects a read.

    WP-CLI refuses to run as root even for harmless reads such as
    ``wp option get``.  That refusal is environmental, not evidence that the
    option is absent.  Retry only a single, already-classified read-only
    command and never add ``--allow-root`` to a write, pipeline, or shell
    expression.  The caller has already passed the normal command approval
    gate; this is an execution recovery for the same approved observation.
    """
    if not output or not _WP_CLI_ROOT_GUARD_RE.search(output):
        return None
    if not _is_read_only_command(command):
        return None
    try:
        argv = shlex.split(command)
    except ValueError:
        return None
    if not argv or Path(argv[0]).name != "wp" or "--allow-root" in argv:
        return None
    return shlex.join([argv[0], "--allow-root", *argv[1:]])


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


def _get_shared_mcp_bridge(workspace_root: str | None = None, servers=None):
    """Return Tamfis Code's standalone MCP client bridge."""
    from .mcp_client import StandaloneMCPBridge
    return StandaloneMCPBridge(workspace_root, servers=servers)


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
# ``0`` means recurse until there are no more reachable directories. There is
# deliberately no hard maximum: discovery must reach the deepest file in a
# deeply nested project tree. The result-size cap is independent and protects
# the model context from an accidental broad listing.
MAX_LIST_DIRECTORY_DEPTH = None
MAX_SEARCH_RESULTS = 200
# How many matches may be RETURNED in one tool result before the remainder is
# handed over as a continuation offset. Small enough to keep a broad query from
# filling the context window, large enough that most real searches finish in one
# page. An explicit max_results may override it, up to MAX_SEARCH_RESULTS.
SEARCH_PAGE_RESULTS = 80
# How many matches are COLLECTED before paging: paging is useless if the pool
# stops at the page size. 2,000 matches at <=500 characters each is bounded
# (~1 MB worst case) and far past what any interactive search needs.
_SEARCH_POOL_LIMIT = 2000


def _sorted_search_matches(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Put matches in a stable order (path, then line) so paging offsets mean
    the same thing on every call.

    ripgrep walks the tree in parallel and emits files in whatever order the
    workers finish, so an unsorted pool differs between two identical
    invocations -- and offset paging over it repeats some matches while
    silently skipping others. Sorting ≤2,000 entries is free next to the
    search itself. The trailing pool/error marker is left exactly where it is;
    the paging call site re-attaches it.
    """
    marker = matches[-1] if matches and matches[-1].get("truncated") else None
    body = matches[:-1] if marker is not None else list(matches)
    if any("error" in entry for entry in body):
        return list(matches)
    try:
        body.sort(key=lambda entry: (str(entry.get("file", "")), int(entry.get("line", 0))))
    except (TypeError, ValueError):
        return list(matches)
    return [*body, marker] if marker is not None else body


def _page_search_matches(
    matches: List[Dict[str, Any]],
    *,
    offset: Optional[int] = None,
    max_results: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Slice search matches into one page plus a continuation pointer.

    Mirrors read_file's paging contract so the agent has ONE mental model for
    "there is more, here is how to get it": the payload ends with a
    `pagination` entry carrying `showing`, `total`, and `next_offset` (None on
    the last page).

    Callers must pass a DETERMINISTICALLY ordered list: an offset is only
    meaningful if both calls see the same order. `_search_code_matches` sorts
    its matches for exactly this reason -- ripgrep walks the tree in parallel,
    so two identical invocations could otherwise page over different
    orderings, repeating some matches while skipping others (caught by
    test_one_page_plus_a_continuation_pointer).
    """
    total = len(matches)
    try:
        page_size = int(max_results) if max_results is not None else SEARCH_PAGE_RESULTS
    except (TypeError, ValueError):
        page_size = SEARCH_PAGE_RESULTS
    page_size = max(1, min(MAX_SEARCH_RESULTS, page_size))
    try:
        start = max(0, int(offset) - 1) if offset is not None else 0
    except (TypeError, ValueError):
        start = 0
    paged_request = offset is not None or max_results is not None
    if not total:
        return matches
    if start >= total:
        return [{
            "note": (
                f"[offset={start + 1} is past the end: {total} match(es) found. "
                f"Re-run without offset to see them from the start.]"
            ),
        }]
    page = matches[start:start + page_size]
    end = start + len(page)
    if end >= total and not paged_request and total <= page_size:
        return matches
    entry: Dict[str, Any] = {
        "showing": f"{start + 1}-{end}",
        "total": total,
        "next_offset": end + 1 if end < total else None,
    }
    if end < total:
        entry["note"] = (
            f"Showing matches {start + 1}-{end} of {total}. Continue with "
            f"offset={end + 1}, max_results={page_size} (same query) for the rest."
        )
    else:
        entry["note"] = f"Showing matches {start + 1}-{end} of {total} (end of results)."
    return [*page, {"pagination": entry}]
# Files larger than this are skipped by search_code -- a single huge
# (often generated/minified) file can otherwise dominate the whole result
# set with one or two enormous match lines.
MAX_SEARCH_FILE_SIZE_BYTES = 2_000_000
MAX_SEARCH_MATCH_CHARS = 500

# Missing-file recovery must inspect the complete authorised source tree. The
# walk still skips dependency/build output directories, but it has no depth or
# file-count ceiling: a valid source file must not be hidden by an arbitrary
# traversal limit.
MAX_READ_RECOVERY_CANDIDATES = 8

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


def kill_background_job(job_id: str, force: bool = False) -> Dict[str, Any]:
    """Terminate a still-running backgrounded job (Claude Code's KillShell
    parity -- until now a Ctrl+B-backgrounded command could be polled forever
    but never stopped by the agent). Only jobs THIS session started can be
    killed (the registry is module-level but only ever populated from this
    process's own execute_command), and only by signal: no shell is spawned,
    so nothing else on the machine is reachable through this tool.

    SIGTERM by default so the child can clean up; force=True escalates to
    SIGKILL for a child ignoring SIGTERM. The process was spawned with
    start_new_session=True, so the kill goes to the child's whole process
    group -- a `make test` that spawned workers dies with its children
    instead of orphaning them. Returns immediately after signalling; poll
    read_background_job to confirm exit (the in-flight communicate() still
    reaps the real return code into the job record)."""
    job = _BACKGROUND_JOBS.get(job_id)
    if job is None:
        return {"success": False, "error": f"No background job found with id {job_id!r}."}
    if job.finished:
        return {
            "success": True, "job_id": job_id, "command": job.command,
            "status": "already_finished", "return_code": job.return_code,
            "message": "Job already finished; nothing to kill.",
        }
    killed = False
    signal_name = "SIGKILL" if force else "SIGTERM"
    errors: list[str] = []
    if hasattr(os, "killpg"):
        try:
            os.killpg(os.getpgid(job.proc.pid), signal.SIGKILL if force else signal.SIGTERM)
            killed = True
        except (ProcessLookupError, PermissionError, OSError) as exc:
            errors.append(str(exc))
    if not killed:
        # Either a platform without killpg, or the group kill raced the
        # child's exit / lacked permission -- try the direct child only.
        try:
            if force:
                job.proc.kill()
            else:
                job.proc.terminate()
            killed = True
        except ProcessLookupError:
            # Exited between the running check and here -- treat as done.
            killed = True
        except OSError as exc:
            errors.append(str(exc))
    if not killed:
        return {
            "success": False, "job_id": job_id, "command": job.command,
            "error": "Could not signal the job's process: " + "; ".join(errors),
        }
    result: Dict[str, Any] = {
        "success": True, "job_id": job_id, "command": job.command,
        "killed": True, "signal": signal_name,
        "status": "signalled",
        "message": (
            f"Sent {signal_name} to the job's process group. Call read_background_job "
            "with this job_id to confirm it exited."
        ),
    }
    if errors:
        result["detail"] = "; ".join(errors)
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
    # Providers sometimes reuse the generic `path` field when they mean the
    # command working directory. Keep execute_command compatible without
    # passing an unknown keyword into its handler.
    "cwd": ("working_directory", "directory", "path"),
    "content": ("text", "new_content", "file_content"),
    "old_string": ("old_text", "old_content"),
    "new_string": ("new_text", "replacement"),
    "command": ("cmd", "shell_command"),
}

def salvage_truncated_tool_arguments(raw_arguments: str) -> dict[str, Any]:
    """Recover as much as possible from a TRUNCATED tool-call argument object.

    A model's tool-call arguments are bounded by its output token limit, so a
    genuinely large `write_file` (a multi-page report, a long generated file)
    arrives with its JSON cut off mid-string. Everything before the cut is real
    work; the runner uses this to keep it instead of discarding the call (see
    the malformed-arguments branch in runner_local.py).

    Returns a dict of the members that could be recovered. A truncated FINAL
    string value is closed and returned as-is (a partial document is far more
    useful than nothing, and the caller tells the model how much was kept so it
    can append the remainder). Returns {} when nothing usable is present.
    """
    text = (raw_arguments or "").strip()
    if not text.startswith("{"):
        return {}

    def _try(blob: str) -> Optional[dict[str, Any]]:
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    # 1. Truncate at the last COMPLETE top-level member and close the object.
    #    Scanning member boundaries rather than guessing an offset means a
    #    nested object/array inside a complete member can never be cut open.
    # A member boundary is only real once a VALUE has been read. Tracking
    # `in_key` separately matters: without it, the closing quote of the KEY
    # ("content") looked like a completed member, and the recovered prefix was
    # `{"path": "/a", "content"}` -- unparseable, so a truncated write
    # salvaged nothing.
    boundary = 1  # just after the opening brace
    depth = 1
    in_string = False
    in_key = False
    expect_key = True  # the first token inside an object is a key
    escaped = False
    for index in range(1, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
                if depth == 1 and not in_key:
                    boundary = index + 1
            continue
        if char == '"':
            in_string = True
            in_key = expect_key
            expect_key = False
            continue
        if char in "{[":
            depth += 1
            expect_key = True
        elif char in "}]":
            depth -= 1
            if depth == 0:
                # The object actually closed -- not truncated at all.
                complete = _try(text[: index + 1])
                return complete or {}
            if depth == 1:
                boundary = index + 1
        elif char == "," and depth == 1:
            boundary = index
            expect_key = True
    recovered = _try(text[:boundary].rstrip().rstrip(",") + "}") or {}

    # 2. The cut may have landed inside the value of the member AFTER the last
    #    complete one (typically `"content"`). Recover the key and whatever of
    #    its string value arrived, so a partial document is preserved.
    tail = text[boundary:]
    key_match = re.match(r'\s*,?\s*"([A-Za-z_][A-Za-z0-9_]*)"\s*:\s*"', tail)
    if key_match:
        partial = tail[key_match.end():]
        if partial.endswith("\\"):
            partial = partial[:-1]
        recovered.setdefault(key_match.group(1), _unescape_partial_json_string(partial))
    return recovered


def _unescape_partial_json_string(text: str) -> str:
    """Decode the JSON escapes in a string that was cut off mid-value.

    A trailing lone backslash (the escape character itself was the last byte)
    is dropped rather than raising, and any other malformed escape is kept
    literally -- the point is to preserve the model's text, not to be strict.
    """
    try:
        return json.loads('"' + text + '"')
    except json.JSONDecodeError:
        pass
    out: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char != "\\" or index + 1 >= len(text):
            out.append(char)
            index += 1
            continue
        nxt = text[index + 1]
        mapping = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f"}
        if nxt in mapping:
            out.append(mapping[nxt])
            index += 2
            continue
        if nxt == "u" and index + 5 < len(text):
            try:
                out.append(chr(int(text[index + 2:index + 6], 16)))
                index += 6
                continue
            except ValueError:
                pass
        out.append(nxt)
        index += 2
    return "".join(out)



async def run_blocking_bounded(fn: Callable[[], Any], timeout: float) -> Any:
    """Run synchronous filesystem/CPU work off the event loop, bounded in time.

    Tool handlers are coroutines, but several did plain blocking work directly
    on the loop (recursive directory walks, whole-tree symbol indexing, the
    pure-Python search fallback). While that ran the ENTIRE terminal froze --
    keystrokes, Esc, Ctrl+C and the status clock included -- because prompt
    input is served by the same loop. The work now runs on a daemon thread (a
    stuck one can never block interpreter shutdown, unlike a default-executor
    worker) and the caller gets ``asyncio.TimeoutError`` after ``timeout``.
    """
    loop = asyncio.get_running_loop()
    future: "asyncio.Future[Any]" = loop.create_future()

    def _settle(setter: Callable[[Any], None], value: Any) -> None:
        if not future.done():
            setter(value)

    def _worker() -> None:
        try:
            value = fn()
        except BaseException as exc:  # noqa: BLE001 - handed to the awaiting task
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(_settle, future.set_exception, exc)
        else:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(_settle, future.set_result, value)

    threading.Thread(target=_worker, name="tamfis-blocking-tool", daemon=True).start()
    return await asyncio.wait_for(future, timeout=timeout)


class MCPServer:
    """MCP server for tool execution"""

    def __init__(
        self, *, workspace_root: Optional[str] = None, session_id: Optional[int] = None,
        console: Optional[Any] = None, renderer: Optional[Any] = None, interactive: bool = False,
        transaction_id: Optional[str] = None,
        attachment_paths: Optional[List[str]] = None,
        allowed_workspace_roots: Optional[List[str]] = None,
        sandbox_policy: Optional[SandboxPolicy] = None,
        external_mcp_servers: Optional[Dict[str, Any]] = None,
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
        # Set the first time build_sandbox_command raises its fail-closed
        # RuntimeError (bwrap missing + fail_if_unavailable). Every command
        # in a turn re-triggers the identical multi-line remediation
        # paragraph otherwise -- confirmed live: two execute_command calls
        # in the same turn each returned the full "Install bubblewrap..."
        # text, which reads as the tool being broken/repeating itself
        # rather than one already-explained, still-true precondition. The
        # first failure still gets the full explanation; later ones in the
        # same turn get a one-line reminder instead.
        self._sandbox_unavailable_warned = False
        self._external_mcp = _get_shared_mcp_bridge(workspace_root, external_mcp_servers)
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
                    "line_start": {
                        "type": "integer", "minimum": 1,
                        "description": "Compatibility alias for offset (1-based first line)",
                    },
                    "line_end": {
                        "type": "integer", "minimum": 1,
                        "description": "Optional inclusive ending line; may be used with line_start",
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
                "instead so the rest of the file (and any concurrent, unrelated edits) survives. "
                "For an existing source file, the live agent must first diagnose/read it and provide "
                "expected_sha256 and explicitly select mode=\"overwrite\"; ordinary writes never "
                "replace existing source code. "
                "Use the extension the language and project actually use -- never '.txt' for code. "
                "For a LARGE document, NEVER send it in one call: your arguments are bounded by "
                "your own output token limit, and a call much over roughly 6,000 characters of "
                "`content` will be cut off mid-file. Send at most ~6,000 characters per call -- "
                "write the first part, then continue with mode=\"append\" calls for the remainder. "
                "Example: write_file(path=\"report.md\", content=\"<part 1: up to 6,000 chars>\") "
                "then write_file(path=\"report.md\", mode=\"append\", content=\"<part 2>\")."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "File content"},
                    "mode": {
                        "type": "string",
                        "enum": ["write", "append", "overwrite"],
                        "description": (
                            "'write' (default) replaces the file; 'append' adds `content` to "
                            "the end of the existing file; 'overwrite' is required for an "
                            "explicit, hash-verified full replacement of existing source code."
                        ),
                    },
                    "expected_sha256": {
                        "type": "string",
                        "description": "SHA-256 of the existing source file after diagnostic read; required for live full replacement",
                    },
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
                "context in old_string to make the match unambiguous. Read the file (or the "
                "relevant range) first and copy whitespace exactly; after editing, re-read the "
                "changed region and run the project's checks to confirm the edit does what you "
                "intended. Use write_file instead for creating a brand-new file. Full replacement "
                "of an existing source file is refused by the live agent unless expected_sha256 is verified."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "old_string": {"type": "string", "description": "Exact text to replace (must match exactly once)"},
                    "new_string": {"type": "string", "description": "Replacement text"},
                    "content": {"type": "string", "description": "Full replacement content for a new file or verified source replacement"},
                    "expected_sha256": {"type": "string", "description": "SHA-256 from the diagnostic read of the existing file"},
                },
                "required": ["path", "old_string", "new_string"],
            },
            handler=self._edit_file,
        )

        self.register_tool(
            name="list_directory",
            description=(
                "List one directory tree. By default depth=1 lists immediate children; "
                "set depth=0 to recurse to the deepest reachable directory, or provide any "
                "positive depth for a finite prefix. Common noise directories "
                "(.git, node_modules, __pycache__, and "
                "similar) are always excluded. For a broad, unfocused request, list the top "
                "level once and then act on what it actually returns -- read_file a specific "
                "file it named, list_directory a specific subdirectory, or use search_code for "
                "a concrete pattern -- rather than repeatedly listing while deciding what to do. "
                "This is the mandatory repository-orientation operation: use it before the "
                "first read of an unresolved repository path instead of guessing a filename."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path"},
                    "depth": {
                        "type": "integer", "minimum": 0,
                        "default": 1,
                        "description": (
                            "Optional recursion depth. 1 lists immediate children; "
                            "0 walks to the deepest reachable directory with no depth limit."
                        ),
                    },
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
                "(.git, node_modules, __pycache__, and similar) are always excluded. Results "
                "are PAGED: when a result ends with a `pagination` entry, its `next_offset` "
                "means there are more matches -- call again with that offset and the same "
                "query instead of re-running a broader or differently-worded search."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search pattern"},
                    "path": {"type": "string", "description": "Search directory"},
                    "file_pattern": {"type": "string", "description": "File pattern to match"},
                    "offset": {
                        "type": "integer", "minimum": 1,
                        "description": (
                            "Optional 1-based first match to return; continue with the "
                            "`next_offset` from a previous result."
                        ),
                    },
                    "max_results": {
                        "type": "integer", "minimum": 1,
                        "maximum": MAX_SEARCH_RESULTS,
                        "description": "Optional matches per page (default 80, maximum 200)",
                    },
                    "limit": {
                        "type": "integer", "minimum": 1,
                        "maximum": MAX_SEARCH_RESULTS,
                        "description": (
                            "Alias for max_results accepted by common search tool callers"
                        ),
                    },
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
            name="kill_background_job",
            description=(
                "Terminate a still-running command that was moved to the background "
                "(an execute_command result with backgrounded=true and a job_id) -- e.g. a "
                "build or dev server you no longer need, or one blocking the task. Sends "
                "SIGTERM to the job's whole process group (children die with it); pass "
                "force=true only if the job ignored an earlier SIGTERM. Finished jobs are "
                "reported, not an error. Confirm the exit with read_background_job afterwards."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "The job_id from the backgrounded execute_command result"},
                    "force": {
                        "type": "boolean",
                        "description": "Send SIGKILL instead of SIGTERM (only after a normal kill was ignored)",
                    },
                },
                "required": ["job_id"],
            },
            handler=self._kill_background_job,
        )

        self.register_tool(
            name="read_archive",
            description=(
                "Read-only look inside a ZIP or TAR archive (.zip .tar .tar.gz .tgz .tar.bz2 .tar.xz), "
                "with NO size limit on the archive and NO extraction to disk: list its files (paged, "
                "optional pattern filter) or read one text member by name (paged like read_file). "
                "Archives inside archives work to any depth: chain them with '!/', e.g. "
                "path='pack.zip!/data/inner.tar.gz' lists the inner archive and adding "
                "member='notes/readme.txt' reads a file from it. Use this instead of read_file on "
                "an archive (read_file rejects binary files) and instead of extract_archive when you "
                "only need to inspect; use extract_archive only when files must be edited or run."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Archive path; chain nested archives with '!/'"},
                    "member": {"type": "string", "description": "Text file inside the (innermost) archive to read; omit to list files"},
                    "pattern": {"type": "string", "description": "When listing: only names containing this text or matching this glob"},
                    "offset": {"type": "integer", "minimum": 1, "description": "1-based first entry (listing) or first line (member) to return"},
                    "limit": {"type": "integer", "minimum": 1, "description": "Entries (default 200, max 1000) or lines (default 800, max 2000) per call"},
                },
                "required": ["path"],
            },
            handler=self._read_archive,
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
                    "max_chars": {"type": "integer", "description": "Maximum extracted text characters per call (default 30000)"},
                    "offset": {"type": "integer", "description": "Character offset into the extracted text to continue a previous read from (use the reported total_chars/truncated to page). Default 0."},
                },
                "required": ["path"],
            },
            handler=self._inspect_artifact,
        )

        self.register_tool(
            name="execute_command",
            description=(
                "Execute a shell command and inspect its real output and exit code -- this "
                "is how you verify your own work (run the project's tests, typecheck, or "
                "build after edits and READ the result before claiming success). Prefer "
                "the project's own documented commands (check AGENTS.md/README, package "
                "scripts, Makefile, pyproject) over invented ones; run check/test/build "
                "commands after every non-trivial edit. To run in a subdirectory, pass "
                "cwd -- do not chain `cd <dir> && ...` into the command string. For "
                "In read-only audit/inspect/plan mode, use this only for process inspection "
                "with `ps`/`pgrep` and optional `grep`/`rg`/`head`/`tail`/`sort`/`uniq` filters; "
                "writes, interpreters, file reads, and process control, "
                "and process control are rejected. Long-running commands must not be daemonized with nohup/setsid/disown or "
                "a trailing '&'. If a command is already running, inspect it and leave it "
                "alone; do not kill/restart it. Use the existing project supervisor/queue "
                "for persistence, or let the user press Ctrl+B to move this exact command "
                "to Tamfis-Code's tracked background-job handle."
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
            name="knowledge_base_search",
            description=(
                "Search TamfisGPT's shared research corpus (real papers/sources it has already "
                "acquired and indexed from its own research feature) for passages relevant to a "
                "query -- a second, complementary source of research evidence alongside web_search, "
                "not a replacement for it. Only works on a host running TamfisGPT (calls its local "
                "internal API); returns a clear error, not a crash, if that's unavailable -- fall "
                "back to web_search in that case. Read-only, no side effects."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 30,
                        "description": "Maximum number of sources to return (default 10)",
                    },
                },
                "required": ["query"],
            },
            handler=self._knowledge_base_search,
        )

        self.register_tool(
            name="knowledge_base_index",
            description=(
                "Add a source you found (via web_search or elsewhere) into TamfisGPT's shared "
                "research corpus, so future research (from this tool or TamfisGPT's own research "
                "feature) can find it via knowledge_base_search too. Only index sources actually "
                "relevant to the research at hand, not every page visited. Only works on a host "
                "running TamfisGPT; returns a clear error, not a crash, if that's unavailable."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Source title"},
                    "url": {"type": "string", "description": "Source URL"},
                    "text": {"type": "string", "description": "The source's relevant text content to index"},
                },
                "required": ["title", "text"],
            },
            handler=self._knowledge_base_index,
        )

        self.register_tool(
            name="memory_search",
            description=(
                "Semantic recall over YOUR OWN saved memory notes (memory_remember's store, "
                "backed by TamfisGPT's vector memory) -- meaning-matched, not keyword-matched, "
                "so 'how do we run the tests again' finds the note that says 'pytest via "
                ".venv/bin/python'. Use it at the START of a task to recall prior gotchas, "
                "conventions, build commands, or user preferences before re-deriving them. "
                "Optionally filter to one project. Distinct from knowledge_base_search (shared "
                "research corpus of external papers) and from save_memory (local key-value "
                "records -- prefer memory_remember when semantic recall matters). Only works on "
                "a host running TamfisGPT; returns a clear error, not a crash, if that's "
                "unavailable. Read-only, no side effects."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to recall, phrased naturally"},
                    "project": {
                        "type": "string",
                        "description": "Only return memories saved for this project name (default: no filter)",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "description": "Maximum memories to return (default 8)",
                    },
                },
                "required": ["query"],
            },
            handler=self._memory_search,
        )

        self.register_tool(
            name="memory_remember",
            description=(
                "Save a memory note into TamfisGPT's vector memory so memory_search can recall it "
                "by MEANING in any future session (e.g. 'pytest lives in .venv/bin/python; never "
                "bare python', 'user prefers PRs under 300 lines', 'deploy needs TAMGPT_TIER_IV_URL "
                "set'). A stable memory_id makes re-saving overwrite the old note -- correct "
                "instead of duplicate. Keep each note short and self-contained. Distinct from "
                "save_memory (local key-value records, no semantic recall) and knowledge_base_index "
                "(external research sources, not your own learnings). Only works on a host running "
                "TamfisGPT; returns a clear error, not a crash, if that's unavailable."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The memory note itself -- short, self-contained"},
                    "memory_id": {
                        "type": "string",
                        "description": "Stable id so re-saving replaces the old note, e.g. '<project>:<topic>' -- omit for a fresh note",
                    },
                    "kind": {
                        "type": "string",
                        "description": "Note type: note (default), command, preference, gotcha, convention",
                    },
                    "project": {
                        "type": "string",
                        "description": "Project name to scope this memory to (recommended; enables project-filtered recall)",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional short tags",
                    },
                },
                "required": ["text"],
            },
            handler=self._memory_remember,
        )

        self.register_tool(
            name="list_external_agent_sessions",
            description=(
                "List coding sessions recorded on this machine by OTHER AI coding agents -- "
                "Claude Code, Codex CLI, GitHub Copilot CLI, OpenCode, Kimi Code -- read-only, "
                "newest first. Use this when the user asks to continue, pick up, or finish work "
                "they started in one of those tools (e.g. \"continue what Codex was doing\", "
                "\"pick up where Claude Code left off\") instead of asking them to re-explain the "
                "task. Defaults to sessions recorded for this same workspace; pass all_workspaces "
                "to search every directory on this machine. Follow up with "
                "read_external_agent_session on whichever session id looks right before acting on "
                "it -- this only returns titles/metadata, not the conversation content."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "tool": {
                        "type": "string",
                        "description": "Restrict to one tool (claude-code, codex, copilot, opencode, kimi-code). Omit to search all of them.",
                    },
                    "all_workspaces": {
                        "type": "boolean",
                        "description": "Search every workspace on this machine instead of just the current one. Default false.",
                    },
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50, "description": "Maximum sessions to return (default 20)."},
                },
            },
            handler=self._list_external_agent_sessions,
        )

        self.register_tool(
            name="read_external_agent_session",
            description=(
                "Read one session's recovered transcript from another AI coding agent (see "
                "list_external_agent_sessions for how to find the tool/session_id pair). Returns "
                "the recent turns of that conversation so you can understand what was being "
                "worked on and continue it -- but nothing it claims was already done should be "
                "trusted without re-verifying against the actual repository state; the transcript "
                "may be stale, partial, or from a different branch/checkout."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "tool": {"type": "string", "description": "The tool name from list_external_agent_sessions, e.g. \"codex\""},
                    "session_id": {"type": "string", "description": "The session id from list_external_agent_sessions"},
                },
                "required": ["tool", "session_id"],
            },
            handler=self._read_external_agent_session,
        )

        self.register_tool(
            name="list_agent_types",
            description=(
                "List built-in and configured subagent types available to this workspace. "
                "Use an exact returned name as agent_type when delegating; never invent a tool "
                "name such as review_agent. Read-only."
            ),
            parameters={"type": "object", "properties": {}},
            handler=self._list_agent_types,
        )

        self.register_tool(
            name="ask_user_question",
            description=(
                "Stop and ask the person at the terminal a real decision, with concrete options "
                "they pick with the arrow keys. Use it when the answer is genuinely theirs to "
                "give and guessing would be costly: (1) BEFORE any hard-to-reverse or "
                "outward-facing action -- deleting or overwriting data, restarting or taking a "
                "service offline (which drops connections), pushing or publishing, spending "
                "money or quota; (2) at a real fork with material trade-offs (two valid "
                "approaches, two conflicting conventions); (3) when the request's scope is "
                "ambiguous in a way reading the code cannot settle. Batch related decisions into "
                "ONE call (up to 4 questions). Give each question 2-4 options, each with a "
                "one-line `description` of what choosing it does; put the option you recommend "
                "FIRST and end its label with \"(Recommended)\". The person can always type "
                "their own answer instead. Do NOT ask what a tool can tell you (read the code, "
                "run the command, check the config first), do not ask trivial or purely "
                "stylistic questions, and do not ask permission for routine work you were "
                "already asked to do. Only works in an interactive terminal; if unavailable the "
                "result says so -- then choose the reversible/safest option or stop and report "
                "the decision you need."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "description": "1-4 questions to ask together",
                        "items": {
                            "type": "object",
                            "properties": {
                                "question": {"type": "string", "description": "The full question, ending with a question mark"},
                                "header": {"type": "string", "description": "A very short label (max ~12 chars), e.g. \"Restart\""},
                                "options": {
                                    "type": "array",
                                    "description": "2-4 choices; recommended first, labelled \"(Recommended)\"",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "label": {"type": "string", "description": "Short choice text (1-5 words)"},
                                            "description": {"type": "string", "description": "What choosing this does / its trade-off"},
                                        },
                                        "required": ["label"],
                                    },
                                },
                                "multiSelect": {"type": "boolean", "description": "Allow choosing several options"},
                            },
                            "required": ["question"],
                        },
                    },
                    "question": {"type": "string", "description": "Single-question shorthand (use `questions` for choices with descriptions)"},
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Single-question shorthand: short suggested answers",
                    },
                },
                "required": [],
            },
            handler=self._ask_user_question,
        )

        self.register_tool(
            name="write_todos",
            description=(
                "Maintain the visible step-by-step plan for the CURRENT task. Call this at the "
                "start of any non-trivial task (roughly: anything needing 3+ tool calls or "
                "touching more than one file) to lay out the steps, then call it again every "
                "time a step completes or the plan genuinely changes -- NOT to narrate routine "
                "progress like 'read file' or 'run tests'. Each call REPLACES the whole list, "
                "so always send every step with its current status; never send only the step "
                "that changed. Keep each task a short imperative outcome ('Fix pagination in "
                "UserList', not 'Investigate pagination'); mark completed only steps that are "
                "actually done and verified. The user watches this list live to see what you "
                "are doing and how far along you are -- keep it honest and current."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "task": {"type": "string", "description": "Short imperative description of the step's OUTCOME"},
                                "completed": {"type": "boolean", "description": "True only when this step is finished AND verified"},
                            },
                            "required": ["task", "completed"],
                        },
                        "description": "The complete step list in execution order (replaces the previous list)",
                    },
                },
                "required": ["todos"],
            },
            handler=self._write_todos,
        )

    @staticmethod
    def _coerce_todos(todos: Any) -> List[Any]:
        """Models send `todos` as a real list, a JSON *string* of one, or a single object. A string
        used to be iterated character by character, matched nothing, and silently CLEARED the list."""
        if isinstance(todos, str):
            text = todos.strip()
            try:
                todos = json.loads(text) if text else []
            except ValueError:
                # not JSON: treat each non-empty line as one open task
                todos = [{"task": line.strip("-*• \t")} for line in text.splitlines() if line.strip("-*• \t")]
        if isinstance(todos, dict):
            todos = todos.get("todos") if isinstance(todos.get("todos"), list) else [todos]
        return list(todos) if isinstance(todos, (list, tuple)) else []

    async def _write_todos(self, todos: List[Dict[str, Any]]) -> str:
        cleaned: List[Dict[str, Any]] = []
        for item in self._coerce_todos(todos)[:50]:
            if not isinstance(item, dict):   # a stray string inside a list is dropped (contract)
                continue
            # accept the Claude-style {"content","status"} shape as well as {"task","completed"}
            task_text = str(item.get("task") or item.get("content") or "").strip()
            if not task_text:
                continue
            done = bool(item.get("completed")) or str(item.get("status") or "").lower() in {"completed", "done"}
            cleaned.append({"task": task_text[:300], "completed": done})
        if self.session_id is not None:
            try:
                from . import state as local_state
                local_state.update_task_state(self.session_id, todo_list=cleaned)
            except Exception:
                pass  # never let plan bookkeeping fail a task
        if not cleaned:
            return "Todo list cleared."
        done = sum(1 for item in cleaned if item["completed"])
        return f"Todo list updated: {done}/{len(cleaned)} steps complete."

    async def _ask_user_question(
        self, question: Optional[str] = None, options: Optional[List[Any]] = None,
        questions: Optional[List[Any]] = None,
    ) -> str:
        from . import ask_user

        try:
            parsed = ask_user.normalize_questions(questions, question, options)
        except ValueError as exc:
            return f"ask_user_question could not be shown: {exc}"
        if self._console is None or not self._interactive:
            return ask_user.unavailable_message()
        # Same ordering discipline as the approval gate: pause the live status AND
        # the live input listener, and wait until it has actually released the
        # terminal, before a new interactive prompt starts reading keys.
        from .render import resume_live_if_active, suspend_live_async_if_active

        await suspend_live_async_if_active(self._renderer)
        try:
            answers = await ask_user.ask_questions(self._console, parsed)
        finally:
            resume_live_if_active(self._renderer)
        if question and not questions:
            # The original single-question form returned just the answer; keep that.
            return answers[0].text()
        return ask_user.answers_to_text(parsed, answers)

    async def _list_external_agent_sessions(
        self, tool: Optional[str] = None, all_workspaces: bool = False, limit: int = 20,
    ) -> List[Dict[str, Any]]:
        from . import external_agents
        if tool and tool not in external_agents.known_tools():
            return [{
                "error": f"Unknown tool '{tool}'. Known tools: {', '.join(external_agents.known_tools())}.",
            }]
        sessions = external_agents.discover_external_sessions(
            workspace_root=None if all_workspaces else self.workspace_root,
            tools=[tool] if tool else None, limit=max(1, min(limit, 50)),
        )
        if not sessions:
            return []
        return [
            {
                "tool": s.tool, "session_id": s.session_id, "title": s.title,
                "cwd": s.cwd, "updated_at": s.updated_at,
            }
            for s in sessions
        ]

    async def _read_external_agent_session(self, tool: str, session_id: str) -> Dict[str, Any]:
        from . import external_agents
        if tool not in external_agents.known_tools():
            return {"error": f"Unknown tool '{tool}'. Known tools: {', '.join(external_agents.known_tools())}."}
        record = external_agents.read_external_session(tool, session_id)
        if record is None:
            return {"error": f"No '{tool}' session '{session_id}' found -- call list_external_agent_sessions again, the id may be stale."}
        return {
            "tool": record["tool"], "session_id": record["session_id"], "title": record["title"],
            "cwd": record["cwd"], "updated_at": record["updated_at"],
            "brief": external_agents.continuation_brief(record),
        }

    async def _list_agent_types(self, **_ignored: Any) -> Dict[str, Any]:
        # Older checkpoints may contain a malformed assistant tool call with
        # synthetic recovery arguments.  This is a no-argument discovery
        # operation, so safely ignore those historical arguments instead of
        # turning recovery into a second failure or provider-switch loop.
        from .agent_definitions import load_agent_definitions
        from .agents import AgentManager

        built_in = AgentManager().list_agents()
        configured = [
            {
                "name": definition.name,
                "description": definition.description,
                "source": definition.source,
            }
            for definition in load_agent_definitions(self.workspace_root)
        ]
        return {"built_in": built_in, "configured": configured}

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
        # Defense in depth for callers outside runner_local.py: provider
        # channel markers must never become part of an executable MCP name.
        # Normalize only to a locally registered tool; an unresolved marked
        # name is rejected before external MCP dispatch as well.
        from .provider_protocols import normalize_tool_call
        canonical_name, _ = normalize_tool_call(
            name, "", allowed_names=set(self.tools),
        )
        if canonical_name in self.tools:
            name = canonical_name
        elif "<|" in str(name):
            return {
                "error": f"Unknown MCP tool: {name}",
                "tool": str(name),
                "success": False,
            }

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
        line_start: Optional[int] = None, line_end: Optional[int] = None,
    ) -> str:
        # Several MCP clients use the more descriptive line_start/line_end
        # vocabulary. Normalize it at the tool boundary so providers can use
        # either schema without producing an unexpected-keyword failure.
        if offset is None and line_start is not None:
            offset = line_start
        if limit is None and line_end is not None and offset is not None:
            try:
                limit = int(line_end) - int(offset) + 1
            except (TypeError, ValueError):
                return "Error: read_file line_start and line_end must be positive integers"
        p = self._resolve_readable_input(path)
        resolved_note = ""
        if not p.exists():
            recovered, candidates = self._recover_missing_read_path(path)
            if recovered is not None:
                p = recovered
                resolved_note = (
                    f"[Resolved requested path '{path}' to workspace file "
                    f"'{self._display_workspace_path(p)}' after complete tree discovery.]\n"
                )
            elif candidates:
                rendered = ", ".join(f"'{item}'" for item in candidates)
                return (
                    f"Error: File '{path}' was not found at that exact path and the workspace "
                    f"search found multiple possible files: {rendered}. Do not guess; retry "
                    "read_file with one of these exact paths."
                )
            else:
                # Only offer a typo suggestion after the complete tree search
                # has failed. Otherwise a shallow sibling such as
                # tamgpt_init.py can mask the real nested tamgpt_api.py.
                corrected = self._correct_path_typos(path)
                if corrected and self._is_allowed_read_path(Path(corrected)):
                    return f"Error: File '{path}' not found. Did you mean '{corrected}'?"
                return f"Error: File '{path}' not found.{self._not_found_hint(path)}"
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
                "models -- look at it there instead of calling read_file. For a ZIP/TAR archive, "
                "use read_archive (list or read members with no size limit, nested to any depth); "
                "extract_archive only when files must be edited or run."
            )
        content = p.read_text(encoding='utf-8', errors='ignore')
        lines = content.splitlines(keepends=True)
        default_page_lines = 800
        requested_page = offset is not None or limit is not None
        if not requested_page and len(lines) <= default_page_lines:
            return resolved_note + content
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
        return resolved_note + f"[Showing lines {start}-{max(end, start)} of {len(lines)}.{continuation}]\n{numbered}"

    def _display_workspace_path(self, path: Path) -> str:
        """Render a recovered path without leaking an unrelated host path."""
        base = Path(self.workspace_root).expanduser().resolve() if self.workspace_root else Path.cwd().resolve()
        try:
            return str(path.resolve().relative_to(base))
        except ValueError:
            return str(path)

    def _is_allowed_read_path(self, path: Path) -> bool:
        try:
            resolved = path.expanduser().resolve()
            roots = self._recovery_roots()
            return any(resolved == root or root in resolved.parents for root in roots)
        except OSError:
            return False

    def _recovery_roots(self) -> list[Path]:
        base = Path(self.workspace_root).expanduser().resolve() if self.workspace_root else Path.cwd().resolve()
        roots = self.allowed_workspace_roots or {base}
        return sorted(
            {root.resolve() for root in roots if root.exists() and root.is_dir()},
            key=lambda item: str(item),
        )

    def _recover_missing_read_path(self, requested: str) -> tuple[Optional[Path], list[str]]:
        """Find a missing read path using the already-authorised workspace tree.

        Recovery is intentionally conservative: an exact relative suffix wins
        only when unique; otherwise a basename match may be used only when it
        is unique. Ambiguous matches are returned to the caller so the model
        must choose an exact path rather than silently reading the wrong file.
        Symlinks and excluded/generated directories stay inside the same
        allowed roots and are never followed outside the scope.
        """
        requested_path = Path(str(requested).strip()).expanduser()
        requested_name = requested_path.name
        if not requested_name or requested_name in {".", ".."}:
            return None, []
        requested_parts = tuple(part for part in requested_path.parts if part not in {"", ".", "..", "/"})
        if not requested_parts:
            return None, []
        roots = self._recovery_roots()
        if not roots:
            return None, []

        # Prefer the already-named parent directory. This is both faster than
        # walking a workspace rooted at /home and handles a common agent error
        # where a status/log prefix is invented (`moe_pretraining.status`) but
        # the discovered directory contains the canonical `status` file. Only
        # a unique direct child is accepted; no arbitrary sibling is chosen.
        try:
            requested_parent = requested_path.parent
            if not requested_parent.is_absolute():
                requested_parent = (Path(self.workspace_root or os.getcwd()) / requested_parent)
            requested_parent = requested_parent.resolve()
            if self._is_allowed_read_path(requested_parent) and requested_parent.is_dir():
                direct_files = sorted(
                    (item.resolve() for item in requested_parent.iterdir() if item.is_file()),
                    key=lambda item: str(item),
                )
                tail = requested_name.rsplit(".", 1)[-1] if "." in requested_name else ""
                direct_tail = [item for item in direct_files if tail and item.name == tail]
                if len(direct_tail) == 1:
                    return direct_tail[0], []
        except OSError:
            pass

        suffix_matches: list[Path] = []
        basename_matches: list[Path] = []
        seen: set[Path] = set()
        for root in roots:
            try:
                for current, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
                    current_path = Path(current)
                    dirnames[:] = [
                        name for name in dirnames
                        if name not in self._PATH_SUGGESTION_SKIP_DIRS
                        and name not in EXCLUDED_DIR_NAMES
                        and not name.endswith(".egg-info")
                    ]
                    for filename in filenames:
                        candidate = (current_path / filename).resolve()
                        if candidate in seen or not candidate.is_file():
                            continue
                        if not any(candidate == allowed or allowed in candidate.parents for allowed in roots):
                            continue
                        seen.add(candidate)
                        relative_parts = tuple(candidate.relative_to(root).parts) if candidate.is_relative_to(root) else ()
                        if filename == requested_name:
                            basename_matches.append(candidate)
                            if relative_parts and len(relative_parts) >= len(requested_parts):
                                if relative_parts[-len(requested_parts):] == requested_parts:
                                    suffix_matches.append(candidate)
            except OSError:
                continue

        def unique(paths: list[Path]) -> list[Path]:
            return list(dict.fromkeys(paths))

        suffix_matches = unique(suffix_matches)
        basename_matches = unique(basename_matches)
        if len(suffix_matches) == 1:
            return suffix_matches[0], []
        candidates = suffix_matches if suffix_matches else basename_matches
        if len(candidates) > 1:
            active = [item for item in candidates if not self._is_archival_recovery_path(item)]
            if len(active) == 1:
                return active[0], []
        if len(candidates) == 1:
            return candidates[0], []
        rendered = [self._display_workspace_path(item) for item in candidates[:MAX_READ_RECOVERY_CANDIDATES]]
        return None, rendered

    @staticmethod
    def _is_archival_recovery_path(path: Path) -> bool:
        """Identify duplicate copies that should not outrank live source."""
        archival_names = {"backup", "backups", ".backup", "dist", "build", "__pycache__"}
        return any(part.lower() in archival_names for part in path.parts)

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
        from .path_utils import clean_path_argument

        base = Path(self.workspace_root) if self.workspace_root else Path.cwd()
        p = Path(clean_path_argument(path))
        if not p.is_absolute():
            p = base / p
        resolved = p.resolve()
        # Models often include the workspace directory name in a relative
        # path after seeing it in a listing (for example, ``finitron/README.md``
        # while the active workspace is already ``/home/finitron``).  Do not
        # turn that valid workspace-relative reference into the impossible
        # ``/home/finitron/finitron/README.md``.  Only accept the de-prefixed
        # form when it resolves inside an authorised root and the requested
        # spelling does not, so ordinary nested directories remain unchanged.
        if self.workspace_root and not Path(clean_path_argument(path)).is_absolute():
            relative = Path(clean_path_argument(path))
            base_resolved = base.resolve()
            if relative.parts and relative.parts[0] == base_resolved.name:
                candidate = (base_resolved.joinpath(*relative.parts[1:])).resolve()
                roots = self.allowed_workspace_roots or {base_resolved}
                candidate_allowed = any(
                    candidate == root or root in candidate.parents
                    for root in roots
                )
                if candidate_allowed and not resolved.exists() and candidate.exists():
                    resolved = candidate
        # The same duplication can arrive as an absolute path when a model
        # combines the workspace shown by the UI with a path already prefixed
        # by that workspace name (``/home/finitron/finitron/README.md``).
        # Normalize one duplicated root component before invoking bounded
        # recovery.  This is safe because the candidate must already exist
        # beneath an authorised workspace root; genuinely ambiguous files
        # still go through the explicit ambiguity diagnostic.
        if self.workspace_root and Path(clean_path_argument(path)).is_absolute():
            base_resolved = base.resolve()
            raw_resolved = Path(clean_path_argument(path)).resolve()
            try:
                relative_parts = raw_resolved.relative_to(base_resolved).parts
            except ValueError:
                relative_parts = ()
            if relative_parts and relative_parts[0] == base_resolved.name:
                candidate = (base_resolved.joinpath(*relative_parts[1:])).resolve()
                roots = self.allowed_workspace_roots or {base_resolved}
                candidate_allowed = any(
                    candidate == root or root in candidate.parents
                    for root in roots
                )
                if candidate_allowed and not resolved.exists() and candidate.exists():
                    resolved = candidate
        if self.workspace_root:
            roots = self.allowed_workspace_roots or {base.resolve()}
            if not any(resolved == root or root in resolved.parents for root in roots):
                rendered = ", ".join(str(root) for root in sorted(roots, key=str))
                raise PermissionError(
                    f"'{path}' resolves outside the workspace; approved roots: ({rendered})"
                )
        return resolved

    _PATH_SUGGESTION_SKIP_DIRS = {
        ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
        "dist", "build", ".mypy_cache", ".pytest_cache", ".ruff_cache",
        ".idea", ".vscode", ".tox", "htmlcov",
    }

    def _suggest_similar_paths(self, missing_name: str, *, limit: int = 5) -> list[str]:
        """Complete workspace-relative search for files sharing `missing_name`'s
        basename.

        A bare "File 'X' not found" invites the same wrong guess again --
        confirmed live: a model asked to read a project file guessed an
        absolute host path that didn't exist and had no way to recover the
        real, workspace-relative one. This gives read_file/edit_file's
        not-found error something concrete to point at instead of leaving
        "resolve the canonical path" (orchestrator/repair.py's own repair
        strategy for this failure class) as an instruction with no tool
        support behind it. Dependency and generated directories are skipped,
        but there is no depth or file-count cutoff.
        """
        base = Path(self.workspace_root) if self.workspace_root else Path.cwd()
        target_name = Path(missing_name).name
        if not target_name or not base.is_dir():
            return []
        matches: list[str] = []
        for root, dirnames, filenames in os.walk(base):
            dirnames[:] = [
                d for d in dirnames
                if d not in self._PATH_SUGGESTION_SKIP_DIRS and not d.endswith(".egg-info")
            ]
            for name in filenames:
                if name == target_name:
                    try:
                        rel = str(Path(root, name).relative_to(base))
                    except ValueError:
                        rel = str(Path(root, name))
                    matches.append(rel)
                    if len(matches) >= limit:
                        return matches
        return matches

    def _correct_path_typos(self, path: str) -> Optional[str]:
        """The existing path this one is most likely a misspelling of, or None.

        Users (and models copying them) mistype names -- "/home/tmafisseo/.../caompgns/caompgns.php" for
        /home/tamfisseo/.../campaigns/campaigns.php -- and a not-found that only says "search for the right
        path" sends the model hunting for a name that cannot exist. Walk the path from the root; at the first
        component that is missing, take the closest real sibling name. Bounded: one directory listing per
        misspelled component, huge directories skipped, and it only answers when EVERY missing component has a
        close match."""
        import difflib

        try:
            raw = Path(path)
            target = raw if raw.is_absolute() else Path(self.workspace_root or os.getcwd()) / raw
            parts = target.parts
            current = Path(parts[0])
            changed = False
            for part in parts[1:]:
                candidate = current / part
                if candidate.exists():
                    current = candidate
                    continue
                if not current.is_dir():
                    return None
                names = os.listdir(current)
                if len(names) > 5000:
                    return None
                match = difflib.get_close_matches(part, names, n=1, cutoff=0.78)
                if not match:
                    return None
                current = current / match[0]
                changed = True
            return str(current) if changed and current.exists() else None
        except (OSError, ValueError):
            return None

    def _written_earlier_this_session(self, path: str) -> bool:
        """True when the session's mutation ledger says this session wrote ``path`` (it has since been
        deleted or moved outside the session)."""
        if self.session_id is None:
            return False
        try:
            from . import state as local_state

            wanted = str(self._resolve_in_workspace(path)) if self.workspace_root else str(path)
            return any(
                str(entry.get("path")) in {wanted, str(path)}
                for entry in local_state.get_session_state(self.session_id).modified_files
            )
        except Exception:
            return False

    def _not_found_hint(self, path: str) -> str:
        if self._written_earlier_this_session(path):
            # The generic hint below ("use list_directory or search_code to find the right path")
            # sent a model that had written this file round the same read/list loop indefinitely.
            return (
                " This session wrote that file earlier, but it no longer exists (deleted or moved outside "
                "this session). Do not keep searching for it: recreate it with write_file only if it is "
                "still needed, otherwise continue without it."
            )
        corrected = self._correct_path_typos(path)
        if corrected:
            return f" A name in that path looks misspelled. Did you mean '{corrected}'?"
        suggestions = self._suggest_similar_paths(path)
        if suggestions:
            return f" Found '{Path(path).name}' at: {', '.join(suggestions)}."
        if self.workspace_root:
            return (
                f" Workspace root is '{self.workspace_root}'; paths are resolved relative to "
                "it. Use list_directory or search_code to find the right path."
            )
        return ""

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

    @staticmethod
    def _is_source_file(path: Path) -> bool:
        return path.suffix.lower() in {
            ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".java", ".go",
            ".rs", ".rb", ".php", ".c", ".cc", ".cpp", ".h", ".hpp", ".cs",
            ".swift", ".kt", ".kts", ".scala", ".sql", ".sh", ".bash", ".zsh",
            ".css", ".scss", ".html", ".vue", ".svelte", ".toml", ".yaml", ".yml",
            ".json",
        }

    def _reject_unverified_source_replacement(
        self, path: Path, original_content: Optional[str], expected_sha256: Optional[str], *, operation: str,
    ) -> Optional[str]:
        """Prevent an agent turn from blindly replacing an existing source file.

        A full replacement is safe for a new file, but dangerous for an existing
        source file because a truncated or stale model response can erase
        unrelated code. The live agent must provide the digest obtained during
        its diagnostic read; targeted ``edit_file`` remains the normal path.
        This applies to every caller; there is no unguarded full-replacement
        escape hatch for an existing source file.
        """
        if original_content is None or not self._is_source_file(path):
            return None
        actual = hashlib.sha256(original_content.encode("utf-8")).hexdigest()
        supplied = str(expected_sha256 or "").strip().lower()
        if not supplied:
            return (
                f"❌ Refused unverified {operation} of existing source file '{path}'. "
                "Read/diagnose it first and provide expected_sha256, or use edit_file "
                "with an exact unique old_string; no existing code was changed."
            )
        if supplied != actual:
            return (
                f"❌ Refused stale {operation} of '{path}': expected_sha256 does not match "
                "the current file. Re-read and re-diagnose it; no existing code was changed."
            )
        return None

    async def _write_file(
        self, path: str, content: str | None = None, mode: str | None = None,
        **aliases: Any,
    ) -> str:  # noqa: D401 - see the mode comment inside
        content = content if content is not None else aliases.pop("text", None)
        content = content if content is not None else aliases.pop("new_content", None)
        content = content if content is not None else aliases.pop("file_content", None)
        if content is None:
            return "❌ Error: write_file requires content"
        # mode="append" is how a LARGE document gets written at all: a single
        # tool call's arguments are bounded by the model's output token limit,
        # so one oversized write_file arrives truncated (see
        # salvage_truncated_tool_arguments in runner_local.py). Writing the
        # document in parts -- an initial write, then appends -- keeps every
        # call inside the limit. Anything other than "append" (the default,
        # including a missing value) means overwrite, so an old caller that
        # never heard of `mode` behaves exactly as before.
        append = str(mode or "write").strip().lower() == "append"
        p = self._resolve_in_workspace(path)
        original_content = p.read_text(encoding="utf-8", errors="ignore") if p.is_file() else None
        source_replacement = original_content is not None and self._is_source_file(p) and not append
        if source_replacement and str(mode or "write").strip().lower() != "overwrite":
            return (
                f"❌ Refused destructive replacement of existing source file '{path}'. "
                "Use edit_file with an exact unique old_string for a patch, or explicitly "
                "request mode=overwrite together with expected_sha256 after diagnostics; "
                "no existing code was changed."
            )
        if not append:
            guard_error = self._reject_unverified_source_replacement(
                p, original_content, aliases.pop("expected_sha256", None), operation="write"
            )
            if guard_error:
                return guard_error
        if append and original_content is not None:
            content = original_content + content
        # No-op guard (confirmed live 2026-09-24: a model "edited" a file it
        # had already put into its requested state, the tool rewrote identical
        # bytes, recorded a +0/-0 "mutation", and printed "✅ Edited" -- the
        # validator then counted that fabricated mutation as progress
        # evidence). Identical content is a truthful result, never a mutation.
        if original_content is not None and content == original_content:
            if append:
                return f"ℹ Nothing appended to '{path}': the new content is byte-identical to the file."
            return (
                f"ℹ No changes written to '{path}': the provided content is byte-identical "
                "to the current file. Do not re-issue this write; the requested state "
                "already exists -- verify it and continue."
            )
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
        if append and original_content is not None:
            return (
                f"✅ Appended {len(content) - len(original_content)} bytes to '{path}' "
                f"(file is now {len(content)} bytes)"
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
            return await self._write_file(
                path, content=full_content,
                mode=aliases.pop("mode", None),
                expected_sha256=aliases.pop("expected_sha256", None),
            )
        if old_string is None or new_string is None:
            return "❌ Error: edit_file requires old_string and new_string, or content for full replacement"
        p = self._resolve_in_workspace(path)
        if not p.is_file():
            return f"❌ Error: File '{path}' not found.{self._not_found_hint(path)}"
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
        # No-op guard: identical bytes (old_string == new_string, or the
        # replacement round-trips to the same content) must not rewrite the
        # file or record a +0/-0 mutation as if work had happened -- the
        # live auto-blog session showed exactly that being counted as edit
        # evidence. Report the truth and point the model at the next action.
        if new_content == original_content:
            return (
                f"ℹ No changes made to '{path}': the replacement produces content identical "
                "to the current file (old_string == new_string, or the section is already "
                "in the requested state). Do not repeat this edit; verify the current "
                "content and move to the next step."
            )
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
    
    async def _list_directory(
        self, path: str = ".", depth: int = 1,
    ) -> List[Dict[str, Any]]:
        """List a directory tree.

        Older callers omit ``depth`` and retain the original immediate-child
        behavior. Newer models sometimes include it automatically in a tool
        call, so accepting it here prevents a schema/handler mismatch from
        terminating the task. ``depth=0`` means unlimited recursion. The
        result count remains bounded independently so a broad listing does not
        flood the model context in one response.
        """
        try:
            p = self._resolve_in_workspace(path)
        except PermissionError as exc:
            return [{"error": str(exc)}]
        if not p.exists():
            corrected = self._correct_path_typos(path)
            hint = f" A name in that path looks misspelled. Did you mean '{corrected}'?" if corrected else ""
            return [{"error": f"Directory '{path}' not found.{hint}"}]
        if not p.is_dir():
            return [{"error": f"'{path}' is not a directory"}]
        try:
            requested_depth = int(depth)
        except (TypeError, ValueError):
            return [{"error": "depth must be an integer"}]
        if requested_depth < 0:
            return [{
                "error": "depth must be a non-negative integer"
            }]

        unlimited = requested_depth == 0
        walk_depth = None if unlimited else requested_depth

        results: list[Dict[str, Any]] = []
        excluded_count = 0
        omitted_count = 0

        async def visit(directory: Path, remaining: int | None) -> None:
            nonlocal excluded_count, omitted_count
            # Recursive filesystem calls through the generic worker-thread
            # wrapper can deadlock on Python 3.13 after the first nested
            # directory operation. Keep this bounded walk cooperative on the
            # event loop instead: yield between entries so prompt input,
            # cancellation, and status events remain responsive.
            await asyncio.sleep(0)
            try:
                children = sorted(directory.iterdir(), key=lambda item: item.name)
            except OSError:
                return
            for item in children:
                await asyncio.sleep(0)
                try:
                    is_dir = item.is_dir()
                    if is_dir and item.name in EXCLUDED_DIR_NAMES:
                        excluded_count += 1
                        continue
                    if len(results) >= MAX_LIST_DIRECTORY_ENTRIES:
                        omitted_count += 1
                        continue
                    entry: Dict[str, Any] = {
                        "name": item.name,
                        "path": str(item),
                        "is_file": item.is_file(),
                        "is_dir": is_dir,
                        "size": item.stat().st_size if item.exists() else 0,
                        "modified": item.stat().st_mtime if item.exists() else 0,
                    }
                    if not unlimited and remaining < requested_depth:
                        entry["depth"] = requested_depth - remaining + 1
                    results.append(entry)
                    if is_dir and (unlimited or remaining > 1):
                        await visit(item, None if unlimited else remaining - 1)
                except OSError:
                    # A disappearing or unreadable child should not invalidate
                    # the rest of a useful directory listing.
                    continue

        try:
            await asyncio.wait_for(visit(p, walk_depth), timeout=30.0)
        except asyncio.TimeoutError:
            return [{"error": (
                f"Listing '{path}' took longer than 30s and was stopped. "
                "Use a narrower path or search_code for a targeted query."
            )}]
        # Recursive traversal is emitted in path order so the result remains
        # deterministic even when directory iteration order differs by host.
        results = sorted(results, key=lambda x: str(x.get("path", "")))
        total = len(results) + omitted_count
        if omitted_count:
            results.append({
                "truncated": True,
                "note": f"{omitted_count} more entrie(s) omitted "
                        f"(showing first {len(results)} of {total}). "
                        "Narrow the path or use search_code for a targeted query.",
            })
        if excluded_count:
            results.append({
                "excluded": True,
                "note": f"{excluded_count} ignored subdirectory name(s) not listed "
                        f"({', '.join(sorted(EXCLUDED_DIR_NAMES))}, when present).",
            })
        return results

    async def _search_code(
        self, query: str, path: str = ".", file_pattern: str = None,
        offset: Optional[int] = None, max_results: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Search for `query`, returning ONE PAGE of matches.

        A broad query in a large repository produces far more matches than a
        model can use (and more than one tool result should carry), so the
        result is paged exactly like read_file: the page is followed by a
        `pagination` entry naming the range, the total, and the `next_offset`
        to continue from -- rather than only telling the model to narrow the
        query, which throws away the fact that the answer is ON match 120.
        """
        if max_results is None:
            max_results = limit
        matches = await self._search_code_matches(query, path, file_pattern)
        if matches and isinstance(matches[0], dict) and matches[0].get("error"):
            return matches
        # The "the pool itself stopped here" marker describes the search, not
        # one of its matches: keep it out of the paging arithmetic (it must not
        # be counted as a match or pushed onto a later page) and re-attach it
        # to every page.
        pool_marker = None
        if matches and isinstance(matches[-1], dict) and matches[-1].get("truncated"):
            pool_marker = matches[-1]
            matches = matches[:-1]
        page = _page_search_matches(matches, offset=offset, max_results=max_results)
        return [*page, pool_marker] if pool_marker is not None else page

    async def _search_code_matches(
        self, query: str, path: str = ".", file_pattern: str = None,
    ) -> List[Dict[str, Any]]:
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
                if len(matches) >= _SEARCH_POOL_LIMIT:
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

            if len(matches) >= _SEARCH_POOL_LIMIT:
                matches.append({
                    "truncated": True,
                    "note": f"There are more than {_SEARCH_POOL_LIMIT} matches; the pool "
                            "stops here. Narrow the query (a more specific pattern, a "
                            "file_pattern glob, or a deeper path) to see the rest.",
                })

            return _sorted_search_matches(matches)
        except FileNotFoundError:
            # `rg` is fast and preferred, but it is not part of Python and is
            # absent from some minimal servers and hosted CI images. A
            # portable install must retain search functionality without a
            # host-specific binary, so use the same bounds and exclusions in
            # a small standard-library fallback.
            deadline = time.monotonic() + 30.0
            try:
                return await run_blocking_bounded(
                    lambda: self._search_code_python(query, resolved_path, file_pattern, deadline=deadline),
                    timeout=35.0,
                )
            except asyncio.TimeoutError:
                return [{"error": "Search timed out"}]

    @staticmethod
    def _search_code_python(
        query: str, root: Path, file_pattern: Optional[str], *, deadline: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        try:
            matcher = re.compile(query)
        except re.error as exc:
            return [{"error": f"Invalid search pattern: {exc}"}]
        matches: List[Dict[str, Any]] = []
        try:
            paths = [root] if root.is_file() else sorted(root.rglob("*"))
            for candidate in paths:
                if len(matches) >= _SEARCH_POOL_LIMIT:
                    break
                if deadline is not None and time.monotonic() > deadline:
                    matches.append({"truncated": True, "note": "Search stopped at its time limit; narrow the path or query."})
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
                            if len(matches) >= _SEARCH_POOL_LIMIT:
                                break
                except (OSError, UnicodeError):
                    continue
        except OSError as exc:
            return [{"error": str(exc)}]
        if len(matches) >= _SEARCH_POOL_LIMIT:
            matches.append({
                "truncated": True,
                "note": f"There are more than {_SEARCH_POOL_LIMIT} matches; narrow the query or path.",
            })
        return _sorted_search_matches(matches)

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

                def _index_and_lookup() -> List[Dict[str, Any]]:
                    indexer.index()
                    return [
                        {"name": sym.name, "kind": sym.kind, "file": sym.file_path, "line": sym.line_start}
                        for sym in indexer.search_symbol(symbol)
                        if sym.name == symbol  # search_symbol matches substrings; only exact names are real definitions
                    ]

                # Whole-tree indexing used to run synchronously on the event
                # loop -- pointed at a large tree it froze the terminal for
                # minutes. Bounded and off-loop now; on timeout the textual
                # reference search below still answers.
                definitions = await run_blocking_bounded(_index_and_lookup, timeout=20.0)
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

    async def _read_archive(
        self, path: str, member: str = "", pattern: str = "",
        offset: Optional[int] = None, limit: Optional[int] = None,
    ) -> str:
        from .archive_reader import read_archive, split_chain

        chain = split_chain(path)
        if not chain:
            return "Error: read_archive needs an archive path"
        try:
            root = self._resolve_readable_input(chain[0])
        except (PermissionError, FileNotFoundError) as exc:
            return f"Error: {exc}"
        if not root.is_file():
            return f"Error: Archive not found: {chain[0]}"
        try:
            start = int(offset or 1)
            page = int(limit) if limit else None
        except (TypeError, ValueError):
            return "Error: read_archive offset and limit must be positive integers"
        return await run_blocking_bounded(
            lambda: read_archive(root, chain[1:], str(member or ""), pattern=str(pattern or ""), offset=start, limit=page),
            timeout=150.0,
        )

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
        # Never report an artifact as complete from the in-memory helper
        # result alone.  The agent and TamfisGPT both rely on this boundary
        # as the evidence that a real file exists before presenting a link or
        # continuing with inspection/archive work.
        if not target.is_file() or target.stat().st_size <= 0:
            raise IOError(f"Artifact generation did not produce a readable file: {target}")
        result = {
            **(result if isinstance(result, dict) else {}),
            "success": True,
            "path": str(target),
            "size_bytes": target.stat().st_size,
            "verified": True,
        }
        if self.session_id is not None:
            from .safety import record_mutation
            record_mutation(
                self.session_id, path=str(target), operation="update" if existed else "create",
                original_content=None, new_content=None, transaction_id=self.transaction_id,
            )
        return result

    async def _inspect_artifact(
        self, path: str, max_chars: int = 30_000, offset: int = 0,
    ) -> Dict[str, Any]:
        from .artifacts import inspect_artifact
        source = self._resolve_readable_input(path)
        if not source.is_file():
            return {"success": False, "error": f"Artifact not found: {path}"}
        try:
            limit = min(max(int(max_chars), 1_000), 100_000)
        except (TypeError, ValueError):
            limit = 30_000
        try:
            skip = max(int(offset), 0)
        except (TypeError, ValueError):
            skip = 0
        return inspect_artifact(source, max_chars=limit, offset=skip)
    
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
        _wp_root_retry: bool = False,
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
            # Commands default to the approved workspace. An explicit
            # require_escalated arrives only after the runner's user approval
            # gate for an external scope, so it is the narrow capability that
            # permits diagnostics such as `wp option get` in /srv/site or
            # `cat /etc/cron.d/...`. Do not make the whole MCP server
            # unrestricted: without this flag the existing workspace boundary
            # remains fail-closed.
            run_dir = self._resolve_in_workspace(cwd or ".")
        except PermissionError as e:
            if sandbox_permissions != "require_escalated":
                return {"error": str(e), "success": False}
            external_cwd = Path(cwd or ".").expanduser()
            if not external_cwd.is_absolute():
                return {"error": str(e), "success": False}
            run_dir = external_cwd.resolve()
        if not run_dir.is_dir():
            return {"error": f"cwd '{cwd}' is not a directory", "success": False}

        # Validation and test commands are package workloads, not interactive
        # shell snippets.  Raise a model-supplied short default (for example
        # 120s) to the workload-aware package floor before wait_for() starts.
        timeout = adaptive_command_timeout(command, run_dir, timeout)

        # A plain `cat file...` is a bounded read, not a shell workload. Some
        # approved external reads (notably /etc/cron.d files) have sporadically
        # spent the whole shell timeout in the approval/sandbox path even
        # though the same file is immediately readable by the host. Resolve
        # every operand through the normal workspace boundary and read it
        # directly; shell features continue through the normal policy path.
        try:
            cat_argv = shlex.split(command)
        except ValueError:
            cat_argv = []
        if cat_argv and cat_argv[0] == "cat" and len(cat_argv) > 1 and all(
            not item.startswith("-") for item in cat_argv[1:]
        ):
            chunks: list[bytes] = []
            try:
                for operand in cat_argv[1:]:
                    try:
                        path = self._resolve_in_workspace(operand)
                    except PermissionError:
                        if sandbox_permissions != "require_escalated" or not Path(operand).expanduser().is_absolute():
                            raise
                        path = Path(operand).expanduser().resolve()
                    if not path.is_file():
                        return {"error": f"cat: {operand}: not a regular file", "success": False}
                    chunks.append(path.read_bytes())
            except (OSError, PermissionError) as exc:
                return {"error": str(exc), "success": False}
            return {
                "stdout": b"".join(chunks).decode("utf-8", errors="replace"),
                "stderr": "", "return_code": 0, "success": True,
                "sandbox": {
                    "active": False, "backend": "direct-read",
                    **({"external_scope_approved": True} if sandbox_permissions == "require_escalated" else {}),
                },
            }

        # Training and frontier jobs are durable, stateful workloads. Killing
        # one merely because the foreground wait ended, then launching a
        # duplicate through nohup, can corrupt the run's meaning and create
        # two writers for the same checkpoint directory. Keep this boundary
        # below the model's prose/tool policy so a generated shell command
        # cannot bypass it.
        lowered_command = command.lower()
        protected_training = any(
            marker in lowered_command
            for marker in ("train_frontier", "train_sft", "train2", "training_queue")
        )
        if protected_training and re.search(
            r"(?:\bkill(?:all)?\b|\bpkill\b|\b fuser\s+[^\n]*--kill\b|\bnohup\b|\bsetsid\b|\bdisown\b|(?<!&)\&(?!&)\s*$)",
            lowered_command,
        ):
            return {
                "error": (
                    "Refusing to kill or daemonize a training job from execute_command. "
                    "If it is running, leave it running and inspect its PID/log/checkpoint. "
                    "Use the repository's supervisor/queue for persistence; never launch a "
                    "duplicate writer for the same checkpoint directory."
                ),
                "success": False,
            }

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
        sandbox_command = None
        argv = (shell, "-lc", command)
        if self.sandbox_policy is not None and self.workspace_root:
            try:
                sandbox_command = build_sandbox_command(
                    command=command, shell=shell, cwd=run_dir,
                    workspace_root=Path(self.workspace_root).expanduser().resolve(),
                    policy=self.sandbox_policy,
                    require_escalated=sandbox_permissions == "require_escalated",
                )
            except RuntimeError as exc:
                if self._sandbox_unavailable_warned:
                    return {
                        "error": (
                            "OS sandbox still unavailable (see the earlier command's error "
                            "this turn for remediation) -- not retrying under kernel isolation."
                        ),
                        "success": False,
                    }
                self._sandbox_unavailable_warned = True
                return {"error": str(exc), "success": False}
            argv = sandbox_command.argv
            if sandbox_command.env_overrides:
                # Corrects HOME/USER/LOGNAME to the workspace's owning
                # account when this command is running dropped-to that uid
                # (see sandbox.py's resolve_workspace_owner) -- applied
                # after the caller's own `environment` override so this
                # always wins for these specific keys; the process uid
                # itself is already correct by this point (bwrap --uid/
                # --gid, or the runuser wrapper), this just keeps tools
                # like npm/git from resolving config/cache against root's
                # home directory despite running as a different uid.
                env.update(sandbox_command.env_overrides)
        try:
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
                    result = {
                        "stdout": stdout.decode('utf-8', errors='ignore'),
                        "stderr": stderr.decode('utf-8', errors='ignore'),
                        "return_code": proc.returncode,
                        "success": proc.returncode == 0,
                        "sandbox": _sandbox_result(sandbox_command),
                    }
                    retry_command = (
                        None if _wp_root_retry else
                        _wp_cli_root_retry_command(command, result["stderr"] or result["stdout"])
                    )
                    if retry_command:
                        recovered = await self._execute_command(
                            retry_command, cwd=cwd, timeout=timeout,
                            environment=environment, shell=shell,
                            sandbox_permissions=sandbox_permissions,
                            approval_metadata=approval_metadata,
                            background_signal=background_signal,
                            _wp_root_retry=True,
                        )
                        recovered["recovery"] = {
                            "kind": "wp_cli_root_guard",
                            "original_command": command,
                            "retry_command": retry_command,
                            "message": (
                                "WP-CLI rejected the approved read because the process was "
                                "running as root; the same read was retried with --allow-root."
                            ),
                        }
                        return recovered
                    return result
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
                result = {
                    "stdout": stdout.decode('utf-8', errors='ignore'),
                    "stderr": stderr.decode('utf-8', errors='ignore'),
                    "return_code": proc.returncode,
                    "success": proc.returncode == 0,
                    "sandbox": _sandbox_result(sandbox_command),
                }
                retry_command = (
                    None if _wp_root_retry else
                    _wp_cli_root_retry_command(command, result["stderr"] or result["stdout"])
                )
                if retry_command:
                    recovered = await self._execute_command(
                        retry_command, cwd=cwd, timeout=timeout,
                        environment=environment, shell=shell,
                        sandbox_permissions=sandbox_permissions,
                        approval_metadata=approval_metadata,
                        background_signal=background_signal,
                        _wp_root_retry=True,
                    )
                    recovered["recovery"] = {
                        "kind": "wp_cli_root_guard",
                        "original_command": command,
                        "retry_command": retry_command,
                        "message": (
                            "WP-CLI rejected the approved read because the process was "
                            "running as root; the same read was retried with --allow-root."
                        ),
                    }
                    return recovered
                return result
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
                # Never hand a child the real TTY: a git hook or credential
                # helper reading it would eat the user's keystrokes (see
                # _execute_command for the full rationale).
                stdin=asyncio.subprocess.DEVNULL,
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

    async def _knowledge_base_search(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Query TamfisGPT's shared research corpus via its internal Tier IV
        endpoint (see tier_iv_orchestration/tamgpt_api.py's /v1/knowledge/search
        on that project, 127.0.0.1-only, no auth -- same trust boundary as
        this tool's own TamfisGPT-routed model calls). Unlike web_search,
        this is inherently TamfisGPT-dependent -- there is no portable
        fallback, so a connectivity failure is reported clearly rather than
        raised, and the caller should fall back to web_search instead.
        """
        query = (query or "").strip()
        if not query:
            raise ValueError("knowledge_base_search requires a non-empty query")
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 10
        limit = max(1, min(limit, 30))

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{_TAMGPT_TIER_IV_BASE}/v1/knowledge/search",
                    json={"query": query, "limit": limit},
                )
        except httpx.HTTPError as exc:
            return {
                "query": query, "results": [],
                "error": f"TamfisGPT knowledge base unreachable ({exc}) -- fall back to web_search.",
            }
        if response.status_code != 200:
            return {
                "query": query, "results": [],
                "error": f"TamfisGPT knowledge search failed (HTTP {response.status_code}) -- fall back to web_search.",
            }
        payload = response.json() or {}
        return {"query": query, "results": payload.get("results") or []}

    async def _knowledge_base_index(self, title: str, text: str, url: str = "") -> Dict[str, Any]:
        """Write a source into TamfisGPT's shared research corpus -- see
        _knowledge_base_search's docstring for the endpoint/trust model."""
        title = (title or "").strip()
        text = (text or "").strip()
        if not title or not text:
            raise ValueError("knowledge_base_index requires a non-empty title and text")

        source = {
            "title": title,
            "url": (url or "").strip(),
            "evidence_units": [{"text": text}],
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{_TAMGPT_TIER_IV_BASE}/v1/knowledge/index",
                    json={"source": source},
                )
        except httpx.HTTPError as exc:
            return {"indexed": False, "error": f"TamfisGPT knowledge base unreachable ({exc})."}
        if response.status_code != 200:
            return {"indexed": False, "error": f"TamfisGPT knowledge index failed (HTTP {response.status_code})."}
        payload = response.json() or {}
        return {"indexed": True, "chunks_indexed": payload.get("chunks_indexed", 0)}

    async def _memory_search(self, query: str, project: str = "", limit: int = 8) -> Dict[str, Any]:
        """Meaning-matched recall over THIS AGENT's own saved memory notes via
        TamfisGPT's agent-memory collection (Tier IV /v1/memory/search ->
        tier_vi_knowledge/embeddings/agent_memory_index.py, a dedicated
        ChromaDB collection -- nothing here can leak into or out of the
        research corpus that knowledge_base_search reads). Same trust model
        and failure contract as _knowledge_base_search: TamfisGPT-dependent,
        no portable fallback, so a connectivity failure is reported clearly
        rather than raised.
        """
        query = (query or "").strip()
        if not query:
            raise ValueError("memory_search requires a non-empty query")
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 8
        limit = max(1, min(limit, 50))

        body: Dict[str, Any] = {"query": query, "limit": limit}
        if (project or "").strip():
            body["project"] = project.strip()
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{_TAMGPT_TIER_IV_BASE}/v1/memory/search",
                    json=body,
                )
        except httpx.HTTPError as exc:
            return {
                "query": query, "results": [],
                "error": f"TamfisGPT vector memory unreachable ({exc}) -- recall is unavailable this session.",
            }
        if response.status_code != 200:
            return {
                "query": query, "results": [],
                "error": f"TamfisGPT memory search failed (HTTP {response.status_code}) -- recall is unavailable this session.",
            }
        payload = response.json() or {}
        return {"query": query, "results": payload.get("results") or []}

    async def _memory_remember(
        self, text: str, memory_id: str = "", kind: str = "note",
        project: str = "", tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Store one memory note into TamfisGPT's agent-memory collection
        (Tier IV /v1/memory/index -- see _memory_search for the trust model).
        A stable memory_id makes the write idempotent: Tier IV upserts the
        single chunk behind that id, so a re-save REPLACES the old text
        instead of duplicating it. memory_id omitted -> derived from a hash
        of the text, so a verbatim re-save is still idempotent while a
        reworded note starts a fresh id (the model should pass an explicit
        id when it intends to update)."""
        text = (text or "").strip()
        if not text:
            raise ValueError("memory_remember requires non-empty text")
        memory_id = (memory_id or "").strip()
        if not memory_id:
            memory_id = f"auto:{hashlib.sha256(text.encode('utf-8')).hexdigest()[:32]}"
        if len(memory_id) > 200:
            raise ValueError("memory_remember requires a memory_id of at most 200 characters")
        clean_tags = [str(tag).strip() for tag in (tags or []) if str(tag).strip()][:10]

        body: Dict[str, Any] = {
            "text": text,
            "memory_id": memory_id,
            "kind": (kind or "note").strip() or "note",
            "project": (project or "").strip(),
            "tags": clean_tags,
            "source_tool": "tamfis-code",
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{_TAMGPT_TIER_IV_BASE}/v1/memory/index",
                    json=body,
                )
        except httpx.HTTPError as exc:
            return {"indexed": False, "error": f"TamfisGPT vector memory unreachable ({exc})."}
        if response.status_code != 200:
            return {"indexed": False, "error": f"TamfisGPT memory index failed (HTTP {response.status_code})."}
        payload = response.json() or {}
        return {"indexed": True, "memory_id": payload.get("memory_id", memory_id)}

    async def _kill_background_job(self, job_id: str, force: bool = False) -> Dict[str, Any]:
        """Thin async wrapper over the module-level kill (see kill_background_job
        for the signal/group reasoning); module-level like the status reader so
        it can also be reused outside an MCPServer instance."""
        return kill_background_job(str(job_id or ""), force=bool(force))

# Convenience function for CLI use
async def call_tool(name: str, **kwargs):
    """Call a tool with given parameters"""
    server = MCPServer()
    from .capability_gateway import TamfisCodeCapabilityGateway
    return await TamfisCodeCapabilityGateway(server).call_tool(name, kwargs)
