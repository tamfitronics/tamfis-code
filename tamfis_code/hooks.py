"""User-configurable pre/post-tool-use hooks for the standalone agent loop.

Matches Claude Code's PreToolUse/PostToolUse hook model at a small, real
scope: a hook is an arbitrary shell command, configured in a settings file
(not hardcoded), that receives a JSON event on stdin and can observe (or,
for PreToolUse, block) a real local tool call before/after mcp.py executes
it. This was a genuine parity gap -- tamfis-code had no equivalent
mechanism at all before this module.

Config precedence mirrors config.py's own layering: user hooks
(~/.config/tamfis-code/hooks.toml, or the platform-equivalent via
config.resolve_config_dir) load first, project hooks
(<project_root>/.tamfis/hooks.toml) load second and run after them. Both
files use the same shape:

    [[pre_tool_use]]
    matcher = "write_file|edit_file"   # regex against the tool name; empty/absent matches every tool
    command = "python3 my_guard.py"

    [[post_tool_use]]
    matcher = "execute_command"
    command = "notify-send 'tamfis-code ran a command'"

Each hook command is run with the event JSON on stdin:
    {"event": "pre_tool_use"|"post_tool_use", "tool_name": ..., "tool_input": {...},
     "tool_output": {...} (post_tool_use only), "session_id": ..., "workspace_root": ...}

PreToolUse: exit code 2 blocks the tool call -- the tool is never actually
executed, and the hook's stderr (falling back to stdout) becomes the denial
reason fed back to the model as the tool result, the same shape an approval
denial already uses. Any other non-zero exit does not block, but its
stderr/stdout still surfaces as a diagnostic. PostToolUse: the tool has
already run, so no exit code can undo it -- stderr/stdout always just
surfaces as additional context appended for the model to see, matching
Claude Code's PostToolUse contract (observe and inform, not veto).

A hook that fails to start, errors, or times out never crashes the turn --
it degrades to a visible diagnostic, the same "never let an optional
integration point take down a real turn" contract already established by
mcp.py's `_import_monorepo_attr` for browser/the shared MCP bridge.
"""

from __future__ import annotations

import asyncio
import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .config import CONFIG_DIR

HOOKS_PATH = CONFIG_DIR / "hooks.toml"
PROJECT_HOOKS_RELATIVE = Path(".tamfis") / "hooks.toml"
HOOK_TIMEOUT_SECONDS = 30
_HOOK_EVENTS = ("pre_tool_use", "post_tool_use")


@dataclass(frozen=True)
class HookDefinition:
    event: str
    matcher: str
    command: str
    source: str


@dataclass(frozen=True)
class HookResult:
    blocked: bool
    message: str
    hook: HookDefinition


def _load_hooks_file(path: Path, source: str) -> list[HookDefinition]:
    if not path.is_file():
        return []
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return []
    hooks: list[HookDefinition] = []
    for event in _HOOK_EVENTS:
        entries = data.get(event)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            command = str(entry.get("command") or "").strip()
            if not command:
                continue
            hooks.append(HookDefinition(
                event=event, matcher=str(entry.get("matcher") or ""), command=command, source=source,
            ))
    return hooks


def load_hooks(project_root: Optional[str] = None) -> list[HookDefinition]:
    """User hooks first, then project hooks (if `project_root` is given and
    has a .tamfis/hooks.toml) -- same ordering as config.py's own layering,
    read fresh once per turn rather than cached (hook edits should take
    effect on the next turn without restarting the process)."""
    hooks = _load_hooks_file(HOOKS_PATH, "user config")
    if project_root is not None:
        hooks += _load_hooks_file(Path(project_root) / PROJECT_HOOKS_RELATIVE, "project config")
    return hooks


def _matches(hook: HookDefinition, tool_name: str) -> bool:
    if not hook.matcher:
        return True
    try:
        return re.search(hook.matcher, tool_name) is not None
    except re.error:
        return False


async def run_tool_hooks(
    hooks: list[HookDefinition],
    event: str,
    *,
    tool_name: str,
    tool_input: dict[str, Any],
    tool_output: Optional[dict[str, Any]] = None,
    session_id: int,
    workspace_root: str,
) -> list[HookResult]:
    """Run every configured hook for `event` whose matcher matches
    `tool_name`, in configured order, and return what each one reported.
    Callers stop at the first `blocked` result for pre_tool_use (a later
    hook's opinion on a call that was already refused doesn't matter);
    every hook still runs for post_tool_use since none of them can block.
    """
    if not hooks:
        return []
    payload = json.dumps({
        "event": event,
        "tool_name": tool_name,
        "tool_input": tool_input,
        "tool_output": tool_output,
        "session_id": session_id,
        "workspace_root": workspace_root,
    }, default=str).encode("utf-8")

    results: list[HookResult] = []
    for hook in hooks:
        if hook.event != event or not _matches(hook, tool_name):
            continue
        try:
            proc = await asyncio.create_subprocess_shell(
                hook.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workspace_root,
            )
        except OSError as exc:
            results.append(HookResult(blocked=False, message=f"Hook failed to start ({exc}): {hook.command}", hook=hook))
            continue
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(payload), timeout=HOOK_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            results.append(HookResult(
                blocked=False,
                message=f"Hook timed out after {HOOK_TIMEOUT_SECONDS}s and was killed: {hook.command}",
                hook=hook,
            ))
            continue
        text = (stderr or stdout or b"").decode("utf-8", errors="ignore").strip()
        if event == "pre_tool_use" and proc.returncode == 2:
            results.append(HookResult(blocked=True, message=text or f"Blocked by hook: {hook.command}", hook=hook))
            break
        if text:
            results.append(HookResult(blocked=False, message=text, hook=hook))
    return results
