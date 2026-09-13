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

    [[pre_tool_use]]
    matcher = "execute_command"
    if_command = "git commit*"   # fnmatch glob against tool_input["command"]; Claude-Code-parity addition
    command = "python3 my_commit_guard.py"

    [[session_interrupted]]
    command = "notify-send 'tamfis-code task was interrupted'"

    [[user_prompt_submit]]
    command = "python3 my_prompt_guard.py"

    [[session_completed]]
    command = "notify-send 'tamfis-code task finished'"

Each hook command is run with the event JSON on stdin:
    {"event": "pre_tool_use"|"post_tool_use", "tool_name": ..., "tool_input": {...},
     "tool_output": {...} (post_tool_use only), "session_id": ..., "workspace_root": ...}

session_interrupted fires whenever the standalone local runner checkpoints
a turn as interrupted -- a provider/tool failure it can't recover from
mid-turn, a tool-call round budget exhausted, or a rejected/invalid final
answer -- anywhere runner_local.py calls its internal
`_persist_turn_checkpoint(status="interrupted", ...)`. It has no
`tool_name`/`matcher` concept (nothing tool-specific triggered it), so
every configured `session_interrupted` hook runs unconditionally; its
payload is instead {"event": "session_interrupted", "session_id": ...,
"workspace_root": ..., "reason": ...}. Like PostToolUse, this is
observe-only -- the interruption already happened, so no exit code can
undo it -- but see run_session_hooks for details.

user_prompt_submit fires once per turn, before the objective is
classified/sent to a provider. Like pre_tool_use, exit code 2 blocks --
here, the whole turn, not just one tool call. Any other hook output is
folded into the objective as additional context, mirroring Claude Code's
"add context" capability for this event. See run_user_prompt_submit_hooks.

session_completed fires once a turn completes successfully -- an
observe-only notification, deliberately NOT a port of Claude Code's real
Stop hook (which can force the agent to keep working by returning
{"decision": "block"}; that has no analog in tamfis-code's synchronous
per-turn hook firing). See run_session_completed_hooks.

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

Claude-Code-parity addition: every matching hook for one event runs
CONCURRENTLY (asyncio.gather), matching Claude Code's own documented "all
matching hooks run in parallel" contract, rather than one at a time.
Hook order is still preserved in the returned list regardless of
completion order. For a blocking-capable event (pre_tool_use,
user_prompt_submit), once the first hook in *configured* order blocks,
later hooks' results are dropped from the returned list the same way
sequential short-circuiting used to behave -- but their subprocesses
still actually run to completion (or timeout) concurrently, since there
is no way to cancel an already-launched hook without breaking Claude
Code's own "hooks don't see each other's output" independence contract.
"""

from __future__ import annotations

import asyncio
import fnmatch
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
_HOOK_EVENTS = (
    "pre_tool_use", "post_tool_use", "session_interrupted",
    "user_prompt_submit", "session_completed",
    "session_start", "session_end", "subagent_stop", "pre_compact", "notification",
)


@dataclass(frozen=True)
class HookDefinition:
    event: str
    matcher: str
    command: str
    source: str
    if_command: str = ""


@dataclass(frozen=True)
class HookResult:
    blocked: bool
    message: str
    hook: HookDefinition
    updated_input: Optional[dict[str, Any]] = None


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
                if_command=str(entry.get("if_command") or ""),
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


def _matches(hook: HookDefinition, tool_name: str, tool_input: Optional[dict[str, Any]] = None) -> bool:
    """`matcher` (a regex against the tool name) must pass first; if the
    hook also sets `if_command` (Claude-Code-parity addition -- a
    simplified, glob-style equivalent of Claude Code's real
    `"if": "Bash(git commit:*)"` syntax, deliberately not porting that
    exact mini-language verbatim), it is additionally matched with
    fnmatch against tool_input["command"]. Only meaningful for tools that
    actually take a `command` argument (execute_command); for any other
    tool, an `if_command` clause simply never matches rather than
    crashing -- there is nothing sensible to compare it against.
    """
    if hook.matcher:
        try:
            if re.search(hook.matcher, tool_name) is None:
                return False
        except re.error:
            return False
    if hook.if_command:
        command = str((tool_input or {}).get("command") or "")
        if not command:
            return False
        if not fnmatch.fnmatch(command, hook.if_command):
            return False
    return True




async def _execute_one_hook(
    hook: HookDefinition, payload: bytes, *, workspace_root: str,
    blocking_capable: bool = False, want_updated_input: bool = False,
) -> Optional[HookResult]:
    """Run a single hook subprocess and interpret its outcome. Returns
    None when the hook produced nothing worth reporting (no message, not
    blocked, no updated_input) -- callers filter these out. Every public
    run_*_hooks function below launches all of its matching hooks
    concurrently via _run_hooks_concurrently and calls this once per hook.
    """
    try:
        proc = await asyncio.create_subprocess_shell(
            hook.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workspace_root,
        )
    except OSError as exc:
        return HookResult(blocked=False, message=f"Hook failed to start ({exc}): {hook.command}", hook=hook)
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(payload), timeout=HOOK_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return HookResult(
            blocked=False,
            message=f"Hook timed out after {HOOK_TIMEOUT_SECONDS}s and was killed: {hook.command}",
            hook=hook,
        )
    stderr_text = stderr.decode("utf-8", errors="ignore").strip()
    stdout_text = stdout.decode("utf-8", errors="ignore").strip()
    text = stderr_text or stdout_text
    if blocking_capable and proc.returncode == 2:
        return HookResult(blocked=True, message=text or f"Blocked by hook: {hook.command}", hook=hook)
    # Claude-Code-parity addition (updatedInput): a pre_tool_use hook that
    # didn't block can instead rewrite the pending call's arguments by
    # printing {"updated_input": {...}} as its entire stdout. Deliberately
    # a flatter, simpler shape than Claude Code's real nested
    # hookSpecificOutput.updatedInput -- tamfis-code has no separate
    # systemMessage/permissionDecision concept to nest this alongside. If
    # stdout parses as that shape, it is not also surfaced as a raw JSON
    # diagnostic message.
    updated_input: Optional[dict[str, Any]] = None
    if want_updated_input:
        try:
            parsed = json.loads(stdout_text)
        except (ValueError, TypeError):
            parsed = None
        if isinstance(parsed, dict) and isinstance(parsed.get("updated_input"), dict):
            updated_input = parsed["updated_input"]
            text = stderr_text
    if text or updated_input is not None:
        return HookResult(blocked=False, message=text, hook=hook, updated_input=updated_input)
    return None


async def _run_hooks_concurrently(
    matching: list[HookDefinition], payload: bytes, *, workspace_root: str,
    blocking_capable: bool = False, want_updated_input: bool = False,
) -> list[HookResult]:
    """Launch every matching hook concurrently (Claude-Code-parity: "all
    matching hooks run in parallel"), then resolve to a deterministic
    result list in configured order regardless of completion order. For a
    blocking-capable event, once the first hook in configured order
    blocked, later hooks' results are dropped from the returned list --
    their subprocesses still ran to completion (or timeout) concurrently;
    only their reported outcome is discarded, matching how a
    sequential-with-early-break design used to behave for the caller.
    """
    if not matching:
        return []
    raw = await asyncio.gather(*(
        _execute_one_hook(
            hook, payload, workspace_root=workspace_root,
            blocking_capable=blocking_capable, want_updated_input=want_updated_input,
        )
        for hook in matching
    ))
    results = [result for result in raw if result is not None]
    if blocking_capable:
        blocking_index = next((i for i, result in enumerate(results) if result.blocked), None)
        if blocking_index is not None:
            results = results[: blocking_index + 1]
    return results


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
    """Run every configured hook for `event` whose matcher (and, for
    pre_tool_use/post_tool_use, if_command) matches, concurrently. For
    pre_tool_use, exit code 2 blocks the tool call, and a non-blocked
    hook may rewrite tool_input via updated_input (see _execute_one_hook).
    post_tool_use is always observe-only -- the tool already ran, so no
    exit code can undo it.
    """
    matching = [hook for hook in hooks if hook.event == event and _matches(hook, tool_name, tool_input)]
    if not matching:
        return []
    payload = json.dumps({
        "event": event,
        "tool_name": tool_name,
        "tool_input": tool_input,
        "tool_output": tool_output,
        "session_id": session_id,
        "workspace_root": workspace_root,
    }, default=str).encode("utf-8")
    return await _run_hooks_concurrently(
        matching, payload, workspace_root=workspace_root,
        blocking_capable=(event == "pre_tool_use"),
        want_updated_input=(event == "pre_tool_use"),
    )


async def run_session_hooks(
    hooks: list[HookDefinition],
    event: str,
    *,
    session_id: int,
    workspace_root: str,
    reason: str = "",
) -> list[HookResult]:
    """Run every configured hook for a session-level `event` (currently only
    "session_interrupted") concurrently -- unlike run_tool_hooks, there is
    no tool_name to match against, so every hook configured for this event
    runs unconditionally. Always observe-only: the exit code is never
    inspected, since a session-level event (a task already checkpointed as
    interrupted) can't be blocked or undone after the fact.
    """
    matching = [hook for hook in hooks if hook.event == event]
    if not matching:
        return []
    payload = json.dumps({
        "event": event,
        "session_id": session_id,
        "workspace_root": workspace_root,
        "reason": reason,
    }, default=str).encode("utf-8")
    return await _run_hooks_concurrently(matching, payload, workspace_root=workspace_root)


async def run_user_prompt_submit_hooks(
    hooks: list[HookDefinition],
    *,
    session_id: int,
    workspace_root: str,
    objective: str,
) -> list[HookResult]:
    """Claude-Code-parity addition: fires once per turn, before the
    objective is classified/sent to a provider, with
    {"event": "user_prompt_submit", "session_id": ..., "workspace_root": ...,
    "objective": ...} on stdin. Every configured hook for the event runs
    unconditionally and concurrently (there is no tool_name to match
    against). Exit code 2 blocks the turn from proceeding at all -- the
    caller is expected to fail the turn with the hook's stderr/stdout as
    the reason, the same convention pre_tool_use already uses for a
    blocked tool call. Any other hook output is returned as additional
    context for the caller to fold into the objective (Claude Code's "add
    context" capability for this event) rather than discarded.
    """
    matching = [hook for hook in hooks if hook.event == "user_prompt_submit"]
    if not matching:
        return []
    payload = json.dumps({
        "event": "user_prompt_submit",
        "session_id": session_id,
        "workspace_root": workspace_root,
        "objective": objective,
    }, default=str).encode("utf-8")
    return await _run_hooks_concurrently(matching, payload, workspace_root=workspace_root, blocking_capable=True)


async def run_session_completed_hooks(
    hooks: list[HookDefinition],
    *,
    session_id: int,
    workspace_root: str,
    summary: str = "",
) -> list[HookResult]:
    """Claude-Code-parity addition: an observe-only completion notification,
    symmetric to run_session_hooks' "session_interrupted" for the failure
    case -- fires once a turn completes successfully, with
    {"event": "session_completed", "session_id": ..., "workspace_root": ...,
    "summary": ...} on stdin.

    This is deliberately NOT a port of Claude Code's real Stop hook: Stop
    can return {"decision": "block"} to force the agent to keep working
    (re-injecting into the same round loop), which has no analog in
    tamfis-code's synchronous per-turn hook firing and would require
    substantial runner_local.py changes to support safely (resuming a
    turn the caller already believes is finished). This only covers the
    observe-only notification half of Stop's contract.
    """
    matching = [hook for hook in hooks if hook.event == "session_completed"]
    if not matching:
        return []
    payload = json.dumps({
        "event": "session_completed",
        "session_id": session_id,
        "workspace_root": workspace_root,
        "summary": summary,
    }, default=str).encode("utf-8")
    return await _run_hooks_concurrently(matching, payload, workspace_root=workspace_root)


async def run_session_start_hooks(
    hooks: list[HookDefinition],
    *,
    session_id: int,
    workspace_root: str,
) -> list[HookResult]:
    """Claude-Code-parity addition: fires once when an interactive REPL
    session begins (see interactive.py's run_interactive wrapper), with
    {"event": "session_start", "session_id": ..., "workspace_root": ...}
    on stdin. Observe-only -- Claude Code's real SessionStart also lets a
    hook persist environment variables for the rest of the session via
    $CLAUDE_ENV_FILE; tamfis-code has no equivalent env-injection point in
    its process model, so this only covers "load context": a hook's
    output is surfaced to the user as a dim diagnostic line by the
    caller, the same way a SessionStart hook loading project context
    would be shown.
    """
    matching = [hook for hook in hooks if hook.event == "session_start"]
    if not matching:
        return []
    payload = json.dumps({
        "event": "session_start",
        "session_id": session_id,
        "workspace_root": workspace_root,
    }, default=str).encode("utf-8")
    return await _run_hooks_concurrently(matching, payload, workspace_root=workspace_root)


async def run_session_end_hooks(
    hooks: list[HookDefinition],
    *,
    session_id: int,
    workspace_root: str,
) -> list[HookResult]:
    """Claude-Code-parity addition: fires once when an interactive REPL
    session ends -- any exit path (Ctrl+C, Ctrl+D, /exit, an uncaught
    exception) -- via interactive.py's run_interactive wrapping the whole
    REPL loop in try/finally, so this always fires exactly once regardless
    of which exit path was taken. Distinct from session_completed, which
    fires per successful *turn*, not per process lifetime. Observe-only
    (cleanup/logging), matching Claude Code's SessionEnd contract -- there
    is nothing left to block once the session is already ending.
    """
    matching = [hook for hook in hooks if hook.event == "session_end"]
    if not matching:
        return []
    payload = json.dumps({
        "event": "session_end",
        "session_id": session_id,
        "workspace_root": workspace_root,
    }, default=str).encode("utf-8")
    return await _run_hooks_concurrently(matching, payload, workspace_root=workspace_root)


async def run_subagent_stop_hooks(
    hooks: list[HookDefinition],
    *,
    session_id: int,
    workspace_root: str,
    task_id: str,
    description: str,
    status: str,
    error: str = "",
) -> list[HookResult]:
    """Claude-Code-parity addition: fires once a delegated swarm sub-task
    finishes (agents.py's AgentManager.execute_tasks -- the single choke
    point every sub-task's success or failure already converges on before
    returning its result dict), with {"event": "subagent_stop",
    "session_id": ..., "workspace_root": ..., "task_id": ...,
    "description": ..., "status": "completed"|"failed", "error": ...} on
    stdin. session_id is the *parent* session that launched the swarm (0
    when there is none, e.g. a one-shot `agent-cmd delegate` invocation),
    not the sub-task's own isolated child session.

    Like session_completed, this is deliberately NOT a port of Claude
    Code's real SubagentStop, which (like Stop) can return
    {"decision": "block"} to force the subagent to keep working --
    observe-only here, for the same reason session_completed's own
    docstring gives: tamfis-code's synchronous per-turn hook firing has no
    analog for resuming a sub-task the caller already believes is
    finished.
    """
    matching = [hook for hook in hooks if hook.event == "subagent_stop"]
    if not matching:
        return []
    payload = json.dumps({
        "event": "subagent_stop",
        "session_id": session_id,
        "workspace_root": workspace_root,
        "task_id": task_id,
        "description": description,
        "status": status,
        "error": error,
    }, default=str).encode("utf-8")
    return await _run_hooks_concurrently(matching, payload, workspace_root=workspace_root)


async def run_pre_compact_hooks(
    hooks: list[HookDefinition],
    *,
    session_id: int,
    workspace_root: str,
) -> list[HookResult]:
    """Claude-Code-parity addition: fires immediately before
    state.py's compact_session_thread folds older turns into the
    conversation summary (tamfis-code's only compaction trigger --
    unlike Claude Code, there is no size-triggered auto-compact, only the
    explicit `/compact` command), with {"event": "pre_compact",
    "session_id": ..., "workspace_root": ...} on stdin. A hook's output
    is folded into the preserved summary by the caller (see
    compact_session_thread's preserve_note parameter) rather than just
    logged, matching Claude Code's "preserve critical info" intent for
    this event. Observe/contribute-only: there is no way to block a
    compaction the user explicitly requested.
    """
    matching = [hook for hook in hooks if hook.event == "pre_compact"]
    if not matching:
        return []
    payload = json.dumps({
        "event": "pre_compact",
        "session_id": session_id,
        "workspace_root": workspace_root,
    }, default=str).encode("utf-8")
    return await _run_hooks_concurrently(matching, payload, workspace_root=workspace_root)


async def run_notification_hooks(
    hooks: list[HookDefinition],
    *,
    session_id: int,
    workspace_root: str,
    message: str,
) -> list[HookResult]:
    """Claude-Code-parity addition: fires whenever tamfis-code delivers a
    notification-style message into a session, with {"event":
    "notification", "session_id": ..., "workspace_root": ...,
    "message": ...} on stdin. Currently wired at background.py's
    update_job_status -- the single real notification tamfis-code already
    sends today (a background job/goal finishing, enqueued as a follow-up
    instruction for the parent session) -- rather than Claude Code's
    broader notion covering permission-request/idle nudges, which have no
    single equivalent choke point in tamfis-code's architecture yet.
    Observe-only: a background job has already finished by the time this
    fires, so there is nothing left to block.
    """
    matching = [hook for hook in hooks if hook.event == "notification"]
    if not matching:
        return []
    payload = json.dumps({
        "event": "notification",
        "session_id": session_id,
        "workspace_root": workspace_root,
        "message": message,
    }, default=str).encode("utf-8")
    return await _run_hooks_concurrently(matching, payload, workspace_root=workspace_root)
