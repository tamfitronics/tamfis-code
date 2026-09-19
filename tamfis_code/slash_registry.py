"""Slash commands added to close the gap with Claude Code, Codex, Kimi Code and Freebuff.

Owner request 2026-09-19: "check for /*** commands missing when compared to
freebuff, claude code, codex and kimi code; consolidate and add any missing".

The comparison (their official command tables / source registries) found the
concepts below that tamfis-code lacked. Vendor- and platform-specific commands
(account login/logout, billing and subscription, IDE / desktop / mobile / voice /
browser integrations, cosmetic themes, telemetry, ads) are deliberately NOT added:
they have no meaning for a standalone agent that hides its backends.

This module is a registry, so the REPL's long `if` chain does not grow a screen
for every command. `dispatch` returns:
  * None          -- not one of these; the REPL's existing handlers take over;
  * HANDLED       -- done, prompt again;
  * Rewrite(text) -- run `text` as if the user had typed it (an alias, or a command
                     that is just a well-formed prompt for an existing mode such as
                     /audit).
Handlers get a SlashContext whose fields the REPL syncs in and out around the call,
because the REPL keeps its state in closure locals.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, Union

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from . import state as local_state


class _Handled:
    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "HANDLED"


HANDLED = _Handled()


@dataclass(frozen=True)
class Rewrite:
    text: str


Result = Union[_Handled, Rewrite]


@dataclass
class SlashContext:
    console: Console
    config: Any
    workspace: Any  # WorkspaceContext; a handler may rebind it (/new)
    conversation_history: list  # the REPL's own list object -- mutate in place
    last_turn: Any = None
    last_response_text: Optional[str] = None
    session: Any = None  # the REPL's PromptSession (for /vim)
    standalone: bool = True
    custom_commands: dict = field(default_factory=dict)
    reload_custom_commands: Optional[Callable[[], int]] = None
    version: str = ""


Handler = Callable[[SlashContext, str], Awaitable[Result]]


@dataclass(frozen=True)
class SlashCommand:
    name: str
    description: str
    handler: Handler
    aliases: tuple[str, ...] = ()
    category: str = "session"


# ---------------------------------------------------------------------------
# Aliases that simply mean an existing command (consolidation)
# ---------------------------------------------------------------------------

ALIAS_REWRITES: dict[str, tuple[str, str]] = {
    "/history": ("/resume", "browse and resume past sessions"),
    "/sessions": ("/resume", "browse and resume past sessions"),
    "/chats": ("/resume", "browse and resume past sessions"),
    "/branch": ("/fork", "branch this conversation into a new session"),
    "/cost": ("/usage", "show usage and credit balance"),
    "/pwd": ("/cwd", "show the current workspace root"),
    "/q": ("/exit", "quit"),
    "/?": ("/help", "show this help"),
    "/image": ("/paste-image", "attach an image from the clipboard"),
    "/rewind": ("/undo", "undo the last turn"),
    "/diagnostics": ("/debug", "show session diagnostics"),
    "/changelog": ("/version", "show the version"),
    "/release-notes": ("/version", "show the version"),
    "/plugin": ("/plugins", "list plugins"),
    "/queued": ("/queue", "show or append queued instructions"),
}


def _split(text: str) -> tuple[str, str]:
    stripped = (text or "").strip()
    name, _, rest = stripped.partition(" ")
    return name.lower(), rest.strip()


def _root(ctx: SlashContext) -> Path:
    return Path(getattr(ctx.workspace, "workspace_root", "") or ".").expanduser()


def _say(ctx: SlashContext, text: str) -> None:
    ctx.console.print(text)


def _dim(ctx: SlashContext, text: str) -> None:
    ctx.console.print(f"[dim]{escape(text)}[/dim]")


def _error(ctx: SlashContext, text: str) -> None:
    ctx.console.print(f"[red]{escape(text)}[/red]")


# ---------------------------------------------------------------------------
# Session commands
# ---------------------------------------------------------------------------


async def _cmd_new(ctx: SlashContext, arg: str) -> Result:
    """Start a fresh conversation in this workspace (Claude /clear, Codex /new,
    Kimi /new, Freebuff /new). /clear in tamfis-code only clears the screen."""
    if not ctx.standalone:
        _error(ctx, "/new is available for standalone local sessions only.")
        return HANDLED
    root = str(_root(ctx))
    known = local_state.all_known_session_ids()
    new_id = (max(known) + 1) if known else 1
    local_state.save_session_state(new_id, workspace_root=root, primary_workspace=root)
    previous = ctx.workspace.session_id
    ctx.workspace = type(ctx.workspace)(session_id=new_id, workspace_root=root)
    ctx.conversation_history[:] = []
    ctx.last_turn = None
    ctx.last_response_text = None
    _say(ctx, f"[green]New session {new_id}.[/green] [dim]The previous one ({previous}) is kept: /resume {previous}[/dim]")
    return HANDLED


def _history_markdown(ctx: SlashContext) -> str:
    state = local_state.get_session_state(ctx.workspace.session_id)
    messages = list(ctx.conversation_history) or list(state.conversation_history or [])
    title = local_state.session_display_title(ctx.workspace.session_id)
    lines = [
        f"# {title}", "",
        f"- Session: {ctx.workspace.session_id}",
        f"- Workspace: {getattr(ctx.workspace, 'workspace_root', '')}",
        f"- Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}", "",
    ]
    for message in messages:
        role = str(message.get("role") or "")
        if role not in {"user", "assistant"}:
            continue
        lines += [f"## {'You' if role == 'user' else 'Assistant'}", "", str(message.get("content") or "").strip(), ""]
    return "\n".join(lines).rstrip() + "\n"


async def _cmd_export(ctx: SlashContext, arg: str) -> Result:
    """Write the conversation to a Markdown file (Claude/Codex/Kimi/Freebuff /export)."""
    text = _history_markdown(ctx)
    if text.count("## ") == 0:
        _dim(ctx, "Nothing to export yet -- this session has no conversation.")
        return HANDLED
    name = arg or f"tamfis-session-{ctx.workspace.session_id}-{time.strftime('%Y%m%d-%H%M%S')}.md"
    path = Path(name).expanduser()
    if not path.is_absolute():
        path = _root(ctx) / path
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        _error(ctx, f"Could not write {path}: {exc}")
        return HANDLED
    _say(ctx, f"[green]Exported[/green] {escape(str(path))} [dim]({text.count(chr(10))} lines)[/dim]")
    return HANDLED


async def _cmd_rename(ctx: SlashContext, arg: str) -> Result:
    """Name this session yourself (Claude/Codex/Kimi /rename). A name you set is
    never overwritten by the automatic AI title."""
    if not arg:
        current = local_state.session_display_title(ctx.workspace.session_id)
        _dim(ctx, f"Current name: {current}. Usage: /rename <new name>")
        return HANDLED
    if local_state.rename_session_title(ctx.workspace.session_id, arg):
        _say(ctx, f"[green]Session renamed:[/green] {escape(local_state.session_display_title(ctx.workspace.session_id))}")
    else:
        _error(ctx, "That is not a usable session name.")
    return HANDLED


async def _cmd_archive(ctx: SlashContext, arg: str) -> Result:
    """Hide this session from the resume picker (Codex /archive). Reversible."""
    local_state.set_session_archived(ctx.workspace.session_id, True)
    _say(ctx, f"[green]Session {ctx.workspace.session_id} archived.[/green] [dim]It no longer shows in the resume picker's default view.[/dim]")
    return HANDLED


async def _cmd_undo(ctx: SlashContext, arg: str) -> Result:
    """Drop the last turn from the conversation (Claude /rewind, Kimi /undo). File
    edits the turn made are NOT reverted -- /diffs and /revert do that."""
    history = ctx.conversation_history
    cut = None
    for index in range(len(history) - 1, -1, -1):
        if history[index].get("role") == "user":
            cut = index
            break
    if cut is None:
        _dim(ctx, "Nothing to undo -- no turn in this session yet.")
        return HANDLED
    del history[cut:]
    state = local_state.get_session_state(ctx.workspace.session_id)
    stored = list(state.conversation_history or [])
    for index in range(len(stored) - 1, -1, -1):
        if stored[index].get("role") == "user":
            state.conversation_history = stored[:index]
            local_state.put_session_state(state)
            break
    ctx.last_turn = None
    ctx.last_response_text = None
    _say(ctx, "[green]Undid the last turn.[/green] [dim]Files it changed are not reverted -- /diffs lists them, /revert <id> restores one.[/dim]")
    return HANDLED


# ---------------------------------------------------------------------------
# Prompt-shaped commands: well-formed prompts for existing modes
# ---------------------------------------------------------------------------


async def _cmd_init(ctx: SlashContext, arg: str) -> Result:
    """Generate AGENTS.md (Claude /init, Codex /init, Kimi /init, Freebuff /init)."""
    exists = (_root(ctx) / "AGENTS.md").exists()
    task = (
        "Analyze this repository and " + ("update the existing AGENTS.md" if exists else "create an AGENTS.md")
        + " at its root so future agent sessions start with the right context. Read the real files "
        "first (manifests, README, CI config, a sample of the source and tests). Cover: what the "
        "project is, how to build / test / lint it (the exact commands you verified), the directory "
        "layout, the conventions actually in use, and any hazards (generated files, secrets, "
        "destructive scripts). Keep it concise and specific to THIS repository -- no generic advice. "
        + (f"Additional focus: {arg}" if arg else "")
    ).strip()
    return Rewrite(f"/agent {task}")


async def _cmd_review(ctx: SlashContext, arg: str) -> Result:
    """Review the working-tree changes (Claude/Codex/Freebuff /review). Read-only."""
    task = (
        "Review the current uncommitted changes in this repository (run `git status` and "
        "`git diff`, including staged and untracked files). Find real problems: bugs, missed edge "
        "cases, security issues, broken or missing tests, and inconsistencies with the surrounding "
        "code. Report each finding with the file and line, why it matters and a concrete fix, "
        "ordered by severity. Do not modify any files. "
        + (f"Focus especially on: {arg}" if arg else "")
    ).strip()
    return Rewrite(f"/audit {task}")


async def _cmd_security_review(ctx: SlashContext, arg: str) -> Result:
    """Security review of the pending changes (Claude /security-review). Read-only."""
    task = (
        "Do a security review of the current uncommitted changes (`git diff`, staged and "
        "untracked files). Look for injection (SQL, shell, template), unsafe deserialization, path "
        "traversal, authentication/authorization gaps, secrets committed to the repo, weak "
        "cryptography, SSRF, unsafe file handling and dependency risks. Report only findings you "
        "can point to in the code, with file and line, the realistic impact and a fix. Do not "
        "modify any files. " + (f"Scope: {arg}" if arg else "")
    ).strip()
    return Rewrite(f"/audit {task}")


async def _cmd_interview(ctx: SlashContext, arg: str) -> Result:
    """Turn a rough request into a spec by interviewing the user (Freebuff /interview)."""
    if not arg:
        _dim(ctx, "Usage: /interview <what you want to build or change>")
        return HANDLED
    task = (
        f"I want to: {arg}\n\nBefore doing any work, interview me to turn this into a precise spec. "
        "Read the relevant code first so your questions are informed, then use ask_user_question "
        "(batch up to 4 related questions per call, 2-4 options each with a one-line description, your "
        "recommendation first and marked (Recommended)). Ask only what the code cannot tell you. "
        "When the open questions are settled, write the spec as a short numbered list of requirements "
        "and acceptance checks, and ask me whether to proceed."
    )
    return Rewrite(f"/chat {task}")


# ---------------------------------------------------------------------------
# Inspection commands
# ---------------------------------------------------------------------------


def _table(*columns: str) -> Table:
    table = Table(show_header=True, header_style="bold")
    for column in columns:
        table.add_column(column, overflow="fold")
    return table


async def _cmd_mcp(ctx: SlashContext, arg: str) -> Result:
    """List configured MCP servers (Claude/Codex/Kimi /mcp)."""
    from .mcp_client import load_mcp_servers

    try:
        servers = load_mcp_servers(str(_root(ctx)))
    except Exception as exc:
        _error(ctx, f"Could not read the MCP configuration: {exc}")
        return HANDLED
    if not servers:
        _dim(ctx, "No MCP servers configured. Add them to ~/.config/tamfis-code/mcp.json, .mcp.json or .tamfis/mcp.json.")
        return HANDLED
    table = _table("SERVER", "TRANSPORT", "TARGET")
    for name, config in sorted(servers.items()):
        target = getattr(config, "url", "") or " ".join(
            [str(getattr(config, "command", "") or ""), *[str(a) for a in (getattr(config, "args", None) or [])]]
        )
        transport = "http" if getattr(config, "url", "") else "stdio"
        table.add_row(name, transport, str(target)[:90])
    ctx.console.print(table)
    return HANDLED


async def _cmd_hooks(ctx: SlashContext, arg: str) -> Result:
    """List configured lifecycle hooks (Claude/Codex/Kimi /hooks)."""
    from .hooks import load_hooks

    try:
        hooks = load_hooks(str(_root(ctx)))
    except Exception as exc:
        _error(ctx, f"Could not read hooks: {exc}")
        return HANDLED
    if not hooks:
        _dim(ctx, "No hooks configured. Define them in ~/.config/tamfis-code/hooks.toml or .tamfis/hooks.toml.")
        return HANDLED
    table = _table("EVENT", "MATCHER", "TYPE", "ACTION", "SOURCE")
    for hook in hooks:
        action = hook.prompt if hook.hook_type == "prompt" else hook.command
        table.add_row(hook.event, hook.matcher or "*", hook.hook_type, " ".join(str(action).split())[:60], str(hook.source))
    ctx.console.print(table)
    return HANDLED


async def _cmd_skills(ctx: SlashContext, arg: str) -> Result:
    """List available skills (Claude/Codex /skills)."""
    try:
        from .openhands.skills import workspace_skill_registry

        skills = workspace_skill_registry(_root(ctx)).list()
    except Exception as exc:
        _error(ctx, f"Could not load skills: {exc}")
        return HANDLED
    if not skills:
        _dim(ctx, "No skills found. Add Markdown skill files under .tamfis/skills/ or a plugin's skill roots.")
        return HANDLED
    table = _table("SKILL", "DESCRIPTION")
    for skill in skills:
        table.add_row(str(getattr(skill, "name", "?")), " ".join(str(getattr(skill, "description", "") or "").split())[:90])
    ctx.console.print(table)
    return HANDLED


async def _cmd_plugins(ctx: SlashContext, arg: str) -> Result:
    """List installed plugins (Claude /plugin, Codex /plugins)."""
    from .plugins import load_plugins

    try:
        plugins = load_plugins()
    except Exception as exc:
        _error(ctx, f"Could not load plugins: {exc}")
        return HANDLED
    if not plugins:
        _dim(ctx, "No plugins installed.")
        return HANDLED
    table = _table("PLUGIN", "VERSION", "PATH")
    for plugin in plugins:
        table.add_row(str(getattr(plugin, "name", "?")), str(getattr(plugin, "version", "") or "-"), str(getattr(plugin, "path", "") or getattr(plugin, "root", "") or "-"))
    ctx.console.print(table)
    return HANDLED


async def _cmd_memory(ctx: SlashContext, arg: str) -> Result:
    """Show the instruction files and durable memory in play (Claude /memory, Codex /memories)."""
    from .workspace import _instruction_chain

    root = _root(ctx).resolve()
    try:
        chain = list(_instruction_chain(root, root))
    except Exception:
        chain = []
    _say(ctx, "[bold]Instruction files loaded into every prompt[/bold]")
    if chain:
        for path in chain:
            try:
                size = Path(path).stat().st_size
            except OSError:
                size = 0
            _say(ctx, f"  {escape(str(path))} [dim]({size:,} bytes)[/dim]")
    else:
        _dim(ctx, "  none -- /init creates an AGENTS.md")
    try:
        from .memory import get_memory_store  # type: ignore[attr-defined]
    except Exception:
        try:
            from .cli import get_memory_store  # type: ignore[attr-defined]
        except Exception:
            get_memory_store = None  # type: ignore[assignment]
    if get_memory_store is not None:
        try:
            records = get_memory_store().list()
        except Exception:
            records = []
        _say(ctx, f"\n[bold]Durable memory[/bold] [dim]({len(records)} record{'s' if len(records) != 1 else ''})[/dim]")
        for record in records[:10]:
            label = getattr(record, "title", None) or getattr(record, "name", None) or getattr(record, "content", "")
            _say(ctx, f"  · {escape(' '.join(str(label).split())[:100])}")
    return HANDLED


async def _cmd_tasks(ctx: SlashContext, arg: str) -> Result:
    """List background tasks (Claude /tasks, /background; Kimi /task; Codex /ps)."""
    from . import background

    jobs = background.list_jobs()
    if not jobs:
        _dim(ctx, "No background tasks. Start one with /background <objective>.")
        return HANDLED
    table = _table("ID", "STATUS", "STARTED", "TASK")
    for job in jobs[-15:]:
        table.add_row(
            str(job.get("job_id") or job.get("id") or "?"), str(job.get("status") or "?"),
            str(job.get("started_at") or job.get("created_at") or "")[:19],
            " ".join(str(job.get("objective") or job.get("command") or "").split())[:70],
        )
    ctx.console.print(table)
    _dim(ctx, "/stop <id> stops one.")
    return HANDLED


async def _cmd_stop(ctx: SlashContext, arg: str) -> Result:
    """Stop a background task (Claude /stop, Codex /stop)."""
    from . import background

    if not arg:
        _dim(ctx, "Usage: /stop <task id>   (see /tasks)")
        return HANDLED
    if background.stop_job(arg.split()[0]):
        _say(ctx, f"[green]Stopped {escape(arg.split()[0])}.[/green]")
    else:
        _error(ctx, f"No running background task '{arg.split()[0]}'.")
    return HANDLED


async def _cmd_version(ctx: SlashContext, arg: str) -> Result:
    """Version and install location (Kimi /version, Claude /release-notes)."""
    import sys

    import tamfis_code

    _say(ctx, f"[bold]tamfis-code {escape(ctx.version or getattr(tamfis_code, '__version__', '?'))}[/bold]")
    _dim(ctx, f"installed at {Path(tamfis_code.__file__).parent}")
    _dim(ctx, f"python {sys.version.split()[0]} on {sys.platform}")
    return HANDLED


async def _cmd_debug(ctx: SlashContext, arg: str) -> Result:
    """Session diagnostics (Kimi /debug, Codebuff /diagnostics, Codex /debug-config)."""
    state = local_state.get_session_state(ctx.workspace.session_id)
    history = ctx.conversation_history or list(state.conversation_history or [])
    chars = sum(len(str(m.get("content") or "")) for m in history)
    lines = [
        f"session            {ctx.workspace.session_id}  ({local_state.session_display_title(ctx.workspace.session_id)})",
        f"workspace          {getattr(ctx.workspace, 'workspace_root', '')}",
        f"messages           {len(history)}  (~{chars // 4:,} tokens of history)",
        f"phase / status     {state.current_phase} / {state.execution_status}",
        f"saved plans        {len(state.saved_plans or [])}  active: {state.active_plan_id or '-'}",
        f"checkpoint         {'yes' if state.turn_checkpoint else 'none'}",
        f"queued messages    {sum(1 for i in (state.queued_user_instructions or []) if i.get('status') == 'queued')}",
    ]
    try:
        from .runtime.resume import describe_resume_point

        point = describe_resume_point(ctx.workspace.session_id)
        if point:
            lines.append(f"resume at          step {point['step']}/{point['total']}: {point['name']}")
    except Exception:
        pass
    for line in lines:
        _say(ctx, escape(line))
    return HANDLED


# ---------------------------------------------------------------------------
# Settings commands
# ---------------------------------------------------------------------------


async def _cmd_add_dir(ctx: SlashContext, arg: str) -> Result:
    """Grant this session access to another directory (Claude/Kimi /add-dir)."""
    if not arg:
        _dim(ctx, "Usage: /add-dir <path>")
        return HANDLED
    path = Path(arg).expanduser()
    if not path.is_absolute():
        path = _root(ctx) / path
    try:
        approved = path.resolve()
    except OSError as exc:
        _error(ctx, f"Cannot resolve {arg}: {exc}")
        return HANDLED
    if not approved.is_dir():
        _error(ctx, f"{approved} is not a directory.")
        return HANDLED
    state = local_state.get_session_state(ctx.workspace.session_id)
    allowed = list(dict.fromkeys([*(state.allowed_workspaces or [state.workspace_root]), str(approved)]))
    local_state.save_session_state(ctx.workspace.session_id, allowed_workspaces=allowed)
    _say(ctx, f"[green]Directory approved for this session:[/green] {escape(str(approved))}")
    return HANDLED


_EFFORT_LEVELS = ("low", "medium", "high", "auto")


async def _cmd_effort(ctx: SlashContext, arg: str) -> Result:
    """Set how hard the model thinks (Claude /effort, Codebuff /reasoning)."""
    from . import runner_local

    level = arg.strip().lower()
    if not level:
        _say(ctx, f"Reasoning effort: [bold]{runner_local.DEFAULT_REASONING_EFFORT}[/bold] [dim](low | medium | high | auto)[/dim]")
        return HANDLED
    if level not in _EFFORT_LEVELS:
        _error(ctx, f"Unknown effort '{level}'. Choose: {', '.join(_EFFORT_LEVELS)}.")
        return HANDLED
    runner_local.DEFAULT_REASONING_EFFORT = level
    note = "chosen per task from its complexity" if level == "auto" else f"pinned to {level} for every task"
    _say(ctx, f"[green]Reasoning effort set to {level}[/green] [dim]({note}; this process only)[/dim]")
    return HANDLED


async def _cmd_vim(ctx: SlashContext, arg: str) -> Result:
    """Toggle vi keybindings in the prompt (Claude/Codex /vim)."""
    session = ctx.session
    if session is None:
        _dim(ctx, "Vi mode needs the interactive prompt.")
        return HANDLED
    from prompt_toolkit.enums import EditingMode

    turn_on = session.editing_mode != EditingMode.VI
    want = arg.strip().lower()
    if want in {"on", "off"}:
        turn_on = want == "on"
    session.editing_mode = EditingMode.VI if turn_on else EditingMode.EMACS
    _say(ctx, f"[green]Vi mode {'on' if turn_on else 'off'}.[/green]")
    return HANDLED


async def _cmd_reload(ctx: SlashContext, arg: str) -> Result:
    """Re-read custom commands (Kimi /reload, Claude /reload-plugins)."""
    count = ctx.reload_custom_commands() if ctx.reload_custom_commands else 0
    _say(ctx, f"[green]Reloaded.[/green] [dim]{count} custom command{'s' if count != 1 else ''} loaded; hooks, skills and MCP settings are read fresh on every task.[/dim]")
    return HANDLED


async def _cmd_feedback(ctx: SlashContext, arg: str) -> Result:
    """Record feedback (Claude/Codex/Kimi/Freebuff /feedback). Saved locally; share the file."""
    if not arg:
        _dim(ctx, "Usage: /feedback <what worked, what did not>")
        return HANDLED
    path = local_state.CONFIG_DIR / "feedback.jsonl"
    entry = {
        "at": datetime.now().isoformat(timespec="seconds"),
        "session": ctx.workspace.session_id,
        "version": ctx.version,
        "feedback": arg,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        _error(ctx, f"Could not save feedback: {exc}")
        return HANDLED
    _say(ctx, f"[green]Thanks -- saved to[/green] {escape(str(path))} [dim](nothing was sent anywhere)[/dim]")
    return HANDLED


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

COMMANDS: tuple[SlashCommand, ...] = (
    SlashCommand("/new", "start a fresh conversation in this workspace (the old one is kept)", _cmd_new, ("/reset",), "session"),
    SlashCommand("/export", "write this conversation to a Markdown file (/export [file])", _cmd_export, (), "session"),
    SlashCommand("/rename", "name this session yourself (/rename <name>)", _cmd_rename, (), "session"),
    SlashCommand("/archive", "hide this session from the resume picker", _cmd_archive, (), "session"),
    SlashCommand("/undo", "drop the last turn from the conversation (edits are not reverted)", _cmd_undo, (), "session"),
    SlashCommand("/init", "analyze the repository and create or update AGENTS.md", _cmd_init, (), "workflow"),
    SlashCommand("/review", "review the uncommitted changes and report problems (read-only)", _cmd_review, (), "workflow"),
    SlashCommand("/security-review", "security review of the uncommitted changes (read-only)", _cmd_security_review, (), "workflow"),
    SlashCommand("/interview", "turn a rough request into a spec by answering the agent's questions", _cmd_interview, (), "workflow"),
    SlashCommand("/mcp", "list configured MCP servers", _cmd_mcp, (), "inspect"),
    SlashCommand("/hooks", "list configured lifecycle hooks", _cmd_hooks, (), "inspect"),
    SlashCommand("/skills", "list available skills", _cmd_skills, (), "inspect"),
    SlashCommand("/plugins", "list installed plugins", _cmd_plugins, (), "inspect"),
    SlashCommand("/memory", "show the instruction files and durable memory in play", _cmd_memory, (), "inspect"),
    SlashCommand("/tasks", "list background tasks", _cmd_tasks, (), "inspect"),
    SlashCommand("/stop", "stop a background task (/stop <id>)", _cmd_stop, (), "inspect"),
    SlashCommand("/version", "show the tamfis-code version and install location", _cmd_version, (), "inspect"),
    SlashCommand("/debug", "session diagnostics: messages, tokens, plan, checkpoint, resume point", _cmd_debug, (), "inspect"),
    SlashCommand("/add-dir", "grant this session access to another directory (/add-dir <path>)", _cmd_add_dir, (), "settings"),
    SlashCommand("/effort", "set how hard the model thinks: low | medium | high | auto", _cmd_effort, (), "settings"),
    SlashCommand("/vim", "toggle vi keybindings in the prompt", _cmd_vim, (), "settings"),
    SlashCommand("/reload", "re-read custom commands", _cmd_reload, (), "settings"),
    SlashCommand("/feedback", "record feedback locally (/feedback <text>)", _cmd_feedback, (), "settings"),
)

_BY_NAME: dict[str, SlashCommand] = {}
for _command in COMMANDS:
    _BY_NAME[_command.name] = _command
    for _alias in _command.aliases:
        _BY_NAME[_alias] = _command


def registered_names() -> set[str]:
    """Every name this registry answers to (commands, their aliases, rewrites)."""
    return set(_BY_NAME) | set(ALIAS_REWRITES)


def slash_command_entries() -> tuple[tuple[str, str], ...]:
    """(name, description) pairs for the REPL's completion and unknown-command
    detection: each command, then each alias as "alias for /x"."""
    entries: list[tuple[str, str]] = []
    for command in COMMANDS:
        entries.append((command.name, command.description))
        for alias in command.aliases:
            entries.append((alias, f"alias for {command.name}"))
    for alias, (target, _description) in ALIAS_REWRITES.items():
        entries.append((alias, f"alias for {target}"))
    return tuple(entries)


def help_text() -> str:
    """The /help section for these commands, grouped by category."""
    titles = {
        "session": "Sessions", "workflow": "Workflows", "inspect": "Inspect", "settings": "Settings",
    }
    lines = []
    for category, title in titles.items():
        rows = [c for c in COMMANDS if c.category == category]
        if not rows:
            continue
        lines.append(f"\n{title}:")
        for command in rows:
            names = command.name + (f" ({', '.join(command.aliases)})" if command.aliases else "")
            lines.append(f"{names:<24}{command.description}")
    return "\n".join(lines)


async def dispatch(text: str, ctx: SlashContext) -> Optional[Result]:
    """Run `text` if it is one of these commands; None otherwise."""
    if not (text or "").lstrip().startswith("/"):
        return None
    name, arg = _split(text)
    if name[1:] in (ctx.custom_commands or {}):
        return None  # the user's own /name.md command wins over a built-in
    if name in ALIAS_REWRITES:
        target = ALIAS_REWRITES[name][0]
        return Rewrite(f"{target} {arg}".strip())
    command = _BY_NAME.get(name)
    if command is None:
        return None
    try:
        return await command.handler(ctx, arg)
    except Exception as exc:  # a broken command must never take the REPL down
        _error(ctx, f"{name} failed: {type(exc).__name__}: {exc}")
        return HANDLED
