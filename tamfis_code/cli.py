"""tamfis-code command-line entry point.

Command surface includes login, workspace/session management, conversational
chat, read-only audit, durable plan/list/execute-plan workflows, full agent
execution, explicit shell commands, approvals, diffs/revert, retries,
background tasks, attach/logs, and the bare `tamfis-code` interactive mode.
Deliberately deferred: JSON/jsonl/sarif output modes,
shell completion, @file/@stdin references, TAMFIS.md hierarchical
instructions, --server/--session remote-session mode, compact as a distinct
command, the network-outage retry state machine, and a durable multi-
channel notification outbox (separate follow-up -- see project memory).
"""

from __future__ import annotations

import asyncio
import functools
import getpass
import os
import re
import sys
from pathlib import Path
from typing import Optional

import click
import httpx
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from . import __version__, state as local_state
from .api_client import (
    AuthRequiredError, RemoteAPIClient, RemoteAPIError, clear_secure_credentials as clear_credentials,
    credential_storage_backend, load_secure_credentials as load_credentials,
    save_secure_credentials as save_credentials,
)
from .config import APPROVAL_MODES, CONFIG_DIR, Config, Credentials, load_config
from .doctor import run_doctor
from .render import StreamRenderer, print_banner, print_error, print_recent_thread, print_resume_plan_status, print_unified_diff
from .runner import (
    ACTIVE_TASK_STATUSES,
    attach_and_stream, follow_session_logs, retry_task_and_stream,
    run_ai_task_and_stream, run_shell_command, submit_ai_task_background,
)
from .tasks import find_recent_task
from .workspace import WorkspaceContext, blocking_dirty_files, context_from_session, discover_local_repository, find_resumable_session, resolve_workspace
from .local_chat import _PROVIDER_ALIASES

# Derived from the real alias table (local_chat.py) rather than hand-typed,
# so a --provider Choice list can't silently drift out of sync with what
# resolve_provider_type() actually accepts -- it used to include "gemini"
# and "apiframe", neither of which was ever a real alias: both passed Click
# validation and then failed with "Unknown local provider" inside
# resolve_provider_type every time.
_PROVIDER_CHOICES = sorted(_PROVIDER_ALIASES.keys())
_PROVIDER_HELP = "tamfis/tamfisgpt (subscription API), hf, nvidia, openrouter, or auto (default)."

EXIT_OK = 0
EXIT_TASK_FAILED = 1
EXIT_INVALID_ARGS = 2
EXIT_AUTH_FAILED = 3
EXIT_RUNTIME_UNAVAILABLE = 6
EXIT_INTERRUPTED = 7
EXIT_LOCAL_STATE_ERROR = 8


def _print_bg_hint(console: Console, session_id: int, task_id: str) -> None:
    console.print(f"[green]backgrounded[/green] · session {session_id} · task {task_id}")
    console.print()
    console.print("  tamfis-code agents")
    console.print(f"  tamfis-code attach {session_id}")
    console.print(f"  tamfis-code logs {session_id}")
    console.print(f"  tamfis-code stop {session_id}")


def _run_async(coro):
    try:
        return asyncio.run(coro)
    except KeyboardInterrupt:
        raise SystemExit(EXIT_INTERRUPTED)


def _use_remote(config: Config, remote_flag: bool) -> bool:
    """A per-command --remote flag always wins; otherwise fall back to the
    persistent config.toml/env `default_backend` setting (see config.py) --
    this is what lets a paid TamfisGPT tenant set it once instead of typing
    --remote on every command."""
    return remote_flag or config.default_backend == "remote"


def async_command(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return _run_async(fn(*args, **kwargs))
    return wrapper


@click.group(invoke_without_command=True)
@click.option("--debug", is_flag=True, default=False, help="Show structured event and tool diagnostics.")
@click.option("--approval", "approval_policy", type=click.Choice(APPROVAL_MODES), default=None, help="Override the configured approval policy for this invocation. Note: 'never' means deny everything outright -- the opposite of 'auto'/'full-auto' (which mean never PROMPT, i.e. auto-approve). It is not a synonym for auto-approve.")
@click.option("--api-base", "api_base", default=None, help="Override the configured Remote API base URL.")
@click.option("--cwd", "cwd_override", type=click.Path(exists=True, file_okay=False), default=None, help="Treat this directory as the workspace instead of the current directory.")
@click.option("--provider", default="auto", help="hf, nvidia, openrouter, or auto (default) -- which provider the bare (no-subcommand) interactive REPL calls directly.")
@click.option("--model", default=None, help="Provider-specific model id for the bare interactive REPL; defaults to that provider's default model.")
@click.option("--remote", is_flag=True, default=False, help="Use the legacy TamfisGPT Remote Workspace backend for the bare interactive REPL instead of calling a provider directly.")
@click.version_option(__version__, prog_name="tamfis-code")
@click.pass_context
def cli(
    ctx: click.Context, debug: bool, approval_policy: Optional[str], api_base: Optional[str],
    cwd_override: Optional[str], provider: str, model: Optional[str], remote: bool,
):
    """TamfisGPT Code -- a standalone terminal coding agent."""
    workspace_root = Path(cwd_override).resolve() if cwd_override else Path.cwd()
    config = load_config(project_root=workspace_root)
    if os.environ.get("NO_COLOR") is not None:
        config.colour = False
        config.sources["colour"] = "env NO_COLOR"
    elif os.environ.get("TERM", "").lower() == "dumb":
        config.colour = False
        config.sources["colour"] = "env TERM=dumb"
    elif (os.environ.get("CI") or not sys.stdout.isatty()) and os.environ.get("FORCE_COLOR") is None:
        config.colour = False
        config.sources["colour"] = "non-interactive output"
    if approval_policy:
        config.approval_policy = approval_policy
        config.sources["approval_policy"] = "--approval flag"
    if api_base:
        config.api_base = api_base
        config.sources["api_base"] = "--api-base flag"
    if debug:
        config.debug = True
        config.sources["debug"] = "--debug flag"
        os.environ["TAMFIS_CODE_DEBUG"] = "1"

    ctx.ensure_object(dict)
    ctx.obj["config"] = config
    ctx.obj["workspace_root"] = workspace_root

    if ctx.invoked_subcommand is None:
        _run_async(_interactive_entry(config, workspace_root, provider, model, remote))


async def _interactive_entry(
    config: Config, workspace_root: Path, provider: str = "auto",
    model: Optional[str] = None, remote: bool = False,
) -> None:
    from .interactive import run_interactive

    console = Console(no_color=not config.colour)

    if not _use_remote(config, remote):
        from .workspace import resolve_local_workspace

        workspace = resolve_local_workspace(workspace_root)
        await run_interactive(None, config, workspace, provider=provider, model=model)
        return

    creds = load_credentials()
    if creds is None:
        print_error(console, "Not authenticated -- run `tamfis-code login` first, or omit --remote for the standalone REPL.")
        raise SystemExit(EXIT_AUTH_FAILED)

    async with RemoteAPIClient(config, creds) as client:
        try:
            workspace = await resolve_workspace(client, workspace_root)
        except AuthRequiredError:
            print_error(console, "Not authenticated -- run `tamfis-code login` first.")
            raise SystemExit(EXIT_AUTH_FAILED)
        except (RemoteAPIError, httpx.HTTPError) as e:
            print_error(console, f"Could not reach TamfisGPT Remote runtime: {e}")
            raise SystemExit(EXIT_RUNTIME_UNAVAILABLE)

        await run_interactive(client, config, workspace)


# -- login / logout ------------------------------------------------------

@cli.command()
@click.option("--email", default=None)
@click.option("--token", "existing_token", default=None, envvar="TAMFIS_CODE_LOGIN_TOKEN",
              help="Use an existing TamfisGPT access token (prefer the environment variable to shell history).")
@click.pass_context
def login(ctx: click.Context, email: Optional[str], existing_token: Optional[str]):
    """Authenticate against the TamfisGPT account system (only needed for --remote commands -- standalone mode never requires login)."""
    config: Config = ctx.obj["config"]
    console = Console(no_color=not config.colour)

    if not email and not existing_token:
        console.print("1. Sign in with email and password")
        console.print("2. Use an existing access token")
        console.print("3. Exit")
        choice = click.prompt("Select an option", type=click.Choice(["1", "2", "3"]), default="1")
        if choice == "3":
            return
        if choice == "2":
            existing_token = getpass.getpass("Access token (input hidden): ")

    if existing_token:
        async def _verify_token():
            creds = Credentials(access_token=existing_token)
            async with RemoteAPIClient(config, creds) as client:
                data = await client.me()
            user = data.get("user") or {}
            if not data.get("authenticated", True):
                raise RemoteAPIError(401, "Token is not authenticated")
            creds.user_id = user.get("id")
            creds.email = user.get("email")
            backend = save_credentials(creds)
            return user, backend

        try:
            user, backend = _run_async(_verify_token())
        except (RemoteAPIError, httpx.HTTPError) as e:
            print_error(console, f"Token login failed: {e}")
            raise SystemExit(EXIT_AUTH_FAILED)
        console.print(f"[green]Logged in[/green] as {user.get('email', 'TamfisGPT user')} · storage={backend}")
        return

    email = email or click.prompt("Email")
    password = getpass.getpass("Password: ")

    async def _do_login():
        async with RemoteAPIClient(config, credentials=None) as client:
            data = await client.login(email, password)
            user = data.get("user") or {}
            creds = Credentials(
                access_token=data["access_token"],
                refresh_token=data.get("refresh_token"),
                user_id=user.get("id"),
                email=user.get("email", email),
            )
            backend = save_credentials(creds)
            return user, backend

    try:
        user, backend = _run_async(_do_login())
    except RemoteAPIError as e:
        print_error(console, f"Login failed: {e}")
        raise SystemExit(EXIT_AUTH_FAILED)

    console.print(f"[green]Logged in[/green] as {user.get('email', email)} (plan: {user.get('plan', 'unknown')}) · storage={backend}")


@cli.command()
@click.pass_context
def logout(ctx: click.Context):
    """End the backend browser session where applicable and remove local credentials."""
    config: Config = ctx.obj["config"]
    console = Console(no_color=not config.colour)
    creds = load_credentials()
    if creds is not None:
        async def _server_logout():
            async with RemoteAPIClient(config, creds) as client:
                try:
                    await client.logout()
                except (RemoteAPIError, httpx.HTTPError):
                    pass  # local token removal must still succeed offline
        _run_async(_server_logout())
    if clear_credentials():
        console.print("[green]Logged out.[/green]")
    else:
        console.print("[dim]Not logged in.[/dim]")


# -- workspace / diagnostics ---------------------------------------------

def _session_for_primary(root: Path) -> Optional[int]:
    resolved = str(root.resolve())
    matches = [
        sid for sid in local_state.all_known_session_ids()
        if local_state.get_session_state(sid).primary_workspace == resolved
        or local_state.get_session_state(sid).workspace_root == resolved
    ]
    return matches[-1] if matches else None


_ABS_PATH_RE = re.compile(r"(?<![\w.])(/[A-Za-z0-9_./+@%:=-]+)")


def _explicit_absolute_paths(objective: str) -> list[Path]:
    return [Path(raw.rstrip(".,;:)]}")) for raw in _ABS_PATH_RE.findall(objective)]


def _project_root_for_target(target: Path) -> Path:
    start = target if target.is_dir() else target.parent
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate.resolve()
    return start.resolve()


@cli.group(name="workspace")
def workspace_group():
    """Manage filesystem roots approved for the current session."""


@workspace_group.command(name="list")
@click.pass_context
def workspace_list(ctx: click.Context):
    console = Console(no_color=not ctx.obj["config"].colour)
    session_id = _session_for_primary(ctx.obj["workspace_root"])
    if session_id is None:
        print_error(console, "No known session for this workspace; run `tamfis-code init` first.")
        raise SystemExit(EXIT_TASK_FAILED)
    state = local_state.get_session_state(session_id)
    for path in state.allowed_workspaces or [state.workspace_root]:
        marker = " (current)" if path == state.current_working_directory else ""
        console.print(f"{path}{marker}")


@workspace_group.command(name="add")
@click.argument("path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.pass_context
def workspace_add(ctx: click.Context, path: Path):
    console = Console(no_color=not ctx.obj["config"].colour)
    session_id = _session_for_primary(ctx.obj["workspace_root"])
    if session_id is None:
        print_error(console, "No known session for this workspace; run `tamfis-code init` first.")
        raise SystemExit(EXIT_TASK_FAILED)
    approved = str(path.resolve())
    state = local_state.get_session_state(session_id)
    allowed = list(dict.fromkeys([*(state.allowed_workspaces or [state.workspace_root]), approved]))
    local_state.save_session_state(session_id, allowed_workspaces=allowed)
    console.print(f"[green]Workspace approved for this session:[/green] {approved}")


@workspace_group.command(name="remove")
@click.argument("path", type=click.Path(path_type=Path))
@click.pass_context
def workspace_remove(ctx: click.Context, path: Path):
    console = Console(no_color=not ctx.obj["config"].colour)
    session_id = _session_for_primary(ctx.obj["workspace_root"])
    if session_id is None:
        print_error(console, "No known session for this workspace; run `tamfis-code init` first.")
        raise SystemExit(EXIT_TASK_FAILED)
    state = local_state.get_session_state(session_id)
    target = str(path.expanduser().resolve())
    if target == state.primary_workspace:
        raise click.UsageError("The primary workspace cannot be removed.")
    local_state.save_session_state(
        session_id, allowed_workspaces=[item for item in state.allowed_workspaces if item != target],
    )
    console.print(f"[green]Workspace removed:[/green] {target}")


@cli.command(name="cwd")
@click.argument("path", required=False, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--remote", is_flag=True, default=False, help="Update the working directory on the legacy TamfisGPT Remote Workspace backend instead of the local session.")
@click.pass_context
@async_command
async def cwd_command(ctx: click.Context, path: Optional[Path], remote: bool):
    config: Config = ctx.obj["config"]
    console = Console(no_color=not config.colour)
    session_id = _session_for_primary(ctx.obj["workspace_root"])
    if session_id is None:
        print_error(console, "No known session for this workspace; run `tamfis-code init` first.")
        raise SystemExit(EXIT_TASK_FAILED)
    state = local_state.get_session_state(session_id)
    if path is None:
        console.print(state.current_working_directory or state.workspace_root)
        return
    target = str(path.resolve())
    if not any(target == allowed or target.startswith(allowed.rstrip("/") + "/") for allowed in state.allowed_workspaces):
        console.print(
            f"Access to this path requires workspace approval:\n\n{target}\n\n"
            f"Approve adding it with:\n  tamfis-code workspace add {target}"
        )
        raise SystemExit(EXIT_TASK_FAILED)

    if not _use_remote(config, remote):
        local_state.save_session_state(session_id, current_working_directory=target)
        discover_local_repository(session_id, Path(target), force=True)
        console.print(f"[green]Working directory:[/green] {target}")
        return

    creds = load_credentials()
    if creds is None:
        print_error(console, "Not authenticated -- run `tamfis-code login` first, or omit --remote to update the local session.")
        raise SystemExit(EXIT_AUTH_FAILED)
    async with RemoteAPIClient(config, creds) as client:
        updated = await client.set_session_cwd(session_id, target)
    resolved = str(updated.get("working_directory") or target)
    local_state.save_session_state(session_id, current_working_directory=resolved)
    discover_local_repository(session_id, Path(resolved), force=True)
    console.print(f"[green]Working directory:[/green] {resolved}")

@cli.command()
@click.option("--remote", is_flag=True, default=False, help="Register/reuse a session on the legacy TamfisGPT Remote Workspace backend instead of a local one.")
@click.pass_context
@async_command
async def init(ctx: click.Context, remote: bool):
    """Open (or reuse) a session for this directory."""
    config: Config = ctx.obj["config"]
    workspace_root: Path = ctx.obj["workspace_root"]
    console = Console(no_color=not config.colour)

    if not _use_remote(config, remote):
        from .workspace import resolve_local_workspace

        workspace = resolve_local_workspace(workspace_root)
        console.print(f"[green]Ready.[/green] session_id={workspace.session_id}  (standalone, local session)")
        console.print(f"workspace_root={workspace.workspace_root}")
        return

    creds = load_credentials()
    if creds is None:
        print_error(console, "Not authenticated -- run `tamfis-code login` first, or omit --remote for a standalone local session.")
        raise SystemExit(EXIT_AUTH_FAILED)

    async with RemoteAPIClient(config, creds) as client:
        try:
            workspace = await resolve_workspace(client, workspace_root)
        except AuthRequiredError:
            print_error(console, "Not authenticated -- run `tamfis-code login` first.")
            raise SystemExit(EXIT_AUTH_FAILED)
        except RemoteAPIError as e:
            print_error(console, str(e))
            raise SystemExit(EXIT_RUNTIME_UNAVAILABLE)

    console.print(f"[green]Ready.[/green] session_id={workspace.session_id} server_id={workspace.server_id}")
    console.print(f"workspace_root={workspace.workspace_root}")


@cli.command()
@click.option("--provider", default="auto", help="hf, nvidia, openrouter, or auto (default).")
@click.option("--remote", is_flag=True, default=False, help="Check the legacy TamfisGPT Remote Workspace backend instead of local provider connectivity.")
@click.pass_context
@async_command
async def doctor(ctx: click.Context, provider: str, remote: bool):
    """Validate provider connectivity (or, with --remote, the legacy backend)."""
    config: Config = ctx.obj["config"]
    workspace_root: Path = ctx.obj["workspace_root"]
    console = Console(no_color=not config.colour)

    if not _use_remote(config, remote):
        from .doctor import _STATUS_STYLE, _diagnose_local_session
        from .local_chat import resolve_provider_type
        from .providers import get_provider_status
        from .workspace import resolve_local_workspace

        try:
            provider_type = resolve_provider_type(provider)
        except ValueError as exc:
            raise click.UsageError(str(exc))
        status = get_provider_status()
        table = Table(show_header=True, header_style="bold")
        for column in ("PROVIDER", "CONFIGURED", "KEY"):
            table.add_column(column)
        any_configured = False
        for name, info in status["config"].items():
            configured = bool(info["api_key_set"]) or name == "tier_iv"
            any_configured = any_configured or configured
            table.add_row(name, "[green]yes[/green]" if configured else "[dim]no[/dim]", info["key_preview"])
        console.print(table)
        console.print(f"[dim]Currently selected: {provider_type.value}  · auto would pick: {status['default']}[/dim]")
        workspace = resolve_local_workspace(workspace_root, discover=False)
        console.print(f"[green]Local session ready[/green]  session_id={workspace.session_id}  workspace_root={workspace.workspace_root}")
        # Session-local diagnostics from actual recorded local turns
        # (context usage, tool-call success rate, plan progress,
        # unresolved validation issues) -- this default/local branch used
        # to stop at provider connectivity and never report any of this,
        # even though state.py already records it all during real runs.
        for result in _diagnose_local_session(workspace_root):
            style = _STATUS_STYLE[result.status]
            console.print(f"[{style}]{result.status:8}[/{style}] {result.name}  [dim]{result.detail}[/dim]")
        if not any_configured:
            print_error(console, "No provider is configured (set HF_TOKEN / NVIDIA_API_KEY / OPENROUTER_API_KEY).")
            raise SystemExit(EXIT_RUNTIME_UNAVAILABLE)
        return

    # Best-effort: if this directory already has (or can idempotently reuse)
    # a session, run_doctor's session/workspace-snapshot/event-replay checks
    # run too -- same resolve_workspace() every other command already calls,
    # so this isn't a new side effect class, just reusing the existing one.
    # Any failure here (no creds, API down) falls through to run_doctor
    # running its own checks and reporting those failures properly instead.
    session_id = None
    creds = load_credentials()
    if creds is not None:
        async with RemoteAPIClient(config, creds) as client:
            try:
                workspace = await resolve_workspace(client, workspace_root)
                session_id = workspace.session_id
            except (AuthRequiredError, RemoteAPIError):
                pass

    ok = await run_doctor(config, console, workspace_root, session_id=session_id)
    if not ok:
        raise SystemExit(EXIT_RUNTIME_UNAVAILABLE)


@cli.command(name="mcp-server")
@click.pass_context
@async_command
async def mcp_server_command(ctx: click.Context):
    """Serve this workspace's read-only tools (read_file, list_directory, search_code,
    find_references, get_git_info) over stdio as a real MCP server, so another agent or
    IDE with MCP support can drive tamfis-code directly. Runs until stdin closes.
    Point an MCP client at: tamfis-code --cwd <this repo> mcp-server"""
    workspace_root: Path = ctx.obj["workspace_root"]
    from .mcp_stdio_server import run_stdio_server

    await run_stdio_server(str(workspace_root))


@cli.command(name="config")
@click.pass_context
def config_command(ctx: click.Context):
    """Show resolved configuration and where each value came from."""
    config: Config = ctx.obj["config"]
    console = Console(no_color=not config.colour)
    table = Table(show_header=True, header_style="bold")
    table.add_column("Setting")
    table.add_column("Value")
    table.add_column("Source", style="dim")
    for key, value in config.as_dict().items():
        table.add_row(key, str(value), config.sources.get(key, "default"))
    table.add_row("credential_storage", credential_storage_backend(), "platform capability")
    console.print(table)


@cli.command()
@click.option("--remote", is_flag=True, default=False, help="List sessions on the legacy TamfisGPT Remote Workspace backend instead of known local sessions.")
@click.option("--all", "show_all", is_flag=True, default=False, help="Also include swarm sub-task child sessions, hidden by default.")
@click.pass_context
@async_command
async def sessions(ctx: click.Context, remote: bool, show_all: bool):
    """List known sessions (local by default, or --remote)."""
    config: Config = ctx.obj["config"]
    console = Console(no_color=not config.colour)

    if not _use_remote(config, remote):
        table = Table(show_header=True, header_style="bold")
        for col in ("ID", "Workspace Root"):
            table.add_column(col)
        for sid in local_state.all_known_session_ids():
            sess_state = local_state.get_session_state(sid)
            if sess_state.is_swarm_child and not show_all:
                continue
            table.add_row(str(sid), sess_state.workspace_root or sess_state.primary_workspace)
        console.print(table)
        return

    creds = load_credentials()
    if creds is None:
        print_error(console, "Not authenticated -- run `tamfis-code login` first, or omit --remote to list local sessions.")
        raise SystemExit(EXIT_AUTH_FAILED)

    async with RemoteAPIClient(config, creds) as client:
        try:
            rows = await client.list_sessions()
        except AuthRequiredError:
            print_error(console, "Not authenticated -- run `tamfis-code login` first.")
            raise SystemExit(EXIT_AUTH_FAILED)
        except RemoteAPIError as e:
            print_error(console, str(e))
            raise SystemExit(EXIT_RUNTIME_UNAVAILABLE)

    table = Table(show_header=True, header_style="bold")
    for col in ("ID", "Server", "Status", "Working Directory", "Commands"):
        table.add_column(col)
    for row in rows:
        table.add_row(str(row.get("id")), str(row.get("server_name")), str(row.get("status")), str(row.get("working_directory") or ""), str(row.get("command_count")))
    console.print(table)


@cli.command()
@click.option("--remote", is_flag=True, default=False, help="Show status from the legacy TamfisGPT Remote Workspace backend instead of the local session.")
@click.pass_context
@async_command
async def status(ctx: click.Context, remote: bool):
    """Show session, task, CWD, and approval status for this workspace."""
    config: Config = ctx.obj["config"]
    workspace_root: Path = ctx.obj["workspace_root"]
    console = Console(no_color=not config.colour)

    if not _use_remote(config, remote):
        from .workspace import resolve_local_workspace

        workspace = resolve_local_workspace(workspace_root, discover=False)
        state = local_state.get_session_state(workspace.session_id)
        console.print(f"session_id={workspace.session_id}  (standalone, local session)  phase={state.current_phase}")
        console.print(f"workspace_root={workspace.workspace_root}")
        console.print(f"repository_root={state.repository_root or '(not a Git repository)'}  branch={state.active_branch or '-'}")
        console.print(f"approval_policy={config.approval_policy}")
        running = state.running_action or {}
        console.print(f"running_action={running.get('purpose', 'none')}  queued={sum(1 for item in state.queued_user_instructions if item.get('status') == 'queued')}")
        console.print(f"modified_files={len(state.modified_files)}  validations={len(state.validation_results)}  unresolved={len(state.unresolved_issues)}")
        console.print(f"saved_plans={len(state.saved_plans)}  active_plan={state.active_plan_id or '-'}")
        return

    creds = load_credentials()
    if creds is None:
        print_error(console, "Not authenticated -- run `tamfis-code login` first, or omit --remote for local status.")
        raise SystemExit(EXIT_AUTH_FAILED)

    state = None
    task_detail = None
    async with RemoteAPIClient(config, creds) as client:
        try:
            workspace = await resolve_workspace(client, workspace_root)
            session_detail = await client.get_session(workspace.session_id)
            state = local_state.get_session_state(workspace.session_id)
            if state.last_task_id:
                try:
                    task_detail = await client.get_task(state.last_task_id)
                except RemoteAPIError as exc:
                    if exc.status_code != 404:
                        raise
        except AuthRequiredError:
            print_error(console, "Not authenticated -- run `tamfis-code login` first.")
            raise SystemExit(EXIT_AUTH_FAILED)
        except RemoteAPIError as e:
            print_error(console, str(e))
            raise SystemExit(EXIT_RUNTIME_UNAVAILABLE)

    if (
        task_detail
        and str(task_detail.get("status")) in {"completed", "failed", "cancelled", "denied"}
        and state.running_action
    ):
        terminal_status = str(task_detail.get("status"))
        local_state.finish_action(
            workspace.session_id,
            str(state.running_action.get("id")),
            status=terminal_status,
            summary=str(task_detail.get("final_answer") or task_detail.get("error") or ""),
        )
        local_state.save_session_state(
            workspace.session_id,
            active_task=None,
            current_phase="report",
            execution_status="idle",
        )
        state = local_state.get_session_state(workspace.session_id)
    console.print(f"session_id={workspace.session_id}  status={session_detail.get('status')}  phase={state.current_phase}")
    console.print(f"workspace_root={workspace.workspace_root}")
    console.print(f"repository_root={state.repository_root or '(not a Git repository)'}  branch={state.active_branch or '-'}")
    console.print(f"approval_policy={config.approval_policy}  api_base={config.api_base}")
    running = state.running_action or {}
    console.print(f"running_action={running.get('purpose', 'none')}  queued={sum(1 for item in state.queued_user_instructions if item.get('status') == 'queued')}")
    console.print(f"modified_files={len(state.modified_files)}  validations={len(state.validation_results)}  unresolved={len(state.unresolved_issues)}")
    console.print(f"saved_plans={len(state.saved_plans)}  active_plan={state.active_plan_id or '-'}")


@cli.command(name="context")
@click.option("--refresh", is_flag=True, help="Force a fresh bounded repository index.")
@click.pass_context
def context_command(ctx: click.Context, refresh: bool):
    """Show the durable, secret-free repository and task context."""
    config: Config = ctx.obj["config"]
    root: Path = ctx.obj["workspace_root"]
    console = Console(no_color=not config.colour)
    matching = [sid for sid in local_state.all_known_session_ids()
                if local_state.get_session_state(sid).workspace_root == str(root)]
    if not matching:
        console.print("[dim]No local session context yet; run `tamfis-code init` first.[/dim]")
        return
    session_id = matching[-1]
    context = discover_local_repository(session_id, root, force=refresh)
    state = local_state.get_session_state(session_id)
    console.print(f"Repository  {context.get('repository_root')}")
    console.print(f"CWD         {context.get('working_directory')}")
    console.print(f"Branch      {context.get('branch') or '-'}")
    console.print(f"Worktree    {'modified' if context.get('dirty') else 'clean'}")
    console.print(f"Task        {(state.active_task or {}).get('objective') or state.conversation_summary or '-'}")
    console.print(f"Phase       {state.current_phase} ({state.execution_status})")
    console.print(f"Indexed     {context.get('indexed_file_count', 0)} files")
    for path in context.get("instruction_files", []):
        console.print(f"  instruction: {path}")


@cli.command(name="reports")
@click.pass_context
def reports_command(ctx: click.Context):
    """Show reports discovered for this repository and verification status."""
    config: Config = ctx.obj["config"]
    root: Path = ctx.obj["workspace_root"]
    console = Console(no_color=not config.colour)
    matching = [sid for sid in local_state.all_known_session_ids()
                if local_state.get_session_state(sid).workspace_root == str(root)]
    if not matching:
        console.print("[dim]No report index yet; run `tamfis-code init` first.[/dim]")
        return
    state = local_state.get_session_state(matching[-1])
    if not state.discovered_reports:
        console.print("[dim]No matching report files discovered in this repository.[/dim]")
        return
    table = Table(show_header=True, header_style="bold")
    for name in ("Modified", "Status", "Title", "Path"):
        table.add_column(name)
    for report in state.discovered_reports:
        table.add_row(str(report.get("modified_at", ""))[:10], str(report.get("verification", "unverified")),
                      str(report.get("title", "")), str(report.get("path", "")))
    console.print(table)


@cli.command(name="plans")
@click.argument("plan_id", required=False)
@click.pass_context
def plans_command(ctx: click.Context, plan_id: Optional[str]):
    """List saved plans, or show one plan by id/prefix."""
    config: Config = ctx.obj["config"]
    root: Path = ctx.obj["workspace_root"]
    console = Console(no_color=not config.colour)
    matching = [sid for sid in local_state.all_known_session_ids()
                if local_state.get_session_state(sid).workspace_root == str(root)]
    if not matching:
        console.print("[dim]No local session context yet; run `tamfis-code init` first.[/dim]")
        return
    session_id = matching[-1]
    state = local_state.get_session_state(session_id)
    if plan_id:
        plan = local_state.get_plan(session_id, plan_id)
        if plan is None:
            print_error(console, "Plan not found or prefix is ambiguous.")
            raise SystemExit(EXIT_TASK_FAILED)
        console.print(
            f"[bold]{plan.get('id')}[/bold] · {plan.get('status', 'ready')}\n"
            f"[dim]Objective:[/dim] {plan.get('objective', '')}"
        )
        console.print(Markdown(str(plan.get("content") or "")))
        return
    if not state.saved_plans:
        console.print("[dim]No saved plans yet. Run `tamfis-code plan <objective>`.[/dim]")
        return
    table = Table(show_header=True, header_style="bold")
    for column in ("ID", "STATUS", "OBJECTIVE", "CREATED"):
        table.add_column(column)
    for item in reversed(state.saved_plans):
        marker = " *" if item.get("id") == state.active_plan_id else ""
        table.add_row(
            f"{item.get('id')}{marker}", str(item.get("status") or "ready"),
            str(item.get("objective") or "")[:80], str(item.get("created_at") or "")[:19],
        )
    console.print(table)


async def _push_live_instruction(config: Config, task_id: str, text: str, classification: str) -> None:
    creds = load_credentials()
    if creds is None:
        raise AuthRequiredError(401, "Not authenticated -- run `tamfis-code login` first.")
    async with RemoteAPIClient(config, creds) as client:
        await client.add_task_instruction(task_id, text, classification)


@cli.command(name="queue")
@click.argument("instruction", nargs=-1)
@click.option("--classification", type=click.Choice(["append", "reprioritise", "pause", "cancel", "replace", "follow_up", "clarification"]), default="append")
@click.option("--priority", type=int, default=100)
@click.pass_context
def queue_command(ctx: click.Context, instruction: tuple[str, ...], classification: str, priority: int):
    """Show or enqueue an instruction for this workspace's active session."""
    config: Config = ctx.obj["config"]
    root: Path = ctx.obj["workspace_root"]
    console = Console(no_color=not config.colour)
    matching = [sid for sid in local_state.all_known_session_ids()
                if local_state.get_session_state(sid).workspace_root == str(root)]
    if not matching:
        print_error(console, "No known session for this workspace; run `tamfis-code init` first.")
        raise SystemExit(EXIT_TASK_FAILED)
    session_id = matching[-1]
    if instruction[:1] == ("remove",):
        if len(instruction) != 2 or not local_state.update_instruction(session_id, instruction[1], "removed"):
            raise click.UsageError("Use `tamfis-code queue remove <queue-id>` with an existing id.")
        console.print(f"[green]Removed queued request:[/green] {instruction[1]}")
    elif instruction:
        text = " ".join(instruction)
        item = local_state.enqueue_instruction(session_id, text, classification=classification, priority=priority)
        queued_now = [entry for entry in local_state.get_session_state(session_id).queued_user_instructions if entry.get("status") == "queued"]
        position = next((index for index, entry in enumerate(queued_now, 1) if entry.get("id") == item.id), len(queued_now))
        console.print(f"[cyan]Queued request:[/cyan] {item.id} · position {position}")

        # "cancel"/"replace"/"reprioritise" already reach a running task by
        # interrupting it (runner.py's watch_instruction_queue, polling this
        # same on-disk queue from whichever process is streaming the task).
        # These three classifications are the live-branch case: guidance
        # that should reach the SAME running task -- possibly streaming in
        # a different terminal -- without killing it. Best-effort: if
        # nothing is running, or the push fails, the instruction still sits
        # in the local queue above for the next REPL turn.
        if classification in {"append", "follow_up", "clarification"}:
            active_task = local_state.get_session_state(session_id).active_task
            task_id = active_task.get("id") if active_task else None
            # A remote-mode active_task can be a concurrently-streaming task
            # in another terminal, reachable via the Remote backend below --
            # a standalone local turn is always synchronous within one
            # process, so there's nothing "live" to push into; the queue
            # just gets consumed on the next turn, silently and correctly.
            if task_id and load_credentials() is not None:
                try:
                    _run_async(_push_live_instruction(config, task_id, text, classification))
                    local_state.update_instruction(session_id, item.id, "running")
                    console.print(f"[green]Sent live to running task[/green] {task_id}")
                except (AuthRequiredError, RemoteAPIError) as e:
                    print_error(console, f"Could not reach the running task ({e}); it will run on the next turn instead.")
    queued = local_state.get_session_state(session_id).queued_user_instructions
    if not queued:
        console.print("[dim]Queue is empty.[/dim]")
        return
    for item in queued:
        console.print(f"  {item.get('id')}  p={item.get('priority')}  {item.get('status')}  {item.get('classification')}  {item.get('text')}")


# -- AI task commands ------------------------------------------------------

async def _run_ai_command(
    config: Config, workspace_root: Path, objective: str, mode: str,
    background: bool = False, model: str = "auto", provider: Optional[str] = None,
    attachment_paths: tuple[str, ...] = (),
) -> int:
    console = Console(no_color=not config.colour)
    creds = load_credentials()
    if creds is None:
        print_error(console, "Not authenticated -- run `tamfis-code login` first.")
        return EXIT_AUTH_FAILED

    async with RemoteAPIClient(config, creds) as client:
        try:
            workspace = await resolve_workspace(client, workspace_root, discover=mode != "chat")
        except AuthRequiredError:
            print_error(console, "Not authenticated -- run `tamfis-code login` first.")
            return EXIT_AUTH_FAILED
        except (RemoteAPIError, httpx.HTTPError) as e:
            print_error(console, f"Could not reach TamfisGPT Remote runtime: {e}")
            return EXIT_RUNTIME_UNAVAILABLE

        attachments = []
        try:
            for raw_path in attachment_paths:
                attachment_path = Path(raw_path).expanduser().resolve()
                if not attachment_path.is_file():
                    print_error(console, f"Attachment not found: {attachment_path}")
                    return EXIT_INVALID_ARGS
                if attachment_path.stat().st_size > 10 * 1024 * 1024:
                    print_error(console, f"Remote task attachments are limited to 10 MB: {attachment_path}")
                    return EXIT_INVALID_ARGS
                attachments.append(await client.upload_attachment(attachment_path))
        except (RemoteAPIError, httpx.HTTPError, OSError) as e:
            print_error(console, f"Could not upload attachment: {e}")
            return EXIT_RUNTIME_UNAVAILABLE

        state = local_state.get_session_state(workspace.session_id)

        # Single-shot commands (`ask`/`exec`/`agent`/...) can be used to
        # continue a paused/background task from a fresh invocation, so a
        # bare "ok"/"yes"/"1"/"step 2" needs the same contextual expansion
        # the interactive REPL applies -- and it must happen before the
        # objective is used for anything else, never rejected for being short.
        from .interactive import contextualize_short_reply
        objective = contextualize_short_reply(
            objective,
            has_context=bool(
                state.last_task_id or state.conversation_summary or state.active_plan_id
                or state.active_task or state.turn_checkpoint or state.conversation_history
            ),
        )

        for requested_path in _explicit_absolute_paths(objective):
            # Explicit local paths are checked factually before the model is
            # involved. This prevents an out-of-scope shell rejection from
            # being misreported as ENOENT (or vice versa).
            try:
                requested_path.lstat()
            except FileNotFoundError:
                console.print(f"File not found: {requested_path}")
                return EXIT_TASK_FAILED
            except PermissionError:
                console.print(f"Permission denied: {requested_path}")
                return EXIT_TASK_FAILED
            if requested_path.is_dir():
                continue
            approved = any(
                str(requested_path.resolve()) == root_path
                or str(requested_path.resolve()).startswith(root_path.rstrip("/") + "/")
                for root_path in state.allowed_workspaces
            )
            if approved:
                continue
            expansion_root = _project_root_for_target(requested_path)
            console.print(
                f"Access to this path requires workspace approval:\n\n{expansion_root}\n\n"
                "Approve adding it to this session's allowed workspaces?"
            )
            if not sys.stdin.isatty() or not click.confirm("Approve", default=False):
                console.print(f"\n  tamfis-code workspace add {expansion_root}")
                return EXIT_TASK_FAILED
            # Grants an ADDITIONAL allowed root (e.g. /tmp) alongside the
            # existing primary workspace -- previously this called
            # set_session_cwd, which instead REPLACES working_directory
            # session-wide, silently abandoning the original repo root for
            # every later command until manually switched back. The server
            # now has a real, separate concept for "also allow this path"
            # (RemoteSession.allowed_workspace_roots), matching what
            # "approve workspace expansion" actually promises the user.
            try:
                expanded = await client.expand_session_workspace(workspace.session_id, str(expansion_root))
            except (AuthRequiredError, RemoteAPIError) as e:
                print_error(console, f"Could not approve workspace expansion: {e}")
                return EXIT_TASK_FAILED
            allowed = list(dict.fromkeys([
                *state.allowed_workspaces,
                *(expanded.get("allowed_workspace_roots") or [str(expansion_root)]),
            ]))
            local_state.save_session_state(workspace.session_id, allowed_workspaces=allowed)
            state = local_state.get_session_state(workspace.session_id)
        if mode in {"coding", "agent", "execute"}:
            repeated_failure = next(
                (
                    issue for issue in state.unresolved_issues
                    if issue.get("type") == "repeated_action_failure" and issue.get("purpose") == objective
                ),
                None,
            )
            if repeated_failure:
                console.print(
                    f"[yellow]▲ This exact objective has already failed {repeated_failure.get('attempts')} "
                    "times in a row.[/yellow]\n"
                    "[dim]Continuing anyway -- consider `tamfis-code plan` to reconsider the approach "
                    "instead of retrying it unchanged.[/dim]"
                )

        dirty_files = state.repository_context.get("dirty_files") or []
        protected_dirty_files = blocking_dirty_files(dirty_files)
        if mode in {"coding", "agent", "execute"} and protected_dirty_files:
            console.print("[yellow]▲ Existing uncommitted changes detected[/yellow]")
            for path in protected_dirty_files[:20]:
                console.print(f"  {path}")
            console.print("[dim]Execution was not started because action-scoped rollback cannot safely distinguish overlapping user edits. Commit/stash them yourself, or use audit/plan mode.[/dim]")
            state.unresolved_issues.append({
                "type": "pre_existing_changes", "status": "blocked",
                "detail": f"{len(protected_dirty_files)} protected dirty paths detected before execute",
            })
            local_state.save_session_state(workspace.session_id, unresolved_issues=state.unresolved_issues[-100:])
            return EXIT_TASK_FAILED

        if background:
            # The task is durable and server-side the instant this call
            # returns (see submit_ai_task_background's docstring) -- this
            # process's own lifetime from here on is irrelevant to the
            # task's. No streaming, no blocking, no approval-answering: use
            # `attach`/`logs` and `approve`/`reject` to interact with it.
            try:
                task = await submit_ai_task_background(
                    client, session_id=workspace.session_id, objective=objective, mode=mode,
                    model=model, provider=provider,
                    attachments=attachments,
                )
            except AuthRequiredError:
                print_error(console, "Not authenticated -- run `tamfis-code login` first.")
                return EXIT_AUTH_FAILED
            except (RemoteAPIError, httpx.HTTPError) as e:
                print_error(console, str(e))
                return EXIT_RUNTIME_UNAVAILABLE
            _print_bg_hint(console, workspace.session_id, str(task["task_id"]))
            return EXIT_OK

        print_banner(console, host=config.api_base, workspace_root=workspace.workspace_root, mode=mode, approval_policy=config.approval_policy)
        renderer = StreamRenderer(console)
        try:
            outcome = await run_ai_task_and_stream(
                client, renderer, console,
                session_id=workspace.session_id, objective=objective, mode=mode,
                # A one-shot command is still a live human sitting at a real
                # terminal watching it stream, unless stdin has been
                # redirected/piped (a script, CI, etc.) -- confirmed live:
                # hardcoding this to False meant the default "ask" approval
                # policy's (y)es/(n)o prompt could never actually appear for
                # `agent`/`ask`/`chat`/`audit`/`plan`, no matter what;
                # every risky action was silently denied instead.
                approval_policy=config.approval_policy, interactive=sys.stdin.isatty(),
                model=model, provider=provider,
                attachments=attachments,
            )
        except AuthRequiredError:
            print_error(console, "Not authenticated -- run `tamfis-code login` first.")
            return EXIT_AUTH_FAILED
        except (RemoteAPIError, httpx.HTTPError) as e:
            print_error(console, str(e))
            return EXIT_RUNTIME_UNAVAILABLE

    if outcome.status == "completed":
        if outcome.summary and not renderer.streamed_final_text:
            console.print(outcome.summary)
        return EXIT_OK
    if outcome.status == "cancelled":
        print_error(console, "Interrupted.")
        return EXIT_INTERRUPTED
    if outcome.status == "detached":
        _print_bg_hint(console, workspace.session_id, outcome.summary or "")
        return EXIT_OK
    print_error(console, outcome.error or f"task {outcome.status}")
    return EXIT_TASK_FAILED


async def _run_local_ai_command(
    config: Config, workspace_root: Path, objective: str, mode: str,
    model: str, provider: Optional[str], attachment_paths: tuple[str, ...],
) -> int:
    """Standalone equivalent of _run_ai_command -- calls a provider directly
    and runs the tool-calling loop locally (runner_local.py), no
    RemoteAPIClient/tamgpt6 backend involved. Reuses every piece of
    _run_ai_command's logic that was already purely local (contextualize_short_reply,
    explicit-path workspace approval, repeated-failure warning, dirty-tree
    guard) -- only the remote-specific pieces (auth, attachment upload,
    remote workspace-expansion RPC, background execution) are replaced or
    dropped, since none of them have a meaningful standalone equivalent yet.
    """
    from .local_chat import resolve_provider_type
    from .providers import ProviderManager
    from .runner_local import (
        _is_resume_request,
        build_vision_content_blocks,
        is_vision_image_path,
        run_local_agent_turn,
    )
    from .safety import READ_ONLY_TOOLS
    from .workspace import resolve_local_workspace

    console = Console(no_color=not config.colour)

    resolved_attachments: list[str] = []
    for raw_path in attachment_paths:
        attachment_path = Path(raw_path).expanduser().resolve()
        if not attachment_path.is_file():
            print_error(console, f"Attachment not found: {attachment_path}")
            return EXIT_INVALID_ARGS
        if attachment_path.stat().st_size > 10 * 1024 * 1024:
            print_error(console, f"Standalone task attachments are limited to 10 MB: {attachment_path}")
            return EXIT_INVALID_ARGS
        resolved_attachments.append(str(attachment_path))

    try:
        provider_type = resolve_provider_type(provider)
    except ValueError as exc:
        raise click.UsageError(str(exc))

    workspace = resolve_local_workspace(workspace_root, discover=mode != "chat")
    state = local_state.get_session_state(workspace.session_id)

    from .interactive import contextualize_short_reply
    objective = contextualize_short_reply(
        objective,
        has_context=bool(
            state.last_task_id or state.conversation_summary or state.active_plan_id
            or state.active_task or state.turn_checkpoint or state.conversation_history
        ),
    )

    for requested_path in _explicit_absolute_paths(objective):
        try:
            requested_path.lstat()
        except FileNotFoundError:
            console.print(f"File not found: {requested_path}")
            return EXIT_TASK_FAILED
        except PermissionError:
            console.print(f"Permission denied: {requested_path}")
            return EXIT_TASK_FAILED
        if requested_path.is_dir():
            continue
        approved = any(
            str(requested_path.resolve()) == root_path
            or str(requested_path.resolve()).startswith(root_path.rstrip("/") + "/")
            for root_path in state.allowed_workspaces
        )
        if approved:
            continue
        expansion_root = _project_root_for_target(requested_path)
        console.print(
            f"Access to this path requires workspace approval:\n\n{expansion_root}\n\n"
            "Approve adding it to this session's allowed workspaces?"
        )
        if not sys.stdin.isatty() or not click.confirm("Approve", default=False):
            console.print(f"\n  tamfis-code workspace add {expansion_root}")
            return EXIT_TASK_FAILED
        # No remote session to notify -- allowed_workspaces is (and always
        # was) purely a local state.py concept for this client.
        allowed = list(dict.fromkeys([*state.allowed_workspaces, str(expansion_root)]))
        local_state.save_session_state(workspace.session_id, allowed_workspaces=allowed)
        state = local_state.get_session_state(workspace.session_id)

    if mode in {"coding", "agent", "execute"}:
        repeated_failure = next(
            (
                issue for issue in state.unresolved_issues
                if issue.get("type") == "repeated_action_failure" and issue.get("purpose") == objective
            ),
            None,
        )
        if repeated_failure:
            console.print(
                f"[yellow]▲ This exact objective has already failed {repeated_failure.get('attempts')} "
                "times in a row.[/yellow]\n"
                "[dim]Continuing anyway -- consider `tamfis-code plan` to reconsider the approach "
                "instead of retrying it unchanged.[/dim]"
            )

    dirty_files = state.repository_context.get("dirty_files") or []
    protected_dirty_files = blocking_dirty_files(dirty_files)
    if mode in {"coding", "agent", "execute"} and protected_dirty_files:
        console.print("[yellow]▲ Existing uncommitted changes detected[/yellow]")
        for path in protected_dirty_files[:20]:
            console.print(f"  {path}")
        console.print("[dim]Execution was not started because action-scoped rollback cannot safely distinguish overlapping user edits. Commit/stash them yourself, or use audit/plan mode.[/dim]")
        state.unresolved_issues.append({
            "type": "pre_existing_changes", "status": "blocked",
            "detail": f"{len(protected_dirty_files)} protected dirty paths detected before execute",
        })
        local_state.save_session_state(workspace.session_id, unresolved_issues=state.unresolved_issues[-100:])
        return EXIT_TASK_FAILED

    read_only_mode = mode in {"chat", "audit", "plan"}
    # Persisted (not cleared on completion, unlike the remote flow's
    # active_task) so a later, separate `tamfis-code retry` invocation can
    # recover both the objective AND the mode that was actually used --
    # without this, retry had no way to know whether the last turn was
    # e.g. "agent" vs "chat" and always guessed "coding".
    # Preserve the prior objective until run_local_agent_turn has had a
    # chance to recover it. Overwriting active_task with the literal word
    # "continue" destroys the last useful legacy resume pointer.
    if not _is_resume_request(objective):
        local_state.save_session_state(workspace.session_id, active_task={"objective": objective, "mode": mode})
    print_banner(console, host=f"local:{provider_type.value}", workspace_root=workspace.workspace_root, mode=mode, approval_policy=config.approval_policy)
    renderer = StreamRenderer(console)
    manager = ProviderManager()
    turn_messages: list[dict[str, str]] = []
    image_attachments = [p for p in resolved_attachments if is_vision_image_path(p)]
    other_attachments = [p for p in resolved_attachments if not is_vision_image_path(p)]
    if other_attachments:
        attachment_lines = "\n".join(f"- {path}" for path in other_attachments)
        turn_messages.append({
            "role": "system",
            "content": (
                "The user explicitly attached the following exact local files for this turn. "
                "They are read-only inputs outside the workspace boundary; use read_file for text "
                "or extract_archive for ZIP/TAR variants. Extract, edit, and repackage outputs must "
                f"stay inside the approved workspace.\n{attachment_lines}"
            ),
        })
    image_content_blocks = build_vision_content_blocks(image_attachments) if image_attachments else None
    if image_attachments:
        image_lines = "\n".join(f"- {path}" for path in image_attachments)
        turn_messages.append({
            "role": "system",
            "content": (
                "The user explicitly attached the following image(s) for this turn. If your model "
                "supports vision, they are included directly in your next user message as real image "
                "content -- look at them, don't call read_file on them (it can't decode binary image "
                "bytes into anything meaningful). If your model does not support vision, you were not "
                "shown the pixels; say so plainly instead of guessing what the image contains.\n"
                f"{image_lines}"
            ),
        })
        if not any(config.vision_supported for config in ProviderManager.PROVIDERS.values()):
            console.print(
                "[yellow]No configured provider supports vision -- the attached image(s) "
                "will be described by path only, not actually seen.[/yellow]"
            )
    turn_messages.append({"role": "user", "content": objective})
    outcome = await run_local_agent_turn(
        manager, provider_type, model if model != "auto" else None, turn_messages,
        console, renderer,
        workspace_root=workspace.workspace_root, session_id=workspace.session_id,
        # See the matching comment in _run_ai_command -- a one-shot command
        # is still an attended terminal unless stdin is redirected/piped.
        approval_policy=config.approval_policy, interactive=sys.stdin.isatty(), read_only=read_only_mode,
        cli_config=config, allow_swarm_tool=True,
        attachment_paths=tuple(resolved_attachments),
        image_content_blocks=image_content_blocks,
    )
    renderer.finish()

    if outcome.status == "completed":
        if outcome.summary and not renderer.streamed_final_text:
            console.print(outcome.summary)
        if outcome.summary:
            local_state.save_session_state(workspace.session_id, conversation_summary=outcome.summary[-4000:])
        if mode == "plan" and outcome.summary:
            saved = local_state.save_plan(workspace.session_id, objective=objective, content=outcome.summary)
            console.print(
                f"[green]Plan saved[/green] · {saved.id} · run `/execute-plan {saved.id}` "
                f"in the REPL or `tamfis-code execute-plan {saved.id}` from the shell"
            )
        return EXIT_OK
    print_error(console, outcome.error or f"task {outcome.status}")
    return EXIT_TASK_FAILED


def _ai_command(mode: str, help_text: str):
    @click.argument("objective", required=False)
    @click.option("--stdin", "read_stdin", is_flag=True, default=False, help="Read the objective from standard input (recommended for very large pasted text).")
    @click.option("--prompt-file", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None, help="Read the objective from a UTF-8 text file.")
    @click.option("--attach", "attachment_paths", multiple=True, type=click.Path(exists=True, dir_okay=False), help="Attach an image or document (repeatable; up to 10 files, 10 MB each).")
    @click.option("--bg", "background", is_flag=True, default=False, help="Submit and return immediately; the task keeps running server-side. Use `tamfis-code agents`/`attach`/`logs` to check on it.")
    @click.option("--model", default="auto", show_default=True, help="Catalog model id, or auto.")
    @click.option("--mode", "mode_override", type=click.Choice(["auto", "coding", "chat", "audit", "plan", "agent", "execute"]), default=None, help="Override this command's task mode.")
    @click.option("--provider", type=click.Choice(_PROVIDER_CHOICES), default=None, help=_PROVIDER_HELP + " Pin one for this task.")
    @click.option("--remote", is_flag=True, default=False, help="Use the legacy TamfisGPT Remote Workspace backend (tamgpt6) instead of calling a provider directly. Deprecated -- standalone (the default) is the supported path going forward.")
    @click.pass_context
    def command(ctx: click.Context, objective: Optional[str], read_stdin: bool, prompt_file: Optional[Path], attachment_paths: tuple[str, ...], background: bool, model: str, mode_override: Optional[str], provider: Optional[str], remote: bool):
        config: Config = ctx.obj["config"]
        workspace_root: Path = ctx.obj["workspace_root"]
        sources = int(bool(objective and objective != "-")) + int(read_stdin or objective == "-") + int(prompt_file is not None)
        if sources != 1:
            raise click.UsageError("Provide exactly one objective, --stdin (or '-'), or --prompt-file.")
        if len(attachment_paths) > 10:
            raise click.UsageError("At most 10 --attach files are allowed per task.")
        effective_mode = mode_override or mode
        if effective_mode == "plan" and background:
            raise click.UsageError(
                "Plan creation must stay attached so the completed plan can be saved locally; omit --bg."
            )
        if background and not _use_remote(config, remote):
            raise click.UsageError(
                "--bg requires --remote (or default_backend = \"remote\" in config.toml): a standalone "
                "local run has no server to keep the task alive once this process exits."
            )
        if prompt_file is not None:
            objective_text = prompt_file.read_text(encoding="utf-8")
        elif read_stdin or objective == "-":
            objective_text = sys.stdin.read()
        else:
            objective_text = objective or ""
        if len(objective_text) > 1_000_000:
            raise click.UsageError("Objective exceeds the 1,000,000 character safety limit.")
        if _use_remote(config, remote):
            exit_code = _run_async(_run_ai_command(
                config, workspace_root, objective_text, effective_mode, background, model, provider, attachment_paths,
            ))
        else:
            exit_code = _run_async(_run_local_ai_command(
                config, workspace_root, objective_text, effective_mode, model, provider, attachment_paths,
            ))
        if exit_code != EXIT_OK:
            raise SystemExit(exit_code)

    command.__doc__ = help_text
    return command


cli.command(name="ask")(_ai_command("coding", "Run a CWD-scoped coding-agent task."))
cli.command(name="chat")(_ai_command("chat", "Use conversational, read-only coding assistance."))
cli.command(name="audit")(_ai_command("audit", "Run a read-only repository audit."))
cli.command(name="plan")(_ai_command("plan", "Produce and save an executable plan without modifying files."))
cli.command(name="agent")(_ai_command("agent", "Run the full coding-agent loop: inspect, edit, and verify."))
cli.command(name="exec")(_ai_command("execute", "Run a tool-using engineering task subject to approval policy."))


@cli.command(name="execute-plan")
@click.argument("plan_id", required=False)
@click.option("--bg", "background", is_flag=True, default=False, help="Execute the plan server-side and return immediately (requires --remote).")
@click.option("--model", default="auto", show_default=True)
@click.option("--provider", type=click.Choice(_PROVIDER_CHOICES), default=None)
@click.option("--remote", is_flag=True, default=False, help="Use the legacy TamfisGPT Remote Workspace backend instead of calling a provider directly.")
@click.pass_context
def execute_plan_command(
    ctx: click.Context, plan_id: Optional[str], background: bool,
    model: str, provider: Optional[str], remote: bool,
):
    """Execute a saved plan (latest/active plan when no id is supplied)."""
    config: Config = ctx.obj["config"]
    root: Path = ctx.obj["workspace_root"]
    console = Console(no_color=not config.colour)
    if background and not _use_remote(config, remote):
        print_error(console, "--bg requires --remote (or default_backend = \"remote\" in config.toml): a standalone local run has no server to keep the task alive once this process exits.")
        raise SystemExit(EXIT_INVALID_ARGS)
    matching = [sid for sid in local_state.all_known_session_ids()
                if local_state.get_session_state(sid).workspace_root == str(root)]
    if not matching:
        print_error(console, "No known session for this workspace; run `tamfis-code init` first.")
        raise SystemExit(EXIT_TASK_FAILED)
    session_id = matching[-1]
    plan = local_state.get_plan(session_id, plan_id)
    if plan is None:
        print_error(console, "Plan not found or prefix is ambiguous. Use `tamfis-code plans`.")
        raise SystemExit(EXIT_TASK_FAILED)
    selected_id = str(plan["id"])
    local_state.update_plan(session_id, selected_id, status="executing")
    objective = local_state.plan_execution_objective(plan)
    if _use_remote(config, remote):
        exit_code = _run_async(_run_ai_command(config, root, objective, "execute", background, model, provider))
    else:
        exit_code = _run_async(_run_local_ai_command(config, root, objective, "execute", model, provider, ()))
    refreshed = local_state.get_session_state(session_id)
    local_state.update_plan(
        session_id, selected_id,
        status="executing" if background and exit_code == EXIT_OK else (
            "completed" if exit_code == EXIT_OK else "failed"
        ),
        execution_task_id=refreshed.last_task_id,
    )
    if exit_code != EXIT_OK:
        raise SystemExit(exit_code)


@cli.command()
@click.argument("command")
@click.option("--bg", "background", is_flag=True, default=False, help="Submit and return immediately; the command keeps running server-side.")
@click.option("--remote", is_flag=True, default=False, help="Use the legacy TamfisGPT Remote Workspace backend instead of running the command directly.")
@click.pass_context
@async_command
async def run(ctx: click.Context, command: str, background: bool, remote: bool):
    """Run an explicit shell command (locally by default, or --remote)."""
    config: Config = ctx.obj["config"]
    workspace_root: Path = ctx.obj["workspace_root"]
    console = Console(no_color=not config.colour)

    if not _use_remote(config, remote):
        if background:
            print_error(console, "--bg requires --remote: a standalone local run has no server to keep the command alive once this process exits.")
            raise SystemExit(EXIT_INVALID_ARGS)
        from .runner_local import run_local_shell_command
        from .workspace import resolve_local_workspace

        workspace = resolve_local_workspace(workspace_root, discover=False)
        outcome = await run_local_shell_command(
            console, workspace_root=workspace.workspace_root, session_id=workspace.session_id,
            # See the matching comment in _run_ai_command -- a one-shot
            # command is still an attended terminal unless stdin is
            # redirected/piped.
            command=command, approval_policy=config.approval_policy, interactive=sys.stdin.isatty(),
        )
        if outcome.status != "completed":
            raise SystemExit(EXIT_TASK_FAILED)
        return

    creds = load_credentials()
    if creds is None:
        print_error(console, "Not authenticated -- run `tamfis-code login` first, or omit --remote to run locally.")
        raise SystemExit(EXIT_AUTH_FAILED)

    async with RemoteAPIClient(config, creds) as client:
        try:
            workspace = await resolve_workspace(client, workspace_root)
            if background:
                cmd = await client.submit_command(workspace.session_id, command)
                console.print(f"[green]backgrounded[/green] · session {workspace.session_id} · command {cmd['id']}")
                console.print(f"  tamfis-code logs {workspace.session_id}")
                console.print(f"  tamfis-code stop {workspace.session_id}")
                return
            outcome = await run_shell_command(
                client, console,
                session_id=workspace.session_id, command=command,
                # See the matching comment in _run_ai_command -- a one-shot
                # command is still an attended terminal unless stdin is
                # redirected/piped.
                approval_policy=config.approval_policy, interactive=sys.stdin.isatty(),
            )
        except AuthRequiredError:
            print_error(console, "Not authenticated -- run `tamfis-code login` first.")
            raise SystemExit(EXIT_AUTH_FAILED)
        except (RemoteAPIError, httpx.HTTPError) as e:
            print_error(console, str(e))
            raise SystemExit(EXIT_RUNTIME_UNAVAILABLE)

    if outcome.status != "completed":
        raise SystemExit(EXIT_TASK_FAILED)


@cli.command()
@click.argument("session_id", type=int, required=False, default=None)
@click.option("--provider", default="auto", help="hf, nvidia, openrouter, or auto (default).")
@click.option("--model", default=None, help="Provider-specific model id; defaults to that provider's default model.")
@click.option("--remote", is_flag=True, default=False, help="Resume a session on the legacy TamfisGPT Remote Workspace backend instead of a local one.")
@click.pass_context
@async_command
async def resume(ctx: click.Context, session_id: Optional[int], provider: str, model: Optional[str], remote: bool):
    """Resume an interrupted or previous session (most recent if no id given), then continue interactively."""
    from .interactive import run_interactive

    config: Config = ctx.obj["config"]
    console = Console(no_color=not config.colour)

    if not _use_remote(config, remote):
        known = local_state.all_known_session_ids()
        if session_id is not None:
            if session_id not in known:
                print_error(console, f"No known local session {session_id}. Use `tamfis-code sessions` to list known sessions.")
                raise SystemExit(EXIT_TASK_FAILED)
            target_id = session_id
        else:
            if not known:
                print_error(console, "No sessions to resume.")
                raise SystemExit(EXIT_TASK_FAILED)
            target_id = known[-1]
        target_state = local_state.get_session_state(target_id)
        workspace = WorkspaceContext(session_id=target_id, workspace_root=target_state.workspace_root or target_state.primary_workspace)
        console.print(f"[green]Resumed session {workspace.session_id}[/green]  workspace_root={workspace.workspace_root}")
        if target_state.conversation_summary:
            console.print(f"[dim]{target_state.conversation_summary[-1000:]}[/dim]")
        print_resume_plan_status(console, target_state)
        await run_interactive(None, config, workspace, provider=provider, model=model)
        return

    creds = load_credentials()
    if creds is None:
        print_error(console, "Not authenticated -- run `tamfis-code login` first, or omit --remote to resume a local session.")
        raise SystemExit(EXIT_AUTH_FAILED)

    async with RemoteAPIClient(config, creds) as client:
        try:
            if session_id is not None:
                workspace = await context_from_session(client, session_id)
            else:
                target = await find_resumable_session(client)
                if target is None:
                    print_error(console, "No sessions to resume.")
                    raise SystemExit(EXIT_TASK_FAILED)
                workspace = await context_from_session(client, target["id"])
        except AuthRequiredError:
            print_error(console, "Not authenticated -- run `tamfis-code login` first.")
            raise SystemExit(EXIT_AUTH_FAILED)
        except (RemoteAPIError, httpx.HTTPError) as e:
            print_error(console, str(e))
            raise SystemExit(EXIT_RUNTIME_UNAVAILABLE)

        console.print(f"[green]Resumed session {workspace.session_id}[/green]  workspace_root={workspace.workspace_root}")
        try:
            thread = await client.get_thread(workspace.session_id)
            print_recent_thread(console, thread.get("messages") or [])
        except (AuthRequiredError, RemoteAPIError):
            pass

        await run_interactive(client, config, workspace)


@cli.command()
@click.argument("task_id", required=False, default=None)
@click.option("--provider", default="auto", help="hf, nvidia, openrouter, or auto (default).")
@click.option("--model", default=None, help="Provider-specific model id; defaults to that provider's default model.")
@click.option("--remote", is_flag=True, default=False, help="Retry a task on the legacy TamfisGPT Remote Workspace backend instead of resending locally.")
@click.pass_context
@async_command
async def retry(ctx: click.Context, task_id: Optional[str], provider: str, model: Optional[str], remote: bool):
    """Retry the last turn in this workspace (standalone), or a specific --remote task."""
    config: Config = ctx.obj["config"]
    workspace_root: Path = ctx.obj["workspace_root"]
    console = Console(no_color=not config.colour)

    if not _use_remote(config, remote):
        if task_id is not None:
            raise click.UsageError("A specific task_id only applies with --remote; standalone retry resends the session's last turn.")
        from .local_chat import resolve_provider_type
        from .providers import ProviderManager
        from .workspace import resolve_local_workspace

        try:
            provider_type = resolve_provider_type(provider)
        except ValueError as exc:
            raise click.UsageError(str(exc))
        workspace = resolve_local_workspace(workspace_root, discover=False)
        state = local_state.get_session_state(workspace.session_id)
        if not state.conversation_summary and not state.active_task:
            print_error(console, "No previous turn in this session to retry.")
            raise SystemExit(EXIT_TASK_FAILED)
        objective = (state.active_task or {}).get("objective") or state.conversation_summary
        mode = (state.active_task or {}).get("mode") or "coding"
        exit_code = _run_async(_run_local_ai_command(config, workspace_root, objective, mode, model, provider, ()))
        if exit_code != EXIT_OK:
            raise SystemExit(exit_code)
        return

    creds = load_credentials()
    if creds is None:
        print_error(console, "Not authenticated -- run `tamfis-code login` first, or omit --remote to retry locally.")
        raise SystemExit(EXIT_AUTH_FAILED)

    async with RemoteAPIClient(config, creds) as client:
        try:
            workspace = await resolve_workspace(client, workspace_root)
            if task_id is None:
                failed = await find_recent_task(client, workspace.session_id, only_status={"failed", "cancelled"})
                if failed is None:
                    print_error(console, "No recent failed task to retry in this workspace.")
                    raise SystemExit(EXIT_TASK_FAILED)
                task_id = failed["id"]

            print_banner(console, host=config.api_base, workspace_root=workspace.workspace_root, mode="retry", approval_policy=config.approval_policy)
            renderer = StreamRenderer(console)
            outcome = await retry_task_and_stream(
                client, renderer, console,
                session_id=workspace.session_id, task_id=task_id, mode=None,
                # See the matching comment in _run_ai_command -- a one-shot
                # command is still an attended terminal unless stdin is
                # redirected/piped.
                approval_policy=config.approval_policy, interactive=sys.stdin.isatty(),
            )
        except AuthRequiredError:
            print_error(console, "Not authenticated -- run `tamfis-code login` first.")
            raise SystemExit(EXIT_AUTH_FAILED)
        except (RemoteAPIError, httpx.HTTPError) as e:
            print_error(console, str(e))
            raise SystemExit(EXIT_RUNTIME_UNAVAILABLE)

    if outcome.status == "completed":
        if outcome.summary and not renderer.streamed_final_text:
            console.print(outcome.summary)
        return
    print_error(console, outcome.error or f"task {outcome.status}")
    raise SystemExit(EXIT_TASK_FAILED)


@cli.command()
@click.argument("n", type=int, required=False, default=10)
@click.option("--remote", is_flag=True, default=False, help="List mutations from the legacy TamfisGPT Remote Workspace backend instead of the local ledger.")
@click.pass_context
@async_command
async def diffs(ctx: click.Context, n: int, remote: bool):
    """List the last N file mutations (write_file/edit_file) in this workspace's session."""
    config: Config = ctx.obj["config"]
    workspace_root: Path = ctx.obj["workspace_root"]
    console = Console(no_color=not config.colour)

    if not _use_remote(config, remote):
        from .workspace import resolve_local_workspace

        workspace = resolve_local_workspace(workspace_root, discover=False)
        mutations = list(reversed(local_state.get_session_state(workspace.session_id).modified_files[-n:]))
        if not mutations:
            console.print("[dim]No file mutations recorded yet in this session.[/dim]")
            return
        table = Table(show_header=True, header_style="bold")
        for col in ("ID", "OP", "PATH", "+/-", "STATUS", "TURN"):
            table.add_column(col)
        for m in mutations:
            table.add_row(
                str(m.get("mutation_id")), str(m.get("operation")), str(m.get("path")),
                f"+{m.get('lines_added')}/-{m.get('lines_removed')}", str(m.get("revert_status")),
                str(m.get("transaction_id") or ""),
            )
        console.print(table)
        return

    creds = load_credentials()
    if creds is None:
        print_error(console, "Not authenticated -- run `tamfis-code login` first, or omit --remote for the local ledger.")
        raise SystemExit(EXIT_AUTH_FAILED)

    async with RemoteAPIClient(config, creds) as client:
        try:
            workspace = await resolve_workspace(client, workspace_root)
            result = await client.list_file_mutations(workspace.session_id, limit=n)
        except AuthRequiredError:
            print_error(console, "Not authenticated -- run `tamfis-code login` first.")
            raise SystemExit(EXIT_AUTH_FAILED)
        except (RemoteAPIError, httpx.HTTPError) as e:
            print_error(console, str(e))
            raise SystemExit(EXIT_RUNTIME_UNAVAILABLE)

    mutations = result.get("mutations") or []
    if not mutations:
        console.print("[dim]No file mutations recorded yet in this session.[/dim]")
        return
    table = Table(show_header=True, header_style="bold")
    for col in ("ID", "OP", "PATH", "+/-", "STATUS"):
        table.add_column(col)
    for m in mutations:
        table.add_row(
            str(m.get("id")), str(m.get("operation")), str(m.get("path")),
            f"+{m.get('lines_added')}/-{m.get('lines_removed')}", str(m.get("revert_status")),
        )
    console.print(table)


@cli.command(name="diff")
@click.argument("mutation_id", required=False)
@click.option("--remote", is_flag=True, default=False, help="Show a diff from the legacy TamfisGPT Remote Workspace backend instead of the local ledger.")
@click.pass_context
@async_command
async def diff_command(ctx: click.Context, mutation_id: Optional[str], remote: bool):
    """Show an agent-recorded unified diff (latest mutation by default)."""
    config: Config = ctx.obj["config"]
    workspace_root: Path = ctx.obj["workspace_root"]
    console = Console(no_color=not config.colour)

    if not _use_remote(config, remote):
        from .workspace import resolve_local_workspace

        workspace = resolve_local_workspace(workspace_root, discover=False)
        mutations = local_state.get_session_state(workspace.session_id).modified_files
        selected = next((item for item in mutations if item.get("mutation_id") == mutation_id), None) if mutation_id else (mutations[-1] if mutations else None)
        if selected is None:
            print_error(console, "Mutation not found in this session." if mutation_id else "No file mutations recorded yet.")
            raise SystemExit(EXIT_TASK_FAILED)
        print_unified_diff(console, str(selected.get("unified_diff") or ""), title=f"{selected.get('path')} · {selected.get('mutation_id')}")
        return

    creds = load_credentials()
    if creds is None:
        print_error(console, "Not authenticated -- run `tamfis-code login` first, or omit --remote for the local ledger.")
        raise SystemExit(EXIT_AUTH_FAILED)
    async with RemoteAPIClient(config, creds) as client:
        try:
            workspace = await resolve_workspace(client, workspace_root)
            result = await client.list_file_mutations(workspace.session_id, limit=200 if mutation_id else 1)
        except AuthRequiredError:
            raise SystemExit(EXIT_AUTH_FAILED)
        except (RemoteAPIError, httpx.HTTPError) as exc:
            print_error(console, str(exc))
            raise SystemExit(EXIT_RUNTIME_UNAVAILABLE)
    mutations = result.get("mutations") or []
    selected = next((item for item in mutations if str(item.get("id")) == mutation_id), None) if mutation_id else (mutations[0] if mutations else None)
    if selected is None:
        print_error(console, "Mutation not found in this session." if mutation_id else "No file mutations recorded yet.")
        raise SystemExit(EXIT_TASK_FAILED)
    print_unified_diff(console, str(selected.get("unified_diff") or ""), title=f"{selected.get('path')} · {selected.get('id')}")


@cli.command(name="changes")
@click.pass_context
def changes_command(ctx: click.Context):
    """Alias for a concise list of agent-recorded file changes."""
    ctx.invoke(diffs, n=25)


@cli.command()
@click.argument("mutation_id")
@click.option("--remote", is_flag=True, default=False, help="Revert a mutation on the legacy TamfisGPT Remote Workspace backend instead of the local ledger.")
@click.pass_context
@async_command
async def revert(ctx: click.Context, mutation_id: str, remote: bool):
    """Revert one file mutation by id (see `tamfis-code diffs`), or every mutation from one turn by its transaction id (the "turn_..." id shown alongside each mutation in `tamfis-code diffs`) -- restores each file to its content before that change, or deletes it if that mutation created the file."""
    config: Config = ctx.obj["config"]
    workspace_root: Path = ctx.obj["workspace_root"]
    console = Console(no_color=not config.colour)

    if not _use_remote(config, remote):
        from .safety import revert_mutation, revert_transaction
        from .workspace import resolve_local_workspace

        workspace = resolve_local_workspace(workspace_root, discover=False)
        if mutation_id.startswith("turn_"):
            try:
                result = revert_transaction(workspace.session_id, mutation_id)
            except ValueError as exc:
                print_error(console, str(exc))
                raise SystemExit(EXIT_TASK_FAILED)
            console.print(f"[green]Reverted {len(result['reverted'])} mutation(s)[/green] from turn {mutation_id}")
            if result["error"]:
                print_error(console, f"Stopped after a failure: {result['error']} -- {len(result['remaining'])} mutation(s) in this turn still not reverted: {', '.join(result['remaining'])}")
                raise SystemExit(EXIT_TASK_FAILED)
            return
        try:
            result = revert_mutation(workspace.session_id, mutation_id)
        except ValueError as exc:
            print_error(console, str(exc))
            raise SystemExit(EXIT_TASK_FAILED)
        console.print(f"[green]Reverted[/green] {result.get('path')}")
        return

    creds = load_credentials()
    if creds is None:
        print_error(console, "Not authenticated -- run `tamfis-code login` first, or omit --remote for the local ledger.")
        raise SystemExit(EXIT_AUTH_FAILED)

    async with RemoteAPIClient(config, creds) as client:
        try:
            workspace = await resolve_workspace(client, workspace_root)
            result = await client.revert_file_mutation(workspace.session_id, mutation_id)
        except AuthRequiredError:
            print_error(console, "Not authenticated -- run `tamfis-code login` first.")
            raise SystemExit(EXIT_AUTH_FAILED)
        except (RemoteAPIError, httpx.HTTPError) as e:
            print_error(console, str(e))
            raise SystemExit(EXIT_RUNTIME_UNAVAILABLE)

    console.print(f"[green]Reverted[/green] {result.get('path')}")


# -- background-session management ----------------------------------------

@cli.command()
@click.option("--remote", is_flag=True, default=False, help="Show running/backgrounded/approval-pending status from the legacy TamfisGPT Remote Workspace backend -- standalone sessions are always synchronous within one process, so this concept only applies remotely.")
@click.pass_context
@async_command
async def agents(ctx: click.Context, remote: bool):
    """List sessions and each one's most recent task/status -- what's running, backgrounded, or waiting on approval.

    Not to be confused with `tamfis-code agent-cmd` (run/list/delegate to sub-agents) --
    similarly-named but unrelated commands."""
    config: Config = ctx.obj["config"]
    console = Console(no_color=not config.colour)

    if not _use_remote(config, remote):
        console.print("[dim]Standalone sessions have no background/approval-pending concept (each run is synchronous). Showing known local sessions -- see `tamfis-code sessions`.[/dim]")
        ctx.invoke(sessions)
        return

    creds = load_credentials()
    if creds is None:
        print_error(console, "Not authenticated -- run `tamfis-code login` first, or omit --remote to list local sessions.")
        raise SystemExit(EXIT_AUTH_FAILED)

    async with RemoteAPIClient(config, creds) as client:
        try:
            sessions_list = await client.list_sessions()
        except AuthRequiredError:
            print_error(console, "Not authenticated -- run `tamfis-code login` first.")
            raise SystemExit(EXIT_AUTH_FAILED)
        except (RemoteAPIError, httpx.HTTPError) as e:
            print_error(console, str(e))
            raise SystemExit(EXIT_RUNTIME_UNAVAILABLE)

        table = Table(show_header=True, header_style="bold")
        for col in ("SESSION", "STATUS", "TASK", "WORKSPACE"):
            table.add_column(col)

        for sess in sessions_list:
            session_id = sess.get("id")
            workspace_dir = str(sess.get("working_directory") or "")
            try:
                latest = await find_recent_task(client, session_id, only_status=None, lookback=1)
            except (AuthRequiredError, RemoteAPIError):
                latest = None
            if latest is None:
                task_col = "[dim](no tasks)[/dim]"
                status_col = str(sess.get("status", ""))
            else:
                task_status = str(latest.get("status", ""))
                task_col = f"{latest.get('id')} ({task_status})"
                status_col = "running" if task_status in ACTIVE_TASK_STATUSES else task_status
            table.add_row(str(session_id), status_col, task_col, workspace_dir)

        console.print(table)
        console.print(
            "[dim]tamfis-code attach <session_id> · tamfis-code logs <session_id> · "
            "tamfis-code stop <session_id>[/dim]"
        )


@cli.command()
@click.argument("session_id", type=int)
@click.pass_context
@async_command
async def attach(ctx: click.Context, session_id: int):
    """Reattach to a session's live (or most recent) task stream (legacy Remote Workspace backend only -- a standalone run is always synchronous within one process, so there's nothing to reattach to). The task is not owned by this connection -- Ctrl+C detaches, it does not stop the task."""
    config: Config = ctx.obj["config"]
    console = Console(no_color=not config.colour)
    creds = load_credentials()
    if creds is None:
        print_error(console, "Not authenticated -- run `tamfis-code login` first.")
        raise SystemExit(EXIT_AUTH_FAILED)

    async with RemoteAPIClient(config, creds) as client:
        try:
            workspace = await context_from_session(client, session_id)
            task = await find_recent_task(client, session_id, only_status=ACTIVE_TASK_STATUSES, lookback=1)
            if task is None:
                task = await find_recent_task(client, session_id, only_status=None, lookback=1)
            if task is None:
                print_error(console, f"Session {session_id} has no tasks to attach to.")
                raise SystemExit(EXIT_TASK_FAILED)

            print_banner(console, host=config.api_base, workspace_root=workspace.workspace_root, mode="attach", approval_policy=config.approval_policy)
            console.print(f"[dim]attached to task {task['id']} ({task.get('status')}) -- Ctrl+C to detach[/dim]")
            renderer = StreamRenderer(console)
            outcome = await attach_and_stream(
                client, renderer, console,
                session_id=session_id, task_id=str(task["id"]),
                approval_policy=config.approval_policy, interactive=True,
            )
        except AuthRequiredError:
            print_error(console, "Not authenticated -- run `tamfis-code login` first.")
            raise SystemExit(EXIT_AUTH_FAILED)
        except (RemoteAPIError, httpx.HTTPError) as e:
            print_error(console, str(e))
            raise SystemExit(EXIT_RUNTIME_UNAVAILABLE)

    if outcome.status == "detached":
        console.print(f"[dim]Detached. The task keeps running.[/dim]")
        console.print(f"  tamfis-code attach {session_id}")
        return
    if outcome.status == "completed":
        if outcome.summary and not renderer.streamed_final_text:
            console.print(outcome.summary)
        return
    print_error(console, outcome.error or f"task {outcome.status}")
    raise SystemExit(EXIT_TASK_FAILED)


@cli.command()
@click.pass_context
def detach(ctx: click.Context):
    """Show the reattach command for this workspace's session -- Ctrl+C/Ctrl+D during `attach` is the actual detach action; this is informational."""
    config: Config = ctx.obj["config"]
    workspace_root: Path = ctx.obj["workspace_root"]
    console = Console(no_color=not config.colour)
    matches = [sid for sid in local_state.all_known_session_ids()
               if local_state.get_session_state(sid).workspace_root == str(workspace_root)]
    if not matches:
        console.print("[dim]No known session for this workspace yet -- run `tamfis-code init` or start a task first.[/dim]")
        return
    session_id = matches[-1]
    console.print(f"[dim]Nothing to disconnect locally (each command is its own short-lived process).[/dim]")
    console.print(f"  tamfis-code attach {session_id}")
    console.print(f"  tamfis-code logs {session_id}")


@cli.command()
@click.argument("session_id", type=int)
@click.option("--follow", "follow", is_flag=True, default=False, help="Stream live output instead of printing recent history and exiting.")
@click.option("--tail", "tail", type=int, default=6, help="Number of recent turns to show when not following.")
@click.pass_context
@async_command
async def logs(ctx: click.Context, session_id: int, follow: bool, tail: int):
    """Show (or follow) a session's event history (legacy Remote Workspace backend only). Read-only -- never answers approvals; use `approve`/`reject` for that."""
    config: Config = ctx.obj["config"]
    console = Console(no_color=not config.colour)
    creds = load_credentials()
    if creds is None:
        print_error(console, "Not authenticated -- run `tamfis-code login` first.")
        raise SystemExit(EXIT_AUTH_FAILED)

    async with RemoteAPIClient(config, creds) as client:
        try:
            if not follow:
                thread = await client.get_thread(session_id)
                print_recent_thread(console, thread.get("messages") or [], limit=tail)
                return
            from_event_id = local_state.get_session_state(session_id).last_event_id
            console.print(f"[dim]following session {session_id} from event {from_event_id} -- Ctrl+C to stop watching[/dim]")
            renderer = StreamRenderer(console)
            await follow_session_logs(client, renderer, console, session_id=session_id, from_event_id=from_event_id)
        except AuthRequiredError:
            print_error(console, "Not authenticated -- run `tamfis-code login` first.")
            raise SystemExit(EXIT_AUTH_FAILED)
        except (RemoteAPIError, httpx.HTTPError) as e:
            print_error(console, str(e))
            raise SystemExit(EXIT_RUNTIME_UNAVAILABLE)


@cli.command()
@click.argument("session_id", type=int)
@click.pass_context
@async_command
async def stop(ctx: click.Context, session_id: int):
    """Cancel the active task in a session (legacy Remote Workspace backend only; server-side cancellation, not just local disconnect). Standalone runs can just be interrupted with Ctrl+C."""
    config: Config = ctx.obj["config"]
    console = Console(no_color=not config.colour)
    creds = load_credentials()
    if creds is None:
        print_error(console, "Not authenticated -- run `tamfis-code login` first.")
        raise SystemExit(EXIT_AUTH_FAILED)

    async with RemoteAPIClient(config, creds) as client:
        try:
            task = await find_recent_task(client, session_id, only_status=ACTIVE_TASK_STATUSES, lookback=1)
            if task is None:
                console.print(f"[dim]No active task in session {session_id}.[/dim]")
                return
            await client.cancel_task(str(task["id"]))
            console.print(f"[green]Cancelled[/green] task {task['id']} in session {session_id}.")
        except AuthRequiredError:
            print_error(console, "Not authenticated -- run `tamfis-code login` first.")
            raise SystemExit(EXIT_AUTH_FAILED)
        except (RemoteAPIError, httpx.HTTPError) as e:
            print_error(console, str(e))
            raise SystemExit(EXIT_RUNTIME_UNAVAILABLE)


async def _decide_approval(ctx: click.Context, approval_id: int, decision: str) -> None:
    config: Config = ctx.obj["config"]
    console = Console(no_color=not config.colour)
    creds = load_credentials()
    if creds is None:
        print_error(console, "Not authenticated -- run `tamfis-code login` first.")
        raise SystemExit(EXIT_AUTH_FAILED)

    async with RemoteAPIClient(config, creds) as client:
        try:
            cmd = await client.approve_command(approval_id, decision)
        except AuthRequiredError:
            print_error(console, "Not authenticated -- run `tamfis-code login` first.")
            raise SystemExit(EXIT_AUTH_FAILED)
        except (RemoteAPIError, httpx.HTTPError) as e:
            print_error(console, str(e))
            raise SystemExit(EXIT_RUNTIME_UNAVAILABLE)
    verb = "Approved" if decision.startswith("approve") else "Rejected"
    console.print(f"[green]{verb}[/green] {approval_id} -> status={cmd.get('status')}")


@cli.command()
@click.argument("approval_id", type=int)
@click.option("--once", "scope", flag_value="approve_once", default="approve_once", help="Approve this command only.")
@click.option("--session", "scope", flag_value="approve_session", help="Approve this safety tier for the session.")
@click.pass_context
@async_command
async def approve(ctx: click.Context, approval_id: int, scope: str):
    """Approve a pending command awaiting approval (legacy Remote Workspace backend only -- standalone mode prompts for approval inline, in the same process)."""
    await _decide_approval(ctx, approval_id, scope)


@cli.command()
@click.argument("approval_id", type=int)
@click.option("--reason", default=None, help="Record a human-readable rejection reason in terminal history.")
@click.pass_context
@async_command
async def reject(ctx: click.Context, approval_id: int, reason: Optional[str]):
    """Reject/deny a pending command awaiting approval (legacy Remote Workspace backend only -- standalone mode prompts for approval inline, in the same process)."""
    await _decide_approval(ctx, approval_id, "deny")
    if reason:
        Console(no_color=not ctx.obj["config"].colour).print(f"Reason: {reason}")


@cli.command(name="inspect")
@click.argument("command_id", type=int)
@click.pass_context
@async_command
async def inspect_command(ctx: click.Context, command_id: int):
    """Show the exact command, CWD/risk metadata, and current status (legacy Remote Workspace backend only)."""
    config: Config = ctx.obj["config"]
    console = Console(no_color=not config.colour)
    creds = load_credentials()
    if creds is None:
        print_error(console, "Not authenticated -- run `tamfis-code login` first.")
        raise SystemExit(EXIT_AUTH_FAILED)
    async with RemoteAPIClient(config, creds) as client:
        cmd = await client.get_command(command_id)
        session_detail = await client.get_session(int(cmd["session_id"]))
    console.print(f"Command:\n{cmd.get('command_text', '')}\n")
    console.print(f"Working directory:\n{session_detail.get('working_directory') or '?'}\n")
    console.print(f"Reason:\n{cmd.get('safety_reason') or 'No reason recorded.'}\n")
    console.print(f"Risk:\n{cmd.get('safety_tier') or '?'}\n")
    console.print(f"Status:\n{cmd.get('status') or '?'}\n")
    console.print(f"Command ID:\n{command_id}")


# ============== NEW COMMANDS ==============

@cli.command('completion')
@click.argument('shell', type=click.Choice(['bash', 'zsh', 'fish', 'powershell']))
def completion_cmd(shell: str):
    """Generate shell completion scripts"""
    from .completion import ShellCompleter
    
    generators = {
        'bash': ShellCompleter.generate_bash,
        'zsh': ShellCompleter.generate_zsh,
        'fish': ShellCompleter.generate_fish,
        'powershell': ShellCompleter.generate_powershell,
    }
    
    click.echo(generators[shell]())


@cli.command('agent-cmd')
@click.argument('action', type=click.Choice(['list', 'run', 'info', 'delegate']))
@click.option('--task', '-t', 'tasks', multiple=True, help='Task description (repeatable for delegate)')
@click.option('--file', '-f', help='File to operate on')
@click.option('--max-concurrency', default=1, show_default=True, help='Max concurrent delegated sub-tasks')
@click.option('--provider', default="auto", help="hf, nvidia, openrouter, or auto (default).")
@click.option('--model', default=None, help="Provider-specific model id; defaults to that provider's default model.")
@click.pass_context
def agent_cmd(ctx: click.Context, action: str, tasks: tuple[str, ...], file: str, max_concurrency: int, provider: str, model: Optional[str]):
    """Run subagents for various tasks (analyze/test/doc-gen, or delegate objectives to concurrent sub-tasks).

    Not to be confused with `tamfis-code agents` (list sessions and their status) --
    similarly-named but unrelated commands. See also `/swarm` and `/delegate` in the
    interactive REPL, which run through this same delegation machinery."""
    import asyncio
    import json
    from .agents import AgentManager

    manager = AgentManager()
    task = tasks[0] if tasks else None

    if action == 'list':
        agents = manager.list_agents()
        click.echo("🤖 Available Agents:")
        for a in agents:
            click.echo(f"  - {a['name']}: {a['description']}")
            click.echo(f"    Capabilities: {', '.join(a['capabilities'])}")
        return

    if action == 'info':
        agents = manager.list_agents()
        for a in agents:
            click.echo(f"\n📋 {a['name']}")
            click.echo(f"  Description: {a['description']}")
            click.echo(f"  Capabilities: {', '.join(a['capabilities'])}")
        return

    if action == 'run':
        if not task:
            click.echo("❌ Please specify a task with --task")
            return

        params = {}
        if file:
            params['file'] = file

        result = asyncio.run(manager.execute_task(task, params))
        if 'error' in result:
            click.echo(f"❌ {result['error']}")
        else:
            click.echo(f"✅ Task completed by {result.get('agent', 'unknown')}")
            click.echo(json.dumps(result.get('result', {}), indent=2))
        return

    if action == 'delegate':
        config: Config = ctx.obj["config"]
        if not config.enable_subagent_delegation:
            click.echo(
                "❌ Subagent delegation is disabled. Enable it with "
                "enable_subagent_delegation = true in config.toml, or "
                "TAMFIS_CODE_ENABLE_SUBAGENT_DELEGATION=1."
            )
            raise SystemExit(EXIT_INVALID_ARGS)
        if not tasks:
            click.echo("❌ Please specify at least one --task")
            raise SystemExit(EXIT_INVALID_ARGS)

        from .local_chat import resolve_provider_type
        from .providers import ProviderManager

        try:
            provider_type = resolve_provider_type(provider)
        except ValueError as exc:
            raise click.UsageError(str(exc))

        workspace_root: Path = ctx.obj["workspace_root"]
        console = Console(no_color=not config.colour)
        provider_manager = ProviderManager()

        results = _run_async(manager.execute_tasks(
            list(tasks), manager=provider_manager, provider=provider_type, model=model,
            console=console, workspace_root=str(workspace_root),
            approval_policy=config.approval_policy, max_concurrency=max_concurrency,
        ))
        for r in results:
            marker = "✅" if r["status"] == "completed" else "❌"
            click.echo(f"{marker} {r['description']}")
            summary = (r.get("result") or {}).get("summary") or (r.get("result") or {}).get("error")
            if summary:
                click.echo(f"   {summary}")
        return


@cli.command('tools')
@click.argument('action', type=click.Choice(['list', 'call']))
@click.option('--name', '-n', help='Tool name')
@click.option('--params', '-p', help='Parameters as JSON')
def tools_cmd(action: str, name: str, params: str):
    """List and call MCP tools"""
    import asyncio
    import json
    from .mcp import MCPServer
    
    server = MCPServer()
    
    if action == 'list':
        tools = asyncio.run(server.list_tools_async())
        click.echo("🔧 Available Tools:")
        for t in tools:
            click.echo(f"  - {t['name']}: {t['description']}")
        return
    
    if action == 'call':
        if not name:
            click.echo("❌ Please specify a tool with --name")
            return

        if name in {"write_file", "execute_command"}:
            click.echo(
                "Approval required: direct mutation through 'tools call' is disabled. "
                "Use 'tamfis-code ask' or 'tamfis-code chat' so the command, working "
                "directory, risk, and approval record are shown first."
            )
            return
        
        params_dict = json.loads(params) if params else {}
        result = asyncio.run(server.call_tool(name, params_dict))
        if result.get('success'):
            click.echo(json.dumps(result.get('result', {}), indent=2))
        else:
            click.echo(f"❌ {result.get('error', 'Unknown error')}")


@cli.command('index')
@click.argument('path', required=False, default='.')
@click.option('--search', '-s', help='Search for symbols')
@click.option('--kind', '-k', help='Filter by symbol kind')
@click.option('--stats', is_flag=True, help='Show index statistics')
def index_cmd(path: str, search: str, kind: str, stats: bool):
    """Index and search code"""
    import json
    from .indexer import CodeIndexer
    from pathlib import Path
    
    root_path = Path(path)
    if not root_path.exists():
        click.echo(f"❌ Path '{path}' not found")
        return
    
    indexer = CodeIndexer(root_path)
    
    if search:
        indexer.load_index()
        results = indexer.search_symbol(search, kind)
        if not results:
            click.echo(f"No symbols found matching '{search}'")
            return
        
        click.echo(f"📋 Found {len(results)} symbols:")
        for r in results:
            click.echo(f"  {r.name} ({r.kind}) - {r.file_path}:{r.line_start}")
        return
    
    if stats:
        indexer.load_index()
        stats_data = indexer.get_stats()
        click.echo("📊 Index Statistics:")
        click.echo(f"  Files: {stats_data['files']}")
        click.echo(f"  Total Symbols: {stats_data['total_symbols']}")
        click.echo(f"  Languages: {stats_data['languages']}")
        click.echo(f"  Symbol Kinds: {stats_data['symbol_kinds']}")
        return
    
    # Index the path
    click.echo(f"📁 Indexing: {path}")
    count = indexer.index()
    click.echo(f"✅ Indexed {count} files")


# The standalone `metrics` command used to start an empty MetricsTracker and
# sleep 60s printing `\r`-overwritten zeros -- nothing ever called .record()
# on it, since it was disconnected from the real streaming path. Superseded
# by StreamRenderer's live token/rate display (render.py), which records real
# per-task estimates as assistant_delta events arrive.


def main() -> None:
    try:
        cli()
    except PermissionError as exc:
        # Without this, a permission/ownership mismatch on CONFIG_DIR (e.g. it
        # was created by a different user) surfaces as a raw traceback on the
        # very first local state write of every single invocation -- the CLI
        # "dies on the same step every time" with no actionable message, and
        # since nothing ever got persisted, the next invocation looks like a
        # fresh start and re-proposes the same plan instead of progressing.
        click.echo(
            f"✗ Permission denied writing local session state: {exc}\n"
            f"  {CONFIG_DIR} (or a file inside it) is likely owned by a different "
            "user than the one running tamfis-code. Check `ls -la "
            f"{CONFIG_DIR}` and fix its ownership, then retry.",
            err=True,
        )
        raise SystemExit(EXIT_LOCAL_STATE_ERROR)


if __name__ == "__main__":
    cli()

@cli.command('providers')
@click.pass_context
def providers_command(ctx: click.Context):
    """Show available AI providers and their status"""
    from .providers import get_provider_status
    from rich.table import Table
    from rich.console import Console
    
    config: Config = ctx.obj["config"]
    console = Console(no_color=not config.colour)
    
    status = get_provider_status()
    
    table = Table(show_header=True, header_style="bold")
    table.add_column("Provider")
    table.add_column("Status")
    table.add_column("Default Model")
    table.add_column("Priority")
    table.add_column("Reasoning")
    
    for p in status.get("available", []):
        status_str = "🟢 Available" if p.get("available") else "🔴 Unavailable"
        reasoning_str = "✅" if p.get("reasoning_supported") else "❌"
        table.add_row(
            p.get("name", "Unknown"), 
            status_str, 
            p.get("default_model", "-"), 
            str(p.get("priority", 999)),
            reasoning_str
        )
    
    console.print(table)
    console.print(f"[dim]Default provider: {status.get('default', 'none')}[/dim]")


@cli.command('local')
@click.argument('objective', required=False)
@click.option('--provider', default="auto", help="hf, nvidia, openrouter, or auto (default).")
@click.option('--model', default=None, help="Provider-specific model id; defaults to that provider's default model.")
@click.option('--no-tools', 'no_tools', is_flag=True, default=False, help="Disable read-only repo tools (read_file/list_directory/search_code/get_git_info) for this turn.")
@click.option('--agent', 'full_agent', is_flag=True, default=False, help="Full read/write/execute tool access (write_file/edit_file/execute_command) via the local risk/approval/mutation-ledger layer, instead of read-only Q&A. Standalone -- no TamfisGPT backend involved.")
@click.option('--repl', 'run_repl', is_flag=True, default=False, help="Start an interactive local chat loop instead of a single turn.")
@click.pass_context
def local_command(ctx: click.Context, objective: Optional[str], provider: str, model: Optional[str], no_tools: bool, full_agent: bool, run_repl: bool):
    """Offline chat with a directly-configured LLM provider -- no TamfisGPT
    account, login, or network round-trip to the backend at all.
    HF/NVIDIA/OpenRouter are available if you've set your own key in the
    environment.

    Read-only repo tools (read_file/list_directory/search_code/get_git_info)
    are available so the model can answer questions about this directory.
    Pass --agent for full read/write/execute capability -- that path has its
    own local risk classifier, approval prompts (per --approval-policy /
    config.toml), and file-mutation ledger (see `tamfis-code diffs`/`revert`
    once wired to it), since there's no server backing it anymore.
    """
    from .local_chat import resolve_provider_type, run_local_turn, stream_local_turn
    from .providers import ProviderManager

    config: Config = ctx.obj["config"]
    console = Console(no_color=not config.colour)

    try:
        provider_type = resolve_provider_type(provider)
    except ValueError as exc:
        raise click.UsageError(str(exc))

    manager = ProviderManager()
    use_tools = not no_tools

    if full_agent:
        from .render import StreamRenderer
        from .runner_local import run_local_agent_turn
        from .workspace import resolve_local_workspace

        workspace = resolve_local_workspace(ctx.obj["workspace_root"])

        async def _one_agent_turn(messages: list) -> str:
            renderer = StreamRenderer(console)
            outcome = await run_local_agent_turn(
                manager, provider_type, model, messages, console, renderer,
                workspace_root=workspace.workspace_root, session_id=workspace.session_id,
                approval_policy=config.approval_policy, interactive=sys.stdin.isatty(),
                cli_config=config, allow_swarm_tool=True,
            )
            renderer.finish()
            if outcome.status != "completed":
                raise RuntimeError(outcome.error or "Task did not complete")
            return outcome.summary or ""

        if run_repl:
            console.print(f"[dim]Local standalone agent mode (session {workspace.session_id}, {workspace.workspace_root}) -- Ctrl+D or /exit to quit.[/dim]")
            history: list = []
            while True:
                try:
                    text = console.input("[bold cyan]you> [/bold cyan]").strip()
                except (EOFError, KeyboardInterrupt):
                    console.print()
                    break
                if not text or text in {"/exit", "/quit"}:
                    break
                history.append({"role": "user", "content": text})
                try:
                    answer = _run_async(_one_agent_turn(history))
                except Exception as exc:
                    print_error(console, str(exc))
                    history.pop()
                    continue
                history.append({"role": "assistant", "content": answer})
            return

        if not objective:
            raise click.UsageError("Provide an objective, or pass --repl for an interactive loop.")
        try:
            _run_async(_one_agent_turn([{"role": "user", "content": objective}]))
        except Exception as exc:
            print_error(console, str(exc))
            raise SystemExit(EXIT_TASK_FAILED)
        return

    async def _one_turn(messages: list) -> str:
        if use_tools:
            answer = await run_local_turn(manager, provider_type, messages, model, console, use_tools=True)
            console.print(answer)
            return answer
        parts: list[str] = []
        async for chunk in stream_local_turn(manager, provider_type, messages, model):
            console.print(chunk, end="")
            parts.append(chunk)
        console.print()
        return "".join(parts)

    if run_repl:
        console.print("[dim]Local/offline mode -- Ctrl+D or /exit to quit.[/dim]")
        history: list = []
        while True:
            try:
                text = console.input("[bold cyan]you> [/bold cyan]").strip()
            except (EOFError, KeyboardInterrupt):
                console.print()
                break
            if not text or text in {"/exit", "/quit"}:
                break
            history.append({"role": "user", "content": text})
            try:
                answer = _run_async(_one_turn(history))
            except Exception as exc:
                print_error(console, str(exc))
                history.pop()
                continue
            history.append({"role": "assistant", "content": answer})
        return

    if not objective:
        raise click.UsageError("Provide an objective, or pass --repl for an interactive loop.")
    try:
        _run_async(_one_turn([{"role": "user", "content": objective}]))
    except Exception as exc:
        print_error(console, str(exc))
        raise SystemExit(EXIT_TASK_FAILED)


@cli.command('screenshot')
@click.argument('url_or_path')
@click.option('--width', '-w', default=1920, help='Screenshot width')
@click.option('--height', '-h', default=1080, help='Screenshot height')
@click.option('--quality', '-q', default=90, help='JPEG quality (1-100)')
@click.option('--format', '-f', default='png', help='Output format (png/jpeg/webp)')
@click.option('--full-page', '-F', is_flag=True, help='Capture full page')
@click.option('--output', '-o', help='Output filename')
@click.pass_context
def screenshot_cmd(ctx: click.Context, url_or_path: str, width: int, height: int, 
                   quality: int, format: str, full_page: bool, output: Optional[str]):
    """Take a real Playwright screenshot of a URL (or copy/render a local image)."""
    config: Config = ctx.obj["config"]
    console = Console(no_color=not config.colour)

    try:
        if url_or_path.startswith(("http://", "https://")):
            if format.lower() != "png":
                raise ValueError("Live browser screenshots use PNG; pass --format png.")
            from .mcp import get_browser_tool_class

            browser_tool_class = get_browser_tool_class()
            if browser_tool_class is None:
                raise RuntimeError(
                    "Live browser screenshots need a co-located tamgpt6 checkout "
                    "(sibling directory, TAMFIS_MONOREPO_ROOT, or running from inside "
                    "one) -- none was found. A local file/image path doesn't need this."
                )
            result = browser_tool_class().execute(
                url=url_or_path,
                action="screenshot",
                viewport_width=width,
                viewport_height=height,
                full_page=full_page,
                screenshot_name=output,
                _trusted_source_client="tamfis-code",
                _trusted_workspace_root=str(ctx.obj["workspace_root"]),
                _trusted_task_id="cli-screenshot",
                _trusted_mode="audit",
            )
            if not result.get("success"):
                raise RuntimeError(result.get("error") or "Browser screenshot failed")
            console.print(f"[green]Screenshot saved:[/green] {result['screenshot_path']}")
            console.print(f"[green]URL:[/green] {result['screenshot_url']}")
        else:
            from .screenshot import ScreenshotOptions, ScreenshotTaker

            taker = ScreenshotTaker()
            options = ScreenshotOptions(
                width=width, height=height, quality=quality,
                format=format, full_page=full_page,
            )
            result = taker.take_screenshot(url_or_path, filename=output, options=options)
            console.print(f"[green]Screenshot saved:[/green] {result}")
    except Exception as e:
        console.print(f"[red]Screenshot failed: {e}[/red]")
        raise SystemExit(EXIT_TASK_FAILED)

# Import enforcer
from .enforcer import TestEnforcer, run_enforcer, add_enforcer_command

# `add_enforcer_command` registers a real `enforce` command backed by methods
# that actually exist on TestEnforcer (--python/--node/--frontend). A previous
# `enforce_cmd` defined directly here duplicated this with --shell/--type
# flags wired to _run_shell_checks/_run_type_checks, neither of which exist on
# TestEnforcer -- using those flags raised AttributeError at runtime.
add_enforcer_command(cli)
