"""Interactive terminal composer with multiline history, explicit chat/audit/
plan/agent modes, durable saved-plan execution, streamed tools and approvals,
session recovery, diffs/revert, model routing, context checkpoints, and queue
management.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from . import state as local_state
from .api_client import AuthRequiredError, RemoteAPIClient, RemoteAPIError
from .config import APPROVAL_MODES, CONFIG_DIR, Config, MODE_ALIASES
from .doctor import run_doctor
from .render import StreamRenderer, print_banner, print_error, print_recent_thread, print_unified_diff
from .runner import resolve_approval_decision, retry_task_and_stream, run_ai_task_and_stream, run_shell_command
from .runner_local import run_local_agent_turn, run_local_shell_command
from .safety import revert_mutation as local_revert_mutation
from .tasks import find_recent_task
from .workspace import (
    WorkspaceContext, blocking_dirty_files, context_from_session, discover_local_repository,
    find_resumable_session, resolve_local_workspace,
)

HELP_TEXT = """\
Natural-language text submits a full coding-agent task (mode: coding).
$ <command>            explicit shell command
/run <command>          explicit shell command
/shell <command>        explicit shell command
/chat <question>         conversational/read-only coding assistance
/audit <objective>      AI audit mode (read-only)
/plan <objective>        create and save an executable plan (no changes)
/plans                   list saved plans for this session
/plan show [plan_id]     show a saved plan (latest/active if omitted)
/execute-plan [plan_id]  execute a saved plan (latest/active if omitted)
/agent <objective>       full coding-agent mode (inspect + edit + verify)
/execute <objective>     AI execute mode (tools + approval policy)

/help                show this help
/status              show session/workspace/approval status
/context             show cached repository/task context
/reports             show the repository report index
/queue               show queued instructions
/queue <instruction> append a follow-up instruction
/cwd                 show the current workspace root
/doctor              run connectivity/auth checks
/resume [session_id]  switch to another session (most recent if omitted)
/retry [task_id]      retry a failed task (most recent failure if omitted)
/agents              list sessions and their latest task status
/delegate <a> | <b>  run objectives a, b, ... as concurrent delegated sub-tasks
                      (requires enable_subagent_delegation = true in config.toml,
                      or TAMFIS_CODE_ENABLE_SUBAGENT_DELEGATION=1)
/diffs [n]           show the last n file mutations in this session (default 10)
/diff [mutation_id]  show a semantic unified diff (latest if omitted)
/revert <mutation_id> restore a file to its content before that mutation (or
                      delete it, if that mutation created the file)
/detach              exit without cancelling anything server-side (same as /exit --
                      nothing in this REPL ties a task's lifetime to this process; see
                      `tamfis-code attach <session_id>` in another terminal to reconnect)
/clear               clear the screen
/compact             save a durable checkpoint of the current task context
/permissions         show approval policy and immutable server safeguards
/mode                show the active approval mode and available modes
/mode <name>         switch mode: manual | accept-edits | auto | plan
/model               show the active model route
/model list [route]  list coding models (route: hf or openrouter)
/model auto          restore Hugging Face -> OpenRouter automatic routing
/model <route> [id]  pin a provider, optionally with a catalog model id
/tools               show the tools exposed to tamfis-code tasks
/pty start [command]  start a persistent background terminal (default: bash)
/pty list             list this session's background terminals
/pty send <id> <text> send input to a background terminal (a trailing \\n
                      submits the line; omit it to send raw keystrokes)
/pty read <id>        show output produced since the last /pty read
/pty kill <id>        terminate a background terminal
/exit                quit (also: /quit, Ctrl+D, Ctrl+C)

Not yet implemented in this pass: /notifications.
"""


class Intent:
    def __init__(self, kind: str, *, command: str = "", objective: str = "", mode: str = "coding"):
        self.kind = kind
        self.command = command
        self.objective = objective
        self.mode = mode


def contextualize_short_reply(raw: str, *, has_context: bool) -> str:
    """Turn terse conversational controls into explicit contextual intents.

    Keep truly new-session input untouched; a bare ``1`` only means "step
    one" when there is an active/prior task to refer to.
    """
    text = raw.strip()
    if not has_context:
        return text
    lowered = text.lower().rstrip(".! ")
    if lowered in {"ok", "okay", "yes", "y", "sure", "go", "proceed"}:
        return "Yes. Proceed with the action or next step you just proposed."
    if lowered in {"no", "n", "stop"}:
        return "No. Do not proceed with the action you just proposed."
    if lowered.isdigit():
        return f"Proceed with step {int(lowered)} from your immediately preceding plan."
    match = re.fullmatch(r"(?:step\s*)?(\d+)", lowered)
    if match:
        return f"Proceed with step {int(match.group(1))} from your immediately preceding plan."
    return text


def parse_intent(raw: str) -> Intent:
    text = raw.strip()
    if text.startswith("$ "):
        return Intent("shell", command=text[2:].strip())
    if text.startswith("/run "):
        return Intent("shell", command=text[5:].strip())
    if text.startswith("/shell "):
        return Intent("shell", command=text[7:].strip())
    if text.startswith("/audit "):
        return Intent("ai", objective=text[7:].strip(), mode="audit")
    if text.startswith("/chat "):
        return Intent("ai", objective=text[6:].strip(), mode="chat")
    if text.startswith("/plan "):
        return Intent("ai", objective=text[6:].strip(), mode="plan")
    if text == "/execute-plan" or text.startswith("/execute-plan "):
        return Intent("saved_plan", command=text[len("/execute-plan"):].strip())
    if text.startswith("/agent "):
        return Intent("ai", objective=text[7:].strip(), mode="execute")
    if text.startswith("/execute "):
        return Intent("ai", objective=text[9:].strip(), mode="execute")
    if text.startswith("/ask "):
        return Intent("ai", objective=text[5:].strip(), mode="coding")
    return Intent("ai", objective=text, mode="coding")


async def run_interactive(
    client: Optional[RemoteAPIClient], config: Config, workspace: WorkspaceContext,
    *, provider: str = "auto", model: Optional[str] = None,
) -> None:
    """Drives the interactive REPL. `client=None` means standalone mode: no
    TamfisGPT Remote Workspace backend, no login -- AI turns/shell commands
    run through runner_local.py's local agent loop calling `provider`
    directly instead. A handful of features that are inherently remote-server
    concepts (background PTY terminals, a remote model catalog) have no
    local equivalent yet and degrade to a clear message rather than crashing;
    everything else (diffs/revert, resume, agents listing, retry, delegate,
    doctor) has a real local implementation in this mode.
    """
    standalone = client is None
    provider_manager = None
    provider_type = None
    last_turn: Optional[tuple[str, str]] = None  # (objective, mode) -- standalone /retry target

    if standalone:
        from .local_chat import resolve_provider_type
        from .providers import ProviderManager

        console = Console(no_color=not config.colour)
        try:
            provider_type = resolve_provider_type(provider)
        except ValueError as exc:
            print_error(console, str(exc))
            return
        provider_manager = ProviderManager()
    else:
        console = Console(no_color=not config.colour)

    print_banner(
        console, host=(f"local:{provider}" if standalone else config.api_base),
        workspace_root=workspace.workspace_root, mode="interactive", approval_policy=config.approval_policy,
    )
    console.print("[dim]Type /help for commands. Paste up to 1,000,000 characters; Alt+Enter adds a newline. Ctrl+D or Ctrl+C exits.[/dim]\n")

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    history_path = CONFIG_DIR / "history"
    bindings = KeyBindings()
    # Local only: which byte offset this REPL has already displayed per PTY
    # id, so /pty read shows only new output. The server (RemotePtySession/
    # pty_broker's ring buffer) is the durable source of truth -- losing
    # this dict on restart just means the next /pty read re-shows output
    # already seen, never data loss. (Remote mode only -- standalone has no
    # PTY backend at all yet, see the /pty handler below.)
    pty_offsets: dict[str, int] = {}

    @bindings.add("enter")
    def _submit(event) -> None:
        event.current_buffer.validate_and_handle()

    @bindings.add("escape", "enter")
    def _newline(event) -> None:
        event.current_buffer.insert_text("\n")

    # multiline=True is essential for bracketed terminal paste: embedded
    # newlines remain part of one objective instead of submitting the first
    # line and treating the remainder as accidental follow-up commands.  The
    # bindings retain familiar Enter-to-submit behaviour.
    session: PromptSession = PromptSession(
        history=FileHistory(str(history_path)), multiline=True, key_bindings=bindings,
    )

    while True:
        queued_item = next((item for item in local_state.get_session_state(workspace.session_id).queued_user_instructions
                            if item.get("status") == "queued"), None)
        if queued_item and queued_item.get("classification") not in {"pause", "cancel"}:
            local_state.update_instruction(workspace.session_id, str(queued_item.get("id")), "running")
            text = str(queued_item.get("text") or "")
            console.print(f"[cyan]◆ Queued instruction[/cyan] {queued_item.get('id')}: {text}")
        elif queued_item:
            local_state.update_instruction(workspace.session_id, str(queued_item.get("id")), "completed")
            console.print(f"[yellow]◆ {queued_item.get('classification')} boundary reached[/yellow]")
            continue
        else:
            try:
                text = await session.prompt_async("tamfis-code> ")
            except KeyboardInterrupt:
            # NOTE: this used to just `continue`, silently redrawing the
            # prompt -- Ctrl+C appeared to do nothing at all, with no
            # documented way to exit except Ctrl+D or typing /exit. Ctrl+C
            # while an AI task/command is actively streaming is a separate,
            # already-correct code path (runner.py's _install_sigint_watcher
            # cancels just that task, not the whole process) -- this only
            # covers the idle prompt, where Ctrl+C exiting is the standard
            # expectation.
                break
            except EOFError:
                break

        text = text.strip()
        if not text:
            continue
        if len(text) > 1_000_000:
            print_error(console, "Objective exceeds the 1,000,000 character safety limit.")
            continue
        if text in ("/exit", "/quit", "/detach"):
            # No task submitted through this REPL outlives this process's
            # lifetime any differently based on which of these three the
            # user types -- background durability comes from `--bg` /
            # `attach` in a fresh invocation, not from how THIS process
            # exits. /detach exists as a documented, discoverable alias
            # matching the Phase 19 slash-command surface, not because it
            # behaves differently from /exit today.
            break
        # A bare "/" fell all the way through to parse_intent() before, which
        # doesn't treat "/" as a prefix of anything (every command check
        # requires an exact match or a trailing space with content) -- it
        # was submitted to the AI as a one-character objective instead of
        # showing the command list the way typing "/" alone is expected to.
        if text in ("/help", "/"):
            console.print(HELP_TEXT)
            if standalone:
                console.print(
                    "[dim]Standalone mode: /pty and /model list have no local equivalent yet (they need a "
                    "persistent server / remote model catalog) -- everything else above (diffs/revert, resume, "
                    "agents, retry, delegate, doctor) runs fully locally, no TamfisGPT backend involved.[/dim]"
                )
            continue
        if text == "/cwd":
            console.print(workspace.workspace_root)
            continue
        if text == "/status":
            state = local_state.get_session_state(workspace.session_id)
            identity_line = (
                f"session_id={workspace.session_id}  (standalone, local session)"
                if standalone else f"session_id={workspace.session_id}  server_id={workspace.server_id}"
            )
            backend_line = (
                f"approval_policy={config.approval_policy}  provider={provider_type.value if provider_type else provider}"
                if standalone else f"approval_policy={config.approval_policy}  api_base={config.api_base}"
            )
            console.print(
                f"{identity_line}\n"
                f"workspace_root={workspace.workspace_root}\n"
                f"repository_root={state.repository_root or '-'}  branch={state.active_branch or '-'}\n"
                f"phase={state.current_phase}  execution={state.execution_status}\n"
                f"queue={sum(1 for item in state.queued_user_instructions if item.get('status') == 'queued')}  "
                f"validations={len(state.validation_results)}  issues={len(state.unresolved_issues)}\n"
                f"saved_plans={len(state.saved_plans)}  active_plan={state.active_plan_id or '-'}\n"
                f"{backend_line}\n"
                f"model={state.selected_model}  route={state.selected_provider or 'auto'}"
            )
            continue
        if text == "/context":
            context = discover_local_repository(workspace.session_id, Path(workspace.workspace_root))
            state = local_state.get_session_state(workspace.session_id)
            console.print(f"repository={context.get('repository_root')}  branch={context.get('branch') or '-'}  dirty={context.get('dirty')}")
            console.print(f"cwd={context.get('working_directory')}  indexed_files={context.get('indexed_file_count')}")
            console.print(f"task={(state.active_task or {}).get('objective') or state.conversation_summary or '-'}")
            for path in context.get("instruction_files", []):
                console.print(f"  instruction: {path}")
            continue
        if text == "/reports":
            discover_local_repository(workspace.session_id, Path(workspace.workspace_root))
            reports = local_state.get_session_state(workspace.session_id).discovered_reports
            if not reports:
                console.print("[dim]No matching reports discovered in this repository.[/dim]")
            for report in reports:
                console.print(f"  {str(report.get('modified_at', ''))[:10]}  {report.get('verification')}  {report.get('path')}")
            continue
        if text == "/plans":
            state = local_state.get_session_state(workspace.session_id)
            if not state.saved_plans:
                console.print("[dim]No saved plans yet. Create one with /plan <objective>.[/dim]")
                continue
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
            continue
        if text == "/plan" or text == "/plan show" or text.startswith("/plan show "):
            plan_id = text[len("/plan show"):].strip() if text.startswith("/plan show") else ""
            plan = local_state.get_plan(workspace.session_id, plan_id or None)
            if plan is None:
                print_error(console, "Plan not found. Use /plans to list saved plans.")
                continue
            console.print(
                f"[bold]{plan.get('id')}[/bold] · {plan.get('status', 'ready')}\n"
                f"[dim]Objective:[/dim] {plan.get('objective', '')}"
            )
            console.print(Markdown(str(plan.get("content") or "")))
            continue
        if text == "/queue" or text.startswith("/queue "):
            arg = text[len("/queue"):].strip()
            if arg:
                item = local_state.enqueue_instruction(workspace.session_id, arg)
                console.print(f"[cyan]Queued[/cyan] {item.id}")
            for item in local_state.get_session_state(workspace.session_id).queued_user_instructions:
                console.print(f"  {item.get('id')}  {item.get('status')}  {item.get('classification')}  {item.get('text')}")
            continue
        if text == "/model" or text.startswith("/model "):
            arg = text[len("/model"):].strip()
            state = local_state.get_session_state(workspace.session_id)
            if not arg:
                console.print(
                    f"model={state.selected_model}  "
                    f"provider={state.selected_provider or 'auto (hf -> openrouter)'}"
                )
                continue
            parts = arg.split()

            if standalone:
                from .local_chat import resolve_provider_type as _resolve_provider_type
                from .providers import ProviderManager as _ProviderManager, ProviderType

                if parts[0].lower() == "list":
                    route_arg = parts[1].lower() if len(parts) > 1 else None
                    manager = _ProviderManager()
                    routes = [route_arg] if route_arg else [p.value for p in manager.PROVIDERS]
                    table = Table(show_header=True, header_style="bold")
                    for column in ("PROVIDER", "MODELS"):
                        table.add_column(column)
                    shown = False
                    for route_name in routes:
                        try:
                            route_type = _resolve_provider_type(route_name)
                        except ValueError:
                            continue
                        pcfg = manager.PROVIDERS.get(route_type)
                        if pcfg:
                            table.add_row(pcfg.name, ", ".join(pcfg.models) or pcfg.default_model)
                            shown = True
                    console.print(table if shown else "[dim]Unknown provider. Use hf, nvidia, openrouter, or ollama.[/dim]")
                    continue
                if parts[0].lower() == "auto":
                    provider_type = ProviderType.AUTO
                    local_state.save_session_state(workspace.session_id, selected_model="auto", selected_provider=None)
                    console.print("[green]Provider routing set to automatic.[/green]")
                    continue
                try:
                    provider_type = _resolve_provider_type(parts[0])
                except ValueError as exc:
                    print_error(console, f"{exc} Usage: /model auto | /model <hf|nvidia|openrouter|ollama> [model-id]")
                    continue
                model_id = parts[1] if len(parts) > 1 else "auto"
                if len(parts) > 2:
                    print_error(console, "Model ids cannot contain spaces.")
                    continue
                model = None if model_id == "auto" else model_id
                local_state.save_session_state(
                    workspace.session_id, selected_model=model_id, selected_provider=parts[0].lower(),
                )
                console.print(f"[green]Pinned {parts[0].lower()} route[/green]  model={model_id}")
                continue

            if parts[0].lower() == "list":
                route = parts[1].lower() if len(parts) > 1 else None
                if route not in (None, "hf", "openrouter"):
                    print_error(console, "Usage: /model list [hf|openrouter]")
                    continue
                api_provider = "huggingface" if route == "hf" else route
                try:
                    result = await client.list_models(api_provider)
                except (AuthRequiredError, RemoteAPIError) as e:
                    print_error(console, str(e))
                    continue
                rows = [m for m in (result.get("models") or []) if "coding" in [str(c).lower() for c in (m.get("categories") or [])]]
                table = Table(show_header=True, header_style="bold")
                for column in ("ID", "PROVIDER", "REASONING", "MAX TOKENS"):
                    table.add_column(column)
                for item in rows[:40]:
                    table.add_row(
                        str(item.get("id")), str(item.get("provider") or item.get("backend") or ""),
                        str(item.get("reasoning") or "-"), str(item.get("maxTokens") or "-"),
                    )
                console.print(table if rows else "[dim]No coding models found for that route.[/dim]")
                continue
            if parts[0].lower() == "auto":
                local_state.save_session_state(
                    workspace.session_id, selected_model="auto", selected_provider=None,
                )
                console.print("[green]Model routing set to automatic: Hugging Face, then OpenRouter.[/green]")
                continue
            route = parts[0].lower()
            if route not in ("hf", "openrouter"):
                print_error(console, "Usage: /model auto | /model <hf|openrouter> [catalog-model-id]")
                continue
            model_id = parts[1] if len(parts) > 1 else "auto"
            if len(parts) > 2:
                print_error(console, "Model ids cannot contain spaces.")
                continue
            if model_id != "auto":
                api_provider = "huggingface" if route == "hf" else route
                try:
                    available = await client.list_models(api_provider)
                except (AuthRequiredError, RemoteAPIError) as e:
                    print_error(console, str(e))
                    continue
                ids = {str(item.get("id")) for item in (available.get("models") or [])}
                if model_id not in ids:
                    print_error(console, f"Unknown {route} model '{model_id}'. Use /model list {route}.")
                    continue
            local_state.save_session_state(
                workspace.session_id, selected_model=model_id, selected_provider=route,
            )
            console.print(f"[green]Pinned {route} route[/green]  model={model_id}")
            continue
        if text == "/tools":
            table = Table(show_header=True, header_style="bold")
            table.add_column("Tool")
            table.add_column("Purpose")
            table.add_column("Safeguard")
            if standalone:
                table.add_row("read_file", "Read a file's contents", "Read-only")
                table.add_row("list_directory", "List a directory's contents", "Read-only")
                table.add_row("search_code", "ripgrep-backed content search", "Read-only")
                table.add_row("get_git_info", "Branch/HEAD/status for a repo path", "Read-only")
                table.add_row("edit_file", "Exact, uniqueness-checked replacement", "Local risk classifier + approval + mutation ledger")
                table.add_row("write_file", "Create or fully replace a file", "Workspace-boundary check + approval + mutation ledger")
                table.add_row("execute_command", "Run a shell command", "Local risk classifier + approval (no sandboxing)")
                table.add_row("browser", "Public Chromium navigation and screenshots", "Only if a monorepo browser tool is co-located")
                console.print(table)
                console.print(
                    "[dim]This is the real local tool set (mcp.py) the standalone agent loop uses -- "
                    "see safety.py for how risk is classified and see /diffs for the mutation ledger.[/dim]"
                )
                continue
            table.add_row("read_file", "Bounded, line-numbered file reads", "Read-only + CWD confinement")
            table.add_row("glob_files", "Find files by filename glob, not full paths", "Read-only + CWD confinement")
            table.add_row("grep_files", "Search repository contents", "Read-only + CWD confinement")
            table.add_row("list_symbols", "List definitions in a source file", "Read-only + CWD confinement")
            table.add_row("find_dependencies", "Inspect imports and require targets", "Read-only + CWD confinement")
            table.add_row("edit_file", "Exact, uniqueness-checked replacement", "Approval + mutation ledger")
            table.add_row("write_file", "Create or fully replace a file", "Approval + mutation ledger")
            table.add_row("extract/repackage_archive", "Inspect and rebuild repository archives", "CWD confinement + approval for writes")
            table.add_row("remote_exec", "Tests, builds, Git and shell operations", "CWD confinement + risk approval")
            table.add_row("browser", "Public Chromium navigation and screenshots", "Clean session + SSRF protection")
            table.add_row("web_search", "Current external research when needed", "Read-only; provider/network policy")
            table.add_row("codex_apps.*", "Optional organization-operated connector gateway", "Unavailable unless explicitly configured")
            table.add_row("claude_apps.*", "Optional organization-operated connector gateway", "Unavailable unless explicitly configured")
            console.print(table)
            console.print("[dim]Use glob_files with patterns like '*.tsx' or 'src/**/*.tsx'. Do not paste a full filesystem path into pattern; use path for the search root instead.[/dim]")
            console.print("[dim]Native coding tools above do not require Codex Apps, Claude Apps, or MCP. Optional connector tools appear only when a real gateway is enabled and successfully discovered.[/dim]")
            continue
        if text == "/pty" or text.startswith("/pty "):
            if standalone:
                print_error(
                    console,
                    "Background terminals require --remote (a persistent server to host them) -- "
                    "there's no standalone equivalent yet. Use $ <command> or /run for one-off local commands.",
                )
                continue
            arg = text[len("/pty"):].strip()
            parts = arg.split(maxsplit=1)
            sub = parts[0].lower() if parts else "list"
            rest = parts[1] if len(parts) > 1 else ""

            if sub == "start":
                shell_command = rest.strip() or "bash"
                try:
                    pty = await client.start_pty(workspace.session_id, shell_command=shell_command)
                    decision = resolve_approval_decision(
                        console, f"[background terminal] {shell_command}", str(pty.get("safety_tier", "medium")),
                        config.approval_policy, interactive=True,
                    )
                    if decision != "approve_once":
                        await client.deny_pty(pty["id"])
                        console.print("[dim]Terminal not started.[/dim]")
                        continue
                    pty = await client.approve_pty(pty["id"])
                except (AuthRequiredError, RemoteAPIError) as e:
                    print_error(console, str(e))
                    continue
                pty_offsets[pty["id"]] = 0
                console.print(f"[green]Started background terminal[/green] {pty['id']}  pid={pty.get('pid')}")
                console.print(f"[dim]/pty send {pty['id']} <text>   /pty read {pty['id']}   /pty kill {pty['id']}[/dim]")
                continue

            if sub == "list":
                try:
                    sessions_list = await client.list_pty(workspace.session_id)
                except (AuthRequiredError, RemoteAPIError) as e:
                    print_error(console, str(e))
                    continue
                if not sessions_list:
                    console.print("[dim]No background terminals in this session. Use /pty start.[/dim]")
                    continue
                table = Table(show_header=True, header_style="bold")
                for column in ("ID", "STATUS", "COMMAND", "PID", "CREATED"):
                    table.add_column(column)
                for item in sessions_list:
                    table.add_row(
                        str(item.get("id"))[:8], str(item.get("status")), str(item.get("shell_command"))[:40],
                        str(item.get("pid") or "-"), str(item.get("created_at") or "")[:19],
                    )
                console.print(table)
                continue

            # Remaining subcommands (send/read/kill) take a target id as
            # their first token; accept either the full UUID or the
            # 8-character prefix /pty list shows.
            id_parts = rest.split(maxsplit=1)
            if not id_parts:
                print_error(console, "Usage: /pty <start|list|send|read|kill> ...")
                continue
            prefix = id_parts[0]
            payload = id_parts[1] if len(id_parts) > 1 else ""
            matches = [pid for pid in pty_offsets if pid == prefix or pid.startswith(prefix)]
            target_id = matches[0] if len(matches) == 1 else prefix

            if sub == "send":
                if not payload:
                    print_error(console, "Usage: /pty send <id> <text>")
                    continue
                try:
                    await client.write_pty(target_id, payload)
                except (AuthRequiredError, RemoteAPIError) as e:
                    print_error(console, str(e))
                continue

            if sub == "read":
                try:
                    result = await client.read_pty(target_id, since=pty_offsets.get(target_id, 0))
                except (AuthRequiredError, RemoteAPIError) as e:
                    print_error(console, str(e))
                    continue
                pty_offsets[target_id] = int(result.get("offset", 0))
                if result.get("data"):
                    console.print(result["data"], end="")
                if not result.get("alive", True):
                    console.print(f"\n[yellow]Terminal {target_id[:8]} is {result.get('status', 'exited')}.[/yellow]")
                continue

            if sub == "kill":
                try:
                    result = await client.kill_pty(target_id)
                except (AuthRequiredError, RemoteAPIError) as e:
                    print_error(console, str(e))
                    continue
                pty_offsets.pop(target_id, None)
                console.print(f"[yellow]Terminal {target_id[:8]} {result.get('status', 'killed')}.[/yellow]")
                continue

            print_error(console, "Usage: /pty <start|list|send|read|kill> ...")
            continue
        if text == "/permissions":
            console.print(f"approval_policy={config.approval_policy}")
            console.print("[dim]Server safeguards always enforce workspace scope, command risk classification, ownership, and approval state; client policy cannot widen them.[/dim]")
            continue
        if text == "/mode" or text.startswith("/mode "):
            arg = text[len("/mode"):].strip().lower()
            if not arg:
                console.print(f"Current mode: [bold]{config.approval_policy}[/bold]")
                console.print(
                    "[dim]/mode manual        prompt for every risky action (the default)\n"
                    "/mode accept-edits  auto-approve safe/medium-risk actions, still prompt for dangerous ones\n"
                    "/mode auto           never prompt (server safeguards still apply)\n"
                    "/mode plan           read-only: propose without executing[/dim]"
                )
                continue
            resolved = MODE_ALIASES.get(arg, arg if arg in APPROVAL_MODES else None)
            if resolved is None:
                print_error(console, f"Unknown mode '{arg}'. Use manual, accept-edits, auto, or plan.")
                continue
            config.approval_policy = resolved
            console.print(f"[green]Mode set to[/green] {arg} [dim]({resolved})[/dim]")
            continue
        if text == "/compact":
            state = local_state.get_session_state(workspace.session_id)
            summary = (state.active_task or {}).get("objective") or state.conversation_summary or "Session checkpoint"
            local_state.checkpoint(workspace.session_id, reason="user_compact", summary=summary)
            console.print("[green]Context checkpoint saved.[/green]")
            continue
        if text == "/doctor":
            if standalone:
                from .providers import get_provider_status as _get_provider_status

                status = _get_provider_status()
                table = Table(show_header=True, header_style="bold")
                for column in ("PROVIDER", "CONFIGURED", "KEY"):
                    table.add_column(column)
                for name, info in status["config"].items():
                    configured = "[green]yes[/green]" if info["api_key_set"] or name == "ollama" else "[dim]no[/dim]"
                    table.add_row(name, configured, info["key_preview"])
                console.print(table)
                console.print(
                    f"[dim]Currently selected: {provider_type.value if provider_type else provider}  "
                    f"· auto would pick: {status['default']}[/dim]"
                )
                continue
            await run_doctor(config, console, Path(workspace.workspace_root), session_id=workspace.session_id)
            continue
        if text == "/agents":
            if standalone:
                for sid in local_state.all_known_session_ids():
                    sess_state = local_state.get_session_state(sid)
                    marker = " *" if sid == workspace.session_id else ""
                    console.print(f"  {sid}  {sess_state.workspace_root or sess_state.primary_workspace}{marker}")
                continue
            try:
                sessions_list = await client.list_sessions()
            except (AuthRequiredError, RemoteAPIError) as e:
                print_error(console, str(e))
                continue
            for sess in sessions_list:
                marker = " *" if sess.get("id") == workspace.session_id else ""
                console.print(f"  {sess.get('id')}  {sess.get('status')}  {sess.get('working_directory') or ''}{marker}")
            continue
        if text == "/delegate" or text.startswith("/delegate "):
            if not config.enable_subagent_delegation:
                print_error(
                    console,
                    "Subagent delegation is disabled. Enable it with enable_subagent_delegation = true "
                    "in config.toml, or TAMFIS_CODE_ENABLE_SUBAGENT_DELEGATION=1.",
                )
                continue
            arg = text[len("/delegate"):].strip()
            descriptions = [part.strip() for part in arg.split("|") if part.strip()]
            if not descriptions:
                print_error(console, "Usage: /delegate <objective 1> | <objective 2> | ...")
                continue
            # Delegation always runs through the standalone local loop
            # (agents.py's DelegatedCodingAgent) regardless of whether this
            # REPL session itself is standalone or --remote -- there's only
            # one implementation now (it was fully converted, not dual-mode).
            from .agents import AgentManager
            manager = AgentManager()
            delegate_manager = provider_manager
            delegate_provider = provider_type
            if delegate_manager is None:
                from .local_chat import resolve_provider_type as _resolve_provider_type
                from .providers import ProviderManager as _ProviderManager

                delegate_manager = _ProviderManager()
                delegate_provider = _resolve_provider_type("auto")
            results = await manager.execute_tasks(
                descriptions, manager=delegate_manager, provider=delegate_provider, model=model,
                console=console, workspace_root=workspace.workspace_root,
                approval_policy=config.approval_policy,
            )
            for r in results:
                marker = "✅" if r["status"] == "completed" else "❌"
                console.print(f"{marker} {r['description']}")
                summary = (r.get("result") or {}).get("summary") or (r.get("result") or {}).get("error")
                if summary:
                    console.print(f"   {summary}")
            continue
        if text == "/diffs" or text.startswith("/diffs "):
            arg = text[len("/diffs"):].strip()
            try:
                limit = int(arg) if arg else 10
            except ValueError:
                print_error(console, f"'{arg}' is not a valid number.")
                continue
            if standalone:
                mutations = local_state.get_session_state(workspace.session_id).modified_files[-limit:]
                mutations = list(reversed(mutations))
            else:
                try:
                    result = await client.list_file_mutations(workspace.session_id, limit=limit)
                except (AuthRequiredError, RemoteAPIError) as e:
                    print_error(console, str(e))
                    continue
                mutations = result.get("mutations") or []
            if not mutations:
                console.print("[dim]No file mutations recorded yet in this session.[/dim]")
            for m in mutations:
                mutation_id = m.get("mutation_id") if standalone else m.get("id")
                status = m.get("revert_status")
                marker = " [reverted]" if status == "reverted" else (" [revert failed]" if status == "revert_failed" else "")
                console.print(
                    f"  {mutation_id}  {m.get('operation')}  {m.get('path')}  "
                    f"+{m.get('lines_added')}/-{m.get('lines_removed')}{marker}"
                )
            continue
        if text == "/diff" or text.startswith("/diff "):
            mutation_id = text[len("/diff"):].strip()
            if standalone:
                mutations = local_state.get_session_state(workspace.session_id).modified_files
                selected = next((item for item in mutations if item.get("mutation_id") == mutation_id), None) if mutation_id else (mutations[-1] if mutations else None)
                if selected is None:
                    print_error(console, "Mutation not found." if mutation_id else "No file mutations recorded yet.")
                else:
                    print_unified_diff(console, str(selected.get("unified_diff") or ""), title=str(selected.get("path") or "Changes"))
                continue
            try:
                result = await client.list_file_mutations(workspace.session_id, limit=200 if mutation_id else 1)
            except (AuthRequiredError, RemoteAPIError) as e:
                print_error(console, str(e))
                continue
            mutations = result.get("mutations") or []
            selected = next((item for item in mutations if str(item.get("id")) == mutation_id), None) if mutation_id else (mutations[0] if mutations else None)
            if selected is None:
                print_error(console, "Mutation not found." if mutation_id else "No file mutations recorded yet.")
            else:
                print_unified_diff(console, str(selected.get("unified_diff") or ""), title=str(selected.get("path") or "Changes"))
            continue
        if text == "/revert" or text.startswith("/revert "):
            arg = text[len("/revert"):].strip()
            if not arg:
                print_error(console, "Usage: /revert <mutation_id> -- see /diffs for recent mutation ids.")
                continue
            if standalone:
                try:
                    result = local_revert_mutation(workspace.session_id, arg)
                except ValueError as e:
                    print_error(console, str(e))
                    continue
                console.print(f"[green]Reverted[/green] {result.get('path')}")
                continue
            try:
                result = await client.revert_file_mutation(workspace.session_id, arg)
            except (AuthRequiredError, RemoteAPIError) as e:
                print_error(console, str(e))
                continue
            console.print(f"[green]Reverted[/green] {result.get('path')}")
            continue
        if text == "/clear":
            console.clear()
            continue
        if text == "/resume" or text.startswith("/resume "):
            arg = text[len("/resume"):].strip()
            if standalone:
                known = local_state.all_known_session_ids()
                if arg:
                    try:
                        target_id = int(arg)
                    except ValueError:
                        print_error(console, f"'{arg}' is not a valid session id.")
                        continue
                    if target_id not in known:
                        print_error(console, f"No known local session {target_id}. Use /agents to list known sessions.")
                        continue
                else:
                    candidates = [sid for sid in reversed(known) if sid != workspace.session_id]
                    if not candidates:
                        console.print("[dim]No other sessions to resume.[/dim]")
                        continue
                    target_id = candidates[0]
                target_state = local_state.get_session_state(target_id)
                workspace = WorkspaceContext(
                    session_id=target_id,
                    workspace_root=target_state.workspace_root or target_state.primary_workspace,
                )
                console.print(f"[green]Resumed session {workspace.session_id}[/green]  workspace_root={workspace.workspace_root}")
                if target_state.conversation_summary:
                    console.print(f"[dim]{target_state.conversation_summary[-1000:]}[/dim]")
                continue
            try:
                if arg:
                    try:
                        target_id = int(arg)
                    except ValueError:
                        print_error(console, f"'{arg}' is not a valid session id.")
                        continue
                    workspace = await context_from_session(client, target_id)
                else:
                    target = await find_resumable_session(client, exclude_session_id=workspace.session_id)
                    if target is None:
                        console.print("[dim]No other sessions to resume.[/dim]")
                        continue
                    workspace = await context_from_session(client, target["id"])
            except (AuthRequiredError, RemoteAPIError) as e:
                print_error(console, str(e))
                continue
            console.print(f"[green]Resumed session {workspace.session_id}[/green]  workspace_root={workspace.workspace_root}")
            try:
                thread = await client.get_thread(workspace.session_id)
                print_recent_thread(console, thread.get("messages") or [])
            except (AuthRequiredError, RemoteAPIError):
                pass
            continue
        if text == "/retry" or text.startswith("/retry "):
            if standalone:
                if last_turn is None:
                    console.print("[dim]No previous turn in this session to retry.[/dim]")
                    continue
                objective, mode = last_turn
                repo_state = local_state.get_session_state(workspace.session_id).repository_context
                if mode in {"coding", "agent", "execute"} and blocking_dirty_files(repo_state.get("dirty_files") or []):
                    print_error(console, "Existing uncommitted changes detected; retry is blocked to preserve user edits.")
                    continue
                renderer = StreamRenderer(console)
                outcome = await run_local_agent_turn(
                    provider_manager, provider_type, model, [{"role": "user", "content": objective}],
                    console, renderer,
                    workspace_root=workspace.workspace_root, session_id=workspace.session_id,
                    approval_policy=config.approval_policy, interactive=True,
                    read_only=mode in {"chat", "audit", "plan"},
                )
                renderer.finish()
                if outcome.status == "completed" and outcome.summary and not renderer.streamed_final_text:
                    console.print(Markdown(outcome.summary))
                if outcome.status != "completed":
                    print_error(console, outcome.error or f"task {outcome.status}")
                console.print()
                continue
            arg = text[len("/retry"):].strip()
            try:
                if arg:
                    task_id = arg
                else:
                    failed = await find_recent_task(client, workspace.session_id, only_status={"failed", "cancelled"})
                    if failed is None:
                        console.print("[dim]No recent failed task to retry.[/dim]")
                        continue
                    task_id = failed["id"]
                renderer = StreamRenderer(console)
                outcome = await retry_task_and_stream(
                    client, renderer, console,
                    session_id=workspace.session_id, task_id=task_id, mode=None,
                    approval_policy=config.approval_policy, interactive=True,
                )
                if outcome.status == "completed" and outcome.summary and not renderer.streamed_final_text:
                    console.print(Markdown(outcome.summary))
                if outcome.status != "completed":
                    print_error(console, outcome.error or f"task {outcome.status}")
            except (AuthRequiredError, RemoteAPIError) as e:
                print_error(console, str(e))
            console.print()
            continue

        # Short replies are meaningful in a stateful coding conversation.
        # Expand them only when this session has prior task context so the
        # server/model receives the intended reference instead of a bare
        # token (and never reject them merely for being short).
        reply_state = local_state.get_session_state(workspace.session_id)
        text = contextualize_short_reply(
            text,
            has_context=bool(reply_state.last_task_id or reply_state.conversation_summary or reply_state.active_plan_id),
        )
        intent = parse_intent(text)
        try:
            if intent.kind == "shell":
                if not intent.command:
                    continue
                if standalone:
                    outcome = await run_local_shell_command(
                        console, workspace_root=workspace.workspace_root, session_id=workspace.session_id,
                        command=intent.command, approval_policy=config.approval_policy, interactive=True,
                    )
                else:
                    outcome = await run_shell_command(
                        client, console,
                        session_id=workspace.session_id, command=intent.command,
                        approval_policy=config.approval_policy, interactive=True,
                    )
            elif intent.kind == "saved_plan":
                plan = local_state.get_plan(workspace.session_id, intent.command or None)
                if plan is None:
                    print_error(console, "Plan not found. Use /plans to list saved plans.")
                    continue
                repo_state = local_state.get_session_state(workspace.session_id).repository_context
                if blocking_dirty_files(repo_state.get("dirty_files") or []):
                    print_error(console, "Existing uncommitted changes detected; plan execution is blocked to preserve user edits.")
                    continue
                plan_id = str(plan["id"])
                local_state.update_plan(workspace.session_id, plan_id, status="executing")
                renderer = StreamRenderer(console)
                model_state = local_state.get_session_state(workspace.session_id)
                plan_objective = local_state.plan_execution_objective(plan)
                if standalone:
                    outcome = await run_local_agent_turn(
                        provider_manager, provider_type, model, [{"role": "user", "content": plan_objective}],
                        console, renderer,
                        workspace_root=workspace.workspace_root, session_id=workspace.session_id,
                        approval_policy=config.approval_policy, interactive=True,
                    )
                    last_turn = (plan_objective, "execute")
                else:
                    outcome = await run_ai_task_and_stream(
                        client, renderer, console,
                        session_id=workspace.session_id,
                        objective=plan_objective, mode="execute",
                        approval_policy=config.approval_policy, interactive=True,
                        model=model_state.selected_model, provider=model_state.selected_provider,
                    )
                if standalone:
                    renderer.finish()
                local_state.update_plan(
                    workspace.session_id, plan_id,
                    status="completed" if outcome.status == "completed" else "failed",
                    execution_task_id=local_state.get_session_state(workspace.session_id).last_task_id,
                )
                if outcome.status == "completed" and outcome.summary and not renderer.streamed_final_text:
                    console.print(Markdown(outcome.summary))
            else:
                if not intent.objective:
                    continue
                if intent.mode == "execute":
                    repo_state = local_state.get_session_state(workspace.session_id).repository_context
                    if blocking_dirty_files(repo_state.get("dirty_files") or []):
                        print_error(console, "Existing uncommitted changes detected; execute mode is blocked to preserve user edits. Use /audit or /plan, or clean the worktree yourself.")
                        continue
                renderer = StreamRenderer(console, objective=intent.objective)
                model_state = local_state.get_session_state(workspace.session_id)
                if standalone:
                    outcome = await run_local_agent_turn(
                        provider_manager, provider_type, model, [{"role": "user", "content": intent.objective}],
                        console, renderer,
                        workspace_root=workspace.workspace_root, session_id=workspace.session_id,
                        approval_policy=config.approval_policy, interactive=True,
                        read_only=intent.mode in {"chat", "audit", "plan"},
                    )
                    renderer.finish()
                    last_turn = (intent.objective, intent.mode)
                    if intent.mode == "plan" and outcome.status == "completed" and outcome.summary:
                        saved = local_state.save_plan(workspace.session_id, objective=intent.objective, content=outcome.summary)
                        console.print(f"[green]Plan saved[/green] · {saved.id} · run /execute-plan {saved.id}")
                else:
                    outcome = await run_ai_task_and_stream(
                        client, renderer, console,
                        session_id=workspace.session_id, objective=intent.objective, mode=intent.mode,
                        approval_policy=config.approval_policy, interactive=True,
                        model=model_state.selected_model,
                        provider=model_state.selected_provider,
                    )
                if outcome.status == "completed" and outcome.summary and not renderer.streamed_final_text:
                    console.print(Markdown(outcome.summary))
        except AuthRequiredError:
            if queued_item:
                local_state.update_instruction(workspace.session_id, str(queued_item.get("id")), "failed")
            print_error(console, "Not authenticated -- run `tamfis-code login` in another terminal, then retry.")
            continue
        except RemoteAPIError as e:
            if queued_item:
                local_state.update_instruction(workspace.session_id, str(queued_item.get("id")), "failed")
            print_error(console, str(e))
            continue
        except Exception as e:
            if queued_item:
                local_state.update_instruction(workspace.session_id, str(queued_item.get("id")), "failed")
            print_error(console, str(e))
            continue

        if queued_item:
            local_state.update_instruction(workspace.session_id, str(queued_item.get("id")), "completed")

        if outcome.status not in ("completed",):
            print_error(console, outcome.error or f"task {outcome.status}")
        console.print()
