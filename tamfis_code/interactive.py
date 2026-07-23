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
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from . import state as local_state
from .api_client import AuthRequiredError, RemoteAPIClient, RemoteAPIError
from .clipboard import copy_to_clipboard
from .config import (
    APPROVAL_MODES,
    CONFIG_DIR,
    Config,
    MODE_ALIASES,
    mode_label_for_policy,
    next_mode_in_cycle,
)
from .agent_definitions import PROJECT_AGENTS_RELATIVE as AGENT_DEFINITIONS_PROJECT_RELATIVE
from .agent_definitions import USER_AGENTS_DIR as AGENT_DEFINITIONS_USER_DIR
from .custom_commands import (
    PROJECT_COMMANDS_RELATIVE,
    USER_COMMANDS_DIR,
    CustomCommand,
    expand_custom_command,
    load_custom_commands,
)
from .doctor import run_doctor
from .live_input import LiveInputListener
from .render import StreamRenderer, print_banner, print_error, print_recent_thread, print_resume_plan_status, print_unified_diff
from .runner import resolve_approval_decision_async, retry_task_and_stream, run_ai_task_and_stream, run_shell_command
from .runner_local import run_local_agent_turn, run_local_shell_command
from .pty import LocalPtyBroker
from .safety import revert_mutation as local_revert_mutation
from .safety import revert_transaction as local_revert_transaction
from .tasks import find_recent_task
from .workspace import (
    WorkspaceContext, blocking_dirty_files, context_from_session, discover_local_repository,
    find_resumable_session, resolve_local_workspace,
)

# A pasted block strictly longer than this many lines is collapsed to a
# placeholder in the input line (Claude Code/Codex-style) instead of
# dumping the whole thing inline and scrolling the terminal -- a short
# paste of a few lines stays visible and directly editable, matching how
# those tools only collapse genuinely large pastes.
PASTE_COLLAPSE_LINE_THRESHOLD = 3

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
/copy                copy the last assistant response to the clipboard (OSC 52 --
                      works over plain SSH, no X11/Wayland/xclip required)
/doctor              run connectivity/auth checks
/resume [session_id]  switch to another session (most recent if omitted)
/retry [task_id]      retry a failed task (most recent failure if omitted)
/agents              list sessions and their latest task status
/delegate <a> | <b>  run objectives a, b, ... as concurrent delegated sub-tasks
                      (requires enable_subagent_delegation = true in config.toml,
                      or TAMFIS_CODE_ENABLE_SUBAGENT_DELEGATION=1)
/swarm <a> | <b>     like /delegate, but with a live aggregate status display and
                      a higher default concurrency; read-only by default -- add
                      --mutate to allow file edits (requires an auto-approving
                      mode: auto, accept-edits, etc., since sub-tasks can't prompt)
/diffs [n]           show the last n file mutations in this session (default 10)
/diff [mutation_id]  show a semantic unified diff (latest if omitted)
/revert <mutation_id> restore a file to its content before that mutation (or
                      delete it, if that mutation created the file)
/revert <turn_id>     revert every mutation from one turn together (the
                      "turn_..." id shown alongside each mutation in /diffs)
/detach              exit without cancelling anything server-side (same as /exit --
                      nothing in this REPL ties a task's lifetime to this process; see
                      `tamfis-code attach <session_id>` in another terminal to reconnect)
/clear               clear the screen
/compact             save a durable checkpoint of the current task context
/permissions         show approval policy and immutable server safeguards
/mode                show the active approval mode and available modes
/mode <name>         switch mode: manual | accept-edits | auto | plan
Shift+Tab            cycle mode without typing a command (shown in the prompt as [mode]);
                     also works while a task is already running, not just at this prompt
message>             while a task is running: type a message and press Enter
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


# Single source of truth for slash-command tab-completion. Names here
# mirror the literal strings the dispatch checks further down this file
# actually match against (`text == "/xxx"` / `text.startswith("/xxx ")`) --
# kept as a plain tuple rather than parsed out of HELP_TEXT above, since
# HELP_TEXT is free-form prose (multi-line descriptions, indentation,
# argument placeholders) that isn't reliably machine-parseable.
SLASH_COMMANDS: tuple[tuple[str, str], ...] = (
    ("/help", "show this help"),
    ("/status", "show session/workspace/approval status"),
    ("/context", "show cached repository/task context"),
    ("/reports", "show the repository report index"),
    ("/queue", "show or append queued instructions"),
    ("/cwd", "show the current workspace root"),
    ("/copy", "copy the last assistant response to the clipboard"),
    ("/doctor", "run connectivity/auth checks"),
    ("/resume", "switch to another session"),
    ("/retry", "retry a failed task"),
    ("/agents", "list sessions and their latest task status"),
    ("/delegate", "run objectives as concurrent delegated sub-tasks"),
    ("/swarm", "fan out independent sub-tasks concurrently (opt-in, safe-by-default read-only)"),
    ("/diffs", "show recent file mutations in this session"),
    ("/diff", "show a semantic unified diff"),
    ("/revert", "restore a file to its content before a mutation"),
    ("/detach", "exit without cancelling anything server-side"),
    ("/clear", "clear the screen"),
    ("/compact", "save a durable checkpoint of the current task context"),
    ("/permissions", "show approval policy and immutable server safeguards"),
    ("/mode", "show or switch the active approval mode"),
    ("/model", "show or switch the active model route"),
    ("/tools", "show the tools exposed to tamfis-code tasks"),
    ("/commands", "list user-defined custom slash commands loaded from .md files"),
    ("/agent-types", "list declarative subagent types available to /delegate and /swarm"),
    ("/pty", "manage a persistent background terminal"),
    ("/exit", "quit"),
    ("/quit", "quit"),
    ("/run", "explicit shell command"),
    ("/shell", "explicit shell command"),
    ("/chat", "conversational/read-only coding assistance"),
    ("/audit", "AI audit mode (read-only)"),
    ("/plan", "create or show a saved executable plan"),
    ("/plans", "list saved plans for this session"),
    ("/execute-plan", "execute a saved plan"),
    ("/agent", "full coding-agent mode (inspect + edit + verify)"),
    ("/execute", "AI execute mode (tools + approval policy)"),
)


class _SlashCommandCompleter(Completer):
    """Tab-completion for slash-commands only. Deliberately inert for
    natural-language input -- offers nothing once the line doesn't start
    with "/", or once a command name is already complete and a space
    follows -- so it never gets in the way of typing an ordinary
    objective, which is most of what gets typed into this REPL.

    `custom_commands`, if given, is the SAME dict object the REPL loop
    refreshes every turn via load_custom_commands (mutated in place, not
    replaced) -- this completer always reflects the latest set without
    needing to be reconstructed when a command file changes mid-session.
    """

    def __init__(self, custom_commands: Optional[dict[str, CustomCommand]] = None) -> None:
        self._custom_commands = custom_commands if custom_commands is not None else {}

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/") or " " in text:
            return
        for name, description in SLASH_COMMANDS:
            if name.startswith(text):
                yield Completion(name, start_position=-len(text), display_meta=description)
        built_in_names = {name for name, _ in SLASH_COMMANDS}
        for name, command in self._custom_commands.items():
            full = f"/{name}"
            if full in built_in_names or not full.startswith(text):
                continue
            yield Completion(full, start_position=-len(text), display_meta=command.description)


class Intent:
    def __init__(self, kind: str, *, command: str = "", objective: str = "", mode: str = "coding"):
        self.kind = kind
        self.command = command
        self.objective = objective
        self.mode = mode


# Bounds standalone in-session conversation history (see run_interactive's
# conversation_history) -- kept generous since runner_local.py's own
# compaction/rollover already handles the actual provider token budget;
# this just stops one very long-lived REPL process from growing the list
# forever in memory.
MAX_STANDALONE_HISTORY_TURNS = 30


def _append_turn_to_history(
    history: list[dict[str, str]], *, objective: str, answer: Optional[str],
) -> None:
    """Record one completed standalone turn for the NEXT turn's context.

    Only a completed turn with real answer text is worth remembering --
    a failed/denied/cancelled turn's objective is still recorded (so a
    follow-up like "try again but skip the tests" has something to refer
    to), but with no assistant turn to pair it with.
    """
    history.append({"role": "user", "content": objective})
    if answer:
        history.append({"role": "assistant", "content": answer})
    del history[: max(0, len(history) - MAX_STANDALONE_HISTORY_TURNS * 2)]


def paste_placeholder(data: str, count: int) -> Optional[tuple[str, str]]:
    """For one bracketed-paste event's raw data: normalize line endings
    (some terminals, e.g. iTerm2, paste \\r\\n -- matches prompt_toolkit's
    own default Keys.BracketedPaste handling), then decide whether it's
    long enough to collapse to a placeholder rather than insert verbatim.

    Returns None for a short paste (insert `data` normally, unchanged
    behaviour) or `(placeholder, normalized_text)` for a long one -- the
    caller inserts `placeholder` into the buffer and remembers
    `normalized_text` under that key so the real content can be
    substituted back in once the line is submitted.
    """
    normalized = data.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized:
        return None
    line_count = normalized.count("\n") + (0 if normalized.endswith("\n") else 1)
    if line_count <= PASTE_COLLAPSE_LINE_THRESHOLD:
        return None
    placeholder = f"[Pasted text #{count} +{line_count} lines]"
    return placeholder, normalized


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


def parse_intent(raw: str, custom_commands: Optional[dict[str, CustomCommand]] = None) -> Intent:
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
    # User-defined custom commands (custom_commands.py) -- checked last, so
    # every built-in slash command above always wins on a name collision.
    # Only a real "/<name>" shape is eligible (not a bare natural-language
    # objective that happens to start with a slash character).
    if custom_commands and text.startswith("/"):
        name, _, arguments = text[1:].partition(" ")
        command = custom_commands.get(name)
        if command is not None:
            return Intent("ai", objective=expand_custom_command(command, arguments.strip()), mode="coding")
    return Intent("ai", objective=text, mode="coding")


async def run_interactive(
    client: Optional[RemoteAPIClient], config: Config, workspace: WorkspaceContext,
    *, provider: str = "auto", model: Optional[str] = None,
) -> None:
    """Drives the interactive REPL. `client=None` means standalone mode: no
    TamfisGPT Remote Workspace backend, no login -- AI turns/shell commands
    run through runner_local.py's local agent loop calling `provider`
    directly instead. A handful of features that are inherently remote-server
    concepts (such as the remote model catalog) remain remote-only, while
    background PTY terminals, diffs/revert, resume, agents listing, retry,
    delegate, and doctor all have local implementations in this mode.
    """
    standalone = client is None
    provider_manager = None
    provider_type = None
    last_turn: Optional[tuple[str, str]] = None  # (objective, mode) -- standalone /retry target
    last_response_text: Optional[str] = None  # most recent completed answer -- /copy target
    # Standalone-only: run_local_agent_turn's `messages` param used to be
    # rebuilt as a single fresh `[{"role": "user", "content": objective}]`
    # on every turn -- confirmed live, this made the interactive session
    # provider-side amnesiac. "yes" got expanded by contextualize_short_reply
    # into "Yes. Proceed with the action or next step you just proposed."
    # and sent as the ENTIRE conversation, with no prior turn attached at
    # all -- the model had no idea what it had just proposed. Bounded to
    # the last MAX_STANDALONE_HISTORY_TURNS turns (not unbounded -- a very
    # long session shouldn't grow this list forever); oversized individual
    # messages within it are still handled by runner_local.py's existing
    # compaction (including old user-turn compaction) and rollover.
    saved_history = local_state.get_session_state(workspace.session_id).conversation_history if standalone else []
    conversation_history: list[dict[str, str]] = [
        {"role": str(item.get("role") or ""), "content": str(item.get("content") or "")}
        for item in saved_history
        if item.get("role") in {"user", "assistant"} and item.get("content")
    ][-MAX_STANDALONE_HISTORY_TURNS * 2:]

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
    console.print(
        "[dim]Type /help for commands. Paste up to 1,000,000 characters; Alt+Enter adds a newline. "
        "While a task runs, a normal message prompt remains available: type and press Enter; "
        "your text is queued without a special shortcut. Shift+Tab cycles mode. Ctrl+D or Ctrl+C exits.[/dim]\n"
    )

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    history_path = CONFIG_DIR / "history"
    bindings = KeyBindings()
    # Local only: which byte offset this REPL has already displayed per PTY
    # id, so /pty read shows only new output. The server (RemotePtySession/
    # pty_broker's or LocalPtyBroker's ring buffer is the source of truth;
    # losing this dict only means the next read re-shows buffered output.
    pty_offsets: dict[str, int] = {}
    local_pty = LocalPtyBroker(cwd=workspace.workspace_root) if standalone else None

    @bindings.add("enter")
    def _submit(event) -> None:
        event.current_buffer.validate_and_handle()

    @bindings.add("escape", "enter")
    def _newline(event) -> None:
        event.current_buffer.insert_text("\n")

    @bindings.add("s-tab")
    def _cycle_mode(event) -> None:
        # Claude-Code-style quick mode switch: manual -> accept-edits ->
        # auto -> plan -> manual, without leaving the input line or having
        # to know the /mode command exists. The prompt's mode indicator
        # (see _prompt_message below) re-renders immediately because
        # prompt_toolkit re-evaluates a callable `message=` on invalidate.
        config.approval_policy = next_mode_in_cycle(config.approval_policy)
        event.app.invalidate()

    # Reported live: pasting a long block (clipboard paste, terminal
    # bracketed-paste mode) inserted the entire raw text into the input
    # line -- unlike Claude Code/Codex, which collapse a large paste to a
    # short placeholder (e.g. "[Pasted text #1 +86 lines]") while editing,
    # substituting the real text back in only once the line is submitted.
    # prompt_toolkit's own default Keys.BracketedPaste binding
    # (key_binding/bindings/basic.py) just inserts the raw data verbatim --
    # there was no collapsing logic anywhere in this codebase to begin
    # with, not a regression. `pending_pastes` maps each placeholder to its
    # real text for the CURRENT line only; reset before every prompt below
    # and consumed right after it returns, so placeholders never leak
    # across turns or get treated as literal user-typed text if the
    # objective happens to echo one back.
    pending_pastes: dict[str, str] = {}
    paste_counter = 0

    @bindings.add(Keys.BracketedPaste)
    def _collapse_large_paste(event) -> None:
        nonlocal paste_counter
        collapsed = paste_placeholder(event.data, paste_counter + 1)
        if collapsed is None:
            event.current_buffer.insert_text(event.data.replace("\r\n", "\n").replace("\r", "\n"))
            return
        placeholder, normalized = collapsed
        paste_counter += 1
        pending_pastes[placeholder] = normalized
        event.current_buffer.insert_text(placeholder)

    def _prompt_message() -> HTML:
        return HTML(f"tamfis-code <ansicyan>[{mode_label_for_policy(config.approval_policy)}]</ansicyan>> ")

    # multiline=True is essential for bracketed terminal paste: embedded
    # newlines remain part of one objective instead of submitting the first
    # line and treating the remainder as accidental follow-up commands.  The
    # bindings retain familiar Enter-to-submit behaviour.
    # Mutated in place (not reassigned) every loop iteration below, so the
    # completer -- constructed once, here -- always reflects the latest
    # command files without needing to be rebuilt.
    custom_commands: dict[str, CustomCommand] = load_custom_commands(workspace.workspace_root)
    session: PromptSession = PromptSession(
        history=FileHistory(str(history_path)), multiline=True, key_bindings=bindings,
        completer=_SlashCommandCompleter(custom_commands), complete_while_typing=True,
    )

    async def _run_saved_plan(plan_id_arg: Optional[str]) -> bool:
        """Execute a saved plan by id (or the most recent if plan_id_arg is
        falsy) -- shared by /execute-plan and the plan-mode "execute now?"
        gate below, so there is exactly one real execution path, not two
        that could silently drift apart. Returns whether a plan was found
        and actually run."""
        nonlocal last_turn, last_response_text
        plan = local_state.get_plan(workspace.session_id, plan_id_arg or None)
        if plan is None:
            print_error(console, "Plan not found. Use /plans to list saved plans.")
            return False
        repo_state = local_state.get_session_state(workspace.session_id).repository_context
        if blocking_dirty_files(repo_state.get("dirty_files") or []):
            print_error(console, "Existing uncommitted changes detected; plan execution is blocked to preserve user edits.")
            return False
        plan_id = str(plan["id"])
        local_state.update_plan(workspace.session_id, plan_id, status="executing")
        renderer = StreamRenderer(console, mode_label=mode_label_for_policy(config.approval_policy))
        model_state = local_state.get_session_state(workspace.session_id)
        plan_objective = local_state.plan_execution_objective(plan)
        if standalone:
            live_input = LiveInputListener(session_id=workspace.session_id, renderer=renderer, cli_config=config)
            live_input.start()
            try:
                outcome = await run_local_agent_turn(
                    provider_manager, provider_type, model,
                    [*conversation_history, {"role": "user", "content": plan_objective}],
                    console, renderer,
                    workspace_root=workspace.workspace_root, session_id=workspace.session_id,
                    approval_policy=config.approval_policy, interactive=True, cli_config=config,
                    allow_swarm_tool=True,
                )
            finally:
                live_input.stop()
            _append_turn_to_history(
                conversation_history, objective=plan_objective,
                answer=outcome.summary if outcome.status == "completed" else None,
            )
            last_turn = (plan_objective, "execute")
        else:
            outcome = await run_ai_task_and_stream(
                client, renderer, console,
                session_id=workspace.session_id,
                objective=plan_objective, mode="execute",
                approval_policy=config.approval_policy, interactive=True,
                model=model_state.selected_model, provider=model_state.selected_provider,
                config=config,
            )
        if standalone:
            renderer.finish()
        local_state.update_plan(
            workspace.session_id, plan_id,
            status="completed" if outcome.status == "completed" else "failed",
            execution_task_id=local_state.get_session_state(workspace.session_id).last_task_id,
        )
        if outcome.status == "completed" and outcome.summary:
            last_response_text = outcome.summary
        if outcome.status == "completed" and outcome.summary and not renderer.streamed_final_text:
            console.print(Markdown(outcome.summary))
        return True

    while True:
        fresh_custom_commands = load_custom_commands(workspace.workspace_root)
        custom_commands.clear()
        custom_commands.update(fresh_custom_commands)
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
            pending_pastes.clear()
            paste_counter = 0
            try:
                text = await session.prompt_async(_prompt_message)
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
            for placeholder, real_text in pending_pastes.items():
                text = text.replace(placeholder, real_text)

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
            if local_pty is not None:
                local_pty.close()
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
                    "[dim]Standalone mode: /model list needs the remote model catalog; /pty, diffs/revert, resume, "
                    "agents, retry, delegate, doctor) runs fully locally, no TamfisGPT backend involved.[/dim]"
                )
            continue
        if text == "/cwd":
            console.print(workspace.workspace_root)
            continue
        if text == "/copy":
            if not last_response_text:
                console.print("[dim]Nothing to copy yet.[/dim]")
            elif copy_to_clipboard(console, last_response_text):
                console.print(f"[dim]Copied {len(last_response_text):,} characters to clipboard.[/dim]")
            else:
                console.print("[dim]Can't copy: output isn't attached to a terminal.[/dim]")
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
                    console.print(table if shown else "[dim]Unknown provider. Use hf, nvidia, or openrouter.[/dim]")
                    continue
                if parts[0].lower() == "auto":
                    provider_type = ProviderType.AUTO
                    local_state.save_session_state(workspace.session_id, selected_model="auto", selected_provider=None)
                    console.print("[green]Provider routing set to automatic.[/green]")
                    continue
                try:
                    provider_type = _resolve_provider_type(parts[0])
                except ValueError as exc:
                    print_error(console, f"{exc} Usage: /model auto | /model <tamfis|hf|nvidia|openrouter> [model-id]")
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
                table.add_row("web_search", "Tavily (if TAVILY_API_KEY set) or DuckDuckGo fallback", "Read-only; self-contained, no monorepo required")
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
        if text == "/commands":
            if not custom_commands:
                console.print(
                    "[dim]No custom commands found. Add one at "
                    f"{USER_COMMANDS_DIR / '<name>.md'} (every session) or "
                    f"{Path(workspace.workspace_root) / PROJECT_COMMANDS_RELATIVE / '<name>.md'} "
                    "(this project only) -- see README.md's Custom commands section.[/dim]"
                )
                continue
            table = Table(show_header=True, header_style="bold")
            table.add_column("Command")
            table.add_column("Description")
            table.add_column("Source")
            for name, command in sorted(custom_commands.items()):
                table.add_row(f"/{name}", command.description, command.source)
            console.print(table)
            continue
        if text == "/agent-types":
            from .agent_definitions import load_agent_definitions
            definitions = load_agent_definitions(workspace.workspace_root)
            if not definitions:
                console.print(
                    "[dim]No declarative subagent types found. Add one at "
                    f"{AGENT_DEFINITIONS_USER_DIR / '<name>.md'} (every session) or "
                    f"{Path(workspace.workspace_root) / AGENT_DEFINITIONS_PROJECT_RELATIVE / '<name>.md'} "
                    "(this project only) -- see README.md's Declarative subagent types section. "
                    "Use with: /delegate --agent <name> ... or /swarm --agent <name> ....[/dim]"
                )
                continue
            table = Table(show_header=True, header_style="bold")
            table.add_column("Agent type")
            table.add_column("Description")
            table.add_column("Model")
            table.add_column("Provider")
            table.add_column("Source")
            for name, definition in sorted(definitions.items()):
                table.add_row(
                    name, definition.description, definition.model or "(shared)",
                    definition.provider or "(shared)", definition.source,
                )
            console.print(table)
            console.print("[dim]Use with: /delegate --agent <name> ... or /swarm --agent <name> ....[/dim]")
            continue
        if text == "/pty" or text.startswith("/pty "):
            arg = text[len("/pty"):].strip()
            parts = arg.split(maxsplit=1)
            sub = parts[0].lower() if parts else "list"
            rest = parts[1] if len(parts) > 1 else ""

            if sub == "start":
                shell_command = rest.strip() or "bash"
                if standalone:
                    try:
                        local_session = local_pty.start(shell_command)  # type: ignore[union-attr]
                    except (OSError, ValueError) as exc:
                        print_error(console, f"Could not start terminal: {exc}")
                        continue
                    pty_offsets[local_session.id] = 0
                    console.print(f"[green]Started local background terminal[/green] {local_session.id[:8]}  pid={local_session.pid}")
                    console.print(f"[dim]/pty send {local_session.id[:8]} <text>   /pty read {local_session.id[:8]}   /pty kill {local_session.id[:8]}[/dim]")
                    continue
                try:
                    pty = await client.start_pty(workspace.session_id, shell_command=shell_command)
                    decision = await resolve_approval_decision_async(
                        console, f"[background terminal] {shell_command}", str(pty.get("safety_tier", "medium")),
                        config.approval_policy, interactive=True, config=config,
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
                if standalone:
                    sessions_list = list(local_pty.sessions.values())  # type: ignore[union-attr]
                    if not sessions_list:
                        console.print("[dim]No background terminals in this session. Use /pty start.[/dim]")
                        continue
                    table = Table(show_header=True, header_style="bold")
                    for column in ("ID", "STATUS", "COMMAND", "PID"):
                        table.add_column(column)
                    for item in sessions_list:
                        table.add_row(item.id[:8], item.status, item.command[:40], str(item.pid))
                    console.print(table)
                    continue
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
                    if standalone:
                        local_pty.write(target_id, payload)  # type: ignore[union-attr]
                    else:
                        await client.write_pty(target_id, payload)
                except (AuthRequiredError, RemoteAPIError) as e:
                    print_error(console, str(e))
                continue

            if sub == "read":
                try:
                    if standalone:
                        session, data, offset = local_pty.read(target_id, since=pty_offsets.get(target_id, 0))  # type: ignore[union-attr]
                        result = {"data": data, "offset": offset, "alive": session.status == "running", "status": session.status}
                    else:
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
                    if standalone:
                        session = local_pty.kill(target_id)  # type: ignore[union-attr]
                        result = {"status": session.status}
                    else:
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
                console.print(f"Current mode: [bold]{mode_label_for_policy(config.approval_policy)}[/bold] ({config.approval_policy})")
                console.print(
                    "[dim]/mode manual        prompt for every risky action (the default)\n"
                    "/mode accept-edits  auto-approve safe/medium-risk actions, still prompt for dangerous ones\n"
                    "/mode auto           auto-approves everything, never prompts (server safeguards still apply)\n"
                    "/mode plan           read-only: propose without executing\n"
                    "Shift+Tab            cycle manual -> accept-edits -> auto -> plan without typing a command\n"
                    "                     (also works mid-task, not just here); type in the message prompt to queue work\n"
                    "Other raw policy values also work directly (--approval-only, no short alias): safe, workspace,\n"
                    "read-only, plan-only, suggest, full-auto, and 'never' -- note 'never' means DENY everything\n"
                    "(the opposite of what it sounds like; it is not a synonym for 'auto').[/dim]"
                )
                continue
            resolved = MODE_ALIASES.get(arg, arg if arg in APPROVAL_MODES else None)
            if resolved is None:
                print_error(
                    console,
                    f"Unknown mode '{arg}'. Use manual, accept-edits, auto, or plan "
                    "(or a raw policy value: safe, workspace, read-only, plan-only, suggest, full-auto, never).",
                )
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
                    configured = "[green]yes[/green]" if info["api_key_set"] or name == "tier_iv" else "[dim]no[/dim]"
                    table.add_row(name, configured, info["key_preview"])
                console.print(table)
                console.print(
                    f"[dim]Currently selected: {provider_type.value if provider_type else provider}  "
                    f"· auto would pick: {status['default']}[/dim]"
                )
                continue
            await run_doctor(config, console, Path(workspace.workspace_root), session_id=workspace.session_id)
            continue
        if text == "/agents" or text == "/agents --all":
            show_all = text.endswith("--all")
            if standalone:
                for sid in local_state.all_known_session_ids():
                    sess_state = local_state.get_session_state(sid)
                    if sess_state.is_swarm_child and not show_all:
                        continue
                    marker = " *" if sid == workspace.session_id else ""
                    label = f"  (swarm: {sess_state.swarm_label})" if sess_state.is_swarm_child else ""
                    console.print(f"  {sid}  {sess_state.workspace_root or sess_state.primary_workspace}{marker}{label}")
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
            delegate_agent_type: Optional[str] = None
            if arg.startswith("--agent "):
                _rest = arg[len("--agent "):].lstrip()
                delegate_agent_type, _, arg = _rest.partition(" ")
                arg = arg.strip()
            descriptions = [part.strip() for part in arg.split("|") if part.strip()]
            if not descriptions:
                print_error(console, "Usage: /delegate [--agent <name>] <objective 1> | <objective 2> | ...")
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
                approval_policy=config.approval_policy, parent_session_id=workspace.session_id,
                agent_types=[delegate_agent_type] * len(descriptions) if delegate_agent_type else None,
            )
            for r in results:
                marker = "✅" if r["status"] == "completed" else "❌"
                console.print(f"{marker} {r['description']}")
                summary = (r.get("result") or {}).get("summary") or (r.get("result") or {}).get("error")
                if summary:
                    console.print(f"   {summary}")
            continue
        if text == "/swarm" or text.startswith("/swarm "):
            if not config.enable_subagent_delegation:
                print_error(
                    console,
                    "Subagent delegation is disabled. Enable it with enable_subagent_delegation = true "
                    "in config.toml, or TAMFIS_CODE_ENABLE_SUBAGENT_DELEGATION=1.",
                )
                continue
            arg = text[len("/swarm"):].strip()
            mutate = False
            for flag in ("--mutate", "--edit"):
                if arg.endswith(flag):
                    mutate = True
                    arg = arg[: -len(flag)].strip()
                    break
            swarm_agent_type: Optional[str] = None
            if arg.startswith("--agent "):
                _rest = arg[len("--agent "):].lstrip()
                swarm_agent_type, _, arg = _rest.partition(" ")
                arg = arg.strip()
            descriptions = [part.strip() for part in arg.split("|") if part.strip()]
            if not descriptions:
                print_error(console, "Usage: /swarm [--agent <name>] <objective 1> | <objective 2> | ... [--mutate]")
                continue
            swarm_manager = provider_manager
            swarm_provider = provider_type
            if swarm_manager is None:
                from .local_chat import resolve_provider_type as _resolve_provider_type
                from .providers import ProviderManager as _ProviderManager

                swarm_manager = _ProviderManager()
                swarm_provider = _resolve_provider_type("auto")
            from .swarm import run_swarm
            try:
                results = await run_swarm(
                    descriptions, manager=swarm_manager, provider=swarm_provider, model=model,
                    console=console, workspace_root=workspace.workspace_root,
                    session_id=workspace.session_id, approval_policy=config.approval_policy,
                    mutate=mutate,
                    agent_types=[swarm_agent_type] * len(descriptions) if swarm_agent_type else None,
                )
            except ValueError as e:
                print_error(console, str(e))
                continue
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
                turn_suffix = f"  (turn {m.get('transaction_id')})" if standalone and m.get("transaction_id") else ""
                console.print(
                    f"  {mutation_id}  {m.get('operation')}  {m.get('path')}  "
                    f"+{m.get('lines_added')}/-{m.get('lines_removed')}{marker}{turn_suffix}"
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
                print_error(console, "Usage: /revert <mutation_id | turn_id> -- see /diffs for recent mutation ids; a turn_... id reverts every mutation from that turn together.")
                continue
            if standalone:
                if arg.startswith("turn_"):
                    try:
                        result = local_revert_transaction(workspace.session_id, arg)
                    except ValueError as e:
                        print_error(console, str(e))
                        continue
                    console.print(f"[green]Reverted {len(result['reverted'])} mutation(s)[/green] from turn {arg}")
                    if result["error"]:
                        print_error(console, f"Stopped after a failure: {result['error']} -- still pending: {', '.join(result['remaining'])}")
                    continue
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
                print_resume_plan_status(console, target_state)
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
                renderer = StreamRenderer(console, mode_label=mode_label_for_policy(config.approval_policy))
                live_input = LiveInputListener(session_id=workspace.session_id, renderer=renderer, cli_config=config)
                live_input.start()
                try:
                    outcome = await run_local_agent_turn(
                        provider_manager, provider_type, model,
                        [*conversation_history, {"role": "user", "content": objective}],
                        console, renderer,
                        workspace_root=workspace.workspace_root, session_id=workspace.session_id,
                        approval_policy=config.approval_policy, interactive=True,
                        read_only=mode in {"chat", "audit", "plan"}, cli_config=config,
                        allow_swarm_tool=True,
                    )
                finally:
                    live_input.stop()
                renderer.finish()
                _append_turn_to_history(
                    conversation_history, objective=objective,
                    answer=outcome.summary if outcome.status == "completed" else None,
                )
                if outcome.status == "completed" and outcome.summary:
                    last_response_text = outcome.summary
                if outcome.status == "completed" and outcome.summary and not renderer.streamed_final_text:
                    console.print(Markdown(outcome.summary))
                if outcome.status == "exited":
                    if local_pty is not None:
                        local_pty.close()
                    break
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
                    approval_policy=config.approval_policy, interactive=True, config=config,
                )
                if outcome.status == "completed" and outcome.summary:
                    last_response_text = outcome.summary
                if outcome.status == "completed" and outcome.summary and not renderer.streamed_final_text:
                    console.print(Markdown(outcome.summary))
                if outcome.status == "exited":
                    if local_pty is not None:
                        local_pty.close()
                    break
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
            has_context=bool(
                reply_state.last_task_id or reply_state.conversation_summary
                or reply_state.active_plan_id or reply_state.active_task
                or reply_state.turn_checkpoint or reply_state.conversation_history
            ),
        )
        submitted_text = text
        intent = parse_intent(text, custom_commands=custom_commands)
        try:
            if intent.kind == "shell":
                if not intent.command:
                    continue
                if standalone:
                    outcome = await run_local_shell_command(
                        console, workspace_root=workspace.workspace_root, session_id=workspace.session_id,
                        command=intent.command, approval_policy=config.approval_policy, interactive=True,
                        config=config,
                    )
                else:
                    outcome = await run_shell_command(
                        client, console,
                        session_id=workspace.session_id, command=intent.command,
                        approval_policy=config.approval_policy, interactive=True, config=config,
                    )
            elif intent.kind == "saved_plan":
                await _run_saved_plan(intent.command)
            else:
                if not intent.objective:
                    continue
                if intent.mode == "execute":
                    repo_state = local_state.get_session_state(workspace.session_id).repository_context
                    if blocking_dirty_files(repo_state.get("dirty_files") or []):
                        print_error(console, "Existing uncommitted changes detected; execute mode is blocked to preserve user edits. Use /audit or /plan, or clean the worktree yourself.")
                        continue
                renderer = StreamRenderer(console, mode_label=mode_label_for_policy(config.approval_policy))
                model_state = local_state.get_session_state(workspace.session_id)
                renderer.handle_event({
                    "event_type": "user_message",
                    "payload": {"content": submitted_text},
                })
                if standalone:
                    live_input = LiveInputListener(session_id=workspace.session_id, renderer=renderer, cli_config=config)
                    live_input.start()
                    try:
                        outcome = await run_local_agent_turn(
                            provider_manager, provider_type, model,
                            [*conversation_history, {"role": "user", "content": intent.objective}],
                            console, renderer,
                            workspace_root=workspace.workspace_root, session_id=workspace.session_id,
                            approval_policy=config.approval_policy, interactive=True,
                            read_only=intent.mode in {"chat", "audit", "plan"}, cli_config=config,
                            allow_swarm_tool=True,
                        )
                    finally:
                        live_input.stop()
                    renderer.finish()
                    _append_turn_to_history(
                        conversation_history, objective=intent.objective,
                        answer=outcome.summary if outcome.status == "completed" else None,
                    )
                    last_turn = (intent.objective, intent.mode)
                    if intent.mode == "plan" and outcome.status == "completed" and outcome.summary:
                        saved = local_state.save_plan(workspace.session_id, objective=intent.objective, content=outcome.summary)
                        console.print(f"[green]Plan saved[/green] · {saved.id}")
                        # Real gated plan-mode UX (Claude Code's Plan Mode
                        # equivalent): the plan above was produced entirely
                        # read-only (mode="plan" -> read_only=True), and
                        # nothing has been executed yet -- this is the one
                        # explicit approval checkpoint between "here's the
                        # plan" and any tool actually mutating the
                        # workspace. Declining leaves it saved for later
                        # (/execute-plan, or /plan again to revise first).
                        if console.is_terminal:
                            try:
                                answer = console.input(
                                    "[cyan]Execute this plan now?[/cyan] [dim](y/N)[/dim] "
                                ).strip().lower()
                            except (KeyboardInterrupt, EOFError):
                                answer = ""
                            if answer in ("y", "yes"):
                                await _run_saved_plan(saved.id)
                            else:
                                console.print(f"[dim]Not executed. Run /execute-plan {saved.id} when ready.[/dim]")
                        else:
                            console.print(f"[dim]Run /execute-plan {saved.id} to execute.[/dim]")
                else:
                    outcome = await run_ai_task_and_stream(
                        client, renderer, console,
                        session_id=workspace.session_id, objective=intent.objective, mode=intent.mode,
                        approval_policy=config.approval_policy, interactive=True,
                        model=model_state.selected_model,
                        provider=model_state.selected_provider,
                        config=config,
                    )
                if outcome.status == "completed" and outcome.summary:
                    last_response_text = outcome.summary
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
