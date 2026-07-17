"""The standalone agent loop: calls an LLM provider directly (via
providers.py's ProviderManager) and runs its own tool-calling loop locally,
with no TamfisGPT Remote Workspace backend involved at all.

Generalizes local_chat.py's run_local_turn (which already proved the basic
send-tools -> parse tool_calls -> execute -> append role:"tool" -> resend
pattern works against HF/NVIDIA NIM/OpenRouter/Ollama) into the primary,
full-capability loop:
  - streaming + tool-calling combined (local_chat.py's run_local_turn has
    tools but no streaming; stream_local_turn streams but has no tools)
  - the full tool set via mcp.py's MCPServer (read/write/edit/execute/etc),
    not just the four read-only tools -- gated per call through safety.py's
    risk classifier and runner.py's existing resolve_approval_decision
  - an open-ended round loop (a high safety-valve cap, not local_chat.py's
    hard MAX_TOOL_ROUNDS=5) with real termination conditions

Emits the same event-dict shape StreamRenderer.handle_event already expects
(assistant_delta/tool_call_requested/tool_output/file_mutation/
approval_required/ai_task_completed/ai_task_failed) so render.py needs no
changes to work with a local loop instead of remote SSE events.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from rich.console import Console

from . import state as local_state
from .mcp import MCPServer
from .providers import ProviderManager, ProviderType
from .render import StreamRenderer
from .runner import TaskOutcome, resolve_approval_decision
from .safety import READ_ONLY_TOOLS, classify_tool_call_risk

# A safety-valve ceiling, not a target -- local_chat.py's MAX_TOOL_ROUNDS=5
# was appropriate for a read-only Q&A loop; a real coding-agent task
# legitimately needs many more tool calls (read several files, make several
# edits, run tests, iterate). This exists only to guarantee termination if
# something is genuinely stuck in a loop, not to cap normal work.
MAX_AGENT_ROUNDS = 200

# If the model requests the exact same tool call(s) (name + arguments,
# unordered) this many rounds in a row, stop rather than let it spin: a
# weaker model asked to check on something that never changes (a health
# endpoint that's never up, a container ID it never filled in) will
# otherwise repeat identically until MAX_AGENT_ROUNDS or a context-window
# error ends it, burning a lot of time/tokens along the way. 2 tolerates a
# legitimate one-off retry; 3 identical rounds in a row is not progress.
MAX_CONSECUTIVE_IDENTICAL_ROUNDS = 2

# Same 4-chars-per-token heuristic render.py uses for its live token counter
# (_CHARS_PER_TOKEN_ESTIMATE) -- good enough to budget against a context
# window, not meant to match a real tokenizer exactly.
_CHARS_PER_TOKEN_ESTIMATE = 4
MAX_TOKENS_PER_REQUEST = 4096
# Leave headroom below the provider's stated context_window: it's a
# conservative estimate already (see providers.py), and this estimate's own
# char/token ratio is approximate too.
_CONTEXT_SAFETY_MARGIN = 0.9


def _estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Rough token estimate for a working-messages list, counting both
    message content and any tool_calls argument strings (which can be large
    for tools like write_file)."""
    total_chars = 0
    for message in messages:
        total_chars += len(str(message.get("content") or ""))
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            total_chars += len(str(function.get("arguments") or ""))
    return total_chars // _CHARS_PER_TOKEN_ESTIMATE


def _trim_tool_outputs(messages: list[dict[str, Any]], target_tokens: int, keep_recent: int = 6) -> bool:
    """Truncate older, large tool-result message contents in place to
    reclaim context budget before a request would otherwise exceed the
    provider's window. Never touches the system message or the most recent
    `keep_recent` messages, since the model needs those intact to reason
    about what it just did. Returns True if anything was trimmed."""
    trimmed_any = False
    boundary = max(1, len(messages) - keep_recent)
    for message in messages[1:boundary]:
        if message.get("role") != "tool":
            continue
        content = str(message.get("content") or "")
        if len(content) > 400:
            message["content"] = (
                content[:200] + "\n...[truncated to reclaim context budget]...\n" + content[-100:]
            )
            trimmed_any = True
        if _estimate_tokens(messages) <= target_tokens:
            break
    return trimmed_any


def _latest_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content") or "").strip().lower()
    return ""


def _is_plain_conversation(messages: list[dict[str, Any]]) -> bool:
    """Return True for turns that cannot reasonably require repository tools.

    This is intentionally conservative.  It prevents greetings and basic
    conversational prompts from sending a large function catalogue to small
    local models such as llama3.2:3b, which may serialize invented function
    calls into ordinary assistant text instead of emitting tool_calls.
    """
    text = _latest_user_text(messages)
    if not text:
        return True
    exact = {
        "hi", "hello", "hey", "hi there", "hello there",
        "good morning", "good afternoon", "good evening",
        "how are you", "how are you?", "thanks", "thank you",
    }
    if text in exact:
        return True
    conversational_prefixes = (
        "who are you", "what can you do", "tell me about yourself",
    )
    return text.startswith(conversational_prefixes)


@dataclass
class _StreamedToolCall:
    call_id: str = ""
    name: str = ""
    arguments: str = ""


def _tool_calls_signature(tool_calls: list[_StreamedToolCall]) -> tuple[tuple[str, str], ...]:
    """Order-independent fingerprint of a round's tool calls, used to detect
    the model repeating the exact same request(s) round after round."""
    return tuple(sorted((tc.name, tc.arguments) for tc in tool_calls))


async def _stream_one_completion(
    client, *, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
    renderer: StreamRenderer,
) -> tuple[str, list[_StreamedToolCall]]:
    """Stream one chat-completions call, forwarding text deltas to the
    renderer as they arrive and accumulating any tool_calls deltas by index
    (the standard OpenAI-compatible streaming tool-call pattern: id/name
    only arrive in the first delta for a given index, arguments arrive as
    incremental string fragments across many deltas)."""
    content_parts: list[str] = []
    tool_calls_by_index: dict[int, _StreamedToolCall] = {}

    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": 0.2,
        "max_tokens": MAX_TOKENS_PER_REQUEST,
    }
    # Do not send an empty tools array. Some OpenAI-compatible servers and
    # small local models behave differently merely because tool mode is
    # present, even when no tool is useful for the current turn.
    if tools:
        request_kwargs["tools"] = tools
        request_kwargs["tool_choice"] = "auto"

    stream = await client.chat.completions.create(**request_kwargs)
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if getattr(delta, "content", None):
            content_parts.append(delta.content)
            renderer.handle_event({"event_type": "assistant_delta", "payload": {"content": delta.content}})
        for tc_delta in getattr(delta, "tool_calls", None) or []:
            slot = tool_calls_by_index.setdefault(tc_delta.index, _StreamedToolCall())
            if tc_delta.id:
                slot.call_id = tc_delta.id
            fn = getattr(tc_delta, "function", None)
            if fn is not None:
                if fn.name:
                    slot.name = fn.name
                if fn.arguments:
                    slot.arguments += fn.arguments

    ordered_calls = [tool_calls_by_index[i] for i in sorted(tool_calls_by_index)]
    return "".join(content_parts), ordered_calls


async def run_local_agent_turn(
    manager: ProviderManager,
    provider: ProviderType,
    model: Optional[str],
    messages: list[dict[str, Any]],
    console: Console,
    renderer: StreamRenderer,
    *,
    workspace_root: str,
    session_id: int,
    approval_policy: str = "ask",
    interactive: bool = True,
    max_rounds: int = MAX_AGENT_ROUNDS,
    read_only: bool = False,
) -> TaskOutcome:
    """Run one user turn to completion against a directly-called provider,
    executing tool calls locally (full tool set, approval-gated) instead of
    delegating to a Remote Workspace backend. Mirrors run_ai_task_and_stream's
    contract (same TaskOutcome shape) so cli.py/interactive.py can drive
    either the local or (while it still exists) remote path interchangeably.

    `read_only=True` restricts both the tool schema offered to the model AND
    (defense in depth, in case a model requests a tool it wasn't offered)
    execution itself to safety.READ_ONLY_TOOLS -- used for chat/audit/plan
    modes, which must never mutate the workspace.
    """
    # Emit lifecycle events before any provider/network operation.  Without
    # these, the Rich live panel remains at its constructor default (idle)
    # while get_client()/chat.completions.create() is resolving or waiting.
    renderer.handle_event({"event_type": "task_started", "payload": {"mode": "local"}})
    renderer.handle_event({"event_type": "context_loading", "payload": {"workspace_root": workspace_root}})

    mcp_server = MCPServer(workspace_root=workspace_root, session_id=session_id)
    if _is_plain_conversation(messages):
        tools: list[dict[str, Any]] = []
    elif read_only:
        tools = mcp_server.tool_schemas_openai(names=list(READ_ONLY_TOOLS))
    else:
        tools = mcp_server.tool_schemas_openai()

    # Plain conversation must not carry the full repository prompt. Besides
    # wasting context, small local models can close an OpenAI-compatible
    # stream without producing visible text when presented with a large coding
    # system prompt for a trivial greeting. Repository-aware turns still get
    # the complete workspace prompt (git branch/dirty status, instruction file
    # contents -- replaces the context tamgpt6 used to assemble server-side).
    if _is_plain_conversation(messages):
        system_prompt = (
            "You are TamfisGPT Code. Respond naturally and concisely to the "
            "user. Do not invent or call tools for ordinary conversation."
        )
    else:
        from .workspace import build_system_prompt
        system_prompt = build_system_prompt(session_id, Path(workspace_root))
    working_messages = [{"role": "system", "content": system_prompt}, *messages]
    session_approved_risks: set[str] = set()
    renderer.handle_event({"event_type": "context_reused", "payload": {"workspace_root": workspace_root}})

    previous_tool_calls_signature: Optional[tuple[tuple[str, str], ...]] = None
    consecutive_identical_rounds = 0

    for _round in range(max_rounds):
        renderer.handle_event({"event_type": "routing_started", "payload": {"requested_provider": provider.value}})
        client = manager.get_client(provider)
        if not client:
            renderer.handle_event({"event_type": "ai_task_failed", "payload": {"error": f"Provider {provider.value} is not available (no client / no valid credentials)."}})
            return TaskOutcome(status="failed", error=f"Provider {provider.value} is not available (no client / no valid credentials).")

        resolved_provider = provider if provider != ProviderType.AUTO else manager._select_best_provider()
        config = manager.PROVIDERS[resolved_provider]
        resolved_model = model or config.default_model

        # Never fire a request already guaranteed to blow the provider's
        # context window -- confirmed live: HF 400'd with inputs(29548) +
        # max_new_tokens(4096) > 32769 after a long tool-calling session grew
        # working_messages unchecked. Try reclaiming budget from old, large
        # tool outputs first; only give up if that's not enough.
        token_budget = int(config.context_window * _CONTEXT_SAFETY_MARGIN) - MAX_TOKENS_PER_REQUEST
        input_tokens = _estimate_tokens(working_messages)
        if input_tokens > token_budget:
            _trim_tool_outputs(working_messages, token_budget)
            input_tokens = _estimate_tokens(working_messages)
        if input_tokens > token_budget:
            message = (
                f"Stopping before round {_round + 1}: this turn has grown to "
                f"~{input_tokens} estimated tokens, too large for "
                f"{resolved_provider.value}'s ~{config.context_window}-token context "
                "window even after trimming old tool output. Start a new turn to "
                "continue (e.g. narrow the objective, or /clear stale context)."
            )
            renderer.handle_event({"event_type": "ai_task_failed", "payload": {"error": message}})
            return TaskOutcome(status="failed", error=message)

        renderer.handle_event({
            "event_type": "model_selected",
            "payload": {"provider": resolved_provider.value, "model": resolved_model},
        })
        renderer.handle_event({
            "event_type": "provider_request_started",
            "payload": {"provider": resolved_provider.value, "model": resolved_model, "round": _round + 1},
        })

        try:
            content, tool_calls = await _stream_one_completion(
                client, model=resolved_model, messages=working_messages, tools=tools, renderer=renderer,
            )
        except Exception as exc:
            renderer.handle_event({"event_type": "ai_task_failed", "payload": {"error": str(exc)}})
            return TaskOutcome(status="failed", error=str(exc))

        if not tool_calls:
            # Some OpenAI-compatible servers occasionally terminate a streamed
            # response with no content even though the same request succeeds
            # non-streaming. Never silently report completion with a blank
            # answer: retry once without streaming and render that canonical
            # response. This also gives a clear error if the provider returned
            # neither text nor tool calls.
            if not content.strip():
                fallback_kwargs: dict[str, Any] = {
                    "model": resolved_model,
                    "messages": working_messages,
                    "stream": False,
                    "temperature": 0.2,
                    "max_tokens": MAX_TOKENS_PER_REQUEST,
                }
                if tools:
                    fallback_kwargs["tools"] = tools
                    fallback_kwargs["tool_choice"] = "auto"
                try:
                    response = await client.chat.completions.create(**fallback_kwargs)
                    message = response.choices[0].message if response.choices else None
                    fallback_content = str(getattr(message, "content", None) or "")
                    if fallback_content:
                        content = fallback_content
                        renderer.handle_event({
                            "event_type": "assistant_delta",
                            "payload": {"content": fallback_content},
                        })
                except Exception as exc:
                    renderer.handle_event({
                        "event_type": "ai_task_failed",
                        "payload": {"error": f"Provider returned an empty stream; fallback failed: {exc}"},
                    })
                    return TaskOutcome(
                        status="failed",
                        error=f"Provider returned an empty stream; fallback failed: {exc}",
                    )
            if not content.strip():
                error = "Provider completed without assistant text or tool calls."
                renderer.handle_event({"event_type": "ai_task_failed", "payload": {"error": error}})
                return TaskOutcome(status="failed", error=error)
            renderer.handle_event({"event_type": "ai_task_completed", "payload": {"status": "completed"}})
            return TaskOutcome(status="completed", summary=content)

        signature = _tool_calls_signature(tool_calls)
        if signature == previous_tool_calls_signature:
            consecutive_identical_rounds += 1
        else:
            consecutive_identical_rounds = 0
        previous_tool_calls_signature = signature
        if consecutive_identical_rounds >= MAX_CONSECUTIVE_IDENTICAL_ROUNDS:
            names = ", ".join(sorted({tc.name for tc in tool_calls})) or "tool call"
            message = (
                f"Stopped after the same {names} request repeated "
                f"{consecutive_identical_rounds + 1} rounds in a row with identical "
                "arguments -- this usually means the model is stuck repeating itself "
                "rather than making progress, not that it's legitimately polling."
            )
            renderer.handle_event({"event_type": "ai_task_failed", "payload": {"error": message}})
            return TaskOutcome(status="failed", error=message)

        working_messages.append({
            "role": "assistant", "content": content or "",
            "tool_calls": [
                {"id": tc.call_id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
                for tc in tool_calls
            ],
        })

        for tc in tool_calls:
            try:
                arguments = json.loads(tc.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}

            risk = classify_tool_call_risk(tc.name, arguments, workspace_root=workspace_root)
            renderer.handle_event({"event_type": "tool_call_requested", "payload": {"name": tc.name, "arguments": arguments}})

            if read_only and tc.name not in READ_ONLY_TOOLS:
                result = {"error": f"'{tc.name}' is not available in read-only mode.", "success": False}
                working_messages.append({"role": "tool", "tool_call_id": tc.call_id, "content": json.dumps(result)})
                renderer.handle_event({"event_type": "tool_output", "payload": {"tool": tc.name, "result": result}})
                continue

            if risk != "read_only" and risk not in session_approved_risks:
                renderer.handle_event({
                    "event_type": "approval_required",
                    "payload": {
                        "command": f"{tc.name}({json.dumps(arguments, default=str)})", "risk_level": risk,
                        "working_directory": workspace_root, "reason": "The agent requested this command.",
                    },
                })
                decision = resolve_approval_decision(console, f"{tc.name}({json.dumps(arguments, default=str)})", risk, approval_policy, interactive)
                if decision == "approve_session":
                    session_approved_risks.add(risk)
                elif decision == "deny":
                    result = {"error": "Denied by approval policy -- try a different, less risky approach.", "success": False}
                    working_messages.append({"role": "tool", "tool_call_id": tc.call_id, "content": json.dumps(result)})
                    renderer.handle_event({"event_type": "tool_output", "payload": {"tool": tc.name, "result": result}})
                    continue

            result = await mcp_server.call_tool(tc.name, arguments)
            renderer.handle_event({"event_type": "tool_output", "payload": {"tool": tc.name, "result": result}})
            working_messages.append({"role": "tool", "tool_call_id": tc.call_id, "content": json.dumps(result, default=str)})

            if tc.name in {"write_file", "edit_file"} and result.get("success"):
                state = local_state.get_session_state(session_id)
                if state.modified_files:
                    mutation = state.modified_files[-1]
                    renderer.handle_event({"event_type": "file_mutation", "payload": {
                        "path": mutation["path"], "lines_added": mutation["lines_added"],
                        "lines_removed": mutation["lines_removed"], "mutation_id": mutation["mutation_id"],
                    }})

    message = f"(Stopped after {max_rounds} tool-call rounds without a final answer -- this usually means the task needs to be narrowed.)"
    renderer.handle_event({"event_type": "ai_task_failed", "payload": {"error": message}})
    return TaskOutcome(status="failed", error=message)


async def run_local_shell_command(
    console: Console, *, workspace_root: str, session_id: int, command: str,
    approval_policy: str, interactive: bool,
) -> TaskOutcome:
    """Standalone equivalent of runner.py's run_shell_command -- executes an
    explicit `$ <command>` / `/run` / `/shell` REPL command locally via
    MCPServer's execute_command tool, gated through the same local risk
    classifier and approval flow as any other tool call, instead of
    submitting it to a Remote command queue."""
    from .safety import classify_command_risk

    action = local_state.start_action(
        session_id, action_type="shell_command", purpose="Run an explicit local shell command",
        risk="policy_classified", detail=command,
    )
    console.print(f"[bold]$[/bold] {command}")

    risk = classify_command_risk(command)
    if risk != "read_only":
        decision = resolve_approval_decision(console, command, risk, approval_policy, interactive)
        if decision == "deny":
            local_state.finish_action(session_id, action.id, status="failed", summary="denied")
            console.print("[dim]Denied.[/dim]")
            return TaskOutcome(status="denied", error="Denied by approval policy")

    mcp_server = MCPServer(workspace_root=workspace_root, session_id=session_id)
    result = await mcp_server.call_tool("execute_command", {"command": command})
    payload = result.get("result") if isinstance(result.get("result"), dict) else result
    stdout = str(payload.get("stdout") or "")
    stderr = str(payload.get("stderr") or "")
    exit_code = payload.get("return_code")
    ok = bool(result.get("success")) and exit_code == 0
    body = stdout.strip()
    if stderr.strip():
        body = (body + "\n" + stderr.strip()).strip()
    if not body:
        body = "Command completed successfully with no output" if ok else f"Command failed with exit code {exit_code}"
    from rich.panel import Panel
    console.print(Panel(body, title=f"exit {exit_code}", border_style="green" if ok else "red"))

    outcome = TaskOutcome(status="completed", summary=stdout) if ok else TaskOutcome(status="failed", error=stderr or f"exit code {exit_code}")
    local_state.finish_action(session_id, action.id, status=outcome.status, summary=f"exit={exit_code}")
    local_state.checkpoint(session_id, reason=f"command_{outcome.status}", summary=f"exit={exit_code}")
    return outcome
