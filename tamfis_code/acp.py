"""Agent Client Protocol (ACP) adapter for IDE-hosted Tamfis Code sessions.

ACP is JSON-RPC 2.0 over newline-delimited stdio. Protocol output is kept on
stdout and all diagnostics go to stderr so editors can safely launch this as a
subprocess.
"""
from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from rich.console import Console

from . import __version__
from .config import Config


JsonObject = dict[str, Any]


@dataclass(slots=True)
class ACPSession:
    session_id: str
    runtime_session_id: int
    cwd: Path
    messages: list[JsonObject] = field(default_factory=list)


class ACPError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


class _ACPRenderer:
    """Translate local runtime streaming events into ACP session updates."""

    def __init__(
        self,
        session_id: str,
        notify: Callable[[str, JsonObject], Awaitable[None]],
    ):
        self.session_id = session_id
        self.notify = notify
        self.background_requested = asyncio.Event()
        self._pending: set[asyncio.Task[Any]] = set()
        self.emitted_text = False

    def handle_event(self, event: JsonObject) -> None:
        event_type = str(event.get("event_type") or "")
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        update: JsonObject | None = None
        if event_type == "assistant_delta" and payload.get("content"):
            self.emitted_text = True
            update = {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": str(payload["content"])},
            }
        elif event_type == "tool_call_requested":
            update = {
                "sessionUpdate": "tool_call",
                "toolCallId": str(payload.get("call_id") or payload.get("id") or "tool"),
                "title": str(payload.get("name") or "Tool"),
                "kind": "other",
                "status": "pending",
                "rawInput": payload.get("arguments") or {},
            }
        elif event_type == "tool_output":
            update = {
                "sessionUpdate": "tool_call_update",
                "toolCallId": str(payload.get("call_id") or payload.get("id") or "tool"),
                "status": "completed",
                "rawOutput": payload.get("result"),
            }
        if update is not None:
            task = asyncio.create_task(
                self.notify("session/update", {"sessionId": self.session_id, "update": update})
            )
            self._pending.add(task)
            task.add_done_callback(self._pending.discard)

    async def drain(self) -> None:
        if self._pending:
            await asyncio.gather(*tuple(self._pending), return_exceptions=True)


class ACPAgent:
    """Small ACP v1 server using the existing Tamfis local agent runtime."""

    def __init__(self, workspace_root: Path, config: Config):
        self.workspace_root = workspace_root.resolve()
        self.config = config
        self.sessions: dict[str, ACPSession] = {}
        self.active_prompts: dict[str, asyncio.Task[Any]] = {}
        self._write_lock = asyncio.Lock()

    async def send(self, payload: JsonObject) -> None:
        encoded = json.dumps(payload, separators=(",", ":"), default=str)
        async with self._write_lock:
            sys.stdout.write(encoded + "\n")
            sys.stdout.flush()

    async def notify(self, method: str, params: JsonObject) -> None:
        await self.send({"jsonrpc": "2.0", "method": method, "params": params})

    def _allowed_cwd(self, value: str | None) -> Path:
        candidate = Path(value or self.workspace_root).expanduser().resolve()
        if not candidate.is_dir():
            raise ACPError(-32602, f"working directory does not exist: {candidate}")
        roots = [self.workspace_root]
        roots.extend(Path(item).expanduser().resolve() for item in self.config.workspace_roots)
        if not any(candidate == root or candidate.is_relative_to(root) for root in roots):
            raise ACPError(-32602, f"working directory is outside configured workspace roots: {candidate}")
        return candidate

    @staticmethod
    def _prompt_text(prompt: Any) -> str:
        if isinstance(prompt, str):
            return prompt
        if not isinstance(prompt, list):
            raise ACPError(-32602, "prompt must be text or an array of content blocks")
        parts: list[str] = []
        for block in prompt:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            elif isinstance(block, dict) and block.get("type") in {"resource", "resource_link"}:
                parts.append(str(block.get("text") or block.get("uri") or ""))
        text = "\n".join(part for part in parts if part).strip()
        if not text:
            raise ACPError(-32602, "prompt contains no supported text content")
        return text

    async def _new_session(self, params: JsonObject) -> JsonObject:
        from .workspace import resolve_local_workspace

        cwd = self._allowed_cwd(params.get("cwd"))
        workspace = resolve_local_workspace(cwd)
        session_id = str(workspace.session_id)
        self.sessions[session_id] = ACPSession(session_id, workspace.session_id, cwd)
        return {"sessionId": session_id}

    async def _load_session(self, params: JsonObject) -> JsonObject:
        from . import state as local_state

        session_id = str(params.get("sessionId") or "")
        if not session_id:
            raise ACPError(-32602, "sessionId is required")
        if session_id not in self.sessions:
            try:
                runtime_id = int(session_id)
            except ValueError as exc:
                raise ACPError(-32602, f"invalid Tamfis session id: {session_id}") from exc
            state = local_state.get_session_state(runtime_id)
            cwd = self._allowed_cwd(params.get("cwd") or state.workspace_root or state.primary_workspace)
            self.sessions[session_id] = ACPSession(session_id, runtime_id, cwd)
        return {"sessionId": session_id}

    async def _run_prompt(self, session: ACPSession, text: str, renderer: _ACPRenderer):
        from .local_chat import resolve_provider_type
        from .providers import ProviderManager
        from .runner_local import run_local_agent_turn

        manager = ProviderManager()
        return await run_local_agent_turn(
            manager,
            resolve_provider_type("auto"),
            None,
            session.messages,
            Console(file=sys.stderr, no_color=True),
            renderer,  # type: ignore[arg-type]
            workspace_root=str(session.cwd),
            session_id=session.runtime_session_id,
            approval_policy=self.config.approval_policy,
            interactive=False,
            cli_config=self.config,
            allow_swarm_tool=True,
        )

    async def _prompt(self, params: JsonObject) -> JsonObject:
        session_id = str(params.get("sessionId") or "")
        session = self.sessions.get(session_id)
        if session is None:
            raise ACPError(-32602, f"unknown session: {session_id}")
        text = self._prompt_text(params.get("prompt"))
        session.messages.append({"role": "user", "content": text})
        renderer = _ACPRenderer(session_id, self.notify)
        current = asyncio.current_task()
        if current is not None:
            self.active_prompts[session_id] = current
        try:
            outcome = await self._run_prompt(session, text, renderer)
            await renderer.drain()
        finally:
            self.active_prompts.pop(session_id, None)
        if outcome.status != "completed":
            raise ACPError(-32603, outcome.error or "Tamfis Code task failed")
        summary = outcome.summary or ""
        if summary and not renderer.emitted_text:
            await self.notify(
                "session/update",
                {
                    "sessionId": session_id,
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": summary},
                    },
                },
            )
        session.messages.append({"role": "assistant", "content": summary})
        return {"stopReason": "end_turn"}

    async def handle(self, method: str, params: JsonObject | None = None) -> JsonObject:
        values = params or {}
        if method == "initialize":
            offered = values.get("protocolVersion", 1)
            if offered != 1:
                raise ACPError(-32602, f"unsupported ACP protocol version: {offered}")
            return {
                "protocolVersion": 1,
                "agentCapabilities": {
                    "loadSession": True,
                    "promptCapabilities": {
                        "image": False,
                        "audio": False,
                        "embeddedContext": True,
                    },
                    "mcpCapabilities": {"http": False, "sse": False},
                },
                "agentInfo": {"name": "Tamfis Code", "version": __version__},
                "authMethods": [],
            }
        if method == "authenticate":
            return {}
        if method == "session/new":
            return await self._new_session(values)
        if method == "session/load":
            return await self._load_session(values)
        if method == "session/prompt":
            return await self._prompt(values)
        if method == "session/cancel":
            session_id = str(values.get("sessionId") or "")
            task = self.active_prompts.get(session_id)
            if task is not None:
                task.cancel()
            return {}
        raise ACPError(-32601, f"method not found: {method}")

    async def _dispatch(self, message: JsonObject) -> None:
        request_id = message.get("id")
        method = message.get("method")
        if not isinstance(method, str):
            if request_id is not None:
                await self.send({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32600, "message": "invalid request"},
                })
            return
        try:
            result = await self.handle(method, message.get("params"))
            if request_id is not None:
                await self.send({"jsonrpc": "2.0", "id": request_id, "result": result})
        except asyncio.CancelledError:
            if request_id is not None:
                await self.send({
                    "jsonrpc": "2.0", "id": request_id,
                    "result": {"stopReason": "cancelled"},
                })
        except ACPError as exc:
            if request_id is not None:
                await self.send({
                    "jsonrpc": "2.0", "id": request_id,
                    "error": {"code": exc.code, "message": str(exc)},
                })
        except Exception as exc:  # protocol boundary: never leak a traceback to stdout
            print(f"ACP error: {exc}", file=sys.stderr)
            if request_id is not None:
                await self.send({
                    "jsonrpc": "2.0", "id": request_id,
                    "error": {"code": -32603, "message": str(exc)},
                })

    async def serve(self) -> None:
        tasks: set[asyncio.Task[Any]] = set()
        while True:
            line = await asyncio.to_thread(sys.stdin.buffer.readline)
            if not line:
                break
            try:
                message = json.loads(line)
                if not isinstance(message, dict):
                    raise ValueError("request must be an object")
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                await self.send({
                    "jsonrpc": "2.0", "id": None,
                    "error": {"code": -32700, "message": f"parse error: {exc}"},
                })
                continue
            task = asyncio.create_task(self._dispatch(message))
            tasks.add(task)
            task.add_done_callback(tasks.discard)
        if tasks:
            await asyncio.gather(*tuple(tasks), return_exceptions=True)


async def run_acp_server(workspace_root: Path, config: Config) -> None:
    await ACPAgent(workspace_root, config).serve()
