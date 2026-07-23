from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from .conversation import Conversation, ConversationState
from .events import EventKind


class EventRenderer:
    """Renderer adapter that records the existing Tamfis agent loop as events."""

    def __init__(self, conversation: Conversation):
        self.conversation = conversation
        self.streamed_final_text = False

    def handle_event(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("event_type") or event.get("event") or event.get("type") or "event")
        payload = dict(event.get("payload") or {})
        mapping = {
            "assistant_delta": EventKind.AGENT_MESSAGE,
            "plan_created": EventKind.PLAN_CREATED,
            "plan_step_progress": EventKind.PLAN_UPDATED,
            "tool_call": EventKind.TOOL_STARTED,
            "tool_result": EventKind.TOOL_FINISHED,
            "approval_required": EventKind.APPROVAL_REQUESTED,
            "ai_task_completed": EventKind.COMPLETED,
            "ai_task_failed": EventKind.ERROR,
        }
        kind = mapping.get(event_type, EventKind.OBSERVATION)
        payload.setdefault("source_event", event_type)
        if event_type == "assistant_delta" and payload.get("content"):
            self.streamed_final_text = True
        self.conversation.emit(kind, payload, actor="agent")


@dataclass(slots=True)
class AgentRunResult:
    status: str
    summary: str = ""
    error: str | None = None


class TamfisAgent:
    """Adapter that drives the proven Tamfis local agent loop via OpenHands events."""

    def __init__(self, conversation: Conversation):
        self.conversation = conversation
        self._task: asyncio.Task[AgentRunResult] | None = None

    async def run(
        self,
        objective: str,
        *,
        provider: str = "auto",
        model: str | None = None,
        approval_policy: str = "ask",
        read_only: bool = False,
        max_rounds: int = 30,
    ) -> AgentRunResult:
        from rich.console import Console
        from tamfis_code.providers import ProviderManager, ProviderType
        from tamfis_code.runner_local import run_local_agent_turn

        if self.conversation.state == ConversationState.RUNNING:
            raise RuntimeError("conversation is already running")
        self.conversation.send_message(objective)
        self.conversation.set_state(ConversationState.RUNNING)
        renderer = EventRenderer(self.conversation)
        manager = ProviderManager()
        try:
            selected = ProviderType(provider)
        except ValueError as exc:
            raise ValueError(f"unsupported provider: {provider}") from exc
        try:
            outcome = await run_local_agent_turn(
                manager,
                selected,
                model,
                [{"role": "user", "content": objective}],
                Console(quiet=True),
                renderer,
                workspace_root=str(self.conversation.workspace.root),
                session_id=abs(hash(self.conversation.id)) % 2_000_000_000,
                approval_policy=approval_policy,
                interactive=False,
                max_rounds=max_rounds,
                read_only=read_only,
            )
        except asyncio.CancelledError:
            self.conversation.cancel()
            raise
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            self.conversation.emit(EventKind.ERROR, {"error": message})
            self.conversation.set_state(ConversationState.FAILED, message)
            return AgentRunResult("failed", error=message)

        if outcome.status == "completed":
            self.conversation.set_state(ConversationState.COMPLETED)
        elif outcome.status == "cancelled":
            self.conversation.set_state(ConversationState.CANCELLED)
        else:
            self.conversation.set_state(ConversationState.FAILED, outcome.error or "agent failed")
        return AgentRunResult(outcome.status, outcome.summary or "", outcome.error)

    def start(self, objective: str, **kwargs: Any) -> asyncio.Task[AgentRunResult]:
        if self._task and not self._task.done():
            raise RuntimeError("agent run already active")
        self._task = asyncio.create_task(self.run(objective, **kwargs))
        return self._task

    def cancel(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
        self.conversation.cancel()
