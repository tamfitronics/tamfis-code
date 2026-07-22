"""Offline/local chat: talk to a directly-configured LLM provider with no
TamfisGPT backend, account, or network round-trip to tamgpt6 at all.

This is a deliberately separate, second model-routing path from the Remote
Workspace backend's Tier IV orchestration -- see providers.py's module
docstring. It exists for genuine directly-configured-provider use with no
TamfisGPT account or server round-trip; it does not replicate the backend's
model health/fallback/quota-tracking, and it never exposes mutating tools
(see local_tools.py). Real conversational value only: read-only repo
inspection via tool-calling, no file writes, no shell execution, no
approval gate (nothing here can mutate anything, so none is needed).
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional

from rich.console import Console

from .local_tools import READ_ONLY_TOOL_SCHEMAS, LocalReadOnlyTools
from .providers import ProviderManager, ProviderType

MAX_TOOL_ROUNDS = 5

_PROVIDER_ALIASES = {
    "hf": ProviderType.HF, "huggingface": ProviderType.HF,
    "nvidia": ProviderType.NVIDIA, "nvidia_nim": ProviderType.NVIDIA,
    "or": ProviderType.OPENROUTER, "openrouter": ProviderType.OPENROUTER,
    "tamfis": ProviderType.TAMFIS, "tamfisgpt": ProviderType.TAMFIS,
    "auto": ProviderType.AUTO,
}


def resolve_provider_type(name: Optional[str]) -> ProviderType:
    key = (name or "auto").strip().lower()
    if key not in _PROVIDER_ALIASES:
        raise ValueError(f"Unknown local provider: {name!r}. Choose one of: {', '.join(sorted(_PROVIDER_ALIASES))}")
    return _PROVIDER_ALIASES[key]


async def run_local_turn(
    manager: ProviderManager,
    provider: ProviderType,
    messages: List[Dict[str, Any]],
    model: Optional[str],
    console: Console,
    *,
    use_tools: bool = True,
) -> str:
    """Run one user turn to completion, resolving any read-only tool calls
    the model requests along the way. Returns the final assistant text.

    Non-streaming by design: accumulating partial tool_call fragments across
    a streamed response is real complexity this conservative first pass
    intentionally skips (see module docstring's scope note) -- a user still
    sees the final answer, just not token-by-token for turns with tool use.
    """
    tools_client = LocalReadOnlyTools() if use_tools else None
    working_messages = list(messages)

    for _ in range(MAX_TOOL_ROUNDS):
        client = manager.get_client(provider)
        if not client:
            raise RuntimeError(f"Provider {provider.value} is not available (no client / no valid credentials).")
        config = manager.PROVIDERS[provider if provider != ProviderType.AUTO else manager._select_best_provider()]
        # No task_profile here (local/offline chat has no classify_task
        # concept) -- select_model(..., None) treats that as "not a
        # demanding task", which for OpenRouter means its free-tier model,
        # matching the same credit-saving default as the standalone loop.
        resolved_model = model or manager.select_model(config, None)

        kwargs: Dict[str, Any] = {}
        if tools_client is not None:
            kwargs["tools"] = READ_ONLY_TOOL_SCHEMAS
            kwargs["tool_choice"] = "auto"

        response = await client.chat.completions.create(
            model=resolved_model, messages=working_messages, stream=False,
            temperature=0.2, max_tokens=4096, **kwargs,
        )
        if not response.choices:
            return ""
        choice = response.choices[0]
        message = choice.message
        tool_calls = getattr(message, "tool_calls", None)

        if not tool_calls:
            return message.content or ""

        working_messages.append({
            "role": "assistant", "content": message.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in tool_calls
            ],
        })
        for tc in tool_calls:
            console.print(f"[dim]· {tc.function.name}({tc.function.arguments})[/dim]")
            try:
                arguments = json.loads(tc.function.arguments or "{}")
                result = await tools_client.call(tc.function.name, arguments)
                content = json.dumps(result, default=str)
            except Exception as exc:
                content = json.dumps({"error": str(exc)})
            working_messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})

    return "(Stopped after exhausting the local tool-call round limit -- try a narrower question.)"


async def stream_local_turn(
    manager: ProviderManager,
    provider: ProviderType,
    messages: List[Dict[str, Any]],
    model: Optional[str],
) -> AsyncIterator[str]:
    """Plain streaming, no tools -- used when the caller already knows this
    turn needs no repo inspection (kept separate from run_local_turn so the
    common single-shot/no-tools case still gets real token streaming)."""
    async for chunk in manager.chat_completion(provider, messages, model=model, stream=True):
        yield chunk
