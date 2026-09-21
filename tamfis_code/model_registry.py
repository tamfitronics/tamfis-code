"""Canonical model capability registry used by local and Tier IV routing."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class ModelCapabilities:
    coding: bool = True
    tool_calling: bool = True
    parallel_tool_calls: bool = False
    structured_output: bool = True
    vision: bool = False
    long_context: bool = False


@dataclass(frozen=True)
class ModelRecord:
    id: str
    provider: str
    capabilities: ModelCapabilities
    context_window: int
    recommended_for: tuple[str, ...] = field(default_factory=tuple)
    quality_tier: str = "balanced"
    cost_tier: str = "medium"


MODELS: dict[str, ModelRecord] = {
    "nvidia/nemotron-3-super-120b-a12b": ModelRecord(
        "nvidia/nemotron-3-super-120b-a12b", "nvidia",
        ModelCapabilities(long_context=True), 128000,
        ("repository_audit", "multi_file_edit", "debugging", "planning", "tool_heavy_execution"),
        "frontier", "medium",
    ),
    "nvidia/nemotron-3-ultra-550b-a55b": ModelRecord(
        "nvidia/nemotron-3-ultra-550b-a55b", "nvidia",
        ModelCapabilities(long_context=True), 128000,
        ("repository_audit", "architecture", "long_context_review"), "frontier", "high",
    ),
    # Direct replacement for NVIDIA's retired nemotron-3-nano-30b-a3b.
    # Live-verified 2026-09-13: HTTP 200 and a native function tool_calls
    # response with valid JSON arguments through NVIDIA NIM.
    "nvidia/nemotron-3.5-lightning-30b-a3b": ModelRecord(
        "nvidia/nemotron-3.5-lightning-30b-a3b", "nvidia",
        ModelCapabilities(), 128000,
        ("multi_file_edit", "debugging", "tool_heavy_execution"), "high", "medium",
    ),
    # Also confirmed live on openrouter (same "moonshotai/kimi-k2.6" id;
    # not a second dict entry since the id is identical) and on HF's
    # router (a distinct id -- "moonshotai/Kimi-K2.6", different casing,
    # see the entry below) -- see providers.py's NVIDIA default_model
    # comment for why NVIDIA's own account-entitlement gap for this model
    # made the extra routes worth confirming and recording.
    "moonshotai/kimi-k2.6": ModelRecord(
        "moonshotai/kimi-k2.6", "nvidia",
        ModelCapabilities(parallel_tool_calls=True, long_context=True), 128000,
        ("multi_file_edit", "tool_heavy_execution", "planning"), "frontier", "medium",
    ),
    # ADDED 2026-09-17 (owner directive: mirror TamfisGPT's model grouping,
    # NIM free models -- kimi-k3, glm-5.3, other tool-calling-verified
    # models): z-ai/glm-5.3 on NVIDIA NIM. Live-verified 2026-09-17 against
    # integrate.api.nvidia.com with a real account key: (1) plain chat 200
    # OK with real content; (2) a real get_weather function-calling probe
    # returned a genuine tool_calls event with valid JSON arguments plus
    # reasoning_content; (3) SSE streaming returned real incremental delta
    # chunks. Mirrors tamgpt6's orchestration.yaml glm-5.3-nim entry
    # (same free-frontier reasoning). Deliberately NOT claiming vision or
    # parallel_tool_calls: the image-url probe timed out twice at 90s and
    # no multi-tool-call probe was run -- untested capabilities stay
    # unclaimed (same discipline as the kimi-k3 entry above).
    "z-ai/glm-5.3": ModelRecord(
        "z-ai/glm-5.3", "nvidia",
        ModelCapabilities(long_context=True), 128000,
        ("multi_file_edit", "planning", "tool_heavy_execution", "debugging"), "frontier", "medium",
    ),
    # ADDED 2026-08-30: live-verified against integrate.api.nvidia.com with
    # a real account key -- plain chat (200 OK), a real function-calling
    # probe (returned a genuine tool_calls event, not narrated text), and
    # NVIDIA's own catalog vision payload shape (text + image_url), which
    # returned an accurate description of the real test image. vision=True
    # is real, not a guess. Deliberately NOT claiming parallel_tool_calls
    # or long_context here even though the sibling kimi-k2.6 entry above
    # sets both -- only a single-tool-call probe and short-context chat
    # were tested this pass, not simultaneous multi-tool calls or long-
    # context behavior on this specific NIM host. context_window kept at
    # 128000 (matching kimi-k2.6) rather than the 1M this model is
    # documented at elsewhere (Ollama Cloud's kimi-k3:cloud comment in
    # providers.py) for the same reason.
    # FIX 2026-09-17: tool_calling was implicitly True (dataclass default)
    # but this entry only ever documented a chat + vision probe -- the same
    # live session also returned a genuine tool_calls event, which is what
    # made tamgpt6 adopt this model as its top free-frontier route. Now
    # explicit so eligible_models(requires_tools=True) is honest about it.
    "moonshotai/kimi-k3": ModelRecord(
        "moonshotai/kimi-k3", "nvidia",
        ModelCapabilities(tool_calling=True, vision=True, long_context=True), 1048576,
        ("multi_file_edit", "planning", "tool_heavy_execution", "vision_assisted_coding"), "frontier", "medium",
    ),
    # Keep direct xAI and OpenRouter relay routes distinct. TamfisGPT's live
    # catalogue classifies direct Grok 4.6 as a Premium-accessible frontier
    # route (Ultima-quality), while the cheaper relay remains Ultra-quality.
    "grok-4.6": ModelRecord(
        "grok-4.6", "grok",
        ModelCapabilities(vision=True, long_context=True), 500000,
        ("repository_audit", "multi_file_edit", "debugging", "planning"),
        "frontier", "high",
    ),
    "x-ai/grok-4.6": ModelRecord(
        "x-ai/grok-4.6", "openrouter",
        ModelCapabilities(vision=True, long_context=True), 500000,
        ("repository_audit", "multi_file_edit", "debugging", "planning"),
        "frontier", "medium",
    ),
    "moonshotai/Kimi-K2.6": ModelRecord(
        "moonshotai/Kimi-K2.6", "hf",
        ModelCapabilities(parallel_tool_calls=True, long_context=True), 128000,
        ("multi_file_edit", "tool_heavy_execution", "planning"), "frontier", "medium",
    ),
    "google/gemini-2.5-flash": ModelRecord(
        "google/gemini-2.5-flash", "openrouter",
        ModelCapabilities(vision=True, long_context=True), 1000000,
        ("long_context_review", "repository_search", "vision_assisted_coding"), "high", "medium",
    ),
    "qwen/qwen3-coder": ModelRecord(
        "qwen/qwen3-coder", "openrouter",
        ModelCapabilities(parallel_tool_calls=True, long_context=True), 256000,
        ("repository_audit", "multi_file_edit", "debugging", "planning", "tool_heavy_execution"),
        "frontier", "medium",
    ),
    "deepseek/deepseek-chat-v3-0324": ModelRecord(
        "deepseek/deepseek-chat-v3-0324", "openrouter",
        ModelCapabilities(parallel_tool_calls=True, long_context=True), 128000,
        ("repository_search", "multi_file_edit", "debugging", "planning"),
        "high", "low",
    ),
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning": ModelRecord(
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning", "nvidia",
        ModelCapabilities(parallel_tool_calls=True, long_context=True), 128000,
        ("repository_audit", "multi_file_edit", "debugging", "planning", "tool_heavy_execution"),
        "frontier", "medium",
    ),
    "deepseek-ai/deepseek-v4-pro": ModelRecord(
        "deepseek-ai/deepseek-v4-pro", "nvidia",
        ModelCapabilities(parallel_tool_calls=True, long_context=True), 1000000,
        ("repository_audit", "multi_file_edit", "debugging", "planning", "tool_heavy_execution"),
        "frontier", "high",
    ),
    "deepseek-ai/deepseek-v4.1-flash": ModelRecord(
        "deepseek-ai/deepseek-v4.1-flash", "nvidia",
        ModelCapabilities(parallel_tool_calls=True, long_context=True), 1000000,
        ("repository_search", "multi_file_edit", "debugging", "tool_heavy_execution"),
        "frontier", "medium",
    ),
    "Qwen/Qwen3.6-35B-A3B": ModelRecord(
        "Qwen/Qwen3.6-35B-A3B", "hf",
        ModelCapabilities(parallel_tool_calls=True, long_context=True), 262144,
        ("repository_audit", "multi_file_edit", "debugging", "planning", "tool_heavy_execution"),
        "frontier", "medium",
    ),
    "Qwen/Qwen3.6-27B": ModelRecord(
        "Qwen/Qwen3.6-27B", "hf",
        ModelCapabilities(parallel_tool_calls=True, long_context=True), 262144,
        ("repository_search", "multi_file_edit", "debugging", "planning"),
        "frontier", "medium",
    ),
    "Qwen/Qwen3-Coder-480B-A35B-Instruct": ModelRecord(
        "Qwen/Qwen3-Coder-480B-A35B-Instruct", "hf",
        ModelCapabilities(parallel_tool_calls=True, long_context=True), 262144,
        ("repository_audit", "multi_file_edit", "debugging", "planning", "tool_heavy_execution"),
        "frontier", "high",
    ),
    "deepseek-ai/DeepSeek-V4-Pro": ModelRecord(
        "deepseek-ai/DeepSeek-V4-Pro", "hf",
        ModelCapabilities(parallel_tool_calls=True, long_context=True), 1048576,
        ("repository_audit", "multi_file_edit", "debugging", "planning", "tool_heavy_execution"),
        "frontier", "high",
    ),
    "deepseek-ai/DeepSeek-V4.1-Flash": ModelRecord(
        "deepseek-ai/DeepSeek-V4.1-Flash", "hf",
        ModelCapabilities(parallel_tool_calls=True, long_context=True, vision=True), 1048576,
        ("repository_search", "multi_file_edit", "debugging", "tool_heavy_execution"),
        "frontier", "medium",
    ),
    # Meta Model API frontier routes. The API is OpenAI-compatible but paid;
    # these are explicit-only in providers.py and are never part of the free
    # automatic pool.
    "muse-spark-1.3": ModelRecord(
        "muse-spark-1.3", "meta",
        ModelCapabilities(parallel_tool_calls=True, long_context=True), 1_048_576,
        ("repository_audit", "multi_file_edit", "debugging", "planning", "tool_heavy_execution"),
        "frontier", "high",
    ),
    "muse-spark-1.2": ModelRecord(
        "muse-spark-1.2", "meta",
        ModelCapabilities(parallel_tool_calls=True, long_context=True), 1_048_576,
        ("repository_audit", "multi_file_edit", "debugging", "planning"),
        "frontier", "high",
    ),
    "muse-spark-1.1": ModelRecord(
        "muse-spark-1.1", "meta",
        ModelCapabilities(parallel_tool_calls=True, long_context=True), 1_048_576,
        ("repository_search", "debugging", "planning"),
        "high", "high",
    ),
    "muse-spark-1.3-contributor": ModelRecord(
        "muse-spark-1.3-contributor", "meta",
        ModelCapabilities(parallel_tool_calls=True, long_context=True), 1_048_576,
        ("repository_audit", "multi_file_edit", "debugging", "planning", "tool_heavy_execution"),
        "frontier", "high",
    ),
    "muse-spark-1.2-contributor": ModelRecord(
        "muse-spark-1.2-contributor", "meta",
        ModelCapabilities(parallel_tool_calls=True, long_context=True), 1_048_576,
        ("repository_audit", "multi_file_edit", "debugging", "planning"),
        "frontier", "high",
    ),
}


def get_model(model_id: str) -> ModelRecord | None:
    return MODELS.get(model_id)


def eligible_models(*, task_type: str, requires_tools: bool, requires_long_context: bool) -> list[ModelRecord]:
    records: Iterable[ModelRecord] = MODELS.values()
    return [
        item for item in records
        if (not requires_tools or item.capabilities.tool_calling)
        and (not requires_long_context or item.capabilities.long_context)
        and (task_type in item.recommended_for or not item.recommended_for)
    ]
