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
    "Qwen/Qwen2.5-7B-Instruct": ModelRecord(
        "Qwen/Qwen2.5-7B-Instruct", "hf",
        ModelCapabilities(long_context=False), 32768,
        ("single_file_explanation", "repository_search"), "balanced", "low",
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
