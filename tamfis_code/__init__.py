"""Tamfis-Code, a standalone terminal coding agent.

The package keeps import-time side effects deliberately minimal. Public
components are loaded lazily so a missing optional module cannot prevent the
CLI or deterministic runtime from starting.
"""
from __future__ import annotations

import os
from importlib import import_module
from typing import Any

# CPR probes make some terminals pause or leave the interactive UI looking
# frozen.  The CLI uses prompt-toolkit, so disable the probe by default; users
# with a terminal that implements CPR can explicitly opt back in.
os.environ.setdefault("PROMPT_TOOLKIT_NO_CPR", "1")

__version__ = "1.6.10"
MIN_COMPATIBLE_API_VERSION = "remote-ai-v2"

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ShellCompleter": (".completion", "ShellCompleter"),
    "MetricsTracker": (".metrics", "MetricsTracker"),
    "StreamMetrics": (".metrics", "StreamMetrics"),
    "AgentManager": (".agents", "AgentManager"),
    "SubAgent": (".agents", "SubAgent"),
    "CodeAnalyzer": (".agents", "CodeAnalyzer"),
    "TestGenerator": (".agents", "TestGenerator"),
    "DocGenerator": (".agents", "DocGenerator"),
    "MCPServer": (".mcp", "MCPServer"),
    "ToolDefinition": (".mcp", "ToolDefinition"),
    "call_tool": (".mcp", "call_tool"),
    "CodeIndexer": (".indexer", "CodeIndexer"),
    "CodeSymbol": (".indexer", "CodeSymbol"),
    "CodeFile": (".indexer", "CodeFile"),
}

__all__ = ["__version__", "MIN_COMPATIBLE_API_VERSION", *_LAZY_EXPORTS]


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    try:
        module = import_module(module_name, __name__)
    except ModuleNotFoundError as exc:
        raise AttributeError(
            f"Optional Tamfis-Code component {name!r} is unavailable because "
            f"module {module_name!r} is not installed."
        ) from exc
    value = getattr(module, attribute)
    globals()[name] = value
    return value
