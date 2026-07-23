"""TamfisGPT Code -- a standalone terminal coding agent.

By default, this package calls an LLM provider directly (NVIDIA NIM,
OpenRouter, or Hugging Face -- see providers.py/runner_local.py) and
runs its own agent loop, tool execution (mcp.py), and local risk
classification/approval/mutation ledger (safety.py) -- no separate backend
process required. Every `ask`/`chat`/`audit`/`plan`/`agent`/`exec` command,
and the interactive REPL, work this way unless `--remote` is passed.

`--remote` still supports the original architecture this package started
as: a thin client to the TamfisGPT Remote Workspace backend (same sessions/
tasks/tools/approvals/events -- see tier_ii_gateway/api/remote.py), which
does the equivalent work server-side. That path is kept for continuity but
is not the primary, developed-going-forward one.

See docs/REMOTE_AGENT_MASTER_SPEC.md, Phase 21, for the original --remote
architecture's spec.
"""

__version__ = "0.6.13"

# Bumped whenever a CLI release requires a minimum backend Remote API
# contract version. Only meaningful for --remote; the standalone path has no
# server to negotiate a contract version with. There is no server-side
# version negotiation yet (`tamfis-code doctor --remote` just checks
# reachability) -- this constant exists so that check has something concrete
# to compare against once one is added, rather than silently assuming
# compatibility forever.
MIN_COMPATIBLE_API_VERSION = "remote-ai-v2"

# New exports
from .completion import ShellCompleter
from .metrics import MetricsTracker, StreamMetrics
from .agents import AgentManager, SubAgent, CodeAnalyzer, TestGenerator, DocGenerator
from .mcp import MCPServer, ToolDefinition, call_tool
from .indexer import CodeIndexer, CodeSymbol, CodeFile

__all__ = [
    # Existing...
    'ShellCompleter',
    'MetricsTracker',
    'StreamMetrics',
    'AgentManager',
    'SubAgent',
    'CodeAnalyzer',
    'TestGenerator',
    'DocGenerator',
    'MCPServer',
    'ToolDefinition',
    'call_tool',
    'CodeIndexer',
    'CodeSymbol',
    'CodeFile',
]
