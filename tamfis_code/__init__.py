"""TamfisGPT Code -- terminal coding agent client.

A client of the same TamfisGPT Remote agent runtime the web workspace uses
(same sessions, tasks, tools, approvals, events, persistence -- see
tier_ii_gateway/api/remote.py). This package does not implement its own
provider routing, tool execution, or agent loop; it talks to the existing
Remote API over HTTP/SSE, the same way the frontend does.

See docs/REMOTE_AGENT_MASTER_SPEC.md, Phase 21, for the full spec this was
built against.
"""

__version__ = "0.1.0"

# Bumped whenever a CLI release requires a minimum backend Remote API
# contract version. There is no server-side version negotiation yet
# (`tamfis-code doctor` just checks reachability) -- this constant exists so
# that check has something concrete to compare against once one is added,
# rather than silently assuming compatibility forever.
MIN_COMPATIBLE_API_VERSION = "remote-ai-v2"

# New exports
from .completion import ShellCompleter
from .metrics import MetricsTracker, StreamMetrics
from .sessions import SessionManager, Session, Message
from .planreview import PlanReviewer, Plan, FileChange, ChangeType
from .agents import AgentManager, SubAgent, CodeAnalyzer, TestGenerator, DocGenerator
from .mcp import MCPServer, ToolDefinition, call_tool
from .indexer import CodeIndexer, CodeSymbol, CodeFile

__all__ = [
    # Existing...
    'ShellCompleter',
    'MetricsTracker',
    'StreamMetrics',
    'SessionManager',
    'Session',
    'Message',
    'PlanReviewer',
    'Plan',
    'FileChange',
    'ChangeType',
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
