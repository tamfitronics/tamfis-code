"""Read-only local filesystem tools for tamfis-code's offline/local chat mode.

Local mode has no server-side approval gate, safety classifier, or mutation-
ledger/audit trail (that infrastructure lives in tamgpt6's Remote Workspace
backend for good reason). Without it, offering anything that mutates state --
writing files, running shell commands -- would be a real safety regression,
not a feature. This module therefore wraps only the read-only subset of
MCPServer's native tools (tamfis_code/mcp.py): read_file, list_directory,
search_code, get_git_info. write_file/execute_command/browser are
intentionally never exposed here.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .mcp import MCPServer

READ_ONLY_TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a local file (read-only).",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "File path"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List the contents of a local directory (read-only).",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Directory path", "default": "."}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search local file contents using ripgrep (read-only).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search pattern"},
                    "path": {"type": "string", "description": "Search directory", "default": "."},
                    "file_pattern": {"type": "string", "description": "Optional glob filter, e.g. '*.py'"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_git_info",
            "description": "Read local git repository metadata: branch, latest commit, dirty status (read-only).",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Repository path", "default": "."}},
            },
        },
    },
]

_READ_ONLY_NAMES = {schema["function"]["name"] for schema in READ_ONLY_TOOL_SCHEMAS}


class LocalReadOnlyTools:
    """Dispatches only read_file/list_directory/search_code/get_git_info.

    Any other tool name (in particular write_file/execute_command/browser,
    which MCPServer also implements) is refused -- this is the enforcement
    point for local mode's read-only boundary, not just documentation of it.
    """

    def __init__(self) -> None:
        self._server = MCPServer()

    async def call(self, name: str, arguments: Dict[str, Any]) -> Any:
        if name not in _READ_ONLY_NAMES:
            raise ValueError(
                f"'{name}' is not available in local/offline mode -- only read-only tools "
                f"({', '.join(sorted(_READ_ONLY_NAMES))}) are exposed without a server-side "
                "approval gate and audit trail."
            )
        handler = getattr(self._server, f"_{name}")
        return await handler(**arguments)
