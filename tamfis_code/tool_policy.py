"""Task-aware minimum tool schema policy."""
from __future__ import annotations

from .routing import TaskProfile, TaskType

# Built-in tools proven non-mutating and available in read-only turns. A
# general shell is deliberately absent: advertising execute_command and then
# rejecting Python/read pipelines at dispatch produced repeated permission
# failures instead of useful inspection. Plugin/external MCP tools are filtered
# separately because their schemas alone do not prove read-only behavior.
#
# ask_user_question is available in every tool-calling task type --
# asking a clarifying question has no side effects on the workspace, and is
# exactly the kind of thing a read-only audit/plan task benefits from most
# (confirmed user request: the agent should be able to pause and ask instead
# of silently guessing when it's genuinely uncertain, e.g. a stated project
# type it can't otherwise verify).
# save_memory belongs here for the same reason as ask_user_question: it has
# no side effects on the workspace (it writes to CONFIG_DIR/memory, not any
# project file), so a read-only audit/inspect/plan turn should still be able
# to record something worth remembering next session.
# list_external_agent_sessions/read_external_agent_session are the same
# shape again: read-only local-disk lookups (another AI coding tool's own
# session files, never this project's), so a plain/inspect/audit turn can
# still act on "continue what Codex was doing" without needing edit-level
# tool access just to look that up.
# write_todos joins every non-plain list for the same reason: it mutates
# only the session's own task-state ledger (never a workspace file), and
# the visible step-by-step plan is exactly as valuable during a read-only
# audit/plan turn as during an edit turn -- the user watches progress
# either way. Claude Code's TodoWrite and Codex's to-do tracking are both
# available in every mode for the same reason.
READ_TOOLS = [
    "list_directory", "search_code", "find_references", "read_file", "read_archive",
    "get_git_info", "ask_user_question", "inspect_artifact", "save_memory",
    "list_external_agent_sessions", "read_external_agent_session",
    "write_todos",
]
# Keep ordinary read-only turns strictly read-only. A resumed machine-generated
# checkpoint gets the separately gated process-inspection surface in
# runner_local.py; exposing shell execution to every audit/question made the
# public tool contract unsafe and broke callers that rely on this boundary.
READ_ONLY_INSPECTION_TOOLS = [*READ_TOOLS]
EDIT_TOOLS = [
    *READ_TOOLS,
    "execute_command",
    "write_file",
    "edit_file",
    "extract_archive",
    "repackage_archive",
    "create_artifact",
]
EXECUTE_TOOLS = [*READ_TOOLS, "execute_command"]
GIT_TOOLS = ["get_git_info", "read_file", "search_code", "find_references", "execute_command", "ask_user_question"]
RESEARCH_TOOLS = [
    "web_search", "browser", "knowledge_base_search", "knowledge_base_index",
    "read_file", "read_archive", "search_code", "find_references", "ask_user_question",
]


def allowed_tools(profile: TaskProfile, *, read_only: bool) -> list[str]:
    if profile.is_plain_conversation:
        return []
    if read_only or profile.task_type in {TaskType.INSPECT, TaskType.AUDIT, TaskType.PLAN}:
        return READ_ONLY_INSPECTION_TOOLS
    if profile.task_type in {TaskType.EDIT, TaskType.DEBUG, TaskType.MIXED, TaskType.QUESTION}:
        return EDIT_TOOLS
    if profile.task_type in {TaskType.TEST, TaskType.EXECUTE}:
        return EXECUTE_TOOLS
    if profile.task_type == TaskType.GIT:
        return GIT_TOOLS
    if profile.task_type == TaskType.RESEARCH:
        return RESEARCH_TOOLS
    return READ_TOOLS
