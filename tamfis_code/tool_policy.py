"""Task-aware minimum tool schema policy."""
from __future__ import annotations

from .routing import TaskProfile, TaskType

# Available in every tool-calling task type, including read-only ones --
# asking a clarifying question has no side effects on the workspace, and is
# exactly the kind of thing a read-only audit/plan task benefits from most
# (confirmed user request: the agent should be able to pause and ask instead
# of silently guessing when it's genuinely uncertain, e.g. a stated project
# type it can't otherwise verify).
READ_TOOLS = [
    "list_directory", "search_code", "find_references", "read_file",
    "get_git_info", "ask_user_question", "execute_command",
]
EDIT_TOOLS = [
    *READ_TOOLS,
    "write_file",
    "edit_file",
    "extract_archive",
    "repackage_archive",
]
EXECUTE_TOOLS = [*READ_TOOLS, "execute_command"]
GIT_TOOLS = ["get_git_info", "read_file", "search_code", "find_references", "execute_command", "ask_user_question"]
RESEARCH_TOOLS = ["web_search", "browser", "read_file", "search_code", "find_references", "ask_user_question"]


def allowed_tools(profile: TaskProfile, *, read_only: bool) -> list[str]:
    if profile.is_plain_conversation:
        return []
    if read_only or profile.task_type in {TaskType.INSPECT, TaskType.AUDIT, TaskType.PLAN}:
        return READ_TOOLS
    if profile.task_type in {TaskType.EDIT, TaskType.DEBUG, TaskType.MIXED, TaskType.QUESTION}:
        return EDIT_TOOLS
    if profile.task_type in {TaskType.TEST, TaskType.EXECUTE}:
        return EXECUTE_TOOLS
    if profile.task_type == TaskType.GIT:
        return GIT_TOOLS
    if profile.task_type == TaskType.RESEARCH:
        return RESEARCH_TOOLS
    return READ_TOOLS
