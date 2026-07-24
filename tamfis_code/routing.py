"""Deterministic task classification used by the standalone runtime."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class TaskType(str, Enum):
    CONVERSATION="conversation"; QUESTION="question"; INSPECT="inspect"; AUDIT="audit"
    PLAN="plan"; EDIT="edit"; DEBUG="debug"; TEST="test"; EXECUTE="execute"
    GIT="git"; RESEARCH="research"; MIXED="mixed"


@dataclass(frozen=True)
class TaskProfile:
    task_type: TaskType
    complexity: str
    requires_tools: bool
    requires_repository_context: bool
    requires_validation: bool
    is_plain_conversation: bool = False


def classify_task(text: str, *, read_only: bool = False) -> TaskProfile:
    value=(text or "").casefold()
    def has(*words: str) -> bool: return any(word in value for word in words)
    if has("audit", "full stack", "entire stack", "review repository", "inspect repository"):
        kind=TaskType.AUDIT
    elif has("debug", "fix bug", "fix error", "traceback", "failing", "broken"):
        kind=TaskType.DEBUG
    elif has("edit", "change", "implement", "add ", "remove ", "rewrite", "refactor", "rebuild", "remodel"):
        kind=TaskType.EDIT
    elif has("test", "pytest", "vitest", "unit test"):
        kind=TaskType.TEST
    elif has("plan", "design", "architecture"):
        kind=TaskType.PLAN
    elif has("git ", "commit", "push", "pull request", "branch"):
        kind=TaskType.GIT
    elif has("research", "web search", "look up"):
        kind=TaskType.RESEARCH
    elif has("run ", "execute ", "restart", "build "):
        kind=TaskType.EXECUTE
    elif has("inspect", "read ", "show ", "find ", "locate", "search"):
        kind=TaskType.INSPECT
    elif value.strip() in {"hi","hello","hey","thanks","thank you"}:
        kind=TaskType.CONVERSATION
    else:
        kind=TaskType.QUESTION
    if read_only and kind in {TaskType.EDIT,TaskType.DEBUG,TaskType.EXECUTE}:
        kind=TaskType.INSPECT
    plain=kind is TaskType.CONVERSATION
    tools=not plain
    repo=kind not in {TaskType.CONVERSATION,TaskType.RESEARCH}
    validation=kind in {TaskType.AUDIT,TaskType.EDIT,TaskType.DEBUG,TaskType.TEST,TaskType.EXECUTE,TaskType.MIXED}
    complexity="high" if kind in {TaskType.AUDIT,TaskType.EDIT,TaskType.DEBUG,TaskType.MIXED} else "medium" if tools else "low"
    return TaskProfile(kind, complexity, tools, repo, validation, plain)
