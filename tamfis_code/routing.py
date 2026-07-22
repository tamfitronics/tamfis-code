"""Deterministic task classification and capability-aware provider routing."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class TaskType(str, Enum):
    CONVERSATION = "conversation"
    QUESTION = "question"
    INSPECT = "inspect"
    AUDIT = "audit"
    PLAN = "plan"
    EDIT = "edit"
    DEBUG = "debug"
    TEST = "test"
    EXECUTE = "execute"
    GIT = "git"
    RESEARCH = "research"
    MIXED = "mixed"


@dataclass(frozen=True)
class TaskProfile:
    task_type: TaskType
    complexity: str
    requires_tools: bool
    requires_repository_context: bool
    requires_long_context: bool
    requires_validation: bool
    preferred_quality_tier: str

    @property
    def is_plain_conversation(self) -> bool:
        return self.task_type == TaskType.CONVERSATION


_GREETINGS = {
    "hi", "hello", "hey", "hi there", "hello there", "good morning",
    "good afternoon", "good evening", "how are you", "how are you?",
    "thanks", "thank you",
}

# Confirmation/closure feedback about work already done ("yeah that bug is
# fixed now, thanks") contains the exact same words ("fix", "bug") the DEBUG
# check below keys on -- "fix" is a literal substring of "fixed", so a plain
# substring match on `has(("fix", "bug", ...))` misclassified pure
# acknowledgment as a fresh DEBUG task. Confirmed live: told to
# `allowed_tools()` and a real edit turn, the model re-applied the same
# already-shipped fix to a file that needed no further changes. Checked
# before the DEBUG/EDIT keyword checks so closure language always wins over
# an incidental "fix"/"bug" mention, on the theory that redundantly
# re-touching an already-fixed file is worse than occasionally missing a
# genuinely new issue mentioned in the same breath (which still surfaces
# normally in the user's next message).
_CLOSURE_SIGNALS = (
    "already fixed", "already resolved", "already working", "already done",
    "is fixed", "it's fixed", "its fixed", "that's fixed", "thats fixed",
    "bug is fixed", "issue is fixed", "confirmed fixed", "confirmed working",
    "works now", "working now", "resolved now", "no need to", "don't need to",
    "dont need to", "no further action", "no more changes needed",
    "nothing more to do", "all good now", "that fixed it", "that solved it",
    "that worked", "no need for further",
)


def classify_task(text: str, *, read_only: bool = False) -> TaskProfile:
    raw = (text or "").strip().lower()
    if not raw or raw in _GREETINGS or raw.startswith(("who are you", "what can you do", "tell me about yourself")):
        return TaskProfile(TaskType.CONVERSATION, "low", False, False, False, False, "economy")

    def has(words: Iterable[str]) -> bool:
        return any(word in raw for word in words)

    if has(_CLOSURE_SIGNALS):
        return TaskProfile(TaskType.CONVERSATION, "low", False, False, False, False, "economy")
    if has(("audit", "entire stack", "whole repository", "whole repo", "end-to-end", "end to end")):
        return TaskProfile(TaskType.AUDIT, "high", True, True, True, True, "frontier")
    if has(("debug", "fix", "repair", "bug", "traceback", "exception", "failing")):
        return TaskProfile(TaskType.DEBUG, "high", True, True, True, True, "frontier")
    if has(("implement", "edit", "modify", "refactor", "rewrite", "add ", "create ", "remove ", "delete ", "patch")) and not read_only:
        return TaskProfile(TaskType.EDIT, "high", True, True, True, True, "frontier")
    if has(("pytest", "test suite", "run tests", "fix tests", "lint", "typecheck", "type check")):
        return TaskProfile(TaskType.TEST, "medium", True, True, False, True, "high")
    if has(("git ", "commit", "push", "pull request", "branch", "merge")):
        return TaskProfile(TaskType.GIT, "medium", True, True, False, True, "high")
    if has(("run ", "execute ", "install ", "build ", "restart ", "systemctl")):
        return TaskProfile(TaskType.EXECUTE, "medium", True, True, False, True, "high")
    if has(("plan", "proposal", "roadmap", "architecture")):
        return TaskProfile(TaskType.PLAN, "medium", True, True, True, False, "high")
    # Checked before INSPECT's broad "search"/"find" catch below, which would
    # otherwise swallow explicitly web-directed phrasing first -- this branch
    # only fires on cues naming the web/internet/a browser, not on ordinary
    # in-repository "search"/"find" requests. TaskType.RESEARCH previously had
    # no classify_task branch at all, so RESEARCH_TOOLS (browser, web_search)
    # was unreachable dead code -- the model could never actually be offered
    # either tool through the normal agent loop, only via the direct
    # `tamfis-code tools call` CLI.
    if has((
        "search the web", "search online", "search the internet", "web search",
        "look up online", "look online", "browse the web", "on the web for",
        "google ", "current price", "latest news", "recent news",
        "what's the latest", "whats the latest", "up to date information",
        "up-to-date information", "current events", "today's news",
    )):
        return TaskProfile(TaskType.RESEARCH, "medium", True, False, False, False, "high")
    if has(("inspect", "analyse", "analyze", "review", "search", "find", "check ", "read file", "repository", "codebase")):
        return TaskProfile(TaskType.INSPECT, "medium", True, True, False, False, "high")
    # requires_tools/requires_validation must stay False here: a generic
    # question is answerable without tool evidence. `read_only` only governs
    # which tools are *offered* (see tool_policy.py) -- it used to also be
    # passed as requires_tools, which made validate_completion fail every
    # plain chat-mode question (confirmed live: "reply with exactly PONG"
    # got flagged "Validation incomplete" with no explanation) purely
    # because no tool call was made, even though none was ever needed.
    return TaskProfile(TaskType.QUESTION, "low", False, read_only, False, False, "balanced")

class Router:
    """Compatibility facade for deterministic classification and provider selection."""

    def __init__(self):
        from .providers import ProviderManager
        self.provider_manager = ProviderManager()

    def select_provider(self, task_profile: TaskProfile, quality_mode: str = "balanced", explicit_provider: str | None = None):
        from .providers import ProviderType

        if explicit_provider:
            provider_type = ProviderType(explicit_provider)
            config = self.provider_manager.PROVIDERS[provider_type]
            return {
                "provider": provider_type.value,
                "model": self.provider_manager.select_model(config, task_profile),
                "selection_reason": f"explicit user selection: {explicit_provider}",
                "capabilities": {
                    "coding_quality": config.coding_quality,
                    "tool_calling": config.tool_calling,
                    "long_context": config.long_context,
                    "context_window": config.context_window,
                },
            }

        resolved, config = self.provider_manager.resolve_route(
            ProviderType.AUTO,
            task_profile,
            quality_mode=quality_mode,
        )
        return {
            "provider": resolved.value,
            "model": self.provider_manager.select_model(config, task_profile),
            "selection_reason": (
                f"capability routing: {config.name} "
                f"(priority {config.priority}, coding_quality {config.coding_quality})"
            ),
            "capabilities": {
                "coding_quality": config.coding_quality,
                "tool_calling": config.tool_calling,
                "long_context": config.long_context,
                "context_window": config.context_window,
            },
        }

    def get_provider_status(self):
        return self.provider_manager.list_available_providers()
