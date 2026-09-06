"""Deterministic task classification and capability-aware provider routing."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
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


class ComplexityLevel(str, Enum):
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


_COMPLEXITY_RANK = {
    # Legacy values remain accepted for callers/tests constructing profiles
    # directly while runtime classification emits the five-level vocabulary.
    "low": 1,
    "medium": 2,
    "high": 3,
    ComplexityLevel.TRIVIAL.value: 0,
    ComplexityLevel.SIMPLE.value: 1,
    ComplexityLevel.MODERATE.value: 2,
    ComplexityLevel.COMPLEX.value: 3,
    ComplexityLevel.VERY_COMPLEX.value: 4,
}


def complexity_at_least(value: str, minimum: ComplexityLevel | str) -> bool:
    floor = minimum.value if isinstance(minimum, ComplexityLevel) else str(minimum)
    return _COMPLEXITY_RANK.get(str(value), 0) >= _COMPLEXITY_RANK.get(floor, 0)


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

_EXPLICIT_READ_ONLY_RE = re.compile(
    r"(?:\bread[ -]?only\b|\bno\s+(?:file\s+)?edits?\b|"
    r"\bdo\s+not\s+(?:edit|modify|change|write|patch)\b|"
    r"\bdon['’]?t\s+(?:edit|modify|change|write|patch)\b|"
    r"\bwithout\s+(?:editing|modifying|changing|writing|patching)\b|"
    r"\brecommendations?\s+only\b)",
    re.IGNORECASE,
)

# Checkpoint recovery appends the newest instruction using this marker (see
# runner_local.py). A prior objective may legitimately say "no file edits",
# while the user later changes direction with "fix it". Treating the combined
# text as one timeless instruction made that stale restriction permanently
# sticky and stripped write/test tools from every resumed turn.
_ADDITIONAL_CONTEXT_MARKER = "additional user context:"
_EXPLICIT_MUTATION_RE = re.compile(
    r"(?:\bfix\b|\brepair\b|\bpatch\b|\bimplement\b|\bapply\b.{0,24}\bfix\b|"
    r"\bmake\b.{0,24}\bchanges?\b|\bedit\b|\bmodify\b|\brewrite\b|"
    r"\bwrite\b.{0,24}\bfiles?\b|\bcommit\b|\bpush\b|\brestart\b|\binstall\b)",
    re.IGNORECASE,
)

_FILE_REFERENCE_RE = re.compile(
    r"(?<![\w.-])[\w@+-]+(?:/[\w@+.-]+)*\.[A-Za-z0-9]{1,10}\b"
)


def estimate_complexity(text: str, task_type: TaskType) -> ComplexityLevel:
    """Estimate orchestration depth from task shape, not one magic word.

    The score combines independent observable signals. A user saying
    "complex" does not itself escalate a task, while a multi-outcome change
    spanning API, persistence, frontend and tests does even without that word.
    """

    raw = (text or "").strip()
    lowered = raw.casefold()
    if not raw or task_type == TaskType.CONVERSATION:
        return ComplexityLevel.TRIVIAL

    base = {
        TaskType.QUESTION: 1,
        TaskType.INSPECT: 1,
        TaskType.RESEARCH: 2,
        TaskType.TEST: 2,
        TaskType.EXECUTE: 2,
        TaskType.GIT: 2,
        TaskType.PLAN: 2,
        TaskType.EDIT: 2,
        TaskType.DEBUG: 2,
        TaskType.AUDIT: 3,
        TaskType.MIXED: 3,
    }.get(task_type, 1)
    score = base

    file_refs = set(_FILE_REFERENCE_RE.findall(raw))
    if len(file_refs) >= 2:
        score += 1
    if len(file_refs) >= 5:
        score += 1

    outcome_signals = len(re.findall(
        r"(?:^|[\n;.]|\bthen\b|\band\b)\s*(?:find|trace|inspect|fix|add|remove|"
        r"implement|update|test|verify|build|deploy|restart|review|document)\b",
        lowered,
    ))
    if outcome_signals >= 3:
        score += 1
    if outcome_signals >= 6:
        score += 1

    component_hits = sum(
        bool(re.search(rf"\b{component}\b", lowered))
        for component in (
            "frontend", "backend", "api", "database", "persistence",
            "worker", "cli", "gateway", "orchestrator", "provider",
        )
    )
    if component_hits >= 2:
        score += 1
    if component_hits >= 4:
        score += 1

    independent_signals = (
        bool(re.search(r"\b(?:whole|entire|repository-wide|end-to-end|cross-file|cross-component)\b", lowered)),
        bool(re.search(r"\b(?:migration|schema|backward compatibility|public api|architecture)\b", lowered)),
        bool(re.search(r"\b(?:intermittent|race condition|unknown root cause|production logs|reproduce)\b", lowered)),
        bool(re.search(r"\b(?:external api|third-party|webhook|oauth|deployment|systemd)\b", lowered)),
        bool(re.search(r"\b(?:unit tests?|integration tests?|typecheck|lint|build)\b", lowered)),
        bool(re.search(r"\b(?:retry|replan|repair loop|independent review|subagents?)\b", lowered)),
    )
    score += sum(independent_signals)
    if len(raw) >= 600:
        score += 1
    if len(raw) >= 1800:
        score += 1

    if score <= 1:
        return ComplexityLevel.SIMPLE
    if score <= 3:
        return ComplexityLevel.MODERATE
    if score <= 6:
        return ComplexityLevel.COMPLEX
    return ComplexityLevel.VERY_COMPLEX


def is_explicit_read_only_request(text: str) -> bool:
    """Return whether the user explicitly prohibited repository mutation.

    This is shared by task routing and downstream honesty/completion guards;
    keeping the interpretation in one place prevents a read-only inspection
    from later being reclassified as an unfinished edit merely because it
    mentions a fix report or says "no file edit".
    """
    value = text or ""
    lowered = value.lower()
    if _ADDITIONAL_CONTEXT_MARKER in lowered:
        latest = lowered.rsplit(_ADDITIONAL_CONTEXT_MARKER, 1)[-1]
        # The latest instruction is authoritative when it explicitly asks
        # for mutation. A bare "continue" still inherits the recovered
        # objective's read-only constraint.
        if _EXPLICIT_MUTATION_RE.search(latest) and not _EXPLICIT_READ_ONLY_RE.search(latest):
            return False
    return _EXPLICIT_READ_ONLY_RE.search(value) is not None


def classify_task(text: str, *, read_only: bool = False) -> TaskProfile:
    raw = (text or "").strip().lower()
    def profile(
        task_type: TaskType,
        requires_tools: bool,
        requires_repository_context: bool,
        requires_long_context: bool,
        requires_validation: bool,
        preferred_quality_tier: str,
    ) -> TaskProfile:
        return TaskProfile(
            task_type,
            estimate_complexity(text, task_type).value,
            requires_tools,
            requires_repository_context,
            requires_long_context,
            requires_validation,
            preferred_quality_tier,
        )
    if not raw or raw in _GREETINGS or raw.startswith(("who are you", "what can you do", "tell me about yourself")):
        return profile(TaskType.CONVERSATION, False, False, False, False, "economy")

    def has(words: Iterable[str]) -> bool:
        return any(word in raw for word in words)

    if has(_CLOSURE_SIGNALS):
        return profile(TaskType.CONVERSATION, False, False, False, False, "economy")
    # A report/status request often names an artefact such as
    # ATTACHMENT_PIPELINE_FIX_PROGRESS.md. The old substring-first DEBUG
    # check treated the incidental "fix" in that filename as an instruction
    # to mutate the repository despite an explicit "no file edit" constraint.
    if is_explicit_read_only_request(raw):
        if has(("audit", "entire stack", "whole repository", "whole repo", "end-to-end", "end to end")):
            return profile(TaskType.AUDIT, True, True, True, False, "frontier")
        return profile(TaskType.INSPECT, True, True, False, False, "high")
    if has(("audit", "entire stack", "whole repository", "whole repo", "end-to-end", "end to end")):
        return profile(TaskType.AUDIT, True, True, True, True, "frontier")
    if has(("debug", "fix", "repair", "bug", "traceback", "exception", "failing")):
        return profile(TaskType.DEBUG, True, True, True, True, "frontier")
    if has(("implement", "edit", "modify", "refactor", "rewrite", "add ", "create ", "remove ", "delete ", "patch")) and not read_only:
        return profile(TaskType.EDIT, True, True, True, True, "frontier")
    if has(("pytest", "test suite", "run tests", "fix tests", "lint", "typecheck", "type check")):
        return profile(TaskType.TEST, True, True, False, True, "high")
    if has(("git ", "commit", "push", "pull request", "branch", "merge")):
        return profile(TaskType.GIT, True, True, False, True, "high")
    if has(("run ", "execute ", "install ", "build ", "restart ", "systemctl")):
        return profile(TaskType.EXECUTE, True, True, False, True, "high")
    if has(("plan", "proposal", "roadmap", "architecture")):
        return profile(TaskType.PLAN, True, True, True, False, "high")
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
        return profile(TaskType.RESEARCH, True, False, False, False, "high")
    if has(("inspect", "analyse", "analyze", "review", "search", "find", "check ", "read file", "repository", "codebase")):
        return profile(TaskType.INSPECT, True, True, False, False, "high")
    # requires_tools/requires_validation must stay False here: a generic
    # question is answerable without tool evidence. `read_only` only governs
    # which tools are *offered* (see tool_policy.py) -- it used to also be
    # passed as requires_tools, which made validate_completion fail every
    # plain chat-mode question (confirmed live: "reply with exactly PONG"
    # got flagged "Validation incomplete" with no explanation) purely
    # because no tool call was made, even though none was ever needed.
    return profile(TaskType.QUESTION, False, read_only, False, False, "balanced")

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
