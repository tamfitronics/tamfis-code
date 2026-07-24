"""The standalone agent loop: calls an LLM provider directly (via
providers.py's ProviderManager) and runs its own tool-calling loop locally,
with no TamfisGPT Remote Workspace backend involved at all.

Generalizes local_chat.py's run_local_turn (which already proved the basic
send-tools -> parse tool_calls -> execute -> append role:"tool" -> resend
pattern works against HF/NVIDIA NIM/OpenRouter) into the primary,
full-capability loop:
  - streaming + tool-calling combined (local_chat.py's run_local_turn has
    tools but no streaming; stream_local_turn streams but has no tools)
  - the full tool set via mcp.py's MCPServer (read/write/edit/execute/etc),
    not just the four read-only tools -- gated per call through safety.py's
    risk classifier and runner.py's existing resolve_approval_decision
  - an open-ended round loop (a high safety-valve cap, not local_chat.py's
    hard MAX_TOOL_ROUNDS=5) with real termination conditions

Emits the same event-dict shape StreamRenderer.handle_event already expects
(assistant_delta/tool_call_requested/tool_output/file_mutation/
approval_required/ai_task_completed/ai_task_failed) so render.py needs no
changes to work with a local loop instead of remote SSE events.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import shlex
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from rich.console import Console

from . import evidence as evidence_store
from . import state as local_state
from .config import Config
from .hooks import load_hooks, run_tool_hooks
from .mcp import MCPServer
from .providers import ProviderManager, ProviderType, reasoning_effort_capable
from .render import StreamRenderer, resume_live_if_active, suspend_live_if_active
from .routing import classify_task
from .orchestrator import (
    AgentOrchestrator,
    ToolEnvelope,
    build_reasoning_plan_prompt,
    parse_reasoning_plan,
    should_plan,
)
from .tool_policy import allowed_tools
from .provider_protocols import normalize_stream_chunk
from .runner import TaskOutcome, resolve_approval_decision_async
from .safety import READ_ONLY_TOOLS, _unified_diff, classify_tool_call_risk, redact_secrets
from .workspace import classify_root

# A safety-valve ceiling, not a target -- local_chat.py's MAX_TOOL_ROUNDS=5
# was appropriate for a read-only Q&A loop; a real coding-agent task
# legitimately needs many more tool calls (read several files, make several
# edits, run tests, iterate). This exists only to guarantee termination if
# something is genuinely stuck in a loop, not to cap normal work.
MAX_AGENT_ROUNDS = 40

# High reasoning is valuable for deliberate architecture work but makes the
# interactive terminal feel stalled on ordinary audits/edits. Medium is the
# responsive default; set TAMFIS_CODE_REASONING_EFFORT=high when depth matters
# more than first-token latency.
DEFAULT_REASONING_EFFORT = os.environ.get("TAMFIS_CODE_REASONING_EFFORT", "medium").strip().lower()


def _reasoning_effort(provider: ProviderType, model: str) -> Optional[str]:
    if not reasoning_effort_capable(provider, model):
        return None
    return DEFAULT_REASONING_EFFORT if DEFAULT_REASONING_EFFORT in {"low", "medium", "high"} else "medium"

# If the model requests the exact same tool call(s) (name + arguments,
# unordered) this many rounds in a row, stop rather than let it spin: a
# weaker model asked to check on something that never changes (a health
# endpoint that's never up, a container ID it never filled in) will
# otherwise repeat identically until MAX_AGENT_ROUNDS or a context-window
# error ends it, burning a lot of time/tokens along the way. 2 tolerates a
# legitimate one-off retry; 3 identical rounds in a row is not progress.
MAX_CONSECUTIVE_IDENTICAL_ROUNDS = 2

# Once a loop is detected (either the identical-repeat or the cycling
# check below), give the model exactly this many chances to self-correct
# -- refuse the repeated call(s), remind it what to do instead, and let it
# try again with tools still available -- before giving up on tool use
# entirely and forcing one final tools-disabled synthesis of whatever it's
# actually found. Matches this codebase's existing bounded-single-extra-
# pass convention (MAX_CONTEXT_ROLLOVERS_PER_TURN, MAX_EMPTY_CONTINUATION_
# RETRIES, MAX_TRUNCATION_CONTINUATIONS) -- one real second chance, not an
# open-ended retry budget that just delays the same dead end.
MAX_LOOP_NUDGE_RETRIES = 1

# Weak OpenAI-compatible models sometimes write a sequence of promises
# ("Let me check...", "Now let me read...") while issuing zero registered
# tool calls.  That prose is not a completed repository task.  Give each
# route one explicit correction before AUTO mode moves to another provider.
MAX_NARRATED_TOOL_RETRIES_PER_PROVIDER = 1
NARRATED_TOOL_CORRECTION = (
    "Your previous response only described future repository actions, but issued no "
    "registered tool call. Do not narrate what you are about to inspect. Call the "
    "appropriate registered tool now, wait for its result, and continue the task. "
    "Only provide a final answer after the requested inspection or action actually ran."
)

# Keep the user-facing completion readable even when a provider streams a
# long audit. Plans and live progress already have their own durable renderer
# panels; the model must not recreate those as a second, drifting bullet list.
FINAL_RESPONSE_FORMAT_INSTRUCTION = (
    "When you finish, write a concise evidence-backed response using exactly these "
    "sections when applicable: Summary, Changes, Verification, and Remaining issues. "
    "Use one short bullet per concrete fact under a section; do not emit nested or "
    "unrelated bullet lists, do not repeat the execution plan, and do not claim a "
    "tool ran unless a real tool result appears in the conversation. If no files "
    "changed, say so plainly under Summary."
)

# Same one-chance-then-fallback shape as narrated tool intent, for the
# distinct failure of giving up outright instead of narrating.
MAX_CAPITULATION_RETRIES_PER_PROVIDER = 1
CAPITULATION_CORRECTION = (
    "Your previous response gave up without issuing a registered tool call, citing an "
    "unclear next step. You were asked to proceed autonomously without further "
    "confirmation. Investigate the repository yourself to find something concrete to "
    "act on -- for example run the test suite or a linter (execute_command), search for "
    "TODO/FIXME/error markers (search_code), inspect recent changes (get_git_info), or "
    "list the working directory (list_directory) -- then act on what you find. Do not "
    "ask the user to clarify the request; make a reasonable judgment call and continue."
)

# Same one-chance-then-fallback shape as narrated tool intent and
# capitulation, for the distinct failure of fabricating a past-tense tool
# result or tool-level refusal instead of either promising to act or
# giving up.
MAX_FABRICATED_RESULT_RETRIES_PER_PROVIDER = 1
FABRICATED_RESULT_CORRECTION = (
    "Your previous response reported a tool result, tool error, or access/permission "
    "restriction, but no registered tool call was actually issued this turn -- that report "
    "was fabricated. Do not describe what a tool found, returned, or denied unless you just "
    "called it and are reporting its real result. Call the appropriate registered tool now "
    "and report only what it actually returns."
)

RESUME_EXECUTION_INSTRUCTION = (
    "You are resuming an unfinished engineering task from durable state. The user's "
    "continuation directive is authoritative: do not declare that there is no clear next "
    "step and do not ask for confirmation or clarification. Recover the original objective, "
    "completed actions, unresolved validation findings, and the latest real tool evidence. "
    "Choose the first concrete unresolved action and call a registered inspection, execution, "
    "editing, or validation tool now. A prose-only statement that the task is stuck is not a "
    "valid completion. Report a blocker only after a real tool result proves it."
)

_AUTONOMOUS_EXECUTION_RE = re.compile(
    r"\b(?:continue|resume|proceed|carry\s+on|go\s+ahead)\b.*"
    r"(?:fix\s+everything|until\s+(?:it(?:'s| is)?\s+)?fix(?:ed)?|do\s+everything|finish\s+everything)",
    re.IGNORECASE | re.DOTALL,
)
_NO_CONFIRMATION_RE = re.compile(
    r"\b(?:do\s*not|don't|dont|without)\s+(?:ask(?:ing)?\s+(?:me\s+)?(?:for\s+)?)?"
    r"(?:confirmation|approval|permission)\b|\bjust\s+go\s+ahead\b",
    re.IGNORECASE,
)

def _requests_autonomous_execution(text: str) -> bool:
    return bool(_AUTONOMOUS_EXECUTION_RE.search(text or ""))

def _requests_no_confirmation(text: str) -> bool:
    return bool(_NO_CONFIRMATION_RE.search(text or ""))

# Same 4-chars-per-token heuristic render.py uses for its live token counter
# (_CHARS_PER_TOKEN_ESTIMATE) -- good enough to budget against a context
# window, not meant to match a real tokenizer exactly.
_CHARS_PER_TOKEN_ESTIMATE = 4
MAX_TOKENS_PER_REQUEST = 4096
# Leave headroom below the provider's stated context_window: it's a
# conservative estimate already (see providers.py), and this estimate's own
# char/token ratio is approximate too.
_CONTEXT_SAFETY_MARGIN = 0.9


# Workspace scoping keeps broad requests such as "audit the full stack" from
# degenerating into a recursive search of an entire parent directory.  The
# resolver deliberately inspects only the workspace root's immediate children,
# identifies real project roots from lightweight markers, and then selects the
# roots named or implied by the user's objective.
_PROJECT_MARKERS = (
    ".git",
    "pyproject.toml",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "requirements.txt",
)
_TAMFIS_STACK_ROOTS = (
    "tamfis-code",
)
_SCOPE_PATH_TOOLS = {
    "read_file",
    "write_file",
    "edit_file",
    "list_directory",
    "search_code",
    "get_git_info",
}


def _is_project_root(path: Path) -> bool:
    try:
        return path.is_dir() and any((path / marker).exists() for marker in _PROJECT_MARKERS)
    except OSError:
        return False


def _is_active_root(path: Path) -> bool:
    """A project root that isn't a dependency/legacy/generated/archived
    directory -- see workspace.classify_root. Used only for *implicit*
    (heuristic) selection; a root the user names explicitly is honored
    regardless of its classification (see _detect_workspace_scope)."""
    return _is_project_root(path) and classify_root(path) == "active"


# Phrases that mark whatever project name follows as off-limits for this
# turn (e.g. "audit the iOS stack, do not touch tamfis-code"). Matched
# against a bounded window of text right after the trigger, not the whole
# objective, so an unrelated later mention of the same name elsewhere in a
# long objective doesn't retroactively un-exclude it.
_EXCLUSION_TRIGGERS = (
    "do not touch", "don't touch", "dont touch",
    "do not modify", "don't modify", "dont modify",
    "do not change", "don't change", "dont change",
    "without touching", "without modifying", "without changing",
    "avoid touching", "avoid modifying", "leave alone", "leave untouched",
    "excluding", "except for", "except",
)
_EXCLUSION_WINDOW_CHARS = 60


def _excluded_root_names(lowered_objective: str, candidate_names: set[str]) -> set[str]:
    """Names from `candidate_names` that the objective explicitly marks as
    off-limits (see _EXCLUSION_TRIGGERS). Confirmed live: an objective like
    "audit the iOS stack, do not touch tamfis-code" was silently ignored --
    the launch directory (tamfis-code) was scoped and read anyway, because
    nothing anywhere parsed this kind of negative instruction at all."""
    excluded: set[str] = set()
    for trigger in _EXCLUSION_TRIGGERS:
        search_from = 0
        while True:
            pos = lowered_objective.find(trigger, search_from)
            if pos == -1:
                break
            window = lowered_objective[pos: pos + len(trigger) + _EXCLUSION_WINDOW_CHARS]
            for name in candidate_names:
                if name and name in window:
                    excluded.add(name)
            search_from = pos + len(trigger)
    return excluded


def _objective_excluded_names(workspace_root: str, objective: str) -> set[str]:
    """Diagnostic-only re-derivation of what _detect_workspace_scope treated
    as excluded, so the "Focused workspace scope" diagnostic can confirm the
    exclusion was actually honored instead of leaving it silent."""
    root = Path(workspace_root).expanduser().resolve()
    lowered = (objective or "").lower()
    try:
        by_name = {child.name.lower() for child in root.iterdir() if child.is_dir()}
    except OSError:
        by_name = set()
    try:
        siblings = {
            child.name.lower() for child in root.parent.iterdir()
            if child.is_dir() and child.resolve() != root
        } if root.parent != root else set()
    except OSError:
        siblings = set()
    return _excluded_root_names(lowered, by_name | siblings | {root.name.lower()})


def _detect_workspace_scope(workspace_root: str, objective: str) -> list[Path]:
    """Resolve the smallest useful set of project roots for this turn.

    Exact directory names mentioned by the user win. Explicit absolute paths
    in the objective are also authoritative task-local scope expansions. This
    matters when the CLI is launched from a parent/admin checkout but the
    user names a sibling project such as `/home/tamfisseo/package.json`.
    A request containing
    "stack" under a configured workspace selects the relevant project roots.
    Otherwise, when the workspace itself is a project, it remains the scope;
    when it is merely a parent directory, only immediate child projects are
    considered -- never arbitrary siblings elsewhere on disk. A root the
    objective explicitly excludes (see _excluded_root_names) is never
    selected, including as the launch-directory fallback -- sibling project
    roots are considered instead so tamfis-code itself running from *inside*
    one of the canonical stacks can still route to an excluded stack's
    siblings (its own children obviously can't contain them).
    """
    root = Path(workspace_root).expanduser().resolve()
    lowered = (objective or "").lower()

    # A clear restrictive directive takes precedence over illustrative,
    # historical or reproduction paths elsewhere in the objective.
    restrictive_patterns = (
        r"(?i)\boperate\s+only\s+(?:inside|within|on)\s+"
        r"(?P<path>/(?:[A-Za-z0-9._~+\-]+/)*[A-Za-z0-9._~+\-]+)",
        r"(?i)\buse\s+only\s+"
        r"(?P<path>/(?:[A-Za-z0-9._~+\-]+/)*[A-Za-z0-9._~+\-]+)",
        r"(?i)\bactive\s+(?:repository|root|workspace)\s*(?:is|:)\s*"
        r"(?P<path>/(?:[A-Za-z0-9._~+\-]+/)*[A-Za-z0-9._~+\-]+)",
    )

    restrictive_roots: list[Path] = []
    restrictive_seen: set[str] = set()

    # First collect paths appearing directly in restrictive phrases.
    for pattern in restrictive_patterns:
        for match in re.finditer(pattern, objective or ""):
            candidate = Path(match.group("path").rstrip(".,;:)]}"))
            try:
                candidate = candidate.resolve()
                if candidate.is_file():
                    candidate = candidate.parent
                if candidate.is_dir() and _is_project_root(candidate):
                    key = str(candidate)
                    if key not in restrictive_seen:
                        restrictive_seen.add(key)
                        restrictive_roots.append(candidate)
            except OSError:
                continue

    # Support list-form directives such as:
    #
    # Operate only inside:
    # - /repo/backend
    # - /repo/frontend
    #
    # Once a restrictive heading is found, collect consecutive absolute-path
    # list entries until ordinary prose resumes. This allows intentional
    # multi-repository tasks without treating later reproduction/example
    # paths as authorised roots.
    lines = (objective or "").splitlines()
    collecting_restrictive_list = False

    for raw_line in lines:
        stripped = raw_line.strip()

        if re.search(
            r"(?i)\b(?:operate|work|act|make changes)\s+only\s+"
            r"(?:inside|within|on)\s*:?\s*$",
            stripped,
        ):
            collecting_restrictive_list = True
            continue

        if not collecting_restrictive_list:
            continue

        match = re.match(
            r"^(?:[-*+]\s+|\d+[.)]\s+)?"
            r"(?P<path>/(?:[A-Za-z0-9._~+\-]+/)*[A-Za-z0-9._~+\-]+)"
            r"\s*[,;.]?\s*$",
            stripped,
        )

        if match:
            candidate = Path(match.group("path")).expanduser()
            try:
                candidate = candidate.resolve()
                if candidate.is_file():
                    candidate = candidate.parent
                if candidate.is_dir() and _is_project_root(candidate):
                    key = str(candidate)
                    if key not in restrictive_seen:
                        restrictive_seen.add(key)
                        restrictive_roots.append(candidate)
            except OSError:
                continue
            continue

        # Blank lines are allowed within a list. Any other prose ends the
        # restrictive-list section so later examples do not expand scope.
        if stripped:
            collecting_restrictive_list = False

    # Also support multiple project roots written on one line:
    #
    #   Operate only inside /repo/backend and /repo/frontend.
    #
    # The narrow phrase regex above intentionally captures one path. Once a
    # restrictive directive is established, collect every existing absolute
    # project path from that same sentence only. Later example/reproduction
    # paths in other sentences must not expand the authorised scope.
    restrictive_trigger = re.compile(
        r"(?i)\b(?:"
        r"operate\s+only\s+(?:inside|within|on)|"
        r"work\s+only\s+(?:inside|within|on)|"
        r"act\s+only\s+(?:inside|within|on)|"
        r"make\s+changes\s+only\s+(?:inside|within|on)|"
        r"use\s+only"
        r")\b"
    )
    absolute_project_path = re.compile(
        r"(?<![\w.-])/(?:[A-Za-z0-9._~+\-]+/)*[A-Za-z0-9._~+\-]+"
    )

    for sentence in re.split(r"(?<=[.!?])\s+|\n+", objective or ""):
        if not restrictive_trigger.search(sentence):
            continue

        for raw_path in absolute_project_path.findall(sentence):
            candidate = Path(raw_path.rstrip(".,;:)]}")).expanduser()
            try:
                candidate = candidate.resolve()
                if candidate.is_file():
                    candidate = candidate.parent
                if candidate.is_dir() and _is_project_root(candidate):
                    key = str(candidate)
                    if key not in restrictive_seen:
                        restrictive_seen.add(key)
                        restrictive_roots.append(candidate)
            except OSError:
                continue

    if restrictive_roots:
        return restrictive_roots

    # An explicitly named existing absolute project path can intentionally
    # expand task scope. The existing workspace-approval layer remains
    # responsible for authorising access outside the launch workspace.
    explicit_roots: list[Path] = []

    for raw_path in re.findall(
        r"(?<![\w.-])/(?:[A-Za-z0-9._~+\-]+/)*[A-Za-z0-9._~+\-]+",
        objective or "",
    ):
        candidate = Path(raw_path.rstrip(".,;:)]}"))
        try:
            candidate = candidate.resolve()
            if candidate.is_file():
                candidate = candidate.parent
            if candidate.is_dir() and _is_project_root(candidate):
                explicit_roots.append(candidate)
        except OSError:
            continue

    if explicit_roots:
        deduped: list[Path] = []
        seen_explicit: set[str] = set()

        for candidate in explicit_roots:
            key = str(candidate)
            if key not in seen_explicit:
                seen_explicit.add(key)
                deduped.append(candidate)

        return deduped

    try:
        children = [child for child in root.iterdir() if child.is_dir()]
    except OSError:
        return [root]

    by_name = {child.name.lower(): child.resolve() for child in children}
    try:
        siblings = {
            child.name.lower(): child.resolve()
            for child in root.parent.iterdir() if child.is_dir() and child.resolve() != root
        } if root.parent != root else {}
    except OSError:
        siblings = {}

    excluded = _excluded_root_names(
        lowered, set(by_name) | set(siblings) | {root.name.lower()}
    )

    def _select_from(names_to_paths: dict[str, Path], *, active_only: bool) -> list[Path]:
        found: list[Path] = []
        for name, child in names_to_paths.items():
            if name in excluded or name not in lowered:
                continue
            if active_only and not _is_active_root(child):
                continue
            if not active_only and not _is_project_root(child):
                continue
            found.append(child)
        return found

    # Explicitly named roots take precedence -- own children first, then
    # (own-child search having found nothing) siblings of the launch
    # directory, so a stack named in the objective is reachable even when
    # it isn't nested inside whatever directory tamfis-code was launched
    # from.
    selected: list[Path] = _select_from(
        {n: c for n, c in by_name.items() if n not in excluded}, active_only=False
    )
    if not selected:
        selected = _select_from(
            {n: c for n, c in siblings.items() if n not in excluded}, active_only=False
        )

    # "the stack" in the TamfisGPT parent means the three canonical tiers,
    # not every folder beneath an arbitrary workspace parent.
    if not selected and "stack" in lowered:
        for source in (by_name, siblings):
            for name in _TAMFIS_STACK_ROOTS:
                if name in excluded:
                    continue
                child = source.get(name.lower())
                if child is not None and _is_active_root(child):
                    selected.append(child)
            if selected:
                break
        if not selected:
            # Never let classification fully break a known canonical
            # shortcut -- fall back to the unfiltered marker check.
            for source in (by_name, siblings):
                for name in _TAMFIS_STACK_ROOTS:
                    if name in excluded:
                        continue
                    child = source.get(name.lower())
                    if child is not None and _is_project_root(child):
                        selected.append(child)
                if selected:
                    break

    if selected:
        # Stable order, no duplicates.
        seen: set[str] = set()
        ordered: list[Path] = []
        for path in selected:
            key = str(path)
            if key not in seen:
                seen.add(key)
                ordered.append(path)
        return ordered

    if _is_project_root(root) and root.name.lower() not in excluded:
        return [root]

    # The launch directory itself is either not a project root, or was
    # explicitly excluded by the objective -- in either case, do not
    # default to it. Fall back to non-excluded active sibling projects
    # (bounded to root.parent's immediate children, same as the
    # "workspace is a parent directory" case below, just one level up)
    # rather than silently touching the excluded root.
    if root.name.lower() in excluded:
        active_siblings = [
            path for name, path in siblings.items()
            if name not in excluded and _is_active_root(path)
        ]
        if active_siblings:
            return active_siblings
        # Truly nothing else usable was found near an explicitly-excluded
        # launch directory -- refuse to silently fall back into it AND
        # refuse to return an empty scope (that disables scoping entirely,
        # see _scope_tool_arguments). The caller's diagnostics event still
        # names the exclusion, so this is visible, not silent.

    # Prefer actively-developed roots over backups/generated output/
    # dependency dirs/legacy folders that also happen to carry a project
    # marker (workspace.classify_root) -- but never let that filtering
    # produce fewer usable roots than a plain marker check would have.
    active_child_projects = [
        child.resolve() for child in children
        if _is_active_root(child) and child.name.lower() not in excluded
    ]
    child_projects = active_child_projects or [
        child.resolve() for child in children
        if _is_project_root(child) and child.name.lower() not in excluded
    ]
    if len(child_projects) == 1:
        return child_projects

    # Multiple unrelated projects under a parent directory are not silently
    # treated as one giant workspace.  Keep the parent as a discovery scope,
    # and the model will receive the immediate project list below.
    return child_projects or [root]


def _apply_mcp_task_scope(
    mcp_server: MCPServer,
    scope_roots: list[Path],
) -> None:
    """Make MCP's filesystem boundary match this turn's resolved scope.

    MCPServer is created before task-specific scope is derived because it is
    also used during repository preparation. Once the user's objective has
    been resolved, replace—not extend—the launch-workspace approval set with
    the exact authoritative project roots for this turn.
    """
    mcp_server.allowed_workspace_roots = {
        Path(root).expanduser().resolve()
        for root in scope_roots
    }


def _scope_instruction(workspace_root: str, scope_roots: list[Path]) -> str:
    roots = "\n".join(f"- {path}" for path in scope_roots)
    return (
        "WORKSPACE SCOPE (authoritative for this turn):\n"
        f"Workspace root: {Path(workspace_root).resolve()}\n"
        f"Target project roots:\n{roots}\n"
        "Operate only inside these target roots unless the user explicitly names "
        "another path. Do not recursively search the workspace parent. For a "
        "multi-project stack, inspect each listed root separately and use focused "
        "queries, bounded result counts, and project markers before reading files. "
        "Do not run repository-wide find/grep/rg from the parent directory. "
        "Every plan step must reference only the target project roots listed above. "
        "Do not mention, inspect or propose files from excluded or unselected sibling "
        "directories. If a planned path falls outside the target roots, discard and "
        "regenerate that plan step before displaying it."
    )


def _resolve_argument_path(value: Any, workspace_root: str) -> Optional[Path]:
    raw = str(value or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path(workspace_root) / path
    try:
        return path.resolve()
    except OSError:
        return path.absolute()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _scope_tool_arguments(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    workspace_root: str,
    scope_roots: list[Path],
    attachment_paths: tuple[str, ...] = (),
) -> tuple[dict[str, Any], Optional[str]]:
    """Normalise or reject a tool call that escapes the resolved task scope."""
    scoped = dict(arguments)
    if not scope_roots:
        return scoped, None

    workspace = Path(workspace_root).resolve()
    path_key = "path" if tool_name in _SCOPE_PATH_TOOLS else None

    if tool_name in {"extract_archive", "repackage_archive"}:
        checks: list[tuple[str, bool]] = []
        if tool_name == "extract_archive":
            source = _resolve_argument_path(scoped.get("path"), workspace_root)
            if source is None:
                return scoped, "Archive path is required."
            exact_attachments = {Path(item).expanduser().resolve() for item in attachment_paths}
            checks.append(("Archive path", source in exact_attachments or any(_is_within(source, root) for root in scope_roots)))
            destination = _resolve_argument_path(scoped.get("destination"), workspace_root)
            if destination is None:
                suffixes = (".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".tbz2", ".txz", ".tar", ".zip")
                source_name = source.name.lower()
                suffix = next((item for item in suffixes if source_name.endswith(item)), "")
                destination_root = scope_roots[0] if len(scope_roots) == 1 else workspace
                destination = destination_root / f"{source.name[:-len(suffix)] if suffix else source.stem}_extracted"
            scoped["path"] = str(source)
            scoped["destination"] = str(destination)
            checks.append(("Extraction destination", any(_is_within(destination, root) for root in scope_roots)))
        else:
            source = _resolve_argument_path(scoped.get("source_dir"), workspace_root)
            output = _resolve_argument_path(scoped.get("output_path"), workspace_root)
            if source is None or output is None:
                return scoped, "Both source_dir and output_path are required."
            scoped["source_dir"] = str(source)
            scoped["output_path"] = str(output)
            checks.extend([
                ("Package source", any(_is_within(source, root) for root in scope_roots)),
                ("Package output", any(_is_within(output, root) for root in scope_roots)),
            ])
        failed = [label for label, allowed in checks if not allowed]
        if failed:
            return scoped, (
                f"{', '.join(failed)} is outside the resolved task scope. Allowed roots: "
                + ", ".join(str(root) for root in scope_roots)
            )
        return scoped, None

    if tool_name == "execute_command":
        cwd = _resolve_argument_path(scoped.get("cwd"), workspace_root) or workspace
        command = str(scoped.get("command") or "").strip()

        # Models commonly emit:
        #
        #   cd /authorised/project && git status
        #
        # even when the tool's cwd remains the shared workspace parent.
        # Recognise one literal leading cd, validate it deterministically,
        # normalise cwd to that authorised root, and remove the redundant cd
        # before ordinary command-path validation.
        leading_cd = re.match(
            r"""^\s*cd\s+
                (?P<target>"[^"]+"|'[^']+'|[^\s;&|]+)
                \s*(?P<separator>&&|;)\s*
                (?P<remainder>.+)$
            """,
            command,
            flags=re.VERBOSE | re.DOTALL,
        )

        if leading_cd is not None:
            raw_target = leading_cd.group("target")

            try:
                parsed_target = shlex.split(raw_target)
            except ValueError:
                parsed_target = []

            if len(parsed_target) != 1:
                return scoped, "Invalid leading cd target in command."

            cd_target = _resolve_argument_path(parsed_target[0], str(cwd))

            if cd_target is None or not any(
                _is_within(cd_target, root)
                for root in scope_roots
            ):
                return scoped, (
                    f"Command cd target is outside the resolved task scope: "
                    f"{cd_target or parsed_target[0]}. Allowed roots: "
                    + ", ".join(str(root) for root in scope_roots)
                )

            cwd = cd_target
            command = leading_cd.group("remainder").strip()
            scoped["cwd"] = str(cwd)
            scoped["command"] = command

        broad_scan = bool(
            re.search(r"(?:^|[;&|]\s*)(?:find|rg|grep)\b", command)
        )

        try:
            command_tokens = shlex.split(command)
        except ValueError:
            command_tokens = []

        absolute_operands: list[Path] = []
        operand_roots: set[Path] = set()

        for token in command_tokens[1:]:
            candidate_text = token.split("=", 1)[-1] if "=" in token else token

            if (
                not candidate_text.startswith("/")
                or candidate_text == "/dev/null"
            ):
                continue

            candidate = Path(candidate_text).expanduser().resolve()
            absolute_operands.append(candidate)

            matched_root = next(
                (
                    root
                    for root in scope_roots
                    if _is_within(candidate, root)
                ),
                None,
            )

            if matched_root is not None:
                operand_roots.add(matched_root)

        if (
            len(scope_roots) == 1
            and cwd == workspace
            and workspace != scope_roots[0]
        ):
            scoped["cwd"] = str(scope_roots[0])
            cwd = scope_roots[0]

        if (
            broad_scan
            and cwd == workspace
            and workspace not in scope_roots
        ):
            all_operands_authorised = (
                bool(absolute_operands)
                and all(
                    any(
                        _is_within(candidate, root)
                        for root in scope_roots
                    )
                    for candidate in absolute_operands
                )
            )

            if all_operands_authorised and len(operand_roots) == 1:
                target_root = next(iter(operand_roots))
                scoped["cwd"] = str(target_root)
                cwd = target_root
            else:
                return scoped, (
                    "Broad parent-directory scan blocked. Run the command "
                    "separately in one of these project roots: "
                    + ", ".join(str(root) for root in scope_roots)
                )

        if not any(_is_within(cwd, root) for root in scope_roots):
            return scoped, (
                f"Command cwd is outside the resolved task scope: {cwd}. "
                "Allowed roots: "
                + ", ".join(str(root) for root in scope_roots)
            )

        for candidate in absolute_operands:
            if not any(
                _is_within(candidate, root)
                for root in scope_roots
            ):
                return scoped, (
                    f"Command path is outside the resolved task scope: "
                    f"{candidate}. Allowed roots: "
                    + ", ".join(str(root) for root in scope_roots)
                )

        scoped["cwd"] = str(cwd)
        scoped["command"] = command
        return scoped, None

    if path_key is None:
        return scoped, None

    requested = _resolve_argument_path(scoped.get(path_key), workspace_root)
    if requested is None:
        if len(scope_roots) == 1:
            scoped[path_key] = str(scope_roots[0])
            return scoped, None
        if tool_name == "search_code":
            return scoped, (
                "Search path is required for a multi-project task. Search one target root "
                "at a time: " + ", ".join(str(root) for root in scope_roots)
            )
        return scoped, None

    # An exact active project root is always a valid target. This must be
    # checked before the common-parent restriction; otherwise a standalone
    # repository such as the Tamfis-Code source checkout is incorrectly
    # rejected merely because unrelated roots also appeared in scope.
    if requested in scope_roots:
        scoped[path_key] = str(requested)
        return scoped, None

    # A request aimed at a genuine common parent is redirected only when
    # there is one narrower target. For a multi-project stack, reject it so
    # every repository is inspected deliberately.
    if requested == workspace and workspace not in scope_roots:
        if len(scope_roots) == 1:
            scoped[path_key] = str(scope_roots[0])
            return scoped, None
        return scoped, (
            "Parent-directory operation blocked for this multi-project stack. Use one "
            "target root at a time: " + ", ".join(str(root) for root in scope_roots)
        )

    if not any(_is_within(requested, root) for root in scope_roots):
        return scoped, (
            f"Path is outside the resolved task scope: {requested}. Allowed roots: "
            + ", ".join(str(root) for root in scope_roots)
        )

    scoped[path_key] = str(requested)
    return scoped, None

# Empty completions occasionally occur after a provider has consumed tool
# results. They are not evidence that the task is complete. Retry the same
# route, then continue through the configured fallback chain before failing.
MAX_EMPTY_CONTINUATION_RETRIES = 2
EMPTY_CONTINUATION_INSTRUCTION = (
    "Continue the current task from the tool results above. "
    "If more inspection is required, call the appropriate registered tools. "
    "If the task is complete, provide a concrete evidence-backed final answer. "
    "Do not return an empty response."
)

# A long, reasoning-heavy final answer (e.g. a full-stack audit report) can
# hit MAX_TOKENS_PER_REQUEST mid-sentence -- confirmed live: a nemotron
# reasoning-model response trailed off mid-paragraph with no closing
# summary, and was accepted as a genuinely complete answer purely because
# it had non-empty content and no tool_calls. finish_reason=="length" (the
# provider's own truncation signal) is the real test; this bounds how many
# times a still-truncated answer gets asked to keep going before giving up
# and labeling it partial rather than looping forever on a pathological
# case (e.g. a model that never naturally reaches finish_reason=="stop").
MAX_TRUNCATION_CONTINUATIONS = 6
TRUNCATION_CONTINUATION_INSTRUCTION = (
    "Your previous response was cut off by the output length limit before it finished. "
    "Continue writing EXACTLY where it left off -- do not repeat anything already written, "
    "do not restart, summarize, or re-introduce the topic. Just keep going from the exact "
    "point it stopped."
)

# Own transport recovery in the runner instead of letting the OpenAI SDK make
# an invisible immediate retry.  The growing pauses prevent a sick endpoint
# from being hammered every few milliseconds while the terminal's live status
# remains active.  Each clean streamed delta is already persisted by the
# caller, so a successful reconnect can continue from the exact durable text.
STREAM_RECONNECT_BACKOFF_SECONDS = (5.0, 15.0, 30.0)
STREAM_RECONNECT_INSTRUCTION = (
    "The provider stream was interrupted after the assistant text above. Continue the SAME "
    "task exactly where that text stopped. Do not repeat or restart the response. Preserve "
    "completed work and tool results; use registered tools if the remaining task requires them."
)


class _InterruptedCompletion(RuntimeError):
    """A completion that exhausted reconnects, retaining its clean partial."""

    def __init__(self, cause: Exception, partial: str = "") -> None:
        super().__init__(str(cause).strip() or type(cause).__name__)
        self.cause = cause
        self.partial = partial


def _messages_with_vision_content(
    messages: list[dict[str, Any]],
    vision_message_index: Optional[int],
    image_blocks: Optional[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Return a shallow-copied messages list with `image_blocks` spliced
    into the message at `vision_message_index` as OpenAI-style multipart
    content, WITHOUT mutating the caller's list or that message in place.

    `working_messages` (the canonical, always-plain-text conversation) must
    never be permanently rewritten into multipart form -- resume/anchor
    matching, deduplication, truncation, and the durable `.memory` snapshot
    all assume `message["content"]` is a plain string via
    `str(message.get("content") or "")`. Rebuilding the multipart form fresh
    for each actual provider call (this is what a stateless chat-completions
    request needs anyway -- the full history, including any image, is
    resent every round regardless) keeps every one of those string-based
    invariants intact for the rest of the turn.
    """
    if not image_blocks or vision_message_index is None or vision_message_index >= len(messages):
        return messages
    target = messages[vision_message_index]
    if target.get("role") != "user":
        return messages
    text = str(target.get("content") or "")
    patched = dict(target)
    patched["content"] = [{"type": "text", "text": text}, *image_blocks]
    return [*messages[:vision_message_index], patched, *messages[vision_message_index + 1:]]


# Image formats every vision-capable provider in providers.PROVIDERS
# (HF, OpenRouter) accepts as an
# OpenAI-style base64 data URI. Anything else attached (PDF, docx, source
# files, ...) stays on the existing plain-text "use read_file/
# extract_archive" attachment path -- it was never an image the model
# could literally see, only ever a path it could act on with a tool.
_VISION_IMAGE_MIME_BY_EXTENSION = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}
# 10 MB matches the existing per-attachment size cap enforced in cli.py
# before any attachment (image or not) reaches this function -- base64
# inflates that by roughly a third, well inside every configured
# provider's context_window/MAX_TOKENS_PER_REQUEST budget.
MAX_VISION_ATTACHMENT_BYTES = 10 * 1024 * 1024


def is_vision_image_path(path: str) -> bool:
    """True if `path`'s extension is a format vision-capable providers
    accept as an inline image, per _VISION_IMAGE_MIME_BY_EXTENSION."""
    return Path(path).suffix.lower() in _VISION_IMAGE_MIME_BY_EXTENSION


def build_vision_content_blocks(paths: list[str]) -> list[dict[str, Any]]:
    """Read each image path and return OpenAI-style `image_url` content
    blocks (base64 data URIs) ready to splice into a user message via
    _messages_with_vision_content. Silently skips a path that isn't a
    recognised image format, is missing, or can't be read -- this is a
    best-effort enhancement over the existing text-attachment path, not a
    new hard failure mode; callers should still tell the model about every
    attached path via the plain-text attachment note either way."""
    blocks: list[dict[str, Any]] = []
    for raw_path in paths:
        if not is_vision_image_path(raw_path):
            continue
        mime = _VISION_IMAGE_MIME_BY_EXTENSION[Path(raw_path).suffix.lower()]
        try:
            data = Path(raw_path).read_bytes()
        except OSError:
            continue
        if not data or len(data) > MAX_VISION_ATTACHMENT_BYTES:
            continue
        encoded = base64.b64encode(data).decode("ascii")
        blocks.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{encoded}"},
        })
    return blocks


def _same_route_reconnectable(manager: Any, exc: Exception) -> bool:
    """Retry transient transport/capacity faults, never account failures."""
    if not hasattr(manager, "is_retryable_provider_error"):
        return False
    if not manager.is_retryable_provider_error(exc):
        return False
    status = manager.provider_error_status(exc) if hasattr(manager, "provider_error_status") else None
    # Credentials/payment will not repair themselves after a sleep.  AUTO can
    # move to the next external route immediately for these statuses.
    return status not in {401, 402, 403}


async def _stream_completion_with_reconnect(
    manager: Any,
    client: Any,
    *,
    provider: ProviderType,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    renderer: StreamRenderer,
    reasoning_effort: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    initial_partial: str = "",
) -> tuple[str, list["_StreamedToolCall"], Optional[str]]:
    """Stream a completion and reconnect without losing or duplicating text.

    Once any text is visible, reconnect calls receive it as an assistant
    prefix and are kept internal until a clean response arrives.  Only the
    novel suffix is then rendered and checkpointed.  This makes provider or
    service restarts look like one continuous answer to the user.
    """
    durable_partial = initial_partial
    last_error: Optional[Exception] = None

    for attempt in range(len(STREAM_RECONNECT_BACKOFF_SECONDS) + 1):
        retrying_partial = bool(durable_partial)
        request_messages = messages
        if retrying_partial:
            request_messages = [
                *messages,
                {"role": "assistant", "content": durable_partial},
                {"role": "system", "content": STREAM_RECONNECT_INSTRUCTION},
            ]

        attempt_parts: list[str] = []

        def remember_attempt(delta: str) -> None:
            attempt_parts.append(delta)
            if not retrying_partial and progress_callback is not None:
                progress_callback(delta)

        try:
            content, calls, finish_reason = await _stream_one_completion(
                client,
                model=model,
                messages=request_messages,
                tools=tools,
                renderer=renderer,
                reasoning_effort=reasoning_effort,
                emit=not retrying_partial,
                progress_callback=remember_attempt,
            )
            if retrying_partial:
                novel = _novel_continuation(durable_partial, content)
                if novel:
                    renderer.handle_event({
                        "event_type": "assistant_delta",
                        "payload": {"content": novel},
                    })
                    if progress_callback is not None:
                        progress_callback(novel)
                content = durable_partial + novel
            return content, calls, finish_reason
        except Exception as exc:
            last_error = exc
            # Only text that was actually rendered is durable. A failed,
            # internal continuation attempt stays hidden so its incomplete
            # suffix cannot leak or be duplicated by the next reconnect.
            if not retrying_partial and attempt_parts:
                durable_partial += "".join(attempt_parts)
            if attempt >= len(STREAM_RECONNECT_BACKOFF_SECONDS) or not _same_route_reconnectable(manager, exc):
                raise _InterruptedCompletion(exc, durable_partial) from exc

            delay = STREAM_RECONNECT_BACKOFF_SECONDS[attempt]
            renderer.handle_event({
                "event_type": "diagnostics",
                "payload": {
                    "content": (
                        f"Stream from {provider.value} was interrupted; keeping this task alive "
                        f"and reconnecting in {int(delay)}s "
                        f"({attempt + 1}/{len(STREAM_RECONNECT_BACKOFF_SECONDS)})."
                    )
                },
            })
            await asyncio.sleep(delay)

    # The loop always returns or raises; this protects type-checkers and any
    # future edit that accidentally changes its bounds.
    assert last_error is not None
    raise _InterruptedCompletion(last_error, durable_partial)


def _novel_continuation(existing: str, continuation: str, *, overlap_window: int = 8_000) -> str:
    """Return only new continuation text, removing a repeated boundary.

    Continuation calls are internal implementation detail. Providers often
    repeat the last sentence despite being told not to; emitting only the
    novel suffix keeps the terminal response visually continuous.
    """
    if not existing or not continuation:
        return continuation
    maximum = min(len(existing), len(continuation), overlap_window)
    for size in range(maximum, 0, -1):
        if existing.endswith(continuation[:size]):
            return continuation[size:]
    return continuation


_RESUME_REQUEST_RE = re.compile(
    r"^(?:yes[,. ]+)?(?:proceed|continue|resume|carry\s+on|go\s+ahead)\b",
    re.IGNORECASE,
)


def _standalone_fallback_chain_names(manager: Any, current: ProviderType) -> list[str]:
    """Return the runtime-safe fallback display order.

    Real ProviderManager exposes fallback_chain_names(); lightweight test
    doubles from older suites do not.  The compatibility fallback remains
    standalone-only and deliberately excludes Tier IV.
    """
    method = getattr(manager, "fallback_chain_names", None)
    if callable(method):
        return list(method(current))
    return [
        provider.value
        for provider in (ProviderType.NVIDIA, ProviderType.HF, ProviderType.OPENROUTER)
        if provider != current
    ]


def _auto_provider_fallback_enabled(manager: Any) -> bool:
    """Use the real manager's billing-safe policy; preserve test doubles."""
    method = getattr(manager, "auto_fallback_enabled", None)
    return bool(method()) if callable(method) else True


def _paid_provider_fallback_enabled(manager: Any) -> bool:
    method = getattr(manager, "paid_fallback_enabled", None)
    return bool(method()) if callable(method) else True


def _is_resume_request(text: str) -> bool:
    """Whether a new prompt explicitly asks to continue prior work."""
    return bool(_RESUME_REQUEST_RE.match(text.strip()))


def _is_real_resume_objective(text: str) -> bool:
    lowered = text.strip().lower()
    return bool(
        lowered
        and not _is_resume_request(text)
        and not lowered.startswith(("active_plan=", "plan_"))
        and lowered not in {
            "understand", "inspect", "plan", "execute", "repair", "validate", "report", "completed",
        }
    )


def _checkpoint_resume_objective(checkpoint: dict[str, Any]) -> str:
    """Recover the whole user objective, including later clarifications.

    A clarification entered after a prematurely-finished model response used
    to replace the actual task in the next checkpoint.  The transcript still
    contained the original request, but routing/scope saw only the newest
    sentence (for example, just "the backend is ...").  Preserve the bounded
    sequence of real user instructions in order and de-duplicate it.
    """
    messages = list(checkpoint.get("messages") or [])
    stored = str(checkpoint.get("objective") or "").strip()

    # Anchor at the user message that created this checkpoint, then cross
    # backwards only over unfinished assistant work (tool calls, narrated
    # promises, or the old amnesia response).  A genuine completed answer is
    # a hard boundary, preventing unrelated earlier conversation objectives
    # from leaking into this resumed task.
    anchor = -1
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if (
            message.get("role") == "user"
            and str(message.get("content") or "").strip().casefold() == stored.casefold()
        ):
            anchor = index
            break
    start = anchor if anchor >= 0 else len(messages)
    for index in range(anchor - 1, -1, -1):
        message = messages[index]
        role = message.get("role")
        content = str(message.get("content") or "").strip()
        if role == "assistant" and not message.get("tool_calls"):
            unfinished = (
                _looks_like_narrated_tool_intent(content)
                or _looks_like_fake_tool_call(content)
                or _looks_like_fabricated_tool_result(content)
                or "don't have context from a previous turn" in content.lower()
            )
            if not unfinished:
                break
        if role == "user" and _is_real_resume_objective(content):
            start = index

    candidates: list[str] = []
    for message in messages[start:]:
        content = str(message.get("content") or "").strip()
        if message.get("role") == "user" and _is_real_resume_objective(content):
            candidates.append(content)
    if _is_real_resume_objective(stored):
        candidates.append(stored)

    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates[-8:]:
        key = candidate.casefold()
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return "\n\nAdditional user context: ".join(unique) or stored


def _close_interrupted_tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Restore provider protocol after interruption during tool execution.

    We cannot safely assume an in-flight command or mutation did *not* run.
    Supplying an explicit unknown-result tool response prevents both a
    malformed assistant/tool transcript and a blind duplicate execution;
    the resumed agent must inspect actual workspace state before retrying.
    """
    answered = {
        str(message.get("tool_call_id"))
        for message in messages
        if message.get("role") == "tool" and message.get("tool_call_id")
    }
    missing: list[tuple[str, str]] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            call_id = str(call.get("id") or "")
            if call_id and call_id not in answered:
                function = call.get("function") or {}
                missing.append((call_id, str(function.get("name") or "tool")))
    if not missing:
        return messages
    repaired = list(messages)
    for call_id, name in missing:
        repaired.append({
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps({
                "success": False,
                "interrupted": True,
                "error": (
                    f"The process was interrupted while {name} was in flight, before its result "
                    "was durably captured. Do not blindly repeat it; inspect current state first."
                ),
            }),
        })
    return repaired


def _workspace_roots_related(left: str, right: str) -> bool:
    """True only for the same workspace or a direct ancestor/descendant."""
    if not left or not right:
        return False
    left_path = Path(left).expanduser().resolve()
    right_path = Path(right).expanduser().resolve()
    return left_path == right_path or left_path in right_path.parents or right_path in left_path.parents


def _select_resume_state(session_id: int, workspace_root: str):
    """Find the best durable context for an explicit resume request.

    Restarting from a project subdirectory can legitimately select a
    different local session than the interrupted parent-workspace turn. A
    related-workspace search is allowed only for explicit resume wording,
    preventing ordinary new messages from inheriting another thread.
    """
    current = local_state.get_session_state(session_id)
    candidates = []
    for candidate_id in local_state.all_known_session_ids():
        candidate = local_state.get_session_state(candidate_id)
        if candidate.is_swarm_child or not _workspace_roots_related(candidate.workspace_root, workspace_root):
            continue
        has_context = bool(
            candidate.turn_checkpoint or candidate.conversation_history
            or candidate.conversation_summary or candidate.completed_actions
            or candidate.context_checkpoints or candidate.active_task
        )
        if has_context:
            candidates.append(candidate)
    if not candidates:
        return current
    # A real interrupted checkpoint always beats an inferred legacy summary;
    # within each class, use actual update time rather than numeric session id.
    return max(candidates, key=lambda state: (
        bool(
            state.turn_checkpoint
            and _is_real_resume_objective(str(state.turn_checkpoint.get("objective") or ""))
        ),
        state.updated_at or "",
    ))


def _legacy_resume_messages(state, incoming_objective: str) -> tuple[list[dict[str, Any]], str]:
    """Build bounded continuation context from pre-v0.4.28 state fields."""
    raw_history = list(state.conversation_history or [])
    history: list[dict[str, Any]] = []
    skip_denial = False
    skip_internal_response = False
    for message in raw_history:
        content = str(message.get("content") or "")
        if (
            message.get("role") == "user"
            and not _is_real_resume_objective(content)
            and content.strip().lower().startswith(("active_plan=", "plan_"))
        ):
            # Old interactive versions accidentally persisted their own
            # internal plan-selection sentinel as if the human had typed it.
            # Drop both the sentinel and the model response it triggered.
            skip_internal_response = True
            continue
        if skip_internal_response and message.get("role") == "assistant":
            skip_internal_response = False
            continue
        if message.get("role") == "user" and _is_resume_request(content):
            skip_denial = True
            continue
        if skip_denial and message.get("role") == "assistant":
            skip_denial = False
            if "don't have context from a previous turn" in content.lower():
                continue
        history.append(message)

    all_actions = list(state.completed_actions or [])[-250:]

    def action_objective(action: dict[str, Any]) -> str:
        purpose = str(action.get("purpose") or "").strip()
        match = re.search(r"\bfor:\s*(.+)$", purpose, re.IGNORECASE | re.DOTALL)
        return (match.group(1) if match else purpose).strip()

    inferred_objective = ""
    for message in reversed(history):
        candidate = str(message.get("content") or "").strip()
        if message.get("role") == "user" and _is_real_resume_objective(candidate):
            inferred_objective = candidate
            break
    if inferred_objective:
        inferred_objective = _checkpoint_resume_objective({
            "objective": inferred_objective,
            "messages": history,
        })
    # Completed actions are timestamped evidence and can be newer than the
    # legacy active_task field (older versions sometimes failed to refresh
    # that field). Prefer their embedded root objective when available.
    if not inferred_objective:
        for action in reversed(all_actions):
            candidate = action_objective(action)
            if _is_real_resume_objective(candidate):
                inferred_objective = candidate
                break
    if not inferred_objective:
        active_objective = str((state.active_task or {}).get("objective") or "").strip()
        if _is_real_resume_objective(active_objective):
            inferred_objective = active_objective

    relevant_actions = [
        action for action in all_actions
        if inferred_objective and action_objective(action) == inferred_objective
    ][-12:]
    if not relevant_actions:
        relevant_actions = [
            action for action in all_actions[-12:]
            if _is_real_resume_objective(action_objective(action))
        ]

    evidence_lines: list[str] = []
    for action in relevant_actions:
        purpose = str(action.get("purpose") or "").strip()
        if not purpose:
            continue
        status = str(action.get("status") or ("completed" if action.get("success") else "failed"))
        summary = str(
            action.get("result_summary") or action.get("stdout") or action.get("stderr") or ""
        ).strip()
        line = f"- [{status}] {purpose}"
        if summary:
            line += f" -> {summary[:600]}"
        evidence_lines.append(line)
    action_paths = list(dict.fromkeys(
        str(path)
        for action in relevant_actions
        for path in (action.get("files_changed") or [])
        if path
    ))
    if action_paths:
        evidence_lines.append("- Files changed for this task: " + ", ".join(action_paths[-20:]))
    # Global mutation/validation/checkpoint fields in legacy state span many
    # turns. Use them only when no objective-linked action evidence exists;
    # otherwise they can leak unrelated work into an explicit resume.
    if not history and not relevant_actions and state.modified_files:
        paths = [str(item.get("path") or "") for item in state.modified_files[-20:] if item.get("path")]
        if paths:
            evidence_lines.append("- Files already modified: " + ", ".join(paths))
    if not history and not relevant_actions and state.validation_results:
        evidence_lines.append(
            "- Recent validation: " + json.dumps(state.validation_results[-5:], default=str)[:2000]
        )
    checkpoint_summary = ""
    if not history and not relevant_actions and state.context_checkpoints:
        checkpoint_summary = str(state.context_checkpoints[-1].get("summary") or "").strip()
    summary = "\n".join(evidence_lines)
    if checkpoint_summary and checkpoint_summary not in summary:
        summary = f"{summary}\n- Last checkpoint: {checkpoint_summary}".strip()
    if not history and not relevant_actions and state.conversation_summary and state.conversation_summary not in summary:
        summary = f"{summary}\n- Last recorded response: {state.conversation_summary[:3000]}".strip()

    recovered: list[dict[str, Any]] = history[-40:]
    if inferred_objective and not any(
        message.get("role") == "user"
        and str(message.get("content") or "").strip() == inferred_objective
        for message in recovered
    ):
        recovered.append({"role": "user", "content": inferred_objective})
    if summary:
        recovered.append({
            "role": "assistant",
            "content": (
                "Durable progress recovered from the interrupted legacy session:\n" + summary[:12_000]
            ),
        })
    if recovered:
        recovered.append({"role": "user", "content": incoming_objective})
    return recovered, inferred_objective

# Confirmed live: a reasoning-heavy provider (NVIDIA nemotron) can get stuck
# narrating the same short phrase back-to-back thousands of times ("We have
# execute_command? Not listed." repeated verbatim) instead of producing real
# output or hitting a normal stop condition. Before this existed, that
# degenerate stream ran all the way to MAX_TOKENS_PER_REQUEST, got labeled
# finish_reason=="length", and the truncation-continuation loop above then
# re-fed the entire, still-growing garbage back to the model and asked it to
# "continue" -- reinforcing the same repetition across up to
# MAX_TRUNCATION_CONTINUATIONS more rounds until the process was OOM-killed.
# Backreference regex: captures a 6-200 char group that repeats immediately
# after itself at least 4 more times (5 total) -- cheap to check against just
# the tail of the accumulated text, and triggers within a couple KB of a
# genuine loop, long before the content can grow large enough to matter.
_DEGENERATE_REPETITION_RE = re.compile(r"(.{6,200}?)\1{4,}", re.DOTALL)
# A separate guard for a common context-confusion failure: the model starts
# replaying a synthetic transcript ("then the user said ... then the
# assistant responded ...") instead of answering the current task. The
# individual paragraphs are often too long or slightly varied for the
# contiguous backreference above, so count the unmistakable transcript
# markers in the rolling tail.
_CONVERSATION_ECHO_USER_MARKER = re.compile(r"\bthen\s+the\s+user\s+said\b", re.IGNORECASE)
_CONVERSATION_ECHO_ASSISTANT_MARKER = re.compile(r"\bthen\s+the\s+assistant\s+(?:said|responded)\b", re.IGNORECASE)
_REPEATED_LONG_SEGMENT_MIN_CHARS = 120


def _has_repeated_long_segment(text: str) -> bool:
    """Detect a long sentence/paragraph/code fragment replayed three times.

    Repetition can be non-contiguous: a model may repeat the same analysis
    paragraph while changing only headings, bullets, or surrounding code.
    Exact contiguous backreferences therefore miss it. Short fragments are
    intentionally ignored to avoid flagging legitimate repeated terms.
    """
    raw_text = text or ""
    # Preserve paragraph-sized units first. Splitting on sentence boundaries
    # alone loses the repeated block when a provider inserts a blank line
    # between copies of an otherwise short three-sentence analysis.
    paragraphs = re.split(r"\n{2,}", raw_text)
    segments = list(paragraphs)
    segments.extend(
        sentence
        for paragraph in paragraphs
        for sentence in re.split(r"(?<=[.!?])\s+", paragraph)
    )
    counts: Counter[str] = Counter()
    for segment in segments:
        normalized = re.sub(r"\s+", " ", segment).strip().casefold()
        if len(normalized) >= _REPEATED_LONG_SEGMENT_MIN_CHARS:
            counts[normalized] += 1
    return any(count >= 3 for count in counts.values())
_DEGENERATE_REPETITION_TAIL_WINDOW = 8_000
_STREAM_QUALITY_LAG_CHARS = 1_024
_CORRUPTED_STREAM_MIN_CHARS = 600
_COMMON_LANGUAGE_TRIGRAMS = frozenset({
    "the", "and", "ing", "ion", "ent", "for", "tio", "ere", "her",
    "ter", "hat", "his", "tha", "ith", "ers", "ati", "all", "wit",
    "not", "you", "this", "str", "int", "sel", "elf", "ret", "urn",
})
_DEGENERATE_REPETITION_CAVEAT = (
    "\n\n⚠ Generation was stopped early: the model got stuck repeating the same "
    "text instead of producing new output. The response above is truncated at "
    "the point the repetition began -- try again, possibly with a narrower or "
    "clearer objective."
)


def _degenerate_repetition_index(text: str) -> Optional[int]:
    """Index into `text` where a back-to-back repeated phrase begins, or
    None if the tail shows no such pattern. Only scans the tail (not the
    whole string) so this stays cheap to call on every streamed chunk."""
    tail = text[-_DEGENERATE_REPETITION_TAIL_WINDOW:]
    match = _DEGENERATE_REPETITION_RE.search(tail)
    if match is None:
        return None
    return (len(text) - len(tail)) + match.start()


def _truncate_degenerate_repetition(text: str) -> str:
    index = _degenerate_repetition_index(text)
    if index is None:
        return text
    return text[:index].rstrip() + _DEGENERATE_REPETITION_CAVEAT


def _corrupted_lexical_stream_index(text: str) -> Optional[int]:
    """Detect provider token-decoder corruption such as long strings of
    recombined lexical fragments (``...Logo...webkit...urp...``). This is
    intentionally stricter than ordinary low-quality prose: it requires a
    dominant uncommon trigram *and* a high proportion of abnormally long or
    repeatedly case-transitioning words. Open fenced code is excluded so
    minified/generated identifiers do not false-positive."""
    tail = text[-_DEGENERATE_REPETITION_TAIL_WINDOW:]
    if len(tail) < _CORRUPTED_STREAM_MIN_CHARS or tail.count("```") % 2:
        return None
    word_matches = list(re.finditer(r"[A-Za-z]{4,}", tail))
    if len(word_matches) < 24:
        return None
    words = [match.group(0) for match in word_matches]
    unusual = sum(
        len(word) >= 20
        or len(re.findall(r"[a-z][A-Z]", word)) >= 2
        for word in words
    )
    if unusual / len(words) < 0.28:
        return None

    counts: Counter[str] = Counter()
    total_letters = sum(len(word) for word in words)
    for word in words:
        lowered = word.lower()
        counts.update(lowered[index:index + 3] for index in range(len(lowered) - 2))
    dominant = next(
        (
            (fragment, count) for fragment, count in counts.most_common()
            if fragment not in _COMMON_LANGUAGE_TRIGRAMS
        ),
        None,
    )
    if dominant is None:
        return None
    fragment, count = dominant
    if count < 24 or (count * 3) / max(total_letters, 1) < 0.09:
        return None
    first = tail.lower().find(fragment)
    if first < 0:
        return None
    return len(text) - len(tail) + first


def _estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Rough token estimate for a working-messages list, counting both
    message content and any tool_calls argument strings (which can be large
    for tools like write_file)."""
    total_chars = 0
    for message in messages:
        total_chars += len(str(message.get("content") or ""))
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            total_chars += len(str(function.get("arguments") or ""))
    return total_chars // _CHARS_PER_TOKEN_ESTIMATE


def _tool_output_for_render(result: dict[str, Any]) -> dict[str, Any]:
    """Flatten MCPServer.call_tool()'s {"result": <actual>, "tool":, "success":}
    envelope into the shape render.py's tool_output handler actually reads
    (content/stdout/stderr/exit_code/error/success at the top level).

    Without this, a real, successful tool call could render no visible line
    at all: read_file's actual payload is a plain string (not under a
    "content" key), list_directory's is a plain list, and execute_command's
    nested dict uses "return_code" where render.py looks for "exit_code" --
    render.py's suppression check (skip rendering a tool-completion envelope
    with none of its recognized keys populated) was silently eating all
    three. The model still sees the unflattened `result` via working_messages
    either way; this only reshapes what the human display reads."""
    flattened = dict(result)
    inner = flattened.pop("result", None)
    if isinstance(inner, str):
        flattened.setdefault("content", inner)
    elif isinstance(inner, list):
        flattened.setdefault("content", f"{len(inner)} item(s)" if inner else "(empty)")
    elif isinstance(inner, dict):
        for key, value in inner.items():
            flattened.setdefault("exit_code" if key == "return_code" else key, value)
    return flattened


def _semantic_tool_failure(tool_name: str, arguments: dict[str, Any], result: dict[str, Any], workspace_root: str) -> Optional[str]:
    """Return a human-readable semantic failure hidden inside a transport-success envelope.

    Some adapters historically returned ``success=True`` merely because the
    Python tool function returned normally, while the actual operation payload
    contained ``Error: ...`` or no meaningful Git data.  The renderer and the
    model must see the operation's truth, not only transport success.
    """
    if result.get("success") is False:
        return str(result.get("error") or "Tool operation failed")

    inner = result.get("result")
    error = result.get("error")
    if error:
        return str(error)

    if isinstance(inner, dict) and inner.get("error"):
        return str(inner.get("error"))

    if isinstance(inner, str):
        stripped = inner.strip()
        lowered = stripped.lower()
        failure_prefixes = (
            "error:",
            "file not found:",
            "permission denied:",
            "not a git repository",
            "fatal:",
            "failed:",
        )
        if lowered.startswith(failure_prefixes):
            return stripped

    if tool_name == "read_file":
        requested = str(arguments.get("path") or "").strip()
        if requested:
            candidate = Path(requested)
            if not candidate.is_absolute():
                candidate = Path(workspace_root) / candidate
            try:
                if not candidate.exists():
                    return f"File not found: {candidate}"
                if not candidate.is_file():
                    return f"Not a file: {candidate}"
            except OSError as exc:
                return f"Unable to inspect file {candidate}: {exc}"

    if tool_name == "get_git_info":
        requested = str(arguments.get("path") or arguments.get("cwd") or workspace_root)
        candidate = Path(requested)
        if not candidate.is_absolute():
            candidate = Path(workspace_root) / candidate
        try:
            is_git = (candidate / ".git").exists()
        except OSError:
            is_git = False
        empty_payload = inner in (None, "", {}, [])
        if not is_git and empty_payload:
            return f"Not a Git repository: {candidate}"

    return None


def _normalise_tool_result(
    tool_name: str,
    arguments: dict[str, Any],
    result: dict[str, Any],
    workspace_root: str,
) -> dict[str, Any]:
    """Convert transport-success envelopes into truthful operation results."""
    normalised = dict(result)
    failure = _semantic_tool_failure(tool_name, arguments, normalised, workspace_root)
    if failure:
        normalised["success"] = False
        normalised["error"] = failure
        if normalised.get("result") in (None, "", {}, []):
            normalised["result"] = {"error": failure}
    else:
        normalised["success"] = bool(normalised.get("success", True))
    return normalised


_AUDIT_PLAN_PATH_RE = re.compile(r"(?<![\w])/(?:[^\s,;()\[\]{}<>]+)")


def _next_audit_plan_file(plan: Any, scope_roots: list[Path]) -> Optional[Path]:
    """Return the first safe, concrete file named by an unfinished audit step.

    This is deliberately narrow: it never invents a path, traverses a
    directory, or executes a command.  It only turns an already-approved
    reasoning-plan step containing an absolute path into the equivalent
    read-only tool operation when a provider failed to emit native tool JSON.
    """
    if plan is None:
        return None
    for step in plan.steps:
        if step.status not in {"pending", "in_progress"}:
            continue
        for raw in _AUDIT_PLAN_PATH_RE.findall(str(step.name)):
            candidate = Path(raw.rstrip(".:")).expanduser()
            try:
                resolved = candidate.resolve()
                if not resolved.is_file():
                    continue
                if any(_is_within(resolved, root.resolve()) for root in scope_roots):
                    return resolved
            except OSError:
                continue
    return None


async def _recover_audit_plan_file(
    *,
    mcp_server: MCPServer,
    orchestrator: AgentOrchestrator,
    renderer: StreamRenderer,
    working_messages: list[dict[str, Any]],
    plan: Any,
    scope_roots: list[Path],
    objective: str,
    round_number: int,
) -> bool:
    """Execute one concrete pending audit read after a malformed completion."""
    path = _next_audit_plan_file(plan, scope_roots)
    if path is None:
        return False
    call_id = f"audit_recovery_{round_number}_{uuid.uuid4().hex[:8]}"
    arguments = {"path": str(path)}
    # Keep the conversation protocol valid even though the provider omitted
    # its native tool call: every synthetic result has a matching assistant
    # tool_calls message and is visibly marked as runtime recovery.
    working_messages.append({
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": "read_file", "arguments": json.dumps(arguments)},
        }],
    })
    envelope = ToolEnvelope(
        tool_call_id=call_id,
        tool_name="read_file",
        arguments=arguments,
        purpose=f"Recover pending audit inspection for: {objective[:160]}",
        cwd=str(path.parent),
    )
    result = await mcp_server.call_tool("read_file", arguments)
    result = _normalise_tool_result("read_file", arguments, result, str(scope_roots[0]))
    envelope.finish(result=result, success=bool(result.get("success")))
    orchestrator.record_tool(envelope)
    renderer.handle_event({
        "event_type": "diagnostics",
        "payload": {"content": f"Recovered pending audit read through read_file: {path}"},
    })
    renderer.handle_event({
        "event_type": "tool_output",
        "payload": {"tool": "read_file", "result": _tool_output_for_render(result)},
    })
    working_messages.append({
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps(result, default=str),
    })
    return True


async def _nonstream_one_completion(
    client,
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> tuple[str, list[_StreamedToolCall]]:
    """Run a non-streaming completion and preserve structured tool calls."""
    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": 0.2,
        "max_tokens": MAX_TOKENS_PER_REQUEST,
    }
    if tools:
        request_kwargs["tools"] = tools
        request_kwargs["tool_choice"] = "auto"

    response = await client.chat.completions.create(**request_kwargs)
    if not response.choices:
        return "", []

    message = response.choices[0].message
    content = str(getattr(message, "content", None) or "")
    calls: list[_StreamedToolCall] = []
    for raw_call in getattr(message, "tool_calls", None) or []:
        function = getattr(raw_call, "function", None)
        calls.append(
            _StreamedToolCall(
                call_id=str(getattr(raw_call, "id", "") or ""),
                name=str(getattr(function, "name", "") or ""),
                arguments=str(getattr(function, "arguments", "") or ""),
            )
        )
    return content, calls


async def _recover_empty_continuation(
    manager: ProviderManager,
    *,
    requested_provider: ProviderType,
    resolved_provider: ProviderType,
    config: Any,
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    renderer: StreamRenderer,
    task_profile: Any,
) -> tuple[str, list[_StreamedToolCall], ProviderType, Any, Any, str]:
    """Recover an empty post-tool continuation without abandoning the task."""
    retry_messages = list(messages) + [
        {"role": "system", "content": EMPTY_CONTINUATION_INSTRUCTION}
    ]
    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_EMPTY_CONTINUATION_RETRIES + 1):
        renderer.handle_event({
            "event_type": "diagnostics",
            "payload": {
                "content": (
                    f"Provider {resolved_provider.value} returned an empty continuation; "
                    f"retrying the same route ({attempt}/{MAX_EMPTY_CONTINUATION_RETRIES})."
                )
            },
        })
        try:
            if attempt == 1:
                content, calls, _finish_reason = await _stream_one_completion(
                    client,
                    model=model,
                    messages=retry_messages,
                    tools=tools,
                    renderer=renderer,
                )
            else:
                content, calls = await _nonstream_one_completion(
                    client,
                    model=model,
                    messages=retry_messages,
                    tools=tools,
                )
                if content:
                    renderer.handle_event({
                        "event_type": "assistant_delta",
                        "payload": {"content": content},
                    })
            if content.strip() or calls:
                return content, calls, resolved_provider, config, client, model
        except Exception as exc:  # provider-specific failures are handled below
            last_error = exc
            break

    if requested_provider == ProviderType.AUTO and _auto_provider_fallback_enabled(manager) and hasattr(manager, "fallback_candidates"):
        failed_provider = resolved_provider
        for candidate in manager.fallback_candidates(failed_provider, task_profile):
            candidate_client = manager.get_client(candidate)
            candidate_config = manager.PROVIDERS.get(candidate)
            if candidate_client is None or candidate_config is None:
                continue
            candidate_model = _select_fallback_model(manager, candidate_config, task_profile)
            renderer.handle_event({
                "event_type": "diagnostics",
                "payload": {
                    "content": (
                        f"Provider {failed_provider.value} produced no continuation; "
                        f"falling back to {candidate.value} / {candidate_model}."
                    )
                },
            })
            try:
                content, calls, _finish_reason = await _stream_one_completion(
                    candidate_client,
                    model=candidate_model,
                    messages=retry_messages,
                    tools=tools if candidate_config.tool_calling else [],
                    renderer=renderer,
                )
                if not content.strip() and not calls:
                    content, calls = await _nonstream_one_completion(
                        candidate_client,
                        model=candidate_model,
                        messages=retry_messages,
                        tools=tools if candidate_config.tool_calling else [],
                    )
                    if content:
                        renderer.handle_event({
                            "event_type": "assistant_delta",
                            "payload": {"content": content},
                        })
                if content.strip() or calls:
                    return (
                        content,
                        calls,
                        candidate,
                        candidate_config,
                        candidate_client,
                        candidate_model,
                    )
            except Exception as exc:
                last_error = exc
                failed_provider = candidate
                continue

    if last_error is not None:
        raise RuntimeError(
            f"Provider continuation remained empty; final recovery error: {last_error}"
        ) from last_error
    raise RuntimeError("Provider continuation remained empty after retries and fallbacks")


def _bounded_text(value: str, *, head: int, tail: int, label: str) -> str:
    """Return a compact, evidence-preserving representation of a large string."""
    if len(value) <= head + tail:
        return value
    omitted = len(value) - head - tail
    return (
        f"{value[:head]}\n"
        f"...[{label}: {omitted} characters omitted; original={len(value)}]...\n"
        f"{value[-tail:]}"
    )


def _compact_json_value(value: Any, *, head: int, tail: int, depth: int = 0) -> Any:
    """Compact arbitrarily large JSON values while preserving valid JSON.

    Historical tool-call arguments are sent back to providers as assistant
    messages. They no longer need to be executable, but they MUST remain valid
    JSON. Raw string slicing with an inserted marker corrupts JSON and can make
    OpenAI-compatible providers reject the whole continuation.
    """
    if depth >= 8:
        return "[nested value compacted]"
    if isinstance(value, str):
        if len(value) <= head + tail:
            return value
        # A structured marker object, not a marker embedded in the string
        # itself: the caller (a tool-call argument, or retrieved evidence)
        # is machine-readable JSON, and an inline "...[N chars omitted]..."
        # marker would silently become part of the string value on the next
        # round-trip. `_bounded_text` (plain inline marker) stays correct for
        # non-JSON display text (tool/assistant message content).
        return {
            "_tamfis_compacted": True,
            "original_chars": len(value),
            "head": value[:head],
            "tail": value[-tail:] if tail else "",
        }
    if isinstance(value, list):
        if len(value) > 24:
            return [
                *[_compact_json_value(item, head=head, tail=tail, depth=depth + 1) for item in value[:10]],
                {"_tamfis_compacted_items": len(value) - 20},
                *[_compact_json_value(item, head=head, tail=tail, depth=depth + 1) for item in value[-10:]],
            ]
        return [_compact_json_value(item, head=head, tail=tail, depth=depth + 1) for item in value]
    if isinstance(value, dict):
        items = list(value.items())
        if len(items) > 40:
            kept = items[:20] + items[-20:]
            compacted = {
                str(key): _compact_json_value(item, head=head, tail=tail, depth=depth + 1)
                for key, item in kept
            }
            compacted["_tamfis_compacted_keys"] = len(items) - len(kept)
            return compacted
        return {
            str(key): _compact_json_value(item, head=head, tail=tail, depth=depth + 1)
            for key, item in items
        }
    return value


def _compact_tool_arguments(arguments: str, *, head: int, tail: int) -> str:
    """Compact one tool-call argument string and always return valid JSON."""
    try:
        parsed = json.loads(arguments or "{}")
    except (json.JSONDecodeError, TypeError, ValueError):
        parsed = {
            "_tamfis_compacted": True,
            "_tamfis_reason": "historical tool arguments were malformed or incomplete",
            "_tamfis_original_chars": len(arguments),
            "preview": _bounded_text(
                str(arguments), head=head, tail=tail, label="malformed arguments compacted"
            ),
        }
    else:
        parsed = _compact_json_value(parsed, head=head, tail=tail)
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"), default=str)


def _trim_message_in_place(
    message: dict[str, Any],
    *,
    head: int,
    tail: int,
    min_len: int,
    compact_assistant_content: bool = True,
) -> bool:
    """Compact every message field that can dominate the provider context.

    This handles three independent growth sources:
    * serialized tool results in role=tool messages;
    * tool_calls[].function.arguments in role=assistant messages;
    * verbose assistant narration/code emitted before or beside tool calls.
    """
    changed = False
    role = message.get("role")
    content = str(message.get("content") or "")

    if role == "tool" and len(content) > min_len:
        message["content"] = _bounded_text(
            content, head=head, tail=tail, label="tool output compacted"
        )
        changed = True
    elif role == "assistant" and compact_assistant_content and len(content) > min_len:
        message["content"] = _bounded_text(
            content, head=head, tail=tail, label="assistant content compacted"
        )
        changed = True
    elif role == "user" and len(content) > min_len:
        # Old user turns (not the current request -- callers exclude the
        # latest user message from normal passes) can dominate the budget
        # just as easily as a large tool result, e.g. a prior turn that
        # pasted a long log/diff as its objective. Never compacted before
        # this fix, no matter how large.
        message["content"] = _bounded_text(
            content, head=head, tail=tail, label="user message compacted"
        )
        changed = True

    if role == "assistant":
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            arguments = str(function.get("arguments") or "")
            if len(arguments) > min_len:
                compacted = _compact_tool_arguments(arguments, head=head, tail=tail)
                if compacted != arguments:
                    function["arguments"] = compacted
                    changed = True
    return changed


def _drop_oldest_completed_cycle(messages: list[dict[str, Any]], *, keep_recent: int) -> bool:
    """Evict one old completed conversational/tool cycle without breaking protocol."""
    if len(messages) <= keep_recent + 1:
        return False

    latest_user_index = max(
        (index for index, message in enumerate(messages) if message.get("role") == "user"),
        default=-1,
    )
    limit = max(1, len(messages) - keep_recent)

    index = 1  # Preserve the leading system instruction.
    while index < limit:
        if index == latest_user_index:
            index += 1
            continue
        message = messages[index]
        role = message.get("role")

        if role == "assistant" and message.get("tool_calls"):
            call_ids = {
                str(call.get("id") or "")
                for call in message.get("tool_calls") or []
            }
            end = index + 1
            while end < len(messages):
                candidate = messages[end]
                if candidate.get("role") != "tool":
                    break
                tool_call_id = str(candidate.get("tool_call_id") or "")
                if call_ids and tool_call_id not in call_ids:
                    break
                end += 1
            del messages[index:end]
            return True

        # Old standalone user/assistant/tool messages can be removed safely.
        del messages[index]
        return True

    return False


def _trim_tool_outputs(messages: list[dict[str, Any]], target_tokens: int, keep_recent: int = 6) -> bool:
    """Iteratively compact working context until it fits the provider budget.

    Normal compaction preserves recent detail. Emergency passes compact all
    non-system history, then evict the oldest completed cycles. The latest user
    request and leading system instruction are never removed.
    """
    trimmed_any = False
    if _estimate_tokens(messages) <= target_tokens:
        return False

    latest_user_index = max(
        (index for index, message in enumerate(messages) if message.get("role") == "user"),
        default=-1,
    )
    boundary = max(1, len(messages) - keep_recent)

    passes = (
        (range(1, boundary), 256, 128, 384),
        (range(boundary, len(messages)), 1200, 300, 1200),
        (range(1, len(messages)), 160, 80, 160),
    )

    for indexes, head, tail, min_len in passes:
        for index in list(indexes):
            if index >= len(messages) or index == latest_user_index:
                continue
            if _trim_message_in_place(
                messages[index], head=head, tail=tail, min_len=min_len
            ):
                trimmed_any = True
            if _estimate_tokens(messages) <= target_tokens:
                return trimmed_any

    # Last resort before evicting whole cycles: the CURRENT request (the
    # latest user message) is normally left untouched above so its full
    # text reaches the model -- but if it alone is large enough to blow the
    # budget (a big pasted log/diff/objective), refusing to ever touch it
    # meant compaction could never succeed for that turn, and it failed
    # immediately on round 1, before any tool call even happened (nothing
    # for _drop_oldest_completed_cycle to evict yet either). Bound it the
    # same way tool/assistant content is bounded: the full text isn't lost,
    # only what's sent to the provider is shortened.
    if latest_user_index != -1 and _estimate_tokens(messages) > target_tokens:
        if _trim_message_in_place(
            messages[latest_user_index], head=2000, tail=500, min_len=3000,
        ):
            trimmed_any = True

    # If compact representations themselves are still too numerous, remove
    # only old completed cycles until the active request fits.
    while _estimate_tokens(messages) > target_tokens:
        if not _drop_oldest_completed_cycle(messages, keep_recent=keep_recent):
            break
        trimmed_any = True

    return trimmed_any

def _latest_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content") or "").strip().lower()
    return ""


def _is_plain_conversation(messages: list[dict[str, Any]]) -> bool:
    """Return True for turns that cannot reasonably require repository tools.

    This is intentionally conservative.  It prevents greetings and basic
    conversational prompts from sending a large function catalogue to small
    local models such as llama3.2:3b, which may serialize invented function
    calls into ordinary assistant text instead of emitting tool_calls.
    """
    text = _latest_user_text(messages)
    if not text:
        return True
    exact = {
        "hi", "hello", "hey", "hi there", "hello there",
        "good morning", "good afternoon", "good evening",
        "how are you", "how are you?", "thanks", "thank you",
    }
    if text in exact:
        return True
    conversational_prefixes = (
        "who are you", "what can you do", "tell me about yourself",
    )
    return text.startswith(conversational_prefixes)


# Verbs that signal the user is asking for an actual code change, not just an
# investigation/question -- deliberately conservative (a false negative here
# just means the honesty check below doesn't fire, same as today; a false
# positive would wrongly flag a legitimate no-op completion as suspicious).
_CHANGE_REQUEST_VERBS = (
    "fix", "repair", "patch", "correct", "debug",
    "add", "implement", "create", "write",
    "update", "change", "modify", "edit", "refactor", "rewrite", "remove", "delete",
)


def _looks_like_change_request(text: str) -> bool:
    value = (text or "").lower()
    # Inspection requests often mention defects "to be fixed" or "for
    # fixing" as the subject of a later turn. Those are not current mutation
    # instructions. The old substring check also treated "fixed" and
    # "fixes" as active requests, which produced the no-files-changed warning
    # after legitimate audits.
    value = re.sub(
        r"\b(?:to\s+be|that\s+(?:need|needs)\s+to\s+be|for)\s+"
        r"(?:fix(?:ed|es|ing)?|repair(?:ed|s|ing)?|patch(?:ed|es|ing)?)\b",
        "",
        value,
    )
    return bool(re.search(
        r"\b(?:" + "|".join(map(re.escape, _CHANGE_REQUEST_VERBS)) + r")\b",
        value,
    ))


# Confirmed live: a weak model can decide a fix is "done" (in prose only,
# never having called write_file/edit_file) and then move straight on to
# restarting/reloading the service that was supposed to pick the fix up --
# the restart itself succeeds, so the turn reads as a clean success even
# though nothing on disk actually changed. By the time the end-of-turn
# no-mutation caveat fires, the (possibly disruptive) restart has already
# run. This can't be blocked outright -- a restart can be a legitimate,
# unrelated step -- but it's exactly the moment a human approver most needs
# the same "nothing has been changed yet" fact the end-of-turn caveat
# already computes, surfaced instead in the approval panel *before* they
# approve it.
_SERVICE_RESTART_RE = re.compile(
    r"\bsystemctl\s+(restart|reload)\b"
    r"|\bservice\s+\S+\s+(restart|reload)\b"
    r"|/etc/init\.d/\S+\s+(restart|reload)\b"
    r"|\b(apachectl|nginx|pm2|supervisorctl)\b[^\n]*\b(restart|reload)\b"
)


def _looks_like_service_restart(command: str) -> bool:
    return bool(_SERVICE_RESTART_RE.search(command or ""))


def _preview_diff_for_tool_call(mcp_server: MCPServer, tool_name: str, arguments: dict[str, Any]) -> Optional[str]:
    """Best-effort unified-diff preview for write_file/edit_file, computed
    WITHOUT writing anything -- read-only preview of exactly what
    safety.record_mutation would compute for real once the call is
    actually approved and executed (same _unified_diff helper). Before
    this, the approval panel for these two tools rendered the raw JSON
    arguments -- for write_file, the entire proposed new file content as a
    JSON string -- instead of a diff. Returns None for any other tool
    (nothing to preview), or if the preview itself can't be meaningfully
    computed (e.g. edit_file's old_string isn't found/unique in the
    current file) -- the real call will report that error itself, and a
    denied preview must never block the approval prompt from appearing."""
    if tool_name not in {"write_file", "edit_file"}:
        return None
    path = arguments.get("path")
    if not path:
        return None
    try:
        resolved = mcp_server._resolve_in_workspace(str(path))
    except (PermissionError, OSError):
        return None

    if tool_name == "write_file":
        new_content = arguments.get("content")
        if new_content is None:
            return None
        try:
            original_content = resolved.read_text(encoding="utf-8", errors="ignore") if resolved.is_file() else None
        except OSError:
            return None
        return _unified_diff(str(path), original_content, str(new_content))

    old_string, new_string = arguments.get("old_string"), arguments.get("new_string")
    if old_string is None or new_string is None or not resolved.is_file():
        return None
    try:
        original_content = resolved.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    if original_content.count(str(old_string)) != 1:
        return None
    new_content = original_content.replace(str(old_string), str(new_string), 1)
    return _unified_diff(str(path), original_content, new_content)


# Confirmed live: a weak model can write out a fenced code block like
# ```python execute_command("...") ``` as if narrating a tool call it never
# actually made -- the round genuinely has zero real tool_calls, so the turn
# completes normally with nothing having happened. This is a much more
# precise signal than a verb heuristic: these are our own tool names, so a
# real, legitimate answer essentially never needs to write one followed by
# an opening paren.
_FAKE_TOOL_CALL_RE = re.compile(
    r"\b(?:read_file|write_file|edit_file|extract_archive|repackage_archive|list_directory|search_code|execute_command|get_git_info|browser|web_search)\s*\("
)

# Confirmed live, a second shape: instead of a paren-style call, a weak
# model (e.g. nvidia/nemotron-3-super) can instead narrate a
# {"tool": "read_file", "argument": {"path": "..."}} (or "name"/"arguments")
# JSON object in plain prose -- no real tool_calls either, and the pattern
# above never matches it (there's no tool-name-immediately-followed-by-paren
# anywhere in valid JSON). Seen repeating itself round after round with
# nothing but this JSON blob until the separate degenerate-repetition guard
# cut generation off mid-stream -- this check lets it be caught cleanly,
# with the normal "nothing actually happened" caveat, well before that.
_FAKE_TOOL_CALL_JSON_RE = re.compile(
    r'"(?:tool|name)"\s*:\s*"(?:read_file|write_file|edit_file|extract_archive|repackage_archive|list_directory|'
    r'search_code|execute_command|get_git_info|browser|web_search)"'
)


def _looks_like_fake_tool_call(text: str) -> bool:
    return bool(_FAKE_TOOL_CALL_RE.search(text) or _FAKE_TOOL_CALL_JSON_RE.search(text))


_NARRATED_TOOL_INTENT_RE = re.compile(
    r"(?:^|[.!?]\s+|\n)\s*(?:"
    r"(?:now\s+)?let\s+me|"
    r"i(?:'ll|\s+will|\s+am\s+going\s+to)|"
    r"(?:now|next|first),?\s+i(?:'ll|\s+will)"
    r")\s+(?:(?:start|begin)\s+by\s+|first\s+|try(?:ing)?\s+to\s+)?(?:"
    r"check(?:ing)?|examin(?:e|ing)|inspect(?:ing)?|read(?:ing)?|"
    r"look(?:ing)?\s+(?:for|at|through)|search(?:ing)?|run(?:ning)?|open(?:ing)?|"
    r"review(?:ing)?|verif(?:y|ying)|explor(?:e|ing)|scan(?:ning)?|list(?:ing)?|"
    # Confirmed live: a weak model (meta/llama-3.1-70b-instruct on NVIDIA
    # NIM) can narrate a future action with a verb outside the original
    # inspection-only list -- "I will try to install the Python debugger
    # pdb" -- and fall through both this and the capitulation guard
    # entirely, landing on _finalize_completed_answer with zero tool
    # calls and no matching pattern. Widened to cover mutation/setup verbs
    # too, not just inspection ones.
    r"install(?:ing)?|writ(?:e|ing)|creat(?:e|ing)|add(?:ing)?|implement(?:ing)?|"
    r"fix(?:ing)?|updat(?:e|ing)|modify(?:ing)?|delet(?:e|ing)|remov(?:e|ing)|"
    r"download(?:ing)?|build(?:ing)?|set(?:ting)?\s+up|configur(?:e|ing)|"
    r"debug(?:ging)?|edit(?:ing)?"
    r")\b",
    re.IGNORECASE,
)

_NARRATED_TOOL_DISPATCH_RE = re.compile(
    r"\b(?:i(?:'ll|\s+will|\s+am\s+going\s+to))\s+"
    r"(?:call|invoke|use)\b[^.!?\n]{0,180}\b(?:registered\s+)?tool(?:s)?\b",
    re.IGNORECASE,
)


def _looks_like_narrated_tool_intent(text: str) -> bool:
    """True only for promises to perform future tool work, not past reports."""
    value = text or ""
    return bool(_NARRATED_TOOL_INTENT_RE.search(value) or _NARRATED_TOOL_DISPATCH_RE.search(value))


# Confirmed live: given an open-ended instruction ("continue until you fix
# everything, don't ask me for confirmation") with no concrete target, a
# weak model can just give up in prose -- zero tool calls, zero repository
# investigation -- and hand back something like "the task is stuck due to
# the lack of a clear next step". Nothing narrated-tool-intent catches this
# (there's no promise of future action to detect), so it fell straight
# through to _finalize_completed_answer and came back as a "completed"
# answer wearing every validator warning at once. An agentic coding tool
# that's just been told to act autonomously should investigate the
# repository itself (tests, lint, TODO/FIXME markers, git status) rather
# than declare itself stuck without ever trying a tool.
_CAPITULATION_RE = re.compile(
    r"lack of a clear next step|no clear next step|unclear next step|"
    r"not sure what (?:to fix|needs? (?:to be )?fix(?:ed|ing)|you (?:want|need) me to)|"
    r"need (?:more information|more details|clarification)\b|"
    r"please (?:specify|clarify)|could you (?:specify|clarify)|"
    r"what would you like me to (?:fix|do)|"
    r"i don'?t have enough information|"
    r"the task is stuck\b",
    re.IGNORECASE,
)


def _looks_like_capitulation(text: str) -> bool:
    """True when the model gave up in prose instead of trying a tool."""
    return bool(_CAPITULATION_RE.search(text or ""))


# Confirmed live (meta/llama-3.1-70b-instruct on NVIDIA NIM): distinct from
# both narrated-intent (a promise of future work) and capitulation (giving
# up), a weak model can fabricate a *past-tense* tool result or tool error
# that never happened -- "the search_code tool has found several
# references to Hono...", or an invented permission refusal ("I encountered
# an access issue... here are the allowed directories: ..." naming
# directories from a different workspace entirely) -- while issuing zero
# real tool calls this round. Neither existing guard catches this because
# the text isn't a promise ("let me...") and isn't giving up ("no clear
# next step..."); it reads as a completed, evidenced answer. Catch it the
# same way: refuse the round, ask for a real tool call, and fall back
# across providers the same one-chance-then-switch way as narrated intent.
_FABRICATED_TOOL_RESULT_RE = re.compile(
    r"the\s+\w+\s+tool\s+(?:has\s+)?(?:found|returned|executed|ran|shows?|revealed|indicates?)|"
    r"(?:the\s+)?results?\s+(?:suggest|indicate)s?\b|"
    r"i\s+encountered\s+an?\s+(?:access|permission)\s+issue|"
    r"(?:here\s+are\s+the\s+)?allowed\s+directories\s+(?:are|include|where\s+i\s+can)|"
    r"i\s+(?:don'?t|do\s+not)\s+have\s+access\s+to\b",
    re.IGNORECASE,
)


def _looks_like_fabricated_tool_result(text: str) -> bool:
    """True when the model reports a past-tense tool result or tool-level
    refusal in prose without any real tool call backing it this round."""
    return bool(_FABRICATED_TOOL_RESULT_RE.search(text or ""))


@dataclass
class _StreamedToolCall:
    call_id: str = ""
    name: str = ""
    arguments: str = ""


_TEXT_TOOL_START = "<tool_call"
_TEXT_TOOL_END_RE = re.compile(r"</tool_call\s*>", re.IGNORECASE)
_TEXT_TOOL_FUNCTION_RE = re.compile(
    r"<function(?:\s*=\s*|\s+name\s*=\s*[\"']?)([A-Za-z_][\w.-]*)[\"']?\s*>",
    re.IGNORECASE,
)
_TEXT_TOOL_PARAMETER_RE = re.compile(
    r"<parameter(?:\s*=\s*|\s+name\s*=\s*[\"']?)([A-Za-z_][\w.-]*)[\"']?\s*>",
    re.IGNORECASE,
)


def _partial_text_tool_prefix_length(text: str) -> int:
    """Keep a possible split ``<tool_call`` prefix between stream chunks."""
    lowered = text.lower()
    for size in range(min(len(lowered), len(_TEXT_TOOL_START) - 1), 0, -1):
        if lowered.endswith(_TEXT_TOOL_START[:size]):
            return size
    return 0


def _parse_text_tool_block(block: str, allowed_names: set[str]) -> Optional[_StreamedToolCall]:
    """Normalize XML-ish tool markup emitted as assistant text.

    Some OpenAI-compatible reasoning models ignore native ``tool_calls`` and
    emit ``<tool_call><function=...><parameter=...>`` instead. Accept only a
    function that was actually offered for this turn; execution still goes
    through the ordinary policy/approval/tool-result pipeline.
    """
    function_match = _TEXT_TOOL_FUNCTION_RE.search(block)
    if function_match is None:
        return None
    name = function_match.group(1)
    if name not in allowed_names:
        return None

    parameters: dict[str, Any] = {}
    matches = list(_TEXT_TOOL_PARAMETER_RE.finditer(block))
    for index, match in enumerate(matches):
        value_start = match.end()
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(block)
        explicit_end = re.search(r"</parameter\s*>", block[value_start:next_start], re.IGNORECASE)
        value_end = value_start + explicit_end.start() if explicit_end else next_start
        value = block[value_start:value_end]
        value = _TEXT_TOOL_END_RE.split(value, maxsplit=1)[0].strip()
        parameters[match.group(1)] = value

    if not parameters:
        return None
    if set(parameters) == {"arguments"}:
        try:
            parsed_arguments = json.loads(str(parameters["arguments"]))
            if isinstance(parsed_arguments, dict):
                parameters = parsed_arguments
        except (TypeError, ValueError):
            pass
    return _StreamedToolCall(
        call_id=f"text_call_{uuid.uuid4().hex[:12]}",
        name=name,
        arguments=json.dumps(parameters),
    )


@dataclass
class _TextToolStreamFilter:
    """Incrementally hide and collect XML-ish textual tool calls."""

    allowed_names: set[str]
    pending: str = ""
    in_tool: bool = False
    tool_blocks: list[str] = field(default_factory=list)

    def feed(self, content: str) -> str:
        self.pending += content
        visible: list[str] = []
        while self.pending:
            if self.in_tool:
                end = _TEXT_TOOL_END_RE.search(self.pending)
                if end is None:
                    break
                self.tool_blocks.append(self.pending[:end.end()])
                self.pending = self.pending[end.end():]
                self.in_tool = False
                continue

            start = self.pending.lower().find(_TEXT_TOOL_START)
            if start >= 0:
                visible.append(self.pending[:start])
                self.pending = self.pending[start:]
                self.in_tool = True
                continue

            keep = _partial_text_tool_prefix_length(self.pending)
            if keep:
                visible.append(self.pending[:-keep])
                self.pending = self.pending[-keep:]
            else:
                visible.append(self.pending)
                self.pending = ""
            break
        return "".join(visible)

    def finish(self) -> tuple[str, list[_StreamedToolCall]]:
        # Only suppress complete, valid, offered tool calls. If the model
        # merely wrote an incomplete/invalid tag, preserve it as text so no
        # content silently disappears.
        parsed: list[_StreamedToolCall] = []
        rejected: list[str] = []
        for block in self.tool_blocks:
            call = _parse_text_tool_block(block, self.allowed_names)
            if call is None:
                rejected.append(block)
            else:
                parsed.append(call)
        trailing = "".join(rejected) + self.pending
        self.pending = ""
        return trailing, parsed


def _select_model(manager: Any, config: Any, task_profile: Any) -> str:
    """manager.select_model(config, task_profile) when the manager supports
    it, falling back to config.default_model otherwise -- same tolerance-
    for-minimal-test-doubles convention as this module's existing
    `hasattr(manager, "fallback_candidates")` checks, since the many
    lightweight fake managers across the test suite only implement the
    handful of ProviderManager methods each specific test actually needs."""
    select = getattr(manager, "select_model", None)
    if select is None:
        return config.default_model
    return select(config, task_profile)


def _select_fallback_model(manager: Any, config: Any, task_profile: Any) -> str:
    """Choose a free fallback model unless paid fallback was opted in."""
    paid = getattr(manager, "paid_fallback_enabled", None)
    if callable(paid) and not paid() and getattr(config, "free_model", None):
        return str(config.free_model)
    return _select_model(manager, config, task_profile)


def _tool_calls_signature(tool_calls: list[_StreamedToolCall]) -> tuple[tuple[str, str], ...]:
    """Order-independent fingerprint of a round's tool calls, used to detect
    the model repeating the exact same request(s) round after round."""
    return tuple(sorted((tc.name, tc.arguments) for tc in tool_calls))


# Width of the rolling window _is_cycling inspects for a repeating period-2
# or period-3 pattern (e.g. read A, read B, read A, read B, ...) -- wide
# enough to distinguish a genuine short cycle from coincidental overlap
# between two otherwise-unrelated rounds, narrow enough to catch it well
# before max_rounds.
_LOOP_WINDOW = 6


def _is_cycling(history: list[tuple[tuple[str, str], ...]]) -> bool:
    """True when the last _LOOP_WINDOW rounds are an exact repeating cycle
    of period 2 or 3. Distinct from the consecutive-identical-round guard,
    which only ever catches a period-1 repeat (the exact same call twice in
    a row) -- a model alternating between two or three distinct calls never
    repeats one call back-to-back, so that guard alone never fires for it."""
    if len(history) < _LOOP_WINDOW:
        return False
    window = history[-_LOOP_WINDOW:]
    for period in (2, 3):
        if all(window[i] == window[i % period] for i in range(len(window))):
            return True
    return False


async def _stream_one_completion(
    client, *, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
    renderer: StreamRenderer, reasoning_effort: Optional[str] = None, emit: bool = True,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> tuple[str, list[_StreamedToolCall], Optional[str]]:
    """Stream one chat-completions call, forwarding text deltas to the
    renderer as they arrive and accumulating any tool_calls deltas by index
    (the standard OpenAI-compatible streaming tool-call pattern: id/name
    only arrive in the first delta for a given index, arguments arrive as
    incremental string fragments across many deltas).

    `reasoning_effort`, when the resolved provider is in
    providers.REASONING_EFFORT_CAPABLE_PROVIDERS, makes some models stream a
    separate `reasoning_content` delta ahead of the real answer (confirmed
    live against NVIDIA NIM) -- forwarded to the renderer as `reasoning_delta`
    events so it can show real "thought for Xs" timing, but never mixed into
    the returned answer text.

    The third return value is the raw `finish_reason` from the provider's
    final chunk (e.g. "stop", "tool_calls", "length") -- confirmed live: a
    reasoning-heavy answer (e.g. a full-stack audit) can hit
    MAX_TOKENS_PER_REQUEST mid-generation, and finish_reason=="length" was
    computed by provider_protocols.py's normalize_stream_chunk but never
    read anywhere -- a truncated, mid-sentence partial answer with no
    tool_calls was indistinguishable from a genuinely complete one, so it
    was accepted as the final answer instead of being continued. Callers
    that don't need it (e.g. the empty-continuation recovery path, which
    already retries on its own condition) may discard it.

    `emit=False` suppresses forwarding reasoning/assistant deltas to the
    renderer -- used for internal, non-user-facing completions (e.g. plan
    generation) whose raw output (JSON) would otherwise print as if it
    were the model's real answer."""
    content_parts: list[str] = []
    tool_calls_by_index: dict[int, _StreamedToolCall] = {}
    finish_reason: Optional[str] = None
    offered_tool_names = {
        str((tool.get("function") or {}).get("name") or "")
        for tool in tools
        if isinstance(tool, dict)
    }
    text_tool_filter = _TextToolStreamFilter(offered_tool_names)

    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": 0.2,
        "max_tokens": MAX_TOKENS_PER_REQUEST,
    }
    # Do not send an empty tools array. Some OpenAI-compatible servers and
    # small local models behave differently merely because tool mode is
    # present, even when no tool is useful for the current turn.
    if tools:
        request_kwargs["tools"] = tools
        request_kwargs["tool_choice"] = "auto"
    if reasoning_effort:
        request_kwargs["reasoning_effort"] = reasoning_effort

    stream = await client.chat.completions.create(**request_kwargs)
    # Bounded rolling tail, not the full accumulated content -- checking a
    # fixed-size window on every chunk keeps this O(1) per chunk regardless
    # of how long a genuinely long (non-degenerate) response gets, instead
    # of re-scanning everything accumulated so far on every delta.
    tail_buffer = ""
    pending_content = ""
    quality_failure_reason: Optional[str] = None

    def forward(content: str) -> None:
        if not content:
            return
        content_parts.append(content)
        if progress_callback is not None:
            progress_callback(content)
        if emit:
            renderer.handle_event({"event_type": "assistant_delta", "payload": {"content": content}})

    stream_error: Optional[Exception] = None
    stream_iterator = stream.__aiter__()
    while True:
        try:
            chunk = await stream_iterator.__anext__()
        except StopAsyncIteration:
            break
        except Exception as exc:
            # Preserve the last clean quarantined tail before reconnecting.
            # Without this, any interrupted response shorter than the quality
            # lag vanished from both the terminal and the durable checkpoint.
            stream_error = exc
            break
        for event in normalize_stream_chunk(chunk, provider=model.split("/", 1)[0] if "/" in model else None, model=model):
            if event.event_type.value == "reasoning_delta":
                reasoning = str(event.payload.get("content") or "")
                if reasoning and emit:
                    renderer.handle_event({"event_type": "reasoning_delta", "payload": {"content": reasoning}})
            elif event.event_type.value == "assistant_delta":
                content = str(event.payload.get("content") or "")
                if content:
                    visible_content = text_tool_filter.feed(content)
                    if visible_content:
                        pending_content += visible_content
                    tail_buffer = (tail_buffer + visible_content)[-_DEGENERATE_REPETITION_TAIL_WINDOW:]
                    if _DEGENERATE_REPETITION_RE.search(tail_buffer):
                        quality_failure_reason = "degenerate_repetition"
                        break
                    if (
                        len(_CONVERSATION_ECHO_USER_MARKER.findall(tail_buffer)) >= 3
                        and len(_CONVERSATION_ECHO_ASSISTANT_MARKER.findall(tail_buffer)) >= 3
                    ):
                        quality_failure_reason = "conversation_echo"
                        break
                    if _has_repeated_long_segment(tail_buffer):
                        quality_failure_reason = "repeated_content"
                        break
                    if _corrupted_lexical_stream_index(tail_buffer) is not None:
                        quality_failure_reason = "corrupted_output"
                        break
                    flush_count = max(0, len(pending_content) - _STREAM_QUALITY_LAG_CHARS)
                    if flush_count:
                        forward(pending_content[:flush_count])
                        pending_content = pending_content[flush_count:]
            elif event.event_type.value == "tool_call_delta":
                index = int(event.payload.get("index") or 0)
                slot = tool_calls_by_index.setdefault(index, _StreamedToolCall())
                slot.call_id = str(event.payload.get("id") or slot.call_id)
                slot.name = str(event.payload.get("name") or slot.name)
                slot.arguments += str(event.payload.get("arguments") or "")
            elif event.event_type.value == "done":
                finish_reason = event.payload.get("reason") or finish_reason
        if quality_failure_reason:
            break

    if quality_failure_reason:
        # Stop pulling further chunks from the provider entirely -- letting
        # the stream run to MAX_TOKENS_PER_REQUEST once a loop is already
        # confirmed just wastes tokens/time/memory on more of the same.
        try:
            await stream.close()
        except Exception:
            pass
        finish_reason = quality_failure_reason
        if emit:
            diagnosis = (
                "Detected corrupted provider token output; discarding it and requesting a clean route."
                if quality_failure_reason == "corrupted_output"
                else "Detected a repeated conversation transcript; discarding it and requesting a clean route."
                if quality_failure_reason == "conversation_echo"
                else "Detected a repeated analysis/code segment; discarding it and requesting a clean route."
                if quality_failure_reason == "repeated_content"
                else "Detected the model repeating itself in a loop; discarding that completion."
            )
            renderer.handle_event({
                "event_type": "diagnostics",
                "payload": {"content": diagnosis},
            })

    trailing_content, textual_calls = text_tool_filter.finish()
    if not quality_failure_reason:
        pending_content += trailing_content
        combined_tail = (tail_buffer + trailing_content)[-_DEGENERATE_REPETITION_TAIL_WINDOW:]
        if _DEGENERATE_REPETITION_RE.search(combined_tail):
            quality_failure_reason = finish_reason = "degenerate_repetition"
        elif _corrupted_lexical_stream_index(combined_tail) is not None:
            quality_failure_reason = finish_reason = "corrupted_output"
        else:
            forward(pending_content)
            pending_content = ""
    if stream_error is not None and not quality_failure_reason:
        raise stream_error
    ordered_calls = (
        [] if quality_failure_reason
        else [tool_calls_by_index[i] for i in sorted(tool_calls_by_index)] + textual_calls
    )
    final_content = "".join(content_parts)
    return final_content, ordered_calls, finish_reason


# Internal context rollover: a segment can be checkpointed out to durable
# evidence storage and restarted from a compact continuation package this
# many times per turn before the agent genuinely gives up. Bounded so a
# pathological task (one whose minimum viable context itself never fits)
# fails cleanly instead of looping forever.
MAX_CONTEXT_ROLLOVERS_PER_TURN = 3

RETRIEVE_EVIDENCE_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "retrieve_evidence",
        "description": (
            "Retrieve full detail from a prior context-rollover evidence segment "
            "(referenced by an evidence_id mentioned in a CONTEXT ROLLOVER system "
            "message). Use this when you need exact prior tool output or file "
            "content that was checkpointed out of the working context during a "
            "rollover, rather than the compact summary that replaced it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "evidence_id": {"type": "string", "description": "The evidence_id to retrieve"},
            },
            "required": ["evidence_id"],
        },
    },
}


SWARM_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "delegate_parallel_tasks",
        "description": (
            "Fan out 2+ genuinely independent sub-objectives to run concurrently "
            "(e.g. 'audit these 3 unrelated modules', 'look into these 3 separate "
            "bug reports'), instead of investigating them one at a time yourself. "
            "Defaults to read-only sub-tasks -- pass mutate=true only when the "
            "sub-objectives genuinely need to edit files AND the user's current "
            "approval mode already auto-approves edits (this call fails with a "
            "clear error otherwise, since a sub-task can't itself prompt for "
            "approval). Do not use this for a single objective, or for objectives "
            "that depend on each other's results."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "minItems": 2,
                    "items": {
                        "anyOf": [
                            {"type": "string"},
                            {
                                "type": "object",
                                "properties": {
                                    "objective": {"type": "string"},
                                    "agent_type": {
                                        "type": "string",
                                        "description": (
                                            "Optional named subagent type (see /agent-types) to run "
                                            "this one sub-task with -- its own system prompt and, if "
                                            "configured, its own model/provider. Omit for the default."
                                        ),
                                    },
                                },
                                "required": ["objective"],
                            },
                        ],
                    },
                    "description": (
                        "One entry per independent sub-objective -- either a plain string, or an "
                        "object with objective (+ optional agent_type to use a declared subagent type)."
                    ),
                },
                "mutate": {
                    "type": "boolean",
                    "description": "Allow sub-tasks to edit files. Defaults to false (read-only/analysis only).",
                },
            },
            "required": ["tasks"],
        },
    },
}


def _parse_swarm_tasks(raw_tasks: list[Any]) -> tuple[list[str], list[Optional[str]]]:
    """Split delegate_parallel_tasks's `tasks` argument (each item either a
    plain string or an {"objective", "agent_type"} object, per
    SWARM_TOOL_SCHEMA's anyOf) into parallel (objectives, agent_types)
    lists AgentManager.execute_tasks/run_swarm expect. A malformed item
    (missing objective, wrong type) is skipped rather than crashing the
    whole delegated call."""
    objectives: list[str] = []
    agent_types: list[Optional[str]] = []
    for item in raw_tasks:
        if isinstance(item, dict):
            objective = str(item.get("objective") or "").strip()
            agent_type = item.get("agent_type")
            agent_type = str(agent_type).strip() if agent_type else None
        else:
            objective = str(item or "").strip()
            agent_type = None
        if not objective:
            continue
        objectives.append(objective)
        agent_types.append(agent_type or None)
    return objectives, agent_types


def _summarise_progress_for_rollover(session_id: int) -> str:
    """Compact summary of task progress so far, drawn from durable
    SessionState -- NOT the provider prompt being reset -- so a rollover's
    continuation package still knows what already happened."""
    state = local_state.get_session_state(session_id)
    lines: list[str] = []
    if state.inspected_files:
        lines.append(f"Files inspected so far: {', '.join(list(state.inspected_files)[-15:])}")
    if state.modified_files:
        recent = [str(m.get("path")) for m in state.modified_files[-10:]]
        lines.append(f"Files modified so far: {', '.join(recent)}")
    recent_actions = state.completed_actions[-10:]
    if recent_actions:
        outcomes = ", ".join(f"{a.get('type')}={a.get('status')}" for a in recent_actions)
        lines.append(f"Recent tool outcomes: {outcomes}")
    if state.unresolved_issues:
        details = "; ".join(str(issue.get("detail", "")) for issue in state.unresolved_issues[-5:])
        lines.append(f"Unresolved: {details}")
    return "\n".join(lines) if lines else "No prior tool activity recorded yet."


def _plan_created_payload(plan: Any, *, title: str) -> dict[str, Any]:
    """render.py's plan_created handler expects {"title", "items": [{"step","status"}]}."""
    return {
        "title": title,
        "items": [{"step": step.name, "status": step.status} for step in plan.steps],
        "assumptions": list(getattr(plan, "assumptions", []) or []),
        "risks": list(getattr(plan, "risks", []) or []),
        "validation_criteria": list(getattr(plan, "validation_criteria", []) or []),
    }


def _plan_message_content(plan: Any, *, heading: str) -> str:
    lines = [heading]
    lines += [f"{step.index}. {step.name}" for step in plan.steps]
    if plan.assumptions:
        lines.append("Assumptions: " + "; ".join(plan.assumptions))
    if plan.risks:
        lines.append("Risks: " + "; ".join(plan.risks))
    return "\n".join(lines)



def _validate_reasoning_plan_scope(
    plan: Any,
    *,
    scope_roots: list[Path],
    objective: str,
) -> tuple[Optional[Any], list[str]]:
    """Remove plan steps that reference unauthorised or non-canonical roots.

    Provider-generated plans are untrusted model output. The scope prompt is
    advisory, so every generated step must also pass deterministic validation
    before it is persisted or displayed.

    Steps without filesystem paths remain eligible. Steps containing absolute
    paths outside ``scope_roots`` are removed. Hidden Claude worktrees are
    excluded from ordinary canonical-repository audits unless the objective
    explicitly requests a worktree.
    """
    if plan is None or not getattr(plan, "steps", None):
        return None, []

    resolved_roots = [
        Path(root).expanduser().resolve()
        for root in scope_roots
    ]
    lowered_objective = (objective or "").lower()
    removed: list[str] = []
    retained: list[Any] = []

    # When all targets are immediate children of one parent, identify sibling
    # project names so relative references such as "llama.cpp/pyproject.toml"
    # cannot bypass the absolute-path check.
    sibling_projects: dict[str, Path] = {}
    parent_candidates = {root.parent for root in resolved_roots}

    if len(parent_candidates) == 1:
        common_parent = next(iter(parent_candidates))
        try:
            for child in common_parent.iterdir():
                if not child.is_dir():
                    continue
                resolved_child = child.resolve()
                if _is_project_root(resolved_child):
                    sibling_projects[child.name.lower()] = resolved_child
        except OSError:
            sibling_projects = {}

    allowed_names = {root.name.lower() for root in resolved_roots}

    for step in list(plan.steps):
        name = str(getattr(step, "name", "") or "").strip()
        if not name:
            removed.append("<empty plan step>")
            continue

        reject = False

        # Validate every absolute path appearing in the step.
        for raw_path in re.findall(
            r"(?<![\w.-])/(?:[A-Za-z0-9._~+\-]+/)*[A-Za-z0-9._~+\-]+",
            name,
        ):
            candidate = Path(raw_path.rstrip(".,;:)]}")).expanduser().resolve()

            if not any(_is_within(candidate, root) for root in resolved_roots):
                reject = True
                break

            normalised = candidate.as_posix().lower()
            if (
                "/.claude/worktrees/" in normalised
                and "worktree" not in lowered_objective
                and str(candidate).lower() not in lowered_objective
            ):
                reject = True
                break

        if reject:
            removed.append(name)
            continue

        # Reject relative references beginning with an unselected sibling
        # project name, for example "llama.cpp/requirements.txt".
        lowered_name = name.lower()

        for sibling_name in sibling_projects:
            if sibling_name in allowed_names:
                continue

            if re.search(
                rf"(?<![A-Za-z0-9._-]){re.escape(sibling_name)}(?:/|\\)",
                lowered_name,
            ):
                reject = True
                break

        if reject:
            removed.append(name)
            continue

        retained.append(step)

    if not retained:
        return None, removed

    # Reindex after filtering so the rendered and persisted plan remains
    # internally consistent.
    for index, step in enumerate(retained, start=1):
        step.index = index
        if step.status not in {"pending", "in_progress", "completed", "failed"}:
            step.status = "pending"

    plan.steps = retained
    return plan, removed


# Classifications that should reach an already-running standalone turn
# (mirrors cli.py's `queue` command choices) vs ones that only ever meant
# "wait for the next turn" even before this existed (e.g. "reprioritise"
# only makes sense against a not-yet-started backlog).
_LIVE_INSTRUCTION_CLASSIFICATIONS = {"append", "follow_up", "clarification", "replace", "cancel", "pause", "exit"}
_LIVE_INSTRUCTION_STOP_CLASSIFICATIONS = {"cancel", "pause", "exit"}


def _claim_live_queued_instructions(session_id: int) -> list[dict[str, Any]]:
    """Every still-`queued` instruction in this session's on-disk queue,
    claimed (marked `running`) so a concurrent second `tamfis-code queue
    ...` process -- or the next REPL turn, if this one finishes first --
    doesn't also pick the same one up. Ordered oldest/highest-priority
    first, matching `enqueue_instruction`'s own sort."""
    state = local_state.get_session_state(session_id)
    claimed = [
        dict(item) for item in state.queued_user_instructions
        if item.get("status") == "queued" and item.get("classification") in _LIVE_INSTRUCTION_CLASSIFICATIONS
    ]
    for item in claimed:
        local_state.update_instruction(session_id, str(item.get("id")), "running")
    return claimed


def _apply_live_queued_instruction(
    live: dict[str, Any], *, session_id: int, working_messages: list[dict[str, Any]], renderer: StreamRenderer,
) -> Optional["TaskOutcome"]:
    """Apply one claimed (already `running`-marked) live instruction.
    Returns a TaskOutcome to end the turn immediately (cancel/pause), or
    None to keep going after splicing the instruction into the conversation
    as a new user turn -- the model sees it on its very next completion
    request, so it can genuinely revise its plan/approach mid-task instead
    of only after the turn ends. Either way, marks the instruction
    `completed` -- it must not stay `running` forever once handled."""
    instruction_id = str(live.get("id") or "")
    classification = str(live.get("classification") or "append")
    text = str(live.get("text") or "")
    local_state.update_instruction(session_id, instruction_id, "completed")

    if classification in _LIVE_INSTRUCTION_STOP_CLASSIFICATIONS:
        renderer.handle_event({
            "event_type": "diagnostics",
            "payload": {
                "content": (
                    f"{classification.capitalize()} requested via a live queued instruction "
                    f"({instruction_id}) from another terminal; stopping this turn."
                )
            },
        })
        if classification == "exit":
            return TaskOutcome(status="exited", error="Exit requested by user")
        summary = f"Task {classification}d by user request" + (f": {text}" if text else ".")
        return TaskOutcome(status="cancelled", error=summary)

    renderer.handle_event({
        "event_type": "diagnostics",
        "payload": {"content": f"Live instruction received mid-task ({classification}, {instruction_id}): {text}"},
    })
    working_messages.append({
        "role": "user",
        "content": (
            f"[Live update sent while you were already working on this task, from another "
            f"terminal -- classification={classification}] {text}\n"
            "Take this into account for the rest of the task -- revise your plan/approach now "
            "if it changes what you should do next, rather than only noting it at the end."
        ),
    })
    return None


_PLANNING_MANIFEST_NAMES = {
    "package.json", "pyproject.toml", "pytest.ini", "setup.cfg", "setup.py",
    "requirements.txt", "Cargo.toml", "go.mod", "composer.json", "pom.xml",
    "build.gradle", "build.gradle.kts", "alembic.ini", "vite.config.ts",
    "vite.config.js", "tsconfig.json",
}
_PLANNING_SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "dist", "build", ".next",
    ".pytest_cache", "__pycache__", ".mypy_cache", ".ruff_cache", "coverage",
    ".venv", "venv", "vendor",
}
_PLANNING_STOP_WORDS = {
    "check", "full", "stack", "status", "fix", "bugs", "bug", "all", "and",
    "the", "this", "that", "with", "from", "into", "without", "rely", "please",
    "workspace", "pipeline", "codebase", "project", "system", "complete",
}


def _planning_keywords(objective: str) -> list[str]:
    """Return a small set of objective terms useful for filename discovery."""
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_-]{2,}", objective.casefold())
    ordered: list[str] = []
    for token in tokens:
        normalised = token.replace("-", "_")
        if normalised in _PLANNING_STOP_WORDS or normalised in ordered:
            continue
        ordered.append(normalised)
    # Mission/workspace/pipeline requests commonly use these exact path terms;
    # retain them even though generic stop-word filtering avoids noisy matches.
    for explicit in ("mission", "missions", "workspace", "pipeline"):
        if explicit in objective.casefold() and explicit not in ordered:
            ordered.append(explicit)
    return ordered[:12]


def _iter_planning_files(root: Path, *, limit: int = 6000):
    """Yield repository files deterministically without walking generated trees."""
    seen = 0
    stack = [root]
    while stack and seen < limit:
        current = stack.pop()
        try:
            entries = sorted(current.iterdir(), key=lambda item: item.name.casefold(), reverse=True)
        except OSError:
            continue
        for entry in entries:
            if entry.name in _PLANNING_SKIP_DIRS:
                continue
            try:
                if entry.is_dir():
                    stack.append(entry)
                elif entry.is_file():
                    seen += 1
                    yield entry
                    if seen >= limit:
                        return
            except OSError:
                continue


def _package_json_facts(path: Path) -> tuple[list[str], list[str]]:
    """Read only declared npm scripts; never infer commands from package presence."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
        return [], []
    scripts = payload.get("scripts") if isinstance(payload, dict) else None
    if not isinstance(scripts, dict):
        return [], []
    names = [str(name) for name in scripts.keys()]
    commands = [f"npm run {name}" for name in names if name not in {"start", "test"}]
    if "test" in names:
        commands.append("npm test")
    if "start" in names:
        commands.append("npm start")
    return names[:30], commands[:30]


def _build_planning_reconnaissance(
    workspace_root: str, scope_roots: list[Path], objective: str,
) -> str:
    """Perform deterministic, read-only discovery before publishing a plan.

    This deliberately does not execute commands. It records exact roots, manifests,
    declared scripts, top-level structure, and objective-matching paths so the model
    cannot invent npm/Alembic/service operations before seeing repository evidence.
    """
    lines: list[str] = []
    keywords = _planning_keywords(objective)
    unique_roots: list[Path] = []
    for candidate in scope_roots or [Path(workspace_root)]:
        resolved = candidate.resolve()
        if resolved not in unique_roots:
            unique_roots.append(resolved)

    for root in unique_roots:
        lines.append(f"ROOT: {root}")
        if not root.exists():
            lines.append("  status: missing")
            continue
        if not root.is_dir():
            lines.append("  status: not a directory")
            continue
        lines.append("  status: accessible directory")
        try:
            top_level = sorted(
                item.name for item in root.iterdir()
                if item.name not in _PLANNING_SKIP_DIRS
            )[:40]
        except OSError as exc:
            lines.append(f"  listing_error: {exc}")
            continue
        lines.append("  top_level: " + (", ".join(top_level) if top_level else "(empty)"))

        manifests: list[str] = []
        relevant_paths: list[str] = []
        declared_scripts: list[str] = []
        verified_commands: list[str] = []
        for file_path in _iter_planning_files(root):
            try:
                relative = file_path.relative_to(root).as_posix()
            except ValueError:
                relative = str(file_path)
            if file_path.name in _PLANNING_MANIFEST_NAMES and len(manifests) < 40:
                manifests.append(relative)
                if file_path.name == "package.json":
                    scripts, commands = _package_json_facts(file_path)
                    declared_scripts.extend(f"{relative}:{name}" for name in scripts)
                    verified_commands.extend(f"{relative} -> {command}" for command in commands)
            folded = relative.casefold().replace("-", "_")
            if keywords and any(keyword in folded for keyword in keywords):
                if len(relevant_paths) < 60:
                    relevant_paths.append(relative)

        lines.append("  manifests: " + (", ".join(manifests) if manifests else "none found"))
        if declared_scripts:
            lines.append("  declared_scripts: " + ", ".join(declared_scripts[:40]))
        else:
            lines.append("  declared_scripts: none found")
        if verified_commands:
            lines.append("  manifest_backed_commands: " + ", ".join(verified_commands[:40]))
        else:
            lines.append("  manifest_backed_commands: none found")
        lines.append(
            "  objective_matching_paths: "
            + (", ".join(relevant_paths[:60]) if relevant_paths else "none found by filename")
        )

    lines.append("PLANNING RULE: do not propose a command unless it appears above as manifest-backed.")
    lines.append("PLANNING RULE: first inspect the named objective-matching paths and manifests; do not guess services or migrations.")
    return "\n".join(lines)


async def _attempt_reasoning_plan(
    client, *, model: str, objective: str, task_profile: Any, session_id: int,
    renderer: StreamRenderer, reconnaissance_summary: Optional[str] = None,
    evidence_summary: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    scope_roots: Optional[list[Path]] = None,
) -> Optional[Any]:
    """Ask the resolved provider for a plan grounded in the real objective
    and real workspace facts (and, for a revision, real evidence gathered
    so far) -- replacing/refining the deterministic template plan.

    Never raises and never blocks the turn on failure: a malformed or
    failed planning response silently returns None, and the caller keeps
    whatever plan it already had (the deterministic template from
    orchestrator.begin(), or the still-valid prior reasoning plan for a
    failed revision attempt).
    """
    repository_context = local_state.get_session_state(session_id).repository_context or {}
    prompt_messages = build_reasoning_plan_prompt(
        objective, task_profile, repository_context,
        reconnaissance_summary=reconnaissance_summary,
        evidence_summary=evidence_summary,
    )
    try:
        content, _tool_calls, finish_reason = await _stream_one_completion(
            client, model=model, messages=prompt_messages, tools=[],
            renderer=renderer, reasoning_effort=reasoning_effort, emit=False,
        )
    except Exception as exc:
        renderer.handle_event({
            "event_type": "diagnostics",
            "payload": {"content": f"Planning request failed ({exc}); using the existing plan."},
        })
        return None
    if finish_reason in {"degenerate_repetition", "conversation_echo", "repeated_content", "corrupted_output"}:
        renderer.handle_event({
            "event_type": "diagnostics",
            "payload": {
                "content": (
                    f"Planning output was rejected ({finish_reason}); the repeated/corrupt plan was not shown "
                    "or archived as a completed answer. Continuing with the durable execution plan."
                )
            },
        })
        return None
    plan = parse_reasoning_plan(
        content,
        objective=objective,
        reconnaissance_summary=reconnaissance_summary,
        workspace_summary=repository_context,
        scope_roots=scope_roots,
    )
    if plan is None:
        renderer.handle_event({
            "event_type": "diagnostics",
            "payload": {"content": "Planning response wasn't a usable plan; using the existing plan."},
        })
        return None
    return plan


def _perform_context_rollover(
    working_messages: list[dict[str, Any]],
    *,
    objective: str,
    scope_roots: list[Path],
    scope_message: dict[str, Any],
    session_id: int,
) -> list[dict[str, Any]]:
    """Checkpoint the full working segment to durable evidence storage, then
    return a minimum-viable continuation context so the SAME task keeps
    going in a fresh internal provider context instead of failing the turn.

    This is the internal-context-rollover step: full messages/tool
    calls/results are never discarded, only moved outside the provider
    prompt (see evidence.py) with a reference the model can retrieve on
    demand via the retrieve_evidence tool.
    """
    progress = _summarise_progress_for_rollover(session_id)
    # evidence_store keeps the FULL, untruncated objective and messages --
    # only what gets re-embedded into the new continuation below is bounded.
    evidence_id = evidence_store.store_segment(
        session_id, objective=objective, messages=working_messages, summary=progress,
    )
    local_state.checkpoint(
        session_id, reason="context_rollover",
        summary=f"Rolled over internal context (evidence_id={evidence_id}). {progress}",
    )
    leading_system = (
        working_messages[0] if working_messages and working_messages[0].get("role") == "system"
        else {"role": "system", "content": "You are a coding agent working directly in a real local repository via tool calls."}
    )
    # Confirmed live: when the objective itself (not tool history) is what
    # drove the turn over budget -- e.g. a large pasted log/diff as the
    # request -- re-embedding it here in full made the new continuation
    # roughly the same size as what was just rolled over, so the very next
    # budget check failed again immediately, burning rollover attempts
    # without ever actually shrinking anything. Bound it the same way
    # oversized tool output is bounded; the full text is never lost -- it's
    # in evidence_id above, retrievable via retrieve_evidence.
    bounded_objective = _bounded_text(objective, head=1500, tail=500, label="objective compacted")
    continuation = {
        "role": "system",
        "content": (
            "CONTEXT ROLLOVER: the working context for this task grew too large for the "
            "provider and was checkpointed, not abandoned -- this is the SAME task, "
            "continuing in a fresh internal context.\n"
            f"Objective: {bounded_objective}\n"
            f"Workspace scope: {', '.join(str(path) for path in scope_roots)}\n"
            f"Progress so far:\n{progress}\n"
            f"Full evidence from before this rollover -- including the exact, untruncated "
            f"objective text above if it was compacted -- is stored under "
            f"evidence_id={evidence_id}. Call retrieve_evidence with this id if you need the "
            "exact prior wording, tool output, or file content back. Continue the task -- do "
            "not redo work already completed above, and do not tell the user to start over or "
            "run /clear; this rollover is transparent to them."
        ),
    }
    latest_user = {"role": "user", "content": bounded_objective}
    return [leading_system, scope_message, continuation, latest_user]


async def run_local_agent_turn(
    manager: ProviderManager,
    provider: ProviderType,
    model: Optional[str],
    messages: list[dict[str, Any]],
    console: Console,
    renderer: StreamRenderer,
    *,
    workspace_root: str,
    session_id: int,
    approval_policy: str = "ask",
    interactive: bool = True,
    max_rounds: int = MAX_AGENT_ROUNDS,
    read_only: bool = False,
    cli_config: Optional[Config] = None,
    allow_swarm_tool: bool = False,
    attachment_paths: tuple[str, ...] = (),
    image_content_blocks: Optional[list[dict[str, Any]]] = None,
) -> TaskOutcome:
    """Run one user turn to completion against a directly-called provider,
    executing tool calls locally (full tool set, approval-gated) instead of
    delegating to a Remote Workspace backend. Mirrors run_ai_task_and_stream's
    contract (same TaskOutcome shape) so cli.py/interactive.py can drive
    either the local or (while it still exists) remote path interchangeably.

    `read_only=True` restricts both the tool schema offered to the model AND
    (defense in depth, in case a model requests a tool it wasn't offered)
    execution itself to safety.READ_ONLY_TOOLS -- used for chat/audit/plan
    modes, which must never mutate the workspace.

    `allow_swarm_tool=True` additionally offers delegate_parallel_tasks
    (swarm.run_swarm) for genuinely independent sub-objectives -- only ever
    set True at real top-level call sites (cli.py/interactive.py), never
    from within a delegated sub-task's own turn (DelegatedCodingAgent.execute
    leaves it at the default False), which gives a hard structural depth-1
    recursion cap instead of a runtime counter.

    `image_content_blocks`, when given (pre-built OpenAI-style
    `{"type": "image_url", ...}` entries -- see cli.py's
    `_build_vision_content_blocks`), are spliced into the most recent user
    message for every provider call in this turn whose resolved route
    actually supports vision (see `_messages_with_vision_content`); routes
    that don't support vision never receive them, and the attached image(s)
    remain visible only via the plain-text attachment note already added to
    `messages` (path only, no pixel content) for those routes.
    """
    incoming_objective = _latest_user_text(messages)
    resume_requested = _is_resume_request(incoming_objective)
    prior_state = (
        _select_resume_state(session_id, workspace_root)
        if resume_requested else local_state.get_session_state(session_id)
    )
    prior_checkpoint = prior_state.turn_checkpoint or {}
    resumed_from_checkpoint = bool(
        prior_checkpoint.get("messages")
        and _is_real_resume_objective(str(prior_checkpoint.get("objective") or ""))
        and resume_requested
    )
    resumed_from_legacy = False
    legacy_objective = ""
    if resumed_from_checkpoint:
        resumed_messages = _close_interrupted_tool_calls(
            list(prior_checkpoint.get("messages") or [])
        )
        partial = str(prior_checkpoint.get("partial_assistant") or "")
        if partial:
            resumed_messages.append({"role": "assistant", "content": partial})
        # Keep the user's new continuation directive explicit.  It may name
        # selected steps ("proceed with 1, 2, and 3"), so it must not be
        # replaced by a generic resume instruction.
        messages = [*resumed_messages, {"role": "user", "content": incoming_objective}]
    elif resume_requested:
        legacy_messages, legacy_objective = _legacy_resume_messages(prior_state, incoming_objective)
        if legacy_messages:
            messages = legacy_messages
            resumed_from_legacy = True
    elif prior_state.conversation_history and not any(
        message.get("role") == "assistant" for message in messages
    ):
        # One-shot invocations start a fresh Python process and traditionally
        # sent only the newest prompt. Rehydrate completed session history so
        # ordinary follow-ups retain the same memory as the interactive REPL.
        leading_system = [message for message in messages if message.get("role") == "system"]
        current_non_system = [message for message in messages if message.get("role") != "system"]
        messages = [*leading_system, *prior_state.conversation_history, *current_non_system]

    # Emit lifecycle events before any provider/network operation.  Without
    # these, the Rich live panel remains at its constructor default (idle)
    # while get_client()/chat.completions.create() is resolving or waiting.
    renderer.handle_event({"event_type": "task_started", "payload": {"mode": "local"}})
    renderer.handle_event({"event_type": "context_loading", "payload": {"workspace_root": workspace_root}})

    # One id for every mutation this turn makes, so a multi-file edit can
    # later be reverted together (safety.revert_transaction) instead of
    # one mutation_id at a time with no way to discover which ids
    # belonged to the same turn.
    mcp_server = MCPServer(
        workspace_root=workspace_root, session_id=session_id,
        console=console, renderer=renderer, interactive=interactive,
        transaction_id=f"turn_{uuid.uuid4().hex[:12]}",
        attachment_paths=list(attachment_paths),
        allowed_workspace_roots=list(prior_state.allowed_workspaces or [workspace_root]),
    )
    # Read fresh once per turn (not cached across turns/process lifetime) so
    # editing hooks.toml takes effect on the next turn without a restart.
    configured_hooks = load_hooks(workspace_root)
    recovered_objective = (
        _checkpoint_resume_objective(prior_checkpoint)
        if resumed_from_checkpoint else (legacy_objective if resumed_from_legacy and legacy_objective else "")
    )
    # Confirmed live: a legacy (pre-v0.4.28) resume's inferred_objective is a
    # best-guess reconstruction from old completed_actions/history and can be
    # something totally unrelated to what the user just typed this turn (a
    # low-complexity leftover question, say). Using it alone as `objective`
    # feeds classify_task stale text instead of the actual new instruction --
    # confirmed live: "continue until you fix everything, don't ask me for
    # confirmation" resumed onto a stale QUESTION-classified objective, so
    # task_profile.requires_tools came back False, the capitulation guard's
    # `task_profile.requires_tools` gate never engaged, and a give-up
    # response sailed straight through as a "completed" answer. Appending
    # the fresh instruction (classify_task is substring-based, so this is
    # additive, not a replacement) keeps genuine multi-step continuation
    # context while still reclassifying for what's actually being asked now.
    objective = (
        recovered_objective
        if recovered_objective and recovered_objective.strip().casefold() == incoming_objective.strip().casefold()
        else (
            f"{recovered_objective}\n\nAdditional user context: {incoming_objective}"
            if recovered_objective else incoming_objective
        )
    )
    orchestrator = AgentOrchestrator(
        session_id=session_id, workspace_root=workspace_root, emit=renderer.handle_event
    )
    orchestration = orchestrator.begin(objective=objective, messages=messages, read_only=read_only)
    task_profile = orchestration.profile
    selected_tool_names = allowed_tools(task_profile, read_only=read_only)
    tools: list[dict[str, Any]] = (
        mcp_server.tool_schemas_openai(names=selected_tool_names) if selected_tool_names else []
    )
    # retrieve_evidence is always safe (a local, read-only lookup by id) and
    # only useful once a rollover has actually happened; offering it
    # whenever any other tool is offered costs one small schema entry and
    # means the model never has to be told about it mid-turn.
    if tools:
        tools = [*tools, RETRIEVE_EVIDENCE_TOOL_SCHEMA]
        if allow_swarm_tool and not read_only and cli_config is not None and cli_config.enable_subagent_delegation:
            tools = [*tools, SWARM_TOOL_SCHEMA]

    working_messages = list(orchestration.context.messages if orchestration.context else messages)
    # Approved roots grant access to explicitly named external paths; they
    # are not additional active audit targets. Keep the current --cwd as the
    # task scope so unrelated approved projects are never mixed into it.
    scope_roots = _detect_workspace_scope(workspace_root, objective)
    _apply_mcp_task_scope(mcp_server, scope_roots)
    scope_message = {
        "role": "system",
        "content": _scope_instruction(workspace_root, scope_roots),
    }
    # Put the scope rule immediately after the leading system instruction so
    # it survives later compaction and remains authoritative in every round.
    insert_at = 1 if working_messages and working_messages[0].get("role") == "system" else 0
    working_messages.insert(insert_at, scope_message)
    working_messages.insert(insert_at + 1, {
        "role": "system",
        "content": FINAL_RESPONSE_FORMAT_INSTRUCTION,
    })
    if resumed_from_checkpoint or resumed_from_legacy or _requests_autonomous_execution(incoming_objective):
        working_messages.insert(insert_at + 1, {"role": "system", "content": RESUME_EXECUTION_INSTRUCTION})
    checkpoint_mode = str(prior_checkpoint.get("mode") or ("read_only" if read_only else "execute"))
    checkpoint_partial_parts: list[str] = []
    last_checkpoint_at = 0.0

    def _persist_turn_checkpoint(
        *, partial_assistant: str = "", status: str = "running", last_error: str = "",
    ) -> None:
        local_state.save_turn_checkpoint(
            session_id,
            objective=objective,
            mode=checkpoint_mode,
            messages=working_messages,
            partial_assistant=partial_assistant,
            status=status,
            last_error=last_error,
        )

    def _remember_stream_delta(delta: str) -> None:
        """Persist streaming text often enough to survive interruption,
        without fsyncing state.json for every provider token."""
        nonlocal last_checkpoint_at
        checkpoint_partial_parts.append(delta)
        now = time.monotonic()
        # Four atomic snapshots per second is effectively realtime for a
        # terminal stream without fsyncing once per token (which can be
        # hundreds of writes per second on fast local models).
        if now - last_checkpoint_at >= 0.25:
            _persist_turn_checkpoint(partial_assistant="".join(checkpoint_partial_parts))
            last_checkpoint_at = now

    _persist_turn_checkpoint()
    scope_diagnostic = "Focused workspace scope: " + ", ".join(str(path) for path in scope_roots)
    excluded_names = _objective_excluded_names(workspace_root, objective)
    if excluded_names:
        scope_diagnostic += f" (excluded per your instruction: {', '.join(sorted(excluded_names))})"
    renderer.handle_event({
        "event_type": "workspace_scope",
        "payload": {"content": scope_diagnostic},
    })
    session_approved_risks: set[str] = set()
    turn_approval_policy = approval_policy
    if approval_policy == "ask" and _requests_no_confirmation(incoming_objective):
        turn_approval_policy = "auto"
        renderer.handle_event({
            "event_type": "diagnostics",
            "payload": {"content": "User authorised automatic approval for ordinary permitted operations in this turn."},
        })
    renderer.handle_event({
        "event_type": "context_reused" if orchestration.context and orchestration.context.reused else "context_rescanned",
        "payload": {"workspace_root": workspace_root},
    })
    if resumed_from_checkpoint or resumed_from_legacy:
        renderer.handle_event({
            "event_type": "diagnostics",
            "payload": {
                "content": (
                    "Resuming the interrupted turn from its last durable checkpoint."
                    if resumed_from_checkpoint
                    else "Resuming from durable progress recorded by the prior Tamfis Code version."
                )
            },
        })

    # Resolve the route once per turn. Re-selecting it on every tool round can
    # make the displayed provider diverge from the client that actually runs.
    # Tier IV (the shared orchestration service) is just another
    # capability-ranked candidate in manager.PROVIDERS/PRIORITY_ORDER here --
    # see providers.py's _check_tier_iv_available -- rather than a separate
    # pre-flight routing-advisor call, since the real service only exposes
    # an execution endpoint, not a routing-decision one.
    renderer.handle_event({"event_type": "routing_started", "payload": {"requested_provider": provider.value, "task_type": task_profile.task_type.value}})
    if provider == ProviderType.AUTO and hasattr(manager, "resolve_route"):
        resolved_provider, config = manager.resolve_route(provider, task_profile, quality_mode="quality")
    else:
        resolved_provider = provider if provider != ProviderType.AUTO else manager._select_best_provider()
        config = manager.PROVIDERS.get(resolved_provider)
    client = manager.get_client(resolved_provider)
    if provider == ProviderType.AUTO and not _paid_provider_fallback_enabled(manager):
        renderer.handle_event({
            "event_type": "diagnostics",
            "payload": {
                "content": (
                    f"Automatic routing selected {resolved_provider.value}; "
                    "safe/free fallback only is enabled. Paid routes require an explicit --provider."
                ),
            },
        })
    if not client or config is None:
        error = f"Provider {resolved_provider.value} is not available (no client / no valid credentials)."
        orchestrator.fail(error)
        renderer.handle_event({"event_type": "ai_task_failed", "payload": {"error": error}})
        return TaskOutcome(status="failed", error=error)
    selected_default_model = (
        _select_fallback_model(manager, config, task_profile)
        if provider == ProviderType.AUTO and not _paid_provider_fallback_enabled(manager)
        else _select_model(manager, config, task_profile)
    )
    resolved_model = model or selected_default_model
    configured_models = set(getattr(config, "models", None) or [])
    if (resumed_from_checkpoint or resumed_from_legacy) and model and configured_models and model not in configured_models:
        renderer.handle_event({
            "event_type": "diagnostics",
            "payload": {
                "content": (
                    f"Checkpoint/requested model {model!r} is not in the current {resolved_provider.value} "
                    f"catalogue; using {selected_default_model!r} instead."
                )
            },
        })
        resolved_model = selected_default_model
    orchestrator.record_route(
        provider=resolved_provider.value, model=resolved_model,
        reason=("explicit selection" if provider != ProviderType.AUTO else "capability-aware automatic routing"),
        fallback_chain=_standalone_fallback_chain_names(manager, resolved_provider),
    )
    orchestrator.start_execution()

    # Replace the deterministic template plan (orchestrator.begin() already
    # made one, synchronously, as a safe fallback) with one grounded in the
    # real objective and real workspace facts -- confirmed live: the
    # template was the same generic "Inspect / Select provider / Execute /
    # Repair / Validate / Report" text for a one-line typo fix and a
    # full-stack audit alike. A failure here (bad JSON, provider error)
    # silently keeps the template; the turn is never blocked on this.
    if should_plan(task_profile):
        planning_reconnaissance = _build_planning_reconnaissance(
            workspace_root, scope_roots, objective,
        )
        renderer.handle_event({
            "event_type": "diagnostics",
            "payload": {"content": "Repository reconnaissance completed before plan generation."},
        })
        reasoning_plan = await _attempt_reasoning_plan(
            client, model=resolved_model, objective=objective, task_profile=task_profile,
            session_id=session_id, renderer=renderer,
            reconnaissance_summary=planning_reconnaissance,
            reasoning_effort=_reasoning_effort(resolved_provider, resolved_model),
            scope_roots=scope_roots,
        )
        if reasoning_plan is not None and orchestrator.run is not None:
            orchestrator.replace_plan(reasoning_plan)
            orchestrator.run.reasoning_plan = True
            plan_message = {
                "role": "system",
                "content": _plan_message_content(
                    reasoning_plan,
                    heading=(
                        "TASK PLAN (grounded in the actual objective and real workspace facts -- "
                        "supersedes any generic plan mentioned above):"
                    ),
                ),
            }
            working_messages.insert(working_messages.index(scope_message) + 1, plan_message)
            renderer.handle_event({
                "event_type": "plan_created",
                "payload": _plan_created_payload(reasoning_plan, title="Plan"),
            })

    previous_tool_calls_signature: Optional[tuple[tuple[str, str], ...]] = None
    consecutive_identical_rounds = 0
    recent_tool_signatures: list[tuple[tuple[str, str], ...]] = []
    any_mutation = False
    rollover_count = 0
    replanned_after_evidence = False
    loop_nudge_count = 0
    narrated_retries: dict[ProviderType, int] = {}
    narrated_failed_providers: set[ProviderType] = set()
    capitulation_retries: dict[ProviderType, int] = {}
    capitulation_failed_providers: set[ProviderType] = set()
    fabricated_result_retries: dict[ProviderType, int] = {}
    fabricated_result_failed_providers: set[ProviderType] = set()
    quality_failed_providers: set[ProviderType] = set()
    audit_recovery_reads = 0
    plan_completion_retries = 0

    async def _finalize_completed_answer(content: str, finish_reason: Optional[str]) -> TaskOutcome:
        """Turn accumulated completion output into the turn's final
        TaskOutcome: continue past a length-truncated cut-off, apply the
        fake-tool-call/no-mutation caveats, run orchestrator validation,
        and emit ai_task_completed. Extracted so _handle_stuck_loop's
        tools-disabled recovery synthesis (below) gets the exact same
        finishing treatment as an ordinary "model stopped calling tools"
        answer, instead of a bare, unvalidated partial answer."""
        nonlocal resolved_provider, config, client, resolved_model
        truncation_rounds = 0
        while finish_reason == "length" and truncation_rounds < MAX_TRUNCATION_CONTINUATIONS:
            truncation_rounds += 1
            continuation_messages = working_messages + [
                {"role": "assistant", "content": content},
                {"role": "user", "content": TRUNCATION_CONTINUATION_INSTRUCTION},
            ]
            try:
                more_content, _more_calls, finish_reason = await _stream_one_completion(
                    client, model=resolved_model, messages=continuation_messages, tools=[],
                    renderer=renderer,
                    reasoning_effort=_reasoning_effort(resolved_provider, resolved_model),
                    emit=False,
                )
            except Exception as exc:
                # A provider/worker request ceiling is a route failure, not a
                # reason to present narrated half-code as a completed task.
                # In auto mode, continue silently on the next eligible route.
                recovered = False
                if (
                    provider == ProviderType.AUTO
                    and _auto_provider_fallback_enabled(manager)
                    and hasattr(manager, "is_retryable_provider_error")
                    and manager.is_retryable_provider_error(exc)
                    and hasattr(manager, "fallback_candidates")
                ):
                    failed_provider = resolved_provider
                    for candidate in manager.fallback_candidates(failed_provider, task_profile):
                        candidate_client = manager.get_client(candidate)
                        candidate_config = manager.PROVIDERS.get(candidate)
                        if candidate_client is None or candidate_config is None:
                            continue
                        candidate_model = _select_fallback_model(manager, candidate_config, task_profile)
                        try:
                            more_content, _more_calls, finish_reason = await _stream_one_completion(
                                candidate_client,
                                model=candidate_model,
                                messages=continuation_messages,
                                tools=[],
                                renderer=renderer,
                                reasoning_effort=_reasoning_effort(candidate, candidate_model),
                                emit=False,
                            )
                        except Exception as candidate_exc:
                            if not manager.is_retryable_provider_error(candidate_exc):
                                break
                            continue
                        resolved_provider = candidate
                        config = candidate_config
                        client = candidate_client
                        resolved_model = candidate_model
                        recovered = True
                        break
                if not recovered:
                    message = (
                        "The provider stopped before finishing and no continuation route was available. "
                        "The exact turn was checkpointed; run `proceed` to resume it without repeating "
                        "completed tool actions."
                    )
                    _persist_turn_checkpoint(partial_assistant=content, status="interrupted")
                    orchestrator.fail(message)
                    renderer.handle_event({"event_type": "ai_task_failed", "payload": {"error": message}})
                    return TaskOutcome(status="failed", error=message)
            if not more_content.strip():
                break
            novel_content = _novel_continuation(content, more_content)
            if novel_content:
                renderer.handle_event({"event_type": "assistant_delta", "payload": {"content": novel_content}})
                content += novel_content
        if finish_reason == "length":
            content += (
                "\n\n⚠ This response is still incomplete after "
                f"{MAX_TRUNCATION_CONTINUATIONS} continuation attempts -- the provider kept hitting "
                "its output-length limit. Treat it as partial; ask again with a narrower scope for "
                "the rest."
            )

        if _looks_like_fake_tool_call(content):
            caveat = (
                "\n\n⚠ This response includes what looks like an unexecuted tool call "
                "(one of this agent's own tool names written out in text/code-block form) "
                "rather than a real action -- nothing beyond what's already listed above "
                "(if anything) actually happened. Ask again, more specifically, if you "
                "need this to actually run."
            )
            renderer.handle_event({"event_type": "assistant_delta", "payload": {"content": caveat}})
            content += caveat
        elif not read_only and not any_mutation and _looks_like_change_request(_latest_user_text(messages)):
            caveat = (
                "\n\n⚠ No files were changed during this task, despite the request "
                "asking for a fix/change. The response above may describe an edit "
                "without having actually made it -- verify the code yourself, or ask "
                "again more specifically (e.g. name the exact file)."
            )
            renderer.handle_event({"event_type": "assistant_delta", "payload": {"content": caveat}})
            content += caveat
        validation = orchestrator.complete(final_text=content, any_mutation=any_mutation)
        if validation.severity == "error":
            message = "Validation failed: " + "; ".join(validation.unresolved)
            renderer.handle_event({
                "event_type": "ai_task_failed",
                "payload": {"error": message, "validation": validation.to_dict()},
            })
            _persist_turn_checkpoint(partial_assistant=content, status="interrupted", last_error=message)
            return TaskOutcome(status="failed", error=message, summary=content)
        if not validation.passed:
            caveat = "\n\n⚠ Validation incomplete: " + "; ".join(validation.unresolved)
            renderer.handle_event({"event_type": "assistant_delta", "payload": {"content": caveat}})
            content += caveat
        renderer.handle_event({"event_type": "ai_task_completed", "payload": {"status": "completed", "validation": validation.to_dict()}})
        local_state.remember_conversation_turn(
            session_id, objective=objective, answer=content, clear_checkpoint=True,
        )
        return TaskOutcome(status="completed", summary=content)

    async def _handle_stuck_loop(
        stuck_reason: str, tool_calls: list["_StreamedToolCall"],
    ) -> Optional[TaskOutcome]:
        """A loop was just detected (identical-repeat or cycling, both
        checked right before this is called). Never just die on the spot:
        refuse the repeated call(s) with a clear reason instead of
        executing them again, and give the model up to
        MAX_LOOP_NUDGE_RETRIES chances to self-correct with tools still
        available. Only once that budget is exhausted does it fall back to
        a final, tools-disabled completion that synthesises whatever was
        actually found into a real answer -- this turn should end
        `completed` (even if only with a partial answer and an honest "ask
        again, narrower" caveat) far more often than it ends `failed` with
        nothing to show for the rounds already spent.

        Returns None when it nudged (the caller should `continue` to the
        next round); returns a TaskOutcome when it produced a final answer
        (the caller should return that immediately).
        """
        nonlocal loop_nudge_count, consecutive_identical_rounds, recent_tool_signatures

        # Every tool_call_id from the assistant message the caller already
        # appended still needs a matching role=="tool" response, or the
        # next completion request is a malformed conversation -- refusing
        # is not the same as skipping.
        refusal = {
            "success": False,
            "error": (
                f"Refused: this round was detected as stuck ({stuck_reason}). "
                "Act on a result you already have, or try a genuinely different next step."
            ),
        }
        for tc in tool_calls:
            working_messages.append({"role": "tool", "tool_call_id": tc.call_id, "content": json.dumps(refusal)})
            renderer.handle_event({"event_type": "tool_output", "payload": {"tool": tc.name, "result": refusal}})

        if loop_nudge_count < MAX_LOOP_NUDGE_RETRIES:
            loop_nudge_count += 1
            renderer.handle_event({
                "event_type": "diagnostics",
                "payload": {
                    "content": f"Detected {stuck_reason} -- reminding the model to change approach "
                               f"({loop_nudge_count}/{MAX_LOOP_NUDGE_RETRIES})."
                },
            })
            working_messages.append({
                "role": "system",
                "content": (
                    f"You just repeated {stuck_reason}, without making progress -- the calls above "
                    "were refused rather than executed again. Do not repeat them. Instead: act on a "
                    "SPECIFIC result you already have (read_file a specific file, list_directory a "
                    "specific subdirectory, search_code for a concrete pattern), or, if the original "
                    "request is too broad to make that concrete choice at all, stop calling tools now "
                    "and give your best current answer/plan based on everything found so far -- be "
                    "explicit about what remains unknown and tell the user what to narrow the request "
                    "to for more detail."
                ),
            })
            # Give the nudge a clean slate rather than re-tripping on the
            # very next round before the model has had a chance to act on
            # it -- a genuine continued loop after this is caught fresh.
            consecutive_identical_rounds = 0
            recent_tool_signatures = []
            return None

        renderer.handle_event({
            "event_type": "diagnostics",
            "payload": {
                "content": f"Still stuck ({stuck_reason}) after a reminder -- disabling tools for one "
                           "final answer instead of failing with nothing to show for it."
            },
        })
        working_messages.append({
            "role": "system",
            "content": (
                "Tool calls are now disabled for the rest of this turn -- you repeated the same "
                "action(s) without making progress even after being told to stop. Give your best "
                "current answer or plan based on everything you've actually found so far. Be explicit "
                "about what remains unknown/unverified, and if the original request was too broad to "
                "finish this way, say so plainly and tell the user what to narrow it to."
            ),
        })
        try:
            recovery_content, _recovery_calls, recovery_finish_reason = await _stream_one_completion(
                client, model=resolved_model, messages=working_messages, tools=[], renderer=renderer,
                reasoning_effort=_reasoning_effort(resolved_provider, resolved_model),
            )
        except Exception as exc:
            message = f"Stuck-loop recovery answer failed too ({exc}); nothing further to try this turn."
            orchestrator.fail(message)
            renderer.handle_event({"event_type": "ai_task_failed", "payload": {"error": message}})
            return TaskOutcome(status="failed", error=message)
        if not recovery_content.strip():
            message = (f"Detected {stuck_reason}, and the tools-disabled recovery answer was empty too. ""Try narrowing the objective to a specific repository, component, file, or concern.")
            orchestrator.fail(message)
            renderer.handle_event({"event_type": "ai_task_failed", "payload": {"error": message}})
            return TaskOutcome(status="failed", error=message)
        return await _finalize_completed_answer(recovery_content, recovery_finish_reason)

    # Computed here, after every working_messages.insert() above (the scope
    # rule and, when should_plan() fired, the grounded plan message) -- both
    # insert BEFORE the user objective message, so computing this any
    # earlier would point at the wrong index by the time the round loop
    # actually reads it. Later mutations from here on are all .append()
    # (tool/assistant messages added AFTER the objective), which never
    # shifts an already-computed earlier index.
    vision_message_index: Optional[int] = None
    if image_content_blocks:
        for _idx in range(len(working_messages) - 1, -1, -1):
            if working_messages[_idx].get("role") == "user":
                vision_message_index = _idx
                break

    for _round in range(max_rounds):
        # User-requested: reach a standalone task that's already running from
        # a SECOND terminal (`tamfis-code queue "..."` against the same
        # session) -- a single terminal can't do this mid-turn since the
        # REPL prompt isn't reading input while this loop is running, but a
        # second process can still write to the same on-disk session queue.
        # Before this existed, cli.py's own `queue` command comment said so
        # explicitly: "a standalone local turn is always synchronous within
        # one process, so there's nothing 'live' to push into." Checking at
        # the top of every round (a natural checkpoint -- never mid-stream)
        # closes that gap for anything queued between rounds.
        for live in _claim_live_queued_instructions(session_id):
            outcome = _apply_live_queued_instruction(
                live, session_id=session_id, working_messages=working_messages, renderer=renderer,
            )
            if outcome is not None:
                return outcome

        # Never fire a request already guaranteed to blow the provider's
        # context window -- confirmed live: HF 400'd with inputs(29548) +
        # max_new_tokens(4096) > 32769 after a long tool-calling session grew
        # working_messages unchecked. Try reclaiming budget from old, large
        # tool outputs first; only give up if that's not enough.
        token_budget = int(config.context_window * _CONTEXT_SAFETY_MARGIN) - MAX_TOKENS_PER_REQUEST
        input_tokens = _estimate_tokens(working_messages)
        # Persisted so `tamfis-code doctor` can report real (if estimated,
        # not provider-reported -- no provider response in this codebase
        # ever surfaces prompt/completion token counts) context usage for
        # this session after the fact, across process invocations.
        local_state.save_session_state(session_id, estimated_context_tokens=input_tokens)
        if input_tokens > token_budget:
            before_compaction = input_tokens
            compacted = _trim_tool_outputs(working_messages, token_budget)
            input_tokens = _estimate_tokens(working_messages)
            if compacted:
                renderer.handle_event({
                    "event_type": "diagnostics",
                    "payload": {
                        "content": (
                            f"Context compacted from ~{before_compaction} to "
                            f"~{input_tokens} estimated tokens for "
                            f"{resolved_provider.value}'s context window."
                        )
                    },
                })
        if input_tokens > token_budget:
            # Compaction alone was not enough. Before ever failing the turn,
            # try (a) an internal context rollover -- persist the full
            # segment as durable evidence and continue the SAME task from a
            # compact continuation package in a fresh internal context --
            # and (b) falling forward to a larger-context provider. Only
            # after both are exhausted (or genuinely inapplicable) does this
            # turn fail and tell the user to start a new one.
            # No longer gated on "has this turn made any tool calls yet" --
            # confirmed live: a large pasted objective (a long log/diff) can
            # blow the budget on round 1, before any tool/assistant message
            # exists at all. That used to skip rollover entirely and fall
            # straight to failure. Rollover is safe and useful regardless of
            # tool history: worst case (nothing has happened yet) it simply
            # persists the oversized objective as evidence and rebuilds a
            # genuinely smaller continuation (see _perform_context_rollover's
            # objective bounding), which is exactly what's needed here.
            if rollover_count < MAX_CONTEXT_ROLLOVERS_PER_TURN:
                rollover_count += 1
                before_rollover = input_tokens
                working_messages[:] = _perform_context_rollover(
                    working_messages, objective=objective, scope_roots=scope_roots,
                    scope_message=scope_message, session_id=session_id,
                )
                input_tokens = _estimate_tokens(working_messages)
                renderer.handle_event({
                    "event_type": "context_rollover",
                    "payload": {
                        "rollover_count": rollover_count,
                        "before_tokens": before_rollover,
                        "after_tokens": input_tokens,
                    },
                })

            if input_tokens > token_budget and provider == ProviderType.AUTO and _auto_provider_fallback_enabled(manager) and hasattr(manager, "fallback_candidates"):
                for candidate in manager.fallback_candidates(resolved_provider, task_profile):
                    candidate_config = manager.PROVIDERS.get(candidate)
                    if candidate_config is None or candidate_config.context_window <= config.context_window:
                        continue
                    candidate_client = manager.get_client(candidate)
                    if candidate_client is None:
                        continue
                    candidate_budget = int(candidate_config.context_window * _CONTEXT_SAFETY_MARGIN) - MAX_TOKENS_PER_REQUEST
                    if input_tokens > candidate_budget:
                        continue
                    candidate_model = _select_fallback_model(manager, candidate_config, task_profile)
                    renderer.handle_event({
                        "event_type": "diagnostics",
                        "payload": {
                            "content": (
                                f"This turn no longer fits {resolved_provider.value}'s "
                                f"~{config.context_window}-token context window; switching to "
                                f"{candidate.value} / {candidate_model} "
                                f"(~{candidate_config.context_window}-token window) to continue."
                            )
                        },
                    })
                    resolved_provider, config, client = candidate, candidate_config, candidate_client
                    resolved_model = candidate_model
                    orchestrator.record_route(
                        provider=resolved_provider.value, model=resolved_model,
                        reason="larger-context provider fallback",
                        fallback_chain=_standalone_fallback_chain_names(manager, resolved_provider),
                    )
                    token_budget = candidate_budget
                    break

            if input_tokens > token_budget:
                message = (
                    f"Stopping before round {_round + 1}: this turn has grown to "
                    f"~{input_tokens} estimated tokens, too large for "
                    f"{resolved_provider.value}'s ~{config.context_window}-token context "
                    "window even after compacting tool calls, tool outputs, assistant content, "
                    "old completed cycles, an internal context rollover, and provider fallback. "
                    "Start a new turn to continue (e.g. narrow the objective, or /clear stale context)."
                )
                orchestrator.fail(message)
                renderer.handle_event({"event_type": "ai_task_failed", "payload": {"error": message}})
                return TaskOutcome(status="failed", error=message)

        renderer.handle_event({
            "event_type": "model_selected",
            "payload": {"provider": resolved_provider.value, "model": resolved_model},
        })
        renderer.handle_event({
            "event_type": "provider_request_started",
            "payload": {"provider": resolved_provider.value, "model": resolved_model, "round": _round + 1},
        })

        checkpoint_partial_parts.clear()
        last_checkpoint_at = 0.0
        _persist_turn_checkpoint()
        try:
            content, tool_calls, finish_reason = await _stream_completion_with_reconnect(
                manager, client, provider=resolved_provider,
                model=resolved_model,
                messages=(
                    _messages_with_vision_content(working_messages, vision_message_index, image_content_blocks)
                    if getattr(config, "vision_supported", False) else working_messages
                ),
                tools=tools, renderer=renderer,
                reasoning_effort=_reasoning_effort(resolved_provider, resolved_model),
                progress_callback=_remember_stream_delta,
            )
        except Exception as exc:
            # Automatic routing must treat provider/account failures as route
            # failures, not task failures. In particular, OpenRouter HTTP 402
            # means that provider is unusable for this turn and should fall
            # through to the next eligible route.
            root_exc = exc.cause if isinstance(exc, _InterruptedCompletion) else exc
            interrupted_partial = (
                exc.partial if isinstance(exc, _InterruptedCompletion)
                else "".join(checkpoint_partial_parts)
            )
            can_fallback = (
                provider == ProviderType.AUTO
                and _auto_provider_fallback_enabled(manager)
                and hasattr(manager, "is_retryable_provider_error")
                and manager.is_retryable_provider_error(root_exc)
                and hasattr(manager, "fallback_candidates")
            )
            fallback_succeeded = False
            last_error: Exception = root_exc
            failed_provider = resolved_provider
            if can_fallback:
                for candidate in manager.fallback_candidates(failed_provider, task_profile):
                    candidate_client = manager.get_client(candidate)
                    candidate_config = manager.PROVIDERS.get(candidate)
                    if candidate_client is None or candidate_config is None:
                        continue
                    candidate_model = _select_fallback_model(manager, candidate_config, task_profile)
                    status = manager.provider_error_status(last_error) if hasattr(manager, "provider_error_status") else None
                    reason = f"HTTP {status}" if status is not None else str(last_error)
                    # A real repair attempt, tracked at the point it's actually
                    # tried -- not just retroactively labelled once every
                    # candidate has already been exhausted. Before this,
                    # repair_attempts/AgentPhase.REPAIR never reflected this
                    # fallback chain at all, even when it succeeded.
                    orchestrator.mark_repair(f"Falling back from {failed_provider.value} to {candidate.value} ({reason})")
                    renderer.handle_event({
                        "event_type": "diagnostics",
                        "payload": {
                            "content": (
                                f"Provider {failed_provider.value} unavailable for this turn ({reason}); "
                                f"falling back to {candidate.value} / {candidate_model}."
                            )
                        },
                    })
                    renderer.handle_event({
                        "event_type": "model_selected",
                        "payload": {
                            "provider": candidate.value,
                            "model": candidate_model,
                            "reason": "automatic provider fallback",
                        },
                    })
                    try:
                        checkpoint_partial_parts.clear()
                        if interrupted_partial:
                            checkpoint_partial_parts.append(interrupted_partial)
                        last_checkpoint_at = 0.0
                        content, tool_calls, finish_reason = await _stream_completion_with_reconnect(
                            manager,
                            candidate_client,
                            provider=candidate,
                            model=candidate_model,
                            messages=(
                                _messages_with_vision_content(working_messages, vision_message_index, image_content_blocks)
                                if getattr(candidate_config, "vision_supported", False) else working_messages
                            ),
                            tools=tools if getattr(candidate_config, "tool_calling", True) else [],
                            renderer=renderer,
                            reasoning_effort=_reasoning_effort(candidate, candidate_model),
                            progress_callback=_remember_stream_delta,
                            initial_partial=interrupted_partial,
                        )
                    except Exception as candidate_exc:
                        last_error = (
                            candidate_exc.cause
                            if isinstance(candidate_exc, _InterruptedCompletion)
                            else candidate_exc
                        )
                        if isinstance(candidate_exc, _InterruptedCompletion):
                            interrupted_partial = candidate_exc.partial
                        failed_provider = candidate
                        if not manager.is_retryable_provider_error(last_error):
                            break
                        continue
                    resolved_provider = candidate
                    config = candidate_config
                    client = candidate_client
                    resolved_model = candidate_model
                    orchestrator.record_route(
                        provider=resolved_provider.value,
                        model=resolved_model,
                        reason="automatic provider fallback",
                        fallback_chain=["nvidia", "hf", "openrouter"],
                    )
                    fallback_succeeded = True
                    break
            if not fallback_succeeded:
                if not can_fallback:
                    orchestrator.mark_repair(f"Provider/tool round failed, no fallback available: {last_error}")
                detail = str(last_error).strip() or type(last_error).__name__
                message = (
                    f"Provider streaming failed on {failed_provider.value} / {resolved_model}: {detail}. "
                    "The exact turn and partial response were checkpointed; type `continue` to resume "
                    "without losing the conversation or completed tool results."
                )
                _persist_turn_checkpoint(
                    partial_assistant=interrupted_partial or "".join(checkpoint_partial_parts),
                    status="interrupted",
                    last_error=message,
                )
                orchestrator.fail(message)
                renderer.handle_event({"event_type": "ai_task_failed", "payload": {"error": message}})
                return TaskOutcome(status="failed", error=message)

        # A provider can repeat across reconnects or agent rounds even when
        # each individual stream stays below the per-stream quality window.
        # Check the complete candidate before it reaches the final-answer
        # path; otherwise a repeated analysis is incorrectly accepted and the
        # no-mutation warning makes it look like a completed task.
        if finish_reason not in {"degenerate_repetition", "conversation_echo", "repeated_content", "corrupted_output"}:
            if _has_repeated_long_segment(content):
                finish_reason = "repeated_content"
            elif _DEGENERATE_REPETITION_RE.search(content[-_DEGENERATE_REPETITION_TAIL_WINDOW:]):
                finish_reason = "degenerate_repetition"

        if finish_reason in {"degenerate_repetition", "conversation_echo", "repeated_content", "corrupted_output"}:
            # A malformed completion must not prevent a concrete audit plan
            # from progressing.  If the plan already names an exact file,
            # consume that read-only step locally before spending another
            # provider attempt on the same broken response.
            if (
                tools
                and task_profile.requires_tools
                and getattr(task_profile.task_type, "value", "") == "audit"
                and orchestrator.run is not None
                and orchestrator.run.reasoning_plan
                and orchestrator.run.plan is not None
                and audit_recovery_reads < 8
                and await _recover_audit_plan_file(
                    mcp_server=mcp_server,
                    orchestrator=orchestrator,
                    renderer=renderer,
                    working_messages=working_messages,
                    plan=orchestrator.run.plan,
                    scope_roots=scope_roots,
                    objective=objective,
                    round_number=_round,
                )
            ):
                audit_recovery_reads += 1
                _persist_turn_checkpoint()
                continue
            failed_quality_provider = resolved_provider
            quality_failed_providers.add(failed_quality_provider)
            checkpoint_partial_parts.clear()
            switched = False
            if provider == ProviderType.AUTO and _auto_provider_fallback_enabled(manager) and hasattr(manager, "fallback_candidates"):
                for candidate in manager.fallback_candidates(failed_quality_provider, task_profile):
                    if candidate in quality_failed_providers:
                        continue
                    candidate_client = manager.get_client(candidate)
                    candidate_config = manager.PROVIDERS.get(candidate)
                    if candidate_client is None or candidate_config is None:
                        continue
                    resolved_provider = candidate
                    config = candidate_config
                    client = candidate_client
                    resolved_model = _select_fallback_model(manager, candidate_config, task_profile)
                    orchestrator.mark_repair(
                        f"Discarding invalid output from {failed_quality_provider.value}; retrying on {candidate.value}"
                    )
                    orchestrator.record_route(
                        provider=resolved_provider.value,
                        model=resolved_model,
                        reason=f"automatic fallback after {finish_reason}",
                        fallback_chain=_standalone_fallback_chain_names(manager, resolved_provider),
                    )
                    renderer.handle_event({
                        "event_type": "model_selected",
                        "payload": {
                            "provider": resolved_provider.value,
                            "model": resolved_model,
                            "selection_reason": f"automatic fallback after {finish_reason}",
                        },
                    })
                    switched = True
                    break
            if switched:
                _persist_turn_checkpoint()
                continue

            reason_text = (
                "corrupted token output"
                if finish_reason == "corrupted_output"
                else "a repeated conversation transcript"
                if finish_reason == "conversation_echo"
                else "repeated analysis/code content"
                if finish_reason == "repeated_content"
                else "a degenerate repetition loop"
            )
            error = (
                f"Provider {failed_quality_provider.value} produced {reason_text}. "
                "The invalid completion was discarded, and no clean fallback provider was available. "
                "The turn is checkpointed; type `continue` after enabling another provider."
            )
            evidence_id = ""
            try:
                evidence_id = evidence_store.store_segment(
                    session_id,
                    objective=objective,
                    messages=[*working_messages, {"role": "assistant", "content": content}],
                    summary=f"Rejected provider output: {reason_text}",
                )
            except Exception:
                # Checkpointing must remain available even if the optional
                # append-only evidence archive is unavailable.
                pass
            if evidence_id:
                error += f" Archived rejected output as {evidence_id}."
            _persist_turn_checkpoint(
                partial_assistant=_bounded_text(content, head=1200, tail=400, label="rejected output"),
                status="interrupted",
                last_error=error,
            )
            orchestrator.fail(error)
            renderer.handle_event({"event_type": "ai_task_failed", "payload": {"error": error}})
            return TaskOutcome(status="failed", error=error)

        # An empty completion after one or more tool results is a provider
        # continuation failure, not a completed task. Recover before entering
        # the final-answer branch so recovered tool calls continue normally.
        if not content.strip() and not tool_calls:
            # Tracked as a real repair attempt at the point it's actually
            # tried, not just retroactively if it fails -- _recover_empty_
            # continuation succeeding is itself a genuine repair, previously
            # invisible to repair_attempts/AgentPhase.REPAIR entirely.
            orchestrator.mark_repair(f"Recovering an empty continuation from {resolved_provider.value}")
            try:
                (
                    content,
                    tool_calls,
                    recovered_provider,
                    recovered_config,
                    recovered_client,
                    recovered_model,
                ) = await _recover_empty_continuation(
                    manager,
                    requested_provider=provider,
                    resolved_provider=resolved_provider,
                    config=config,
                    client=client,
                    model=resolved_model,
                    messages=working_messages,
                    tools=tools,
                    renderer=renderer,
                    task_profile=task_profile,
                )
                # _recover_empty_continuation's own internal completion calls
                # don't track finish_reason -- it's genuinely unknown for
                # recovered content, so the truncation-continuation check
                # below must not act on whatever finish_reason happened to
                # be set by the round that was JUST recovered from (empty).
                finish_reason = None
                if recovered_provider != resolved_provider:
                    resolved_provider = recovered_provider
                    config = recovered_config
                    client = recovered_client
                    resolved_model = recovered_model
                    orchestrator.record_route(
                        provider=resolved_provider.value,
                        model=resolved_model,
                        reason="empty-continuation provider fallback",
                        fallback_chain=["nvidia", "hf", "openrouter"],
                    )
            except Exception as exc:
                error = str(exc)
                orchestrator.fail(error)
                renderer.handle_event({
                    "event_type": "ai_task_failed",
                    "payload": {"error": error},
                })
                return TaskOutcome(status="failed", error=error)

        if not tool_calls:
            if not content.strip():
                error = "Provider continuation recovery produced no assistant text or tool calls."
                orchestrator.fail(error)
                renderer.handle_event({"event_type": "ai_task_failed", "payload": {"error": error}})
                return TaskOutcome(status="failed", error=error)

            # Some compatible models provide useful planning prose but omit
            # the structured tool call entirely.  For a concrete audit plan,
            # recover the next exact file read before treating the prose as a
            # narrated-tool failure or switching providers.
            if (
                tools
                and task_profile.requires_tools
                and getattr(task_profile.task_type, "value", "") == "audit"
                and orchestrator.run is not None
                and orchestrator.run.reasoning_plan
                and orchestrator.run.plan is not None
                and audit_recovery_reads < 8
                and await _recover_audit_plan_file(
                    mcp_server=mcp_server,
                    orchestrator=orchestrator,
                    renderer=renderer,
                    working_messages=working_messages,
                    plan=orchestrator.run.plan,
                    scope_roots=scope_roots,
                    objective=objective,
                    round_number=_round,
                )
            ):
                audit_recovery_reads += 1
                _persist_turn_checkpoint()
                continue

            # "Let me check..." is a promise, not evidence that a repository
            # check happened.  Keep it in the transcript, explicitly require
            # a real registered tool call, and retry.  AUTO mode abandons a
            # route that repeats the behaviour after its bounded correction.
            if tools and _looks_like_narrated_tool_intent(content):
                working_messages.append({"role": "assistant", "content": content})
                working_messages.append({"role": "system", "content": NARRATED_TOOL_CORRECTION})
                attempts = narrated_retries.get(resolved_provider, 0)
                if attempts < MAX_NARRATED_TOOL_RETRIES_PER_PROVIDER:
                    narrated_retries[resolved_provider] = attempts + 1
                    orchestrator.mark_repair(
                        f"Converting narrated repository work into real tool calls on {resolved_provider.value}"
                    )
                    renderer.handle_event({
                        "event_type": "diagnostics",
                        "payload": {
                            "content": (
                                "The model described a future inspection without running it; "
                                "requesting the registered tool call now."
                            )
                        },
                    })
                    _persist_turn_checkpoint()
                    continue

                narrated_failed_providers.add(resolved_provider)
                switched = False
                if provider == ProviderType.AUTO and _auto_provider_fallback_enabled(manager) and hasattr(manager, "fallback_candidates"):
                    for candidate in manager.fallback_candidates(resolved_provider, task_profile):
                        if candidate in narrated_failed_providers:
                            continue
                        candidate_client = manager.get_client(candidate)
                        candidate_config = manager.PROVIDERS.get(candidate)
                        if candidate_client is None or candidate_config is None:
                            continue
                        old_provider = resolved_provider
                        resolved_provider = candidate
                        config = candidate_config
                        client = candidate_client
                        resolved_model = _select_fallback_model(manager, candidate_config, task_profile)
                        orchestrator.mark_repair(
                            f"Falling back from {old_provider.value}: model repeatedly narrated tools without calling them"
                        )
                        orchestrator.record_route(
                            provider=resolved_provider.value,
                            model=resolved_model,
                            reason="automatic fallback after unexecuted narrated tool work",
                            fallback_chain=_standalone_fallback_chain_names(manager, resolved_provider),
                        )
                        renderer.handle_event({
                            "event_type": "diagnostics",
                            "payload": {
                                "content": (
                                    f"{old_provider.value} repeatedly described tool work without executing it; "
                                    f"falling back to {resolved_provider.value} / {resolved_model}."
                                )
                            },
                        })
                        switched = True
                        break
                if switched:
                    _persist_turn_checkpoint()
                    continue

                error = (
                    "The available model repeatedly described repository actions without issuing a registered "
                    "tool call. The exact turn was checkpointed; type `continue` after enabling another "
                    "tool-capable provider, or select one explicitly."
                )
                _persist_turn_checkpoint(status="interrupted", last_error=error)
                orchestrator.fail(error)
                renderer.handle_event({"event_type": "ai_task_failed", "payload": {"error": error}})
                return TaskOutcome(status="failed", error=error)

            if tools and _looks_like_fabricated_tool_result(content):
                working_messages.append({"role": "assistant", "content": content})
                working_messages.append({"role": "system", "content": FABRICATED_RESULT_CORRECTION})
                attempts = fabricated_result_retries.get(resolved_provider, 0)
                if attempts < MAX_FABRICATED_RESULT_RETRIES_PER_PROVIDER:
                    fabricated_result_retries[resolved_provider] = attempts + 1
                    orchestrator.mark_repair(
                        f"Converting a fabricated tool result/refusal into a real tool call on {resolved_provider.value}"
                    )
                    renderer.handle_event({
                        "event_type": "diagnostics",
                        "payload": {
                            "content": (
                                "The model reported a tool result or tool-level refusal that no real tool call "
                                "backs; requesting the registered tool call now."
                            )
                        },
                    })
                    _persist_turn_checkpoint()
                    continue

                fabricated_result_failed_providers.add(resolved_provider)
                switched = False
                if provider == ProviderType.AUTO and _auto_provider_fallback_enabled(manager) and hasattr(manager, "fallback_candidates"):
                    for candidate in manager.fallback_candidates(resolved_provider, task_profile):
                        if candidate in fabricated_result_failed_providers:
                            continue
                        candidate_client = manager.get_client(candidate)
                        candidate_config = manager.PROVIDERS.get(candidate)
                        if candidate_client is None or candidate_config is None:
                            continue
                        old_provider = resolved_provider
                        resolved_provider = candidate
                        config = candidate_config
                        client = candidate_client
                        resolved_model = _select_fallback_model(manager, candidate_config, task_profile)
                        orchestrator.mark_repair(
                            f"Falling back from {old_provider.value}: model fabricated a tool result/refusal without calling it"
                        )
                        orchestrator.record_route(
                            provider=resolved_provider.value,
                            model=resolved_model,
                            reason="automatic fallback after fabricated tool result",
                            fallback_chain=_standalone_fallback_chain_names(manager, resolved_provider),
                        )
                        renderer.handle_event({
                            "event_type": "diagnostics",
                            "payload": {
                                "content": (
                                    f"{old_provider.value} repeatedly fabricated tool results/refusals without "
                                    f"executing them; falling back to {resolved_provider.value} / {resolved_model}."
                                )
                            },
                        })
                        switched = True
                        break
                if switched:
                    _persist_turn_checkpoint()
                    continue

                error = (
                    "The available model repeatedly reported fabricated tool results or refusals without issuing "
                    "a registered tool call. The exact turn was checkpointed; type `continue` after enabling "
                    "another tool-capable provider, or select one explicitly."
                )
                _persist_turn_checkpoint(status="interrupted", last_error=error)
                orchestrator.fail(error)
                renderer.handle_event({"event_type": "ai_task_failed", "payload": {"error": error}})
                return TaskOutcome(status="failed", error=error)

            tool_backed_recovery_required = bool(
                tools and (
                    task_profile.requires_tools
                    or resume_requested
                    or _requests_autonomous_execution(incoming_objective)
                    or _looks_like_change_request(objective.lower())
                )
            )
            if tool_backed_recovery_required and _looks_like_capitulation(content):
                working_messages.append({"role": "assistant", "content": content})
                working_messages.append({"role": "system", "content": CAPITULATION_CORRECTION})
                attempts = capitulation_retries.get(resolved_provider, 0)
                if attempts < MAX_CAPITULATION_RETRIES_PER_PROVIDER:
                    capitulation_retries[resolved_provider] = attempts + 1
                    orchestrator.mark_repair(
                        f"Redirecting {resolved_provider.value} away from giving up without a tool call"
                    )
                    renderer.handle_event({
                        "event_type": "diagnostics",
                        "payload": {
                            "content": (
                                "The model gave up citing an unclear next step without trying a tool; "
                                "requesting it investigate the repository instead."
                            )
                        },
                    })
                    _persist_turn_checkpoint()
                    continue

                capitulation_failed_providers.add(resolved_provider)
                switched = False
                if provider == ProviderType.AUTO and _auto_provider_fallback_enabled(manager) and hasattr(manager, "fallback_candidates"):
                    for candidate in manager.fallback_candidates(resolved_provider, task_profile):
                        if candidate in capitulation_failed_providers:
                            continue
                        candidate_client = manager.get_client(candidate)
                        candidate_config = manager.PROVIDERS.get(candidate)
                        if candidate_client is None or candidate_config is None:
                            continue
                        old_provider = resolved_provider
                        resolved_provider = candidate
                        config = candidate_config
                        client = candidate_client
                        resolved_model = _select_fallback_model(manager, candidate_config, task_profile)
                        orchestrator.mark_repair(
                            f"Falling back from {old_provider.value}: model gave up without trying a tool"
                        )
                        orchestrator.record_route(
                            provider=resolved_provider.value,
                            model=resolved_model,
                            reason="automatic fallback after unprompted capitulation",
                            fallback_chain=_standalone_fallback_chain_names(manager, resolved_provider),
                        )
                        renderer.handle_event({
                            "event_type": "diagnostics",
                            "payload": {
                                "content": (
                                    f"{old_provider.value} gave up without trying a tool; "
                                    f"falling back to {resolved_provider.value} / {resolved_model}."
                                )
                            },
                        })
                        switched = True
                        break
                if switched:
                    _persist_turn_checkpoint()
                    continue

                error = (
                    "The available model gave up citing an unclear next step without issuing a registered "
                    "tool call, and no clean fallback provider was available. The exact turn was checkpointed; "
                    "type `continue` after enabling another tool-capable provider, or select one explicitly, "
                    "or name a specific file/error to fix."
                )
                _persist_turn_checkpoint(status="interrupted", last_error=error)
                orchestrator.fail(error)
                renderer.handle_event({"event_type": "ai_task_failed", "payload": {"error": error}})
                return TaskOutcome(status="failed", error=error)

            pending_plan_steps = []
            if orchestrator.run is not None and orchestrator.run.reasoning_plan and orchestrator.run.plan is not None:
                pending_plan_steps = [
                    step.name for step in orchestrator.run.plan.steps
                    if step.status in {"pending", "in_progress"}
                ]
            if (
                tools and task_profile.requires_tools
                and getattr(task_profile.task_type, "value", "") == "audit"
                and pending_plan_steps
                and plan_completion_retries < 2
            ):
                plan_completion_retries += 1
                working_messages.append({"role": "assistant", "content": content})
                working_messages.append({
                    "role": "system",
                    "content": (
                        "The audit is not complete. The execution plan still has pending steps: "
                        + "; ".join(pending_plan_steps)
                        + ". Continue inspecting the remaining workspace and recover from any failed read "
                        "with a focused alternative path. Do not provide the final report until every plan "
                        "step has either completed with evidence or is explicitly reported as blocked."
                    ),
                })
                renderer.handle_event({
                    "event_type": "diagnostics",
                    "payload": {
                        "content": (
                            "The model attempted to finish before the audit plan was complete; "
                            "requesting the remaining inspection steps."
                        )
                    },
                })
                _persist_turn_checkpoint()
                continue

            return await _finalize_completed_answer(content, finish_reason)

        signature = _tool_calls_signature(tool_calls)
        if signature == previous_tool_calls_signature:
            consecutive_identical_rounds += 1
        else:
            consecutive_identical_rounds = 0
        previous_tool_calls_signature = signature

        # Consecutive-identical detection only catches a period-1 repeat
        # (the exact same call every round). A model can just as easily
        # get stuck alternating between two or three distinct calls
        # (A,B,A,B,... or A,B,C,A,B,C,...) -- rediscovering an unchanged
        # workspace, cycling between the same files -- without ever
        # repeating one call twice in a row. Detect that too, from a short
        # rolling window, rather than waiting for max_rounds.
        recent_tool_signatures.append(signature)
        del recent_tool_signatures[:-_LOOP_WINDOW]

        stuck_reason: Optional[str] = None
        if consecutive_identical_rounds >= MAX_CONSECUTIVE_IDENTICAL_ROUNDS:
            names = ", ".join(sorted({tc.name for tc in tool_calls})) or "tool call"
            stuck_reason = (
                f"the model is stuck repeating the same {names} request "
                f"{consecutive_identical_rounds + 1} rounds in a row "
                "with identical arguments"
            )
        elif _is_cycling(recent_tool_signatures):
            names = ", ".join(sorted({name for sig in recent_tool_signatures for name, _ in sig})) or "tool call"
            stuck_reason = f"a repeating cycle of tool calls ({names}) across the last {_LOOP_WINDOW} rounds"

        # The model's own tool_calls message is echoed back regardless of
        # whether this round turns out to be stuck -- every tool_call_id
        # needs a matching role=="tool" response either way (a real result
        # below, or _handle_stuck_loop's refusal instead).
        working_messages.append({
            "role": "assistant", "content": content or "",
            "tool_calls": [
                {"id": tc.call_id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
                for tc in tool_calls
            ],
        })
        # Persist the native tool-call ids before dispatch. If the process is
        # killed inside a command, resume can close those ids as interrupted
        # and inspect reality instead of executing the action twice.
        _persist_turn_checkpoint()

        if stuck_reason is not None:
            outcome = await _handle_stuck_loop(stuck_reason, tool_calls)
            if outcome is not None:
                return outcome
            continue  # nudged -- give the model one more round to self-correct

        for tc in tool_calls:
            try:
                arguments = json.loads(tc.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}

            guard = orchestrator.guard_tool_call(tc.name, arguments)
            if not guard.allowed:
                result = {
                    "success": False,
                    "error": guard.reason,
                    "runtime_blocked": True,
                }
                working_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.call_id,
                    "content": json.dumps(result),
                })
                renderer.handle_event({
                    "event_type": "tool_output",
                    "payload": {"tool": tc.name, "result": result},
                })
                if guard.terminal:
                    _persist_turn_checkpoint(status="failed", last_error=guard.reason)
                    orchestrator.fail(guard.reason)
                    renderer.handle_event({"event_type": "ai_task_failed", "payload": {"error": guard.reason}})
                    return TaskOutcome(status="failed", error=guard.reason)
                continue

            if tc.name == "retrieve_evidence":
                # A pure local lookup by id -- read-only, no workspace scope
                # or approval gate applies (it isn't a filesystem/shell tool
                # at all), so it's dispatched before either.
                renderer.handle_event({"event_type": "tool_call_requested", "payload": {"name": tc.name, "arguments": arguments}})
                segment = evidence_store.load_segment(session_id, str(arguments.get("evidence_id") or ""))
                if segment is None:
                    result: dict[str, Any] = {
                        "success": False,
                        "error": f"No evidence segment found for evidence_id={arguments.get('evidence_id')!r}.",
                    }
                else:
                    result = {
                        "success": True,
                        "result": {
                            "evidence_id": segment.get("evidence_id"),
                            "objective": segment.get("objective"),
                            "summary": segment.get("summary"),
                            "message_count": segment.get("message_count"),
                            # Bounded immediately (rather than relying on next
                            # round's compaction pass) so retrieving a huge
                            # prior segment can never itself blow the budget.
                            "messages": _compact_json_value(segment.get("messages"), head=2000, tail=500),
                        },
                    }
                working_messages.append({"role": "tool", "tool_call_id": tc.call_id, "content": json.dumps(result, default=str)})
                renderer.handle_event({"event_type": "tool_output", "payload": {"tool": tc.name, "result": _tool_output_for_render(result)}})
                continue

            if tc.name == "delegate_parallel_tasks":
                # Not a filesystem/shell tool either -- no workspace scope
                # or per-call approval gate applies to the call itself
                # (mutation_policy_allows_swarm is the gate here, checked
                # once up front by run_swarm rather than per file/command).
                from .swarm import run_swarm

                renderer.handle_event({"event_type": "tool_call_requested", "payload": {"name": tc.name, "arguments": arguments}})
                sub_tasks, sub_agent_types = _parse_swarm_tasks(arguments.get("tasks") or [])
                mutate = bool(arguments.get("mutate", False))
                if len(sub_tasks) < 2:
                    result = {"success": False, "error": "delegate_parallel_tasks requires at least 2 independent tasks."}
                else:
                    suspend_live_if_active(renderer)
                    try:
                        swarm_results = await run_swarm(
                            sub_tasks, manager=manager, provider=provider, model=model, console=console,
                            workspace_root=workspace_root, session_id=session_id,
                            approval_policy=turn_approval_policy, mutate=mutate,
                            agent_types=sub_agent_types if any(sub_agent_types) else None,
                        )
                        result = {"success": True, "result": swarm_results}
                    except ValueError as e:
                        result = {"success": False, "error": str(e)}
                    finally:
                        resume_live_if_active(renderer)
                working_messages.append({"role": "tool", "tool_call_id": tc.call_id, "content": json.dumps(result, default=str)})
                renderer.handle_event({"event_type": "tool_output", "payload": {"tool": tc.name, "result": _tool_output_for_render(result)}})
                continue

            arguments, scope_error = _scope_tool_arguments(
                tc.name,
                arguments,
                workspace_root=workspace_root,
                scope_roots=scope_roots,
                attachment_paths=attachment_paths,
            )
            renderer.handle_event({"event_type": "tool_call_requested", "payload": {"name": tc.name, "arguments": arguments}})
            if scope_error:
                result = {
                    "error": scope_error,
                    "success": False,
                    "scope_roots": [str(path) for path in scope_roots],
                }
                working_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.call_id,
                    "content": json.dumps(result),
                })
                renderer.handle_event({
                    "event_type": "tool_output",
                    "payload": {"tool": tc.name, "result": result},
                })
                continue

            risk = classify_tool_call_risk(tc.name, arguments, workspace_root=workspace_root)

            if read_only and risk != "read_only":
                result = {
                    "error": f"'{tc.name}' is not available in read-only mode with these arguments.",
                    "success": False,
                }
                working_messages.append({"role": "tool", "tool_call_id": tc.call_id, "content": json.dumps(result)})
                renderer.handle_event({"event_type": "tool_output", "payload": {"tool": tc.name, "result": result}})
                continue

            if risk != "read_only" and risk not in session_approved_risks:
                # The live status line redraws on its own timer (independent
                # of anything else writing to the console) -- confirmed live
                # via a pty capture: suspending it *after* the panel below
                # printed still let one or two stray spinner frames render
                # between the panel and the blocking prompt, because the
                # background refresh thread can tick in the gap before
                # suspend takes effect. Suspend BEFORE printing the durable
                # approval panel (not just before the prompt) so nothing can
                # race the card's visibility, then restore it once a
                # decision is made.
                suspend_live_if_active(renderer)
                orchestrator.waiting_for_approval(f"Approve {tc.name}")
                reason = "The agent requested this command."
                if (
                    tc.name == "execute_command" and not any_mutation
                    and _looks_like_service_restart(str(arguments.get("command") or ""))
                    and _looks_like_change_request(_latest_user_text(messages))
                ):
                    reason += (
                        " ⚠ No files have been changed yet this task -- verify the "
                        "intended fix was actually written before restarting/reloading "
                        "this service."
                    )
                # Real credentials (e.g. a `mysql -pSECRET` invocation built
                # from a password the model just read out of wp-config.php)
                # must still reach the actual tool call below -- only the
                # human-facing/logged rendering gets the redacted copy.
                display_arguments = arguments
                if isinstance(arguments.get("command"), str):
                    display_arguments = {**arguments, "command": redact_secrets(arguments["command"])}
                diff_preview = _preview_diff_for_tool_call(mcp_server, tc.name, arguments)
                if diff_preview is not None:
                    # The diff panel (below) already shows the real change;
                    # a full write_file's entire proposed file content
                    # doesn't need to be duplicated as a raw JSON blob too.
                    display_command = f"{tc.name}(path={arguments.get('path')!r})"
                else:
                    display_command = f"{tc.name}({json.dumps(display_arguments, default=str)})"
                renderer.handle_event({
                    "event_type": "approval_required",
                    "payload": {
                        "command": display_command, "risk_level": risk,
                        "working_directory": workspace_root, "reason": reason,
                        "diff": diff_preview,
                    },
                })
                # display_preview=False -- the event above already rendered
                # the full command/cwd/reason/risk card (render.py's
                # approval_required handler), so the prompt doesn't draw a
                # second, less-informative panel for the same approval
                # (matches runner.py's remote-path fix for the same trap).
                try:
                    decision = await resolve_approval_decision_async(
                        console, display_command, risk, turn_approval_policy, interactive,
                        display_preview=False, config=cli_config,
                    )
                finally:
                    resume_live_if_active(renderer)
                if decision == "approve_session":
                    session_approved_risks.add(risk)
                elif decision == "deny":
                    result = {"error": "Denied by approval policy -- try a different, less risky approach.", "success": False}
                    working_messages.append({"role": "tool", "tool_call_id": tc.call_id, "content": json.dumps(result)})
                    renderer.handle_event({"event_type": "tool_output", "payload": {"tool": tc.name, "result": result}})
                    continue

            if configured_hooks:
                pre_hook_results = await run_tool_hooks(
                    configured_hooks, "pre_tool_use", tool_name=tc.name, tool_input=arguments,
                    session_id=session_id, workspace_root=workspace_root,
                )
                blocking = next((r for r in pre_hook_results if r.blocked), None)
                for hook_result in pre_hook_results:
                    renderer.handle_event({
                        "event_type": "diagnostics",
                        "payload": {"content": f"[hook:{hook_result.hook.event}] {hook_result.message}"},
                    })
                if blocking is not None:
                    result = {"error": f"Blocked by hook: {blocking.message}", "success": False}
                    working_messages.append({"role": "tool", "tool_call_id": tc.call_id, "content": json.dumps(result)})
                    renderer.handle_event({"event_type": "tool_output", "payload": {"tool": tc.name, "result": result}})
                    continue

            envelope = ToolEnvelope(
                tool_call_id=tc.call_id or f"call_round_{_round}", tool_name=tc.name, arguments=arguments,
                purpose=f"Execute {tc.name} for: {objective[:160]}", risk=risk,
                requires_approval=risk != "read_only", cwd=str(arguments.get("cwd") or workspace_root),
            )
            result = await mcp_server.call_tool(tc.name, arguments)
            result = _normalise_tool_result(tc.name, arguments, result, workspace_root)
            envelope.finish(result=result, success=bool(result.get("success")))
            observation = orchestrator.record_tool(envelope)
            if observation.terminal:
                working_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.call_id,
                    "content": json.dumps(result, default=str),
                })
                _persist_turn_checkpoint(status="failed", last_error=observation.reason)
                renderer.handle_event({"event_type": "ai_task_failed", "payload": {"error": observation.reason}})
                return TaskOutcome(status="failed", error=observation.reason)
            renderer.handle_event({
                "event_type": "tool_output",
                "payload": {"tool": tc.name, "result": _tool_output_for_render(result)},
            })
            working_messages.append({
                "role": "tool",
                "tool_call_id": tc.call_id,
                "content": json.dumps(result, default=str),
            })

            if configured_hooks:
                post_hook_results = await run_tool_hooks(
                    configured_hooks, "post_tool_use", tool_name=tc.name, tool_input=arguments,
                    tool_output=result, session_id=session_id, workspace_root=workspace_root,
                )
                for hook_result in post_hook_results:
                    renderer.handle_event({
                        "event_type": "diagnostics",
                        "payload": {"content": f"[hook:{hook_result.hook.event}] {hook_result.message}"},
                    })
                    working_messages.append({
                        "role": "system",
                        "content": f"Hook feedback after {tc.name}: {hook_result.message}",
                    })

            if tc.name in {"write_file", "edit_file", "extract_archive", "repackage_archive"} and result.get("success"):
                any_mutation = True
                state = local_state.get_session_state(session_id)
                if state.modified_files:
                    mutation = state.modified_files[-1]
                    renderer.handle_event({"event_type": "file_mutation", "payload": {
                        "path": mutation["path"], "lines_added": mutation["lines_added"],
                        "lines_removed": mutation["lines_removed"], "mutation_id": mutation["mutation_id"],
                    }})

        # The assistant tool-call message and every matching tool result are
        # now protocol-complete.  Save immediately before any optional
        # replanning/provider request so a kill here never repeats a tool
        # that already ran (especially a file mutation or command).
        _persist_turn_checkpoint()

        # Adaptive replanning: the plan above (whether reasoning-based or
        # the deterministic template) was necessarily made before any tool
        # had actually run -- a guess. Once real evidence exists (this
        # round executed at least one tool call), revise it once, grounded
        # in what was actually found, rather than letting the turn keep
        # working off a guess for its whole duration. Bounded to once per
        # turn -- this is a course-correction, not continuous re-planning.
        if (
            tool_calls and not replanned_after_evidence and should_plan(task_profile)
            and orchestrator.run is not None and orchestrator.run.plan is not None
        ):
            replanned_after_evidence = True
            revised_plan = await _attempt_reasoning_plan(
                client, model=resolved_model, objective=objective, task_profile=task_profile,
                session_id=session_id, renderer=renderer,
                reconnaissance_summary=planning_reconnaissance,
                evidence_summary=_summarise_progress_for_rollover(session_id),
                reasoning_effort=_reasoning_effort(resolved_provider, resolved_model),
                scope_roots=scope_roots,
            )
            if revised_plan is not None:
                orchestrator.replace_plan(revised_plan)
                working_messages.append({
                    "role": "system",
                    "content": _plan_message_content(
                        revised_plan,
                        heading=(
                            "PLAN REVISED based on evidence gathered so far -- supersedes the "
                            "earlier plan entirely:"
                        ),
                    ),
                })
                renderer.handle_event({
                    "event_type": "plan_created",
                    "payload": _plan_created_payload(revised_plan, title="Plan (revised)"),
                })

    message = f"(Stopped after {max_rounds} tool-call rounds without a final answer -- this usually means the task needs to be narrowed.)"
    _persist_turn_checkpoint(status="interrupted", last_error=message)
    orchestrator.fail(message)
    renderer.handle_event({"event_type": "ai_task_failed", "payload": {"error": message}})
    return TaskOutcome(status="failed", error=message)


async def run_local_shell_command(
    console: Console, *, workspace_root: str, session_id: int, command: str,
    approval_policy: str, interactive: bool, config: Optional[Config] = None,
) -> TaskOutcome:
    """Standalone equivalent of runner.py's run_shell_command -- executes an
    explicit `$ <command>` / `/run` / `/shell` REPL command locally via
    MCPServer's execute_command tool, gated through the same local risk
    classifier and approval flow as any other tool call, instead of
    submitting it to a Remote command queue."""
    from .safety import classify_command_risk

    display_command = redact_secrets(command)
    action = local_state.start_action(
        session_id, action_type="shell_command", purpose="Run an explicit local shell command",
        risk="policy_classified", detail=display_command,
    )
    console.print(f"[bold]$[/bold] {display_command}")

    risk = classify_command_risk(command)
    if risk != "read_only":
        decision = await resolve_approval_decision_async(console, display_command, risk, approval_policy, interactive, config=config)
        if decision == "deny":
            local_state.finish_action(session_id, action.id, status="failed", summary="denied")
            console.print("[dim]Denied.[/dim]")
            return TaskOutcome(status="denied", error="Denied by approval policy")

    mcp_server = MCPServer(workspace_root=workspace_root, session_id=session_id)
    result = await mcp_server.call_tool("execute_command", {"command": command})
    payload = result.get("result") if isinstance(result.get("result"), dict) else result
    stdout = str(payload.get("stdout") or "")
    stderr = str(payload.get("stderr") or "")
    exit_code = payload.get("return_code")
    ok = bool(result.get("success")) and exit_code == 0
    body = stdout.strip()
    if stderr.strip():
        body = (body + "\n" + stderr.strip()).strip()
    from .render import _render_result_block
    _render_result_block(console, ok=ok, label=f"exit {exit_code}", content=body)

    outcome = TaskOutcome(status="completed", summary=stdout) if ok else TaskOutcome(status="failed", error=stderr or f"exit code {exit_code}")
    local_state.finish_action(session_id, action.id, status=outcome.status, summary=f"exit={exit_code}")
    local_state.checkpoint(session_id, reason=f"command_{outcome.status}", summary=f"exit={exit_code}")
    return outcome
