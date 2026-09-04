"""Evidence-based validation and completion integrity."""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any
from pathlib import Path

from ..routing import TaskProfile, TaskType


@dataclass
class ValidationReport:
    passed: bool
    checks: list[dict[str, Any]] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    severity: str = "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": self.checks,
            "unresolved": self.unresolved,
            "severity": self.severity,
        }


_UNSUPPORTED_INSPECTION_CLAIM_RE = re.compile(
    r"\b(?:i\s+(?:have\s+)?(?:reviewed|examined|inspected|audited|analysed|analyzed)|"
    r"i\s+now\s+have\s+the\s+full\s+picture|"
    r"after\s+(?:reviewing|examining|inspecting|auditing|analysing|analyzing)|"
    r"the\s+(?:review|inspection|audit|analysis)\s+(?:shows|found|identified|revealed)|"
    # Confirmed live (meta/llama-3.1-70b-instruct on NVIDIA NIM): a weak
    # model can fabricate a *specific* tool's past-tense result -- "the
    # search_code tool has found several references to..." -- without ever
    # issuing that tool call. This reads as reported evidence exactly like
    # the phrases above, just naming the tool instead of saying "I".
    r"the\s+\w+\s+tool\s+(?:has\s+)?(?:found|returned|shows?|revealed|indicates?)|"
    r"(?:the\s+)?results?\s+(?:suggest|indicate)s?\b)\b",
    re.IGNORECASE,
)

_VERIFIED_NO_CHANGE_RE = re.compile(
    r"\b(?:"
    r"(?:issue|error|bug|failure|problem)\s+(?:is|was|has\s+been)\s+"
    r"(?:now\s+)?(?:already\s+)?(?:fixed|resolved|working)|"
    r"already\s+(?:fixed|resolved|implemented|correct|present|working)|"
    r"no\s+(?:code|file|source)?\s*changes?\s+(?:were\s+)?(?:needed|required|necessary)|"
    r"(?:fix|change|implementation|conflict\s+clause)\s+(?:is|was)\s+already\s+"
    r"(?:present|applied|deployed|correct)|"
    r"running\s+(?:server|service|deployment)\b[^.!?\n]{0,120}\b(?:correct|fixed|resolved)"
    r")\b",
    re.IGNORECASE,
)

_MUTATION_CLAIM_RE = re.compile(
    r"\b(?:i\s+(?:have\s+)?(?:updated|changed|edited|rewritten|replaced|implemented|fixed)|"
    r"changes?\s+(?:made|applied)|(?:updated|rewritten|modified)\s+(?:the\s+)?(?:file|code|worker|configuration)|"
    r"(?:file|code|worker|configuration)\s+(?:was|has\s+been)\s+(?:updated|rewritten|modified))\b",
    re.IGNORECASE,
)

_NEGATED_MUTATION_PREFIX_RE = re.compile(
    r"\b(?:no|zero)\s+(?:files?|code|source|configuration|changes?)\b[^.!?\n]{0,120}$|"
    r"\b(?:did\s+not|didn't|was\s+not|were\s+not|weren't)\b[^.!?\n]{0,80}$",
    re.IGNORECASE,
)

_SERVICE_RESTART_CLAIM_RE = re.compile(
    r"\b(?:i\s+(?:have\s+)?(?:restarted|reloaded)|(?:system|service|server|worker|backend)\s+restart(?:ed)?|"
    r"(?:service|server|worker|backend)\s+(?:is|was)\s+(?:now\s+)?live)\b",
    re.IGNORECASE,
)

_LIVE_VERIFICATION_CLAIM_RE = re.compile(
    r"\b(?:verified\s+(?:the\s+)?(?:live|running|production|frontend|endpoint|api)|"
    r"(?:frontend|endpoint|api|integration|pipeline)\s+(?:is|was|has\s+been)\s+(?:now\s+)?(?:working|verified|healthy)|"
    r"(?:will|does)\s+(?:now\s+)?(?:pull|return|load|display)\s+(?:data|results?)\s+successfully)\b",
    re.IGNORECASE,
)

_CLAIMED_PATH_RE = re.compile(
    r"`((?:/[^`\n]+|(?:[A-Za-z0-9_.@+-]+/)+[A-Za-z0-9_.@+-]+|"
    # Bare single-segment filename with an extension (e.g. `tool_registry.py`,
    # no directory component) -- a commit/changes summary listing files by
    # basename alone used to never match here at all, since every other
    # alternative requires a "/". _looks_like_file_path's extension
    # whitelist + existence check already guards this against false
    # positives (a version string like `v1.2` has no recognized extension
    # and doesn't exist on disk, so it's filtered out downstream).
    r"[A-Za-z0-9_-]+\.[A-Za-z0-9]{1,8}|\.env))`"
)

# How far around a mutation-claim phrase ("I updated `x`") to look for a
# backticked path before treating it as claimed-changed. Restricting the
# scan to this window -- instead of the whole report -- stops an unrelated
# backticked mention elsewhere in a long report (an API route quoted for
# context, a test file cited as pre-existing coverage, a product name) from
# being pulled in as if the model had claimed to change it.
_CLAIMED_PATH_WINDOW = 200

_PATH_EXTENSION_RE = re.compile(
    r"\.(?:php|css|scss|sass|less|js|jsx|mjs|cjs|ts|tsx|json|jsonc|xml|yaml|yml|toml|md|mdx|"
    r"txt|pot|po|mo|py|pyi|go|rs|java|kt|kts|rb|sh|bash|zsh|html|htm|svg|png|jpe?g|gif|webp|"
    r"ico|docx|xlsx|pptx|pdf|sql|env|ini|cfg|conf|lock|c|h|cpp|hpp|cs|swift|vue|graphql|proto)$",
    re.IGNORECASE,
)


def _looks_like_file_path(path: Path) -> bool:
    """Filter out backticked tokens that read like a path but are not a file.

    Pasted terminal output and error logs routinely contain absolute-looking
    tokens in backticks that are REST/API routes (`/api/v1/chat/models`,
    `/health`) or bare product/component names, not filesystem paths a
    coding agent could plausibly mutate. Only a token that already exists as
    a real file, or carries a recognizable source/config file extension, is
    treated as a claimed changed path.
    """
    try:
        if path.is_file():
            return True
    except OSError:
        pass
    return bool(_PATH_EXTENSION_RE.search(path.name))

_MUTATING_TOOLS = {
    "write_file", "edit_file", "create_file", "patch_file",
    "extract_archive", "repackage_archive",
    "create_artifact",
}

_VALIDATION_EVIDENCE_TOOLS = {
    "execute_command", "get_git_info", "read_file", "search_code", "list_directory",
}


def verified_no_change_completion(
    *, tool_records: list[dict[str, Any]], final_text: str,
) -> bool:
    """Accept a genuine evidence-backed no-op, not a narrated fake edit.

    Coding agents sometimes discover that the requested fix is already in
    the checked-out code and deployed runtime. Requiring a fresh mutation in
    that case encourages meaningless file touches and turns a truthful
    verification into a false failure. The exception is deliberately narrow:
    the report must explicitly say no change was needed/already resolved,
    inspection or validation must have succeeded, no mutating tool may have
    been attempted, and the latest command (if any) must be green.
    """
    if not _VERIFIED_NO_CHANGE_RE.search(final_text or ""):
        return False
    if any(item.get("tool_name") in _MUTATING_TOOLS for item in tool_records):
        return False
    if not any(
        item.get("tool_name") in _VALIDATION_EVIDENCE_TOOLS
        and item.get("success") is True
        for item in tool_records
    ):
        return False
    commands = [item for item in tool_records if item.get("tool_name") == "execute_command"]
    if commands:
        latest = commands[-1]
        if latest.get("success") is not True or latest.get("exit_code") not in (None, 0):
            return False
    return True


def _claims_completed_inspection(final_text: str) -> bool:
    """Return True only for prose that asserts a real inspection occurred.

    Generic completion words such as ``Fixed`` or ``Done`` remain warning-level
    when evidence is missing, preserving the existing caveat behaviour.  This
    targets the dangerous cross-workspace failure where a provider says it
    reviewed or inspected code despite recording no successful tool result.
    """
    return bool(_UNSUPPORTED_INSPECTION_CLAIM_RE.search(final_text or ""))


def _claims_mutation(final_text: str) -> bool:
    """Distinguish a real mutation claim from an explicit no-change report.

    A read-only audit naturally says things such as "no file, environment
    variable, service, or configuration was changed." The old bare regex
    matched that sentence's trailing "configuration was changed" phrase and
    rejected a correct audit as an unsupported edit.
    """
    text = final_text or ""
    for match in _MUTATION_CLAIM_RE.finditer(text):
        line_start = max(
            text.rfind("\n", 0, match.start()),
            text.rfind(".", 0, match.start()),
            text.rfind("!", 0, match.start()),
            text.rfind("?", 0, match.start()),
        ) + 1
        if not _NEGATED_MUTATION_PREFIX_RE.search(text[line_start:match.start()]):
            return True
    return False


def _successful_changed_paths(tool_records: list[dict[str, Any]], workspace_root: str) -> set[Path]:
    root = Path(workspace_root or ".").resolve()
    changed: set[Path] = set()
    for item in tool_records:
        if item.get("success") is not True or item.get("tool_name") not in _MUTATING_TOOLS:
            continue
        candidates = list(item.get("files_changed") or [])
        argument_path = (item.get("arguments") or {}).get("path")
        if argument_path:
            candidates.append(argument_path)
        for raw_path in candidates:
            candidate = Path(str(raw_path)).expanduser()
            changed.add((candidate if candidate.is_absolute() else root / candidate).resolve())
    return changed


def _successful_commands(tool_records: list[dict[str, Any]]) -> list[str]:
    return [
        str((item.get("arguments") or {}).get("command") or "").strip()
        for item in tool_records
        if item.get("tool_name") == "execute_command"
        and item.get("success") is True
        and item.get("exit_code") in (None, 0)
    ]


def validate_completion(
    *, profile: TaskProfile, tool_records: list[dict[str, Any]],
    any_mutation: bool, final_text: str, objective: str = "", workspace_root: str = "",
) -> ValidationReport:
    checks: list[dict[str, Any]] = []
    unresolved: list[str] = []
    successful_tools = [item for item in tool_records if item.get("success") is True]
    verified_no_change = verified_no_change_completion(
        tool_records=tool_records, final_text=final_text,
    )
    tool_evidence_passed = bool(successful_tools) or not profile.requires_tools
    checks.append({"name": "tool_evidence_recorded", "passed": tool_evidence_passed})
    if not tool_evidence_passed:
        unresolved.append("This task type requires tool evidence, but no successful tool call was recorded.")

    # Completion-integrity gates apply to the words in the final report,
    # independent of task classification. A malformed or vague objective can
    # be classified as QUESTION, but that must never permit a model to claim
    # files were updated, a service restarted, or production verified when
    # its own tool ledger proves otherwise.
    mutation_claimed = _claims_mutation(final_text)
    if mutation_claimed:
        changed_paths = _successful_changed_paths(tool_records, workspace_root)
        mutation_supported = bool(changed_paths) and any_mutation
        checks.append({"name": "reported_mutation_supported", "passed": mutation_supported})
        if not mutation_supported:
            unresolved.append("The response claims files or configuration were changed, but no successful file mutation supports that claim.")

        if workspace_root:
            root = Path(workspace_root).resolve()
            text = final_text or ""
            claimed_paths: list[Path] = []
            for mutation_match in _MUTATION_CLAIM_RE.finditer(text):
                window_start = max(0, mutation_match.start() - _CLAIMED_PATH_WINDOW)
                window_end = min(len(text), mutation_match.end() + _CLAIMED_PATH_WINDOW)
                window = text[window_start:window_end]
                for raw_path in _CLAIMED_PATH_RE.findall(window):
                    candidate = Path(raw_path).expanduser()
                    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
                    if not _looks_like_file_path(resolved):
                        continue
                    if resolved not in claimed_paths:
                        claimed_paths.append(resolved)
            missing_claimed_paths = [path for path in claimed_paths if path not in changed_paths]
            if claimed_paths:
                checks.append({
                    "name": "reported_paths_changed",
                    "passed": not missing_claimed_paths,
                    "claimed": [str(path) for path in claimed_paths],
                })
                if missing_claimed_paths:
                    unresolved.append(
                        "The response claims these paths were changed without successful mutation evidence: "
                        + ", ".join(str(path) for path in missing_claimed_paths[:20])
                    )

    successful_commands = _successful_commands(tool_records)
    if _SERVICE_RESTART_CLAIM_RE.search(final_text or ""):
        restart_supported = any(re.search(r"\b(?:systemctl|service|docker(?:\s+compose)?)\b.*\brestart\b", command, re.I) for command in successful_commands)
        checks.append({"name": "reported_restart_supported", "passed": restart_supported})
        if not restart_supported:
            unresolved.append("The response claims a service restart, but no successful restart command supports that claim.")

    if _LIVE_VERIFICATION_CLAIM_RE.search(final_text or ""):
        live_check_supported = any(
            re.search(r"\b(?:curl|wget|httpie|pytest|vitest|playwright|npm\s+(?:run\s+)?test|pnpm\s+(?:run\s+)?test)\b", command, re.I)
            for command in successful_commands
        )
        checks.append({"name": "reported_live_verification_supported", "passed": live_check_supported})
        if not live_check_supported:
            unresolved.append("The response claims live/frontend/API success, but no successful endpoint or behavioral test supports that claim.")

    # A real, successful `git commit` this turn is an unambiguous authorship-
    # claim trigger on its own: describing what a commit contains implicitly
    # presents those changes as this turn's own work, even when the report's
    # wording carefully avoids first-person mutation language ("Committed
    # the verified ... fixes" never matches _MUTATION_CLAIM_RE's "I fixed/
    # changed" patterns). Confirmed live: a turn found code already sitting
    # uncommitted on disk -- written by a completely different session
    # hours earlier -- ran `git commit` on it, and reported a file-by-file
    # "Changes" summary describing that code's behavior as if it had just
    # written it, with no disclosure that the work predated this turn.
    # Scoped strictly to turns with a real successful `git commit` (not just
    # any mutation claim) so this doesn't reintroduce the false-positive
    # class the windowed/file-path-filtered _CLAIMED_PATH_RE scan above was
    # built to avoid -- a bare `git commit` having actually happened is a
    # much stronger, rarer signal than generic mutation wording.
    committed_this_turn = any(
        re.search(r"\bgit\s+commit\b", command, re.I) for command in successful_commands
    )
    if committed_this_turn and workspace_root:
        root = Path(workspace_root).resolve()
        text = final_text or ""
        changed_paths = _successful_changed_paths(tool_records, workspace_root)
        claimed_paths: list[Path] = []
        for raw_path in _CLAIMED_PATH_RE.findall(text):
            candidate = Path(raw_path).expanduser()
            resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
            if not _looks_like_file_path(resolved):
                continue
            if resolved not in claimed_paths:
                claimed_paths.append(resolved)
        unattributed_paths = [path for path in claimed_paths if path not in changed_paths]
        if claimed_paths:
            checks.append({
                "name": "commit_authorship_disclosed",
                "passed": not unattributed_paths,
                "claimed": [str(path) for path in claimed_paths],
            })
            if unattributed_paths:
                unresolved.append(
                    "This turn ran `git commit`, but the response describes these paths' changes without "
                    "any successful edit tool call for them in this session: "
                    + ", ".join(str(path) for path in unattributed_paths[:20])
                    + ". If this code was already present (written by a prior/other session) rather than "
                    "authored this turn, the report must say so explicitly instead of presenting it as this "
                    "turn's own work."
                )

    if profile.task_type in {TaskType.EDIT, TaskType.DEBUG}:
        mutation_requirement_met = any_mutation or verified_no_change
        checks.append({
            "name": "mutation_recorded",
            "passed": mutation_requirement_met,
            "accepted_verified_no_change": verified_no_change,
        })
        if not mutation_requirement_met:
            unresolved.append("The request required a code change, but no successful file mutation was recorded.")

        # A successful transport response is not enough for a project build:
        # every explicitly requested output path must exist and be non-empty.
        # This catches the common long-generation failure where a model builds
        # the first few files in a supplied tree and then declares the whole
        # theme/plugin complete.
        expected_paths = _explicit_output_paths(objective)
        if expected_paths and workspace_root:
            root = Path(workspace_root).resolve()
            missing: list[str] = []
            for requested in expected_paths:
                candidate = (root / requested).resolve()
                if root not in candidate.parents and candidate != root:
                    continue
                try:
                    valid = candidate.is_file() and candidate.stat().st_size > 0
                except OSError:
                    valid = False
                if not valid:
                    missing.append(requested)
            checks.append({"name": "explicit_outputs_exist", "passed": not missing, "expected": expected_paths})
            if missing:
                unresolved.append("Explicitly requested output files are missing or empty: " + ", ".join(missing[:30]))

    if profile.requires_validation:
        # FIX: for EDIT/DEBUG specifically, a bare read_file/search_code/
        # list_directory/get_git_info call used to count as "validation
        # evidence" -- a model could report a bug fixed after only reading
        # the file it edited, with zero execute_command calls, and this
        # check passed. Narrowed to real execution evidence for exactly the
        # task types where "I fixed it" is a testable claim; other
        # requires_validation task types (e.g. INSPECT-like work, where a
        # read/search genuinely is the deliverable) keep the broader set.
        evidence_tools = (
            {"execute_command"} if profile.task_type in {TaskType.EDIT, TaskType.DEBUG}
            else _VALIDATION_EVIDENCE_TOOLS
        )
        validated = any(
            item.get("tool_name") in evidence_tools and item.get("success")
            for item in tool_records
        )

        # Verification is ordered evidence.  A later failed check/build must
        # invalidate an earlier successful command; otherwise a model can run
        # `build`, fail `check`, and still be allowed to report "all checks
        # passed".  The runner may ask the model to repair and retry, but it
        # must never call the turn complete while the latest verification is
        # red.
        commands = [item for item in tool_records if item.get("tool_name") == "execute_command"]
        latest_command = commands[-1] if commands else None
        latest_command_failed = bool(
            latest_command
            and (
                latest_command.get("success") is not True
                or latest_command.get("exit_code") not in (None, 0)
            )
        )
        checks.append({
            "name": "latest_command_clean",
            "passed": not latest_command_failed,
        })
        if latest_command_failed:
            command = latest_command.get("arguments", {}).get("command", "verification command")
            unresolved.append(
                f"The latest verification command failed: {str(command)[:160]}. "
                "The task cannot be reported complete until it is repaired and rerun successfully."
            )
            validated = False

        # A successful mutation alone is not validation for code edits.  A
        # read/inspection or command result is required, and failed latest
        # verification always wins over earlier evidence.
        checks.append({"name": "validation_evidence", "passed": validated})
        if not validated:
            unresolved.append("No successful validation or inspection tool result was recorded.")

    checks.append({"name": "non_empty_report", "passed": bool(final_text.strip())})

    passed = not unresolved and all(c["passed"] for c in checks)
    severity = "pass"
    if not passed:
        severity = "warning"
        if profile.requires_validation and any(
            check["name"] == "latest_command_clean" and not check["passed"]
            for check in checks
        ):
            severity = "error"
        if (
            profile.task_type in {TaskType.EDIT, TaskType.DEBUG}
            and not any_mutation
            and not verified_no_change
        ):
            # A requested edit/debug task without a recorded file mutation is
            # not a completed task. Warnings used to flow through
            # orchestrator.complete(), causing the CLI to claim success after
            # the model merely described a change.
            severity = "error"
        if profile.requires_tools and not successful_tools and _claims_completed_inspection(final_text):
            severity = "error"
            unresolved.append(
                "The response claims repository inspection or review, but no successful tool evidence supports that claim."
            )
        if mutation_claimed or _SERVICE_RESTART_CLAIM_RE.search(final_text or "") or _LIVE_VERIFICATION_CLAIM_RE.search(final_text or ""):
            severity = "error"
        if any(check["name"] == "commit_authorship_disclosed" and not check["passed"] for check in checks):
            severity = "error"

    return ValidationReport(passed, checks, unresolved, severity=severity)


_OUTPUT_PATH_RE = re.compile(
    r"(?<![\w./-])([\w.-]+(?:/[\w.@+-]+)+\.(?:php|css|scss|sass|js|jsx|ts|tsx|json|xml|yaml|yml|toml|md|txt|pot|po|mo|py|go|rs|java|rb|sh|html|htm|svg|png|jpe?g|webp|docx|xlsx|pptx|pdf))\b",
    re.IGNORECASE,
)


def _explicit_output_paths(objective: str) -> list[str]:
    """Extract files the user actually asked the agent to produce.

    Objectives often contain source files, tests, patch prefixes, and model
    names as context.  Treating every filename as an output contract made a
    successful edit fail after completion because unrelated files were never
    supposed to be created.  Require a nearby production verb and discard
    unified-diff ``a/``/``b/`` prefixes.
    """
    paths: list[str] = []
    for match in _OUTPUT_PATH_RE.finditer(objective or ""):
        value = match.group(1).strip().replace("\\", "/")
        if value.startswith(("http://", "https://")) or ".." in Path(value).parts:
            continue
        if value.startswith(("a/", "b/")):
            value = value[2:]
        # Use the current sentence/line, not a broad character window: a
        # nearby word such as ``implementation`` in an inspection sentence
        # must not turn a referenced file into a deliverable.
        prefix = (objective or "")[:match.start()]
        boundary = max(prefix.rfind("."), prefix.rfind("!"), prefix.rfind("?"), prefix.rfind(":"), prefix.rfind("\n"))
        context = prefix[boundary + 1:]
        production_intent = re.search(
            r"\b(?:create|generate|build|write|produce|scaffold|deliver|export|save|output)\w*\b",
            objective or "",
            re.IGNORECASE,
        )
        if not production_intent and not re.search(
            r"\b(?:create|generate|build|write|produce|scaffold|deliver|add|implement|export|save|output)\w*\b",
            context,
            re.IGNORECASE,
        ):
            continue
        if value not in paths:
            paths.append(value)
    return paths[:200]
