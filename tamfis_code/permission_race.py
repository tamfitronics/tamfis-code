"""Permission racing: parallel static / classifier / UI safety checks.

Sequential approval makes a safety check the latency floor of every tool call:
the agent waits for a static rule check, then a classifier, then a human, and
the human wait is what makes the loop feel like it stalls. Racing instead runs
the checks concurrently and takes the first decisive answer -- so a benign
read-only call is decided by the static rules in microseconds, a destructive
one is killed by the deny-list before a human has even read the prompt, and a
genuinely ambiguous one still ends up in front of the user.

Winner-takes-all, with the asymmetry that keeps it *safer* than a single
sequential check rather than just faster:

* a static DENY always wins -- it cancels the other processes outright, so a
  user's reflexive "yes" at the prompt cannot resurrect `rm -rf /`;
* a classifier may only *approve* when the approval policy already permits
  approving that risk, and never for `dangerous` risk -- the fast path never
  widens what is allowed, it only shortens the wait;
* a classifier may *deny* within the auto-approve window even when the policy
  would have waved the call through (that is the whole safety win);
* the UI answer wins the moment the human answers.

This module is pure orchestration: it makes no provider calls of its own except
through the optional classifier callable, so it is fully unit-testable.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable, Optional

DECISION_APPROVE = "approve_once"
DECISION_DENY = "deny"
DECISION_PROMPT = "prompt"

WINNER_STATIC = "static"
WINNER_CLASSIFIER = "classifier"
WINNER_UI = "ui"
WINNER_POLICY = "policy"
WINNER_NONE = "none"

# How long an already-auto-approved call waits for the classifier's opinion.
# Two-tier on purpose, because the tradeoff genuinely differs:
#   * ordinary auto tiers (auto/safe/workspace/accept-edits) keep a human in
#     the loop and already pre-approve the call, so they wait ZERO -- the
#     classifier is a network round trip and would become the new latency
#     floor for every write_file;
#   * the explicit zero-stop tier ("full-auto") has no human anywhere, so it
#     waits ZERO_STOP_AUTO_WINDOW_MS for the one automated guardrail it has.
DEFAULT_AUTO_WINDOW_MS = 0
ZERO_STOP_AUTO_WINDOW_MS = 500
DEFAULT_CLASSIFIER_TIMEOUT_S = 2.5
# How long the decision waits for losing racers to finish cancelling before moving on.
CANCEL_GRACE_SECONDS = 5.0

# The approval families that mean "yes" at the UI.
_APPROVE_DECISIONS = frozenset({"approve_once", "approve", "approve_session", "allow"})
_DENY_DECISIONS = frozenset({"deny", "denied", "reject", "no"})

# Catastrophic, effectively irreversible commands. Deliberately tight: these
# are the cases where no policy -- including full-auto -- should proceed
# silently. Workspace-scoped destructive commands (`rm -rf build`, `rm -rf
# /tmp/x`) are NOT here: they are dangerous risk, which the ordinary policy
# already gates.
_CATASTROPHIC_PATTERNS: tuple[tuple[str, str], ...] = (
    # The trailing lookahead accepts the command's own end as well as the
    # quoting/punctuation of a serialized tool call (`{"command": "rm -rf /"}`),
    # so the race sees the same command the shell eventually would.
    ("root_recursive_delete", r"\brm\s+(?:-[a-zA-Z]*[rR][a-zA-Z]*\s+|--recursive\s+)+(?:-[a-zA-Z]*f[a-zA-Z]*\s+|--force\s+)*(?:/\*?|~|\$HOME|\$\{HOME\})(?=[\s\"'`,;:)}]|$)"),
    ("no_preserve_root", r"--no-preserve-root"),
    ("raw_device_write", r"\bdd\b[^\n]*\bof\s*=\s*/dev/(?:sd|nvme|hd|vd|mmcblk)"),
    ("filesystem_format", r"\b(?:mkfs(?:\.\w+)?|wipefs|fdisk|parted)\b[^\n]*/dev/"),
    ("device_overwrite", r">\s*/dev/(?:sd|nvme|hd|vd|mmcblk)"),
    ("fork_bomb", r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:"),
    ("root_permissions", r"\bchmod\s+(?:-[a-zA-Z]+\s+)*777\s+/(?=[\s\"'`,;:)}]|$)"),
    ("system_power", r"\b(?:shutdown|poweroff|halt|reboot)\b(?:\s+-[a-zA-Z]+)*(?:\s+(?:now|0|/\w+))?\s*(?=[\"'`,;:)}]|$)"),
    ("wipe_system_tree", r"\brm\s+(?:-[a-zA-Z]+\s+)+/(?:etc|usr|var|boot|bin|sbin|lib|lib64|opt|sys|proc)(?:/\S*)?(?=[\s\"'`,;:)}]|$)"),
    ("authorized_keys_overwrite", r">\s*~?/?\.?ssh/authorized_keys"),
    ("history_scrub", r"\b(?:history\s+-c|shred\b[^\n]*(?:/var/log|\.bash_history))"),
)


def _flatten_arguments(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    try:
        return json.dumps(arguments or {}, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(arguments)


def catastrophic_match(text: str) -> Optional[str]:
    """Return the name of the first catastrophic deny-list pattern matching
    ``text``, or None. Pure and cheap -- this is the static process's work."""
    haystack = text or ""
    if not haystack:
        return None
    for name, pattern in _CATASTROPHIC_PATTERNS:
        try:
            if re.search(pattern, haystack, re.IGNORECASE | re.MULTILINE):
                return name
        except re.error:  # pragma: no cover - a bad pattern must not deny everything
            continue
    return None


@dataclass
class StaticVerdict:
    """Process A's answer. ``decisive`` False means "no opinion -- race it"."""

    decisive: bool = False
    decision: Optional[str] = None
    reason: str = ""
    pattern: Optional[str] = None

    @property
    def denies(self) -> bool:
        return self.decision == DECISION_DENY


def static_verdict(
    tool_name: str,
    arguments: Any,
    *,
    risk: str = "",
    approved_commands: Iterable[str] = (),
    read_only_tools: Iterable[str] = (),
) -> StaticVerdict:
    """Process A -- the regex deny-list (and the trivially-safe fast path).

    * read-only tools: decisively safe (this is what makes the common case
      instant -- no classifier call, no prompt, no waiting);
    * a catastrophic deny-list hit: decisively DENY, whoever else is asking;
    * a command this session already approved verbatim: decisively safe;
    * anything else: no opinion.
    """
    name = str(tool_name or "")
    if risk == "read_only" or name in set(read_only_tools or ()):
        return StaticVerdict(True, DECISION_APPROVE, "read-only tool", None)

    payload = _flatten_arguments(arguments)
    matched = catastrophic_match(payload)
    if matched:
        return StaticVerdict(
            True, DECISION_DENY,
            f"command matches the catastrophic deny-list rule '{matched}'", matched,
        )

    if isinstance(arguments, dict):
        command = str(arguments.get("command") or "").strip()
        if command and command in set(approved_commands or ()):
            return StaticVerdict(True, DECISION_APPROVE, "command already approved this session", None)
    return StaticVerdict(False, None, "", None)


@dataclass
class RaceOutcome:
    """The winner of the race, plus enough detail to explain it afterwards."""

    decision: str
    winner: str = WINNER_NONE
    elapsed_ms: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def denied(self) -> bool:
        return self.decision in _DENY_DECISIONS


def _normalize_decision(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    lowered = str(value).strip().lower()
    if lowered in _APPROVE_DECISIONS:
        return DECISION_APPROVE
    if lowered in _DENY_DECISIONS:
        return DECISION_DENY
    return None


async def race_permission(
    tool_name: str,
    arguments: Any,
    *,
    risk: str,
    policy: str,
    interactive: bool,
    ui_prompt: Optional[Callable[[], Awaitable[Optional[str]]]] = None,
    classifier: Optional[Callable[[str], Awaitable[Optional[str]]]] = None,
    policy_decision: Optional[Callable[[str, str, bool], Optional[str]]] = None,
    approved_commands: Iterable[str] = (),
    read_only_tools: Iterable[str] = (),
    auto_window_ms: Optional[int] = None,
    classifier_timeout_s: float = DEFAULT_CLASSIFIER_TIMEOUT_S,
) -> RaceOutcome:
    """Run the permission race for one tool call and return the winner.

    ``policy_decision`` is ``runner._decision_for_policy`` (injected, not
    imported, so this module stays free of the CLI layer). ``ui_prompt`` is the
    real prompt (already bound to console/policy/config by the caller) and is
    only invoked when the policy genuinely requires a human answer.
    """
    started = time.monotonic()

    def elapsed() -> float:
        return round((time.monotonic() - started) * 1000, 3)

    static = static_verdict(
        tool_name, arguments, risk=risk, approved_commands=approved_commands,
        read_only_tools=read_only_tools,
    )
    if static.decisive:
        return RaceOutcome(
            static.decision or DECISION_APPROVE, WINNER_STATIC, elapsed(),
            {"reason": static.reason, "pattern": static.pattern},
        )

    implied: Optional[str] = None
    if policy_decision is not None:
        try:
            implied = _normalize_decision(policy_decision(policy, risk, interactive))
        except Exception:
            implied = None

    # The classifier may only approve when the policy already permits approval
    # -- the fast path must never widen what is allowed. The one exception is
    # the explicitly opted-in zero-stop tier ("full-auto"), where there is no
    # consent to bypass in the first place; a classifier still cannot approve
    # `dangerous` risk even there. A classifier DENY wins in every tier.
    classifier_may_approve = str(risk).lower() != "dangerous" and (
        implied == DECISION_APPROVE or (implied is None and str(policy).lower() == "full-auto")
    )

    tasks: dict[asyncio.Task[Any], str] = {}
    detail: dict[str, Any] = {"risk": risk, "policy": policy, "classifier_approval_allowed": classifier_may_approve}

    if classifier is not None:
        payload = _flatten_arguments(arguments)
        async def _run_classifier() -> Optional[str]:
            try:
                raw = await asyncio.wait_for(classifier(payload), timeout=classifier_timeout_s)
            except asyncio.TimeoutError:
                detail["classifier_timeout"] = True
                return None
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                detail["classifier_error"] = f"{type(exc).__name__}: {exc}"
                return None
            # Accept either raw model prose or an already-normalized decision.
            return _normalize_intent(raw) or _normalize_decision(raw)
        tasks[asyncio.ensure_future(_run_classifier())] = WINNER_CLASSIFIER

    want_ui = ui_prompt is not None and implied is None
    if want_ui:
        async def _run_ui() -> Optional[str]:
            try:
                return _normalize_decision(await ui_prompt())
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                detail["ui_error"] = f"{type(exc).__name__}: {exc}"
                # A prompt that failed is not an approval: fail closed.
                return DECISION_DENY
        tasks[asyncio.ensure_future(_run_ui())] = WINNER_UI

    if not tasks:
        decision = implied if implied is not None else DECISION_DENY
        winner = WINNER_POLICY if implied is not None else WINNER_NONE
        detail["reason"] = "no racing processes available"
        return RaceOutcome(decision, winner, elapsed(), detail)

    if auto_window_ms is None:
        window_ms = ZERO_STOP_AUTO_WINDOW_MS if str(policy).lower() == "full-auto" else DEFAULT_AUTO_WINDOW_MS
    else:
        window_ms = max(0, auto_window_ms)
    if not want_ui:
        detail["auto_window_ms"] = window_ms
    deadline = None if want_ui else (time.monotonic() + window_ms / 1000)
    pending = set(tasks)
    try:
        while pending:
            remaining = None
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    # Auto-approve window elapsed with no verdict: proceed, which
                    # is exactly what the policy already decided.
                    detail["auto_window_expired"] = True
                    break
            done, pending = await asyncio.wait(
                pending, timeout=remaining, return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                continue
            for task in done:
                winner = tasks[task]
                try:
                    verdict = task.result()
                except (asyncio.CancelledError, Exception) as exc:  # noqa: B014 - one boundary for both
                    detail[f"{winner}_error"] = f"{type(exc).__name__}: {exc}"
                    continue
                if verdict == DECISION_DENY:
                    detail["denied_by"] = winner
                    return RaceOutcome(DECISION_DENY, winner, elapsed(), detail)
                if verdict == DECISION_APPROVE:
                    if winner == WINNER_CLASSIFIER and not classifier_may_approve:
                        detail["classifier_approval_ignored"] = True
                        continue
                    return RaceOutcome(DECISION_APPROVE, winner, elapsed(), detail)
    finally:
        for task in pending:
            task.cancel()
        if pending:
            # Await the cancellations so no "task was destroyed but it is
            # pending" warning can escape into the user's terminal -- but BOUNDED: a racer that
            # ignores cancellation (a prompt app tearing down, a classifier stuck in a client call)
            # used to block this await forever, i.e. the approved tool never ran and the run sat on
            # "Reviewing the tool result…" indefinitely. The decision is already made; move on.
            _done, _stuck = await asyncio.wait(pending, timeout=CANCEL_GRACE_SECONDS)
            for _task in _stuck:
                _task.add_done_callback(lambda t: t.cancelled() or t.exception())  # never "exception was never retrieved"

    if implied is not None:
        return RaceOutcome(implied, WINNER_POLICY, elapsed(), detail)
    # A prompt was required and nothing decisive came back (UI error, or the
    # process went away): failing closed is the only safe default.
    detail["reason"] = "no decisive verdict"
    return RaceOutcome(DECISION_DENY, WINNER_NONE, elapsed(), detail)


# --------------------------------------------------------------------------
# Process B: the fast intent classifier (AI)
# --------------------------------------------------------------------------

CLASSIFIER_SYSTEM_PROMPT = (
    "You are the command-safety classifier for a coding agent. You are shown one "
    "tool call and must decide whether executing it is safe.\n"
    "Answer UNSAFE if the call could: destroy data outside the workspace, "
    "overwrite system files, escalate privileges, exfiltrate credentials or "
    "secrets, disable security/auditing/logging, install persistence, or perform "
    "an irreversible remote action.\n"
    "Otherwise answer SAFE.\n"
    "Reply with exactly one word: SAFE or UNSAFE."
)


def classifier_messages(payload: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Tool call:\n{payload[:2000]}"},
    ]


def _normalize_intent(value: Optional[str]) -> Optional[str]:
    """Map a model's reply to DECISION_APPROVE/DECISION_DENY, or None when the
    reply is not usable (a non-answer must never be read as an approval)."""
    if not value:
        return None
    token = re.findall(r"[A-Za-z]+", str(value))[:1]
    if not token:
        return None
    word = token[0].lower()
    if word in {"unsafe", "dangerous", "malicious", "deny", "no"}:
        return DECISION_DENY
    if word in {"safe", "allow", "yes", "ok", "approve"}:
        return DECISION_APPROVE
    return None


def make_intent_classifier(
    *,
    timeout: float = DEFAULT_CLASSIFIER_TIMEOUT_S,
    temperature: float = 0.0,
    max_tokens: int = 16,
) -> Optional[Callable[[str], Awaitable[Optional[str]]]]:
    """Build the real classifier callable (Process B) over the providers
    system's own routing machinery.

    Returns None when providers are unavailable -- the race then simply runs
    without that process instead of failing the call. Never raises: a
    classifier is an optimization, not a dependency.
    """
    try:
        from .providers import ProviderManager, ProviderType
    except Exception:
        return None

    async def classifier(payload: str) -> Optional[str]:
        try:
            manager = ProviderManager()
        except Exception:
            return None
        text = ""
        try:
            chunks: list[str] = []
            async for chunk in manager.chat_completion(
                ProviderType.AUTO,
                classifier_messages(payload),
                stream=False,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_effort="low",
                timeout=timeout,
            ):
                chunks.append(str(chunk or ""))
            text = "".join(chunks).strip()
        except Exception:
            return None
        return _normalize_intent(text)

    return classifier
