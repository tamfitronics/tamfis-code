"""Deterministic intent checks before an agent is allowed to act.

The model remains responsible for nuanced engineering decisions, but obvious
ambiguity and requests that conflict with basic safety/legal practice should
not reach the tool loop as an unqualified instruction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any


@dataclass(frozen=True)
class PreflightQuestion:
    question: str
    header: str
    options: list[dict[str, str]]


@dataclass(frozen=True)
class PreflightResult:
    proceed: bool
    objective: str
    question: PreflightQuestion | None = None
    reason: str = ""
    warnings: tuple[str, ...] = field(default_factory=tuple)


# These are intentionally conservative phrases. They identify a request that
# needs a human decision; they do not attempt to decide whether a particular
# person or organization is legally permitted to act.
_UNSAFE_PATTERNS = (
    (re.compile(r"\b(?:bypass|circumvent|evade)\b.{0,40}\b(?:auth|login|license|paywall|security|rate limit)\b", re.I),
     "I can help design or test an authorized, documented solution, but I will not bypass access controls, licensing, or rate limits."),
    (re.compile(r"\b(?:steal|dump|exfiltrate|harvest)\b.{0,50}\b(?:password|credential|token|personal|private|customer)\b", re.I),
     "I can help with consent-based data handling, redaction, or incident response, but not credential or private-data extraction."),
    (re.compile(r"\b(?:malware|ransomware|keylogger|credential stealer|botnet|ddos)\b", re.I),
     "I can help with defensive analysis, detection, sandboxing, or a benign proof of concept—not deployment or abuse."),
    (re.compile(r"\b(?:mass|bulk)\s+(?:submit|spam|scrape)\b|\bsubmit\s+to\s+(?:all|hundreds|thousands)\b", re.I),
     "I can help build a rate-limited, consent-based integration, but not unsolicited bulk submission or spam automation."),
)

_AMBIGUOUS_PATTERNS = (
    re.compile(r"^(?:fix|repair|change|update|improve|refactor|review|audit)\s+(?:it|this|that|everything|all)\b", re.I),
    re.compile(r"\b(?:make|do)\s+(?:it|this)\s+(?:better|work|smart|professional)\b", re.I),
    re.compile(r"\b(?:fix|audit|improve)\s+everything\b", re.I),
)


def preflight_intent(objective: str, *, workspace_root: str = "") -> PreflightResult:
    """Classify only cases that need a human decision before model/tool work."""
    text = " ".join((objective or "").split())
    if not text:
        return PreflightResult(False, text, reason="An objective is required.")

    for pattern, warning in _UNSAFE_PATTERNS:
        if pattern.search(text):
            return PreflightResult(
                False,
                text,
                question=PreflightQuestion(
                    "How should I redirect this request?",
                    "Safety boundary",
                    [
                        {"label": "Use a lawful defensive approach (Recommended)", "description": warning},
                        {"label": "Explain the authorized design first", "description": "Provide architecture, threat modeling, or a safe test plan without executing the risky operation."},
                        {"label": "Stop", "description": "Do not send this request to the coding agent."},
                    ],
                ),
                reason=warning,
                warnings=(warning,),
            )

    for pattern in _AMBIGUOUS_PATTERNS:
        if pattern.search(text):
            return PreflightResult(
                False,
                text,
                question=PreflightQuestion(
                    "What concrete outcome and scope do you want before I change anything?",
                    "Clarify scope",
                    [
                        {"label": "Inspect and recommend first (Recommended)", "description": "Read the relevant project and report the highest-impact issue without changing files."},
                        {"label": "Fix the named component", "description": "You will specify the component, files, or failing behavior in your next message."},
                        {"label": "Create a plan only", "description": "Produce a grounded plan and wait for approval before edits."},
                    ],
                ),
                reason="The request does not identify a concrete component, failure, or acceptance criterion.",
            )

    return PreflightResult(True, text)


def apply_preflight_answer(result: PreflightResult, answer: str) -> str:
    """Convert a selected preflight option into an explicit safe instruction."""
    choice = (answer or "").casefold()
    if "inspect" in choice or "defensive" in choice or "authorized design" in choice:
        return f"{result.objective}\n\nPreflight decision: inspect first and do not make changes until scope and authorization are clear."
    if "plan" in choice:
        return f"{result.objective}\n\nPreflight decision: create a grounded plan only; wait for approval before edits."
    return f"{result.objective}\n\nPreflight decision: proceed only within the explicitly authorized, lawful scope."
