"""Context-aware classification of live user steering messages."""
from __future__ import annotations

from enum import Enum


class SteeringIntent(str, Enum):
    APPROVAL = "approval"
    REJECTION = "rejection"
    CANCELLATION = "cancellation"
    CORRECTION = "correction"
    PRIORITY_CHANGE = "priority_change"
    QUESTION = "question"
    FOLLOW_UP = "follow_up"


def classify_live_input(text: str, *, approval_visible: bool = False) -> SteeringIntent:
    cleaned = " ".join(text.strip().lower().split())
    if approval_visible and cleaned in {"y", "yes", "approve", "a", "always"}:
        return SteeringIntent.APPROVAL
    if approval_visible and cleaned in {"n", "no", "reject"}:
        return SteeringIntent.REJECTION
    if cleaned in {"cancel", "stop", "abort", "esc"}:
        return SteeringIntent.CANCELLATION
    if cleaned.startswith(("do not ", "don't ", "correction:", "actually ")):
        return SteeringIntent.CORRECTION
    if "priority" in cleaned or cleaned.startswith("use "):
        return SteeringIntent.PRIORITY_CHANGE
    if cleaned.endswith("?"):
        return SteeringIntent.QUESTION
    return SteeringIntent.FOLLOW_UP
