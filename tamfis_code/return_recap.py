"""A short "Conversation recap" for a user who is coming back to a session.

    ─ Conversation recap ────────────────────────────────────────────
      Objective: build a geo-aware BetPredict operator list ...
      Where it stands: a first compliant version was built and deployed ...
      Next: no next step was recorded

Shown when a session is resumed, when the user returns to an idle prompt after being away, and by `/recap`
(the long turn-by-turn card stays under `/summary`). Built only from what the session durably recorded --
no model call, so it is instant, free, and cannot invent a next step: when nothing was recorded it says so.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from . import state as local_state

NO_NEXT_STEP = "no next step was recorded"
AWAY_RECAP_DEFAULT_MINUTES = 10.0


@dataclass
class ReturnRecap:
    objective: str
    standing: str
    next_step: str
    files: list[str] = field(default_factory=list)


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _first_sentences(text: str, limit: int) -> str:
    text = re.sub(r"[#*`>_|]+", "", " ".join(str(text or "").split()))
    picked = ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if not sentence:
            continue
        if picked and len(picked) + len(sentence) + 1 > limit:
            break
        picked = f"{picked} {sentence}".strip()
        if len(picked) >= limit * 0.6:
            break
    return _clip(picked or text, limit)


def _next_step(state: Any, session_id: int, last_answer: str) -> str:
    try:
        from .runtime.resume import describe_resume_point

        point = describe_resume_point(session_id)
        if point:
            action = f" -- {point['next_action']}" if point.get("next_action") else ""
            return _clip(f"continue at step {point['step']}/{point['total']}: {point['name']}{action}", 240)
    except Exception:
        pass
    plan = next((p for p in reversed(state.saved_plans or []) if p.get("id") == state.active_plan_id), None)
    if plan:
        pending = next((s for s in plan.get("steps") or [] if str(s.get("status")) in {"pending", "in_progress"}), None)
        if pending:
            return _clip(str(pending.get("description") or pending.get("title") or pending.get("name") or "the next plan step"), 240)
    try:
        from .runtime.ledger import load_ledger

        ledger = load_ledger(str(session_id))
        if ledger is not None and ledger.next_action and ledger.status in {"running", "partial", "blocked", "checkpointing"}:
            return _clip(ledger.next_action, 240)
    except Exception:
        pass
    try:
        from .interactive import next_message_suggestion

        suggestion = next_message_suggestion(last_answer, None, state=state)
        if suggestion:
            return _clip(suggestion, 240)
    except Exception:
        pass
    return NO_NEXT_STEP


def build_return_recap(session_id: int) -> Optional[ReturnRecap]:
    """The recap for ``session_id``, or None when the session has nothing recorded to recap."""
    state = local_state.get_session_state(session_id)
    turns = local_state._extract_turns(state.conversation_history or [])
    if not turns and not (state.conversation_summary or "").strip() and not state.saved_plans:
        return None

    plan = next((p for p in reversed(state.saved_plans or []) if p.get("id") == state.active_plan_id), None)
    objective = ""
    if plan and plan.get("objective"):
        objective = str(plan["objective"])
    elif (state.active_task or {}).get("objective") and state.execution_status not in {"completed", "idle"}:
        objective = str(state.active_task["objective"])
    elif turns:
        objective = turns[-1]["objective"]
    else:
        objective = state.session_title or ""

    last_answer = turns[-1]["answer"] if turns else ""
    parts: list[str] = []
    status = str(state.execution_status or "")
    if status in {"failed", "interrupted", "cancelled"}:
        parts.append(f"the last run {status}")
    if plan and plan.get("steps"):
        steps = plan["steps"]
        done = sum(1 for s in steps if str(s.get("status")) == "completed")
        parts.append(f"plan {done}/{len(steps)} steps done")
    files = [
        str(m.get("path")) for m in (state.modified_files or [])
        if m.get("path") and m.get("revert_status") != "reverted"
    ]
    unique_files = list(dict.fromkeys(reversed(files)))[:4]
    if unique_files:
        parts.append("changed " + ", ".join(p.rsplit("/", 1)[-1] for p in unique_files))
    outcome = _first_sentences(last_answer or state.conversation_summary, 260) if (last_answer or state.conversation_summary) else ""
    standing = "; ".join(filter(None, [outcome, ", ".join(parts)])) or "no progress was recorded"
    return ReturnRecap(
        objective=_clip(objective, 300) or "not recorded",
        standing=standing,
        next_step=_next_step(state, session_id, last_answer),
        files=unique_files,
    )


def render_return_recap(console: Any, recap: ReturnRecap) -> None:
    from rich.markup import escape
    from rich.rule import Rule
    from rich.text import Text

    console.print(Rule(Text("Conversation recap", style="dim"), style="dim", characters="─"))
    import textwrap

    width = max(40, int(getattr(console, "width", 100) or 100) - 2)
    for label, value in (("Objective", recap.objective), ("Where it stands", recap.standing), ("Next", recap.next_step)):
        head = f"  {label}: "
        lines = textwrap.wrap(value, width=width, initial_indent=head, subsequent_indent=" " * len(head)) or [head]
        first, rest = lines[0][len(head):], lines[1:]
        text = Text(head, style="bold") + Text(first, style="dim" if value == NO_NEXT_STEP else "")
        for line in rest:
            text.append("\n" + line, style="dim" if value == NO_NEXT_STEP else "")
        console.print(text, highlight=False, soft_wrap=True)
    console.print()


def away_threshold_seconds() -> float:
    """Idle time at the prompt before the recap is shown; 0 disables (TAMFIS_CODE_AWAY_RECAP_MINUTES)."""
    try:
        return max(0.0, float(os.environ.get("TAMFIS_CODE_AWAY_RECAP_MINUTES", AWAY_RECAP_DEFAULT_MINUTES))) * 60.0
    except (TypeError, ValueError):
        return AWAY_RECAP_DEFAULT_MINUTES * 60.0


def print_return_recap(console: Any, session_id: int) -> bool:
    recap = build_return_recap(session_id)
    if recap is None:
        return False
    render_return_recap(console, recap)
    return True
