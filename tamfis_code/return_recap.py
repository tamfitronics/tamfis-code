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
    text = " ".join(str(text or "").split())
    text = re.sub(r"[#>|]+|`|\*\*", "", text)     # markdown marks only; identifiers (_decode, token_id) stay intact
    text = re.sub(r"^(?:summary|overview)\s*[:\-•]?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+[-•]\s+", " ", text)
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


def _strip_context_chain(objective: str) -> str:
    """The first sentence of a "<task>\\n\\nAdditional user context: ..." chain -- the original task."""
    head = re.split(r"\s*Additional user context:", str(objective or ""), maxsplit=1)[0]
    return head.strip() or str(objective or "")


def _clean_recap_text(text: str) -> str:
    """Remove presentation syntax and repair only high-confidence typos."""
    value = str(text or "")
    try:
        from .text_corrector import correct_objective_text

        value = correct_objective_text(value).corrected
    except Exception:
        pass
    value = re.sub(r"```.*?```", " ", value, flags=re.DOTALL)
    value = re.sub(r"(?:^|\s)[#>*`]+", " ", value)
    value = re.sub(r"\*+", "", value)
    return " ".join(value.split()).strip(" -:;,.\t\n")


def _smart_objective(objective: str) -> str:
    """Turn a noisy request into a compact intent statement.

    This is deliberately evidence-free: it rewrites what the user wants, not
    what supposedly happened. Status and conclusions are built separately
    from the task ledger below.
    """
    cleaned = _clean_recap_text(_strip_context_chain(objective))
    if not cleaned:
        return "not recorded"

    lowered = cleaned.casefold()
    bulk_failure = bool(re.search(
        r"\bbulk\b.{0,80}\b(?:not work(?:ing)?|fail(?:s|ed|ing)?|no progress|stuck|stall(?:ed|ing)?)\b",
        lowered,
    ))
    post_failure = bool(re.search(
        r"\bposts?\b.{0,70}\b(?:not (?:be )?generated|generation.{0,20}fail|fail(?:s|ed|ing)?.{0,20}generat)",
        lowered,
    ))
    if bulk_failure or post_failure:
        subjects: list[str] = []
        if bulk_failure:
            subjects.append("stalled bulk operations")
        if post_failure:
            subjects.append("post-generation failures")
        scope = ""
        if re.search(r"\b(?:all|three)\s+(?:wp|wordpress)\s+sites?\b", lowered):
            scope = " across all WordPress sites"
        names_match = re.search(
            r"\b(?:especially|particularly)(?:\s+for|\s+on)?\s+([A-Za-z][A-Za-z0-9_-]*(?:\s+and\s+[A-Za-z][A-Za-z0-9_-]*)?)",
            cleaned,
            flags=re.IGNORECASE,
        )
        emphasis = f", especially {names_match.group(1)}" if names_match else ""
        if len(subjects) == 2:
            subject_text = f"{subjects[0]} and {subjects[1]}"
        else:
            subject_text = subjects[0]
        return _clip(f"Diagnose and fix {subject_text}{scope}{emphasis}", 240)

    # Generic cleanup for long conversational requests: remove politeness
    # and background asides, then retain task-bearing clauses. Short, already
    # clear objectives stay intact.
    if len(cleaned) <= 180 and len(re.split(r"[.!?]", cleaned)) <= 2:
        return _clip(cleaned, 240)
    clauses = re.split(r"(?<=[.!?;])\s+|\bthen also\b", cleaned, flags=re.IGNORECASE)
    task_clauses = [
        clause for clause in clauses
        if not re.search(r"^\s*(?:meanwhile|for context|background|i (?:had|have|was))\b", clause, re.IGNORECASE)
    ]
    concise = " ".join(task_clauses or clauses)
    concise = re.sub(r"\b(?:so\s+)?please\b", "", concise, flags=re.IGNORECASE)
    concise = re.sub(r"\bre-?investigate deeply and see why\b", "Diagnose", concise, flags=re.IGNORECASE)
    return _first_sentences(concise, 240) or _clip(cleaned, 240)


def _clean_next_action(text: str) -> str:
    value = _clean_recap_text(text)
    value = re.sub(r"^(?:next(?:\s+step)?|todo|action)\s*[:\-]\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^\s*(?:[-+]|\d+[.)])\s*", "", value)
    if value.casefold() in {"", "none", "done", "completed", "n/a", "no next step"}:
        return NO_NEXT_STEP
    return _clip(value, 240)


def _next_step(state: Any, session_id: int, last_answer: str) -> str:
    try:
        from .runtime.resume import describe_resume_point

        point = describe_resume_point(session_id)
        if point:
            action = f" -- {point['next_action']}" if point.get("next_action") else ""
            return _clean_next_action(f"continue at step {point['step']}/{point['total']}: {point['name']}{action}")
    except Exception:
        pass
    plan = next((p for p in reversed(state.saved_plans or []) if p.get("id") == state.active_plan_id), None)
    if plan:
        pending = next((s for s in plan.get("steps") or [] if str(s.get("status")) in {"pending", "in_progress"}), None)
        if pending:
            return _clean_next_action(str(pending.get("description") or pending.get("title") or pending.get("name") or "the next plan step"))
    try:
        from .runtime.ledger import load_ledger

        ledger = load_ledger(str(session_id))
        if ledger is not None and ledger.next_action and ledger.status in {"running", "partial", "blocked", "checkpointing"}:
            return _clean_next_action(ledger.next_action)
    except Exception:
        pass
    # A completed structured plan has no verified remaining step. Do not
    # resurrect a speculative "Next:" line from the model's last prose (the
    # source of unsupported items such as "Check worker connectivity**").
    if plan:
        return NO_NEXT_STEP
    try:
        from .interactive import next_message_suggestion

        suggestion = next_message_suggestion(last_answer, None, state=state)
        if suggestion:
            return _clean_next_action(suggestion)
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
        objective = _strip_context_chain(str(plan["objective"]))
    elif (state.active_task or {}).get("objective") and state.execution_status not in {"completed", "idle"}:
        objective = _strip_context_chain(str(state.active_task["objective"]))
    elif turns:
        # The latest turn a person actually wrote: a submitted "Repair the failed plan step ..." or
        # "Continue from the saved checkpoint ..." suggestion is machinery, not the task.
        from .runner_local import _is_machine_generated_objective, _is_resume_request

        human = [t["objective"] for t in turns if not _is_machine_generated_objective(t["objective"])
                 and not _is_resume_request(t["objective"])]
        objective = _strip_context_chain(human[-1] if human else turns[-1]["objective"])
    else:
        objective = state.session_title or ""

    objective = _smart_objective(objective)
    last_answer = turns[-1]["answer"] if turns else ""
    parts: list[str] = []
    status = str(state.execution_status or "")
    if status in {"failed", "interrupted", "cancelled"}:
        parts.append(f"the last run {status}")
    if plan and plan.get("steps"):
        steps = plan["steps"]
        from .plan_panel import plan_progress_label
        done = sum(1 for s in steps if str(s.get("status")) == "completed")
        parts.append(f"plan {done}/{len(steps)} steps done ({plan_progress_label(steps)})")
    # Prefer the task-scoped ledger.  SessionState.modified_files is a safety
    # history across the whole session and may contain edits from an earlier
    # objective; using it here was the source of stale Finitron paths showing
    # up in a later task's recap.
    edits: list[dict[str, Any]] = []
    try:
        from .runtime.ledger import load_ledger

        ledger = load_ledger(str(session_id))
        if ledger is not None:
            edits = [
                {
                    "path": edit.file,
                    "operation": edit.operation,
                    "revert_status": "none",
                }
                for edit in ledger.edits
                if edit.applied and edit.file
            ]
    except Exception:
        edits = []
    if not edits:
        edits = [
            m for m in (state.modified_files or [])
            if m.get("path") and m.get("revert_status") != "reverted"
        ]
    files = [str(m.get("path")) for m in edits if m.get("path")]
    unique_files = list(dict.fromkeys(reversed(files)))[:4]
    if unique_files:
        grouped: dict[str, list[str]] = {"added": [], "updated": [], "removed": []}
        for item in edits:
            path = str(item.get("path") or "")
            if not path:
                continue
            operation = str(item.get("operation") or "update").casefold()
            category = "added" if operation in {"create", "add", "added"} else (
                "removed" if operation in {"delete", "remove", "removed"} else "updated"
            )
            grouped[category].append(path.rsplit("/", 1)[-1])
        labels = [
            f"{name} {', '.join(list(dict.fromkeys(values))[:4])}"
            for name, values in grouped.items() if values
        ]
        parts.append("; ".join(labels))
    # Structured task facts outrank assistant prose. The old implementation
    # copied the opening of the last answer, turning tentative reasoning such
    # as "So source_min is 300..." into an asserted project status.
    structured = bool(plan or unique_files or status in {"failed", "interrupted", "cancelled"})
    validation = ""
    try:
        from .runtime.ledger import load_ledger

        ledger = load_ledger(str(session_id))
        if ledger is not None:
            structured = True
            tests = [test for test in ledger.tests if test.status != "not_run"]
            if tests:
                passed = sum(1 for test in tests if test.status == "passed")
                validation = f"validation {passed}/{len(tests)} passed"
            elif ledger.status in {"running", "partial", "blocked", "checkpointing"}:
                validation = "no completed validation is recorded"
            if ledger.status in {"partial", "blocked", "failed"}:
                parts.insert(0, f"task status is {ledger.status}")
    except Exception:
        pass
    outcome = ""
    if not structured and (last_answer or state.conversation_summary):
        outcome = _first_sentences(last_answer or state.conversation_summary, 260)
    standing = "; ".join(filter(None, [outcome, ", ".join(parts), validation])) or "no verified progress was recorded"
    return ReturnRecap(
        objective=objective,
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

    console_width = max(40, int(getattr(console, "width", 100) or 100))
    for label, value in (("Objective", recap.objective), ("Where it stands", recap.standing), ("Next", recap.next_step)):
        # Wrap the value *inside* the available width. The previous renderer
        # wrapped the already-prefixed line at the full console width, then
        # Rich wrapped it a second time and introduced table-like `│` glyphs
        # plus continuation text under the label column. Keep one stable
        # left-aligned key/value layout instead.
        head = f"  {label}:"
        value_width = max(20, console_width - len(head) - 1)
        lines = textwrap.wrap(
            str(value or ""), width=value_width,
            break_long_words=True, break_on_hyphens=False,
        ) or [""]
        value_style = "dim" if value == NO_NEXT_STEP else ""
        text = Text(head, style="bold") + Text(" " + lines[0], style=value_style)
        continuation_indent = " " * (len(head) + 1)
        for line in lines[1:]:
            text.append("\n" + continuation_indent + line, style=value_style)
        console.print(text, highlight=False, soft_wrap=False, overflow="fold")
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
