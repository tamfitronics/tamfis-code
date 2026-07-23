"""Layered, recoverable context assembly for local agent turns."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import state as local_state
from ..routing import TaskProfile
from ..workspace import build_system_prompt, discover_local_repository


@dataclass
class ContextBundle:
    messages: list[dict[str, Any]]
    layers: dict[str, Any]
    reused: bool


def build_context_bundle(
    *, session_id: int, workspace_root: str, objective: str,
    profile: TaskProfile, conversation_messages: list[dict[str, Any]],
    plan: dict[str, Any] | None = None,
) -> ContextBundle:
    if profile.is_plain_conversation:
        system = (
            "You are TamfisGPT Code. Respond naturally and concisely. "
            "Do not invent or call tools for ordinary conversation."
        )
        layers = {"policy": system, "objective": objective}
        return ContextBundle([{"role": "system", "content": system}, *conversation_messages], layers, True)

    before = local_state.get_session_state(session_id)
    previous_fingerprint = before.discovery_fingerprint
    repository = discover_local_repository(session_id, Path(workspace_root))
    state = local_state.get_session_state(session_id)
    reused = bool(previous_fingerprint and previous_fingerprint == state.discovery_fingerprint)
    system = build_system_prompt(session_id, Path(workspace_root))
    recent_tools = state.completed_actions[-8:]
    layers = {
        "policy": system,
        "objective": objective,
        "workspace_summary": repository,
        "relevant_prior_turns": conversation_messages[-12:],
        "retrieved_files": list(state.inspected_files)[-20:],
        "recent_tool_results": recent_tools,
        "active_plan": plan or {},
        "validation_state": state.validation_results[-10:],
    }
    # A short preview only -- the real, full objective is already present
    # verbatim as the latest message in conversation_messages below. Before
    # this fix, the full (unbounded) objective was duplicated here too:
    # for a large pasted objective (a long log/diff as the request), this
    # put a second full copy inside the leading system message, which none
    # of runner_local.py's compaction passes can touch (role=="system" is
    # deliberately left alone -- it carries essential workspace
    # instructions that must survive compaction), so the turn kept failing
    # on token budget even after the user-facing copy was compacted.
    objective_preview = objective if len(objective) <= 400 else f"{objective[:400]}... [{len(objective) - 400} more characters in the actual request below]"
    # `plan` (an ExecutionPlan.to_dict()) carries its own `objective` field,
    # a second full copy of the same text -- str()'d into the prompt below
    # via `plan or 'none'`. Bound that copy the same way, for the same
    # reason (should_plan() often applies to exactly the complex/high-token
    # tasks most likely to carry a large objective in the first place).
    bounded_plan = dict(plan) if plan else None
    if bounded_plan is not None and "objective" in bounded_plan:
        bounded_plan["objective"] = objective_preview
    supplemental = (
        "\n\nActive orchestration context (recoverable from local state):\n"
        f"Objective (preview -- see the actual latest user message for the full request): {objective_preview}\n"
        f"Repository fingerprint: {state.discovery_fingerprint}\n"
        f"Active plan: {bounded_plan or 'none'}\n"
        f"Recent validation evidence: {state.validation_results[-5:]}"
    )
    return ContextBundle(
        [{"role": "system", "content": system + supplemental}, *conversation_messages],
        layers, reused,
    )
