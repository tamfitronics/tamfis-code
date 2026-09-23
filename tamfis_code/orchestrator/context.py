"""Layered, recoverable context assembly for local agent turns."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import state as local_state
from ..routing import TaskProfile
from ..workspace import build_system_prompt, discover_local_repository
from ..openhands.skills import skill_prompt
from .compression import CacheBoundary, signature_view
from .coding_prompt import assemble_coding_messages, CODING_PROMPT_VERSION

# Elastic-injection caps (Stage 3): how many recently inspected files may
# contribute a signature view, and how large each view may be. Small on
# purpose -- the layer exists to give structure without re-inflating context.
_SIGNATURE_FILES = 6
_SIGNATURE_FILE_CHARS = 2_000


@dataclass
class ContextBundle:
    messages: list[dict[str, Any]]
    layers: dict[str, Any]
    reused: bool


def _signature_layer(state: Any, root: Path | None = None, *, limit: int = _SIGNATURE_FILES) -> tuple[str, list[str]]:
    """Stage 3 (elastic) injection for the context layer.

    Instead of listing only paths (or, worse, whole file bodies) for files
    the session already touched, inject each file's signature/docstring view
    plus a re-read pointer: enough structure to reason about the code, a
    fraction of the tokens. Best-effort -- an unreadable path is skipped, so
    this layer can never break context assembly.
    """
    rendered: list[str] = []
    included: list[str] = []
    for raw_path in list(getattr(state, "inspected_files", []) or [])[-limit:]:
        path = str(raw_path)
        if root is not None and not Path(path).is_absolute():
            path = str(Path(root) / path)
        try:
            view = signature_view(path, max_chars=_SIGNATURE_FILE_CHARS)
        except Exception:  # pragma: no cover - defensive: layer is optional
            continue
        if not view:
            continue
        rendered.append(f"--- {path} (signature view) ---\n{view}")
        included.append(path)
    if not rendered:
        return "", []
    return (
        "\n\nSignatures of files already inspected this session "
        "(bodies deliberately omitted -- re-read a path for its full body):\n"
        + "\n".join(rendered),
        included,
    )


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
    # Static, cacheable prefix: workspace instructions, rules, tools. Kept
    # byte-identical turn to turn so the provider's prefix cache can hit.
    static_prefix = build_system_prompt(session_id, Path(workspace_root))
    skills = skill_prompt(workspace_root, objective) or ""
    signature_text, signature_files = _signature_layer(state, Path(workspace_root))
    recent_tools = state.completed_actions[-8:]
    layers = {
        "policy": static_prefix,
        "coding_prompt_version": CODING_PROMPT_VERSION,
        "skills": skills,
        "objective": objective,
        "workspace_summary": repository,
        "relevant_prior_turns": conversation_messages[-12:],
        "retrieved_files": list(state.inspected_files)[-20:],
        "signature_files": signature_files,
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
    # Cache boundary: static prefix first, volatile per-turn state in its own
    # system message after it. Appending the volatile parts to the static
    # message (as this used to) invalidated the provider's prefix cache every
    # single turn -- the fingerprint/goal/validation evidence changes on
    # every request, and caching is prefix-based.
    volatile_parts = [part for part in (skills, supplemental, signature_text) if part]
    # The old workspace prompt is retained as the repository/runtime evidence
    # layer for compatibility.  The shared prompt module supplies the
    # platform and orchestration layers before it, making precedence explicit
    # for every provider and every resume/fallback that reuses this bundle.
    static_messages, prompt_sections = assemble_coding_messages(
        repository_instructions=static_prefix,
        session_context="\n\n".join(volatile_parts),
        conversation_messages=(),
    )
    static_prompt = "\n\n".join(str(message["content"]) for message in static_messages[:3])
    boundary = CacheBoundary(static_prompt, static_messages[3]["content"])
    layers["prompt_sections"] = [section.name for section in prompt_sections]
    layers["cache_boundary"] = {
        "static_chars": len(boundary.static_prefix),
        "volatile_chars": len(boundary.volatile_suffix),
    }
    return ContextBundle(
        [*boundary.as_system_messages(), *conversation_messages],
        layers, reused,
    )
