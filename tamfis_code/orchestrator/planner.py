"""Deterministic planning policy for complex coding turns, plus reasoning-
based plan generation (build_reasoning_plan_prompt/parse_reasoning_plan).

The deterministic create_plan() below is a fixed template -- the same
generic "Inspect / Select provider / Execute / Repair / Validate / Report"
steps regardless of what the task actually is. It exists as an always-
available synchronous fallback (no provider call, cannot fail). The
reasoning path (runner_local.py calls build_reasoning_plan_prompt, sends it
to the resolved provider, then feeds the raw response to
parse_reasoning_plan) produces a plan grounded in the real objective and
real workspace facts instead -- confirmed live: the old template was
functionally the same "plan" for a one-line typo fix and a full-stack
audit, i.e. not a plan at all, just boilerplate methodology text.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from ..routing import TaskProfile, TaskType

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```\s*$")
MAX_REASONING_PLAN_STEPS = 8


@dataclass
class PlanStep:
    index: int
    name: str
    status: str = "pending"
    evidence: list[str] = field(default_factory=list)


@dataclass
class ExecutionPlan:
    objective: str
    assumptions: list[str]
    components: list[str]
    steps: list[PlanStep]
    validation_criteria: list[str]
    risks: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def should_plan(profile: TaskProfile) -> bool:
    return profile.complexity == "high" or profile.task_type in {
        TaskType.AUDIT, TaskType.EDIT, TaskType.DEBUG, TaskType.TEST, TaskType.MIXED,
    }


def create_plan(objective: str, profile: TaskProfile) -> ExecutionPlan | None:
    if not should_plan(profile):
        return None
    steps = [
        PlanStep(1, "Inspect the relevant repository context and manifests"),
        PlanStep(2, "Select a capable provider/model and the minimum required tools"),
        PlanStep(3, "Execute the requested work and observe every tool result"),
    ]
    if profile.requires_validation:
        steps.extend([
            PlanStep(4, "Repair failures or incomplete changes using observed evidence"),
            PlanStep(5, "Validate the resulting repository state"),
        ])
    steps.append(PlanStep(len(steps) + 1, "Report only claims supported by recorded evidence"))
    return ExecutionPlan(
        objective=objective,
        assumptions=["Existing working functionality must be preserved"],
        components=["repository", "provider routing", "tool runtime"],
        steps=steps,
        validation_criteria=["No unsupported completion claims", "Requested changes or findings are evidenced"],
        risks=["Provider tool incompatibility", "Unrelated workspace mutations", "Context-window exhaustion"],
    )


_REASONING_PLAN_SYSTEM = (
    "You are the planning stage of a repository engineering agent. Produce a short, "
    "CONCRETE execution plan using only the objective and repository evidence supplied. "
    "Never invent package managers, scripts, services, migrations, frameworks, directories, "
    "or commands. A command may appear in the plan only when the evidence explicitly shows "
    "the corresponding manifest or script. Do not use vague steps such as 'inspect the repo', "
    "'look for bugs', 'manually test', 'ensure dependencies are installed', or 'select a "
    "provider'. Name the actual repositories, files, modules, routes, stores, services, tests, "
    "or manifests that will be traced. For audits and fixes, begin with the most relevant "
    "evidence-backed code path, then verification. Respond with ONLY a JSON object, no prose "
    "and no markdown fences, matching exactly this shape:\n"
    '{"steps": ["...", "..."], "assumptions": ["..."], "risks": ["..."]}\n'
    f"Use 2 to {MAX_REASONING_PLAN_STEPS} steps. Each step must be a short, specific sentence. "
    "If evidence is insufficient for a command, plan a read-only inspection of the named "
    "manifest or source path first instead of guessing the command."
)



def build_reasoning_plan_prompt(
    objective: str, profile: TaskProfile, workspace_summary: dict[str, Any],
    *, reconnaissance_summary: Optional[str] = None,
    evidence_summary: Optional[str] = None,
) -> list[dict[str, str]]:
    """Build a one-shot, tool-free planning request from verified facts.

    ``reconnaissance_summary`` is deterministic read-only discovery performed
    before the first plan is shown. ``evidence_summary`` is only for a later
    revision after real tool observations exist.
    """
    languages = ", ".join(workspace_summary.get("detected_languages") or []) or "unknown"
    frameworks = ", ".join(workspace_summary.get("frameworks") or []) or "none detected"
    test_commands = ", ".join(workspace_summary.get("test_commands") or []) or "none verified"
    build_commands = ", ".join(workspace_summary.get("build_commands") or []) or "none verified"
    manifests = ", ".join(workspace_summary.get("project_manifests") or []) or "none recorded"
    repository_root = str(workspace_summary.get("repository_root") or "unknown")

    user_lines = [
        f"Objective: {objective}",
        f"Task type: {profile.task_type.value}",
        f"Repository root: {repository_root}",
        f"Detected languages: {languages}",
        f"Detected frameworks: {frameworks}",
        f"Recorded manifests: {manifests}",
        f"Verified test commands: {test_commands}",
        f"Verified build commands: {build_commands}",
    ]
    if reconnaissance_summary:
        user_lines.append(
            "\nDETERMINISTIC READ-ONLY RECONNAISSANCE (authoritative; use these exact "
            "paths and scripts, and do not invent anything absent from this section):\n"
            + reconnaissance_summary
        )
    else:
        user_lines.append(
            "\nNo deterministic reconnaissance was available. Do not propose install, "
            "build, migration, start, or test commands. Limit the plan to reading the "
            "explicitly recorded repository root and manifests first."
        )
    if evidence_summary:
        user_lines.append(
            "\nREAL TOOL EVIDENCE GATHERED DURING EXECUTION (this is a PLAN REVISION; "
            "supersede assumptions that conflict with this evidence):\n" + evidence_summary
        )
    return [
        {"role": "system", "content": _REASONING_PLAN_SYSTEM},
        {"role": "user", "content": "\n".join(user_lines)},
    ]


def parse_reasoning_plan(raw_content: str, *, objective: str) -> Optional[ExecutionPlan]:
    """Parse a planning completion's raw text into an ExecutionPlan.

    Never raises -- returns None for anything that isn't a valid,
    non-empty plan (malformed JSON, empty steps, wrong shape). Callers must
    fall back to the deterministic create_plan() on None; a plan that
    fails to parse is not evidence the task itself failed.
    """
    text = (raw_content or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        stripped = _CODE_FENCE_RE.sub("", text).strip()
        try:
            data = json.loads(stripped)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
    if not isinstance(data, dict):
        return None

    raw_steps = data.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        return None
    steps: list[PlanStep] = []
    for item in raw_steps[:MAX_REASONING_PLAN_STEPS]:
        name = str(item).strip()
        if name:
            steps.append(PlanStep(len(steps) + 1, name))
    if not steps:
        return None

    def _string_list(key: str) -> list[str]:
        raw = data.get(key)
        if not isinstance(raw, list):
            return []
        return [str(item).strip() for item in raw if str(item).strip()]

    return ExecutionPlan(
        objective=objective,
        assumptions=_string_list("assumptions") or ["Existing working functionality must be preserved"],
        components=[],
        steps=steps,
        validation_criteria=["No unsupported completion claims", "Requested changes or findings are evidenced"],
        risks=_string_list("risks") or ["Context-window exhaustion"],
    )