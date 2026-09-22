"""Evidence-constrained planning for Tamfis-Code.

The planner is repository-agnostic. It never assumes Python, Node, pytest,
npm, Alembic, Docker, Git, a source layout, or a test runner. Plans may use
only paths and commands verified by deterministic reconnaissance.

Existing public APIs remain compatible:
- PlanStep
- ExecutionPlan
- should_plan
- create_plan
- build_reasoning_plan_prompt
- parse_reasoning_plan
"""
from __future__ import annotations

import json
import re
import shlex
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from ..routing import ComplexityLevel, TaskProfile, TaskType, complexity_at_least

MAX_REASONING_PLAN_STEPS = 8
MAX_ASSUMPTIONS = 6
MAX_RISKS = 8
MAX_EVIDENCE_ITEMS = 12
# A genuinely large objective (spanning many components/roots) drafts as
# several per-phase plans instead of forcing MAX_REASONING_PLAN_STEPS steps
# to cover work that doesn't fit in one reliable LLM draft. Bounded, not
# unbounded, for the same reliability reason MAX_REASONING_PLAN_STEPS is
# bounded: each phase is still one call, one 8-step cap.
MAX_PLAN_PHASES = 6

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![\w.-])/(?:[A-Za-z0-9._~+\-]+/)*[A-Za-z0-9._~+\-]+"
)
_BACKTICK_RE = re.compile(r"`([^`\n]+)`")
_COMMAND_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"python(?:\d+(?:\.\d+)*)?\s+-m\s+pytest|pytest|"
    r"npm|npx|pnpm|yarn|bun|deno|"
    r"cargo|go|dotnet|mvn|gradle|./gradlew|"
    r"make|cmake|meson|ninja|"
    r"php|composer|bundle|rake|mix|"
    r"docker|podman|kubectl|helm|"
    r"bash|sh|powershell|pwsh"
    r")\b",
    re.IGNORECASE,
)

# Marker names are recognition hints only. Referencing one in a plan requires
# that deterministic reconnaissance actually found it.
_KNOWN_MANIFEST_NAMES = {
    "package.json", "pnpm-workspace.yaml", "yarn.lock", "package-lock.json",
    "bun.lock", "bun.lockb", "deno.json", "deno.jsonc", "pyproject.toml",
    "pytest.ini", "tox.ini", "setup.py", "setup.cfg", "requirements.txt",
    "poetry.lock", "pdm.lock", "uv.lock", "cargo.toml", "cargo.lock",
    "go.mod", "go.sum", "pom.xml", "build.gradle", "build.gradle.kts",
    "gradlew", "composer.json", "gemfile", "mix.exs", "makefile",
    "cmakelists.txt", "dockerfile", "docker-compose.yml",
    "docker-compose.yaml", "compose.yml", "compose.yaml", "alembic.ini",
    "vitest.config.ts", "vitest.config.js", "jest.config.ts",
    "jest.config.js", "playwright.config.ts", "playwright.config.js",
    "tsconfig.json",
}

_COMMAND_INTENT_RE = re.compile(
    r"\b(?:run|execute|build|test|lint|format|typecheck|compile|migrate|"
    r"install|start|stop|restart|deploy|package|publish)\b",
    re.IGNORECASE,
)
_READ_ONLY_INTENT_RE = re.compile(
    r"\b(?:inspect|read|review|trace|map|inventory|locate|compare|examine|"
    r"identify|summarise|analyze|analyse)\b",
    re.IGNORECASE,
)


@dataclass
class PlanStep:
    index: int
    name: str
    status: str = "pending"
    evidence: list[str] = field(default_factory=list)
    # 0 = ungrouped (today's exact meaning for every existing/legacy plan).
    # >0 = 1-based index into ExecutionPlan.phase_names, set only by
    # merge_phase_plans for a genuinely large, multi-phase objective.
    phase: int = 0


@dataclass
class ExecutionPlan:
    objective: str
    assumptions: list[str]
    components: list[str]
    steps: list[PlanStep]
    validation_criteria: list[str]
    risks: list[str]
    # Empty = today's exact flat-plan shape. Non-empty only when
    # merge_phase_plans built this plan from several per-phase drafts;
    # phase_names[i-1] names PlanStep.phase == i.
    phase_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def _reindex(self) -> None:
        for index, step in enumerate(self.steps, start=1):
            step.index = index

    def deduplicate_steps(self) -> None:
        """Drop repeated action items while preserving their first status/evidence."""
        unique: list[PlanStep] = []
        seen: set[str] = set()
        for step in self.steps:
            key = " ".join(step.name.split()).casefold()
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(step)
        self.steps = unique
        self._reindex()

    def add_step(self, name: str, *, after: int | None = None, evidence: list[str] | None = None) -> PlanStep:
        step = PlanStep(0, name.strip(), evidence=list(evidence or []))
        if not step.name:
            raise ValueError("plan step name cannot be empty")
        if after is None:
            self.steps.append(step)
        else:
            position = max(0, min(int(after), len(self.steps)))
            self.steps.insert(position, step)
        self._reindex()
        return step

    def edit_step(self, index: int, *, name: str | None = None, status: str | None = None) -> PlanStep:
        if index < 1 or index > len(self.steps):
            raise IndexError("plan step index out of range")
        step = self.steps[index - 1]
        if name is not None:
            cleaned = name.strip()
            if not cleaned:
                raise ValueError("plan step name cannot be empty")
            step.name = cleaned
        if status is not None:
            if status not in {"pending", "in_progress", "completed", "blocked", "skipped"}:
                raise ValueError("invalid plan step status")
            step.status = status
        return step

    def remove_step(self, index: int) -> PlanStep:
        if index < 1 or index > len(self.steps):
            raise IndexError("plan step index out of range")
        removed = self.steps.pop(index - 1)
        self._reindex()
        return removed

    def next_pending(self, *, phase: int | None = None) -> PlanStep | None:
        """First step still pending, in plan order.

        `phase=None` (default) scans the whole plan, in step order --
        because merge_phase_plans lays out steps in phase order, this walks
        phase 1's steps to completion before ever reaching phase 2's,
        exactly the linear turn-by-turn advancement _advance_plan_step
        already implements by hand. `phase=N` narrows to just that phase,
        for a caller that already knows which phase it's driving.
        """
        return next(
            (
                step for step in self.steps
                if step.status == "pending" and (phase is None or step.phase == phase)
            ),
            None,
        )

    def remaining_step_count(self, *, phase: int | None = None) -> int:
        return sum(
            1 for step in self.steps
            if step.status != "completed" and (phase is None or step.phase == phase)
        )


def plan_phase_count(plan: ExecutionPlan) -> int:
    """1 for an ordinary flat plan (today's exact shape); len(phase_names)
    for a plan merge_phase_plans built from several per-phase drafts."""
    return len(plan.phase_names) or 1


def merge_phase_plans(objective: str, phase_plans: list[tuple[str, ExecutionPlan]]) -> ExecutionPlan:
    """Combine one ExecutionPlan per phase (each independently drafted and
    evidence-validated by _attempt_phase_plans, one call per phase) into a
    single ExecutionPlan, the same shape every other consumer of
    ExecutionPlan.steps already expects -- steps concatenated in phase
    order and tagged with PlanStep.phase, not a nested structure, so
    _advance_plan_step/edit_plan_step/the /plans renderer/state.py's
    update_plan_steps all keep working unmodified across phases.

    `phase_plans` is `[(phase_name, plan), ...]` in the order phases should
    run. A phase whose drafting pass produced nothing usable is dropped
    silently (same "fewer real phases than requested" tolerance
    _generate_video_script_content's precedent uses) -- raises only if
    every phase came back empty, since then there is nothing to merge.
    """
    steps: list[PlanStep] = []
    phase_names: list[str] = []
    assumptions: list[str] = []
    risks: list[str] = []
    validation_criteria: list[str] = []
    components: list[str] = []

    for phase_name, plan in phase_plans:
        if plan is None or not getattr(plan, "steps", None):
            continue
        phase_index = len(phase_names) + 1
        phase_names.append(phase_name)
        for step in plan.steps:
            steps.append(PlanStep(0, step.name, status=step.status, evidence=list(step.evidence), phase=phase_index))
        assumptions.extend(plan.assumptions)
        risks.extend(plan.risks)
        validation_criteria.extend(plan.validation_criteria)
        components.extend(plan.components)

    if not steps:
        raise ValueError("merge_phase_plans: no phase produced any usable steps")

    merged = ExecutionPlan(
        objective=objective,
        assumptions=_dedupe_strings(assumptions),
        components=_dedupe_strings(components),
        steps=steps,
        validation_criteria=_dedupe_strings(validation_criteria),
        risks=_dedupe_strings(risks),
        phase_names=phase_names,
    )
    merged._reindex()
    return merged


@dataclass
class PlannerEvidence:
    """Structured facts extracted from deterministic reconnaissance."""

    roots: list[Path] = field(default_factory=list)
    existing_paths: set[Path] = field(default_factory=set)
    manifest_paths: set[Path] = field(default_factory=set)
    connected_paths: set[Path] = field(default_factory=set)
    verified_commands: set[str] = field(default_factory=set)
    languages: set[str] = field(default_factory=set)
    frameworks: set[str] = field(default_factory=set)
    raw_summary: str = ""

    @property
    def has_repository_facts(self) -> bool:
        return bool(
            self.roots
            or self.existing_paths
            or self.manifest_paths
            or self.verified_commands
        )

    def path_is_authorised(self, candidate: Path) -> bool:
        """Security boundary only: is `candidate` inside an authorised root.

        Deliberately does NOT require the path to exist. Confirmed live: a
        plan step naming the file a create/add/scaffold objective is about
        to bring into existence (e.g. "Create utils.py with square()") was
        silently dropped by validate_plan_step because reconnaissance is
        read-only and runs before anything is created, so it can never see
        a not-yet-created path -- for an objective made up entirely of such
        steps, every step was rejected, `parse_reasoning_plan` returned
        None, and the turn fell all the way back to the generic template
        plan. Requiring existence conflated "may this turn touch this
        location" with "does this already exist"; only the former is
        actually a security boundary. This only governs whether a plan
        *step description* survives -- it does not authorise any actual
        file write, which still goes through the tool layer's own
        workspace-root/sandbox/approval gates regardless of what the plan
        said.
        """
        resolved = _safe_resolve(candidate)
        if resolved is None:
            return False
        if self.roots:
            return any(_is_within(resolved, root) for root in self.roots)
        # No configured roots at all is not the normal live path (scope_roots
        # always includes at least the workspace root) but keep the original,
        # stricter existence-only behaviour for that unconfigured case.
        return resolved.exists()

    def path_was_discovered(self, candidate: Path) -> bool:
        """Return whether a path is both authorised and relevant to this stack.

        Authorised-root membership is only the security boundary. When
        reconnaissance supplied an architecture graph, ordinary source files
        must also be connected through an import, entry point, route, service,
        build dependency, manifest reference, test target, or an explicit
        objective match.

        Repository roots and verified manifests remain valid structural
        anchors. When no graph evidence exists, retain backwards-compatible
        validation rather than pretending a graph was discovered -- an
        authorised, not-yet-existing path is accepted here (see
        path_is_authorised) precisely so a legitimate creation target is not
        rejected merely for not existing yet at read-only reconnaissance time.
        """
        resolved = _safe_resolve(candidate)
        if resolved is None or not self.path_is_authorised(resolved):
            return False

        if resolved in self.manifest_paths or resolved in self.roots:
            return True

        if not self.connected_paths:
            # No architecture graph to check relevance against -- the guard
            # above already confirmed `resolved` is inside an authorised
            # root, which is all this backwards-compatible fallback ever
            # required for something that already exists. A not-yet-existing
            # path additionally needs a real parent directory (the root
            # itself counts) so a wholly fabricated, multi-level-deep path
            # still isn't accepted just for naming an authorised root as an
            # ancestor.
            return resolved.exists() or resolved.parent.exists()

        if resolved in self.connected_paths:
            return True

        # A directory is relevant when it contains a connected descendant.
        if resolved.is_dir() and any(
            _is_within(connected, resolved)
            for connected in self.connected_paths
        ):
            return True

        # A discovered child may be represented by a connected parent module.
        if any(
            connected.is_dir() and _is_within(resolved, connected)
            for connected in self.connected_paths
        ):
            return True

        # A not-yet-existing path sitting next to (or inside a directory
        # that is itself inside/below) evidence reconnaissance already
        # confirmed relevant is grounded the same way an existing sibling
        # file would be -- e.g. a new test file proposed alongside an
        # existing, objective-matched test module. Only the create/add
        # case needs this: an existing-but-unconnected file is correctly
        # still rejected above, on the theory that a real path the agent
        # can already read is not evidence it's relevant on its own.
        if not resolved.exists():
            parent = resolved.parent
            if parent.exists() and any(
                _is_within(connected, parent) or _is_within(parent, connected)
                for connected in self.connected_paths
            ):
                return True

        return False

    def manifest_name_was_discovered(self, name: str) -> bool:
        lowered = Path(name).name.lower()
        return any(path.name.lower() == lowered for path in self.manifest_paths)

    def command_is_verified(self, command: str) -> bool:
        candidate = _normalise_command(command)
        if not candidate:
            return False
        return any(
            candidate == verified
            or candidate.startswith(verified + " ")
            or verified.startswith(candidate + " ")
            for verified in self.verified_commands
        )


_FORMAL_PLAN_RE = re.compile(
    r"\b(?:multi[- ]file|multiple files|several files|across (?:the )?(?:repo|repository|stack)|"
    r"end[- ]to[- ]end|architecture|architectural|migrat(?:e|ion)|redesign|large[- ]scale|"
    r"whole (?:repo|repository|stack)|entire (?:repo|repository|stack)|complex|roadmap|"
    r"coding agent|reasoning (?:ability|abilities|quality)|agentic (?:ability|abilities|workflow)|"
    r"developer experience)\b",
    re.IGNORECASE,
)
_FILE_REFERENCE_RE = re.compile(r"(?<![\w.-])[\w@+-]+(?:/[\w@+.-]+)*\.[A-Za-z0-9]{1,10}\b")


def should_plan(profile: TaskProfile, objective: str | None = None) -> bool:
    """Use formal plans only when task shape warrants their latency/noise.

    ``objective=None`` preserves the legacy profile-only API for callers that
    genuinely have no request text. Runtime callers always provide it.
    """
    if objective is None:
        return complexity_at_least(profile.complexity, ComplexityLevel.COMPLEX) or profile.task_type in {
            TaskType.AUDIT, TaskType.EDIT, TaskType.DEBUG, TaskType.TEST, TaskType.MIXED,
        }
    if profile.task_type in {TaskType.PLAN, TaskType.AUDIT, TaskType.MIXED}:
        return True
    text = objective.strip()
    if profile.task_type not in {TaskType.EDIT, TaskType.DEBUG, TaskType.TEST}:
        return False
    if _FORMAL_PLAN_RE.search(text) or len(text) >= 320:
        return True
    return len(set(_FILE_REFERENCE_RE.findall(text))) >= 2


def is_formal_planning_objective(profile: TaskProfile, objective: str) -> bool:
    """Narrower than should_plan(): true only for should_plan's PLAN/AUDIT/
    MIXED and explicit-scale (_FORMAL_PLAN_RE / >=320 chars) branches --
    deliberately excludes should_plan's narrowest trigger (>=2 distinct
    file references for an EDIT/DEBUG/TEST objective), which is the
    cheapest and most common way `/plan` fires today and must keep costing
    exactly one LLM call. Gates the multi-pass drafting extension (outline,
    per-phase calls, verify/critique pass) in runner_local.py so a plan
    that already fits in one call never pays for passes it doesn't need.
    """
    text = objective.strip()
    if profile.task_type in {TaskType.PLAN, TaskType.AUDIT, TaskType.MIXED}:
        return True
    if profile.task_type not in {TaskType.EDIT, TaskType.DEBUG, TaskType.TEST}:
        return False
    return bool(_FORMAL_PLAN_RE.search(text)) or len(text) >= 320


def create_plan(
    objective: str,
    profile: TaskProfile,
    *,
    reconnaissance_summary: Optional[str] = None,
    workspace_summary: Optional[dict[str, Any]] = None,
    scope_roots: Optional[Sequence[str | Path]] = None,
) -> ExecutionPlan | None:
    """Create a safe deterministic fallback plan without guessed technology."""
    if not should_plan(profile, objective):
        return None

    evidence = build_planner_evidence(
        reconnaissance_summary=reconnaissance_summary,
        workspace_summary=workspace_summary or {},
        scope_roots=scope_roots,
    )

    # Step text is deliberately terse -- Claude Code/Codex-style single-line
    # action items, not full sentences. This is the deterministic plan shown
    # to the user immediately (before any reasoning-plan revision), so its
    # verbosity was the first, most visible instance of the "plans are too
    # long per item" complaint. Every constraint that matters (which root,
    # which manifest, which verified command) is still named explicitly;
    # only the connective, explanatory prose around it is trimmed.
    steps: list[PlanStep] = []
    if evidence.roots:
        for root in evidence.roots[:4]:
            steps.append(
                PlanStep(
                    len(steps) + 1,
                    f"Inventory `{root}` for objective-relevant components.",
                    evidence=[f"path:{root}"],
                )
            )
    else:
        steps.append(
            PlanStep(
                1,
                "Review the workspace inventory for objective-relevant components.",
            )
        )

    # 2026-09-17 (plan dedup + de-padding, owner report: template plans
    # listed four /tmp test dirs, two checkpoints and a generic validation
    # command as if they were the work): the deterministic plan must not
    # pad itself with boilerplate that merely repeats what later steps do.
    # Key changes:
    #   * manifests already covered by an inventory step are not re-read
    #     by a separate "Read project metadata" step -- that step is only
    #     added for manifests OUTSIDE inventoried roots;
    #   * a validation step naming a command nobody agreed to run is
    #     dropped when no verified command exists ("Validate using only
    #     commands discovered during execution" is execution guidance,
    #     not a plan step);
    #   * the generic inventory step is skipped entirely when a connected
    #     path step already makes the work concrete.
    if evidence.manifest_paths:
        outside_roots = [
            path for path in evidence.manifest_paths
            if not any(
                str(path).startswith(str(root).rstrip("/") + "/")
                or Path(str(path)).parent == Path(str(root))
                for root in evidence.roots
            )
        ]
        if outside_roots:
            paths = sorted(outside_roots, key=str)[:6]
            rendered = ", ".join(str(path) for path in paths)
            steps.append(
                PlanStep(
                    len(steps) + 1,
                    f"Read project metadata: {rendered}.",
                    evidence=[f"path:{path}" for path in paths],
                )
            )

    if evidence.connected_paths:
        paths = sorted(evidence.connected_paths, key=str)[:6]
        rendered = ", ".join(str(path) for path in paths)
        steps.append(PlanStep(
            len(steps) + 1,
            f"Trace objective-relevant paths: {rendered}.",
            evidence=[f"path:{path}" for path in paths],
        ))
    else:
        steps.append(PlanStep(
            len(steps) + 1,
            f"Trace the code paths governing: {objective.strip()[:140]}.",
        ))

    if profile.task_type in {TaskType.EDIT, TaskType.DEBUG, TaskType.MIXED}:
        steps.append(
            PlanStep(
                len(steps) + 1,
                "Apply the smallest evidence-backed change; preserve unrelated behaviour.",
            )
        )

    if profile.requires_validation and evidence.verified_commands:
        command = sorted(evidence.verified_commands)[0]
        steps.append(
            PlanStep(
                len(steps) + 1,
                f"Validate with `{command}`; investigate any failure.",
                evidence=[f"command:{command}"],
            )
        )

    steps.append(
        PlanStep(
            len(steps) + 1,
            "Report evidence-backed findings, changes, validations, and risks.",
        )
    )

    return ExecutionPlan(
        objective=objective,
        assumptions=[
            "Existing working functionality and user-authored changes must be preserved."
        ],
        components=[str(root) for root in evidence.roots],
        steps=steps[:MAX_REASONING_PLAN_STEPS],
        validation_criteria=[
            "Every referenced path exists inside an authorised root.",
            "Every executed command is derived from discovered repository metadata or an explicit user instruction.",
            "No completion claim is made without observed tool evidence.",
        ],
        risks=[
            "Repository evidence may be incomplete until targeted inspection is performed.",
            "Generated, vendored, backup, or hidden trees may resemble active source and must not be treated as canonical without evidence.",
        ],
    )


_REASONING_PLAN_SYSTEM = f"""
You are the evidence-constrained planning stage of a general-purpose repository
engineering agent.

You are not planning for any particular repository, company, language, framework,
package manager, test runner, deployment system, or operating system.

NON-NEGOTIABLE RULES

1. First understand the user's intent from the exact objective: identify the
   requested outcome, explicit constraints, affected scope, and acceptance
   evidence. Preserve those requirements throughout the plan.
2. Reconcile that intent with the supplied current-state/repository evidence
   before drafting or displaying any plan. If evidence conflicts with the
   request, surface the conflict as an assumption or risk; do not silently
   substitute a generic workflow.
3. Start from the supplied deterministic reconnaissance. Do not start from model priors or conventional project layouts.
4. Never invent a file, directory, manifest, script, command, service, migration,
   route, module, framework, package manager, or test runner.
5. A path may appear only when it is present in the authoritative reconnaissance
   or is an explicitly supplied authorised root.
6. A command may appear only when it is listed as a verified command in the
   authoritative reconnaissance or explicitly requested by the user.
7. Do not assume that pyproject.toml, package.json, pytest, npm, Alembic, Docker,
   Git, tests, migrations, src, app, frontend, or backend exist.
8. Do not include provider selection, generic methodology, or vague steps such as
   'inspect the repository', 'look for bugs', 'ensure dependencies', or 'run
   tests'. Name the verified target and purpose.
8a. Keep "action" a single short imperative line (roughly 12 words), the way a
   terse engineering checklist reads -- not a full sentence explaining itself.
   "purpose" is separate and is not shown next to the action, so do not repeat
   it inside "action".
8b. Never pad the plan with recon bookkeeping that repeats earlier steps:
   if a step already inventories a root, do not add another step to "read
   metadata" for files inside that same root, and never spend a step on
   "report findings" or "plan next steps" -- reporting is not plan work.
9. When evidence is insufficient, plan a bounded read-only inventory of an
   authorised root or a verified path. Do not fill gaps with guesses.
10. For multi-root work, keep each root explicit. Never collapse the common parent
   into a workspace target and never rewrite a supplied absolute path.
11. Put execution or mutation after evidence gathering. Put validation after the
   intended change. Audits remain read-only unless the objective requests fixes.
12. Use between 2 and {MAX_REASONING_PLAN_STEPS} steps.

Return ONLY one JSON object with this exact shape:

{{
  "steps": [
    {{
      "action": "short specific action",
      "targets": ["/verified/absolute/path"],
      "command": null,
      "purpose": "why this step is needed",
      "evidence": ["path:/verified/absolute/path"]
    }}
  ],
  "assumptions": [],
  "risks": []
}}

For a verified command, put the exact command in "command" and cite it as
"command:<exact verified command>" in evidence.

Do not return markdown fences or prose outside the JSON object.
""".strip()

# Appended only to the top-level (non-phase-scoped) draft call -- a
# per-phase call already knows its own phase and must not recurse into
# declaring further sub-phases of itself.
_REASONING_PLAN_MULTI_PHASE_ADDENDUM = f"""

11. If, and only if, this objective genuinely spans multiple independent
    components/roots/subsystems too large for {MAX_REASONING_PLAN_STEPS} steps to
    cover honestly, also set "multi_phase": true and "phase_outline" to an
    ordered list of {{"name": "<short phase name>", "description": "<1-2 sentence
    scope of this phase>"}} covering the whole objective, at most {MAX_PLAN_PHASES}
    phases. Leave "multi_phase" false (or omit it) for anything that fits in one
    normal plan -- this is the uncommon case, not the default.

When "multi_phase" is true, "steps" may be empty or a short top-level summary --
the real per-phase steps are drafted separately, one call per phase.
""".strip()


def _reasoning_plan_system_prompt(*, allow_multi_phase: bool) -> str:
    if not allow_multi_phase:
        return _REASONING_PLAN_SYSTEM
    return _REASONING_PLAN_SYSTEM + "\n\n" + _REASONING_PLAN_MULTI_PHASE_ADDENDUM


def reasoning_plan_system_prompt() -> str:
    """Public accessor for the base (non-multi-phase) reasoning-plan system
    prompt, for a caller building its own one-off tool-free completion in
    the same style (e.g. runner_local.py's _verify_and_critique_plan) --
    avoids either duplicating this prompt or reaching across the module
    boundary for the private constant.
    """
    return _REASONING_PLAN_SYSTEM


def build_reasoning_plan_prompt(
    objective: str,
    profile: TaskProfile,
    workspace_summary: dict[str, Any],
    *,
    reconnaissance_summary: Optional[str] = None,
    evidence_summary: Optional[str] = None,
    scope_roots: Optional[Sequence[str | Path]] = None,
    phase_scope: Optional[dict[str, Any]] = None,
) -> list[dict[str, str]]:
    """Build a tool-free planning request from verified repository facts.

    `phase_scope`, when supplied, narrows this call to drafting just one
    phase of a larger, already-outlined objective: `{"name": str,
    "description": str, "other_phases": [str, ...]}`. The system prompt
    drops the multi-phase self-reporting addendum in this case (a per-phase
    call must not itself declare further phases) and the payload gains a
    "phase_scope" key instructing the model to draft only this phase's
    steps, using the other phases' names purely as non-duplication context.
    """
    evidence = build_planner_evidence(
        reconnaissance_summary=reconnaissance_summary,
        workspace_summary=workspace_summary,
        scope_roots=scope_roots,
    )

    payload: dict[str, Any] = {
        "objective": objective,
        "task_type": getattr(profile.task_type, "value", str(profile.task_type)),
        "requires_validation": bool(getattr(profile, "requires_validation", False)),
        "intent_contract": {
            "user_request": objective,
            "task_type": getattr(profile.task_type, "value", str(profile.task_type)),
            "requires_tools": bool(getattr(profile, "requires_tools", False)),
            "requires_repository_context": bool(getattr(profile, "requires_repository_context", False)),
            "requires_validation": bool(getattr(profile, "requires_validation", False)),
            "instruction": (
                "Before drafting steps, restate the requested outcome, constraints, "
                "scope, and acceptance evidence internally from this exact request. "
                "Do not replace it with a generic inspect/execute/test template."
            ),
        },
        "authoritative_reconnaissance": {
            "authorised_roots": [str(root) for root in evidence.roots],
            "discovered_paths": [
                str(path) for path in sorted(evidence.existing_paths, key=str)[:240]
            ],
            "discovered_manifests": [
                str(path) for path in sorted(evidence.manifest_paths, key=str)[:80]
            ],
            "verified_commands": sorted(evidence.verified_commands)[:80],
            "detected_languages": sorted(evidence.languages),
            "detected_frameworks": sorted(evidence.frameworks),
        },
    }

    if reconnaissance_summary:
        payload["raw_reconnaissance"] = reconnaissance_summary
    else:
        payload["reconnaissance_warning"] = (
            "No deterministic reconnaissance was supplied. Do not propose any "
            "path-specific or command-bearing step. Limit the plan to obtaining a "
            "bounded inventory of the authorised roots."
        )

    if evidence_summary:
        payload["observed_execution_evidence"] = evidence_summary
        payload["revision_instruction"] = (
            "REVISION: Replace assumptions contradicted by observed execution "
            "evidence, ground every changed step in real findings, and do not "
            "repeat completed work."
        )

    if phase_scope:
        payload["phase_scope"] = phase_scope
        payload["phase_instruction"] = (
            f"This objective was already broken into phases. Draft ONLY the "
            f"{MAX_REASONING_PLAN_STEPS}-step-or-fewer plan for phase "
            f"'{phase_scope.get('name', '')}' ({phase_scope.get('description', '')}). "
            f"Do not draft steps for the other phases listed in phase_scope."
            f"other_phases -- they are shown only so you don't duplicate their work."
        )

    return [
        {"role": "system", "content": _reasoning_plan_system_prompt(allow_multi_phase=phase_scope is None)},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def extract_phase_outline(raw_content: str) -> Optional[list[dict[str, str]]]:
    """Pull just the top-level draft's self-reported phase_outline out of
    the same raw completion parse_reasoning_plan already consumes, without
    duplicating its step-validation logic. Returns None when the model
    didn't set multi_phase=true or supplied no usable phase names -- the
    caller (_is_formal_planning_objective's corroboration check in
    runner_local.py) treats that as "not actually multi-phase" regardless
    of what multi_phase itself said, so a bare, empty, or malformed
    phase_outline is equivalent to not having declared one at all.
    """
    data = _load_json_object(raw_content)
    if not isinstance(data, dict) or not data.get("multi_phase"):
        return None
    raw_outline = data.get("phase_outline")
    if not isinstance(raw_outline, list):
        return None
    outline: list[dict[str, str]] = []
    for item in raw_outline[:MAX_PLAN_PHASES]:
        if not isinstance(item, dict):
            continue
        name = " ".join(str(item.get("name") or "").split())
        if not name:
            continue
        outline.append({"name": name, "description": " ".join(str(item.get("description") or "").split())})
    return outline or None


def parse_reasoning_plan(
    raw_content: str,
    *,
    objective: str,
    reconnaissance_summary: Optional[str] = None,
    workspace_summary: Optional[dict[str, Any]] = None,
    scope_roots: Optional[Sequence[str | Path]] = None,
    rejection_log: Optional[list[str]] = None,
) -> Optional[ExecutionPlan]:
    """Parse and evidence-validate a reasoning plan.

    Unsupported, invented, non-existent, or out-of-scope steps are removed.
    Returns None when no usable steps remain.

    `rejection_log`, when supplied, is appended with one short reason per
    dropped raw step -- purely diagnostic, so callers can tell whether a
    fallback to the generic template happened because the model proposed
    nothing usable at all vs. because grounding evidence was too thin to
    verify what it proposed (see _attempt_reasoning_plan's diagnostics).
    """
    data = _load_json_object(raw_content)
    if data is None:
        return None

    evidence = build_planner_evidence(
        reconnaissance_summary=reconnaissance_summary,
        workspace_summary=workspace_summary or {},
        scope_roots=scope_roots,
    )

    raw_steps = data.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        return None

    strict_evidence = bool(
        reconnaissance_summary
        or workspace_summary
        or scope_roots
        or evidence.has_repository_facts
        or evidence.connected_paths
    )

    accepted: list[PlanStep] = []
    accepted_names: set[str] = set()
    for raw_step in raw_steps[:MAX_REASONING_PLAN_STEPS]:
        candidate = _parse_step_candidate(raw_step)
        if candidate is None:
            if rejection_log is not None:
                rejection_log.append(f"unparseable step: {raw_step!r}"[:200])
            continue

        name, targets, command, declared_evidence = candidate

        if strict_evidence:
            step_rejection_reasons: list[str] = []
            validated = validate_plan_step(
                name=name,
                targets=targets,
                command=command,
                evidence=evidence,
                rejection_reasons=step_rejection_reasons if rejection_log is not None else None,
            )
            if validated is None:
                if rejection_log is not None:
                    reason = step_rejection_reasons[0] if step_rejection_reasons else "unverified claim"
                    rejection_log.append(f"{reason}: {name!r}"[:200])
                continue
            rendered_name, validated_evidence = validated
        else:
            # Backwards-compatible parsing for callers that provide only the
            # model JSON. Runtime planning always supplies reconnaissance and
            # therefore uses the strict evidence path above.
            rendered_name = " ".join(name.split())
            validated_evidence = []

        # Providers occasionally repeat a validation action verbatim (often
        # after a continuation/retry).  Repeating it in the visible plan
        # creates a fake extra step and can make an already-completed check
        # appear pending again.
        step_key = " ".join(rendered_name.split()).casefold()
        if step_key in accepted_names:
            continue
        accepted_names.add(step_key)

        accepted.append(
            PlanStep(
                index=len(accepted) + 1,
                name=rendered_name,
                evidence=_dedupe_strings(
                    [*declared_evidence, *validated_evidence]
                )[:MAX_EVIDENCE_ITEMS],
            )
        )

    if not accepted:
        return None

    assumptions = _safe_string_list(
        data.get("assumptions"),
        MAX_ASSUMPTIONS,
    ) or [
        "The plan will be revised when repository evidence contradicts an initial step."
    ]
    risks = _safe_string_list(
        data.get("risks"),
        MAX_RISKS,
    ) or [
        "Initial repository evidence may be incomplete until the relevant dependency graph is inspected."
    ]

    return ExecutionPlan(
        objective=objective,
        assumptions=assumptions,
        components=[str(root) for root in evidence.roots],
        steps=accepted,
        validation_criteria=[
            "Every referenced path exists inside an authorised root.",
            "Every source target belongs to the pertinent repository graph when graph evidence is available.",
            "Every command is backed by deterministic reconnaissance or an explicit user instruction.",
            "Every completion claim is supported by observed tool evidence.",
        ],
        risks=risks,
    )


def validate_plan_step(
    *,
    name: str,
    targets: Sequence[str],
    command: Optional[str],
    evidence: PlannerEvidence,
    rejection_reasons: Optional[list[str]] = None,
) -> Optional[tuple[str, list[str]]]:
    """Return a grounded step, or None when it contains unsupported claims.

    `rejection_reasons`, when supplied, gets exactly one short diagnostic
    string appended at the point of rejection -- for _sync_plan_progress-
    style observability into why a plan collapsed to the generic template,
    not for control flow.
    """
    def _reject(reason: str) -> None:
        if rejection_reasons is not None:
            rejection_reasons.append(reason)

    clean_name = " ".join(str(name or "").split())
    if not clean_name:
        _reject("empty action text")
        return None

    validated_evidence: list[str] = []
    resolved_targets: list[Path] = []

    # A partial credit: an unverified entry in the structured `targets` list
    # (metadata, not prose the step's action text asserts) is dropped on its
    # own rather than discarding a step that also names other, genuinely
    # grounded targets -- one bad target field entry among several good ones
    # doesn't mean the whole step is a hallucination. Only when EVERY
    # supplied target fails does that signal the step is really about
    # something unverifiable, so the whole-step rejection still applies then.
    # Paths/commands asserted inside the action text itself (below) keep the
    # strict all-or-nothing rule -- there's no way to "drop" a claim that's
    # baked into the prose the user will actually read.
    dropped_targets: list[str] = []
    for raw_target in targets:
        target = _safe_resolve(Path(str(raw_target)).expanduser())
        if target is None or not evidence.path_was_discovered(target):
            dropped_targets.append(str(raw_target))
            continue
        resolved_targets.append(target)
        validated_evidence.append(f"path:{target}")

    if targets and not resolved_targets:
        _reject(f"no supplied target was in discovered evidence: {dropped_targets!r}")
        return None

    for raw_path in _ABSOLUTE_PATH_RE.findall(clean_name):
        target = _safe_resolve(Path(raw_path.rstrip(".,;:)]}")))
        if target is None or not evidence.path_was_discovered(target):
            _reject(f"path mentioned in action not in discovered evidence: {raw_path!r}")
            return None
        if target not in resolved_targets:
            resolved_targets.append(target)
            validated_evidence.append(f"path:{target}")

    lowered_name = clean_name.lower()
    for manifest_name in _KNOWN_MANIFEST_NAMES:
        if (
            manifest_name in lowered_name
            and not evidence.manifest_name_was_discovered(manifest_name)
        ):
            _reject(f"manifest named but not discovered: {manifest_name!r}")
            return None

    clean_command = _normalise_command(command or "")
    if clean_command:
        if not evidence.command_is_verified(clean_command):
            _reject(f"command not verified: {clean_command!r}")
            return None
        validated_evidence.append(f"command:{clean_command}")
    else:
        embedded_commands = [
            item.strip()
            for item in _BACKTICK_RE.findall(clean_name)
            if _COMMAND_PREFIX_RE.search(item)
        ]
        for embedded in embedded_commands:
            if not evidence.command_is_verified(embedded):
                _reject(f"embedded command not verified: {embedded!r}")
                return None
            validated_evidence.append(f"command:{_normalise_command(embedded)}")

        if (
            _COMMAND_INTENT_RE.search(clean_name)
            and not _READ_ONLY_INTENT_RE.search(clean_name)
            and not embedded_commands
        ):
            _reject("command-intent language with no verified/embedded command")
            return None

    rendered = clean_name
    if clean_command and f"`{clean_command}`" not in rendered:
        rendered = f"{rendered.rstrip('.')} using `{clean_command}`."

    return rendered, _dedupe_strings(validated_evidence)


def build_planner_evidence(
    *,
    reconnaissance_summary: Optional[str],
    workspace_summary: dict[str, Any],
    scope_roots: Optional[Sequence[str | Path]] = None,
) -> PlannerEvidence:
    """Convert deterministic reconnaissance and workspace facts into evidence."""
    evidence = PlannerEvidence(raw_summary=reconnaissance_summary or "")

    for root in scope_roots or ():
        resolved = _safe_resolve(Path(root).expanduser())
        if resolved is not None and resolved.is_dir():
            evidence.roots.append(resolved)

    # Confirmed live: a session whose actual objective is scoped to one
    # subproject (e.g. /home/tamfisseo) but whose cached repository_context
    # resolved to a much broader wrapping git root (e.g. /home, itself a
    # real repo spanning many unrelated sibling projects) produced a plan
    # inventorying /tmp, sibling projects' manifests, and another session's
    # scratchpad files -- none relevant to the objective. scope_roots is the
    # narrower, per-turn security boundary already threaded through tool
    # authorization elsewhere in runner_local.py; when it's available it is
    # a strictly better answer to "what should this plan be scoped to" than
    # the broad, HEAD/dirty-fingerprint-cached repository_root, so only fall
    # back to repository_root when no scope_roots were given at all.
    if not evidence.roots:
        repository_root = workspace_summary.get("repository_root")
        if repository_root:
            resolved = _safe_resolve(Path(str(repository_root)).expanduser())
            if resolved is not None and resolved.is_dir():
                evidence.roots.append(resolved)

    for key in ("project_manifests", "manifests"):
        for item in _iter_values(workspace_summary.get(key)):
            path = _safe_resolve(Path(str(item)).expanduser())
            if path is not None and path.exists():
                evidence.manifest_paths.add(path)
                evidence.existing_paths.add(path)

    for key in (
        "connected_paths",
        "imported_paths",
        "dependency_paths",
        "entrypoints",
        "route_paths",
        "service_paths",
        "stack_paths",
        "objective_matching_paths",
        "referenced_paths",
    ):
        for item in _iter_values(workspace_summary.get(key)):
            candidate = _safe_resolve(Path(str(item)).expanduser())
            if candidate is not None and candidate.exists():
                evidence.connected_paths.add(candidate)
                evidence.existing_paths.add(candidate)

    for key in ("test_commands", "build_commands", "lint_commands", "commands"):
        for item in _iter_values(workspace_summary.get(key)):
            command = _normalise_command(str(item))
            if command:
                evidence.verified_commands.add(command)

    evidence.languages.update(
        str(item).strip()
        for item in _iter_values(workspace_summary.get("detected_languages"))
        if str(item).strip()
    )
    evidence.frameworks.update(
        str(item).strip()
        for item in _iter_values(workspace_summary.get("frameworks"))
        if str(item).strip()
    )

    summary = reconnaissance_summary or ""
    for raw_path in _ABSOLUTE_PATH_RE.findall(summary):
        path = _safe_resolve(Path(raw_path.rstrip(".,;:)]}")))
        if path is None or not path.exists():
            continue
        evidence.existing_paths.add(path)
        if path.is_file() and path.name.lower() in _KNOWN_MANIFEST_NAMES:
            evidence.manifest_paths.add(path)

    current_summary_root: Path | None = None
    for line in summary.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()

        if lowered.startswith(("root:", "project_root:", "repository_root:")):
            value = stripped.split(":", 1)[1].strip()
            path = _safe_resolve(Path(value).expanduser())
            if path is not None and path.is_dir():
                current_summary_root = path
                evidence.roots.append(path)
                evidence.existing_paths.add(path)

        if lowered.startswith("manifests:"):
            value = stripped.split(":", 1)[1].strip()
            if value.lower() not in {"", "none", "none found", "unknown"}:
                for raw_item in re.split(r"[,;]", value):
                    candidate_text = raw_item.strip()
                    if not candidate_text:
                        continue
                    candidate_path = Path(candidate_text).expanduser()
                    if not candidate_path.is_absolute() and current_summary_root is not None:
                        candidate_path = current_summary_root / candidate_path
                    candidate = _safe_resolve(candidate_path)
                    if candidate is not None and candidate.exists():
                        evidence.manifest_paths.add(candidate)
                        evidence.existing_paths.add(candidate)

        if lowered.startswith((
            "connected_path:",
            "connected_paths:",
            "imported_path:",
            "imported_paths:",
            "dependency_path:",
            "dependency_paths:",
            "entrypoint:",
            "entrypoints:",
            "route_path:",
            "route_paths:",
            "service_path:",
            "service_paths:",
            "stack_path:",
            "stack_paths:",
            "objective_matching_path:",
            "objective_matching_paths:",
            "referenced_path:",
            "referenced_paths:",
        )):
            value = stripped.split(":", 1)[1].strip()
            for raw_item in re.split(r"[,;]", value):
                raw_item = raw_item.strip()
                if not raw_item:
                    continue
                candidate_path = Path(raw_item).expanduser()
                if not candidate_path.is_absolute() and current_summary_root is not None:
                    candidate_path = current_summary_root / candidate_path
                candidate = _safe_resolve(candidate_path)
                if candidate is not None and candidate.exists():
                    evidence.connected_paths.add(candidate)
                    evidence.existing_paths.add(candidate)

        if "manifest_backed_commands:" in lowered or "verified_commands:" in lowered:
            value = stripped.split(":", 1)[1].strip()
            if value.lower() not in {"", "none", "none found", "unknown"}:
                for command in _split_command_list(value):
                    normalised = _normalise_command(command)
                    if normalised:
                        evidence.verified_commands.add(normalised)

        if lowered.startswith(("languages:", "detected_languages:")):
            evidence.languages.update(_split_simple_list(stripped.split(":", 1)[1]))

        if lowered.startswith(("frameworks:", "detected_frameworks:")):
            evidence.frameworks.update(_split_simple_list(stripped.split(":", 1)[1]))

    evidence.roots = _minimal_unique_roots(evidence.roots)

    if evidence.roots:
        evidence.existing_paths = {
            path
            for path in evidence.existing_paths
            if any(_is_within(path, root) for root in evidence.roots)
        }
        evidence.manifest_paths = {
            path
            for path in evidence.manifest_paths
            if any(_is_within(path, root) for root in evidence.roots)
        }
        evidence.connected_paths = {
            path
            for path in evidence.connected_paths
            if any(_is_within(path, root) for root in evidence.roots)
        }

    return evidence


def _parse_step_candidate(
    raw_step: Any,
) -> Optional[tuple[str, list[str], Optional[str], list[str]]]:
    if isinstance(raw_step, str):
        name = raw_step.strip()
        return (name, [], None, []) if name else None

    if not isinstance(raw_step, dict):
        return None

    action = str(
        raw_step.get("action")
        or raw_step.get("name")
        or raw_step.get("step")
        or ""
    ).strip()
    # Every plan step used to render as "<action> — <purpose>.", stitching
    # the model's full "why this step is needed" sentence onto every single
    # line. Claude Code/Codex plans read as short, scannable action items --
    # this made ours consistently two to three times longer per step for no
    # added information (the model already grounds *why* in the objective/
    # evidence surfaced elsewhere; validate_plan_step below still requires
    # `action` alone to be a real, evidence-backed, specific step). `purpose`
    # is still accepted from the model (kept for schema stability/backwards
    # compatibility with any caller reading raw JSON) but no longer widens
    # the rendered name.
    name = action
    targets = [
        str(item).strip()
        for item in _iter_values(raw_step.get("targets"))
        if str(item).strip()
    ]
    command_value = raw_step.get("command")
    command = None if command_value in (None, "") else str(command_value).strip()
    declared_evidence = [
        str(item).strip()
        for item in _iter_values(raw_step.get("evidence"))
        if str(item).strip()
    ]

    return (name, targets, command, declared_evidence) if name else None


def _load_json_object(raw_content: str) -> Optional[dict[str, Any]]:
    text = (raw_content or "").strip()
    if not text:
        return None

    candidates = [text, _CODE_FENCE_RE.sub("", text).strip()]
    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        candidates.append(text[first:last + 1])

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if isinstance(data, dict):
            return data
    return None


def _normalise_command(command: str) -> str:
    text = " ".join(str(command or "").strip().split())
    if not text:
        return ""
    try:
        return shlex.join(shlex.split(text))
    except ValueError:
        return text


def _split_command_list(value: str) -> list[str]:
    if " || " in value:
        items = value.split(" || ")
    elif ";" in value:
        items = value.split(";")
    else:
        items = value.split(",")
    return [item.strip() for item in items if item.strip()]


def _split_simple_list(value: str) -> set[str]:
    return {
        item.strip()
        for item in re.split(r"[,;]", value)
        if item.strip() and item.strip().lower() not in {"none", "unknown"}
    }


def _iter_values(value: Any) -> Iterable[Any]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        return value
    return (value,)


def _safe_string_list(value: Any, limit: int) -> list[str]:
    return _dedupe_strings(
        str(item).strip()
        for item in _iter_values(value)
        if str(item).strip()
    )[:limit]


def _dedupe_strings(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _safe_resolve(path: Path) -> Optional[Path]:
    try:
        return path.resolve()
    except (OSError, RuntimeError):
        try:
            return path.absolute()
        except OSError:
            return None


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _minimal_unique_roots(roots: Sequence[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        resolved = _safe_resolve(root)
        if resolved is None or not resolved.is_dir():
            continue
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        unique.append(resolved)
    return unique
