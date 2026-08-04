"""Fail-closed workspace grants and target resolution.

The launch directory is the only implicit workspace. Additional roots are
usable when they are named explicitly by absolute path in the user objective.
The local runner persists those explicit roots in the session grant before
tool execution; inferred product or sibling names never expand scope.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![\w.-])/(?:[A-Za-z0-9._~+\-]+/)*[A-Za-z0-9._~+\-]+"
)
# Strips any "scheme://host/path" span before the absolute-path scan runs.
# Without this, a URL like "https://www.tamfitronics.com/robots.txt" in the
# objective (e.g. quoted from a prior tool result, or the user pasting a
# link) has its "//www.tamfitronics.com/robots.txt" portion matched by
# _ABSOLUTE_PATH_RE as if "/www.tamfitronics.com/robots.txt" were a real
# filesystem path -- which then fails closed as "outside the workspace
# grant" even though nothing filesystem-related was ever requested.
_URL_RE = re.compile(r"\b\w[\w+.-]*://\S+")


class WorkspaceAuthorityError(PermissionError):
    """Raised when the requested target is outside the active workspace grant.

    Carries the denied/allowed paths as structured fields (in addition to
    the fully-rendered message str() already used by str(exc)/tests) so a
    caller like render.py can lay them out as a real list instead of one
    long comma-joined sentence.
    """

    def __init__(self, message: str, *, denied: tuple[Path, ...] = (), allowed: tuple[Path, ...] = ()) -> None:
        super().__init__(message)
        self.denied_targets = denied
        self.allowed_roots = allowed


@dataclass(frozen=True)
class WorkspaceGrant:
    launch_root: Path
    allowed_roots: tuple[Path, ...]

    @classmethod
    def create(cls, launch_root: str | Path, allowed_roots: Iterable[str | Path] = ()) -> "WorkspaceGrant":
        launch = Path(launch_root).expanduser().resolve()
        roots: list[Path] = [launch]
        for raw in allowed_roots:
            candidate = Path(raw).expanduser().resolve()
            if candidate not in roots:
                roots.append(candidate)
        return cls(launch_root=launch, allowed_roots=tuple(roots))

    def contains(self, path: str | Path) -> bool:
        candidate = Path(path).expanduser().resolve()
        return any(candidate == root or root in candidate.parents for root in self.allowed_roots)


@dataclass(frozen=True)
class WorkspaceResolution:
    roots: tuple[Path, ...]
    explicit_paths: tuple[Path, ...] = field(default_factory=tuple)
    denied_targets: tuple[Path, ...] = field(default_factory=tuple)


def _project_root(path: Path) -> Path:
    candidate = path.resolve()
    start = candidate if candidate.is_dir() else candidate.parent
    for root in (start, *start.parents):
        if any(
            (root / marker).exists()
            for marker in (".git", "pyproject.toml", "package.json", "Cargo.toml", "go.mod")
        ):
            return root
    return start


def explicit_absolute_targets(objective: str) -> tuple[Path, ...]:
    targets: list[Path] = []
    seen: set[str] = set()
    scrubbed = _URL_RE.sub(" ", objective or "")
    for raw in _ABSOLUTE_PATH_RE.findall(scrubbed):
        candidate = Path(raw.rstrip(".,;:)]}")).expanduser()
        # Terminal hints, pasted transcripts, and pasted error logs commonly
        # contain absolute-looking tokens that are not filesystem paths at
        # all: slash commands (`/status`), REST/API routes copied from a
        # backend error (`/api/v1/chat/models`, `/health`), or URL segments.
        # None of these name a real target, and treating them as one used to
        # fail the whole turn closed with "outside the active workspace
        # grant" the moment a user pasted an error message containing one.
        # A token only becomes an actionable workspace target once it
        # resolves to something that actually exists on disk.
        if not candidate.exists():
            continue
        try:
            candidate = _project_root(candidate)
        except OSError:
            continue
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            targets.append(candidate)
    return tuple(targets)


def auto_grant_explicit_targets(
    *, launch_root: str | Path, objective: str, allowed_roots: Iterable[str | Path] = ()
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Add explicit, usable targets to a local session grant.

    Naming an absolute path in the current user objective is direct scope
    authorization. Only a resolved directory that already exists is added;
    inferred names, pasted slash commands, and unresolved root-level tokens
    cannot widen the grant.
    """
    grant = WorkspaceGrant.create(launch_root, allowed_roots)
    roots = list(grant.allowed_roots)
    added: list[Path] = []
    for target in explicit_absolute_targets(objective):
        if grant.contains(target) or not target.is_dir():
            continue
        roots.append(target)
        added.append(target)
        grant = WorkspaceGrant.create(launch_root, roots)
    return tuple(roots), tuple(added)


def infer_named_external_project(launch_root: Path, objective: str) -> tuple[Path, ...]:
    """Detect named sibling projects without granting access to them.

    This is diagnostic only. It catches requests such as "audit TamfisGPT"
    launched from /home/tamfisseo and forces the user to switch/add a workspace
    rather than silently inspecting the sibling.

    A project name mentioned as background context (for example, "TamfisGPT
    uses port 5173") is not a workspace target. Require a project-directed
    action before applying this diagnostic, otherwise ordinary comparisons and
    dependency notes are incorrectly rejected before the model sees them.
    """
    text = objective or ""
    lowered = re.sub(r"[^a-z0-9]+", "", text.lower())
    if not lowered or launch_root.parent == launch_root:
        return ()
    target_action_re = re.compile(
        r"\b(?:audit|inspect|review|read|edit|fix|update|modify|change|search|"
        r"check|test|build|deploy|run|open|scan|work\s+on|look\s+(?:at|inside))\b",
        re.IGNORECASE,
    )
    matches: list[Path] = []
    try:
        siblings = list(launch_root.parent.iterdir())
    except OSError:
        return ()
    launch_key = re.sub(r"[^a-z0-9]+", "", launch_root.name.lower())
    for sibling in siblings:
        if not sibling.is_dir() or sibling.resolve() == launch_root:
            continue
        key = re.sub(r"[^a-z0-9]+", "", sibling.name.lower())
        aliases = {key}
        if key == "tamfisgpt":
            aliases.update({"tamfisgptios", "tamfisaios"})
        if key == "tamfisseo":
            aliases.update({"tamfisseopro"})
        if not key or key == launch_key:
            continue
        # Only classify the named sibling as a target when an action verb
        # occurs shortly before it. This preserves the safety check for
        # "audit TamfisGPT" while allowing context such as "TamfisGPT uses
        # port 5173" to remain an answer/implementation request in the
        # current workspace.
        alias_pattern = "|".join(re.escape(alias) for alias in sorted(aliases, key=len, reverse=True))
        for alias_match in re.finditer(rf"\b(?:{alias_pattern})\b", text, re.IGNORECASE):
            preceding_text = text[max(0, alias_match.start() - 100):alias_match.start()]
            if target_action_re.search(preceding_text):
                matches.append(sibling.resolve())
                break
    return tuple(matches)


def resolve_workspace_targets(
    *, launch_root: str | Path, objective: str, allowed_roots: Iterable[str | Path] = ()
) -> WorkspaceResolution:
    grant = WorkspaceGrant.create(launch_root, allowed_roots)
    explicit = explicit_absolute_targets(objective)
    denied = tuple(path for path in explicit if not grant.contains(path))
    if denied:
        rendered = ", ".join(str(path) for path in denied)
        allowed = ", ".join(str(path) for path in grant.allowed_roots)
        raise WorkspaceAuthorityError(
            f"Requested target is outside the active workspace grant: {rendered}. "
            f"Active roots: {allowed}. Restart Tamfis-Code from the target directory or run `tamfis-code workspace add PATH` before retrying.",
            denied=denied, allowed=grant.allowed_roots,
        )

    selected: list[Path] = []
    for path in explicit:
        if grant.contains(path) and path not in selected:
            selected.append(path)

    if not selected:
        # A sibling's product/repository name is context, not a filesystem
        # target. Cross-workspace access is requested with an explicit
        # absolute path (and handled by the normal grant/approval flow above);
        # silently treating a name mention as a target rejects valid requests
        # such as port or dependency comparisons before intent classification.
        # No external files were inspected when a name is mentioned without
        # an explicit path; the current launch root remains the only scope.
        selected = [grant.launch_root]

    return WorkspaceResolution(roots=tuple(selected), explicit_paths=explicit)
