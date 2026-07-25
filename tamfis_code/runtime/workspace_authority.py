"""Fail-closed workspace grants and target resolution.

The launch directory is the only implicit workspace. Additional roots are
usable only when they are both named explicitly by absolute path in the user
objective and already present in the durable session grant list. Product names
or sibling directory names never expand scope by themselves.
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
    return candidate.parent if candidate.is_file() else candidate


def explicit_absolute_targets(objective: str) -> tuple[Path, ...]:
    targets: list[Path] = []
    seen: set[str] = set()
    scrubbed = _URL_RE.sub(" ", objective or "")
    for raw in _ABSOLUTE_PATH_RE.findall(scrubbed):
        candidate = Path(raw.rstrip(".,;:)]}")).expanduser()
        try:
            candidate = _project_root(candidate)
        except OSError:
            continue
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            targets.append(candidate)
    return tuple(targets)


def infer_named_external_project(launch_root: Path, objective: str) -> tuple[Path, ...]:
    """Detect named sibling projects without granting access to them.

    This is diagnostic only. It catches requests such as "audit TamfisGPT"
    launched from /home/tamfisseo and forces the user to switch/add a workspace
    rather than silently inspecting the sibling.
    """
    lowered = re.sub(r"[^a-z0-9]+", "", (objective or "").lower())
    if not lowered or launch_root.parent == launch_root:
        return ()
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
        if key and key != launch_key and any(alias in lowered for alias in aliases):
            matches.append(sibling.resolve())
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
        named_external = infer_named_external_project(grant.launch_root, objective)
        if named_external:
            rendered = ", ".join(str(path) for path in named_external)
            raise WorkspaceAuthorityError(
                f"The request appears to target {rendered}, which is outside the current workspace "
                f"{grant.launch_root}. No external files were inspected. Restart Tamfis-Code from that "
                "project, or run `tamfis-code workspace add PATH` and retry with the explicit absolute path.",
                denied=named_external, allowed=grant.allowed_roots,
            )
        selected = [grant.launch_root]

    return WorkspaceResolution(roots=tuple(selected), explicit_paths=explicit)
