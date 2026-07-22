"""Resolves the launch directory into a Remote session, idempotently.

Implements Phase 21's "CLI entry behaviour": running `tamfis-code` from a
directory treats that directory as workspace_root, never scans the wider
box, and reuses (rather than re-creates) the local server/session pair for
repeated runs from the same directory.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .api_client import RemoteAPIClient
from . import state as local_state

REUSABLE_STATUSES = {"idle", "active"}
INSTRUCTION_NAMES = {
    "AGENTS.md", "CLAUDE.md", "CODEX.md", "TAMFIS.md", ".tamfis",
    "CONTRIBUTING.md", "README.md",
}
REPORT_RE = re.compile(
    r"(report|audit|analysis|architecture|implementation|codex|claude|tamfis[-_ ]?code|"
    r"terminal|agent|roadmap|plan|findings|gaps|status|handover|review)", re.I,
)
REPORT_SUFFIXES = {".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".log", ".pdf", ".docx"}
IGNORED_PARTS = {
    ".git", ".cache", ".config", ".local", ".tox", ".nox",
    "node_modules", "vendor", "site-packages", ".venv", "venv",
    "dist", "build", "__pycache__", "output", "uploads", "tmp",
}
DISPOSABLE_UNTRACKED_PARTS = {
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".tox", ".nox", ".coverage", "htmlcov", ".DS_Store",
}
MAX_INDEX_FILES = 20_000

MANIFEST_LANGUAGE_MAP = {
    "package.json": ("JavaScript/TypeScript", "npm"),
    "pyproject.toml": ("Python", "pip"),
    "requirements.txt": ("Python", "pip"),
    "setup.py": ("Python", "pip"),
    "setup.cfg": ("Python", "pip"),
    "Cargo.toml": ("Rust", "cargo"),
    "go.mod": ("Go", "go"),
    "pom.xml": ("Java", "maven"),
    "build.gradle": ("Java/Kotlin", "gradle"),
    "composer.json": ("PHP", "composer"),
    "Gemfile": ("Ruby", "bundler"),
}


# Cheap, name/marker-based signals only (never a content scan) for
# classifying a *candidate* workspace root discovered under a parent
# directory -- used by runner_local.py's multi-stack scoping so a stale
# backup or a generated build output isn't silently treated as an equally
# valid target alongside the real, actively-developed stacks.
# WordPress installs frequently have neither package.json NOR composer.json
# -- a plain WP site/theme/plugin checkout is just PHP files with no
# dependency manifest at all -- so MANIFEST_LANGUAGE_MAP's filename->language
# lookup alone silently detects nothing for one of the most common web
# stacks. Confirmed live: with no language/framework signal in the workspace
# facts handed to the model, it fell back to guessing Node/React conventions
# (looking for package.json) even when the user's own objective said
# "this is a WordPress site, not a React component."
_WORDPRESS_CORE_MARKERS = frozenset({
    "wp-config.php", "wp-load.php", "wp-settings.php", "wp-cron.php",
    "wp-login.php", "wp-blog-header.php", "wp-mail.php",
})
_WORDPRESS_THEME_HEADER_RE = re.compile(r"Theme Name\s*:", re.I)
_WORDPRESS_PLUGIN_HEADER_RE = re.compile(r"Plugin Name\s*:", re.I)

_PROJECT_MARKER_NAMES = frozenset(MANIFEST_LANGUAGE_MAP) | _WORDPRESS_CORE_MARKERS | {
    ".git", "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
    "compose.yml", "compose.yaml",
}
_DEPENDENCY_DIR_NAMES = {"node_modules", "vendor", ".venv", "venv", "site-packages"}
_ARCHIVED_NAME_MARKERS = ("backup", "backups", "bak", "archive", "archived")
_GENERATED_NAME_MARKERS = ("dist", "build", "generated", "_generated", ".generated", "output")
_LEGACY_NAME_MARKERS = ("legacy", "deprecated", "-old", "_old", "old-", "old_")

STACK_ROLES = ("active", "dependency", "legacy", "generated", "archived", "unrelated")


def has_project_marker(path: Path) -> bool:
    try:
        return path.is_dir() and any((path / marker).exists() for marker in _PROJECT_MARKER_NAMES)
    except OSError:
        return False


def classify_root(path: Path) -> str:
    """Classify a candidate workspace root: active/dependency/legacy/
    generated/archived/unrelated. Best-effort and local-signal-only --
    a stack that doesn't match any marker just isn't a project root at all
    ("unrelated"), it isn't evidence of anything more.
    """
    name = path.name.lower()
    if not path.is_dir():
        return "unrelated"
    if name in _DEPENDENCY_DIR_NAMES:
        return "dependency"
    if any(marker in name for marker in _ARCHIVED_NAME_MARKERS):
        return "archived"
    if any(marker in name for marker in _GENERATED_NAME_MARKERS):
        return "generated"
    if any(marker in name for marker in _LEGACY_NAME_MARKERS):
        return "legacy"
    if not has_project_marker(path):
        return "unrelated"
    return "active"


def _project_metadata(root: Path, files: list[Path]) -> dict[str, Any]:
    """Build bounded, model-facing metadata for the current project root."""
    root = Path(root).resolve()

    # Prefer top-level files when duplicate names occur deeper in the tree.
    by_name: dict[str, Path] = {}
    for path in files:
        current = by_name.get(path.name)
        if current is None:
            by_name[path.name] = path
            continue
        try:
            if len(path.relative_to(root).parts) < len(current.relative_to(root).parts):
                by_name[path.name] = path
        except ValueError:
            continue

    manifests = [
        str(by_name[name])
        for name in MANIFEST_LANGUAGE_MAP
        if name in by_name
    ]
    languages = {
        MANIFEST_LANGUAGE_MAP[name][0]
        for name in by_name
        if name in MANIFEST_LANGUAGE_MAP
    }
    package_managers = {
        MANIFEST_LANGUAGE_MAP[name][1]
        for name in by_name
        if name in MANIFEST_LANGUAGE_MAP
    }
    frameworks: set[str] = set()

    package_json = by_name.get("package.json")
    if package_json is not None:
        try:
            package_text = package_json.read_text(
                encoding="utf-8", errors="replace"
            ).lower()
        except OSError:
            package_text = ""
        for marker, label in (
            ('"react"', "React"),
            ('"vite"', "Vite"),
            ('"next"', "Next.js"),
            ('"hono"', "Hono"),
            ('"vue"', "Vue"),
            ('"@angular/core"', "Angular"),
            ('"@nestjs/core"', "NestJS"),
            ('"express"', "Express"),
        ):
            if marker in package_text:
                frameworks.add(label)

    if "manage.py" in by_name:
        frameworks.add("Django")
    if "alembic.ini" in by_name:
        frameworks.add("Alembic")

    detected = _discover_project_type(root)
    language = detected.get("language")
    framework = detected.get("framework")
    package_manager = detected.get("package_manager")
    if language and language != "unknown":
        # _discover_project_type distinguishes JavaScript vs TypeScript (via
        # tsconfig.json); MANIFEST_LANGUAGE_MAP's package.json entry above
        # only ever adds the generic "JavaScript/TypeScript". When both
        # fire for the same Node project, prefer the more specific label
        # instead of reporting both for one project (confirmed live: a
        # plain style.css + package.json project listed detected_languages
        # as both "JavaScript" and "JavaScript/TypeScript").
        if str(language) in {"JavaScript", "TypeScript"}:
            languages.discard("JavaScript/TypeScript")
        languages.add(str(language))
    if framework:
        frameworks.add(str(framework))
    if package_manager:
        package_managers.add(str(package_manager))

    # WordPress is the authoritative primary stack when core/theme/plugin
    # markers are present. A package.json inside a theme/plugin (or even at
    # the site root for asset tooling) must not relabel the site itself as a
    # JavaScript/TypeScript application or make npm its primary package
    # manager. Keep those manifests in project_manifests for evidence, but
    # expose only the runtime stack here.
    is_wordpress = "WordPress" in frameworks
    if is_wordpress:
        languages = {"PHP"}
        package_managers = {
            "composer" if (root / "composer.json").is_file() else "none"
        }

    test_commands: list[str] = []
    build_commands: list[str] = []
    if not is_wordpress and ((root / "pyproject.toml").exists() or (root / "pytest.ini").exists()):
        test_commands.append("pytest -q")
    if not is_wordpress and (root / "package.json").exists():
        test_commands.append("npm test")
        build_commands.append("npm run build")
    if not is_wordpress and (root / "Cargo.toml").exists():
        test_commands.append("cargo test")
        build_commands.append("cargo build")
    if not is_wordpress and (root / "go.mod").exists():
        test_commands.append("go test ./...")
        build_commands.append("go build ./...")
    if (root / "composer.json").exists():
        test_commands.append("composer test")

    service_files = sorted({
        str(path)
        for path in files
        if path.suffix == ".service"
        or path.name.endswith("-capacity.conf")
        or path.name in {
            "docker-compose.yml",
            "docker-compose.yaml",
            "compose.yml",
            "compose.yaml",
            "Caddyfile",
        }
    })

    important_dirs: list[str] = []
    try:
        for path in root.iterdir():
            if not path.is_dir() or path.name in IGNORED_PARTS:
                continue
            important_dirs.append(str(path))
            if len(important_dirs) >= 50:
                break
    except OSError:
        pass

    return {
        "project_manifests": manifests,
        "detected_languages": sorted(languages),
        "package_managers": sorted(package_managers),
        "frameworks": sorted(frameworks),
        "important_directories": important_dirs,
        "test_commands": test_commands,
        "build_commands": build_commands,
        "service_definitions": service_files,
        "service_endpoint_facts": _service_endpoint_facts(root, files),
    }


def _service_endpoint_facts(root: Path, files: list[Path]) -> list[dict[str, Any]]:
    """Extract bounded, source-labelled endpoint evidence from repository
    configuration. A proxy listener/upstream is deliberately not labelled an
    application port; only an application process bind is strong evidence for
    that claim. This keeps model-facing facts portable without hard-coding any
    TamfisGPT/VPS ports into the installed package."""
    root = Path(root).resolve()
    facts: list[dict[str, Any]] = []
    candidates = [
        path for path in files
        if path.suffix == ".service"
        or path.name.endswith("-capacity.conf")
        or path.name in {
            "Caddyfile", "docker-compose.yml", "docker-compose.yaml",
            "compose.yml", "compose.yaml",
        }
    ][:80]

    def add(path: Path, role: str, port: str, evidence: str) -> None:
        try:
            source = str(path.resolve().relative_to(root))
        except ValueError:
            source = str(path)
        fact = {
            "role": role, "port": int(port), "source": source,
            "evidence": evidence.strip()[:240],
        }
        if fact not in facts:
            facts.append(fact)

    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")[:64_000]
        except OSError:
            continue
        if path.name == "Caddyfile":
            for match in re.finditer(r"(?m)^\s*(?P<address>(?:https?://)?[^\s{]*:(?P<port>\d{2,5}))\s*\{", text):
                add(path, "proxy_listener", match.group("port"), match.group(0))
            for match in re.finditer(r"(?m)^\s*reverse_proxy\s+(?P<address>[^\s#]+:(?P<port>\d{2,5}))", text):
                add(path, "proxy_upstream", match.group("port"), match.group(0))
            continue

        for match in re.finditer(
            r"(?m)^\s*(?:ExecStart=.*\b(?:gunicorn|uvicorn)\b.*?--bind|"
            r"ExecStart=.*\buvicorn\b.*?--port)\s+(?P<address>[^\s]+?[:=])?(?P<port>\d{2,5})(?:\s|$)",
            text,
        ):
            add(path, "application_process_bind", match.group("port"), match.group(0))

        if path.name.endswith((".yml", ".yaml")):
            for match in re.finditer(r"(?m)^\s*-\s*[\"']?(?P<host>\d{2,5}):(?P<container>\d{2,5})[\"']?\s*$", text):
                add(path, "container_published_port", match.group("host"), match.group(0))
                add(path, "container_internal_port", match.group("container"), match.group(0))
    return facts[:100]


@dataclass
class WorkspaceContext:
    session_id: int
    workspace_root: str
    # None for a purely local session (see resolve_local_workspace) -- only
    # meaningful for the legacy Remote-backend path (resolve_workspace),
    # which this field exists to support until that path is retired.
    server_id: Optional[int] = None


def _next_local_session_id() -> int:
    known = local_state.all_known_session_ids()
    return (max(known) + 1) if known else 1


def resolve_local_workspace(cwd: Optional[Path] = None, *, discover: bool = True) -> WorkspaceContext:
    """Resolve a launch directory into a purely local session -- no network
    calls, no RemoteAPIClient, no remote-assigned session/server id.

    Reuses a prior session for the same workspace_root (matched by
    `primary_workspace`, the same durable-root concept `resolve_workspace`
    already used) rather than minting a new one on every invocation; a
    fresh id is allocated via `_next_local_session_id()` (one past the
    highest session id state.py already knows about) when no match exists.

    Swarm sub-task child sessions (is_swarm_child, see
    resolve_swarm_subtask_workspace) are deliberately excluded from this
    match -- they share the same workspace_root by design, but are not a
    session an ordinary caller should ever land back in.
    """
    workspace_root = str((cwd or Path.cwd()).resolve())

    local_match = next((
        sid for sid in reversed(local_state.all_known_session_ids())
        if local_state.get_session_state(sid).primary_workspace == workspace_root
        and not local_state.get_session_state(sid).is_swarm_child
    ), None)
    session_id = local_match if local_match is not None else _next_local_session_id()

    local_state.save_session_state(session_id, workspace_root=workspace_root)
    if discover:
        discover_local_repository(session_id, Path(workspace_root))
    return WorkspaceContext(session_id=session_id, workspace_root=workspace_root)


def resolve_swarm_subtask_workspace(
    workspace_root: Path, *, parent_session_id: Optional[int] = None, label: str = "",
) -> WorkspaceContext:
    """Like resolve_local_workspace, but for a concurrent swarm sub-task:
    always mints a fresh session id instead of reusing the same-
    workspace_root match resolve_local_workspace intentionally does.

    Concurrent sub-tasks sharing one session_id would race on state.json's
    single-value fields (current_phase/running_action/active_task/...) --
    only queued_user_instructions/saved_plans are merge-safe there. Giving
    each concurrent sub-task its own child session (same real filesystem
    workspace_root, distinct state.json row, tagged is_swarm_child=True so
    it can be filtered out of default session listings) sidesteps that
    without adding locking to state.py's persistence layer, which every
    other command in this codebase also depends on.

    parent_session_id is best-effort context (None when there's no
    pre-existing session to record -- e.g. `agent-cmd delegate` is a
    one-shot CLI invocation with no "current session" at all) -- it is
    NOT what marks this as a swarm child for hide/show purposes; that's
    is_swarm_child, always True here regardless of whether a real parent
    was known. A live-caught bug briefly used parent_session_id is not
    None as that marker, which silently failed to hide any child session
    minted with no real parent (confirmed via agent-cmd delegate).
    """
    resolved_root = str(Path(workspace_root).resolve())
    session_id = _next_local_session_id()
    local_state.save_session_state(
        session_id, workspace_root=resolved_root,
        parent_session_id=parent_session_id, is_swarm_child=True, swarm_label=label,
    )
    return WorkspaceContext(session_id=session_id, workspace_root=resolved_root)


async def _get_or_create_local_server(client: RemoteAPIClient) -> dict:
    servers = await client.list_servers()
    for server in servers:
        if server.get("transport_type") == "local":
            return server
    created = await client.register_local_server("local-vps")
    return created["server"]


async def resolve_workspace(
    client: RemoteAPIClient, cwd: Optional[Path] = None, *, discover: bool = True,
) -> WorkspaceContext:
    workspace_root = str((cwd or Path.cwd()).resolve())

    server = await _get_or_create_local_server(client)
    server_id = server["id"]

    # A session can move to an explicitly approved sibling workspace. Reuse
    # it by its durable primary root rather than creating a new conversation
    # just because its current working directory changed.
    local_match = next((
        sid for sid in reversed(local_state.all_known_session_ids())
        if local_state.get_session_state(sid).primary_workspace == workspace_root
    ), None)
    if local_match is not None:
        try:
            detail = await client.get_session(local_match)
        except Exception:
            detail = None
        if detail and str(detail.get("status", "")).lower() in REUSABLE_STATUSES:
            state = local_state.get_session_state(local_match)
            current = str(detail.get("working_directory") or state.current_working_directory or workspace_root)
            local_state.save_session_state(
                local_match, workspace_root=workspace_root,
                current_working_directory=current,
            )
            if discover:
                discover_local_repository(local_match, Path(current))
            return WorkspaceContext(session_id=local_match, server_id=server_id, workspace_root=current)

    sessions = await client.list_sessions()
    existing = next(
        (
            s for s in sessions
            if s.get("server_id") == server_id
            and s.get("working_directory") == workspace_root
            and str(s.get("status", "")).lower() in REUSABLE_STATUSES
        ),
        None,
    )
    if existing is not None:
        local_state.save_session_state(existing["id"], workspace_root=workspace_root)
        if discover:
            discover_local_repository(existing["id"], Path(workspace_root))
        return WorkspaceContext(session_id=existing["id"], server_id=server_id, workspace_root=workspace_root)

    created = await client.create_session(server_id, workspace_root)
    local_state.save_session_state(created["id"], workspace_root=workspace_root)
    if discover:
        discover_local_repository(created["id"], Path(workspace_root))
    return WorkspaceContext(session_id=created["id"], server_id=server_id, workspace_root=workspace_root)


async def context_from_session(client: RemoteAPIClient, session_id: int) -> WorkspaceContext:
    """Builds a WorkspaceContext from an existing session_id -- used by
    `/resume` and `tamfis-code resume`. Raises RemoteAPIError (404) if the
    session does not exist or is not owned by the authenticated user, same
    ownership check every other Remote endpoint already enforces."""
    detail = await client.get_session(session_id)
    workspace_root = detail.get("working_directory") or ""
    local_state.save_session_state(session_id, workspace_root=workspace_root)
    if workspace_root:
        discover_local_repository(session_id, Path(workspace_root))
    return WorkspaceContext(
        session_id=detail["id"],
        server_id=detail["server_id"],
        workspace_root=workspace_root,
    )


async def find_resumable_session(client: RemoteAPIClient, *, exclude_session_id: Optional[int] = None) -> Optional[dict]:
    """Most recent non-closed session other than the one already active --
    `list_sessions()` is ordered by created_at desc server-side."""
    sessions = await client.list_sessions()
    for session in sessions:
        if exclude_session_id is not None and session.get("id") == exclude_session_id:
            continue
        if str(session.get("status", "")).lower() in REUSABLE_STATUSES:
            return session
    return None


def _git(root: Path, *args: str) -> str:
    """Run a bounded, argument-safe Git query (never through a shell)."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True,
            timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _indexable_files(root: Path) -> list[Path]:
    def allowed(path: Path) -> bool:
        try:
            parts = path.relative_to(root).parts
        except ValueError:
            return False
        return not any(
            part in IGNORED_PARTS
            or (part.startswith(".") and part not in {".github", ".tamfis"})
            for part in parts
        )

    output = _git(root, "ls-files", "-co", "--exclude-standard")
    if output:
        result: list[Path] = []
        seen: set[Path] = set()

        def append_file(path: Path) -> bool:
            if not allowed(path) or not path.is_file() or path in seen:
                return False
            seen.add(path)
            result.append(path)
            return len(result) >= MAX_INDEX_FILES

        for item in output.splitlines():
            path = root / item
            if not allowed(path):
                continue
            if path.is_dir():
                # A parent Git repository reports a nested repository as one
                # directory entry (for example `backend/`) rather than listing
                # its files. Expand only that named directory, applying the
                # exact same cache/dependency/hidden filters and global bound.
                # Otherwise model context sees the project name but none of
                # its manifests, instructions, service units, or source.
                try:
                    nested_paths = path.rglob("*")
                    for nested in nested_paths:
                        if append_file(nested):
                            return result
                except OSError:
                    continue
            elif append_file(path):
                break
        return result
    found: list[Path] = []
    for path in root.rglob("*"):
        if not allowed(path):
            continue
        if path.is_file():
            found.append(path)
            if len(found) >= MAX_INDEX_FILES:
                break
    return found


def blocking_dirty_files(status_lines: list[str]) -> list[str]:
    """Return entries that may represent user-authored work.

    Tracked changes always block execute mode. Only untracked, well-known
    disposable test/interpreter artefacts are ignored.
    """
    blocking: list[str] = []
    for raw in status_lines:
        line = str(raw).rstrip()
        if not line:
            continue
        status = line[:2]
        path_text = line[3:].strip() if len(line) > 3 else ""
        parts = {part for part in Path(path_text).parts if part not in {".", ""}}
        if status == "??" and parts.intersection(DISPOSABLE_UNTRACKED_PARTS):
            continue
        blocking.append(line)
    return blocking


def _report_title(path: Path) -> str:
    if path.suffix.lower() in {".md", ".txt"}:
        try:
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[:40]:
                if line.lstrip().startswith("#"):
                    return line.lstrip("# ").strip()
        except OSError:
            pass
    return path.stem.replace("_", " ").replace("-", " ")





def _discover_project_type(workspace_root: Path) -> dict[str, Any]:
    """Detect the primary project type from bounded, local filesystem signals.

    The function deliberately inspects only the root and a few conventional
    source directories. It never recursively scans the wider host.
    """
    root = Path(workspace_root).expanduser().resolve()

    def result(
        language: str,
        *,
        framework: Optional[str] = None,
        package_manager: Optional[str] = None,
    ) -> dict[str, Any]:
        value: dict[str, Any] = {
            "language": language,
            "package_manager": package_manager,
        }
        if framework:
            value["framework"] = framework
        return value

    def has_any(*names: str) -> bool:
        return any((root / name).exists() for name in names)

    # WordPress must be checked before generic PHP/Node detection. A site may
    # contain package.json in a theme while WordPress remains the primary stack.
    wordpress_core = (
        (root / "wp-content").is_dir()
        and ((root / "wp-admin").is_dir() or (root / "wp-includes").is_dir())
    ) or any((root / marker).is_file() for marker in _WORDPRESS_CORE_MARKERS)

    wordpress_theme = False
    style_css = root / "style.css"
    if not wordpress_core and style_css.is_file():
        try:
            wordpress_theme = bool(
                _WORDPRESS_THEME_HEADER_RE.search(
                    style_css.read_text(encoding="utf-8", errors="replace")[:4000]
                )
            )
        except OSError:
            wordpress_theme = False

    wordpress_plugin = False
    if not wordpress_core and not wordpress_theme:
        try:
            candidates = [
                path for path in root.iterdir()
                if path.is_file() and path.suffix.lower() == ".php"
            ][:100]
        except OSError:
            candidates = []
        for candidate in candidates:
            try:
                header = candidate.read_text(
                    encoding="utf-8", errors="replace"
                )[:4000]
            except OSError:
                continue
            if _WORDPRESS_PLUGIN_HEADER_RE.search(header):
                wordpress_plugin = True
                break

    if wordpress_core or wordpress_theme or wordpress_plugin:
        return result(
            "PHP",
            framework="WordPress",
            package_manager="composer" if (root / "composer.json").is_file() else None,
        )

    # Python frameworks and projects.
    if (root / "manage.py").is_file():
        return result("Python", framework="Django", package_manager="pip")

    if has_any("pyproject.toml", "requirements.txt", "setup.py", "setup.cfg", "Pipfile", "poetry.lock"):
        framework: Optional[str] = None
        likely_files = [root / "main.py", root / "app.py"]
        for directory in (root, root / "src", root / "app"):
            if directory.is_dir():
                try:
                    likely_files.extend(list(directory.glob("*.py"))[:50])
                except OSError:
                    pass
        for path in likely_files:
            if not path.is_file():
                continue
            try:
                sample = path.read_text(encoding="utf-8", errors="replace")[:12000]
            except OSError:
                continue
            if "FastAPI(" in sample or "from fastapi" in sample:
                framework = "FastAPI"
                break
            if "from flask" in sample or "Flask(" in sample:
                framework = "Flask"
                break
        return result("Python", framework=framework, package_manager="pip")

    try:
        root_python = any(root.glob("*.py"))
    except OSError:
        root_python = False
    if root_python:
        return result("Python", package_manager="pip")

    # Node.js / JavaScript / TypeScript.
    package_json = root / "package.json"
    if package_json.is_file():
        import json

        dependencies: dict[str, Any] = {}
        package_manager = "npm"
        if (root / "pnpm-lock.yaml").is_file():
            package_manager = "pnpm"
        elif (root / "yarn.lock").is_file():
            package_manager = "yarn"
        try:
            payload = json.loads(package_json.read_text(encoding="utf-8"))
            dependencies = {
                **(payload.get("dependencies") or {}),
                **(payload.get("devDependencies") or {}),
            }
        except (OSError, ValueError, TypeError):
            dependencies = {}

        framework = None
        for key, label in (
            ("next", "Next.js"),
            ("react", "React"),
            ("vue", "Vue"),
            ("@angular/core", "Angular"),
            ("@nestjs/core", "NestJS"),
            ("hono", "Hono"),
            ("express", "Express"),
        ):
            if key in dependencies:
                framework = label
                break
        language = "TypeScript" if (root / "tsconfig.json").is_file() else "JavaScript"
        return result(language, framework=framework, package_manager=package_manager)

    # Other manifest-driven stacks.
    checks = (
        (("composer.json",), "PHP", None, "composer"),
        (("Cargo.toml", "Cargo.lock"), "Rust", None, "cargo"),
        (("go.mod", "go.sum"), "Go", None, "go"),
        (("pom.xml",), "Java", None, "maven"),
        (("build.gradle.kts",), "Kotlin", None, "gradle"),
        (("build.gradle",), "Java/Kotlin", None, "gradle"),
        (("Gemfile", "Rakefile"), "Ruby", None, "bundler"),
        (("Package.swift",), "Swift", None, "swift"),
        (("pubspec.yaml", "pubspec.yml"), "Dart", "Flutter", "pub"),
        (("mix.exs",), "Elixir", None, "mix"),
        (("project.clj",), "Clojure", None, "lein"),
        (("deps.edn",), "Clojure", None, "clojure"),
        (("stack.yaml",), "Haskell", None, "stack"),
        (("cabal.project",), "Haskell", None, "cabal"),
        (("build.sbt",), "Scala", None, "sbt"),
        (("cpanfile",), "Perl", None, "cpanm"),
        (("Makefile.PL",), "Perl", None, "perl"),
    )
    for names, language, framework, manager in checks:
        if has_any(*names):
            return result(
                language,
                framework=framework,
                package_manager=manager,
            )

    try:
        if any(root.glob("*.csproj")):
            return result("C#", framework=".NET", package_manager="dotnet")
        if any(root.glob("*.fsproj")):
            return result("F#", framework=".NET", package_manager="dotnet")
        if any(root.glob("*.rockspec")):
            return result("Lua", package_manager="luarocks")
        if any(root.glob("*.php")):
            return result("PHP", package_manager=None)
    except OSError:
        pass

    return result("unknown", package_manager=None)


def discover_local_repository(session_id: int, workspace_root: Path, *, force: bool = False) -> dict[str, Any]:
    """Cache local CLI context until Git HEAD or dirty-file count changes.

    The model-facing server snapshot remains authoritative. This lightweight
    index powers offline `/context` and `/reports`, and records the worktree
    state that existed before a task begins.
    """
    root_text = _git(workspace_root, "rev-parse", "--show-toplevel")
    root = Path(root_text).resolve() if root_text else workspace_root.resolve()
    branch = _git(root, "branch", "--show-current") or None
    head = _git(root, "rev-parse", "HEAD") or "no-head"
    dirty_lines = _git(root, "status", "--short").splitlines()
    fingerprint_inputs = [str(root), head, *dirty_lines]
    for name in sorted(set(MANIFEST_LANGUAGE_MAP) | INSTRUCTION_NAMES):
        path = root / name
        if path.is_file():
            try:
                fingerprint_inputs.append(f"{name}:{path.stat().st_mtime_ns}:{path.stat().st_size}")
            except OSError:
                pass
    fingerprint = hashlib.sha256("|".join(fingerprint_inputs).encode()).hexdigest()
    current = local_state.get_session_state(session_id)
    if not force and current.discovery_fingerprint == fingerprint and current.repository_context:
        return current.repository_context

    files = _indexable_files(root)
    instructions: list[str] = []
    reports: list[dict[str, Any]] = []
    for path in files:
        if path.name in INSTRUCTION_NAMES:
            instructions.append(str(path))
        if path.suffix.lower() in REPORT_SUFFIXES and REPORT_RE.search(path.name):
            try:
                modified = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
            except OSError:
                modified = ""
            reports.append({
                "path": str(path), "title": _report_title(path), "modified_at": modified,
                "scope": "repository", "verification": "unverified",
            })

    context = {
        "repository_root": str(root), "working_directory": str(workspace_root.resolve()),
        "branch": branch, "head": None if head == "no-head" else head,
        "dirty": bool(dirty_lines), "dirty_files": dirty_lines[:100],
        "blocking_dirty_files": blocking_dirty_files(dirty_lines)[:100],
        "instruction_files": sorted(instructions), "indexed_file_count": len(files),
        "indexed_at": datetime.now(timezone.utc).isoformat(),
        **_project_metadata(root, files),
    }
    local_state.save_session_state(
        session_id, repository_root=str(root), current_working_directory=str(workspace_root.resolve()),
        active_branch=branch, repository_context=context,
        discovered_reports=sorted(reports, key=lambda item: item["modified_at"], reverse=True),
        discovery_fingerprint=fingerprint,
    )
    return context


# Total instruction-file content budget for the system prompt -- generous
# enough for a real AGENTS.md/CLAUDE.md, bounded so a huge README doesn't
# blow the context window before the actual objective is even sent.
MAX_INSTRUCTION_CHARS = 12_000


def build_system_prompt(session_id: int, workspace_root: Path, *, force_discovery: bool = False) -> str:
    """Build the standalone agent loop's system prompt from real local
    workspace awareness (git branch/dirty status, instruction file
    contents) -- replacing the context tamgpt6 used to assemble server-side.
    Reuses discover_local_repository's existing fingerprint-cached scan
    rather than re-walking the repo on every call.
    """
    context = discover_local_repository(session_id, workspace_root, force=force_discovery)
    lines = [
        "You are a coding agent working directly in a real local repository via tool calls. "
        "Verify with tools before claiming something is done or correct. Prefer minimal, "
        "targeted changes over broad rewrites.",
        "Never describe a fix, edit, or command in your written response without actually "
        "calling the corresponding tool (write_file/edit_file/execute_command/etc) in this "
        "same turn. A code block in your text is not a change -- if the task requires "
        "changing a file, call the tool that changes it before you say you've changed it. "
        "If you're unsure which file actually defines something, use read_file or "
        "search_code to find it first; do not guess a file's contents from its name.",
        "When calling write_file to create a new source file, the path's extension must "
        "match the real language of the content you're writing (.py, .js, .ts, .go, .php, "
        ".css, etc.) -- never fall back to a generic '.txt' (or any other wrong extension) "
        "for code, even as a placeholder you intend to rename later. Match whatever filename "
        "or extension the user's request itself specifies; otherwise use the extension "
        "the target language and the rest of the project actually use.",
        "Never call list_directory (or any other read-only tool) again with the exact same "
        "arguments you already used earlier in this same task -- you already have that "
        "result; re-issuing it is not progress and will end the task early as a stuck loop. "
        "For a broad request (e.g. \"audit the entire system for vulnerabilities\") that "
        "doesn't name a specific file or directory: list_directory the top level ONCE, then "
        "immediately act on what it actually returned -- read_file a specific file it "
        "listed, list_directory a specific subdirectory it named, or search_code for a "
        "concrete pattern -- rather than re-listing the same path while you decide what to "
        "do. If the request is too broad to make that concrete next choice at all, say so "
        "and ask the user to narrow it (a specific component, directory, or concern) instead "
        "of stalling on repeated top-level listings.",
        "Before checking whether any local service is 'healthy' or 'running', you must "
        "first find its REAL configured port -- do not use 8080/3000/5000/8000 or any "
        "other common default unless you have actually confirmed that's the real one. "
        "Concrete required steps, in order: (1) search_code for \"port\" (or read "
        "config.yaml/.env/docker-compose.yml/package.json, whichever exists) to find the "
        "actual configured port; (2) only then curl/request that exact port. Getting ANY "
        "HTTP response back from a guessed port is NOT evidence the intended service is "
        "healthy -- an entirely different, unrelated process can easily be listening on a "
        "common default port instead. A Caddy/Nginx/Apache listener or reverse_proxy upstream "
        "is proxy topology, not proof of an application's own process bind or service identity; "
        "never relabel a proxy port as the application port. Prefer a service unit's ExecStart "
        "bind, a container's explicit internal/published port mapping, or a live process command, "
        "and cite that exact evidence in the answer. The same applies to any other environment-specific "
        "value (host, container/process ID, file path, env var): find the real one via a "
        "tool call before using it, never assume it from a common default.",
        "Before running any install/build/start command against a project (or one component "
        "of a multi-component stack), first find out what kind of project it actually is -- "
        "list_directory it and look for package.json (Node/npm), pyproject.toml/"
        "requirements.txt (Python), go.mod (Go), Cargo.toml (Rust), composer.json (PHP), "
        "wp-config.php/wp-load.php/a wp-content directory (WordPress -- often has NO "
        "package.json or composer.json at all; do not assume Node just because it's a web "
        "project), Dockerfile/docker-compose.yml, etc. Do not default to npm install/npm "
        "start (or assume Node/React at all) just because a component is called a "
        "'backend'/'site'/'package' or is mentioned alongside a Node frontend -- and never "
        "override an explicit statement in the user's own objective about what kind of "
        "project this is (e.g. 'this is a WordPress site, not a React component') with your "
        "own guess; run the command/inspection that actually matches what's really there.",
        f"Workspace root: {context['working_directory']}",
    ]
    # discover_local_repository always sets repository_root (falling back to
    # workspace_root itself when `git rev-parse --show-toplevel` fails) --
    # `head` is the real "is this actually a Git repo" signal, since it's
    # only None when that git call failed (see its `"no-head"` sentinel).
    if context.get("head"):
        lines.append(f"Git repository root: {context['repository_root']}  branch: {context.get('branch') or '(detached HEAD)'}")
        if context.get("dirty"):
            lines.append(
                f"The working tree already has {len(context.get('dirty_files') or [])} uncommitted change(s) -- "
                "be careful not to conflate your own edits with pre-existing ones when reporting what changed."
            )
    else:
        lines.append("This directory is not a Git repository.")

    endpoint_facts = context.get("service_endpoint_facts") or []
    if endpoint_facts:
        rendered = []
        for fact in endpoint_facts[:30]:
            rendered.append(
                f"- {fact.get('role')} port {fact.get('port')} from "
                f"{fact.get('source')}: {fact.get('evidence')}"
            )
        lines.append(
            "Repository service-endpoint evidence (configuration evidence only; live status still "
            "requires a matching process/health check). Roles are not interchangeable:\n"
            + "\n".join(rendered)
        )

    instruction_files = context.get("instruction_files") or []
    remaining_budget = MAX_INSTRUCTION_CHARS
    instruction_blocks = []
    for path_str in instruction_files:
        if remaining_budget <= 0:
            break
        try:
            content = Path(path_str).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        snippet = content[:remaining_budget]
        remaining_budget -= len(snippet)
        truncated = " (truncated)" if len(snippet) < len(content) else ""
        instruction_blocks.append(f"--- {path_str}{truncated} ---\n{snippet}")
    if instruction_blocks:
        lines.append("\nProject instructions found in this repository:\n" + "\n\n".join(instruction_blocks))

    return "\n".join(lines)
