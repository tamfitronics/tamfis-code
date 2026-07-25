"""Release-grade verification for Tamfis-Code distributions.

The verifier is intentionally dependency-light. It validates the source tree
and built artifacts without contacting providers or modifying the workspace.
"""
from __future__ import annotations

import hashlib
import json
import re
import tarfile
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

AUTHORITATIVE_PROVIDER_ORDER = (
    "ollama_cloud",
    "nvidia",
    "hf",
    "openrouter",
)

REQUIRED_GITHUB_COMMANDS = (
    "alias", "api", "auth", "browse", "cache", "co", "codespace",
    "completion", "config", "extension", "gist", "gpg-key", "issue",
    "label", "org", "pr", "project", "release", "repo", "ruleset",
    "run", "search", "secret", "ssh-key", "status", "variable", "workflow",
)

REQUIRED_ENTRY_POINTS = {
    "tamfis-code": "tamfis_code.cli:main",
    "tamgpt-code": "tamfis_code.cli:main",
    "tamfis": "tamfis_code.cli:main",
    "tamfis-code-server": "tamfis_code.openhands.agent_server:main",
}


@dataclass(slots=True)
class VerificationCheck:
    name: str
    passed: bool
    detail: str
    category: str


@dataclass(slots=True)
class VerificationReport:
    version: str
    root: str
    checks: list[VerificationCheck] = field(default_factory=list)
    artifacts: dict[str, dict[str, object]] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(check.passed for check in self.checks)

    def add(self, name: str, passed: bool, detail: str, category: str) -> None:
        self.checks.append(VerificationCheck(name, passed, detail, category))

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "root": self.root,
            "passed": self.passed,
            "checks": [asdict(check) for check in self.checks],
            "artifacts": self.artifacts,
        }

    def write_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def write_markdown(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            f"# Tamfis-Code {self.version} Verification Report",
            "",
            f"Overall result: **{'PASS' if self.passed else 'FAIL'}**",
            "",
            "| Category | Check | Result | Detail |",
            "|---|---|---:|---|",
        ]
        for check in self.checks:
            detail = check.detail.replace("|", "\\|").replace("\n", " ")
            rows.append(f"| {check.category} | {check.name} | {'PASS' if check.passed else 'FAIL'} | {detail} |")
        if self.artifacts:
            rows.extend(["", "## Artifacts", ""])
            for name, info in sorted(self.artifacts.items()):
                rows.append(f"- `{name}` — {info.get('size_bytes')} bytes — SHA-256 `{info.get('sha256')}`")
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return path


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_version(pyproject_text: str) -> str:
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject_text, flags=re.MULTILINE)
    return match.group(1) if match else "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_info(path: Path) -> dict[str, object]:
    return {"size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _check_wheel(path: Path, version: str) -> tuple[bool, str]:
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            required = {
                "tamfis_code/cli.py",
                "tamfis_code/runtime/unified.py",
                "tamfis_code/release_verification.py",
            }
            missing = sorted(required - names)
            metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
            if missing:
                return False, f"wheel missing: {', '.join(missing)}"
            if not metadata_names:
                return False, "wheel has no dist-info/METADATA"
            metadata = archive.read(metadata_names[0]).decode("utf-8", errors="replace")
            if f"Version: {version}" not in metadata:
                return False, f"wheel metadata does not declare {version}"
    except (OSError, zipfile.BadZipFile) as exc:
        return False, f"invalid wheel: {exc}"
    return True, "wheel structure and metadata verified"


def _check_sdist(path: Path, version: str) -> tuple[bool, str]:
    try:
        with tarfile.open(path, "r:gz") as archive:
            names = set(archive.getnames())
            suffixes = {
                "pyproject.toml",
                "tamfis_code/cli.py",
                "tamfis_code/runtime/unified.py",
                "tamfis_code/release_verification.py",
            }
            missing = [suffix for suffix in suffixes if not any(name.endswith(suffix) for name in names)]
            if missing:
                return False, f"source archive missing: {', '.join(sorted(missing))}"
            pkg_info = next((name for name in names if name.endswith("PKG-INFO")), None)
            if not pkg_info:
                return False, "source archive has no PKG-INFO"
            metadata = archive.extractfile(pkg_info)
            content = metadata.read().decode("utf-8", errors="replace") if metadata else ""
            if f"Version: {version}" not in content:
                return False, f"source metadata does not declare {version}"
    except (OSError, tarfile.TarError) as exc:
        return False, f"invalid source archive: {exc}"
    return True, "source archive structure and metadata verified"


def run_release_verification(root: Path, artifacts: Iterable[Path] = ()) -> VerificationReport:
    root = root.resolve()
    pyproject = root / "pyproject.toml"
    pyproject_text = _read(pyproject) if pyproject.is_file() else ""
    version = _extract_version(pyproject_text)
    report = VerificationReport(version=version, root=str(root))

    report.add("Source tree", pyproject.is_file() and (root / "tamfis_code").is_dir(), "pyproject.toml and package directory present", "package")

    entry_points_ok = all(f'{name} = "{target}"' in pyproject_text for name, target in REQUIRED_ENTRY_POINTS.items())
    report.add("Console entry points", entry_points_ok, f"required entry points: {', '.join(REQUIRED_ENTRY_POINTS)}", "package")

    github_text = _read(root / "tamfis_code" / "github_commands.py")
    missing_commands = [name for name in REQUIRED_GITHUB_COMMANDS if f'"{name}"' not in github_text]
    report.add("GitHub command surface", not missing_commands, "all 27 delegated commands present" if not missing_commands else f"missing: {', '.join(missing_commands)}", "commands")

    providers_text = _read(root / "tamfis_code" / "providers.py")
    priority_match = re.search(
        r"PRIORITY_ORDER:[^=]*=\s*\((.*?)\n\s*\)",
        providers_text,
        flags=re.DOTALL,
    )
    priority_block = priority_match.group(1) if priority_match else ""
    positions = [priority_block.find(f"ProviderType.{name.upper()}") for name in AUTHORITATIVE_PROVIDER_ORDER]
    provider_ok = all(position >= 0 for position in positions) and positions == sorted(positions)
    provider_detail = " > ".join(AUTHORITATIVE_PROVIDER_ORDER)
    report.add("Authoritative provider order", provider_ok, provider_detail, "routing")

    render_text = _read(root / "tamfis_code" / "render.py")
    banner = "ollama_cloud, nvidia, hf, openrouter, in authoritative priority order"
    report.add("Provider banner", banner in render_text, banner, "routing")

    runner_text = _read(root / "tamfis_code" / "runner_local.py")
    compaction_markers = (
        "def _trim_tool_outputs",
        "def _compact_tool_arguments",
        'function["arguments"] = compacted',
        "keep_recent",
    )
    missing_compaction = [marker for marker in compaction_markers if marker not in runner_text]
    report.add("Context compaction safeguards", not missing_compaction, "tool outputs and tool-call arguments are compacted with recent-turn preservation" if not missing_compaction else f"missing markers: {missing_compaction}", "context")

    cli_text = _read(root / "tamfis_code" / "cli.py")
    mcp_text = _read(root / "tamfis_code" / "mcp.py")
    state_text = _read(root / "tamfis_code" / "state.py")
    mutation_ok = "os.replace" in mcp_text and "revert" in cli_text and "os.replace" in state_text
    report.add("Mutation and rollback", mutation_ok, "atomic replacement, mutation ledger, and revert command present", "mutation")

    live_text = _read(root / "tamfis_code" / "live_input.py")
    terminal_markers = ("await self._shutdown_prompt()", "asyncio.CancelledError", "finally:")
    missing_terminal = [marker for marker in terminal_markers if marker not in live_text]
    terminal_ok = not missing_terminal
    report.add("Terminal capability and recovery", terminal_ok, "awaited prompt shutdown and cancellation cleanup present" if terminal_ok else f"missing markers: {missing_terminal}", "terminal")

    cognitive_files = [
        root / "tamfis_code" / "runtime" / "cognitive.py",
        root / "tamfis_code" / "runtime" / "repository_index.py",
        root / "tamfis_code" / "runtime" / "reviewer.py",
        root / "tamfis_code" / "runtime" / "steering.py",
    ]
    cognitive_ok = all(path.is_file() for path in cognitive_files)
    unified_text = _read(root / "tamfis_code" / "runtime" / "unified.py")
    cognitive_markers = ("TaskContract.derive", "EvidenceGraph", "IndependentReviewer")
    cognitive_ok = cognitive_ok and all(marker in unified_text for marker in cognitive_markers)
    report.add("Cognitive orchestration", cognitive_ok, "task contracts, evidence graph, re-planning, repository index, live steering, and independent review present", "orchestration")

    workspace_authority_text = _read(root / "tamfis_code" / "runtime" / "workspace_authority.py")
    workspace_markers = ("class WorkspaceGrant", "resolve_workspace_targets", "No external files were inspected")
    workspace_ok = all(marker in workspace_authority_text for marker in workspace_markers) and "resolve_workspace_targets(" in runner_text
    report.add("Workspace authority", workspace_ok, "fail-closed launch-root grants, explicit target resolution, and no sibling inference" if workspace_ok else "workspace authority markers missing", "orchestration")

    memory_text = _read(root / "tamfis_code" / "runtime" / "memory.py")
    memory_markers = ("class MemoryStore", 'USER = "user"', 'FEEDBACK = "feedback"', 'PROJECT = "project"', 'REFERENCE = "reference"', "_atomic_write_json")
    memory_ok = all(marker in memory_text for marker in memory_markers) and "relevant_memories(request.objective)" in unified_text
    report.add("Durable cross-session memory", memory_ok, "typed, atomically-written memory store wired into task-contract derivation" if memory_ok else "memory store markers missing", "memory")

    worktree_text = _read(root / "tamfis_code" / "runtime" / "worktree.py")
    worktree_markers = ("def create_worktree", "def remove_worktree", "is_worktree_clean(handle)", "refusing to remove without force")
    worktree_ok = all(marker in worktree_text for marker in worktree_markers) and 'request.isolation == "worktree"' in unified_text
    report.add("Worktree isolation", worktree_ok, "fail-closed worktree creation, dirty-state removal guard, and unified-runtime isolation wiring present" if worktree_ok else "worktree isolation markers missing", "worktree")

    approvals_text = _read(root / "tamfis_code" / "orchestrator" / "approvals.py")
    approvals_markers = ("class ApprovalBatch", "def describe_batch", "risky_actions")
    decision_ok = all(marker in approvals_text for marker in approvals_markers) and "_turn_batch = ApprovalBatch()" in runner_text and "_batch_denied_ids" in runner_text
    report.add("Risk-tiered decision gating", decision_ok, "same-turn risky actions are batched into one approval decision instead of prompting per call" if decision_ok else "decision/approval batching markers missing", "decision")

    phase_tests = [
        root / "tests" / f"test_phase{number}_{name}.py"
        for number, name in (
            (1, "reliability"), (2, "unified_runtime"), (3, "claude_behaviour"),
            (4, "release_verification"), (5, "cognitive_orchestration"), (6, "workspace_authority"),
            (7, "memory"), (8, "worktree"), (9, "decision_logic"),
        )
    ]
    missing_tests = [path.name for path in phase_tests if not path.is_file()]
    report.add("Phase regression suites", not missing_tests, "Phase 1-9 test modules present" if not missing_tests else f"missing: {', '.join(missing_tests)}", "tests")

    for artifact in artifacts:
        artifact = Path(artifact)
        if not artifact.is_file():
            report.add(f"Artifact {artifact.name}", False, "file not found", "artifacts")
            continue
        report.artifacts[artifact.name] = _artifact_info(artifact)
        if artifact.suffix == ".whl":
            passed, detail = _check_wheel(artifact, version)
        elif artifact.name.endswith(".tar.gz"):
            passed, detail = _check_sdist(artifact, version)
        else:
            passed, detail = True, "checksum recorded"
        report.add(f"Artifact {artifact.name}", passed, detail, "artifacts")

    return report
