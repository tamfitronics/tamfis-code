import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from tamfis_code.orchestrator.planner import (
    build_reasoning_plan_prompt,
    parse_reasoning_plan,
)


def _profile():
    return SimpleNamespace(
        task_type=SimpleNamespace(value="audit"),
        requires_validation=True,
    )


def test_prompt_does_not_assume_project_technology():
    prompt = build_reasoning_plan_prompt(
        "Audit the project",
        _profile(),
        {},
        reconnaissance_summary="root: /tmp/example\nmanifest_backed_commands: none found",
    )

    system = prompt[0]["content"]

    assert "Do not assume that pyproject.toml" in system
    assert "Do not start from model priors" in system


def test_invented_manifest_is_rejected():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()

        raw = json.dumps({
            "steps": [{
                "action": f"Inspect {root / 'pyproject.toml'}",
                "targets": [str(root / "pyproject.toml")],
                "command": None,
                "purpose": "confirm pytest configuration",
                "evidence": [],
            }],
            "assumptions": [],
            "risks": [],
        })

        plan = parse_reasoning_plan(
            raw,
            objective="Audit the repository",
            reconnaissance_summary=f"root: {root}\nmanifest_backed_commands: none found",
            scope_roots=[root],
        )

        assert plan is None


def test_invented_pytest_command_is_rejected():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()

        raw = json.dumps({
            "steps": [{
                "action": "Run backend tests",
                "targets": [str(root)],
                "command": "pytest -q",
                "purpose": "validate the backend",
                "evidence": [],
            }],
            "assumptions": [],
            "risks": [],
        })

        plan = parse_reasoning_plan(
            raw,
            objective="Audit the repository",
            reconnaissance_summary=(
                f"root: {root}\n"
                "manifest_backed_commands: none found"
            ),
            scope_roots=[root],
        )

        assert plan is None


def test_verified_command_is_accepted():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        manifest = root / "package.json"
        manifest.write_text(
            '{"scripts":{"test":"vitest run"}}',
            encoding="utf-8",
        )

        raw = json.dumps({
            "steps": [{
                "action": "Run the verified test script",
                "targets": [str(root)],
                "command": "npm test",
                "purpose": "validate the project",
                "evidence": ["command:npm test"],
            }],
            "assumptions": [],
            "risks": [],
        })

        plan = parse_reasoning_plan(
            raw,
            objective="Validate the project",
            reconnaissance_summary=(
                f"root: {root}\n"
                f"manifest: {manifest}\n"
                "manifest_backed_commands: npm test"
            ),
            scope_roots=[root],
        )

        assert plan is not None
        assert len(plan.steps) == 1
        assert "npm test" in plan.steps[0].name


def test_out_of_scope_existing_path_is_rejected():
    with tempfile.TemporaryDirectory() as authorised:
        with tempfile.TemporaryDirectory() as outside:
            authorised_root = Path(authorised).resolve()
            outside_root = Path(outside).resolve()
            outside_file = outside_root / "package.json"
            outside_file.write_text("{}", encoding="utf-8")

            raw = json.dumps({
                "steps": [{
                    "action": f"Inspect {outside_file}",
                    "targets": [str(outside_file)],
                    "command": None,
                    "purpose": "inspect dependencies",
                    "evidence": [],
                }],
                "assumptions": [],
                "risks": [],
            })

            plan = parse_reasoning_plan(
                raw,
                objective="Audit authorised repository",
                reconnaissance_summary=f"root: {authorised_root}",
                scope_roots=[authorised_root],
            )

            assert plan is None


def test_real_discovered_path_is_accepted():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        source = root / "src"
        source.mkdir()
        file_path = source / "main.custom"
        file_path.write_text("content", encoding="utf-8")

        raw = json.dumps({
            "steps": [{
                "action": f"Read {file_path}",
                "targets": [str(file_path)],
                "command": None,
                "purpose": "trace the objective-relevant implementation",
                "evidence": [f"path:{file_path}"],
            }],
            "assumptions": [],
            "risks": [],
        })

        plan = parse_reasoning_plan(
            raw,
            objective="Audit the implementation",
            reconnaissance_summary=(
                f"root: {root}\n"
                f"objective_matching_paths: {file_path}\n"
                "manifest_backed_commands: none found"
            ),
            scope_roots=[root],
        )

        assert plan is not None
        assert str(file_path) in plan.steps[0].name


def test_existing_but_unconnected_file_is_rejected_when_graph_exists():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()

        active = root / "src" / "active.py"
        unrelated = root / "archive" / "old_copy.py"
        active.parent.mkdir()
        unrelated.parent.mkdir()
        active.write_text("from core import run\n", encoding="utf-8")
        unrelated.write_text("legacy = True\n", encoding="utf-8")

        raw = json.dumps({
            "steps": [{
                "action": f"Inspect {unrelated}",
                "targets": [str(unrelated)],
                "command": None,
                "purpose": "review the implementation",
                "evidence": [],
            }],
            "assumptions": [],
            "risks": [],
        })

        plan = parse_reasoning_plan(
            raw,
            objective="Audit the active stack",
            reconnaissance_summary=(
                f"root: {root}\n"
                f"connected_paths: {active}\n"
                f"objective_matching_paths: {active}\n"
            ),
            scope_roots=[root],
        )

        assert plan is None


def test_connected_import_path_is_accepted():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()

        entrypoint = root / "app" / "main.py"
        dependency = root / "core" / "service.py"
        entrypoint.parent.mkdir()
        dependency.parent.mkdir()
        entrypoint.write_text(
            "from core.service import run\n",
            encoding="utf-8",
        )
        dependency.write_text(
            "def run():\n    return True\n",
            encoding="utf-8",
        )

        raw = json.dumps({
            "steps": [{
                "action": f"Inspect {dependency}",
                "targets": [str(dependency)],
                "command": None,
                "purpose": "trace the imported service used by the entry point",
                "evidence": [f"path:{dependency}"],
            }],
            "assumptions": [],
            "risks": [],
        })

        plan = parse_reasoning_plan(
            raw,
            objective="Audit the active application stack",
            reconnaissance_summary=(
                f"root: {root}\n"
                f"entrypoints: {entrypoint}\n"
                f"imported_paths: {dependency}\n"
                f"connected_paths: {entrypoint}, {dependency}\n"
            ),
            scope_roots=[root],
        )

        assert plan is not None
        assert len(plan.steps) == 1
        assert str(dependency) in plan.steps[0].name


def test_connected_directory_allows_real_descendant():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()

        feature_root = root / "src" / "features" / "chat"
        target = feature_root / "MessageBubble.tsx"
        feature_root.mkdir(parents=True)
        target.write_text("export const MessageBubble = () => null;\n", encoding="utf-8")

        raw = json.dumps({
            "steps": [{
                "action": f"Inspect {target}",
                "targets": [str(target)],
                "command": None,
                "purpose": "review the active chat rendering path",
                "evidence": [f"path:{target}"],
            }],
            "assumptions": [],
            "risks": [],
        })

        plan = parse_reasoning_plan(
            raw,
            objective="Audit the chat stack",
            reconnaissance_summary=(
                f"root: {root}\n"
                f"connected_paths: {feature_root}\n"
            ),
            scope_roots=[root],
        )

        assert plan is not None
