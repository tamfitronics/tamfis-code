from pathlib import Path

from tamfis_code.orchestrator.validator import _explicit_output_paths, validate_completion
from tamfis_code.routing import classify_task


def test_extracts_explicit_project_tree_files_without_urls():
    objective = """Create complete theme:
enterprise-theme/style.css
enterprise-theme/functions.php
enterprise-theme/assets/js/app.js
See https://example.com/file.css
"""
    assert _explicit_output_paths(objective) == [
        "enterprise-theme/style.css",
        "enterprise-theme/functions.php",
        "enterprise-theme/assets/js/app.js",
    ]


def test_complete_project_cannot_finish_with_missing_tree_files(tmp_path: Path):
    present = tmp_path / "theme" / "style.css"
    present.parent.mkdir()
    present.write_text("/* Theme Name: Enterprise */")
    objective = "Create complete project with theme/style.css and theme/functions.php"
    report = validate_completion(
        profile=classify_task(objective),
        tool_records=[{
            "tool_name": "write_file", "success": True,
            "arguments": {"path": "theme/style.css"},
        }, {
            "tool_name": "read_file", "success": True,
            "arguments": {"path": "theme/style.css"},
        }],
        any_mutation=True, final_text="Complete.", objective=objective,
        workspace_root=str(tmp_path),
    )
    assert report.passed is False
    assert any("theme/functions.php" in item for item in report.unresolved)


def test_complete_project_passes_output_contract_when_every_file_exists(tmp_path: Path):
    for relative in ("theme/style.css", "theme/functions.php"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("content")
    objective = "Create complete project with theme/style.css and theme/functions.php"
    report = validate_completion(
        profile=classify_task(objective),
        tool_records=[
            {"tool_name": "write_file", "success": True, "arguments": {"path": "theme/style.css"}},
            {"tool_name": "execute_command", "success": True, "exit_code": 0, "arguments": {"command": "test -s theme/functions.php"}},
        ],
        any_mutation=True, final_text="Complete.", objective=objective,
        workspace_root=str(tmp_path),
    )
    assert report.passed is True
