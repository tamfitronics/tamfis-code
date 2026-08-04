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


def test_mutation_claim_ignores_api_routes_and_unrelated_backticked_mentions(tmp_path: Path):
    # A model's report legitimately mentions a REST route, a product name,
    # and a pre-existing test file in backticks near completion language.
    # None of those are filesystem paths the agent claimed to change, so
    # they must not trip the "claims files changed without mutation
    # evidence" gate as long as the actually-edited file is supported.
    edited = tmp_path / "catalog.py"
    edited.write_text("x", encoding="utf-8")
    objective = "fix the model catalog tier bug"
    final_text = (
        "I updated `catalog.py` to fix the tier filter. The endpoint `/api/v1/chat/models` "
        "now reflects this, as covered by `TamfisGPT Code`. See `/health` for the readiness "
        "probe and `/home/tests/test_model_catalog_tier_enforcement.py` for existing coverage."
    )
    report = validate_completion(
        profile=classify_task(objective),
        tool_records=[
            {
                "tool_name": "edit_file", "success": True,
                "arguments": {"path": "catalog.py"}, "files_changed": ["catalog.py"],
            },
            {
                "tool_name": "execute_command", "success": True, "exit_code": 0,
                "arguments": {"command": "pytest"},
            },
        ],
        any_mutation=True, final_text=final_text, objective=objective,
        workspace_root=str(tmp_path),
    )
    assert report.passed is True
    assert report.unresolved == []


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
