from pathlib import Path

import pytest

from tamfis_code.runtime.workspace_authority import (
    WorkspaceAuthorityError,
    auto_grant_explicit_targets,
    explicit_absolute_targets,
    resolve_workspace_targets,
)


def _project(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "pyproject.toml").write_text('[project]\nname="x"\n', encoding="utf-8")
    return path


def test_launch_root_is_only_implicit_workspace(tmp_path: Path):
    current = _project(tmp_path / "tamfisseo")
    _project(tmp_path / "tamfisgpt")
    result = resolve_workspace_targets(
        launch_root=current,
        objective="audit the full stack TamfisGPT IOS",
        allowed_roots=[current],
    )
    assert result.roots == (current.resolve(),)


def test_named_project_context_does_not_change_workspace_target(tmp_path: Path):
    current = _project(tmp_path / "tamfisseo")
    _project(tmp_path / "tamfisgpt")
    result = resolve_workspace_targets(
        launch_root=current,
        objective="Use port 3001 because TamfisGPT uses port 5173 for dev.",
        allowed_roots=[current],
    )
    assert result.roots == (current.resolve(),)


def test_explicit_external_path_requires_prior_grant(tmp_path: Path):
    current = _project(tmp_path / "tamfisseo")
    other = _project(tmp_path / "tamfisgpt")
    with pytest.raises(WorkspaceAuthorityError):
        resolve_workspace_targets(
            launch_root=current,
            objective=f"audit {other}",
            allowed_roots=[current],
        )


def test_explicit_granted_external_path_is_selected(tmp_path: Path):
    current = _project(tmp_path / "tamfisseo")
    other = _project(tmp_path / "tamfisgpt")
    result = resolve_workspace_targets(
        launch_root=current,
        objective=f"audit {other}",
        allowed_roots=[current, other],
    )
    assert result.roots == (other.resolve(),)


def test_unrelated_absolute_file_does_not_expand_to_sibling(tmp_path: Path):
    current = _project(tmp_path / "tamfisseo")
    target = current / "README.md"
    target.write_text("x", encoding="utf-8")
    assert explicit_absolute_targets(f"read {target}") == (current.resolve(),)


def test_pasted_status_hint_is_not_an_absolute_target(tmp_path: Path):
    current = _project(tmp_path / "tamfisseo")
    assert explicit_absolute_targets("ready · /status for session, task, cwd") == ()
    result = resolve_workspace_targets(
        launch_root=current,
        objective="ready · /status for session, task, cwd",
        allowed_roots=[current],
    )
    assert result.roots == (current.resolve(),)


def test_explicit_external_path_is_automatically_added_to_grant(tmp_path: Path):
    current = _project(tmp_path / "tamfisseo")
    other = _project(tmp_path / "tamfisgpt")
    target = other / "src" / "App.tsx"
    target.parent.mkdir()
    target.write_text("export default null", encoding="utf-8")

    allowed, added = auto_grant_explicit_targets(
        launch_root=current,
        objective=f"read {target}",
        allowed_roots=[current],
    )

    assert added == (other.resolve(),)
    assert allowed == (current.resolve(), other.resolve())
    result = resolve_workspace_targets(
        launch_root=current,
        objective=f"read {target}",
        allowed_roots=allowed,
    )
    assert result.roots == (other.resolve(),)


def test_pasted_api_routes_and_error_text_are_not_absolute_targets(tmp_path: Path):
    # A user pasting an error message that quotes REST routes and a
    # nonexistent multi-segment file path (copied from a backend log, not a
    # real request to access it) used to be treated as an explicit
    # filesystem target and fail the whole turn closed with "outside the
    # active workspace grant". Only tokens that exist on disk are targets.
    current = _project(tmp_path / "tamfisseo")
    objective = (
        "Validation failed: claims these paths were changed without mutation evidence: "
        "/api/v1/chat/models, /home/tests/test_model_catalog_tier_enforcement.py, /health"
    )
    assert explicit_absolute_targets(objective) == ()
    result = resolve_workspace_targets(
        launch_root=current,
        objective=objective,
        allowed_roots=[current],
    )
    assert result.roots == (current.resolve(),)


def test_runner_integrates_authority_and_freshness_rules():
    text = Path("tamfis_code/runner_local.py").read_text(encoding="utf-8")
    assert "resolve_workspace_targets(" in text
    assert "SUPERSEDED" in text
    assert "current source" in text
    assert "do not present a final audit conclusion" in text
