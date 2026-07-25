from pathlib import Path

import pytest

from tamfis_code.runtime.workspace_authority import (
    WorkspaceAuthorityError,
    explicit_absolute_targets,
    resolve_workspace_targets,
)


def _project(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "pyproject.toml").write_text('[project]\nname="x"\n', encoding="utf-8")
    return path


def test_launch_root_is_only_implicit_workspace(tmp_path: Path):
    current = _project(tmp_path / "tamfisseo")
    other = _project(tmp_path / "tamfisgpt")
    with pytest.raises(WorkspaceAuthorityError) as exc:
        resolve_workspace_targets(
            launch_root=current,
            objective="audit the full stack TamfisGPT IOS",
            allowed_roots=[current],
        )
    assert str(other) in str(exc.value)
    assert "No external files were inspected" in str(exc.value)


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


def test_runner_integrates_authority_and_freshness_rules():
    text = Path("tamfis_code/runner_local.py").read_text(encoding="utf-8")
    assert "resolve_workspace_targets(" in text
    assert "SUPERSEDED" in text
    assert "current source" in text
    assert "do not present a final audit conclusion" in text
