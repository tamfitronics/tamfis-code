from pathlib import Path

import pytest
from click.testing import CliRunner

from tamfis_code.cli import cli
from tamfis_code.github_automation import install_pr_review_workflow


def test_generated_pr_review_workflow_is_safe_and_actionable(tmp_path: Path):
    path = install_pr_review_workflow(tmp_path)
    body = path.read_text()
    assert "pull_request:" in body
    assert "pull-requests: write" in body
    assert "--approval read-only" in body
    assert "secrets.TAMFIS_API_KEY" in body
    assert "gh pr comment" in body
    with pytest.raises(FileExistsError):
        install_pr_review_workflow(tmp_path)


def test_github_automation_cli_installs_workflow(tmp_path: Path):
    result = CliRunner().invoke(cli, ["--cwd", str(tmp_path), "github-automation", "install-review"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / ".github" / "workflows" / "tamfis-code-review.yml").is_file()
