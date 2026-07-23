from click.testing import CliRunner

from tamfis_code.cli import cli
from tamfis_code.github_commands import GITHUB_COMMANDS


def test_requested_github_command_surface_is_registered():
    missing = sorted(set(GITHUB_COMMANDS) - set(cli.commands))
    assert not missing, f"Missing GitHub-compatible commands: {missing}"


def test_version_is_pinned_to_0613():
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.6.13" in result.output
