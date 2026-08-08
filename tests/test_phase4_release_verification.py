from pathlib import Path

from tamfis_code.github_commands import GITHUB_COMMANDS
from tamfis_code.release_verification import (
    AUTHORITATIVE_PROVIDER_ORDER,
    REQUIRED_GITHUB_COMMANDS,
    run_release_verification,
)


def test_required_github_command_surface_is_complete():
    assert tuple(GITHUB_COMMANDS) == REQUIRED_GITHUB_COMMANDS
    assert len(GITHUB_COMMANDS) == 27


def test_nvidia_nim_is_authoritative_priority_one():
    assert AUTHORITATIVE_PROVIDER_ORDER == (
        "nvidia", "ollama_cloud", "hf", "openrouter"
    )


def test_source_tree_passes_phase4_release_gate():
    root = Path(__file__).resolve().parents[1]
    report = run_release_verification(root)
    failures = [f"{check.name}: {check.detail}" for check in report.checks if not check.passed]
    assert report.passed, failures
