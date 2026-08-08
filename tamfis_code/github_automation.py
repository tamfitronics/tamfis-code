"""Generate first-party GitHub Actions workflows backed by tamfis-code."""
from __future__ import annotations

from pathlib import Path

import click


PR_REVIEW_WORKFLOW = """name: Tamfis Code review

on:
  pull_request:
    types: [opened, synchronize, reopened, ready_for_review]

permissions:
  contents: read
  pull-requests: write

concurrency:
  group: tamfis-review-${{ github.event.pull_request.number || github.ref }}
  cancel-in-progress: true

jobs:
  review:
    if: github.event.pull_request.draft == false
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - name: Install Tamfis Code
        run: python -m pip install --disable-pip-version-check tamfis-code
      - name: Review pull request
        env:
          TAMFIS_API_KEY: ${{ secrets.TAMFIS_API_KEY }}
          GH_TOKEN: ${{ github.token }}
          NO_COLOR: '1'
        run: |
          tamfis-code --cwd . --approval read-only local --agent \
            "Review pull request #${{ github.event.pull_request.number }}. Inspect the diff against ${{ github.event.pull_request.base.sha }}, identify only actionable correctness, security, and test issues, and finish with a concise Markdown review." \
            | tee tamfis-review.md
          gh pr comment "${{ github.event.pull_request.number }}" --body-file tamfis-review.md
"""


def install_pr_review_workflow(root: Path, *, force: bool = False) -> Path:
    destination = root.resolve() / ".github" / "workflows" / "tamfis-code-review.yml"
    if destination.exists() and not force:
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(PR_REVIEW_WORKFLOW, encoding="utf-8")
    return destination


@click.group("github-automation")
def github_automation_group() -> None:
    """Install Tamfis Code GitHub Actions automation."""


@github_automation_group.command("install-review")
@click.option("--force", is_flag=True, help="Replace the generated Tamfis workflow if it exists.")
@click.pass_context
def install_review(ctx: click.Context, force: bool) -> None:
    root: Path = ctx.find_root().obj["workspace_root"]
    try:
        destination = install_pr_review_workflow(root, force=force)
    except FileExistsError as exc:
        raise click.ClickException(
            f"{exc.args[0]} already exists; pass --force to replace this generated workflow"
        ) from exc
    click.echo(f"Installed {destination}")
    click.echo("Add TAMFIS_API_KEY as a GitHub Actions repository secret before enabling the workflow.")


def register_github_automation(root: click.Group) -> None:
    root.add_command(github_automation_group)
