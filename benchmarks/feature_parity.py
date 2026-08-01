#!/usr/bin/env python3
"""Repeatable, evidence-backed coding-agent feature parity benchmark.

Pass 1 checks that every claimed TamfisGPT Code capability still has source
evidence. Pass 2 (``--verify``) runs focused behavioral tests for the highest
risk capabilities. Competitor columns mean "documented by the vendor", not
that implementations are identical or equally good.
"""
from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Feature:
    name: str
    evidence_file: str
    evidence_pattern: str
    kimi: str = "yes"
    claude: str = "yes"
    codex: str = "yes"
    tamfis: str = "yes"


FEATURES = (
    Feature("Interactive terminal agent", "tamfis_code/interactive.py", r"run_interactive"),
    Feature("Read, edit, and execute tools", "tamfis_code/mcp.py", r"execute_command"),
    Feature("Approval modes", "tamfis_code/config.py", r"APPROVAL_MODES"),
    Feature("Workspace sandbox", "tamfis_code/sandbox.py", r"SandboxPolicy"),
    Feature("Plans and visible progress", "tamfis_code/render.py", r"plan_step_progress"),
    Feature("Durable resume", "tamfis_code/state.py", r"turn_checkpoint"),
    Feature("Context compaction", "tamfis_code/state.py", r"context_checkpoints"),
    Feature("Image input", "tamfis_code/runner_local.py", r"build_vision_content_blocks"),
    Feature("Browser and screenshots", "tamfis_code/cli.py", r"screenshot_cmd"),
    Feature("Web research tool", "tamfis_code/tool_policy.py", r"web_search"),
    Feature("MCP client", "tamfis_code/mcp_client.py", r"StandaloneMCPBridge"),
    Feature("MCP server", "tamfis_code/cli.py", r"mcp-server", kimi="partial", codex="partial"),
    Feature("Agent skills", "tamfis_code/openhands/skills.py", r"SkillRegistry"),
    Feature("Lifecycle hooks", "tamfis_code/hooks.py", r"run_tool_hooks"),
    Feature("Custom agent definitions", "tamfis_code/agent_definitions.py", r"load_agent_definitions"),
    Feature("Parallel subagents / swarm", "tamfis_code/swarm.py", r"run_swarm"),
    Feature("Background jobs and PTY", "tamfis_code/mcp.py", r"read_background_job"),
    Feature("JSON and JSONL output", "tamfis_code/render.py", r"StructuredRenderer"),
    Feature("IDE-native integration", "tamfis_code/cli.py", r"mcp-server", tamfis="partial"),
    Feature("Remote/background tasks", "tamfis_code/runner.py", r"submit_ai_task_background", kimi="partial", claude="partial"),
    Feature("GitHub workflow surface", "tamfis_code/github_commands.py", r"GITHUB_COMMANDS", kimi="partial", tamfis="partial"),
    Feature("Scheduled automations", "tamfis_code/openhands/automation.py", r"AutomationScheduler", kimi="partial", claude="partial", tamfis="partial"),
    Feature("Worktree isolation", "tamfis_code/runtime/worktree.py", r"create_worktree", kimi="partial", claude="partial"),
    Feature("Diff ledger and revert", "tamfis_code/state.py", r"mutation", kimi="partial", codex="partial"),
)

VERIFY_TESTS = (
    "tests/test_agent_definitions.py",
    "tests/test_hooks.py",
    "tests/test_mcp_stdio_server.py",
    "tests/test_swarm.py",
    "tests/test_safety.py",
    "tests/test_tamfis_code_render.py",
)


def score(value: str) -> float:
    return {"yes": 1.0, "partial": 0.5, "no": 0.0}[value]


def source_pass() -> list[str]:
    failures: list[str] = []
    for feature in FEATURES:
        path = ROOT / feature.evidence_file
        if not path.is_file() or re.search(feature.evidence_pattern, path.read_text(errors="replace")) is None:
            failures.append(feature.name)
    return failures


def print_report() -> None:
    headers = ("Feature", "Tamfis", "Kimi", "Claude", "Codex")
    rows = [(f.name, f.tamfis, f.kimi, f.claude, f.codex) for f in FEATURES]
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    print("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))
    print()
    for column in ("tamfis", "kimi", "claude", "codex"):
        total = sum(score(getattr(feature, column)) for feature in FEATURES)
        print(f"{column.title():7}: {total:g}/{len(FEATURES)} ({total / len(FEATURES):.1%})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="also run focused behavioral tests")
    args = parser.parse_args()
    print_report()
    failures = source_pass()
    print(f"Static evidence: {'PASS' if not failures else 'FAIL'}")
    if failures:
        print("Missing evidence: " + ", ".join(failures))
        return 1
    if args.verify:
        completed = subprocess.run(
            ["python3", "-m", "pytest", "-q", *VERIFY_TESTS], cwd=ROOT, check=False,
        )
        print(f"Behavioral verification: {'PASS' if completed.returncode == 0 else 'FAIL'}")
        return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
