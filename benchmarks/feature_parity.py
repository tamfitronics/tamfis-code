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
    Feature("Approval modes", "tamfis_code/permissions.py", r"def decide_permission"),
    Feature("Workspace sandbox", "tamfis_code/sandbox.py", r"SandboxPolicy"),
    Feature("Plans and visible progress", "tamfis_code/orchestrator/engine.py", r"_tool_matches_plan_step"),
    Feature("Durable resume", "tamfis_code/state.py", r"turn_checkpoint"),
    Feature("Session branching / fork", "tamfis_code/state.py", r"def fork_session_state"),
    Feature("Context compaction", "tamfis_code/state.py", r"context_checkpoints"),
    # The four supremacy pillars. Their evidence is the mechanism itself
    # (a class/function that must exist in source), not a claim in docs.
    Feature(
        "Multi-stage context compression", "tamfis_code/orchestrator/compression.py",
        r"class CompressionCascade", kimi="partial", claude="yes", codex="partial",
    ),
    Feature(
        "Prompt cache boundary", "tamfis_code/orchestrator/context.py",
        r"CacheBoundary", kimi="partial", claude="yes", codex="partial",
    ),
    Feature(
        "Permission racing (parallel safety)", "tamfis_code/permission_race.py",
        r"async def race_permission", kimi="no", claude="partial", codex="partial",
    ),
    Feature(
        "Coordinator/worker approval mailbox", "tamfis_code/mailbox.py",
        r"def claim_next", kimi="no", claude="partial", codex="no",
    ),
    Feature("Image input", "tamfis_code/runner_local.py", r"build_vision_content_blocks"),
    Feature("Browser and screenshots", "tamfis_code/cli.py", r"screenshot_cmd"),
    Feature("Web research tool", "tamfis_code/tool_policy.py", r"web_search"),
    Feature("MCP client", "tamfis_code/mcp_client.py", r"StandaloneMCPBridge"),
    Feature("MCP server", "tamfis_code/cli.py", r"mcp-server", kimi="partial", codex="partial"),
    Feature("Agent skills", "tamfis_code/openhands/skills.py", r"SkillRegistry"),
    Feature("Lifecycle hooks", "tamfis_code/hooks.py", r"run_tool_hooks"),
    Feature("Custom agent definitions", "tamfis_code/agent_definitions.py", r"load_agent_definitions"),
    Feature("Parallel subagents / swarm", "tamfis_code/swarm.py", r"run_swarm"),
    Feature("Background jobs and PTY", "tamfis_code/background.py", r"notification_delivered"),
    Feature("JSON and JSONL output", "tamfis_code/render.py", r"StructuredRenderer"),
    Feature("IDE-native integration", "tamfis_code/acp.py", r"class ACPAgent"),
    Feature("Remote/background tasks", "tamfis_code/runner.py", r"submit_ai_task_background", kimi="partial", claude="partial"),
    Feature("GitHub workflow surface", "tamfis_code/github_automation.py", r"install_pr_review_workflow", kimi="partial"),
    Feature("Scheduled automations", "tamfis_code/automation_commands.py", r"serve_automations", kimi="partial", claude="partial"),
    Feature("Worktree isolation", "tamfis_code/runtime/worktree.py", r"create_worktree", kimi="partial", claude="partial"),
    Feature("Diff ledger and revert", "tamfis_code/state.py", r"mutation", kimi="partial", codex="partial"),
)

VERIFY_TESTS = (
    "tests/test_swarm.py",
    "tests/test_context_compression.py",
    "tests/test_permission_race.py",
    "tests/test_mailbox.py",
    "tests/test_plugins.py",
    "tests/test_agent_definitions.py",
    "tests/test_hooks.py",
    "tests/test_mcp_stdio_server.py",
    "tests/test_swarm.py",
    "tests/test_safety.py",
    "tests/test_tamfis_code_render.py",
    "tests/test_acp.py",
    "tests/test_automation_commands.py",
    "tests/test_github_automation.py",
)

# These are regression scenarios rather than source-presence proxies. Keep the
# list explicit so the report says what behavior was exercised.
BEHAVIOR_SCENARIOS = (
    ("permission precedence and protected paths", "tests/test_permissions.py"),
    ("background result reinjection exactly once", "tests/test_background_lifecycle.py"),
    ("natural-language background and goal controls", "tests/test_tamfis_code_intent.py"),
    ("read-only request enforcement", "tests/test_reasoning_plan.py::ReasoningPlanIntegrationTests::test_natural_language_no_edit_constraint_blocks_mutating_shell_calls"),
    ("simple fixes skip formal planning", "tests/test_reasoning_plan.py::ReasoningPlanIntegrationTests::test_simple_single_file_fix_skips_formal_planning"),
    ("failed plans replan once", "tests/test_reasoning_plan.py::ReasoningPlanIntegrationTests::test_plan_is_revised_once_tool_evidence_invalidates_it"),
    ("plan progress uses semantic evidence", "tests/test_orchestrator.py"),
    ("large-file reads paginate by line", "tests/test_mcp.py::TestMCPServer::test_read_file_supports_line_pagination_and_continuation_hint"),
    ("read-only inspection pipelines stay usable", "tests/test_safety.py::ClassifyToolCallRiskTests::test_common_inspection_pipelines_are_read_only"),
    ("duplicate evidence terminates reconnaissance loops", "tests/test_runtime_controller.py::test_different_actions_returning_duplicate_evidence_trigger_stall"),
    ("provider route is announced once per task", "tests/test_tamfis_code_render.py::StreamRendererTests::test_same_route_is_announced_only_once_per_task"),
    ("session forks isolate subsequent conversation", "tests/test_session_fork.py"),
    ("saved plans support detached local execution", "tests/test_cli_commands.py::StandaloneDefaultDispatchTests::test_execute_plan_bg_without_remote_spawns_a_detached_job"),
    ("shell commands support detached local execution", "tests/test_cli_commands.py::StandaloneInfoCommandTests::test_run_bg_without_remote_spawns_a_detached_job"),
    ("queued follow-ups can be recalled and edited", "tests/test_live_input.py::CtrlTInjectsFollowUpTests::test_submitting_recalled_text_updates_queue_without_duplication"),
    # Pillar 1 -- context invincibility.
    (
        "200k-token conversation keeps state through the structured summary",
        "tests/test_context_compression.py::test_200k_token_conversation_still_remembers_a_variable_from_100_messages_ago",
    ),
    (
        "cacheable static prefix stays byte-identical across turns",
        "tests/test_context_compression.py::test_context_bundle_keeps_a_stable_static_prefix_and_puts_volatile_state_second",
    ),
    (
        "superseded file reads keep their signatures, not their bodies",
        "tests/test_context_compression.py::test_stage3_prunes_superseded_file_reads_to_a_signature_view",
    ),
    # Pillar 2 -- permission racing.
    (
        "catastrophic commands are denied before any prompt",
        "tests/test_permission_race.py::test_catastrophic_command_is_denied_without_prompting_the_human",
    ),
    (
        "read-only calls never wait on a classifier or a human",
        "tests/test_permission_race.py::test_read_only_calls_are_decided_by_the_static_process_alone",
    ),
    (
        "classifier deny beats a later human approval",
        "tests/test_permission_race.py::test_classifier_deny_beats_a_human_approval_that_arrives_later",
    ),
    # Pillar 3 -- the mailbox pattern.
    (
        "concurrent coordinators never double-claim a request",
        "tests/test_mailbox.py::test_concurrent_coordinators_never_double_claim",
    ),
    (
        "swarm workers are gated by the coordinator mailbox",
        "tests/test_mailbox.py::test_swarm_runs_workers_through_the_coordinator_mailbox",
    ),
    (
        "a worker cannot approve its own request",
        "tests/test_mailbox.py::test_a_worker_cannot_approve_its_own_request",
    ),
    # Pillar 4 -- protocol and plugin bridge.
    (
        "broken plugin manifests degrade instead of crashing",
        "tests/test_plugins.py::test_a_broken_manifest_degrades_instead_of_crashing",
    ),
    (
        "ACP JSON-RPC framing, cancel, and session updates",
        "tests/test_acp.py",
    ),
    (
        "the live tool dispatch blocks a catastrophic command before it runs",
        "tests/test_runner_permission_race.py::RunnerPermissionRaceTests::test_a_catastrophic_command_is_blocked_before_dispatch",
    ),
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
    headers = ("Feature", "TAMFIS-CODE", "Kimi", "Claude", "Codex")
    rows = [(f.name, f.tamfis, f.kimi, f.claude, f.codex) for f in FEATURES]
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    print("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))
    print()
    labels = {"tamfis": "TAMFIS-CODE", "kimi": "Kimi", "claude": "Claude", "codex": "Codex"}
    for column in ("tamfis", "kimi", "claude", "codex"):
        total = sum(score(getattr(feature, column)) for feature in FEATURES)
        print(f"{labels[column]:11}: {total:g}/{len(FEATURES)} ({total / len(FEATURES):.1%})")


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
        scenario_tests = [node_id for _, node_id in BEHAVIOR_SCENARIOS]
        completed = subprocess.run(
            ["python3", "-m", "pytest", "-q", *VERIFY_TESTS, *scenario_tests],
            cwd=ROOT,
            check=False,
        )
        result = "PASS" if completed.returncode == 0 else "FAIL"
        print(f"Behavioral verification: {result} ({len(BEHAVIOR_SCENARIOS)} named scenarios)")
        for name, _ in BEHAVIOR_SCENARIOS:
            print(f"  - {name}")
        return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
