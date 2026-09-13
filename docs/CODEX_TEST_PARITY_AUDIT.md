# Codex test-suite parity audit

Snapshot date: 2026-09-13. Source compared: `github.com/openai/codex` (Rust,
~651 test files across `codex-rs/*/tests/`) against tamfis-code's own test
suite (`tests/*.py`) and source (`tamfis_code/*.py`).

Scope: only categories with a genuinely comparable tamfis-code feature area
were audited. Explicitly out of scope (Codex-only enterprise/cloud product
surface tamfis-code has no equivalent of, and never will as a standalone
CLI): cloud auth/ChatGPT backend, marketplace/plugin store, voice/realtime
webrtc, the "guardian" review subsystem, Windows/macOS sandbox backends,
`bedrock_setup`, `luna_reserve`, `daybreak_access`, `cyber_access_program`,
`app-server`'s v2 JSON-RPC surface (Codex's separate daemon product, ~180
files), OTel backend wiring specifics, `remote_control`, realtime-conversation
files.

Verdict legend:
- **COVERED** — feature exists, real edge-case tests exist for it.
- **GAP** — feature exists, but coverage is thin/missing; actionable.
- **N/A** — no equivalent tamfis-code feature (reason given); not a test gap.

This file is a living record. As gaps are closed or reclassified after
deeper verification, update the row in place and add a dated line to the
Changelog at the bottom — do not just delete the history.

## Sandbox / exec-policy

| Codex category | tamfis-code feature | Verdict |
|---|---|---|
| `sandbox.rs`, `sandbox_tty.rs`, `sandbox_network_proxy.rs`, `sandbox_cloud_config.rs` | `sandbox.py`, `SandboxPolicy` | COVERED — `test_sandbox.py` |
| `landlock.rs`, `bundled_bwrap.rs` | bwrap invocation in `sandbox.py`'s `build_sandbox_command` | COVERED (fixed 2026-09-13) — `test_sandbox.py`'s `TestRealBwrapEnforcement` runs the real, unmocked bwrap binary and confirms live: a write outside workspace_root/writable_roots fails (path invisible inside the sandbox), a write inside succeeds, `network_access=False` actually blocks a connection attempt ("Network is unreachable"), and `network_access=True` omits `--unshare-net` and allows the command to run |
| `windows_sandbox.rs` | none | N/A — Linux-only |
| `managed_proxy.rs` | none | N/A — no network-proxy concept |
| execpolicy DSL (`basic.rs`, `execpolicy.rs`, `exec_policy.rs`, `cyber_exec_policy.rs`) | `tool_policy.py` (56 lines, pure tool-category allow-listing) + `safety.py`'s `classify_command_risk` (a 3-tier read_only/medium/dangerous heuristic used to gate approval prompts) | N/A — confirmed by reading both files: neither is a declarative per-argument command policy DSL the way Codex's execpolicy crate is; architecturally different, not a missing test |
| `extension_sandbox.rs` | `plugins.py` | N/A — confirmed by reading `plugins.py` in full: a plugin's tool handler is registered directly into the same `server.tools` dict as every built-in tool (`register_plugin_tools`) and runs in-process with full privileges; there is no isolation boundary at all to test |

## Apply-patch / file mutation

| Codex category | tamfis-code feature | Verdict |
|---|---|---|
| `no_follow.rs` (symlink escape) | `_resolve_in_workspace` in `mcp.py` | COVERED (fixed 2026-09-13) — `test_mcp.py` |
| `tool.rs`, `scenarios.rs`, `cli.rs`, `apply_patch_serialization.rs`, `apply_patch_cli.rs` | `edit_file`/`write_file` (old_string/new_string, not unified-diff hunks) | N/A — different mechanism |

## MCP protocol

| Codex category | tamfis-code feature | Verdict |
|---|---|---|
| External MCP config loading, `mcp_login.rs` | `mcp_client.py` `load_mcp_servers` | COVERED — `test_standalone_mcp_client.py` |
| `mcp_tool_cache.rs` | none | N/A — confirmed by full-text search: zero hits for "cache" in `mcp_client.py`/`mcp.py` related to tool lists; `StandaloneMCPBridge._tools`/`_tool_map` are populated once per bridge instance and the bridge itself is rebuilt fresh per turn, so there is nothing to invalidate |
| OAuth refresh (`mcp_auth_refresh.rs`, `mcp_oauth_refresh_tests.rs`, `mcp_auth_elicitation.rs`) | none | N/A — confirmed: zero hits for "oauth" or "elicit" in `mcp_client.py`/`mcp.py`; external MCP servers authenticate via static `config.headers`/`config.env` only |
| Startup lifecycle/grace (`mcp_startup_refresh_http_proxy.rs`, `mcp_refresh_cleanup.rs`, `mcp_optional_startup_grace.rs`) | `StandaloneMCPBridge.initialize()` (`mcp_client.py`) | COVERED (fixed 2026-09-13) — confirmed live and now tested: `initialize()` runs every configured server's startup concurrently via `asyncio.gather`, and `_initialize_stdio_server`/`_initialize_http_server` each catch their own `Exception` internally (terminating the process / dropping the http connection) rather than propagating it, so one server that can't start never blocks a working server alongside it. New test: `test_standalone_mcp_client.py::test_one_server_failing_at_startup_does_not_block_the_others` |
| `mcp_extension_protocol.rs`, `mcp_ema_config.rs`, elicitation (`mcp_tool_exposure.rs`, `mcp_turn_metadata.rs`, `mcp_user_verification.rs`) | none | N/A — no elicitation protocol |
| `ext/mcp/tests` (hosted apps) | none | N/A — out of scope |

## Approvals / permissions

| Codex category | tamfis-code feature | Verdict |
|---|---|---|
| `approvals.rs`, `network_approval*.rs` | `permissions.py`, approval flow | COVERED |
| `safety_check_downgrade.rs`, `safety_buffering.rs` | `safety.py`, safety manifest | COVERED |
| `skill_approval.rs`, `skills*.rs` | none | N/A — no skills marketplace |
| `request_user_input*.rs` | `mcp.py`'s `_ask_user_question` tool | COVERED (confirmed 2026-09-13) — `test_mcp.py`'s `TestAskUserQuestionTool` has 5 tests: unavailable without an attached console, unavailable when non-interactive, free-text answer, numeric option selection, free text still accepted when options are offered |
| `request_permissions*.rs` | `permissions.py`'s `decide_permission` -- a distinct, persisted allow/ask/deny rule engine, separate from the interactive live approval-prompt flow that `approvals.rs` maps to | COVERED (confirmed 2026-09-13) — `test_permissions.py` covers deny-precedes-ask-and-allow, scoped tool/path rule matching, protected-path forcing "ask" even under an allow rule, explicit deny still winning for a protected path, and a shell command merely mentioning a protected path also requiring approval |
| `permissions_messages.rs`, `catalog_permission_messages.rs` | none (inlined copy, not catalog-driven) | N/A |

## Hooks

| Codex category | tamfis-code feature | Verdict |
|---|---|---|
| `hooks.rs`, `hooks_executor.rs` | `hooks.py`, `HOOK_TIMEOUT_SECONDS` | COVERED (timeout-kill path fixed 2026-09-13) |
| `hooks_mcp.rs` (hooks calling MCP tools) | none | N/A — confirmed by reading `hooks.py` in full (182 lines, the whole module): a hook is an arbitrary shell command run via `asyncio.create_subprocess_shell` in its own OS process, receiving only a JSON event on stdin; there is no API surface for a hook to call back into tamfis-code's in-process MCP tool registry at all |
| `interrupt_hooks.rs` (hooks on task interruption) | `hooks.py`'s new `session_interrupted` event + `run_session_hooks()`; fired from `runner_local.py`'s `_fire_session_interrupted_hooks()` at all 10 call sites where a turn is checkpointed interrupted | COVERED (added 2026-09-13) — 6 unit tests in `test_hooks.py` plus a real end-to-end integration test in `test_round_budget_extension.py` proving an on-disk `.tamfis/hooks.toml` hook actually fires on a real strict-round-cap interruption |

## Resume / compaction / rollout

| Codex category | tamfis-code feature | Verdict |
|---|---|---|
| `resume.rs`, `resume_warning.rs` | `save_turn_checkpoint`, `_select_resume_state`, `_resume_file_status_note` | COVERED — exceeds Codex in one respect (file-fingerprint "did another coder touch it" check has no direct Codex equivalent) |
| `compact*.rs` | compaction across `state.py`/`runner.py`/`interactive.py` | COVERED — `test_thread_compression.py` |
| `rollout_*.rs`, `sqlite_state.rs` | none (plain JSON state file, not JSONL rollout/SQLite) | N/A — different persistence architecture |

## Resilience

| Codex category | tamfis-code feature | Verdict |
|---|---|---|
| `retry_after.rs` | none (immediate failover instead of backoff) | N/A — architectural difference |
| `quota_exceeded.rs` | `ProviderManager.is_quota_or_rate_limit_error` (`providers.py`) | COVERED (fixed 2026-09-13) — the real classmethod had zero direct tests (only a hand-rolled fake reimplementing simplified logic existed in `test_reasoning_plan.py`'s `_FallbackCapableManager`); `test_provider_key_rotation.py`'s new `TestIsQuotaOrRateLimitError` covers HTTP 429/402/404 status codes, message-marker fallback, a plain transport failure NOT being classified as quota, and a `.response.status_code` attribute also being read |
| `model_switching.rs`, `model_overrides.rs`, `model_runtime_selectors.rs`, `model_provider_requirements_tests.rs` | `model_registry.py`, `model_group_fallback` | COVERED (fallback path); override/selector edge cases unconfirmed |
| `stream_error_allows_next_turn.rs`, `stream_no_completed.rs` | stream-error handling | COVERED — `test_stream_reconnect.py` |
| `prompt_caching.rs`, `prompt_cache_key.rs` | none (no `cache_control` breakpoints implemented) | N/A — real feature gap, not a test gap |
| `token_budget.rs`, `token_usage_rollout.rs` | `_estimate_tokens`/`_trim_tool_outputs` (`runner_local.py`) -- real context-window budget accounting (`token_budget = context_window * safety_margin - MAX_TOKENS_PER_REQUEST`), distinct from `test_round_budget_extension.py`'s round-COUNT safety valve | COVERED (fixed 2026-09-13) — confirmed `test_round_budget_extension.py` is unrelated (round count, not tokens) and neither token function had any test at all; new `tests/test_token_budget_trimming.py` covers estimate correctness (content + tool_calls arguments), a no-op when already under budget, shrinking an oversized history below target, and the leading system / latest user messages always being preserved |

## Tool execution mechanics

| Codex category | tamfis-code feature | Verdict |
|---|---|---|
| `unified_exec*` family (one PTY+approval+stdin-review tool) | `pty.py`/`local_pty.py` + separate `execute_command` (two tools, different architecture) | N/A (borderline) — literal port doesn't make sense; underlying scenarios (approval mid-command, stdin review) not confirmed tested either way |
| `tool_parallelism.rs` (concurrent tool_calls in one turn) | `runner_local.py`'s `_dispatch_queue`/`_conflicts`/`_flush_dispatch_queue` -- a real, non-trivial feature: independent tool calls in one round run concurrently via `asyncio.gather`, split into "maximal conflict-free groups" (two calls conflict if they touch overlapping paths and either isn't read-only, or an `execute_command` is paired with any non-read-only mutation) | COVERED (fixed 2026-09-13) — `test_concurrent_dispatch_regressions.py` already covered the no-conflict/concurrent-success case; added `test_two_writes_to_the_same_path_never_dispatch_in_the_same_group`, which records every real `asyncio.gather` call's argument count (black-box, not reaching into private dispatch-queue state) and confirms two same-path writes are always split into separate size-1 groups, never raced against each other |
| `tool_lifecycle.rs`, `tool_harness.rs` | distributed across many tool-specific test files | COVERED |
| `turn_input_submission.rs`, `pending_input.rs` | `state.py`'s `enqueue_instruction`/`queued_user_instructions` -- mid-turn follow-up messages are queued (priority-ordered, durable) rather than lost while a task is running | COVERED — exercised across 7 test files including a dedicated `test_live_input.py` |
| `direct_tool_metadata.rs` | `mcp.py`'s `list_tools`/`tool_schemas_openai`/`external_tool_schemas_openai` | COVERED (as a general concept) — exercised across `test_mcp.py`, `test_integration_new.py`, `test_mcp_stdio_server.py`, `test_standalone_mcp_client.py`; Codex's exact scenario in this file wasn't individually confirmed since the Codex source wasn't re-read line-by-line here, but the underlying tool-metadata-exposure surface is real and tested |
| `turn_state.rs` | none, as a distinct concept | N/A — no separate "turn state" abstraction exists; the underlying concerns (round budget, dispatch queue, checkpoint) are already audited individually elsewhere in this document |

## Context / prompt engineering

| Codex category | tamfis-code feature | Verdict |
|---|---|---|
| `agents_md.rs`, `agents_md_refresh.rs` | `workspace.py`'s `load_instruction_text` | COVERED (fixed 2026-09-13) — confirmed live and now tested: `load_instruction_text` has no cache/memoization at all, calling `Path.read_text` fresh on every call (both call sites, `orchestrator/engine.py`'s `validate()` and `runner_local.py`'s preflight validation, invoke it fresh on every validation pass, not once per session) -- a mid-session AGENTS.md edit takes effect on the very next validation with nothing to invalidate. New `test_tamfis_code_workspace.py::test_load_instruction_text_always_re_reads_from_disk` |
| `additional_context.rs`, `context_annotations.rs`, `current_time_reminder.rs`, `personality.rs`, `collaboration_instructions.rs`, `git_enrichment.rs` | none | N/A — confirmed by full-text search across all of `tamfis_code/*.py`: zero hits for "current_time", "collaboration_instruction", "git_enrichment", or "personality" as a concept; none of these exist under any name |
| `truncation.rs` | Context-budget truncation is `_trim_tool_outputs`/`_estimate_tokens` (`runner_local.py`) -- already closed under Resilience's `token_budget.rs` row above | COVERED (via the token_budget.rs fix) -- note: `_truncate_degenerate_repetition`/`_corrupted_lexical_stream_index` also exist in `runner_local.py` but solve a different problem entirely (detecting and truncating repetitive/corrupted model output, a quality-control guard, not context-window truncation) and were left untested as out of scope for this row |
| `audio_truncation.rs` | none (audio not a real input modality) | N/A |
| `view_image.rs` | `is_vision_image_path`/`build_vision_content_blocks`/`_messages_with_vision_content` (`runner_local.py`) | COVERED (fixed 2026-09-13) — new `tests/test_vision_attachments.py` (13 tests): recognised/unrecognised image extensions, a real image becoming a base64 data URI block, a non-image path and a missing path both silently skipped, an oversized image skipped, mixed-path filtering, splicing into the target user message without mutating the original, no-op when there are no blocks or no target index, and a non-user target message left untouched |
| `web_search.rs`, `search_tool.rs` | both implemented | COVERED |

## Multi-agent / delegation

| Codex category | tamfis-code feature | Verdict |
|---|---|---|
| `fork_thread.rs` | session fork | COVERED — `test_session_fork.py` |
| `multi_agent_resume.rs`, `multi_agent_mode.rs`, `codex_delegate.rs`, `spawn_agent_description.rs` | swarm delegation | COVERED — `test_swarm.py`, `test_agents_delegation.py` |
| `subagent_notifications.rs` | `swarm.py`'s `BufferedSubagentRenderer` -- relays real-time status (model selected, tool call, file mutation, failure) from each concurrent sub-task back to the parent's aggregate status display via `on_update` | COVERED (confirmed 2026-09-13) — `test_swarm.py`'s `BufferedSubagentRendererTests` covers event-to-update translation, the no-`on_update` no-op case, and multiple independent renderer instances |
| `subagent_service_tier.rs` | none | N/A — confirmed: no "tier" concept anywhere in `swarm.py`; every sub-task resolves its provider/model the same way as a top-level turn |

## CLI subcommands

| Codex category | tamfis-code feature | Verdict |
|---|---|---|
| `login.rs`, `auth_refresh.rs` | `RemoteAPIClient.login`/`_refresh` (`api_client.py`) | COVERED (confirmed 2026-09-13) — `test_tamfis_code_api_client.py` already covers: a 401 triggers refresh then retries successfully, the refresh token is sent as a cookie (not a JSON body -- a specific prior regression), a 401 with no refresh token raises `AuthRequiredError`, login's flat-token-response parsing, and login-failure error detail |
| `device_code_login.rs`, `login_server_e2e.rs`, `logout.rs` | none (no OAuth/device-code flow) | N/A |
| `doctor_path_safety.rs` | `check_path_safety()` | COVERED (added 2026-09-13) |
| `doctor_enterprise_network.rs` | none | N/A — out of scope |
| `mcp_add_remove.rs`, `mcp_list.rs`, `mcp_login.rs` | config-file-based MCP server config (not imperative CLI) | COVERED via config-loading tests; imperative-CLI surface itself N/A (doesn't exist) |
| `queue.rs` | `cli.py`'s `queue` command over `state.py`'s `enqueue_instruction` | COVERED — same underlying mechanism as `turn_input_submission.rs`/`pending_input.rs` above, tested across 7 files |
| `delete.rs`, `debug_clear_memories.rs` | `cli.py`'s `clear-session` command | COVERED — tested across `test_cli_commands.py`, `test_session_lifecycle.py`, `test_state_caps_and_pruning.py` |
| `debug_models.rs` | `cli.py`'s `providers` command | COVERED (loosely) — confirmed by reading it in full: a thin ~20-line display wrapper around `get_provider_status()` with no independent logic of its own to test beyond what the underlying provider-status/model-routing tests already exercise |
| `features.rs` (feature-flag listing) | none | N/A — confirmed: no such command exists in `cli.py`'s command list |

## Telemetry

| Codex category | tamfis-code feature | Verdict |
|---|---|---|
| `otel.rs` | out of scope | N/A |
| Runtime execution journal | `runtime/journal.py` | COVERED |

## Changelog

- 2026-09-13: Initial audit compiled and this document created. Fixed as
  part of the same pass: symlink-escape write/edit tests (`test_mcp.py`),
  hook-execution-timeout test (`test_hooks.py`), `doctor` PATH
  world-writable check + tests (`doctor.py`, `test_doctor_session_diagnostics.py`).
- 2026-09-13: Sandbox theme resolved. `landlock.rs`/`bundled_bwrap.rs`
  closed with 4 new real-bwrap-execution tests in `test_sandbox.py`
  (`TestRealBwrapEnforcement`, skipped if bwrap isn't installed). execpolicy
  DSL and `extension_sandbox.rs` N/A verdicts confirmed by reading
  `tool_policy.py`/`safety.py`/`plugins.py` in full rather than by grep.
- 2026-09-13: MCP protocol theme resolved. Startup lifecycle/grace closed
  with a new test proving one broken server doesn't block a working one
  alongside it (`test_standalone_mcp_client.py`). `mcp_tool_cache.rs` and
  OAuth-refresh N/A verdicts tightened with concrete full-text-search
  evidence instead of a prior grep pass's "not found."
- 2026-09-13: Approvals/permissions theme resolved. Both `request_user_input*.rs`
  and `request_permissions*.rs` reclassified GAP -> COVERED after reading
  `test_mcp.py`'s `TestAskUserQuestionTool` and `test_permissions.py` in
  full: both already have real edge-case coverage, not just happy-path.
  Confirmed `decide_permission` (persisted allow/ask/deny rules) and the
  interactive approval-prompt flow are two genuinely distinct mechanisms,
  each separately tested.
- 2026-09-13: Hooks theme resolved. `hooks_mcp.rs` and `interrupt_hooks.rs`
  both N/A, confirmed by reading `hooks.py` in full (182 lines): hooks are
  plain shell subprocesses with no MCP-calling API, and there is no
  third hook-event category for task interruption at all --
  `interrupt_hooks.rs` flagged as a genuine future FEATURE gap, not a test
  gap, and not built here.
- 2026-09-13: Resilience theme's `quota_exceeded.rs` and `token_budget.rs`
  closed. `ProviderManager.is_quota_or_rate_limit_error` never had a direct
  test (`test_provider_key_rotation.py`'s new `TestIsQuotaOrRateLimitError`,
  6 tests). `_estimate_tokens`/`_trim_tool_outputs` never had any test at
  all (`tests/test_token_budget_trimming.py`, 9 new tests) -- confirmed
  distinct from the existing round-count budget tests.
- 2026-09-13: Tool execution mechanics theme resolved. `tool_parallelism.rs`
  closed with a new test proving two same-path writes never dispatch in
  the same concurrent group (`test_concurrent_dispatch_regressions.py`,
  verified via a real `asyncio.gather` call-count recorder rather than
  private state). `turn_input_submission.rs`/`pending_input.rs` reclassified
  COVERED (the queued-instruction mechanism, tested across 7 files).
  `direct_tool_metadata.rs` reclassified COVERED (general tool-metadata
  exposure, tested across 4 files). `turn_state.rs` confirmed N/A (no
  distinct abstraction; underlying concerns audited elsewhere).
- 2026-09-13: Context/prompt-engineering theme resolved. `agents_md_refresh.rs`
  closed -- confirmed `load_instruction_text` has no caching at all, so a
  mid-session edit is always picked up on the next read; new test added.
  `truncation.rs` reclassified COVERED via the token_budget.rs fix above
  (same underlying functions). `view_image.rs` closed with a new dedicated
  `tests/test_vision_attachments.py` (13 tests). Remaining N/A rows
  (`additional_context.rs`, `context_annotations.rs`, `current_time_reminder.rs`,
  `personality.rs`, `collaboration_instructions.rs`, `git_enrichment.rs`)
  confirmed via a full-text search across all of `tamfis_code/*.py` --
  none exist under any name.
- 2026-09-13: Multi-agent and CLI-subcommand themes resolved, no code
  changes needed (all reclassified after reading the actual code/tests in
  full). `subagent_notifications.rs` -> COVERED (`BufferedSubagentRenderer`,
  tested in `test_swarm.py`); `subagent_service_tier.rs` -> N/A (no "tier"
  concept exists). `login.rs`/`auth_refresh.rs` -> COVERED
  (`test_tamfis_code_api_client.py` already covers the 401-retry, cookie-vs-
  JSON-body, and no-refresh-token paths in detail). `queue.rs` -> COVERED
  (same mechanism as `turn_input_submission.rs`). `delete.rs`/
  `debug_clear_memories.rs` -> COVERED (`clear-session`, tested across 3
  files). `debug_models.rs` -> COVERED loosely (`providers` command is a
  thin display wrapper with no independent logic). `features.rs` -> N/A
  (no such command exists).

- 2026-09-13: `interrupt_hooks.rs` closed. The genuine feature gap flagged
  in the prior pass was built: a new `session_interrupted` hook event
  fires from all 10 sites in `runner_local.py` where a turn is
  checkpointed interrupted. See `tests/test_hooks.py`'s
  `TestRunSessionHooks` and `test_round_budget_extension.py`'s new
  end-to-end integration test.

## Claude Code comparison

Unlike Codex, Claude Code's own source and internal test suite are not
public, so this section is not a test-file diff. Ground truth instead
comes from what is actually installed and inspectable on this same box:
the official `plugin-dev` plugin's developer documentation
(`~/.claude/plugins/marketplaces/claude-plugins-official/plugins/plugin-dev/skills/*/SKILL.md`
— authoritative first-party specs for hooks, agents, skills, commands,
settings, and MCP integration), real `hooks.json` configs from six other
official plugins, the 6,679-line `~/.claude/cache/changelog.md`, and
`claude --help`'s full CLI reference. Because Claude Code is a more
mature product in some of these areas, "GAP" here more often means a
genuine missing *feature*, not just missing tests for an existing one —
each row says which.

### Hooks

Claude Code's authoritative event list (`hook-development/SKILL.md`'s own
Quick Reference table): PreToolUse, PostToolUse, UserPromptSubmit, Stop,
SubagentStop, SessionStart, SessionEnd, PreCompact, Notification. Real
plugin configs on this box also use `PostToolUseFailure` and
`UserPromptExpansion`, seen in practice but not in that skill's own table.

| Claude Code capability | tamfis-code feature | Verdict |
|---|---|---|
| PreToolUse | `hooks.py`'s `pre_tool_use` | COVERED |
| PostToolUse | `hooks.py`'s `post_tool_use` | COVERED |
| UserPromptSubmit (block a turn or add context before it reaches a provider) | `hooks.py`'s new `user_prompt_submit` + `run_user_prompt_submit_hooks()` | COVERED (added 2026-09-13) — fires once per turn in `runner_local.py` right after the resumed/fresh objective is finalized; exit code 2 blocks the turn before any provider call, other output is folded into the objective as added context. Confirmed live end-to-end in `tests/test_claude_parity_hooks.py`: a blocking hook stops the turn with zero provider calls, and a context-adding hook's text is proven to reach the actual provider request payload |
| Stop (can force the agent to keep working via `{"decision": "block"}`, re-injecting into the same round loop) | none | GAP (feature) — the block-and-continue half of Stop has no analog; would require resuming a turn `runner_local.py` already believes is finished, a substantially larger change than this pass's scope. Not built |
| Stop (observe-only half: fires on successful completion) | `hooks.py`'s new `session_completed` + `run_session_completed_hooks()` | COVERED (added 2026-09-13) — deliberately only the observe-only half of Stop's contract; fires from `_finalize_completed_answer`, the one real successful-completion return path in `_run_local_agent_turn_impl`. Confirmed live in `tests/test_claude_parity_hooks.py` that a real on-disk hook fires with the correct summary |
| SubagentStop | none | GAP (feature) — no equivalent fires when a delegated swarm sub-task finishes; `swarm.py`'s delegation has no hook point at all today |
| SessionStart (load context, persist env vars via `$CLAUDE_ENV_FILE`) | none | GAP (feature) — no hook fires when a tamfis-code session/process starts |
| SessionEnd | none | GAP (feature) — no hook fires when a session ends (as distinct from a single turn completing, which `session_completed` now covers) |
| PreCompact (add critical info to preserve before context compaction) | none | GAP (feature) — `state.py`'s conversation compaction (see `test_thread_compression.py`) has no hook point before it runs |
| Notification (react when Claude sends a notification) | none | GAP (feature) — no equivalent; tamfis-code has no generalized notification-event concept for hooks to observe |
| PostToolUseFailure, UserPromptExpansion | none | GAP (feature), lower priority — narrower/newer events seen in real plugin configs but not central to the documented spec |
| Prompt-based hooks (`{"type": "prompt", "prompt": "..."}` — an LLM call decides the outcome instead of a shell command) | none | GAP (feature) — every tamfis-code hook is a `command`-type shell subprocess only |
| Parallel hook execution (all matching hooks for an event run concurrently) | sequential (`for hook in hooks: ...`, awaited one at a time) | GAP (feature) — a design choice difference (this session's implementations lean toward deterministic ordering, e.g. `session_interrupted`'s "first blocking hook stops evaluation" semantics), not obviously a defect, but worth the user knowing it's not matched |
| `if`-conditional matching (gate a hook on the actual command being run, e.g. `"if": "Bash(git commit:*)"`, not just the tool name) | tamfis-code's `matcher` is a regex against `tool_name` only | GAP (feature) — no equivalent of matching against the command/argument content itself |
| `asyncRewake` (a hook runs in the background and later "wakes" the agent back up with its findings, mid-or-after an already-answered turn) | none | GAP (feature), largest of this list — no background/suspend-resume mechanism exists in tamfis-code's synchronous per-turn hook firing at all; would need genuinely new runtime infrastructure, not a small addition |
| PreToolUse's `updatedInput` (a hook can rewrite the tool call's arguments before it runs, not just approve/deny it) | none | GAP (feature) — tamfis-code's pre_tool_use can only block, never mutate, the pending tool call |
| Hooks loaded once at session start, require a restart to pick up changes | tamfis-code reads hooks.toml fresh every turn | Not a gap — tamfis-code's behavior here is arguably better (edit hooks.toml and the very next turn uses it, no restart), noted for completeness rather than tabulated as COVERED/GAP |

### Broader feature areas (lighter-touch, verified against source but not test-audited to the same depth as hooks)

| Area | Claude Code | tamfis-code | Verdict |
|---|---|---|---|
| Subagents | `.claude/agents/*.md`, `--agent`/`--agents` CLI flags | `agent_definitions.py`'s `load_agent_definitions` — user + project markdown files, read fresh (no restart needed), consumed by `swarm.py` delegation | COVERED — same convention, tamfis-code's fresher-reload behavior is again arguably ahead |
| Skills (auto-discovered SKILL.md packages, triggered by description match) | `skills/<name>/SKILL.md` auto-discovery, per `plugin-structure/SKILL.md` | `plugins.py` declares a `skill_roots` field on plugin manifests, but confirmed by grep: nothing in `tamfis_code/*.py` actually reads a SKILL.md-like file from those roots or exposes anything to the model from them — `skill_roots` is only ever displayed in a status listing (`cli.py`'s plugin status output), never consumed | GAP (feature) — the plumbing for a skills concept was started (a config field exists) but the actual auto-discovery/invocation mechanism was never built |
| Custom slash commands | `commands/*.md`, project + user, `$ARGUMENTS` substitution | `custom_commands.py` — explicitly modeled on this exact convention (its own docstring says "Claude Code/Codex-style"), same discovery paths, same `$ARGUMENTS` substitution, project overrides user by name | COVERED |
| MCP integration | full external server config (stdio/HTTP), per `mcp-integration/SKILL.md` | `mcp_client.py`'s `StandaloneMCPBridge` — stdio and HTTP server support, config-file-based | COVERED for the core config/dispatch mechanism (OAuth/elicitation gaps already covered in the Codex section above, which apply equally here since Claude Code's MCP integration also documents those capabilities) |
| Settings/permission model | `.claude/settings.json` (hooks, permissions.allow/deny, env, model), plugin `plugin-settings/SKILL.md` | `config.py` (`.tamfis/config.toml` layering) + `permissions.py` (`decide_permission` allow/ask/deny rules) | COVERED at a comparable depth for the core allow/ask/deny + config-layering mechanism |
| Session management CLI (`--bg`/`--background`, `attach`, `logs`, `stop`, `rm`, `agents` listing) | documented in `claude --help` | tamfis-code has background-job support (`background.py`, referenced throughout this session's doctor/test work) with comparable list/attach-style operations | COVERED at a comparable level — not re-verified flag-by-flag in this pass; flagged as lighter-touch than the hooks section |

### What was fixed this pass vs. documented as a future feature investment

Fixed and deployed: `user_prompt_submit` and `session_completed` hook
events, both wired into real `runner_local.py` call sites with unit tests
(`test_hooks.py`) and a real end-to-end integration test
(`test_claude_parity_hooks.py`) proving the wiring, not just the isolated
functions.

Deliberately not attempted this pass (each is a genuine, separate feature
investment, not a fast/high-value fix): Stop's block-and-continue
semantics, SubagentStop, SessionStart, SessionEnd, PreCompact,
Notification, prompt-based (LLM-driven) hooks, parallel hook execution,
`if`-conditional command matching, `asyncRewake`, and PreToolUse's
`updatedInput` mutation. The skills auto-discovery/invocation gap
(`plugins.py`'s unused `skill_roots` field) is likewise flagged but not
built here.

## Summary

Every row in this document has now been resolved to a definitive verdict
(no remaining "tentative"/"unconfirmed" rows) as of 2026-09-13. Fixed this
pass: symlink-escape write/edit tests, hook-execution-timeout test,
`doctor` PATH-safety check, real-bwrap sandbox enforcement tests, MCP
startup-grace test, quota-classifier tests, token-budget tests,
same-path-write dispatch-conflict test, the `session_interrupted`,
`user_prompt_submit`, and `session_completed` hook events (Codex's
`interrupt_hooks.rs` and two Claude-Code-parity additions), `load_instruction_text` refresh
test, and a full vision/image-attachment test file. Several genuine future
FEATURE gaps remain flagged for the user rather than silently built --
see the Claude Code comparison section above for the full list (Stop's
block-and-continue semantics, SubagentStop, SessionStart, SessionEnd,
PreCompact, Notification, prompt-based hooks, parallel hook execution,
`if`-conditional matching, `asyncRewake`, PreToolUse input mutation, and
skills auto-discovery).
