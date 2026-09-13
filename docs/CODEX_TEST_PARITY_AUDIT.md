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
| `request_user_input*.rs` | `ask_user_question`-equivalent | GAP — standalone coverage unconfirmed |
| `request_permissions*.rs` | likely same mechanism as approvals | GAP — needs confirmation it's actually distinct before deciding N/A vs GAP |
| `permissions_messages.rs`, `catalog_permission_messages.rs` | none (inlined copy, not catalog-driven) | N/A |

## Hooks

| Codex category | tamfis-code feature | Verdict |
|---|---|---|
| `hooks.rs`, `hooks_executor.rs` | `hooks.py`, `HOOK_TIMEOUT_SECONDS` | COVERED (timeout-kill path fixed 2026-09-13) |
| `hooks_mcp.rs` (hooks calling MCP tools) | unconfirmed | GAP — needs confirmation hooks can invoke MCP tools at all |
| `interrupt_hooks.rs` (hooks on task interruption) | unconfirmed | GAP — needs confirmation |

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
| `quota_exceeded.rs` | `quota` handling in `providers.py`/`local_chat.py`/`runner_local.py` | GAP — no dedicated test found |
| `model_switching.rs`, `model_overrides.rs`, `model_runtime_selectors.rs`, `model_provider_requirements_tests.rs` | `model_registry.py`, `model_group_fallback` | COVERED (fallback path); override/selector edge cases unconfirmed |
| `stream_error_allows_next_turn.rs`, `stream_no_completed.rs` | stream-error handling | COVERED — `test_stream_reconnect.py` |
| `prompt_caching.rs`, `prompt_cache_key.rs` | none (no `cache_control` breakpoints implemented) | N/A — real feature gap, not a test gap |
| `token_budget.rs`, `token_usage_rollout.rs` | token-budget concept in `runner_local.py` | GAP — unclear if token-level (vs. round-level) accounting is covered by `test_round_budget_extension.py` |

## Tool execution mechanics

| Codex category | tamfis-code feature | Verdict |
|---|---|---|
| `unified_exec*` family (one PTY+approval+stdin-review tool) | `pty.py`/`local_pty.py` + separate `execute_command` (two tools, different architecture) | N/A (borderline) — literal port doesn't make sense; underlying scenarios (approval mid-command, stdin review) not confirmed tested either way |
| `tool_parallelism.rs` (concurrent tool_calls in one turn) | unconfirmed | GAP — concurrent tool-call ordering/semantics not confirmed tested |
| `tool_lifecycle.rs`, `tool_harness.rs` | distributed across many tool-specific test files | COVERED |
| `turn_state.rs`, `turn_input_submission.rs`, `pending_input.rs`, `direct_tool_metadata.rs` | unconfirmed | GAP — needs investigation before classifying |

## Context / prompt engineering

| Codex category | tamfis-code feature | Verdict |
|---|---|---|
| `agents_md.rs`, `agents_md_refresh.rs` | `workspace.py` `load_instruction_text`/`_instruction_chain` | GAP — hot-refresh mid-session (re-reading after a live edit) not confirmed tested |
| `additional_context.rs`, `context_annotations.rs`, `current_time_reminder.rs`, `personality.rs`, `collaboration_instructions.rs`, `git_enrichment.rs` | not found | N/A — needs one more confirmation pass before fully closing |
| `truncation.rs` | truncation logic in `providers.py`/`runner_local.py` | GAP — unclear which test covers this specifically |
| `audio_truncation.rs` | none (audio not a real input modality) | N/A |
| `view_image.rs` | vision/image_content_blocks support across `cli.py`/`render.py`/`providers.py`/`mcp.py`/`runner_local.py` | GAP — feature clearly exists, no dedicated test file found |
| `web_search.rs`, `search_tool.rs` | both implemented | COVERED |

## Multi-agent / delegation

| Codex category | tamfis-code feature | Verdict |
|---|---|---|
| `fork_thread.rs` | session fork | COVERED — `test_session_fork.py` |
| `multi_agent_resume.rs`, `multi_agent_mode.rs`, `codex_delegate.rs`, `spawn_agent_description.rs` | swarm delegation | COVERED — `test_swarm.py`, `test_agents_delegation.py` |
| `subagent_notifications.rs`, `subagent_service_tier.rs` | "subagent" referenced across `swarm.py` etc. | GAP — notification/service-tier semantics specifically unconfirmed |

## CLI subcommands

| Codex category | tamfis-code feature | Verdict |
|---|---|---|
| `login.rs`, `auth_refresh.rs` | `tamfis-code login`, `config.py`/`api_client.py` | GAP — refresh-on-401 behavior not confirmed tested |
| `device_code_login.rs`, `login_server_e2e.rs`, `logout.rs` | none (no OAuth/device-code flow) | N/A |
| `doctor_path_safety.rs` | `check_path_safety()` | COVERED (added 2026-09-13) |
| `doctor_enterprise_network.rs` | none | N/A — out of scope |
| `mcp_add_remove.rs`, `mcp_list.rs`, `mcp_login.rs` | config-file-based MCP server config (not imperative CLI) | COVERED via config-loading tests; imperative-CLI surface itself N/A (doesn't exist) |
| `features.rs`, `queue.rs`, `delete.rs`, `debug_models.rs`, `debug_clear_memories.rs` | unconfirmed | GAP — needs investigation before classifying |

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
