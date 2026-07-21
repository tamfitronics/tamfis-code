# tamfis-code status

This file is the single source of truth for tamfis-code's current state. It
is **updated in place** as things change -- it does not get superseded by a
new dated snapshot next time. If you're reading a dated report elsewhere
(`TAMFISGPT_PARITY_AUDIT_*.md`, `*_Gap_Report.md`, `*_Completion_Report.md`
in the parent directory) instead of this file, treat it as historical only:
those are point-in-time snapshots that go stale the moment code changes, and
that staleness has already caused wasted work once (2026-07-17 -- work was
nearly re-planned off two reports dated 2026-07-12 and 2026-07-16, both
already superseded by same-day code changes).

**Rule going forward: don't create a new dated audit file for tamfis-code.
Update this one.**

## Post-release CWD scope, live input, plan progress, model routing, evidence, and fallback fixes (2026-07-22, working tree)

Fixed the remaining plan-visibility gap identified after v0.6.1:

- `plan_created` and every `plan_step_progress` event now print a durable
  scrollback snapshot instead of relying on Rich's transient Live region.
- Each visible item now shows an explicit status: `pending`, `in_progress`,
  `completed`, or `failed`.
- Progress remains visible after assistant streaming starts and the spinner is
  stopped.
- Remote `plan_step_progress` events are now persisted into the active saved
  plan, matching the local orchestrator path.
- Added a regression test covering plan creation, Live shutdown, and a later
  completed/in-progress transition.
- Automatic coding routes no longer default to NVIDIA's general
  `meta/llama-3.1-70b-instruct`, which repeatedly produced narrated tool
  claims in this workflow. NVIDIA now defaults to the verified
  `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` route; OpenRouter's paid
  coding route now defaults to `qwen/qwen3-coder`. TamfisGPT subscription
  routing remains preferred when `TAMFIS_API_KEY` is configured.
- Kimi remains listed only as an explicit selectable model; it is not an
  automatic route and cannot spend credits unless the user chooses it.
- The fabricated-result guard now rejects transcript-style claims such as
  `list_directory tool has found ...` and `execute_command tool has executed
  ...` unless real tool-call evidence exists. Added direct regression tests.
- Final completions now use a compact `Summary / Changes / Verification /
  Remaining issues` contract with flat, non-nested bullets; durable plan and
  progress panels remain the single source of truth for execution tracking.
- HTTP 400 responses containing NVIDIA's `DEGRADED function cannot be
  invoked` route error are now retryable, so AUTO switches to the next
  eligible model/provider instead of stopping with a checkpointed failure.
  Unrelated HTTP 400 errors remain non-retryable.
- Task scope now honors explicit existing absolute project paths in the user
  objective, so a launch from an admin checkout cannot silently constrain a
  request for `/home/tamfisseo` to `/home/tamfisgpt`.
- Ctrl+T live input now detects the control byte even when the terminal groups
  it with adjacent bytes, persists the queued instruction, acknowledges its
  queue ID, and leaves it for the next safe agent-round boundary.

Verification after this fix: **854 tests passed** with an isolated writable
config directory (3 existing collection/deprecation warnings). These changes
are assigned to release **0.6.4**; GitHub/PyPI publication is the remaining
release step for this working tree. Earlier release tags are not rewritten.

## Release gate and live queue UX (2026-07-21, v0.6.1)

Commercial-readiness verification completed for the portable standalone
runtime. The TamfisGPT subscription provider is now the preferred standalone
route when `TAMFIS_API_KEY` is configured; model traffic stays behind the
subscription API while workspace files, commands, PTY/TTY sessions,
approvals, plans, state, and mutation evidence remain local.

The live terminal path now supports a real `Ctrl+T` `queue next>` editor while
a task is streaming. Submitted input is shown back to the user with a durable
queue ID, concurrent editors are prevented from competing for stdin, and
queued follow-ups are applied at the next safe agent-round boundary. The
existing second-terminal `tamfis-code queue` path remains compatible.

Also fixed in this release: durable plan snapshots surviving Rich Live
shutdown, local PTY lifecycle and bounded output handling, standalone client
tool-call contracts, narrated fake-tool recovery, release test discovery, and
package documentation for installation and subscription configuration.

Verification completed:

- Tamfis-Code full suite: **850 passed** with isolated writable config state.
- TamGPT6 OpenAI-compatibility/admin API-key tests: **8 passed**.
- Release build: `tamfis_code-0.6.1.tar.gz` and
  `tamfis_code-0.6.1-py3-none-any.whl` built successfully with isolated
  build dependencies.
- The source release includes `USAGE_INSTALL_RELEASE.md` through
  `MANIFEST.in`.

Deployment status: GitHub repository `tamfitronics/tamfis-code` now has the
verified `agent/commercial-release-0.6.1` branch and draft PR #1. The
`v0.6.1` GitHub Actions Trusted Publishing workflow completed successfully,
and `tamfis-code==0.6.1` is publicly available on PyPI. There is no matching
`tamfitronics/tamgpt6` repository. No PyPI secret was required or stored.

Current release record:

- Release commit: `72b9382` (`Release commercial standalone runtime 0.6.1`).
- Status follow-up commit: `b804473` (`Update release status after PyPI deployment`).
- Release tag: `v0.6.1`.
- Draft PR: [tamfitronics/tamfis-code#1](https://github.com/tamfitronics/tamfis-code/pull/1).
- Successful deployment run:
  [GitHub Actions run 29875141492](https://github.com/tamfitronics/tamfis-code/actions/runs/29875141492).
- Published package:
  [PyPI tamfis-code 0.6.1](https://pypi.org/project/tamfis-code/0.6.1/).
- The first CI attempt exposed that `rg` is not guaranteed on hosted/minimal
  systems. `mcp.py` now falls back to bounded standard-library search, and
  the corrected CI run passed before publishing.
- Published-checkout CI passed **869 tests**; the local source checkout passed
  **850 tests** with an isolated writable config directory. The difference is
  the repository's retained historical Ollama-only test, which was removed
  from the release branch because Ollama is not a supported provider in the
  current provider registry.
- TamGPT6 backend validation passed **8 tests** covering the OpenAI-compatible
  client-tool path and admin API-key lifecycle.
- The release artifact contains local tools, PTY/TTY support, approvals,
  planning, durable state, live queue input, subscription routing, hooks,
  swarm/delegation, OpenHands-compatible runtime pieces, and the client-side
  tool execution contract.
- GitHub token credentials were used only through environment-backed commands;
  no token was committed. The token was exposed once in a command-generated
  remote URL during the initial push and must be revoked/rotated by the
  account owner.

Known deployment boundary: the GitHub branch/PR is still a draft and requires
the repository owner to review and merge it. PyPI 0.6.1 is already published;
future releases should use the existing `publish.yml` Trusted Publishing
workflow and a new version tag.

## Live-reported: execute_command's `environment` argument also crashed the same way (2026-07-21, v0.4.42)

Direct follow-up to v0.4.41 below, same session: user pasted another live crash, `'str' object has no attribute 'items'`. Same root cause, same tool, different parameter: `_execute_command`'s `environment: Optional[Dict[str, str]] = None` is only a type hint, and `env.update({str(k): str(v) for k, v in environment.items()})` assumed a real dict unconditionally. A real tool call sent `environment` as something other than an actual object (most likely a JSON-encoded string), and `.items()` crashed exactly as reported.

Given this is now the *second* type-mismatch crash found in the same handler in one session, did a full sweep of every other tool handler in `mcp.py` for the same pattern (any non-string-typed parameter used without runtime coercion) before shipping just the one-line fix: `write_file`/`edit_file`/`list_directory`/`search_code`/`find_references`/`extract_archive`/`repackage_archive`/`get_git_info` only take string parameters (no risk); `web_search`'s `max_results` was already defensively coerced (`build web_search`, v0.4.33); `ask_user_question`'s `options` degrades gracefully (enumerates whatever it's given) rather than crashing; `browser`'s numeric/object fields delegate to an external `BrowserTool` outside this package, out of scope. `execute_command`'s `timeout` and `environment` were the only two real crash risks, and both are now fixed.

Fixed the same way as `timeout`: `environment` is only applied when `isinstance(environment, dict)` is true; anything else is treated as "no override" instead of crashing the whole command.

Full suite: 742/742 passing (was 741, +1 new). Live-verified against the real installed package with both malformed arguments at once (`environment: "{}"`, `timeout: "300"`) -- runs cleanly. Version bumped to `0.4.42`, rebuilt, force-reinstalled, diff-verified identical, `tamfis-code --version` confirmed on the real `$PATH` binary.

## Live-reported: mode switch invisible mid-task, and a real execute_command crash (2026-07-21, v0.4.41)

User pasted a long real transcript. Two distinct, confirmed bugs in it, both fixed:

**1. Mid-task mode switch (Shift+Tab) wasn't visible the way Claude Code shows it.** `live_input.py`'s `_cycle_mode()` only ever printed a one-time scrolling `◆ Mode switched to X` diagnostic line -- exactly reproduced in the transcript (`· diagnostics: ◆ Mode switched to auto...` then `...plan...`), which the next few lines of tool/streaming output immediately pushed off-screen. Unlike Claude Code, where the current mode is a persistent, always-visible part of the UI, there was no ongoing indicator during a run -- only the idle REPL prompt showed `[mode]`, which isn't visible while a task is actually streaming.

Fixed by folding the mode label into the SAME persistent Live status line the spinner/elapsed-time/tip text already occupies: `StreamRenderer.__init__` gained an optional `mode_label` param (every existing non-interactive caller omitting it is unaffected -- no tag shown), `_build_status()` prepends `[mode]` when set, and a new `set_mode_label()` method updates it and forces an immediate `_refresh_live()` so a switch is visible the instant it happens. `live_input.py`'s `_cycle_mode()` now calls both the existing diagnostic line (durable transcript record) AND `set_mode_label()` (immediate persistent visibility). Wired at all 3 real interactive-turn `StreamRenderer` construction sites in `interactive.py` (main turn dispatch, `_run_saved_plan`, `/retry`'s standalone path) with the session's current `approval_policy` so the tag is present from turn start, not just after the first switch.

**A real markup bug caught by the test suite, not by eyeballing it**: the first implementation built the tag as `f"[cyan][{label}][/cyan]"` and fed it straight to `Text.from_markup()` -- rich parses `[accept-edits]` itself as an (unrecognized, silently-dropped) markup tag, not literal bracket text, so the entire tag vanished with no error. Fixed by escaping the literal opening bracket (`\\[`).

**2. Real crash: `execute_command` broke on both `npm test` and `npm run build`** in the same transcript -- `RuntimeWarning: coroutine 'Process.communicate' was never awaited` immediately followed by `✗ Ran command / '<=' not supported between instances of 'str' and 'int'`. Root cause: `mcp.py`'s `_execute_command(..., timeout: int = 60, ...)` -- a type hint only, never enforced -- received a real tool call with `"timeout": "300"` (a **string**; visible in the transcript's own approval panel), which reached `asyncio.wait_for(proc.communicate(), timeout=timeout)` unmodified. `wait_for`'s own internal `timeout <= 0` check then raised the exact `str`/`int` comparison `TypeError` *before* it ever awaited the `proc.communicate()` coroutine it had already created -- explaining both symptoms as one root cause, not two. A model outputting a numeric tool argument as a string is a common, not exotic, tool-calling failure mode.

Fixed with defensive coercion at the top of `_execute_command`: `int(timeout)` with a fallback to the 60s default on `TypeError`/`ValueError` or a non-positive value.

**Not chased this pass, needs more information**: the user separately flagged `npm run build` running in what looked like the wrong directory (`cwd: "/home/tamfisseo"`). The manifest-check safeguard (`manifest_rules`) would have refused the run outright if that directory had no `package.json` -- since it proceeded (into the crash above, not a manifest-refusal), a real `package.json` does exist there, so this reads as the model choosing a plausible-but-wrong cwd within the user's actual layout rather than a code-level cwd-resolution bug. Flagged, not fixed -- needs the user's real directory structure to diagnose further.

Full suite: 741/741 passing (was 735, +6 new: 3 mode-label tests in `test_tamfis_code_render.py`, 3 `execute_command` string/invalid/non-positive-timeout tests in `test_mcp.py`). Live-verified both fixes against the real installed package (not source tree): a real `"timeout": "300"` string call now runs cleanly instead of crashing, and `set_mode_label()` correctly updates the rendered status line text. Version bumped to `0.4.41`, rebuilt, force-reinstalled, diff-verified identical, `tamfis-code --version` confirmed on the real `$PATH` binary.

## kimi-k2.6 also registered on OpenRouter and HF (2026-07-21, v0.4.40)

Direct follow-up to the v0.4.39 fix below. User asked to check whether kimi-k2.6 is available through other providers rather than only routing around NVIDIA's account-entitlement gap -- confirmed live on both:

- **OpenRouter**: `moonshotai/kimi-k2.6` (same id/casing as NVIDIA) is present in OpenRouter's real `/v1/models` catalog, and a real chat-completions call against it returned a genuine `402 insufficient credits` -- i.e. OpenRouter itself accepts and would run the model; this account simply has no balance right now, a materially different (and better) signal than NVIDIA's `404 Not Found for account`.
- **Hugging Face**: `moonshotai/Kimi-K2.6` returns a real `200` with a genuine completion. Casing is load-bearing here -- confirmed live that the lowercase `moonshotai/kimi-k2.6` (NVIDIA's own id casing) 400s with `model_not_found` on HF's router; only the differently-cased id resolves.

Added both to `providers.py`'s respective `models` lists (not as either provider's `default_model` -- this is additional selectable inventory, not an AUTO-routing behavior change) and a matching second entry in `model_registry.py` keyed on the HF casing. Live-verified on the real installed `$PATH` binary: `tamfis-code --approval auto ask "say OK and nothing else" --provider hf --model moonshotai/Kimi-K2.6` completed with a real `OK` from the real model.

Full suite: 735/735 passing (was 733, +2 new). Version bumped to `0.4.40`, rebuilt, force-reinstalled, diff-verified identical, `tamfis-code --version` confirmed on the real `$PATH` binary.

## Live-reported: NVIDIA default model (kimi-k2.6) was a hard 404 for a real account (2026-07-21, v0.4.39)

User pasted a real failure straight from the running CLI: `Task failed: Provider streaming failed on nvidia / moonshotai/kimi-k2.6: Error code: 404 ... "Function '...': Not found for account '...'"`. v0.4.32 had deliberately switched NVIDIA's `default_model` to `moonshotai/kimi-k2.6` to avoid nemotron's occasional lexical-loop corruption -- confirmed live (real HTTP calls against the real account) that kimi-k2.6 is a **hard, permanent 404**, not a transient outage: it's listed in NVIDIA's general `/v1/models` catalog (200, present) but this account has no actual deployment/entitlement to invoke it (`nvidia/nemotron-3-super-120b-a12b` separately hit a transient 429 in the same check, `meta/llama-3.1-70b-instruct` returned a clean real `200` with a real completion). Root cause: a third-party marketplace NIM model commonly needs separate account enablement that NVIDIA's own first-party models don't, and this ordinary API key never had it.

Fixed in `providers.py`: NVIDIA's `default_model` changed to `meta/llama-3.1-70b-instruct` (confirmed live before switching, not guessed), `kimi-k2.6` demoted in the `models` list (still selectable, not removed from the catalog -- it may work fine for an account that does have entitlement). Also added `"not found for account"` to `is_retryable_provider_error`'s message-marker list -- this is an account-entitlement failure, not a bad request, so AUTO mode should fall back to the next candidate provider/model instead of hard-failing the whole turn if this specific NIM error shape is ever hit again (e.g. a user explicitly pins a model they don't have access to).

Live-verified the exact fix: `tamfis-code --approval auto ask "say OK and nothing else" --provider nvidia` on the real installed `$PATH` binary now resolves to `meta/llama-3.1-70b-instruct` and completes cleanly, reproducing and closing the user's exact reported failure.

Full suite: 733/733 passing (was 731, +2 new: `test_nvidia_default_model_is_not_loop_prone_nemotron_or_unentitled_kimi` replacing the now-stale `test_nvidia_uses_kimi_instead_of_loop_prone_nemotron_by_default`, plus `test_nim_account_entitlement_404_is_retryable`/`test_plain_404_without_the_entitlement_shape_is_not_retryable` for the new marker -- the latter guards against over-broadly treating every 404 as retryable). Version bumped to `0.4.39`, rebuilt, force-reinstalled, diff-verified identical, `tamfis-code --version` confirmed on the real `$PATH` binary.

## Gated plan mode -- a real "execute this plan now?" checkpoint (2026-07-21, v0.4.38)

Sixth and final item off the 2026-07-20 orchestration parity audit (see v0.4.33-37 entries below) -- closes the punch list entirely. `/plan` mode already ran entirely read-only and saved the resulting plan (`local_state.save_plan`, `/plans`, `/execute-plan`), but nothing ever asked the user whether to execute it: they had to notice the "run /execute-plan <id>" hint text and manually type a second, separate command. That's prompt convention, not a structural gate -- Claude Code's Plan Mode asks "would you like to proceed?" the moment the plan is ready.

Refactored the `/execute-plan` dispatch body (previously inlined under `elif intent.kind == "saved_plan":`) into a shared `_run_saved_plan(plan_id)` closure inside `run_interactive` -- both `/execute-plan` and the new gate call through the exact same real execution path (no second, potentially-drifting copy). Right after a plan-mode turn completes and the plan is saved, the REPL now asks **"Execute this plan now? (y/N)"** via `console.input()` (same blocking-input-inside-async pattern `mcp.py`'s `_ask_user_question` already established) -- gated on `console.is_terminal` so a non-interactive/piped session gets the old "run /execute-plan" hint instead of hanging on a prompt nothing will ever answer. Approving calls `_run_saved_plan` immediately with the exact plan just proposed (not a fresh prompt); declining leaves it saved exactly as before, with `/execute-plan`, `/plans`, or `/plan` again (to revise) all still available.

Live-verified in two parts: a real pty-driven session (`pexpect`) against a live NVIDIA NIM completion, and a direct call confirming a plan-shaped read-only objective genuinely reaches `status: "completed"` with a real non-empty summary against the real provider (`read_only=True` correctly denied a write_file attempt the model incorrectly tried on a differently-phrased, execution-flavored objective in an earlier pty run -- a pre-existing model-behavior characteristic having nothing to do with this change, not a defect in the gate). The REPL-level integration coverage (below) is the real proof of the gate logic itself, run against a stubbed provider so it's deterministic.

Full suite: 731/731 passing (was 728, +3 new: `tests/test_plan_mode_gate.py`, driving the real `run_interactive` REPL loop end-to-end with `run_local_agent_turn` stubbed and a real `rich.console.Console` forced into terminal mode -- approving triggers a real second execution call with `saved_plans[0]["status"] == "completed"`, declining makes zero second call with status staying `"ready"`, and an ordinary non-plan-mode turn is confirmed to never trigger the gate at all). Documented in `README.md`'s new "Plan mode" section. Version bumped to `0.4.38`, rebuilt, force-reinstalled, diff-verified identical (only the pre-existing harmless `workspace.py.bak`), `tamfis-code --version` confirmed on the real `$PATH` binary.

**This closes the entire 2026-07-20 orchestration parity audit** (see [[feedback_tamfis_code_rival_orchestration_directive]] in memory) -- all 6 items done: web_search + RESEARCH routing (v0.4.33), vision/image attachments (v0.4.34), pre/post-tool-use hooks (v0.4.35), custom slash commands (v0.4.36), declarative subagent types (v0.4.37), gated plan mode (v0.4.38). The standing directive itself has no finish line (keep closing gaps opportunistically as they're found in future sessions), but this specific audit's punch list is fully closed.

## Declarative subagent types (2026-07-21, v0.4.37)

Fifth item off the same audit (see v0.4.33-36 entries below): delegation already existed for real (`agents.py`'s `DelegatedCodingAgent`, `swarm.py`'s hardened fan-out, the model-callable `delegate_parallel_tasks` tool) but only as ad-hoc task strings -- every sub-task shared one model/provider and got no specialised instructions, unlike Claude Code's `.claude/agents/*.md` named subagent definitions.

Added `tamfis_code/agent_definitions.py`: a markdown file in `~/.config/tamfis-code/agents/<name>.md` (every session) or `<project>/.tamfis/agents/<name>.md` (one project, replaces same-named user definition outright -- same override-not-merge precedent as v0.4.36's custom commands, for the same reason) declares a named subagent type: optional frontmatter `description`/`model`/`provider`, body is a system-prompt prefix.

Wired through the existing primitives rather than reimplementing them: `DelegatedCodingAgent` gained an `extra_system_prompt` param (prepended as a real system message ahead of the sub-task's objective, not merged into it). `AgentManager.execute_tasks` gained an optional `agent_types: List[Optional[str]]` parallel to `descriptions` (every existing caller omitting it is completely unaffected) -- when a name resolves to a real definition, that ONE sub-task's model/provider (via `local_chat.resolve_provider_type`) and system prompt are overridden for just that sub-task; an unknown name is a no-op, not a failure, so one bad name can't take down the whole fan-out. `swarm.run_swarm` forwards the same param through. `SWARM_TOOL_SCHEMA`'s `tasks` items gained an `anyOf` (plain string, unchanged, OR `{"objective", "agent_type"}` object) so the model itself can pick a subagent type per sub-task when it calls `delegate_parallel_tasks` mid-turn -- this is what actually closes the parity gap (the orchestrating model deciding which specialised subagent handles which piece of work), not just a human pre-scripting it. `/delegate --agent <name> ...` and `/swarm --agent <name> ...` apply one definition to every task in that REPL invocation (the simpler CLI-flag shape; per-task selection is the tool-call path's job). New `/agent-types` REPL command lists what's loaded.

Live-verified against the REAL installed package (`sys.path` stripped of the source checkout before import): a real `.tamfis/agents/reviewer.md` (frontmatter setting `model: moonshotai/kimi-k2.6`, `provider: nvidia`) was discovered and, through the real `execute_tasks` path, correctly resolved `provider` to the real `ProviderType.NVIDIA` enum member (not just a string), overrode `model`, and applied the system prompt -- confirmed by inspecting the actual kwargs `DelegatedCodingAgent.__init__` received.

Full suite: 728/728 passing (was 713, +15 new: `tests/test_agent_definitions.py` module coverage, 4 new `ExecuteTasksAgentTypesTests` in `test_swarm.py` covering override/fallback/mixed-list/prompt-only cases, 4 new `_parse_swarm_tasks` tests in `test_runner_local.py`). Documented in `README.md`'s new "Declarative subagent types" section. Version bumped to `0.4.37`, rebuilt, force-reinstalled, diff-verified identical (only the pre-existing harmless `workspace.py.bak`), `tamfis-code --version` confirmed on the real `$PATH` binary.

**This closes every item from the 2026-07-20 audit except plan-mode gating** (see [[feedback_tamfis_code_rival_orchestration_directive]] for the standing tracker) -- plan mode remains prompt convention (`should_plan`/`_attempt_reasoning_plan` producing a `TASK PLAN` system message) rather than a distinct gated propose→user-approves→execute phase; not touched this pass.

## User-defined custom slash commands (2026-07-21, v0.4.36)

Fourth item off the same audit (see v0.4.33/34/35 entries below): `SLASH_COMMANDS` (`interactive.py`) was a hardcoded Python tuple -- no way for a user to add a `/command` of their own without editing source, unlike Claude Code/Codex's custom-prompt mechanism.

Added `tamfis_code/custom_commands.py`: drop a markdown file into a commands directory and get a new `/<name>` command on the next REPL turn, no code changes. `~/.config/tamfis-code/commands/<name>.md` (platform-native location, same resolver `config.py`/`hooks.py` already use) applies every session; `<project>/.tamfis/commands/<name>.md` applies to one project. Unlike hooks.py (where user and project hooks both fire), a project command with the same name REPLACES the user one -- matches how a project `.tamfis/config.toml` layer already overrides user config, since a project-specific command should win outright, not run alongside a same-named personal one. File shape: filename (minus `.md`) is the command name; an optional `---\ndescription: ...\n---` frontmatter block sets the description shown in `/commands` and tab-completion; the rest of the file is the prompt template. `$ARGUMENTS` in the template is replaced with whatever the user typed after the command name; a template with no `$ARGUMENTS` placeholder still gets the typed text appended on a new line so it's never silently dropped.

Wired into `interactive.py`: `parse_intent` gained an optional `custom_commands` dict, checked last (every built-in slash command -- `/plan`, `/agent`, etc. -- always wins on a name collision) before the final plain-AI-objective fallback. `_SlashCommandCompleter` now also accepts a `custom_commands` dict for tab-completion, and it's the SAME dict object the REPL's main loop mutates in place (`clear()` + `update()`) every turn via a fresh `load_custom_commands()` call -- so both dispatch and tab-completion pick up a newly added/edited command file on the very next turn without restarting the process or rebuilding the completer. New `/commands` REPL command lists what's currently loaded (name/description/source), matching the existing `/tools`/`/agents` discoverability pattern.

Live-verified against the real installed package (not the source tree -- `sys.path` explicitly stripped of the tamfis-code checkout before import): a real `.tamfis/commands/greet.md` with frontmatter was discovered, its description read correctly, and `/greet Ada` expanded through `parse_intent` into the real templated objective with `$ARGUMENTS` substituted.

Full suite: 713/713 passing (was 696, +17 new: `tests/test_custom_commands.py` module coverage, 4 new `parse_intent` tests in `test_tamfis_code_intent.py` including the built-in-wins-on-collision guarantee, 3 new completer tests in `test_interactive_standalone.py` including the live-mutation-without-rebuild contract). Documented in `README.md`'s new "Custom commands" section. Version bumped to `0.4.36`, rebuilt, force-reinstalled, diff-verified identical (only the pre-existing harmless `workspace.py.bak`), `tamfis-code --version` confirmed on the real `$PATH` binary.

## Real pre/post-tool-use hooks, settings-file driven (2026-07-20, v0.4.35)

Third item off the same audit (see the v0.4.33/v0.4.34 entries below): tamfis-code had no equivalent at all to Claude Code's PreToolUse/PostToolUse hook mechanism -- zero hits for "hook" anywhere in the source before this.

Added `tamfis_code/hooks.py`: hooks are arbitrary shell commands configured in `hooks.toml` (no code changes required) -- `~/.config/tamfis-code/hooks.toml` (platform-native location, same resolver as everything else in `config.py`) applies to every session, and/or `<project>/.tamfis/hooks.toml` applies to one project; both apply together, project hooks running after user hooks, matching `config.py`'s own layering precedent. Each `[[pre_tool_use]]`/`[[post_tool_use]]` entry has an optional `matcher` (regex against the tool name; empty matches every tool) and a `command`. A hook receives the event as JSON on stdin (`event`, `tool_name`, `tool_input`, `tool_output` for post_tool_use, `session_id`, `workspace_root`). `pre_tool_use`: exit code 2 blocks the call before mcp.py ever executes it -- the tool is never run, and the hook's stderr (falling back to stdout) becomes the denial reason the model sees, the same shape an approval denial already uses. Any other exit code doesn't block. `post_tool_use`: the tool already ran by the time this fires, so nothing can undo it -- stderr/stdout always just surfaces as additional context appended to the conversation for the model to see. A hook that fails to start, errors, or times out (30s cap) degrades to a visible diagnostic instead of failing the turn, matching the "never let an optional integration point take down a real turn" contract `mcp.py`'s `_import_monorepo_attr` already established for browser/the shared MCP bridge.

Wired into `runner_local.py`'s single canonical tool-dispatch site (the one place every real local tool call in the standalone loop actually executes): hooks are loaded once per turn (fresh, not cached across turns, so editing `hooks.toml` takes effect on the next turn without a restart), fire immediately before/after `mcp_server.call_tool(...)`, and both directions emit a `diagnostics` render event so hook activity is visible in the terminal, not silent.

Live-verified two ways: a full integration test through `run_local_agent_turn` with a fake LLM client but a REAL spawned hook subprocess (not mocked) proving both the block and the feedback-injection paths actually work, and a real end-to-end run against the actual installed `$PATH` binary with a real NVIDIA NIM completion (`meta/llama-3.1-70b-instruct`, after `moonshotai/kimi-k2.6` 404'd and `nvidia/nemotron-3-super-120b-a12b` hit a transient 429) -- a project `.tamfis/hooks.toml` blocking `execute_command` correctly stopped a real `echo hello123` request server-side-approved-but-hook-vetoed, with the hook's stderr message visible in the terminal exactly where expected.

Full suite: 696/696 passing (was 680, +16 new: `tests/test_hooks.py` unit coverage for the module in isolation, plus 2 integration tests in `test_runner_local.py`). Documented in `README.md`'s new "Hooks" section. Version bumped to `0.4.35`, rebuilt, force-reinstalled, diff-verified identical (only the pre-existing harmless `workspace.py.bak`), `tamfis-code --version` confirmed on the real `$PATH` binary.

**Deliberately scoped down from Claude Code's full hook surface**: only `pre_tool_use`/`post_tool_use` (the highest-value pair) -- no `SessionStart`/`Stop`/`UserPromptSubmit`/notification-style hooks in this pass. `pre_tool_use` fires after the existing approval-decision flow, not before it (a human still gets asked to approve a mutating call even if a hook would go on to block it) -- interleaving hooks with the approval prompt itself would have meant touching `resolve_approval_decision_async`'s internal logic, a larger, riskier change deliberately not bundled here. Both are real, honest scope choices, not silent gaps -- flagged for a future pass if a real need for either surfaces.

## Real vision/image content for --attach in the standalone loop (2026-07-20, v0.4.34)

Second item off the same fresh orchestration-parity audit (see the v0.4.33 entry below). Audit found `--attach`ed images never actually reached the model as pixels: `cli.py`'s `_run_local_ai_command` turned every attachment (image or not) into a single plain-text system note ("use read_file for text..."), and `mcp.py`'s `_read_file` would then `read_text(encoding='utf-8', errors='ignore')` a binary PNG/JPEG, silently dropping every invalid byte and handing the model plausible-looking garbage instead of an error or the real image.

Added real OpenAI-style multipart vision content: `cli.py` now splits `--attach`ed files into images (`.png`/`.jpg`/`.jpeg`/`.gif`/`.webp`) vs everything else, base64-encodes images into `image_url` content blocks (`runner_local.build_vision_content_blocks`), and passes them into `run_local_agent_turn(..., image_content_blocks=...)`. Inside the loop, a new `_messages_with_vision_content` helper splices those blocks into the current user message **only for the actual provider call**, gated per-call on that route's real `ProviderConfig.vision_supported` flag (HF/OpenRouter/Ollama; not TIER_IV/NVIDIA) -- routes without vision support never receive the image, only the existing plain-text path note.

**Deliberate design constraint, confirmed against the codebase before writing code**: the canonical `working_messages` list threading through the whole turn is never permanently rewritten into multipart form. At least 6 call sites in `runner_local.py` (resume/anchor matching, `_legacy_resume_messages`, `_latest_user_text`, dedup) do `str(message.get("content") or "")` against user messages and compare it to plain objective text -- persisting a list-typed `content` into `working_messages` itself would have silently broken resume matching for any turn that included an image. Instead the multipart form is rebuilt fresh from the plain-text canonical message at each of the two `_stream_completion_with_reconnect` call sites (primary route + AUTO fallback candidate), which is what a stateless chat-completions request needs anyway -- the full history is resent every round regardless.

**A real placement bug caught by a live-traced repro, not by the mocked unit tests**: the first implementation computed `vision_message_index` immediately after `working_messages` was built, but `working_messages.insert()` for the scope-instruction message (and, when `should_plan()` fires, the grounded-plan message) both insert **before** the user objective, shifting its real index by 1 or 2 after the fact. The mocked `_FakeClient`-based test caught this immediately (`content[0]` came back as `'w'` -- indexing into the still-plain-text string, proving the splice silently no-op'd). Fixed by moving the index computation to immediately before the round loop starts, after every `.insert()` has already happened; later mutations are all `.append()`, which never shifts an earlier index.

Also fixed a related pre-existing correctness gap while auditing `_read_file`: it now sniffs the first 8000 bytes for a null byte (the same binary-detection heuristic `file`/git use) and returns a clear "looks like a binary file" error instead of silently returning `errors='ignore'` mangled garbage -- applies to any binary file, not just images.

Live-verified with a real generated PNG (400x200, black rectangle outline + red "TAMFIS42" text) against real configured vision routes (OpenRouter `google/gemini-2.5-flash`, HF `Qwen/Qwen2.5-VL-72B-Instruct`, the latter looked up live via HF's own `/v1/models` endpoint since the two vision model names already hardcoded in `providers.py` for HF turned out to be stale/unavailable for this token). **Both real accounts are out of credits in this environment** (OpenRouter HTTP 402 insufficient credits, HF HTTP 402 depleted monthly credits) -- a genuine billing limitation, not a code defect: confirmed by tracing the actual outgoing request right before the network call and seeing a correctly-shaped real base64 PNG data URI inside a proper `[{"type": "text", ...}, {"type": "image_url", ...}]` block, and by both providers returning a well-formed billing error (not a request-validation error), meaning the payload was accepted as valid far enough to be billed. Re-ran the same trace against the real installed `$PATH` binary via `tamfis-code ask --attach` -- identical clean, non-crashing billing error, confirming the deploy matches source-tree behavior. A full successful round-trip (model actually describing the image back) was not obtainable in this environment for that reason; noted honestly rather than claimed.

Full suite: 680/680 passing (was 670, +10 new tests: binary-guard regression in `test_mcp.py`, `VisionContentTests` pure-function coverage and 2 `RunLocalAgentTurnTests` integration tests in `test_runner_local.py` proving image blocks reach the fake provider only when `vision_supported=True`). Version bumped to `0.4.34`, rebuilt, force-reinstalled, diff-verified identical (only the pre-existing harmless `workspace.py.bak`), `tamfis-code --version` confirmed on the real `$PATH` binary.

**Known limitation, noted not fixed this pass**: `vision_supported` is a provider-level flag, not per-model -- e.g. Ollama's `default_model` (`llama3.2:3b`) isn't a vision model even though the Ollama provider entry has `vision_supported=True` (because some of its OTHER listed models, e.g. `llava:7b`/`tamgpt-vision-pro:latest`, are). An image attached during an Ollama-fallback turn with the non-vision default model would be spliced in and likely ignored/error by that specific model. Low real-world impact (Ollama is `local_only`, lowest priority, zero weight in AUTO routing) but a real accuracy gap in the existing data model, not introduced by this change -- flagged for a future pass, not chased here.

## Real, self-contained web_search tool + fixed dead RESEARCH task-type routing (2026-07-20, v0.4.33)

Closed the first item off a fresh Claude Code/Codex/Kimi orchestration parity audit (standing directive: keep tamfis-code's orchestration -- not model quality -- rivaling those tools). Audit found `web_search` claimed as a real capability in the `--remote` `/tools` table, and separately found `RESEARCH_TOOLS` (`tool_policy.py`) already listed `browser` -- but `TaskType.RESEARCH` had **zero branch in `routing.classify_task`** that ever returned it, meaning both `browser` and any future research tool were unreachable dead code through the normal agent loop; the model could only ever use them via the direct `tamfis-code tools call` CLI escape hatch, never mid-turn on its own.

Added a real, self-contained `web_search` tool to the standalone local-tool path (`mcp.py`): Tavily primary if `TAVILY_API_KEY` is set, DuckDuckGo HTML-endpoint fallback otherwise (no key required, always available). Deliberately **not** implemented via `_import_monorepo_attr` reuse of tamgpt6's `WebSearchManager` (the pattern `browser` uses) -- explicitly corrected by the user mid-session: tamfis-code must keep working when installed standalone on a machine that never had tamgpt6 on it (same portability bar already set for config/state paths in v0.4.30), so the dual-provider Tavily/DuckDuckGo logic is implemented natively in `mcp.py` using the already-declared `httpx` dependency, not imported from the monorepo.

Fixed the dead-routing bug: `routing.py`'s `classify_task` now has a real RESEARCH branch (web-directed phrasing: "search the web", "look up online", "google ", "latest news", etc.), checked before INSPECT's broad "search"/"find" catch-all so ordinary in-repo search requests are unaffected. `web_search` added to `RESEARCH_TOOLS` alongside `browser`. Standalone `/tools` table (`interactive.py`) and the fake-tool-call narration guard's tool-name regexes (`runner_local.py`) updated to include it.

**A real bug caught immediately by the test suite**: `import html` at module scope in `mcp.py` collided with `_parse_duckduckgo_html(html: str, ...)`'s own parameter name of the same name, shadowing the module inside the function and breaking `html.unescape()` with `AttributeError: 'str' object has no attribute 'unescape'` -- caught by 6 failing tests the moment HTML-entity unescaping was added (DuckDuckGo results were showing raw `&#x27;`/`&amp;` in titles/snippets). Fixed by renaming the parameter to `html_text`. Same general "don't reuse an already-imported name as a local/parameter name" gotcha previously seen in `project_tamgpt6_tier_iv_default_and_mode_switch`.

Live-verified twice against the real network, not just mocks: once confirming Tavily-primary/DuckDuckGo-fallback both independently return real results (Python.org pages), and again after the `html` fix where Tavily happened to return a real HTTP 432 ("exceeds your plan's set usage limit") -- confirming the non-200 fallback path works under a genuine live failure condition, not just a simulated one. Final live check via the actual installed `$PATH` binary: `tamfis-code tools list` shows `web_search`, `tamfis-code tools call --name web_search --params '...'` returns real structured Node.js LTS results from DuckDuckGo.

**Concurrent-editing note**: while writing `tests/test_routing.py`, another session's edit briefly appeared inside a just-written test function (two extra assertions expecting `INSPECT` to have `requires_long_context=True`/`preferred_quality_tier="frontier"`, which doesn't match current `routing.py` behavior) -- removed to restore the test as actually authored, per the established shared-checkout caution (see `project_tamgpt6_concurrent_sessions_risk`). No production code was affected.

Full suite: 670/670 passing (was 656 baseline, +14 new tests across `test_mcp.py`, `test_routing.py`, `test_orchestrator.py`). Version bumped to `0.4.33`, rebuilt, force-reinstalled, diff-verified identical (only the same pre-existing harmless `workspace.py.bak`), `tamfis-code --version` confirmed `0.4.33` on the real `$PATH` binary.

**Remaining items from the same audit pass, not yet done**: no real vision/image content-block path for `--attach`ed images in the standalone loop, no hooks system, no user-extensible custom slash commands, no declarative subagent types (`.claude/agents/*.md`-equivalent), plan mode is still prompt convention rather than a distinct gated propose-then-approve phase. Tracked as ongoing work, not blocking.

## Durable stream reconnects, clean provider routing, and grounded service ports (2026-07-20, v0.4.32)

Fixed the reported always-cutting-off stream path in the standalone agent. The OpenAI SDK no longer performs an immediate invisible retry; Tamfis Code owns recovery and visibly backs off for 5/15/30 seconds on transient disconnect, timeout, capacity, and service-restart failures while keeping the live task active. Clean streamed text is checkpointed before reconnect. The reconnect request receives that exact assistant prefix, continues the same task with its existing tool results, suppresses internal retry text, de-duplicates any repeated boundary, then displays only the novel suffix. Account/payment failures move to the next eligible route immediately. If every route remains unavailable, the turn is explicitly `interrupted` with the exact partial response and error in `.memory`, ready for `continue`; it is never falsely finalized as complete.

Added a 1,024-character quality quarantine plus two independent corrupt-output guards. Exact repetition loops and the reproduced recombined-token/lexical corruption (`...urp...webkit...`) are stopped early, kept out of terminal output and checkpoints, and retried on a clean external route. AUTO priority is now Tier IV, NVIDIA, OpenRouter, Hugging Face, then Ollama; NVIDIA uses Kimi K2.6 instead of the loop-prone Nemotron default. The turn always prints its real provider/model, while the banner now says `Runtime: standalone · Provider: auto (external providers first; Ollama last fallback)` rather than the misleading `Host: local:auto`. The provider table also displays real priority values instead of weights.

Closed the port-grounding/context pollution that led to incorrect 8080/8081 claims. Git-backed discovery now applies ignore filtering to every path, excludes user caches, dependency trees, generated outputs, and virtual environments, and safely expands nested-project directory entries that a parent Git repository otherwise reports as a single opaque path. Live indexing of `/home/tamfisgpt` fell from the prior 20,000-file cap to under 4,800 relevant files, with zero `.cache` or `node_modules` hits. Portable, source-labelled endpoint extraction distinguishes application process binds, reverse-proxy listeners/upstreams, and container published/internal ports. Live evidence now identifies `tamgpt6/tamfis-gpt-capacity.conf` port 9500 and `tamgpt6/tamgpt-orchestrator-capacity.conf` port 9555 as `application_process_bind`; proxy topology can no longer be relabelled as the application service port. No TamfisGPT/VPS port or home path is hard-coded into the installable package.

Regression coverage includes interrupted-stream continuation without duplicated text, no-output failure only after the backoff budget, corrupted-token quarantine and clean-route retry, external-before-Ollama ordering, Kimi default selection, truthful banner/model display, hidden cache/dependency exclusion, nested-repository expansion, and application-vs-proxy endpoint roles. Focused suite: **149/149 passing**; full source suite: **656/656 passing** with a portable writable config root. Built both the wheel and source archive without network/build isolation, upgraded the real system installation from 0.4.31 to **0.4.32**, and verified `/usr/local/bin/tamfis-code` imports from `/usr/local/lib/python3.13/dist-packages`, reports v0.4.32, exposes priority `tier_iv → nvidia → openrouter → hf → ollama`, uses Kimi K2.6, carries the 5/15/30 reconnect policy and corruption guard, excludes cache/dependency files, and discovers the real 9500/9555 binds.

## Real tool execution after narrated intent + actionable stream recovery (2026-07-20, v0.4.31)

Fixed the exact live failure where `check your previous status and continue` produced several paragraphs of “Let me examine/check/read...” without a single registered tool call, then a provider disconnect rendered only `Task failed: An error occurred during streaming`. Realtime `.memory` was working and retained the full transcript; the executor was wrongly accepting future-tense tool narration as a completed answer. It now detects bounded, explicit promises to inspect/read/search/run repository state, records the prose in the durable transcript, injects a corrective instruction requiring the registered tool call, and retries once on that route. If the route repeats the behaviour, `auto` switches to another eligible provider; with no route left it marks the turn interrupted instead of completed and tells the user how to resume.

The generic OpenAI-compatible SDK wrapper `An error occurred during streaming` is now classified as a retryable route/transport failure, so automatic provider fallback applies even when the SDK discarded the underlying HTTP body. If every route fails, the checkpoint now records `status: interrupted`, partial assistant text, and an actionable `last_error` containing provider/model context and the exact `continue` recovery instruction; it no longer leaves a misleading `running` checkpoint beside a context-free error.

Resume objective reconstruction now keeps the original task together with later user clarification (for example, the original status check plus `tamgpt6`/`tamfis-frontend` locations) while stopping at the previous genuine completed-answer boundary, so unrelated older conversation objectives cannot leak in. The same reconstruction now handles old processes that incorrectly cleared their checkpoint after accepting narration as completion, and legacy history drops accidental internal `active_plan=...` prompt/response pairs. “check ...” requests are classified as repository inspection and therefore routed with required tool capability. Full source suite: **644/644 passing** with an isolated portable config home; the five failures seen without that override were pre-existing sandbox-only writes to the intentionally read-only `/root/.config`, not code regressions. The wheel was rebuilt and the real `/usr/local/bin/tamfis-code` installation upgraded from 0.4.30 to **0.4.31**.

## Worldwide-installable, platform-native runtime storage (2026-07-20, v0.4.30)

Clarified and hardened the packaging boundary after the user correctly rejected any VPS-specific storage design. No `/home/tamfisgpt` or `/root` path was present in the runtime config resolver—the displayed path was `Path.home()` resolving on this VPS—but the fallback was Linux-centric. `config.resolve_config_dir()` now has an explicit portable contract: `TAMFIS_CODE_CONFIG_HOME` wins everywhere; Windows uses `%APPDATA%\tamfis-code` (then `%LOCALAPPDATA%`, then the normal home fallback), Linux/Unix honors `$XDG_CONFIG_HOME/tamfis-code` and otherwise uses `~/.config/tamfis-code`, and macOS uses `~/Library/Application Support/tamfis-code`. Config, credentials, state, evidence, history, and `.memory/session-<id>.json` all derive from that runtime-resolved per-user directory—not the source checkout, wheel build host, or `site-packages`.

The README now documents the cross-platform locations and portable/container override. Four platform-resolution regression tests cover explicit override, XDG/fallback Linux, macOS, and Windows AppData. Focused config/runner/CLI verification: **135/135 passing**; full suite: **638/638 passing**.

## Canonical realtime `.memory` and legacy/cross-workspace resume recovery (2026-07-20, v0.4.29)

Closed the remaining restart failure reproduced immediately after v0.4.28: a user typed `continue from where you stopped`, but the selected local session had no new-format checkpoint because the interrupted work predated v0.4.28, so the model answered that it had no context. Tamfis Code now maintains a canonical owner-only, secret-redacted, atomic memory mirror at `~/.config/tamfis-code/.memory/session-<id>.json`. It is refreshed by every session-state transition and therefore at turn start, at up to four times per second while text streams, before tool dispatch, after tool results, on interruption, and on completion. Unlike the transient turn checkpoint, the `.memory` snapshot remains after completion and retains bounded conversation history, active/running task, partial checkpoint, recent actions/checkpoints, mutations, validations, and unresolved issues.

Explicit `continue`/`proceed`/`resume` requests now recover across local sessions whose workspace roots are the same or direct ancestor/descendant paths, ordered by real update time with a valid interrupted checkpoint taking priority. This handles restarting from `/home/tamfisgpt` versus `/home/tamfisgpt/tamgpt6` without opening unrelated-workspace conversations. For pre-v0.4.28 sessions, a bounded legacy recovery package is constructed from timestamped objective-linked tool actions and the latest real user instruction. A stale `active_task`, synthetic `active_plan=...` checkpoint, unrelated prior-turn mutations/artifacts, or a previous failed “continue”/“I have no context” exchange no longer overrides or leaks into the recovered task. The one-shot CLI also preserves the prior active objective until recovery has read it instead of overwriting it with the literal continuation prompt.

Live-state verification first recovered the legacy objective `please execute the benchmark.py` and its recorded provider timeout/tool evidence from the related `tamgpt6` child workspace, then correctly advanced to the user's newer real instruction to inspect `STATUS.md`, `README.md`, and memory while rejecting a synthetic `active_plan=...` checkpoint. Verification: **634/634 tests passing**.

## Durable interrupted-turn memory and transparent provider continuation (2026-07-20, v0.4.28)

Closed the interruption/amnesia failure reproduced by the user. Standalone sessions now persist bounded completed conversation history plus an atomic live-turn checkpoint containing the original objective, provider/tool transcript, partial assistant output, and execution status. Interactive and one-shot invocations rehydrate completed history automatically. An explicit `proceed`/`continue`/`resume` request restores the interrupted transcript—including named selections such as “proceed with 1, 2, and 3”—rather than sending a contextless new prompt. Tool-call IDs are checkpointed before dispatch and results immediately afterward; if the process dies while a tool is in flight, resume closes the unmatched call with an unknown/interrupted result and instructs the model to inspect real state instead of blindly repeating a possibly completed command or mutation. State remains secret-redacted, owner-only, bounded, and atomically replaced.

Output-length continuation is now an internal implementation detail: the normal `Response was cut off ... continuation (1/6)` diagnostic is no longer rendered, continuation output is streamed as one visual answer, repeated boundary text is removed, and continuation calls still cannot invoke tools. If the current provider hits a retryable worker/request ceiling (including the reproduced `ResourceExhausted ... 32/32`) during continuation, auto routing silently tries the next eligible provider. If no route exists, the partial turn is marked interrupted and checkpointed instead of being mislabeled completed and followed by a misleading fake-tool caveat.

Added compatibility recovery for OpenAI-compatible models that emit `<tool_call><function=...><parameter=...>` as streamed assistant text instead of native `tool_calls`. Complete valid calls are hidden from user output, normalized, and sent through the same scope, risk, approval, execution, and tool-result pipeline; invalid, incomplete, unknown, or unoffered markup remains text. Read-only inspection routing can now offer `execute_command`, but only a conservative allowlist of inspection commands (`find`, `rg`, `ls`, read-only git commands, etc.) classifies read-only; shell control/redirection, `find -delete/-exec`, `sed -i`, unknown commands, and mutations remain blocked in read-only mode or approval-gated elsewhere.

Two full-suite gaps surfaced and were fixed during verification: `find_references` now reuses a per-turn temporary symbol index (unchanged files are no longer reparsed on every call, without polluting the repository/home directory), and the no-live-config async approval path no longer deadlocks in workers with an exhausted/unavailable default thread executor. Verification: **632/632 tests passing** with a writable isolated config home; focused continuation/memory/safety/interactive suite **170/170 passing**. The wheel was built without network/build isolation, the prior system installation was upgraded from 0.4.27 to **0.4.28**, and the real `/usr/local/bin/tamfis-code --version` plus imports from `/usr/local/lib/python3.13/dist-packages` were smoke-verified after installation.

## Standalone attachments, safe archive round-trip, and generated-media stream parity (2026-07-20, v0.4.27)

Closed the uploaded-project gap without weakening workspace isolation. `--attach` now works in the standalone/local-provider path (the previously hard-coded rejection is gone): the exact user-selected files are passed to the turn as read-only inputs, but their parent directories do not become readable or writable workspaces. `read_file` was moved onto the same workspace/explicit-attachment resolver, which also closes an older boundary bug where that one tool resolved arbitrary absolute paths directly from process cwd while write tools were correctly confined.

Added model-callable `extract_archive` and `repackage_archive` tools for ZIP, TAR, TAR.GZ/TGZ, TAR.BZ2/TBZ2, and TAR.XZ/TXZ. They preserve binary members, enforce workspace-scoped extraction/output, reject traversal, symlinks/hardlinks/special TAR entries, non-empty extraction targets, 5,000-file/250 MB expansion limits, and package self-inclusion. Extraction validates the complete member table before writing and publishes from a private staging directory; packaging writes a temporary archive then atomically replaces the final output. Both tools are included only in mutating task modes, go through risk/approval classification and task-scope checks, count as real mutation evidence, populate tool-envelope `files_changed`, and are covered by the fake-tool-call narration guard.

Streaming parity was aligned with TamfisGPT's canonical payloads: the provider normalizer now accepts canonical `event` as well as `event_type`/`type`, preserves top-level generated-artifact payloads, and recognizes `artifact_generated`, `file_generated`, `image_generated`, `video_generated`, and `diff_available`. The terminal renderer handles all aliases and uses file/image/video-specific labels and URL fields instead of silently dropping them.

Public model-group audit result: TamfisGPT backend and frontend both expose the same seven capability groups (`auto`, `core`, `reason`, `code`, `research`, `vision`, `agentic`). Tamfis Code's provider/model registry is intentionally a different, provider-facing layer; no conflicting public-group list was found or introduced.

Verification: focused archive/safety/policy/provider/renderer suite 126/126 passing. Full-suite result is recorded below once the final run completes. Version bumped consistently in `pyproject.toml` and `tamfis_code.__init__`. The system-installed CLI has not yet been rebuilt/reinstalled in this entry, so do not claim the real `$PATH` binary is v0.4.27 until that deployment step is completed and verified.

## Added safe parallel "swarm" execution + bundled parity cleanup (2026-07-19, v0.4.26)

User asked for Kimi-"swarm"-style parallel multi-agent execution. Investigation found the hard part already existed: `AgentManager.execute_tasks` (`agents.py`) already did real concurrent fan-out (`asyncio.Semaphore` + `asyncio.gather`, tests already proving genuine overlap), wired via `agent-cmd delegate` (CLI) and `/delegate` (REPL) -- both defaulting to `max_concurrency=1` for real, confirmed reasons, not caution for its own sake:

1. **Terminal-rendering collision**: every concurrent `DelegatedCodingAgent` built its own `StreamRenderer` on the *same shared* `Console` -- `StreamRenderer` unconditionally starts a `rich.live.Live` region on a TTY with no guard against another `Live` already active on that console.
2. **Session/state collision**: `resolve_local_workspace()` intentionally collapses same-`workspace_root` calls onto one shared `session_id` -- correct for its existing callers, wrong for concurrent swarm sub-tasks, which then race on single-value `state.json` fields (`current_phase`/`running_action`/`active_task`) that aren't merge-safe.
3. **Silent-deny mutation gap**: `DelegatedCodingAgent` always runs `interactive=False`; under the default `"ask"` policy, a non-interactive caller's mutating tool call is silently denied with no signal anywhere.
4. **No mid-conversation trigger**: both existing entry points required a human to pre-write the task list up front -- no way for the model itself to decide, mid-turn, that a request decomposes into independent parallel work (the actual Claude Code/Codex/Kimi UX).

Fixed all four, then exposed it properly, in five phases (all done same day, all tests real, full suite green throughout):

1. **Distinct session identity per concurrent sub-task** -- new `workspace.resolve_swarm_subtask_workspace()` mints a fresh child session per sub-task (`parent_session_id`/`swarm_label`/`is_swarm_child` fields added to `state.SessionState`) instead of `resolve_local_workspace`'s same-workspace_root reuse. **Found and fixed two real bugs, neither caught by unit tests alone**:
   - While writing the tests: swarm child sessions were polluting `resolve_local_workspace`'s own reuse-matching for *ordinary* (non-swarm) callers afterward (a test proved it: `1 != 3`) -- fixed by excluding swarm children from that match.
   - **Only surfaced by live verification against the real installed binary, after the unit suite was already green**: the initial hide-from-default-listings filter keyed off `parent_session_id is not None` -- but `agent-cmd delegate` is a one-shot CLI invocation with no pre-existing session to record as a parent, so its child sessions got `parent_session_id=None`, indistinguishable from an ordinary session. `tamfis-code sessions` (default) silently kept showing them. Confirmed live: ran a real 2-task `agent-cmd delegate` against a scratch repo, `sessions` (no `--all`) still listed both children. Fixed by adding a dedicated `is_swarm_child: bool` field, set unconditionally on every child regardless of whether a real parent was known -- `parent_session_id` is now correctly just best-effort context, not the tag itself. Re-verified live after the fix: a fresh delegate run's children are correctly hidden by default and shown with `--all`.
   `/agents` and `agent-cmd sessions` both hide swarm child sessions by default now (`--all` to show them).
2. **Buffered sub-agent renderer** -- new `tamfis_code/swarm.py` module: `BufferedSubagentRenderer` implements the same informal protocol `StreamRenderer` exposes but never constructs a `rich.live.Live` at all, by construction (zero collision risk, not lock-based avoidance). `DelegatedCodingAgent`/`AgentManager.execute_tasks` gained an optional `renderer_factory` param (`None` default preserves every existing caller's exact behavior).
3. **Non-silent mutation gate** -- `mutation_policy_allows_swarm()` mirrors the exact auto-approving policy groupings `runner.py`'s `_decision_for_policy` already hard-codes (a dedicated test fails if the two ever drift). Swarm defaults to **read-only**; mutation requires explicit opt-in, checked *up front*, refusing to start with an actionable message rather than a silent per-call deny discovered N calls deep.
4. **`/swarm` REPL command** -- `swarm.run_swarm()`: validates the mutation gate, drives one aggregate `Live` status display (gated on TTY) via the buffered-renderer factory, calls `AgentManager.execute_tasks` (the same fan-out primitive, nothing reimplemented). `/swarm <a> | <b> | ... [--mutate]` wired into `interactive.py`, mirroring `/delegate`'s existing dispatch shape.
5. **Model-callable swarm tool** -- `delegate_parallel_tasks` (`SWARM_TOOL_SCHEMA` in `runner_local.py`, same shape/placement as the existing `retrieve_evidence` pattern), gated behind a new `allow_swarm_tool` param on `run_local_agent_turn` (default `False`, every existing call site's behavior untouched). **This is what actually closes the Claude Code/Kimi parity gap**: the orchestrating model itself can now decide mid-turn to fan out independent sub-objectives, instead of only a human being able to pre-script it via CLI flags or a slash command. **Reentrancy guard**: `allow_swarm_tool=True` only set at genuine top-level call sites (`cli.py` x2, `interactive.py` x3) -- never at `DelegatedCodingAgent.execute()`'s inner call, giving a hard structural depth-1 recursion cap (a delegated sub-task's own turn can never itself offer `delegate_parallel_tasks`) instead of a runtime counter.
6. **Raised `run_swarm`'s own default `max_concurrency`** from 1 to 3 (only once 1-5 above actually landed) -- `AgentManager.execute_tasks`'s bare default and `agent-cmd delegate --max-concurrency`'s CLI default are both deliberately left at 1; the higher default applies only to this now-hardened surface.

**Also fixed, bundled in the same pass** (small, independent, cheap):
- **`/delegate`/`agent-cmd delegate` had zero dedicated test coverage** before this (only `AgentManager.execute_tasks` itself was tested) -- added it first, before touching either's shared internals, so the Phase 1 changes to their shared code path had a real regression net under them.
- **`"never"` approval-policy naming footgun**: means deny-everything, the *opposite* of what it sounds like next to "auto"/"full-auto" (which mean never *prompt*, i.e. auto-approve). `/mode`'s own help text used to describe "auto" as "never prompt" right next to the unrelated "never" policy, and its error message claimed only 4 values were valid when "never" (and 5 other raw values) were silently accepted too. Fixed via disambiguating help text in `/mode`, `--approval`'s CLI help, and a comment on `config.APPROVAL_MODES` -- no rename (would break existing configs), decorative-clarity fix only.
- **`agent`/`agents`/`agent-cmd` naming collision**: added cross-referencing text to each command's docstring/`--help` (no rename, avoids breaking scripts/muscle memory); a durable test asserts each `--help` output mentions the others so this can't silently drift back.
- **Tier IV "Model: unknown" cosmetic bug**: an empty resolved model (NIM routes leave `config.default_model` blank by design, letting the provider pick its own default) showed as "Model: unknown" in `--debug` output, reading as a real problem when nothing was actually unknown. Now shows "(provider default)" instead -- a display-only fix (`render.py`), not a change to model resolution itself.
- **"Loose top-level file" workspace-scope gap** -- explicitly deferred, not touched this pass: a document that isn't inside a resolvable project-root subdirectory still can't be targeted by name; needs its own design pass, unrelated to swarm/naming and would have diluted focus here.

New tests: `tests/test_swarm.py` (19 tests -- mutation-policy drift detection, zero-`Live`-collision, renderer-factory wiring, `run_swarm` mutation-gate/mode/concurrency-default behavior), `tests/test_tamfis_code_workspace.py` (`ResolveSwarmSubtaskWorkspaceTests`, 4 tests, including the regression proof above), `tests/test_runner_local.py` (`SwarmToolSchemaGatingTests` + `SwarmToolDispatchTests`, 8 tests), `tests/test_interactive_standalone.py` (`/delegate` coverage + `/swarm` dispatch + `/mode`'s never-disambiguation, 14 tests), `tests/test_cli_commands.py` (`agent-cmd delegate` coverage + naming-collision help, 5 tests), `tests/test_tamfis_code_config.py` (2 tests), `tests/test_tamfis_code_render.py` (1 test).

Full suite: 620/620 passing (was 564 at the start of this pass). Version bumped to `0.4.26`, rebuilt, force-reinstalled into the real system location, diff-verified identical (aside from the same pre-existing, harmless `workspace.py.bak` noted in an earlier entry), `tamfis-code --version` confirmed `0.4.26` on the actual `$PATH` binary. Live-verified end to end in a scratch repo with a real NVIDIA NIM completion: `agent-cmd delegate` with 2 concurrent read-only tasks both completed correctly with no terminal corruption, and (after the `is_swarm_child` fix above) their child sessions are correctly hidden from `sessions`/`/agents` by default and shown with `--all`.

## Large clipboard pastes now collapse to a placeholder, Claude Code/Codex-style (2026-07-19, v0.4.25)

User reported: pasting a long block of text into the interactive prompt inserted the whole raw text into the input line and scrolled the terminal, instead of collapsing to something like `[Pasted text #1 +86 lines]` while typing (real content still submitted underneath) the way Claude Code/Codex do. Confirmed this was a genuine gap, not a regression -- `interactive.py`'s `KeyBindings()` never overrode prompt_toolkit's default `Keys.BracketedPaste` handler (`key_binding/bindings/basic.py`), which just does `event.current_buffer.insert_text(event.data)` verbatim; there was no collapsing logic anywhere in this codebase.

Added: a new pure `paste_placeholder(data, count)` helper (`interactive.py`) that normalizes line endings (matches prompt_toolkit's own default handling of `\r\n`/`\r`, e.g. iTerm2 pastes) and, for a paste strictly longer than `PASTE_COLLAPSE_LINE_THRESHOLD` (3) lines, returns a placeholder like `[Pasted text #1 +86 lines]` plus the real normalized text -- otherwise `None` (a short paste of a few lines still inserts and displays normally, matching how Claude Code/Codex only collapse genuinely large pastes). A new `@bindings.add(Keys.BracketedPaste)` handler in the interactive REPL calls this, inserting the placeholder into the buffer instead of the raw text and remembering `{placeholder: real_text}` in a per-turn `pending_pastes` dict (cleared before every prompt). Right after `session.prompt_async()` returns, every placeholder in the submitted line is expanded back to its real text before anything else (slash-command dispatch, `contextualize_short_reply`, `parse_intent`) ever sees it -- the model/rest of the pipeline always receives the full original pasted content, never the placeholder string.

New tests in `tests/test_interactive_standalone.py` (`PastePlaceholderTests`, 8 tests) cover the threshold boundary (exactly at vs. one over), CRLF/CR normalization, empty paste, and that the placeholder's `#N` count isn't hardcoded. Verified end-to-end against a REAL `prompt_toolkit.Application` (not just the pure function in isolation) using `prompt_toolkit.input.create_pipe_input()` to feed an actual bracketed-paste escape sequence (`\x1b[200~...\x1b[201~`) for an 86-line block through `run_interactive` itself, with `run_local_agent_turn` mocked to capture what objective actually got submitted: confirmed it exactly matches the original 86-line pasted content, not the placeholder string.

Full suite: 564/564 passing (was 555; +9 new tests). Version bumped to `0.4.25`, rebuilt, force-reinstalled, diff-verified identical (aside from the same harmless `workspace.py.bak` noted in the entry below), `tamfis-code --version` confirmed `0.4.25` on the real `$PATH` binary.

**Scope note**: only the main interactive REPL's objective-entry prompt got this treatment (the one place a large paste is actually likely) -- the separate approval-decision sub-prompt (`runner.py`'s `_prompt`) was left untouched, since pasting a large block there isn't a realistic scenario (it only ever expects y/n/a).

## Stuck-loop guard now self-corrects instead of just failing (2026-07-19, v0.4.24)

Direct follow-up to the v0.4.23 entry below -- user explicitly asked "why did you not make it self-correct" after the prompt-only fix. Restructured `runner_local.py`'s loop guard (both the identical-repeat and the cycling check) so it no longer fails the turn on the spot the moment it trips:

1. **First trip**: the repeated tool call(s) are refused (a real `role: "tool"` result explaining why, not executed again -- still satisfies the tool-calling protocol's one-response-per-call-id requirement) and a system reminder is appended telling the model to act on a specific result it already has, or say the request is too broad and ask the user to narrow it. The turn continues -- the model gets one real, bounded (`MAX_LOOP_NUDGE_RETRIES = 1`) chance to self-correct with tools still available.
2. **Second trip** (nudge budget exhausted): tools are disabled for one final completion asking the model to synthesise its best current answer/plan from whatever it actually found, run through the exact same finishing logic (`_finalize_completed_answer`, extracted from the old inline "no tool_calls" branch so both paths share it: truncation-continuation handling, fake-tool-call/no-mutation caveats, orchestrator validation) as an ordinary completed turn. Only if that recovery answer is *itself* empty or errors does the turn actually fail -- now a fallback of last resort instead of the first response to a detected loop.

New tests in `tests/test_runner_local.py`: the old single "fails immediately" test was renamed/restructured to
`test_identical_tool_call_repeated_eventually_fails_if_recovery_also_empty` (still fails, but only after confirming a refusal + nudge + a real (empty) recovery attempt all genuinely happened first) plus two new ones proving the self-correction actually works --
`test_identical_tool_call_self_corrects_after_one_nudge` (model gives a real answer right after the one nudge, no forced recovery needed) and
`test_identical_tool_call_recovers_via_tools_disabled_synthesis` (model keeps repeating past the nudge, but the final tools-disabled completion produces a real answer, confirmed sent with tools actually disabled).

**Note on concurrent editing**: mid-session, the user was also independently modifying `runner_local.py`'s exact stuck-loop message wording (`stuck_reason` phrasing, an added "Try narrowing the objective..." suggestion) and, separately, fixed a real live incident in `workspace.py`/`enforcer.py` -- npm being run against Python/WordPress projects (`_discover_project_type` added, `enforcer.py`'s `_run_frontend_tests` now gates on it before ever invoking npm). Caught via failing tests after a fresh full-suite run (`test_ordinary_stylesheet_without_a_theme_header_is_not_wordpress`, `test_missing_package_json_sets_no_package_status`) rather than by any file-conflict error -- both were pre-existing tests asserting stale expectations from before that concurrent fix, not regressions in it. Fixed one real rough edge surfaced by the first of those (a Node project with both `package.json` and `MANIFEST_LANGUAGE_MAP`'s generic `"JavaScript/TypeScript"` label AND `_discover_project_type`'s more specific `"JavaScript"`/`"TypeScript"` label both ended up in `detected_languages` at once -- now the specific one supersedes the generic one), then updated both stale tests to match the new, more accurate intentional behavior. See [[project_tamgpt6_concurrent_sessions_risk]]-style lesson: re-verify the full suite after any "a lot has changed" signal before trusting your own in-context understanding of a shared file.

Full suite: 555/555 passing. Version bumped to `0.4.24`, rebuilt, force-reinstalled, diff-verified identical (aside from a harmless `workspace.py.bak` backup file left by the concurrent edit, correctly excluded from the wheel), `tamfis-code --version` confirmed `0.4.24` on the real `$PATH` binary.

## Live-reported: broad "investigate the entire system" request got stuck re-listing the same directory (2026-07-19, v0.4.23)

User pasted a real transcript: `tamfis-code [auto]> investigate the entire system for vulnerability, ect. and create a bug fix plan` called `list_directory` on the same path twice, then the task failed with `runner_local.py`'s existing loop guard (`MAX_CONSECUTIVE_IDENTICAL_ROUNDS = 2` -- 3 identical rounds in a row trips it, by design, to avoid burning rounds on a model stuck polling something that never changes).

Investigated whether this was the guard misfiring (a real bug elsewhere making the tool result invisible to the model, e.g. the tamgpt6-side "message normalizer drops tool history" class of bug -- see [[project_tamgpt6_document_generation_pipeline]]) or a genuine model-gets-stuck case. Checked: `working_messages.append({"role": "tool", "tool_call_id": tc.call_id, ...})` correctly correlates every tool result to its call, and `mcp.py`'s `_list_directory` returns a real, informative, correctly-sorted listing (name/is_file/is_dir/size/modified) for all 30 entries, well under its truncation threshold -- the tool and message plumbing are both fine. Root cause is a genuine gap instead: `workspace.py`'s `build_system_prompt` had zero guidance steering the model toward acting on a directory listing it already has (read a specific file it named, recurse into a specific subdirectory, narrow scope) versus re-listing the same top-level path while it decides what to do -- exactly the failure mode a small/weaker model hits on an extremely broad, unscoped request like "the entire system."

Two fixes, both low-risk (prompt/message text only, no control-flow changes):
1. Added an explicit rule to `build_system_prompt`: never repeat list_directory (or any read-only tool) with identical arguments already used this task; after listing a directory, either read_file/list_directory/search_code something *specific* it actually returned, or tell the user the request is too broad and ask them to narrow it.
2. The loop guard's own failure message now suggests narrowing the request to a specific component/directory/concern instead of just reporting "stuck" with no next step -- matches this codebase's existing tone (`mcp.py`'s own `list_directory` truncation note already suggests "narrow the path or use search_code").

New test in `tests/test_tamfis_code_workspace.py`
(`test_warns_against_repeating_identical_read_only_calls`) locks in the new
system-prompt rule; extended the existing
`test_identical_tool_call_repeated_stops_before_max_rounds` in
`tests/test_runner_local.py` to also assert the failure message suggests
narrowing. Full suite: 550/550 passing (was 549). Version bumped to
`0.4.23`, rebuilt, force-reinstalled, diff-verified identical,
`tamfis-code --version` confirmed `0.4.23` on the real `$PATH` binary.

**Not fixed this pass, deliberately**: this doesn't guarantee the loop can
never recur for a sufficiently vague/large request on a weak enough
model -- it's a prompt nudge, not a structural guarantee. A more thorough
fix would give the loop guard a self-correction nudge (one bounded extra
round with a system reminder, mirroring the `MAX_FILE_GEN_NUDGE_RETRIES`-
style pattern already used elsewhere for "model narrates instead of
calling the right tool") instead of failing outright with zero synthesis
of whatever was found so far -- not attempted here since it's a bigger,
riskier change than a same-day drive-by fix warrants; worth doing as its
own pass if this recurs after the prompt fix.

## Reduced default terminal noise per turn -- 3 always-on setup lines gated behind debug (2026-07-19, v0.4.22)

User asked for tamfis-code's terminal UI to not "overbloat" while a task runs, explicitly comparing it to Claude Code's clean default. Audited every place `render.py`'s `StreamRenderer` prints something and found the real always-on noise was narrow and concrete: three lines printed unconditionally on **every single turn**, before the model even starts producing an answer, with nothing actionable in the common case:

1. The workspace-scope diagnostic (`"Focused workspace scope: ..."`) -- previously sent as a generic `{"event_type": "diagnostics", ...}` with no dedicated handler in `render.py`, so it silently fell through to the renderer's "unrecognised event type" fallback and printed with an ugly redundant `"diagnostics: "` prefix.
2. `context_reused`/`context_rescanned` ("Reusing workspace context…" / "Workspace rescanned…").
3. `model_selected` ("Provider: X · Model: Y · reason").

Everything else that goes through `render.py`'s generic `"diagnostics"` event type (retries, provider fallbacks, context compaction, truncation continuations, live-instruction/mode-switch/follow-up confirmations from v0.4.20/21, cancel/pause) was deliberately left alone -- those are already conditional/exceptional (only fire when something notable happens), not routine per-turn noise, and hiding them would be a real regression (silently swallowing legitimate signal), not a bloat fix.

Fix: gave the workspace-scope diagnostic its own dedicated event type (`workspace_scope`, was reusing the generic `"diagnostics"` type) and gated all three behind the existing `self.debug` flag (`TAMFIS_CODE_DEBUG=1` / `--debug`, already a documented CLI flag -- no new flag needed). `model_selected`'s non-printing side effect (`self._selected_provider`, used later for the "Executing with X..." status text) still always runs even when the print is suppressed. New tests in `tests/test_tamfis_code_render.py` (`test_routine_per_turn_setup_lines_are_hidden_by_default`, `test_routine_per_turn_setup_lines_show_in_debug_mode`) lock in both the default-hidden behavior and that `--debug`/`TAMFIS_CODE_DEBUG=1` still shows them, plus that the `_selected_provider` side effect isn't lost.

Verified visually with a real Console render before/after: default output now goes straight from a tool-call line to the answer, with the 3 setup lines gone entirely; `renderer.debug = True` brings them all back unchanged.

Full suite: 549/549 passing (was 547; +2 new render tests). Version bumped to `0.4.22`, rebuilt, force-reinstalled, diff-verified identical against `/usr/local/lib/python3.13/dist-packages/tamfis_code/`, `tamfis-code --version` confirmed `0.4.22` on the real `$PATH` binary.

## Same-terminal live input (mode switch + follow-up) while a task is running, and a "repeats an already-fixed bug" classifier bug (2026-07-19, v0.4.20)

Two more same-day fixes, continuing directly from the v0.4.19 entry below.

**1. Root-caused a live incident**: user told a running tamfis-code session
a particular bug was already fixed, and it went and re-applied the same
edit again instead of recognising there was nothing left to do. Root
cause in `routing.py`'s `classify_task()`: the DEBUG check is a plain
substring match on `("debug", "fix", "repair", "bug", ...)`, and "fix" is
literally a substring of "fixed" -- a pure closure/confirmation message
("yeah that bug is fixed now, thanks") matched it exactly like a genuine
new bug report, got handed edit tools via `tool_policy.allowed_tools()`,
and the model dutifully "fixed" a file that needed no further changes.
Fixed by adding a `_CLOSURE_SIGNALS` check (checked before DEBUG/EDIT) that
routes messages like "already fixed", "is fixed", "no need to touch it
again", "confirmed working", etc. to `TaskType.CONVERSATION` instead --
deliberately makes closure language win over an incidental "fix"/"bug"
mention in the same message, since redundantly re-touching an
already-fixed file is worse than occasionally missing a genuinely new
issue mentioned in the same breath (which still surfaces normally in the
user's next message). New tests in `tests/test_routing.py`
(`test_closure_confirmation_is_conversation_not_debug`,
`test_closure_confirmation_variants_are_conversation`,
`test_genuine_debug_request_is_unaffected` -- confirms "please fix the bug
in calc.py" still classifies as DEBUG, unaffected).

**2. Same-terminal mode-switch/follow-up injection while a task is
running** (explicit user request, chose "same-terminal hotkey, matches
Claude Code exactly" over a lower-risk second-terminal-only alternative
when asked). Investigation found this was only a *partial* gap:
mode-switching via Shift+Tab already worked live, mid-task, but only at
the exact moment an approval prompt is showing (`runner.py`'s `_prompt`
already had its own Shift+Tab binding mutating the same live `Config`
object `resolve_approval_decision` reads). What was missing: (a)
switching mode at any OTHER moment during a running turn (while the model
is just streaming/thinking, no approval pending), and (b) injecting a new
instruction without a second terminal (a second terminal + `tamfis-code
queue "..." --classification follow_up` already worked --
`runner_local.py`'s `_claim_live_queued_instructions`/
`_apply_live_queued_instruction` already poll the on-disk queue at the top
of every round -- but needed a second terminal/process).

Added `tamfis_code/live_input.py`'s `LiveInputListener`: a cbreak-mode raw
stdin reader running concurrently (via `asyncio.add_reader`) with an
already-streaming standalone turn, recognising exactly two keys and
dropping everything else silently:
- **Shift+Tab** directly cycles `cli_config.approval_policy` (the same
  live object, same `next_mode_in_cycle`) -- no queue involved, takes
  effect on the very next approval decision, at any point in the turn.
- **Ctrl+T** suspends the Live display, reads one line via
  `loop.run_in_executor(None, input, ...)` (doesn't block the event loop,
  so the model's response keeps streaming while the human types), and
  enqueues it via the EXISTING `local_state.enqueue_instruction(...,
  classification="follow_up")` -- an in-process producer for
  infrastructure that already existed, zero new queue plumbing.

Wired into `interactive.py`'s 3 standalone `run_local_agent_turn` call
sites only (`start()`/`stop()` in try/finally) -- the one-shot CLI
subcommands were deliberately left alone, since a one-shot command exits
after its one turn anyway and there's no "next round" REPL context for a
mode switch to matter to.

Extended `render.py`'s `StreamRenderer.suspend_live()`/`resume_live()` (the
same pair every approval gate already calls) to also pause/resume the
attached `live_input_listener`, so the raw byte reader and a blocking
`console.input()` approval prompt never race for the same fd -- and to
also stop/not-immediately-restart the streaming-assistant Markdown Live
(added in v0.4.19) across a Ctrl+T interjection, so suspending the input
listener mid-answer doesn't visually corrupt the in-progress code render;
the next `assistant_delta` lazily recreates it from the still-intact
buffer.

Cross-platform: `termios`/`tty` import is guarded (`ImportError` ->
`_TTY_AVAILABLE = False`), so the whole feature is a clean no-op on
Windows or any non-TTY/piped invocation, same as every other TTY-gated
feature in this codebase.

**Verified two ways**: unit tests (`tests/test_live_input.py`, 9 tests --
Shift+Tab dispatch, partial-escape-sequence buffering, unrecognised bytes
dropped, Ctrl+T enqueues a real follow-up instruction via a real
(tmp-redirected) `state.json`, blank interject queues nothing, pause/
resume/start/stop are safe no-ops off a real TTY, `suspend_live`/
`resume_live` correctly delegate to an attached listener and are safe with
none attached) AND a real pty-backed smoke test (not just unit-level
mocks): opened a real pty pair, pointed the process's actual stdin at the
slave end, started the listener, wrote raw Shift+Tab bytes (`\x1b[Z`) to
the master end while a live status spinner was running, confirmed
`approval_policy` cycled from `ask` to `accept-edits` in real time, and
confirmed `termios.tcgetattr` on the pty exactly matches its pre-listener
state after `stop()` -- the specific failure mode that would otherwise
leave a user's real shell broken.

**Follow-up same day (v0.4.21) -- the rough edge above is now actually
closed, not just documented**, per explicit user request ("close it,
don't leave any messy thing for me"). Root cause of the rough edge: the
interjection prompt used a blocking `input()` in a thread-pool executor,
which has no way to coordinate with anything else concurrently writing to
the same terminal. Replaced it with a real `PromptSession.prompt_async()`
(native asyncio, no thread) wrapped in prompt_toolkit's own
`patch_stdout(raw=True)` context manager -- this is prompt_toolkit's
official, battle-tested mechanism for exactly this scenario: any text a
concurrently-running coroutine writes to stdout/stderr while a Application
is reading input gets safely inserted above the active input line and the
line is redrawn beneath it, instead of interleaving with/corrupting it.
`raw=True` keeps Rich's own ANSI colour codes intact instead of having
patch_stdout escape them as literal text. No custom buffering/arbitration
system needed -- this was the appropriately-scoped existing tool for the
job.

Verified two ways: a new unit test
(`test_interject_wraps_the_prompt_in_patch_stdout`) asserts `sys.stdout`
is actually `prompt_toolkit.patch_stdout.StdoutProxy` while the mocked
prompt runs, and is restored to the real stdout afterward; AND a real
pty-backed end-to-end run (not mocked) -- opened a real pty, redirected
the process's actual stdin/stdout to it, triggered Ctrl+T, "typed" a full
line plus Enter through the master fd while a renderer diagnostic fired
concurrently (the exact race the rough edge was about), and confirmed the
line was captured correctly and landed in the real (tmp-redirected)
on-disk instruction queue with the right text and `follow_up`
classification.

Full suite: 547/547 passing (534 in the v0.4.19 entry below, +9 from
`tests/test_live_input.py`, +3 from `tests/test_routing.py`, +1 new
patch_stdout regression test). Version bumped to `0.4.21`, rebuilt,
force-reinstalled, diff-verified identical against
`/usr/local/lib/python3.13/dist-packages/tamfis_code/`, `tamfis-code
--version` confirmed `0.4.21` on the real `$PATH` binary.

## User-reported bugs: raw code output, wrong file extension (2026-07-19, v0.4.19)

Two live user reports, standalone (non-`--remote`) path:

1. **Generated code showed up as unformatted raw text in the terminal**,
   not as a distinct highlighted code block. Root cause in `render.py`:
   `StreamRenderer.handle_event`'s `assistant_delta` branch printed every
   streamed chunk straight to the console (`console.print(content,
   end="")`) with zero Markdown parsing -- a fenced ` ```python ... ``` `
   block from the model rendered as literal backticks and plain text,
   never syntax-highlighted. (`interactive.py`/`cli.py` already import and
   use `rich.markdown.Markdown` elsewhere, but only for a *fallback*
   one-shot summary print that's dead in the common case, since
   `streamed_final_text` is set on the very first delta.) Fixed by
   buffering the assistant text for the block currently streaming
   (`self._assistant_buffer`) and re-rendering it as `Markdown` through a
   second `rich.live.Live` handle (`self._assistant_live`) on every delta,
   TTY only -- non-TTY/redirected output deliberately keeps the old raw
   print (the right behavior for a plain-text file; Live doesn't degrade
   gracefully off a real terminal anyway). Buffer/live reset in
   `_close_assistant()` so each block between tool-call rounds gets its
   own clean render instead of concatenating onto the last. Manually
   verified: a streamed `def add(a, b): return a + b` fence now renders
   with real Python token coloring on a dark code background instead of
   raw backtick text.
2. **Generated files still landed with a `.txt` extension instead of the
   real source extension** (e.g. `.py`, `.js`). `write_file`'s tool schema
   (`mcp.py`) is fully generic (`path`/`content`, no extension guidance),
   and the standalone system prompt (`workspace.py`'s
   `build_system_prompt`) had zero instruction about matching a new
   file's extension to its actual content/language -- unlike tamgpt6's
   `tamgpt_orchestration.py`, which already has this exact class of bug
   documented and prompt-level-guarded (`_fix_hallucinated_urls`-adjacent
   HARD RULEs, see project memory). Fixed by adding an explicit
   instruction to `build_system_prompt`: match the extension to the real
   language, never default to `.txt` (or any other wrong extension) for
   code, honor whatever filename/extension the user's request specifies.
   No code-level content-sniffing guard added deliberately -- classifying
   "is this code" from raw text is inherently fuzzy/false-positive-prone;
   the prompt-level fix is the same shape as the one already proven out
   on the tamgpt6 side for this bug class.

Both fixes are prompt/render-layer only, no schema or tool-execution
changes. Full suite: 534/534 passing (32/32 in
`tests/test_tamfis_code_render.py` specifically). Version bumped to
`0.4.19`, rebuilt, force-reinstalled into the real `$PATH` location
(`/usr/local/lib/python3.13/dist-packages/tamfis_code/`), diff-verified
identical to source, `tamfis-code --version` confirmed `0.4.19` on the
real binary.

**Separately raised, not yet scoped or implemented:** user also flagged
"in-place user input" and "mode switch while a task is running" as
missing. `Shift+Tab` mode-cycling (manual/accept-edits/auto/plan) already
exists (`interactive.py`, `KeyBindings` on the `PromptSession`), but only
between turns, at the next prompt -- there is no way today to switch mode
or inject input while a turn is actively streaming/running tools. This is
a real feature gap (would need a way to interrupt/steer a running
`asyncio` agent-loop turn safely), not a bug fix -- needs its own design
pass before implementation, not bundled into this entry.

## Parity audit continued: all 7 remaining gaps closed, plus a real production incident (2026-07-18, v0.4.18)

Direct continuation of the v0.4.17 entry below, same day, same session --
after shipping the first 3 (lowest-risk) fixes, closed the rest of the
11-item checklist: repair-retry tracking + plan-state resume (#1),
indexer incremental re-indexing (#2a), reference-resolution tool (#2b),
diff preview before approval (#4a), transactional multi-file revert (#4b),
REPL tab-completion (#8), and MCP server-exposure (#6) -- plus a real
production bug found and fixed mid-session (see below). Every item below
has real tests, verified passing, and is deployed to the real system
location, not just committed to the source tree.

**Production incident, found and fixed mid-session**: `tests/test_orchestrator.py`
had no state isolation -- unlike every other stateful test file, it never
redirected `state_module.CONFIG_DIR`/`STATE_PATH` to a temp directory, so
every `AgentOrchestrator` call in it wrote directly into the real
`/root/.config/tamfis-code/state.json`. One test in particular
(`test_huge_objective_is_not_duplicated_unbounded_into_the_system_prompt`,
session_id `9013`) intentionally uses a ~300KB objective string; with no
isolation, each repeated full-suite run during this session appended
another huge entry to that session's `saved_plans` (capped at 50 entries,
~150KB each). By the time this was caught (full-suite runs had gone from
~90s to several minutes and counting), the real state file had grown to
**14.8MB** -- confirmed via `py-spy dump` mid-hang, which showed the suite
stuck inside `redact_secrets`/`_sanitize` serializing that one bloated
file on every single `save_session_state` call, real usage included, not
just these tests. A near-identical smaller version of the same bug was
found in `test_doctor_session_diagnostics.py`'s `DiagnoseLocalSessionTests`
(added earlier this session) -- caught proactively before it could repeat.
  - Fix: added the same `CONFIG_DIR`/`STATE_PATH`-redirecting `setUp`/
    `tearDown` every other test file already uses to both files.
  - Cleanup (done with explicit user approval, real production data, full
    backup taken first): removed exactly the two bloated test-artifact
    sessions (`9013`, `9015`) from the real state.json --
    14,554,175 -> 1,715,409 bytes. Left every other session (real usage
    history, ids 1-50ish, and this session's own smaller test sessions)
    untouched.
  - Verified: a targeted re-run of both fixed test files left
    state.json's byte size exactly unchanged (proof the leak is closed,
    not just slowed), and the subsequent full-suite run finished in
    **24.5s** (was multiple minutes and climbing before the fix).

1. **#1 Agentic loop -- repair attempts and plan-state resume.**
   `orchestrator.mark_repair()` previously fired exactly once, immediately
   before giving up entirely -- real recovery (provider fallback chains,
   empty-continuation recovery) already existed in `runner_local.py` but
   was invisible to `repair_attempts`/`AgentPhase.REPAIR` even when it
   *succeeded*. Moved `mark_repair()` to fire at the point each real
   recovery attempt is actually tried (once per fallback candidate; once
   before `_recover_empty_continuation`), so the orchestrator's phase
   tracking now honestly reflects real repair activity instead of being a
   terminal-failure label. Separately, `/resume` (both `tamfis-code
   resume` and the REPL's `/resume`) showed only a conversation summary --
   a plan left mid-execution (some steps done, one in_progress, others
   pending) was completely invisible once resumed, even though state.py
   has carried real, live step-status data since the v0.4.17 fix below.
   Added `render.py`'s `print_resume_plan_status()`, wired into both
   resume paths.
   - Tests: `test_successful_fallback_is_tracked_as_a_real_repair_attempt`
     and an added assertion on the existing empty-continuation-recovery
     test in `tests/test_runner_local.py`; `PrintResumePlanStatusTests` in
     `tests/test_tamfis_code_render.py`; resume-plan-status tests in
     `tests/test_interactive_standalone.py` and `tests/test_cli_commands.py`.

2. **#2a Indexer incremental re-indexing.** `CodeIndexer.index()` always
   did a full re-parse of every matching file regardless of what changed
   -- its own `force` parameter was accepted but never read anywhere.
   Added `CodeFile.mtime_ns`, compared against the current filesystem
   state before re-parsing (skip if size+mtime match); a fresh
   `CodeIndexer` instance (a new CLI invocation) now loads the previous
   on-disk index first, so this holds across process runs too, not just
   repeated in-process calls. Also added deletion pruning for a whole-root
   `index()` call (no explicit `paths`) -- a file removed since the last
   index no longer lingers in results forever.
   - Tests: 5 new tests in `tests/test_indexer.py` (unchanged-file skip,
     modified-file reparse, `force=True` override, cross-instance
     persistence, deletion pruning).

3. **#2b Reference-resolution tool.** `references.py` was never actual
   cross-file reference resolution -- it's `@file`/`@folder` prompt-mention
   expansion, a real but unrelated feature. No find-usages/go-to-definition
   capability existed under any name, and nothing was callable by the
   model mid-turn. Added `MCPServer._find_references(symbol, path)`
   (registered as tool `find_references`): combines `CodeIndexer.search_symbol()`
   for real definitions with a whole-word `search_code` pass (reused, not
   reimplemented) for every reference across the codebase. Wired into
   `safety.READ_ONLY_TOOLS` (no approval needed) and `tool_policy.py`'s
   `READ_TOOLS`/`GIT_TOOLS`/`RESEARCH_TOOLS` (available by default
   alongside `search_code`).
   - Tests: `tests/test_find_references.py` (6 tests, including a check
     that a second call in the same directory doesn't re-parse unchanged
     files -- direct synergy with the indexer fix above). Proactively
     patched `Path.home()` in these tests so `CodeIndexer`'s default
     `~/.tamfis/index/` location is never touched by the test suite (the
     same class of bug as the state.json incident, caught before it could
     repeat here).

4. **#4a Diff preview before write/edit approval.** The approval panel for
   `write_file`/`edit_file` rendered raw JSON arguments -- for
   `write_file`, the entire proposed new file content as a JSON string --
   instead of a diff, even though real diff rendering (`print_unified_diff`)
   already existed and was correctly wired, just only ever run post-hoc via
   `tamfis-code diffs`. Added `runner_local._preview_diff_for_tool_call()`
   (read-only; computes the same diff `safety.record_mutation` would, via
   the same `_unified_diff` helper, without writing anything), a new
   `plan_step_progress`-style dedicated payload field (`"diff"`) on the
   `approval_required` event, and a `render.py` handler that prints it via
   `print_unified_diff` right in the approval panel. The fallback command
   label for these two tools was also shortened (`write_file(path=...)`)
   since the diff panel already shows the actual change.
   - Tests: `PreviewDiffForToolCallTests` (5 tests) and an integration
     test proving the live approval event carries a real diff, in
     `tests/test_runner_local.py`; 2 render tests in
     `tests/test_tamfis_code_render.py`.

5. **#4b Transactional multi-file revert.** `revert_mutation` reverted one
   `mutation_id` at a time with no shared identifier across a turn, so a
   multi-file edit had no way to be reverted together, or even discover
   which mutation ids belonged to the same turn. Added a `transaction_id`
   (one per turn, minted by `MCPServer.__init__`, threaded through both
   `record_mutation` call sites in `mcp.py`) and `safety.revert_transaction()`
   -- reverts every not-yet-reverted mutation in a transaction, in reverse
   chronological order (correct for sequential edits to the same file
   within one turn), stopping and reporting exactly what's still pending
   on the first failure rather than pressing on into unknown state. Wired
   into `tamfis-code revert <turn_id>` (auto-detected by the `turn_`
   prefix) and the REPL's `/revert <turn_id>`; `tamfis-code diffs`/`/diffs`
   now show the turn id alongside each mutation.
   - Tests: `RevertTransactionTests` (6 tests) in `tests/test_safety.py`;
     CLI and REPL integration tests in `tests/test_cli_commands.py` and
     `tests/test_interactive_standalone.py`.

6. **#8 REPL tab-completion.** `PromptSession` had no `completer=` at all
   -- Tab did nothing while typing, for slash-commands or otherwise. Added
   `SLASH_COMMANDS` (the canonical list, matching the dispatch checks
   further down `interactive.py`) and `_SlashCommandCompleter`, wired into
   the REPL's `PromptSession`. Deliberately inert once the line doesn't
   start with `/` or a space follows the command name, so it never
   interferes with typing an ordinary natural-language objective.
   - Tests: `SlashCommandCompleterTests` (5 tests) in
     `tests/test_interactive_standalone.py`.

7. **#6 MCP server-exposure.** tamfis-code could consume external MCP
   tools (`mcp.py`'s `MCPServer` is a client-side facade for that
   direction) but could not be driven by another agent/IDE as an MCP
   server itself -- no stdio/JSON-RPC listener, no `mcp` SDK dependency
   declared (though already installed system-wide and unused -- same
   undeclared-dependency gap this project has already fixed for
   `openai`/`pytest-asyncio`). Added `tamfis_code/mcp_stdio_server.py`,
   built on the real `mcp` SDK's low-level `Server` + `stdio_server`,
   wrapping the existing `MCPServer` as the tool executor. Deliberately
   exposes only a fixed read-only subset (`read_file`, `list_directory`,
   `search_code`, `find_references`, `get_git_info`) -- an external MCP
   client has no equivalent of the approval-gated agent loop, so anything
   that could mutate the workspace or run commands is refused at the tool-
   call boundary, not just omitted from the listing. New command:
   `tamfis-code mcp-server`. Declared `mcp>=1.2` in `pyproject.toml`.
   - Tests: `tests/test_mcp_stdio_server.py` (8 tests: tool-listing scope,
     real delegation to a workspace-scoped `MCPServer`, the refusal
     boundary enforced at call time too, not just in the listing) plus a
     CLI wiring test in `tests/test_cli_commands.py`.
   - **Verified live, not just under pytest**: a real JSON-RPC round trip
     (`initialize` -> `tools/list` -> `tools/call("find_references", ...)`)
     against `tamfis-code mcp-server` in a scratch repo returned the
     correct real tool schemas and correctly found a real function
     definition -- the full protocol path works end to end, not just the
     Python-level plumbing.

- Tests: 530/530 passing (`python3 -m pytest -q`, up from the 478 at the
  end of the v0.4.17 pass below, 63 up from this session's original 467
  baseline), finishing in **24.5s** (was climbing past several minutes
  before the state.json fix above). `python3 -m py_compile` clean.
- Version bumped to `0.4.18` in both `pyproject.toml` and
  `tamfis_code/__init__.py`, wheel rebuilt and reinstalled into
  `/usr/local/lib/python3.13/dist-packages/tamfis_code/`
  (`--force-reinstall --no-deps --break-system-packages`); `diff -rq`
  against the source tree is clean; `tamfis-code --version` reports
  `0.4.18`. `tamfis-code doctor` and `tamfis-code mcp-server` both
  smoke-tested for real from the reinstalled system binary.

### What's left from the original 11-item checklist

All 11 items now have real, working, tested implementations. Two
deliberate scope boundaries worth knowing about, not gaps:

- Multi-file revert (#4b) is "stop and report exactly what's left" on
  failure, not true atomic all-or-nothing rollback -- the underlying
  filesystem writes were never transactional to begin with, and claiming
  otherwise would be dishonest. `revert_transaction`'s `remaining` field
  always tells you precisely what still needs attention.
- The outbound MCP server (#6) exposes read-only tools only, by design --
  making it configurably unsafe (write/execute over MCP, with no local
  approval gate to catch a bad call) was explicitly out of scope for this
  pass.

One naming footgun noted in the original audit but not touched: the
`"never"` approval policy means deny-everything, not "never ask, always
allow" -- worth a doc note or rename in a future pass, low priority.

## Claude Code / Codex CLI parity audit + 3 local-mode gaps fixed (2026-07-18, v0.4.17)

An external "bring tamfis-code to Claude Code/Codex CLI parity" spec assumed
a much earlier/rougher codebase than what's actually here. Before writing
any code, ran a 5-way parallel read-only audit against the real source for
all 11 checklist items (agentic loop, codebase understanding, tool use,
diff editing, provider routing, MCP, sub-agents, interactive/non-interactive
modes, workspace awareness, diagnostics, task management). Most items were
already solid and real (tool surface/safety gating, diff-vs-overwrite
editing, provider routing+degradation, sub-agent delegation, REPL streaming,
git-aware workspace scoping) -- confirmed by tracing actual call chains and
matching against existing tests, not by reading docstrings. The audit found
the real gaps mostly clustered in one pattern: working code that existed but
was never wired into the default **local** (non-`--remote`) execution path,
which is what most users actually run. Picked the three lowest-risk,
highest-impact items from that cluster to fix this pass; the rest are
listed as still-open below.

1. **The project's own instruction-file convention (`TAMFIS.md`, `.tamfis`)
   was invisible to the live agent loop.** `workspace.py`'s
   `INSTRUCTION_NAMES` (the list that actually feeds `build_system_prompt`,
   confirmed via `discover_local_repository` -> `_indexable_files` ->
   `build_system_prompt`) only recognized `AGENTS.md`/`CLAUDE.md`/
   `CODEX.md`/`CONTRIBUTING.md`/`README.md`. A separate, unrelated,
   never-called implementation (`references.py`'s `InstructionManager` /
   `instructions.py`'s `get_instruction_context`) documented `TAMFIS.md` and
   `.tamfis` as this project's own convention -- but that code path has no
   caller anywhere outside itself and its own tests, confirmed via
   whole-tree grep. Rather than wiring in a second, duplicate live
   mechanism, extended the one that's actually live. Fix: added
   `"TAMFIS.md"` and `".tamfis"` to `workspace.py`'s `INSTRUCTION_NAMES`.
   The dead `InstructionManager`/`get_instruction_context` path was left
   alone (out of scope this pass -- `references.py`'s separate `@file`/
   `@folder` mention-expansion feature, `ReferenceResolver`, is real and
   still uses that module).
   - Test: `test_includes_tamfis_md_instruction_file` in
     `tests/test_tamfis_code_workspace.py`.

2. **Local-mode plan step status was created once and never updated again
   -- visible progress was fake for the default execution path.** Traced
   `state.update_plan_steps`'s only real caller: `runner.py:384`, the
   `--remote` backend client path (mirrors server-computed status). The
   local path (`orchestrator/engine.py` + `runner_local.py`, what most
   users run) called `save_plan` once in `AgentOrchestrator.begin()` and
   never touched step status again -- every plan step stayed `"pending"`
   regardless of real progress, even though `render.py` was already fully
   built (with an explicit "best-effort approximation" comment) to display
   status changes that never arrived. Separately found: when
   `runner_local.py` replaced the synchronous deterministic template plan
   with the real reasoning-grounded plan (`orchestrator.run.plan =
   reasoning_plan`, two call sites), it never re-persisted to
   `state.saved_plans` -- so `get_plan()`/`tamfis-code plan` kept returning
   the generic boilerplate template forever, not the real plan actually
   driving the turn.
   - Fix: `engine.py` gained `OrchestrationRun.plan_id`,
     `AgentOrchestrator.replace_plan()` (persists a plan swap under a fresh
     plan id instead of only updating in-memory state), and
     `_advance_plan_step()` (called from `record_tool()` on every observed
     tool result -- advances a step cursor proportional to real tool-call
     count, reserving the final step exclusively for `complete()`/`fail()`
     so it's never marked done before validation actually happens).
     `complete()` marks all steps completed on a passing validation report;
     `fail()` marks the in-progress step failed. Runner call sites updated
     to call `orchestrator.replace_plan(...)` instead of assigning
     `orchestrator.run.plan` directly.
   - A **new event type**, `plan_step_progress`, carries these per-round
     updates -- deliberately distinct from `plan_created` (which means "a
     new/revised plan now exists," reprints the plan banner, and resets the
     spinner phase). The first implementation reused `plan_created` for
     this and broke two existing integration tests
     (`test_reasoning_plan.py`) that count exactly one `plan_created` event
     per real plan creation/revision -- caught by the test suite before
     shipping, not after. `render.py` handles the new event type by
     refreshing the live step markers in place only, with no banner
     reprint and no phase change.
   - Tests: `test_tool_results_advance_plan_step_status`,
     `test_complete_marks_all_plan_steps_completed_on_pass`,
     `test_fail_marks_in_progress_step_failed`,
     `test_replace_plan_persists_new_plan_as_the_active_saved_plan` in
     `tests/test_orchestrator.py`.

3. **`doctor.py` had zero diagnostics for the default local mode.** It only
   ever checked the `--remote` backend (Tier III API reachability, remote
   transport server registration, session/event-replay integrity) -- no
   coverage of the four directly-called providers, token/context usage,
   tool-call success rate, or plan-step failures for local runs. Added
   `_diagnose_local_providers()` (reuses the existing, real
   `get_provider_status()` -- the same check `tamfis-code providers`
   already relies on, not a reimplementation) and `_diagnose_local_session()`
   (reads real, already-persisted evidence from `state.py`:
   `estimated_context_tokens` -- a new field, populated from the per-round
   token-budget estimate `runner_local.py` already computes but never
   persisted anywhere -- `completed_actions` for tool-call success rate,
   `saved_plans`/`active_plan_id` for step-progress summary,
   `unresolved_issues` for validation gaps). Also fixed a related
   correctness bug found in the same code: `doctor` reported a hard `FAIL`
   for "no `--remote` credentials" even for a pure local-only user who
   never needed them (the default mode requires none, per README) --
   downgraded to `WARNING`, with a real `FAIL` reserved for credentials
   that exist but are rejected/expired.
   - **Near-miss caught before shipping, not after**: the first version of
     this fix added the new checks only inside `run_doctor()` -- which
     turned out to be dead code for this purpose, because `cli.py`'s
     `doctor` command only calls `run_doctor()` when `--remote` is passed;
     the default/local branch (`if not _use_remote(...)`) is a separate,
     shorter code path that returns early and never reaches it. A live
     smoke test (`tamfis-code doctor` in a scratch directory) surfaced this
     immediately -- the new diagnostics simply didn't appear. Fixed by also
     calling `_diagnose_local_session()` directly from the local branch in
     `cli.py`, and added a CLI-level regression test
     (`test_doctor_reports_local_session_diagnostics_by_default` in
     `tests/test_cli_commands.py`) so this class of "helper function is
     correct but never reached from the real command" bug can't silently
     regress again.
   - Tests: `DiagnoseLocalProvidersTests`, `DiagnoseLocalSessionTests` in
     `tests/test_doctor_session_diagnostics.py`;
     `test_doctor_reports_local_session_diagnostics_by_default` in
     `tests/test_cli_commands.py`.

4. **Incidental fix**: `__version__` in `tamfis_code/__init__.py` is a
   second, separate copy of the version string from `pyproject.toml`'s
   `version` -- they can drift (found this at 0.4.16/0.4.16, in sync by
   luck, while bumping to 0.4.17: `tamfis-code --version` kept reporting
   the old version after the pyproject.toml bump alone). Both must be
   bumped together; no single-source fix was made (would be a
   restructuring change, out of scope this pass) -- noted here so the next
   version bump doesn't repeat this.

- Tests: 478/478 passing (`python3 -m pytest -q`, up from the 467 baseline
  at the start of this session). `python3 -m py_compile` clean.
- Version bumped to `0.4.17` in both `pyproject.toml` and
  `tamfis_code/__init__.py`, wheel rebuilt and reinstalled into the real
  system location, verified via `tamfis-code --version` (reports 0.4.17)
  and a clean `diff -rq` against
  `/usr/local/lib/python3.13/dist-packages/tamfis_code/`. Also smoke-tested
  `tamfis-code doctor` for real (not just under pytest) from both the
  source tree and the reinstalled system binary, in a fresh scratch
  directory, confirming the new local-session diagnostics lines actually
  print.

### Still open from the same 11-item parity audit (not attempted this pass)

Full findings are in this session's conversation history, not duplicated
into a file per the anti-audit-sprawl rule above. Summary of what's
genuinely missing or partial, ranked roughly by lift:

- **#1 Agentic loop**: `AgentOrchestrator`'s `REPAIR` phase is a
  failure-classification label, not a real retry-and-continue loop --
  `mark_repair()` is always immediately followed by `fail()` at both call
  sites in `runner_local.py`. Real resilience exists elsewhere (provider
  fallback chains, context rollover, truncation-continuation retries) but
  none of it routes through `AgentPhase.REPAIR` or increments
  `repair_attempts`. Separately, `/resume` reloads a conversation summary
  and drops into a fresh REPL -- it does not resume a saved plan's step
  state; `state.checkpoint()` only fires on terminal outcomes, never at a
  mid-plan boundary.
- **#2 Codebase understanding**: `indexer.py`'s `index()` always does a
  full re-parse of every file on every call -- no hash/mtime comparison
  against the previous index; its own `force` parameter is accepted but
  never read. Not exposed as a tool the model can call mid-turn either
  (only reachable via the human-invoked `tamfis-code index`/`search`
  subcommands). `references.py` is **not** cross-file reference resolution
  -- it's `@file`/`@folder` prompt-mention expansion. True find-usages/
  go-to-definition doesn't exist under any name in this codebase.
- **#4 Diff-based editing**: the approval panel shown before a
  `write_file`/`edit_file` call is approved renders raw JSON arguments
  (the entire new file content, for `write_file`), not a diff -- real diff
  rendering (`print_unified_diff`) exists and is correctly wired to real
  data, but only runs post-hoc via `tamfis-code diffs`/`/diffs`, never as
  a pre-approval preview. `revert_mutation` reverts one mutation_id
  correctly but isn't transactional -- a multi-file edit in one turn has
  no shared transaction id, so a partial revert can leave a turn
  half-reverted.
- **#6 MCP**: consuming external MCP tools works (`mcp.py`'s `MCPServer` is
  a client-side facade calling `server.call_tool(...)`). Exposing
  tamfis-code's own tools outward, as an MCP server other agents/IDEs could
  drive, does not exist anywhere -- no stdio/JSON-RPC listener, no `mcp`
  SDK dependency in `pyproject.toml`, no entry point.
- **#8 Interactive mode**: the REPL (`prompt_toolkit.PromptSession` in
  `interactive.py`) has no `completer=` configured -- no tab-completion for
  slash-commands or file paths while typing. (`completion.py` is unrelated
  -- it only generates static shell-level completion scripts for top-level
  CLI command names, e.g. `tamfis-code completion bash >> ~/.bashrc`.)
  Also noted, not fixed: the `"never"` approval policy name means
  deny-everything, the opposite of what the name suggests -- a naming
  footgun worth a doc note or rename in a future pass.

## Fixed today, two more (2026-07-18, v0.4.16)

Live incident report against `nvidia/nemotron-3-super-120b-a12b`, WordPress
site audit (`/home/finima/www`, `wp theme list --allow-root` and a direct
`mysql` query against `wp_options` to identify the active theme).

1. **A live database password from `wp-config.php` was echoed in cleartext
   into the approval panel and the "Running command · ..." status line.**
   The agent (correctly) queried `wp_options` directly via `mysql -u finima
   -p<real password> -h 127.0.0.1 ...` after `wp` refused to run as root
   without `--allow-root`. The raw command string, real password included,
   was rendered verbatim in two places: `render.py`'s `_tool_action_label`
   (the "→ Running command · ..." / "✓ Ran command · ..." lines, which
   print `arguments["command"]` directly) and `runner_local.py`'s
   `approval_required` panel (`f"{tc.name}({json.dumps(arguments, ...)})"`)
   -- both display/log surfaces, neither previously redacted anything.
   - Fix: new `safety.redact_secrets()` masks (a) `mysql`/`mariadb`/`psql`-
     family inline `-p<value>` (only when a SQL-client binary name is
     present, and only a standalone `-p` token -- not the literal "-p"
     substring inside `--password`), (b) `--password=`/`--password ` for
     any command, (c) URL-embedded `user:pass@host` credentials. Wired in
     at the two display sites above -- `render.py`'s `_tool_action_label`
     redacts `arguments["command"]` before building the status-line label,
     and `runner_local.py` builds a redacted `display_arguments` copy for
     the approval-panel event and the `resolve_approval_decision` prompt
     text, while the *real*, unredacted `arguments` still reaches the
     actual tool call (redaction is display/logging-only, never sent to
     execution). Also applied to `run_local_shell_command`'s explicit `$
     <command>` REPL path (both the console echo and what gets persisted
     via `local_state.start_action`'s `detail`).
   - Tests: `RedactSecretsTests` (6 cases) in `tests/test_safety.py`;
     `ToolActionLabelSecretRedactionTests` (2 cases) in
     `tests/test_tamfis_code_render.py`.
2. **A second, previously-uncaught shape of the "model narrates a fake tool
   call instead of making a real one" failure mode.** The existing
   `_looks_like_fake_tool_call` regex only matched a paren-style call
   (`read_file(...)`) -- this model instead wrote a
   `{"tool": "read_file", "argument": {"path": "..."}}` JSON object in
   plain prose, repeatedly, round after round, with zero real `tool_calls`
   each time. No `funcname(` substring exists anywhere in valid JSON like
   that, so it never matched, and the turn kept "succeeding" with nothing
   having happened until the separate degenerate-repetition guard cut
   generation off mid-stream as a last resort -- a much blunter, less
   informative stop than the existing fake-tool-call caveat.
   - Fix: added `_FAKE_TOOL_CALL_JSON_RE` in `runner_local.py`, matching
     `"tool"` or `"name"` JSON keys mapped to one of this agent's own tool
     names; `_looks_like_fake_tool_call` now checks both patterns.
   - Tests: `test_fake_json_shaped_tool_call_in_text_gets_a_caveat` in
     `tests/test_runner_local.py`.
- Tests: 467/467 passing. `python3 -m py_compile` clean.
- Version bumped to `0.4.16`, wheel rebuilt and reinstalled into the real
  system location, verified via `tamfis-code --version` and a diff against
  `/usr/local/lib/python3.13/dist-packages/tamfis_code/`.

## Fixed today, two more (2026-07-18, v0.4.15)

1. **`allowed_tools()` forced read-only tools onto misclassified edit
   requests -- fixed in a prior pass this same day but never packaged.**
   `tool_policy.py`'s `allowed_tools()` forced `READ_TOOLS` (no
   `write_file`/`edit_file`/`execute_command`) whenever
   `routing.classify_task()`'s keyword matcher guessed `TaskType.QUESTION`
   -- its catch-all fallback for anything not matching an explicit edit
   trigger word ("edit", "modify", "add ", "create ", "implement", etc.) --
   regardless of whether the session was actually read-only. A real edit
   request phrased without one of those exact words (e.g. "make the
   TamfisPress child theme full-width") got misclassified as QUESTION and
   handed only read tools, so the model correctly reported it had no
   file-editing tool instead of doing the edit. Fixed: only `INSPECT`/
   `AUDIT`/`PLAN` force `READ_TOOLS` unconditionally now; `QUESTION` falls
   into the `EDIT_TOOLS` bucket alongside `EDIT`/`DEBUG`/`MIXED` whenever
   `read_only` is false (offering edit tools doesn't force their use for a
   genuine question). Tests:
   `test_misclassified_edit_request_still_gets_edit_tools_outside_read_only_mode`,
   `test_question_is_still_read_only_when_explicitly_in_read_only_mode` in
   `tests/test_orchestrator.py`. **This fix landed in the source tree
   earlier today but was never rebuilt/reinstalled -- another instance of
   the same deployment gap documented below (v0.4.9->v0.4.10 entry): the
   real `tamfis-code` on `$PATH` was still running the old, buggy
   `tool_policy.py` right up until this version's install.**
2. **A model that only narrates a fix (never actually calling
   `write_file`/`edit_file`) could still get a service restart approved
   and executed as if the fix had landed.** Confirmed live: user asked to
   make a WordPress child theme full-width; the agent's final answer
   claimed `style.css` had been updated and restarted `php8.5-fpm`/`caddy`
   to pick it up -- the existing end-of-turn "no files were changed"
   caveat correctly fired afterward, but only *after* the (real, executed)
   service restart had already happened. The approval panel is the last
   point a human can catch this before it happens, and it had no signal
   at all that nothing had been written yet.
   - Fix: `runner_local.py` gained `_looks_like_service_restart()` (matches
     `systemctl restart|reload`, `service X restart|reload`,
     `/etc/init.d/X restart|reload`, and `apachectl`/`nginx`/`pm2`/
     `supervisorctl` restart/reload invocations). When an `execute_command`
     call matching this pattern comes up for approval, `any_mutation` is
     still `False`, and the objective looks like a change request
     (`_looks_like_change_request`), the approval panel's `reason` field
     now appends "⚠ No files have been changed yet this task -- verify the
     intended fix was actually written before restarting/reloading this
     service." This doesn't block the restart (a restart can be a
     legitimate, unrelated step) -- it surfaces the same fact the
     end-of-turn caveat already computes, but early enough for a human
     approver to act on it.
   - Tests: `test_service_restart_before_any_mutation_gets_a_warning_in_the_approval_reason`,
     `test_service_restart_after_a_real_mutation_gets_no_warning` in
     `tests/test_runner_local.py`.
- Tests: 458/458 passing. Wheel rebuilt and reinstalled into the real
  system location (`pip install --break-system-packages --force-reinstall
  --no-deps dist/tamfis_code-0.4.15-py3-none-any.whl`); confirmed via
  `tamfis-code --version` on the actual `$PATH` binary, and via a direct
  diff of `tool_policy.py`/`runner_local.py` against
  `/usr/local/lib/python3.13/dist-packages/tamfis_code/` (both were
  stale there before this install -- the source tree already had fix #1
  above, but it had never made it past this checkout).
- Version bumped to `0.4.15`.

## Added today, live mid-task instruction queue for standalone turns (2026-07-18, v0.4.14)

User asked: "Can it also update plan while it is working and user types in
an update prompt?" Checked the actual code before answering rather than
guessing -- the honest answer was previously no, for the standalone
(default, primary) path specifically. `cli.py`'s own `queue` command
already said so in a comment: "a standalone local turn is always
synchronous within one process, so there's nothing 'live' to push into."
The genuinely-live version (`runner.py`'s `watch_instruction_queue`,
polling mid-task) only ever existed for the legacy `--remote` backend.
True same-terminal live typing isn't feasible while a turn is streaming
(the REPL prompt isn't reading input during that time) without much
deeper prompt_toolkit/concurrency changes -- out of scope here. What *is*
now built: a **second terminal** running `tamfis-code queue "..."` against
the same session can reach an already-running standalone task.

- `runner_local.py`'s round loop (`run_local_agent_turn`) now claims
  (`_claim_live_queued_instructions`) any `queued`-status instruction at
  the top of every round -- a natural, never-mid-stream checkpoint --
  restricted to the classifications that meaningfully reach a live task
  (`append`/`follow_up`/`clarification`/`replace`/`cancel`/`pause`;
  `reprioritise` is deliberately excluded, it only makes sense against a
  not-yet-started backlog and is left for the next turn as before).
- `append`/`follow_up`/`clarification`/`replace` are spliced into
  `working_messages` as a new user-role message before the next completion
  request, explicitly telling the model to revise its plan/approach now,
  not just note it at the end -- so the very next round's request already
  includes it.
- `cancel`/`pause` end the turn immediately with `TaskOutcome(status=
  "cancelled", ...)`, before any further provider call.
- Either way the instruction is marked `completed` (not left stuck at
  `running` forever).
- New tests in `tests/test_runner_local.py` (4 cases): a live `follow_up`
  reaches the next completion request and is marked completed; a live
  `cancel` stops the turn before any provider call is even made; a
  non-live classification (`reprioritise`) is correctly left `queued`, not
  consumed.
- Tests: 454/454 passing. Wheel rebuilt and reinstalled into the real
  system location; `tamfis-code --version` confirms `0.4.14`.
- Version bumped to `0.4.14`.

## Added today, interactive clarifying-question tool (2026-07-18, v0.4.13)

User request, made explicitly after the WordPress/package.json bug above:
the agent should be able to pause and ask a real clarifying question
instead of silently guessing when it's genuinely uncertain about something
only the human can resolve -- the same underlying failure pattern (guessed
Node/React instead of checking or asking) that motivated it.

- New tool `ask_user_question` (`mcp.py`): takes a `question` and an
  optional `options` list. When the round loop has a real attached
  interactive terminal, it prints the question (rich `Panel`, matching the
  approval-gate panel's look) and blocks on `console.input()` for the
  answer -- a numeric reply against a supplied `options` list resolves to
  that option's text, anything else is returned verbatim. When no
  interactive terminal is attached (piped/non-tty/no console wired at
  all), it returns a clear "unavailable, proceed on your best judgement
  and state your assumption" message instead of blocking forever or
  crashing -- same fail-safe-closed default as `resolve_approval_decision`.
- Uses the same suspend/resume-live-status-line discipline around the
  panel+block as the approval gate (see the v0.4.5 ordering fix above), so
  the background spinner can't render between the question and the
  prompt.
- Classified `read_only` in `safety.py` (no workspace side effects, so it
  never goes through the mutation-approval gate -- it has its own
  interactive gate instead) and added to every task-type's tool list in
  `tool_policy.py`, including read-only ones (`READ_TOOLS`, so `audit`/
  `plan`/`chat` tasks -- exactly where the motivating bug happened -- can
  use it too). `runner_local.py`'s `run_local_agent_turn` now constructs
  `MCPServer` with the real `console`/`renderer`/`interactive` it already
  has in scope; every other existing `MCPServer()` call site (tests,
  `tools`/`screenshot` debug commands, the single-shot `execute_command`
  path) is unaffected since all three new constructor params default to
  "unavailable."
- Live-verified: `tamfis-code tools list` shows the new tool registered on
  the real installed binary.
- New tests: `TestAskUserQuestionTool` in `tests/test_mcp.py` (5 cases --
  unavailable with no console at all, unavailable with a console but
  `interactive=False`, free-text answer, numeric option resolution, free
  text still accepted alongside offered options), plus one addition each
  to `tests/test_safety.py` (read-only classification) and
  `tests/test_orchestrator.py` (offered even in read-only/audit mode).
- Tests: 451/451 passing. Wheel rebuilt and reinstalled into the real
  system location; `tamfis-code --version` confirms `0.4.13` on the actual
  `$PATH` binary.
- Version bumped to `0.4.13`.

## Fixed today, WordPress/PHP project detection + banner branding (2026-07-18, v0.4.12)

User report: "It keeps looking for packages.json even when it is WordPress
sites. Again not smart" -- from a real transcript where the objective
literally said "this is a wordpress package, not a react component" and the
agent still went hunting Node/React conventions.

Root cause, confirmed by reading the actual system prompt and metadata
code, not guessed:
1. `workspace.py`'s `build_system_prompt` had a hardcoded project-type-
   detection instruction listing package.json/pyproject.toml/go.mod/
   Cargo.toml/Dockerfile as the things to check for -- composer.json (PHP)
   and any WordPress marker were never in that list at all, and nothing
   told the model to defer to the user's own explicit statement about
   project type over its own guess.
2. `_project_metadata`'s `MANIFEST_LANGUAGE_MAP` is a filename->language
   lookup keyed on manifest files (package.json, pyproject.toml, etc). A
   WordPress install/theme/plugin routinely has **no** package.json or
   composer.json at all (confirmed: a real WP site is often just PHP files
   with no dependency manifest) -- so `detected_languages`/`frameworks`
   came back completely empty, giving the model zero grounding signal and
   leaving Node/React as its only guess.
- Fix: (1) the system prompt's project-type-detection line now explicitly
  lists composer.json (PHP) and wp-config.php/wp-load.php/wp-content
  (WordPress -- noting it often has no manifest at all) alongside the
  existing entries, and explicitly says not to override the user's own
  stated project type with a guess; (2) `_project_metadata` gained real
  marker-based WordPress detection independent of any manifest file: core
  markers (`wp-config.php`/`wp-load.php`/`wp-settings.php`/etc), a
  `style.css` "Theme Name:" header, or a top-level `*.php` file's "Plugin
  Name:" header (top-level only, not a full recursive scan, so this stays
  cheap even on a large real WP install) all mark `PHP`/`WordPress` in
  `detected_languages`/`frameworks`. `_PROJECT_MARKER_NAMES` (used by
  `classify_root`/`has_project_marker` for multi-stack scoping) also
  gained the WordPress core markers, so a bare WP checkout with no
  manifest is now correctly classified `active`, not `unrelated`.
- Live-verified against a real synthetic WordPress checkout (wp-config.php
  + a themed style.css, no package.json/composer.json anywhere):
  `detected_languages` now correctly reports `['PHP']` and `frameworks`
  reports `['WordPress']`, where before both were empty.
- New tests: `WordpressProjectMetadataTests` in
  `tests/test_tamfis_code_workspace.py` (5 tests -- wp-config-only with no
  manifest, theme style.css header, plugin header, a false-positive guard
  confirming an ordinary non-WordPress style.css does NOT trigger, and the
  classify_root/active-not-unrelated case).
- Also added, same user request: a "by Tamfis Nig. Ltd" line under the
  banner title (`render.py`'s `print_banner`) -- live-verified via a real
  pty capture against the installed binary.
- Tests: 445/445 passing. Wheel rebuilt and reinstalled into the real
  system location (`tamfis-code --version` confirms `0.4.12` on the actual
  `$PATH` binary).
- Version bumped to `0.4.12`.

## Fixed today, degenerate-repetition loop (2026-07-18, v0.4.11)

User pasted a real transcript from a live session (`--provider nvidia`,
`nvidia/nemotron-3-super-120b-a12b`, continuing a saved audit against
`/home/finima/www`): after `search_code` came back empty, the model's
"answer" turned into thousands of repetitions of the literal phrase "We
have execute_command? Not listed." -- verbatim, back-to-back -- and the
process was eventually OS-`Killed` (OOM).

Root cause, confirmed by reading the full path end to end:
1. The task classified as `audit`, so `tool_policy.py`'s `allowed_tools`
   correctly restricted it to `READ_TOOLS` only (`execute_command`
   intentionally excluded from audit/read-only tasks -- this part is
   working as designed, not a bug). The model, on discovering no
   `execute_command` tool was offered, didn't gracefully fall back -- it
   spiraled into repeating the same short sentence instead of producing
   real output or reaching a normal stop.
2. That degenerate stream ran all the way to `MAX_TOKENS_PER_REQUEST`,
   which set `finish_reason=="length"` -- indistinguishable, to the
   existing v0.4.8 truncation-continuation logic, from a genuinely
   truncated *legitimate* answer. It re-fed the entire (already huge, still
   growing) garbage back to the model as its own prior turn and asked it to
   "continue", which reinforced the same repetition. This repeated for up
   to `MAX_TRUNCATION_CONTINUATIONS` (6) more rounds, each one larger than
   the last (confirmed in the pasted transcript: "continuation (1/6)",
   "(2/6)", "(3/6)" each visibly longer), until memory exhaustion killed
   the process.
- Fix: `runner_local.py` gained `_DEGENERATE_REPETITION_RE` (a
  backreference regex: a 6-200 char group repeating immediately after
  itself 4+ more times) and checks a bounded rolling tail buffer (not the
  full accumulated content -- O(1) per streamed chunk, not O(n), so a
  genuinely long legitimate answer pays no extra cost) on every
  `assistant_delta` chunk inside `_stream_one_completion`. The moment a
  repeat is confirmed, the provider stream is closed immediately
  (`await stream.close()`) instead of being allowed to run to the token
  cap, `finish_reason` is set to a new sentinel (`"degenerate_repetition"`,
  never equal to `"length"`) so the truncation-continuation `while` loop
  correctly never fires for it, and the returned content is truncated back
  to just before the repeated block began, with a clear caveat appended
  (`_truncate_degenerate_repetition`). Net effect: at most one wasted
  completion call instead of up to seven compounding ones, and the
  process no longer grows unbounded memory on this failure mode.
- Verified no false positives: a 3,000-word genuinely non-repeating answer
  does not trip the detector (checked directly against the installed
  package, not just synthetic unit-test chunks).
- New test: `test_degenerate_repetition_stops_generation_early_instead_of_looping_forever`
  in `tests/test_runner_local.py` -- asserts the round completes with a
  captioned/truncated answer and exactly one completion call was made (no
  continuation loop fired).
- Not attempted here (separate concern, model behavior not a code defect):
  making the model itself handle "tool not offered" more gracefully in
  audit mode. The fix here is the general safety net for ANY provider/model
  getting stuck in a text-generation loop, regardless of what triggers it --
  same category of protection as the existing `_is_cycling` tool-call-loop
  guard, just for raw generated text instead of tool calls.
- Tests: 440/440 passing. `python3 -m build` clean; wheel installed into the
  real system location (`pip install --break-system-packages
  --force-reinstall --no-deps dist/tamfis_code-0.4.11-py3-none-any.whl`,
  confirmed via `tamfis-code --version` on the actual `$PATH` binary, not
  just the source tree).
- Version bumped to `0.4.11`.

## Last verified: 2026-07-18 (re-verification pass, no code changes)

Re-verified the mode-switching / approval-gate / adaptive-reasoning work
already recorded below (v0.4.8-v0.4.10) is genuinely live, not just
unit-tested -- this is exactly the kind of claim that turned out to be
stale before (see the CRITICAL deployment gap note). Findings:

- `tamfis-code --version` on the real `$PATH` binary
  (`/usr/local/bin/tamfis-code`, `/usr/local/lib/python3.13/dist-packages`)
  reports `0.4.10`, matching `pyproject.toml`/`__init__.py`/source tree --
  no deployment drift this time.
- `python3 -m pytest -q`: 439/439 passing.
- Live pty capture of the real interactive REPL: startup banner shows
  `Mode: interactive   Approval: ask`, prompt renders `tamfis-code
  [manual]>`. Three real Shift+Tab (`\x1b[Z`) keypresses sent to the pty
  cycled the visible indicator `[manual] -> [accept-edits] -> [auto] ->
  [plan]` exactly as designed -- confirmed byte-for-byte in captured
  output, not inferred from code reading alone.
- No new bugs found in this pass. Nothing below this line needed a code
  change.

## Last verified: 2026-07-17 ~16:45 WAT

Verified by direct inspection of the running code/services in this
environment, not by re-reading prior reports.

- Version: 0.4.10 (see the "Fixed today..." sections below --
  `__init__.py`'s `__version__` had drifted from `pyproject.toml`'s
  declared version; both are kept in sync now, guarded by
  `tests/test_version_consistency.py`). Package builds clean (wheel + sdist
  in `dist/`, `SHA256SUMS` present). Not yet published anywhere (no git
  commit exists anywhere in this checkout -- `git status` at the repo root
  shows zero commits). **Installed into the real system location this
  time** (`/usr/local/lib/python3.13/dist-packages`, what
  `/usr/local/bin/tamfis-code` on `$PATH` actually runs) -- see the
  "CRITICAL deployment gap" note below; earlier version bumps in this file
  were only ever verified against the source tree and a throwaway venv,
  never the real installed command.
- Tests: 439/439 passing (`python3 -m pytest -q`).
- `tamfis-code doctor`: all four providers (nvidia, hf, openrouter, ollama)
  configured and reachable; local session bootstrap works.
- CLI surface: ~40 top-level commands, no subgroups (the `session` group was
  removed today -- see Fixed, below).
- Playwright chromium **is** installed (`~/.cache/ms-playwright/chromium-1223`
  etc.) -- contradicts the 2026-07-16 audit's claim that the browser runtime
  was missing. Already fixed by the time that audit's finding would matter.
- Hard-coded OAuth credential defaults in `tier_ii_gateway/dependencies/auth.py`
  (flagged P0 in the 2026-07-16 audit) are **already fixed** -- all client
  ID/secret values now come from `os.getenv(...)` with no source-level
  default.

## Fixed today (2026-07-17)

1. **Duplicate/orphaned session system.** `sessions.py`'s `SessionManager`
   (SQLite, uuid ids) backed a `session` command group (`create`/`list`/
   `resume`/`delete`/`fork`/`clean`) that was never populated by any real
   `ask`/`agent`/`chat`/interactive code path -- only `state.py`'s
   `local_state` (int ids) is. `session resume` was a literal stub. Removed
   the dead group and module entirely; the real system (`init`/`resume`/
   `sessions`/`attach`/`status`) already covers this correctly. 10 tests
   removed with it (they only tested the dead module). 363 tests remain,
   all passing.
2. **Fabricated shell completions.** `completion.py`'s command list was
   hand-written and wrong -- `edit`/`review`/`ingest`/`version` don't exist
   as commands; most real commands (`agent`, `resume`, `diffs`, `plans`,
   `tools`, ~30 others) were missing. Rewritten to introspect the real
   Click CLI (`cli.commands`), so it can't drift out of sync again.

## Fixed today, continued (2026-07-17)

3. **Tier IV routing contract mismatch -- fixed.** `tier_iv.py`'s
   `TierIVRouter` POSTed to `{base_url}/api/v1/orchestration/route`
   expecting a routing *decision*; the real, live Tier IV service in
   `tamgpt6` (`tier_iv_orchestration/tamgpt_api.py`, port 9555, confirmed
   running) only ever exposed `/health`, `/v1/chat/completions`,
   `/chat/completions`, `/chat/stream` -- confirmed via direct `curl`,
   `/api/v1/orchestration/route` returned 404. Effect before the fix: Tier
   IV routing silently no-op'd (safe fallback to local provider selection)
   whenever `TAMFIS_TIER_IV_URL` was configured.
   - Fix: `tier_iv.py` deleted. Tier IV is now a genuine fifth entry in
     `providers.py`'s `ProviderManager.PROVIDERS`/`PRIORITY_ORDER`
     (`ProviderType.TIER_IV`), calling the real `/v1/chat/completions`
     endpoint like nvidia/hf/openrouter/ollama already do, gated by a new
     `_check_tier_iv_available()` (opt-in only -- `TAMFIS_TIER_IV_URL` or
     `TAMFIS_TIER_IV_ENABLED=true`, plus a live `/health` probe -- so a
     default install/test run never touches localhost:9555 unasked;
     confirmed offline test runs still don't probe it).
   - Also fixed in the same pass: `--provider` accepted `gemini`/`apiframe`
     as valid `click.Choice` values in the `ask`/`agent`/`chat`/`audit`/
     `plan`/`exec`/`execute-plan` commands, neither of which was ever a
     real provider alias -- both passed CLI validation and then always
     crashed with "Unknown local provider" inside `resolve_provider_type`.
     The Choice list is now derived from the real alias table
     (`local_chat._PROVIDER_ALIASES`) instead of hand-typed, and `tier_iv`/
     `tier4` were added to that table as real aliases.
   - Live-verified end to end, not just unit-tested: with
     `TAMFIS_TIER_IV_URL=http://127.0.0.1:9555`, `tamfis-code providers`
     and `doctor` correctly show Tier IV available and auto-selected;
     `tamfis-code chat "reply with exactly PONG" --provider tier_iv`
     round-tripped through the real port-9555 service and returned `PONG`.
     Without the env var set, Tier IV correctly shows unavailable and
     default routing is unaffected (falls through to nvidia as before).
   - Known minor gap, not yet fixed: the CLI's route-display shows
     "Model: unknown" for Tier IV turns (its `default_model` is
     intentionally `""`, deferring model choice to Tier IV's own
     orchestrator) -- the real model Tier IV used isn't plumbed back from
     the response into the displayed route record. Cosmetic only; the
     actual completion is correct.
   - Tests: 361/361 passing (2 removed with `tier_iv.py`'s own dead unit
     tests, which tested the old, now-removed contract).

## Fixed today, continued further (2026-07-17, later pass)

4. **`tamfis-code enforce` tested the wrong project entirely -- fixed.**
   `enforcer.py`'s `TestEnforcer` hard-coded `self.base = Path("/home/tamfisgpt")`,
   `self.backend = self.base / "tamgpt6"`, `self.frontend = self.base /
   "tamfis-frontend"` -- so running `tamfis-code enforce` from *any*
   workspace (including tamfis-code's own repo, or any other project on any
   other machine) silently ignored the actual workspace and instead ran
   tamgpt6's ~700-test suite and looked for a sibling `tamfis-frontend`
   directory. The old test suite even asserted this as intended behavior
   (`test_backend_and_frontend_paths_are_siblings_under_base`). For a
   standalone CLI whose whole pitch is "point it at any repo," this made
   `enforce` non-functional everywhere except one specific developer's home
   directory.
   - Fix: `TestEnforcer` now takes the real, resolved `workspace_root`
     (same one every other command uses via `ctx.obj["workspace_root"]`,
     respecting `--cwd`) and auto-detects what to run from it, reusing
     `workspace._project_metadata`'s existing manifest-based detection
     (`pyproject.toml`/`pytest.ini` -> pytest, `package.json` -> npm,
     `Cargo.toml` -> cargo) instead of assuming a fixed sibling layout.
     `enforce_cmd` now takes `@click.pass_context` to get the resolved root.
   - Tests rewritten to match (`tests/test_enforcer.py`) -- including a
     regression test asserting the workspace root is honored rather than a
     hard-coded path. 362/362 passing.
   - Live-verified: `tamfis-code enforce --python` run from tamfis-code's
     own repo now correctly discovers and runs tamfis-code's own 32 test
     files (confirmed real per-file pass/fail output), not tamgpt6's.

5. **`tamfis-code index` indexed stale packaging output -- fixed.**
   `indexer.py`'s ignore list was missing `build`/`dist` -- confirmed live:
   `tamfis-code index . -s ProviderManager` returned the same class twice,
   once from the real source and once from a stale pre-edit copy under
   `build/lib/tamfis_code/providers.py` (left over from an earlier `pip
   install -e .` / wheel build). Added `build`/`dist` to the ignore list
   (matching `workspace.py`'s existing `IGNORED_PARTS`, which already
   excluded them for the same reason). Live-verified: re-indexing dropped
   from 108 files (with the duplicate) to 69, and the search now returns
   exactly one correct hit. 362/362 tests still passing.

6. **`tamfis-code screenshot <url>` always crashed -- fixed, now actually
   works.** `mcp.py`'s `get_browser_tool_class()` is documented to return
   `None` (never raise) when a tamgpt6 monorepo checkout isn't co-located,
   and explicitly tells callers to "report that clearly rather than crash."
   `cli.py`'s `screenshot_cmd` ignored that contract and called
   `get_browser_tool_class()().execute(...)` with no `None` check --
   confirmed live: `tamfis-code screenshot https://example.com` crashed
   with the bare `'NoneType' object is not callable`, no matter what.
   Root cause of *why* it was always `None`: the "co-located" search only
   walked upward from tamfis-code's own install location looking for a
   `tier_iv_orchestration` directory directly, so it could only ever find
   tamgpt6 if tamfis-code were running from *inside* a tamgpt6 checkout --
   never the common case (confirmed as this environment's own real layout)
   of tamgpt6 and tamfis-code sitting side by side as sibling directories.
   - Fix: `_import_monorepo_attr` now also checks each ancestor's `tamgpt6`
     child directory (plus an explicit `TAMFIS_MONOREPO_ROOT` override), so
     the sibling-checkout layout is actually found. `screenshot_cmd` now
     also has an honest `None` check with a clear message, for whenever a
     monorepo genuinely isn't available at all.
   - Live-verified end to end: `tamfis-code screenshot https://example.com`
     now finds the real tamgpt6 `BrowserTool` and produces a genuine
     1920x1080 PNG (confirmed with `file`) instead of crashing. 362/362
     tests still passing.

7. **The orchestrator falsely flagged nearly every plain chat answer as
   "Validation incomplete" -- fixed.** Found by actually reading through
   `orchestrator/engine.py`/`validator.py`/`planner.py` end to end (the
   "newly built" 0.4.0 orchestration layer from today's `UPGRADE_REPORT.md`)
   rather than just knowing it existed. Two compounding bugs:
   1. `routing.py`'s `classify_task` set `requires_tools=read_only` for the
      generic `QUESTION` fallback -- conflating "tools are *permitted* in
      this mode" with "this task *requires* tool evidence to be valid." A
      trivial question like "reply with exactly PONG" got marked as
      requiring tool evidence purely because it ran in `chat` mode, even
      though answering it needs no tool at all.
   2. `orchestrator/validator.py`'s `tool_evidence_recorded` check could
      fail the whole report without ever appending anything to
      `unresolved` -- unlike the other two checks, which do explain
      themselves. `runner_local.py`'s caveat is built purely from
      `"; ".join(validation.unresolved)`, so the failure rendered as a
      bare, unexplained `⚠ Validation incomplete: ` with nothing after the
      colon (confirmed live, twice, in the Tier IV verification pass
      above -- and reproduced again identically with the `nvidia`
      provider, proving it wasn't Tier-IV-specific).
   - Fix: (1) the `QUESTION` fallback no longer derives `requires_tools`
     from `read_only`; (2) `tool_evidence_recorded` now explains itself in
     `unresolved` when it fails, as defense in depth for any other future
     case that hits it.
   - Two regression tests added to `tests/test_orchestrator.py`. Live-
     verified: the exact `chat ... --provider tier_iv` "PONG" turn that
     originally surfaced this now completes with no caveat at all.
     364/364 tests passing.

8. **`rm -rf` risk detection missed common flag spellings -- fixed.** Read
   through `safety.py` (the local risk classifier -- safety-critical, gates
   what gets auto-approved) carefully rather than skimming it. The single
   regex for detecting a dangerous recursive-force delete only matched
   combined short flags (`-rf`/`-fr` and letter-order variants of *that
   combined token*) -- confirmed live: `rm -r -f x`, `rm -f -r x`,
   `rm --recursive --force x`, `rm -r --force x`, and `rm --force -r x`
   (all equally common, realistic ways to write the same command) were
   silently classified as only `medium` risk instead of `dangerous`.
   - Fix: replaced the single regex with `_is_dangerous_rm`, which checks
     for a recursive flag and a force flag anywhere in the command's
     whitespace-split tokens, independent of order/spacing/combination/
     long-vs-short form. Same threshold as before (still requires *both*
     recursive and force together -- `rm -r somedir` or `rm -f file` alone
     stay `medium`, unchanged), just order/format-agnostic detection of
     that same combination. Verified no false positives on lookalike words
     (`confirm -r -f`, `term -r -f` don't match `\brm\b`).
   - 6 new regression tests in `tests/test_safety.py` (3 for the missed
     variants, 1 confirming the unchanged medium-risk threshold, 1 for the
     lookalike-word check). 367/367 tests passing, no regressions.

9. **Added: real "thought for Xs" + rotating tips in the live status line
   (user-requested, matching Claude Code CLI's own
   `Brewing… (2m 18s · ↓ 8.9k tokens · thought for 12s)` / rotating `Tip:`
   line).** tamfis-code already had a live spinner status line
   (`render.py`'s `StreamRenderer`, gated on being a real TTY) with elapsed
   time and token count -- it just never showed reasoning duration or tips.
   - Confirmed live against the real NVIDIA NIM API that `reasoning_effort`
     (already sent for nvidia/openrouter/tier_iv) genuinely triggers a
     separate `reasoning_content` stream delta ahead of the real answer --
     this was being silently dropped everywhere it already flowed. Added
     `EventType.REASONING_DELTA`, extraction in `provider_protocols.py`,
     and threading through `runner_local.py`'s primary + first-fallback
     streaming call sites (`REASONING_EFFORT_CAPABLE_PROVIDERS` promoted to
     a shared constant in `providers.py` so this and the pre-existing
     `chat_completion` usage can't drift into two different lists). Live-
     verified end to end with `--debug` against real NVIDIA output: the raw
     reasoning text prints separately, the final answer stays clean, timing
     freezes once real content starts and stays visible in the status line
     for the rest of the turn (does not keep incrementing forever).
   - Added a `_TIPS` rotating-hint line (starts after 4s elapsed, rotates
     every 8s) with 12 tips. Every single one was individually re-verified
     against real code/live behavior before shipping, not just written from
     memory -- caught and fixed two inaccurate ones in review: a Ctrl+C tip
     claiming turn-only cancellation (actually exits the whole process in
     standalone/local mode -- the SIGINT-cancels-just-the-task behavior only
     exists in the legacy `--remote` path), and a `/compact` tip claiming it
     "condenses conversation history" (it actually just saves a checkpoint
     bookmark, no token/context reduction happens at all). Also added a
     test (`test_all_tips_reference_real_commands_only`) that fails if any
     future tip names a command that isn't actually registered in the CLI,
     specifically to stop this class of drift from recurring the way
     `completion.py`'s fabricated command list once did.
   - 12 new/updated tests across `test_provider_protocols.py` and
     `test_tamfis_code_render.py`. 376/376 tests passing throughout, no
     regressions. Live-verified twice through the real CLI end to end.

## Verified working, no fix needed (ruled out during this pass)

- `runner.py`'s `--remote` provider list (`gemini`/`apiframe`/etc) looked
  like the same "hand-typed, drifted" bug as the standalone `--provider`
  Choice list, but isn't -- checked against tamgpt6's actual
  `tier_ii_gateway/api/remote.py` `RemoteTaskIn` Pydantic schema, which
  genuinely accepts those exact literal values server-side. Same for the
  `mode` Choice list. Left unchanged.
- `agent-cmd delegate`: gated behind `enable_subagent_delegation`/
  `TAMFIS_CODE_ENABLE_SUBAGENT_DELEGATION=1`, off by default -- this is an
  honest, typed gate (clear message telling you how to enable it), not a
  broken stub. Live-verified it actually works end to end once enabled
  (real file inspection, real tool calls, real completion).
- `mcp.py`/`agents.py`/`metrics.py` exports genuinely wired into
  `interactive.py`/`runner_local.py`/`render.py` (unlike the removed
  `sessions.py`) -- confirmed via direct import-site grep.

## Fixed today, continued further still (2026-07-17, later pass, v0.4.5)

A large rebuild spec (context compaction, internal rollover, workspace
scoping, truthful tool results, visible approval gate, loop detection) was
handed in again mid-session as if none of it existed yet. It does exist --
this is exactly the trap this file's header warns about (nearly re-planned
off two stale reports on 2026-07-16/17 already). Verified against the
actual code before touching anything:

- Workspace scoping (`_detect_workspace_scope`/`classify_root`/
  `_scope_tool_arguments` in `runner_local.py`/`workspace.py`), context
  compaction + internal rollover + evidence retrieval + larger-context
  provider fallback (`_trim_tool_outputs`/`_perform_context_rollover`/
  `evidence.py`/`retrieve_evidence` tool), truthful tool-result
  normalisation (`_semantic_tool_failure`/`_normalise_tool_result`),
  repeated-tool-loop detection (`_is_cycling` + consecutive-identical
  guard), and the Tier IV decommissioning were all already implemented and
  covered by real tests (`test_workspace_scope.py`, `test_mcp_search_bounds.py`,
  `test_context_rollover.py`) from earlier passes today. Not re-implemented.
- One real bug found and fixed: `_compact_json_value`'s string-compaction
  branch (`runner_local.py`) embedded its size marker as text inside a
  plain string (`"...[N chars omitted]..."`) instead of the structured
  `{"_tamfis_compacted": true, ...}` object the spec's own already-written
  test (`test_context_rollover.py::test_300k_char_tool_call_argument_stays_valid_json_after_compaction`)
  expected -- 398/399 tests passing before this fix, 399/399 after.
- Version drift found and fixed: `tamfis_code/__init__.py`'s `__version__`
  was hard-coded to `"0.4.2"` while `pyproject.toml` had already moved to
  `0.4.4` -- confirmed live, a freshly built-and-installed wheel's
  `tamfis-code --version` reported the stale `0.4.2`. Nothing kept the two
  in sync since pyproject.toml's version isn't `dynamic`. Fixed, bumped
  both to `0.4.5`, and added `tests/test_version_consistency.py` so this
  class of drift fails CI instead of silently shipping again.
- **Visible approval gate had a real ordering bug -- fixed.** Confirmed
  live via a pty capture (raw terminal bytes, not just unit tests): the
  live status line was suspended *after* the approval panel already
  printed (both in `runner_local.py`'s local loop and `runner.py`'s
  `_stream_task` remote-event handling), only immediately before the
  blocking `resolve_approval_decision` prompt. Live's background refresh
  thread redraws on its own timer independent of anything else writing to
  the console, so a stray spinner frame ("⠴ Waiting… (0s)") could still
  render *between* the approval panel and the "Approve?" question --
  confirmed present before the fix and gone after, via two pty captures
  (raw bytes preserved in the fix commit's test). Fix: suspend the live
  status line before the panel event is even emitted, not just before the
  prompt, in both call sites; resume only after the decision is made.
  Regression test: `test_live_status_line_is_suspended_before_the_approval_panel_prints`.
- Added the 3 remaining spec-listed regression-test scenarios that had no
  dedicated coverage yet: nested semantic tool failure inside a
  transport-success envelope (`test_nested_semantic_failure_inside_a_transport_success_envelope_is_reported_as_failed`),
  continuation recovery after an empty post-tool provider response
  (`test_empty_provider_continuation_after_a_tool_round_is_recovered_not_treated_as_done`),
  and the visible-approval-gate ordering test above.
- Tests: 406/406 passing. `python3 -m py_compile` and `python3 -m build`
  both clean; wheel installs and imports correctly from a venv outside the
  source tree; `tamfis-code --version` now correctly reports `0.4.5`.

## Fixed today, yet another pass (2026-07-17, evening, v0.4.6)

User report: the agent still "cuts short" mid-task reporting a context-
length-exceeded error, instead of using the rollover/continuation machinery
above. Investigated live (a fork with fresh eyes, not from memory) rather
than assuming the earlier pass already covered this. It didn't -- found
**three compounding bugs**, all in the same failure mode: a large objective
(a big pasted log/diff/paste as the request, or one that accumulates size
over a task) could blow the token budget in a way none of the existing
compaction/rollover machinery could actually fix, because none of it ever
looked at the objective text itself:

1. **`_trim_tool_outputs`/`_trim_message_in_place` never compacted
   `role=="user"` content at all**, ever -- only `tool`/`assistant`. A huge
   objective (as the sole message, before any tool call) was structurally
   immune to compaction.
2. **The rollover gate required prior tool/assistant history
   (`has_tool_history`)** -- so a huge objective blowing the budget on
   round 1, before any tool call has happened yet, skipped rollover
   entirely and fell straight to failure.
3. **`orchestrator/context.py`'s `build_context_bundle` embedded the full,
   unbounded objective a SECOND time** inside the leading system message's
   "Active orchestration context" supplemental text, and a THIRD time
   inside the plan dict's own `objective` field (`f"Active plan: {plan}"`,
   for any plan-worthy/high-complexity task, e.g. "fix"/"debug"). Neither
   copy is reachable by compaction, since `role=="system"` content is
   deliberately left untouched (it carries essential workspace
   instructions that must survive compaction) -- so even after fixing (1)
   and (2), a plan-worthy task with a huge objective still failed, because
   two full copies of the same 300K-character text remained, invisible to
   every trimming pass. This is what made the earlier "context-window
   failure" fix (above, from the initial spec pass) look complete in tests
   but still fail live: none of that pass's tests used a large `user`-role
   message or a plan-worthy classification together.
   - Fixes: (1) added a `role=="user"` branch to `_trim_message_in_place`
     for old user turns, plus a final last-resort pass in
     `_trim_tool_outputs` that bounds even the *current* (latest) user
     message if nothing else got the turn under budget (full text is never
     lost -- only what's sent to the provider is shortened); (2) removed
     the `has_tool_history` gate -- rollover is safe and useful regardless
     (worst case it persists the oversized objective as evidence and
     rebuilds a genuinely smaller continuation); `_perform_context_rollover`
     itself was also re-embedding the full objective unbounded in its new
     continuation (defeating its own purpose -- the "smaller" continuation
     was roughly the same size as what was rolled over), now bounded there
     too, full text still retrievable via `retrieve_evidence`; (3)
     `build_context_bundle` now uses a 400-char objective preview in the
     supplemental text and in the plan dict's `objective` field, instead of
     the full text -- the real, complete objective is already present
     exactly once, as the actual latest user message.
   - Verified live end-to-end (not just unit tests): a 300,000-character
     objective that previously failed immediately on round 1 with
     "Stopping before round 1: ~150,723 estimated tokens..." now completes
     normally in round 1 after compaction alone (~1,475 estimated tokens,
     no rollover even needed for that case).
   - Tests: `test_huge_single_objective_is_compacted_instead_of_failing_round_one`,
     `test_context_budget_exceeded_stops_before_request` (now additionally
     asserts a `context_rollover` event actually fires, not just that
     failure is total, for a budget rollover genuinely cannot rescue),
     `test_rollover_bounds_a_huge_objective_instead_of_reproducing_the_same_size`,
     `test_huge_objective_is_not_duplicated_unbounded_into_the_system_prompt`.
     409/409 passing.
- Version bumped to `0.4.6` (behavior fix, not just tests/docs).

## Fixed today, yet another pass still (2026-07-18, v0.4.7)

User report, with a real transcript: ran `tamfis-code` from inside its own
repo (`/home/tamfisgpt/tamfis-code`) with the objective "audit the
TamfisGPT iOS full stack. Identify it and do not touch tamfis-code" -- the
very first thing it did was scope to `/home/tamfisgpt/tamfis-code` and
start reading `README.md` there. The exact thing it was told not to do.

Root cause, confirmed by direct reproduction (`_detect_workspace_scope`
called with the real objective against the real directory layout, not
guessed): two compounding gaps in `runner_local.py`.

1. **Nothing anywhere parsed a "do not touch X" / negative instruction.**
   `_detect_workspace_scope` only ever had positive-selection logic
   (explicitly-named roots, the "stack" shortcut) -- there was no concept
   of an excluded root at all, so naming tamfis-code specifically to rule
   it out had zero effect on the resolved scope.
2. **The "stack" shortcut only ever looked at the launch directory's own
   children.** `by_name` is built from `Path(workspace_root).iterdir()` --
   when `workspace_root` IS one of the three canonical stacks
   (`tamgpt6`/`tamfis-code`/`tamfis-frontend`), its own children can never
   include its *siblings*, so the shortcut silently found nothing and fell
   through to `if _is_project_root(root): return [root]` -- tamfis-code
   itself, exactly backwards from "do not touch tamfis-code."
   - Fix: added `_excluded_root_names()` (matches negation trigger phrases
     -- "do not touch", "don't modify", "except", "excluding", etc. --
     against a bounded window of text after the trigger, so an unrelated
     later mention of the same name doesn't retroactively un-exclude it)
     and threaded exclusion through every selection path in
     `_detect_workspace_scope`: explicit-name matching, the "stack"
     shortcut, and the final launch-directory fallback (which now refuses
     to default to `[root]` when `root` itself is excluded, falling back to
     non-excluded active *sibling* projects instead). The "stack" shortcut
     and explicit-name matching now also check the launch directory's
     siblings (`root.parent.iterdir()`), not just its own children, so a
     stack named in the objective is reachable even when it isn't nested
     inside the directory tamfis-code happens to be running from.
     `_scope_tool_arguments`'s existing enforcement (reject any path not
     `_is_within` a resolved scope root) needed no changes -- once an
     excluded root is correctly absent from `scope_roots`, the existing
     mechanism already blocks it.
   - The "Focused workspace scope" diagnostic now also states which name(s)
     were excluded per the objective, so the exclusion is visibly
     confirmed, not just silently applied.
   - Live-verified against the real repo layout (not just synthetic temp
     dirs): `_detect_workspace_scope("/home/tamfisgpt/tamfis-code", "audit
     the TamfisGPT iOS full stack. Identify it and do not touch
     tamfis-code")` now resolves to `[tamgpt6, tamfis-frontend]`, excluding
     tamfis-code entirely.
   - Known remaining limitation, not fixed: "TamfisGPT iOS" does not exist
     as an actual project directory anywhere in this environment -- only a
     loose top-level document, `/home/tamfisgpt/TamfisGPT IOS
     _Upgrade..txt`. Loose top-level files (as opposed to project-root
     subdirectories) are still outside every resolvable scope
     configuration; "identify it" for a case like this still requires the
     user to point at the right file/location directly. Not attempted here
     -- out of scope for this fix, which was specifically about honoring an
     explicit exclusion instruction.
   - Tests: `test_do_not_touch_excludes_the_launch_directory_and_routes_to_siblings`,
     `test_excluded_root_name_is_never_selected_even_when_explicitly_named`,
     `test_excluded_root_names_matches_within_a_bounded_window_of_the_trigger`.
     412/412 passing.
- Version bumped to `0.4.7`.

## Fixed today, two more (2026-07-18, v0.4.8)

Two more user reports, same session.

1. **A long final answer (e.g. a full-stack audit) could trail off
   mid-sentence and still be accepted as "done."** User pasted a real
   transcript: a reasoning-heavy audit response stopped mid-paragraph with
   no closing summary. Root cause: `provider_protocols.py`'s
   `normalize_stream_chunk` already computes `finish_reason` from the
   provider's own `DONE` signal (`"length"` means the completion was cut
   off by `MAX_TOKENS_PER_REQUEST`, not that the model was actually done)
   and wraps it in a `DONE` canonical event -- but `_stream_one_completion`
   in `runner_local.py` never had a branch for that event type at all, so
   it was computed and immediately discarded. A non-empty, no-tool-calls
   response with `finish_reason=="length"` was indistinguishable from a
   genuinely complete one, so a mid-sentence partial was accepted as the
   final answer with no attempt to continue it -- unlike the existing
   empty-continuation recovery, which only ever handled the *opposite*
   case (zero content).
   - Fix: `_stream_one_completion` now returns `finish_reason` as a third
     value (all 4 call sites updated). When the "final answer" branch sees
     `finish_reason=="length"`, it now asks the model to continue exactly
     where it left off (`tools=[]` -- purely "keep writing", never a new
     tool-call opportunity) and appends the result, repeating up to
     `MAX_TRUNCATION_CONTINUATIONS=6` times until it naturally reaches
     `finish_reason!="length"`. If still truncated after the cap, the
     answer is labeled partial rather than presented as if it were
     complete. The empty-continuation recovery path (a separate, pre-
     existing mechanism) doesn't track `finish_reason` internally, so
     `finish_reason` is explicitly reset to `None` after it runs --
     otherwise a stale flag from the round it just recovered from could
     misfire against unrelated recovered content.
   - Tests: `test_truncated_final_answer_is_continued_not_accepted_as_complete`,
     `test_truncated_final_answer_gives_up_after_the_continuation_cap`.

2. **No visible/quick way to see or switch the approval mode.** A `/mode`
   command already existed (manual/accept-edits/auto/plan), but nothing
   showed the CURRENT mode anywhere persistent -- only the one-time startup
   banner or an on-demand `/status`/`/mode` (with no argument) query. User
   expected Claude-Code-style behavior: an always-visible indicator plus a
   quick keypress to cycle, not a command you have to already know exists.
   - Fix: `config.py` gained `mode_label_for_policy()` (raw policy ->
     short /mode name, falling back to the raw value for
     `--approval`-only policies with no short alias) and
     `next_mode_in_cycle()` (manual -> accept-edits -> auto -> plan -> ...,
     starting fresh from manual if the current policy isn't in the named
     cycle at all). `interactive.py`'s REPL prompt is now a dynamic
     `HTML` callable (prompt_toolkit re-evaluates callables on redraw)
     showing `tamfis-code [mode]>` instead of a static string, and a new
     Shift+Tab (`"s-tab"`) key binding cycles the mode and calls
     `event.app.invalidate()` to redraw the indicator immediately, without
     leaving the input line. `/mode`'s own output and the startup hint
     line now mention Shift+Tab too.
   - Tests: `ModeCycleTests` in `tests/test_tamfis_code_config.py` (5 new
     tests covering the label mapping and cycle order/wraparound/fallback).
- Tests: 419/419 passing. `python3 -m py_compile`/`python3 -m build` clean;
  wheel installs and imports correctly from a venv outside the source
  tree; `tamfis-code --version` reports `0.4.8`.
- Version bumped to `0.4.8`.

## Fixed today, two more still (2026-07-18, v0.4.9)

1. **The standalone interactive REPL was provider-side amnesiac across
   turns -- fixed.** User report: "the system is not thread and context
   aware," clarified as "loses earlier conversation" within the same
   session. Confirmed live: all three `run_local_agent_turn` call sites in
   `interactive.py` (plain objective, `/execute-plan`, `/retry`) built
   `messages` as a fresh single-element `[{"role": "user", "content":
   objective}]` on every turn -- no prior user/assistant exchange was ever
   attached. `orchestrator/context.py`'s `build_context_bundle` even
   computes `layers["relevant_prior_turns"] = conversation_messages[-12:]`
   already, but that value was informational-only and never actually used
   to build the real prompt sent to the provider. Worst concretely: a
   follow-up like "yes" gets expanded by `contextualize_short_reply` into
   "Yes. Proceed with the action or next step you just proposed." and sent
   as the model's ENTIRE conversation -- referring to a proposal the model
   had literally never seen.
   - Fix: `interactive.py` now keeps an in-memory `conversation_history`
     (bounded to `MAX_STANDALONE_HISTORY_TURNS=30` turns) across the REPL
     loop, appended via `_append_turn_to_history` after every completed
     standalone turn (a failed/denied/cancelled turn still records its own
     objective, just with no paired assistant answer), and prepended to
     `messages` on every subsequent turn. Oversized old turns within that
     history are still handled by `runner_local.py`'s existing compaction
     (including the recent role=="user" compaction fix) and rollover --
     nothing new needed there.
   - Tests: `test_second_turn_includes_the_first_turns_conversation_in_messages`,
     `test_failed_turn_still_records_its_objective_without_a_missing_answer`
     in `tests/test_interactive_standalone.py`.
2. **The approval gate's (y)es/(n)o/(a)lways prompt could never actually
   appear for any one-shot command -- fixed.** Separate user report:
   "Approval gate yes and no doesn't appear visible." Root cause:
   `cli.py`'s `_run_ai_command`/`_run_local_ai_command` (the shared
   implementation behind `agent`/`ask`/`chat`/`audit`/`plan`) and the
   `run`/`retry` commands all hardcoded `interactive=False` -- literally
   labeled "one-shot commands never block on a human prompt" -- regardless
   of whether a real human was sitting at an attached terminal watching it
   stream. `resolve_approval_decision`'s default "ask" policy short-circuits
   straight to `deny` when `interactive=False`, so the approval PANEL
   (command/cwd/reason/risk) still rendered, but the actual "(y)es/(n)o"
   question was never reachable at all -- every risky/dangerous action was
   silently denied, no matter what. `cli.py`'s own `local_command` (a
   different, legacy entry point) already had the correct pattern
   (`interactive=sys.stdin.isatty()`), just never applied to the primary
   command surface.
   - Fix: all 5 hardcoded `interactive=False` sites in `cli.py` (lines
     covering `_run_ai_command`, `_run_local_ai_command`, `run`, `retry`)
     now use `interactive=sys.stdin.isatty()`, matching the existing
     correct precedent -- piped/redirected/CI invocations (`isatty()==False`)
     still safely auto-deny as before, no regression there.
   - Tests: `test_standalone_chat_is_interactive_when_stdin_is_a_real_tty`,
     `test_standalone_chat_stays_non_interactive_when_stdin_is_piped` in
     `tests/test_cli_commands.py`.
- Tests: 423/423 passing. `python3 -m py_compile`/`python3 -m build` clean;
  wheel installs and imports correctly from a venv outside the source
  tree; `tamfis-code --version` reports `0.4.9`.
- Version bumped to `0.4.9`.

## CRITICAL deployment gap found and fixed (2026-07-18, same session)

The user kept reporting that fixes from this session ("approval gate still
not visible" *after* the ordering fix, then again *after* the
`sys.stdin.isatty()` fix) weren't taking effect. Root cause had nothing to
do with the code itself: **the real `tamfis-code` on `$PATH`
(`/usr/local/bin/tamfis-code`, backed by `/usr/local/lib/python3.13/dist-packages`)
was a completely separate install, stuck at version 0.4.2/0.4.4, that had
never once been updated with any change made in this checkout.** Every
fix this session was verified against `python3 -m pytest` (which imports
the source tree directly) and a disposable venv at `/tmp/tamfis_verify_venv`
-- the actual command the user runs was never touched. `python3 -c "import
tamfis_code"` happened to resolve to the source tree too (cwd on
`sys.path`), which masked this during earlier verification -- only
`pip show tamfis-code` / `which tamfis-code` / running the real
`tamfis-code --version` binary revealed the mismatch.

Fixed by installing the built wheel directly into the real system location
(`pip install --break-system-packages --force-reinstall --no-deps
dist/tamfis_code-*.whl`, confirmed necessary and sufficient here: the
existing install already lived in system site-packages this way, not a
venv). **Going forward: every version bump in this file must end with
this same reinstall, not just a build into `dist/` or an install into the
throwaway verification venv, or the user will keep running stale code
indefinitely.**

## Fixed today, adaptive orchestration (2026-07-18, v0.4.10)

User: "the orchestrator should be very smart" -- specifically, adapting
and reasoning, not a fixed plan template. Confirmed by reading
`orchestrator/planner.py`: `create_plan()` returned the exact same
generic steps ("Inspect the relevant repository context and manifests" /
"Select a capable provider/model..." / "Execute the requested work...")
for every plan-worthy task regardless of what was actually being asked --
a one-line typo fix and a full-stack audit got literally the same "plan."
It was also purely embedded as system-prompt text; nothing ever tracked
or acted on it programmatically, and it was NEVER revised once created,
no matter what the agent actually found while working.

- **Real, task-specific plans.** `orchestrator/planner.py` gained
  `build_reasoning_plan_prompt()`/`parse_reasoning_plan()` (pure,
  independently tested functions: build a one-shot planning prompt from
  the real objective + real workspace facts -- detected languages,
  frameworks, test/build commands, already known from
  `discover_local_repository`, no extra tool calls needed -- and parse a
  JSON `{"steps": [...], "assumptions": [...], "risks": [...]}` response
  into an `ExecutionPlan`, tolerating markdown code fences, never
  raising). `runner_local.py`'s new `_attempt_reasoning_plan()` calls this
  via a tool-free, non-visible (`emit=False`) completion request right
  after the provider/model routes and before the round loop starts, for
  any `should_plan(profile)`-eligible task (audit/edit/debug/test/mixed or
  high complexity) -- gated so read-only-but-plan-worthy tasks (e.g.
  `audit`/`plan` CLI modes) still get a real plan. On ANY failure
  (malformed JSON, empty steps, provider error) it silently keeps
  whichever plan was already in effect -- `orchestrator.begin()`'s
  synchronous deterministic template is always made first specifically as
  this fallback, so a planning failure never blocks or degrades the turn.
- **Adaptive replanning.** The initial plan is necessarily a guess -- made
  before any tool has actually run. Once the first round with real
  tool_calls has executed, `runner_local.py` triggers exactly one
  revision, grounded in `_summarise_progress_for_rollover()`'s real
  evidence summary (files inspected/modified, tool outcomes, unresolved
  issues -- already existed for context-rollover continuations, reused
  as-is here). Bounded to once per turn via `replanned_after_evidence`
  (a course-correction, not continuous re-planning) -- matches this
  codebase's existing bounded-single-extra-pass pattern
  (`MAX_CONTEXT_ROLLOVERS_PER_TURN`, `MAX_EMPTY_CONTINUATION_RETRIES`,
  `MAX_TRUNCATION_CONTINUATIONS`).
- Both the initial and revised plan are inserted into `working_messages`
  as their own system message (not just baked into the original leading
  system prompt, which is never rebuilt mid-turn) and emitted as a real
  `plan_created` event (`title: "Plan"` / `"Plan (revised)"`) through
  render.py's existing plan-display machinery -- the live status line and
  `/status` now show the actual task-specific plan, not a placeholder.
- Test-suite impact: `should_plan()`-eligible objectives ("fix the bug in
  calc.py", "add a file", etc.) now consume one extra fake completion
  round for the initial plan (and, if that round also had tool_calls, one
  more for the one-time replan) in every affected `_FakeClient`-based
  test -- 3 existing tests updated (`test_single_tool_call_round_then_completion`,
  `test_change_request_completed_with_no_mutation_gets_a_caveat`,
  `test_live_status_line_is_suspended_before_the_approval_panel_prints`).
  Confirmed the blast radius is otherwise contained: `QUESTION`-classified
  objectives (the majority of this suite's fake-client tests) never
  trigger planning at all, and `classify_task()` already treats a
  `read_only`-forced EDIT-shaped objective as QUESTION, not EDIT, so no
  further tests were affected.
- New tests: `tests/test_reasoning_plan.py` (16 tests) -- prompt/parse unit
  tests (well-formed plan, code-fenced JSON, missing/empty steps, malformed
  JSON, non-dict JSON, step cap, evidence-summary/REVISION wording) plus
  full-loop integration tests (task-specific plan replaces the template,
  malformed response falls back silently, a QUESTION-type task never pays
  for a planning call at all, and the plan is revised exactly once after
  real tool evidence exists).
- Tests: 439/439 passing. `python3 -m py_compile`/`python3 -m build` clean.
- Version bumped to `0.4.10` -- installed into the REAL system location
  this time (see the deployment-gap note above), confirmed via
  `tamfis-code --version` on the actual `$PATH` binary, not just a
  throwaway venv.

## Known UX rough edge (not a correctness bug, not yet actioned)

- Three similarly-named but functionally distinct commands: `agent` (full
  orchestrator coding-agent loop), `agents` (list sessions/status), and
  `agent-cmd` (separate lightweight local sub-agents: code_analyzer,
  test_generator, doc_generator -- verified these do real file analysis,
  not canned output). Naming is a footgun for anyone expecting Claude-Code
  parity; not fixed yet.
