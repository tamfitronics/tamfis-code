# Tamfis-Code 0.4.0 Full Orchestration Upgrade

## Diagnosis

The uploaded 0.3.0 archive had capability-aware provider scoring and several safety fixes, but the primary local runner still owned a monolithic model/tool loop. Planning, phases, tool evidence, validation and completion integrity were not managed by a dedicated persistent orchestrator. Tool selection was also broader than necessary, Tier IV had no canonical client contract, and command execution could still inherit the caller process directory when `cwd` was omitted.

## Implemented

- Added `tamfis_code/orchestrator/` with a persistent state machine covering understand, inspect, route, plan, execute, observe, repair, validate, report, approval, completed and failed phases.
- Added deterministic executable plans for complex turns and persisted them through the existing session state store.
- Added layered, recoverable context assembly using repository fingerprints, prior turns, retrieved files, tool results, active plan and validation state.
- Added canonical provider/tool events and structured `ToolEnvelope` evidence records.
- Added evidence-based completion validation and unresolved-issue persistence.
- Added a canonical model capability registry.
- Added task-aware minimum tool gating.
- Added a Tier IV routing request/decision contract with configured-Tier-IV-first and safe local fallback.
- Added provider stream normalisation for OpenAI-compatible chunks, Ollama native events, Anthropic-style deltas and canonical Tier IV events.
- Resolved provider/model once per turn, preventing route drift between tool rounds.
- Added real renderer support for orchestrator phases.
- Expanded workspace context with manifests, languages, package managers, frameworks, important directories, test/build commands and service definitions.
- Strengthened repository fingerprinting with manifest and instruction-file metadata.
- Extended `execute_command` with real workspace-default `cwd`, timeout, environment overrides, shell selection and approval metadata.
- Added local-manifest guards for npm/pnpm/yarn, pip, Cargo, Go, Maven and Gradle commands. Parent manifests are deliberately ignored.
- Preserved fabricated-tool-call detection, empty-stream retry, missing-file truthfulness, cwd boundary enforcement, mutation ledger and approval controls.
- Fixed packaging so the new `tamfis_code.orchestrator` subpackage is included in the wheel.

## Provider policy

Automatic coding priority is Tier IV when explicitly configured, then the strongest eligible local route based on capabilities. The local fallback chain is NVIDIA, Hugging Face, OpenRouter, then Ollama. Explicit provider/model pins continue to override automatic routing. Ollama remains suitable for explicit local-only/offline use and lightweight conversation, but models without reliable tool calling do not receive coding tools.

## Validation

- Full tests: 370 passed, 2 collection warnings.
- Python compilation: passed.
- Wheel build: passed.
- Wheel content smoke test: passed; orchestrator and Tier IV modules are included.
- Isolated target installation: passed.
- CLI `--help` smoke test: passed.
- Greeting policy smoke test: `hello` receives zero tools.
- Command cwd/manifest/environment regression tests: passed.
- Provider protocol normalisation tests: passed.

The original archive contained 355 tests. The earlier stated 374-test baseline was not present in this supplied source tree. This release adds 15 tests and finishes with 370 passing tests.

## Configuration

- `TAMFIS_TIER_IV_URL`: enables and points to the shared Tier IV router, e.g. `http://127.0.0.1:9555`.
- `TAMFIS_TIER_IV_ENABLED=true|false`: explicit Tier IV routing toggle.
- `TAMFIS_ACCESS_TOKEN`: optional bearer token for Tier IV.
- Existing provider keys remain: `NVIDIA_API_KEY`, `HF_TOKEN`, `OPENROUTER_API_KEY`.
- `OLLAMA_BASE_URL`: optional Ollama OpenAI-compatible base URL.

## Remaining external dependency

The uploaded package did not include `/home/tamfisgpt/tamgpt6`, so the Tier IV server endpoint itself could not be modified or live-tested here. Tamfis-Code now implements the client contract and falls back safely when the endpoint is unavailable. The connected Tier IV service must expose `POST /api/v1/orchestration/route` using the documented request/decision fields for shared routing to become authoritative.

---

# Tamfis-Code 0.4.17 Claude Code / Codex CLI Parity Audit

This is a separate, later pass than the 0.4.0 upgrade above -- see STATUS.md
(the actively-maintained single source of truth) for the full, current
write-up of this session's fixes and remaining gaps; this section is a
short pointer for anyone reading UPGRADE_REPORT.md specifically.

## Diagnosis

An external parity spec asked for Claude Code/Codex CLI-level capability
across 11 checklist items, written as if against a much earlier codebase.
A 5-way parallel read-only audit against the real source (not the spec's
assumptions) found most items already solid and real. The genuine gaps
clustered around one pattern: working code that existed but was never
wired into the default **local** (non-`--remote`) execution path, which is
what most users actually run -- the `--remote` backend kept getting the
complete version of features (plan step status updates, doctor
diagnostics) while local mode silently got a stub or nothing.

## Implemented

- `workspace.py`: added `TAMFIS.md`/`.tamfis` to `INSTRUCTION_NAMES` so the
  project's own documented instruction-file convention is actually picked
  up by the live system prompt (it previously wasn't, despite a separate,
  dead, never-called implementation elsewhere already documenting that
  convention).
- `orchestrator/engine.py`: plan step status now advances in the local
  path as real tool results are observed (`_advance_plan_step`, called
  from `record_tool()`), persists correctly (`OrchestrationRun.plan_id`,
  `replace_plan()` so a reasoning-plan swap doesn't leave `state.saved_plans`
  stuck on the generic template), and resolves on completion/failure. A
  new `plan_step_progress` event (distinct from `plan_created`) carries
  these updates to `render.py` without re-triggering the "new plan" banner.
- `doctor.py` + `cli.py`: added local-mode diagnostics (`_diagnose_local_providers`,
  `_diagnose_local_session`) covering provider health, estimated context
  token usage (a new `state.py` field, `estimated_context_tokens`),
  tool-call success rate, and plan-step progress -- wired into the actual
  default/local `doctor` CLI branch, not just the `--remote`-only
  `run_doctor()` function (a first pass got this wrong; a real `tamfis-code
  doctor` smoke test caught it before shipping). Also corrected a
  local-mode false `FAIL` on missing `--remote` credentials to `WARNING`.
- `state.py`: added `estimated_context_tokens` field.
- Incidental: `tamfis_code/__init__.py`'s `__version__` and
  `pyproject.toml`'s `version` are two separate hardcoded copies that can
  drift -- both bumped to `0.4.17` this pass; not consolidated to a single
  source (would be a restructuring change, left for a future pass).

## Validation

- Full tests: 478 passed (up from a 467 baseline), 0 failed.
  `python3 -m pytest -q`.
- `python3 -m py_compile` clean.
- Wheel rebuilt (`python3 -m build --wheel`) and reinstalled into
  `/usr/local/lib/python3.13/dist-packages/tamfis_code/`
  (`--force-reinstall --no-deps --break-system-packages`); `diff -rq`
  against the source tree is clean; `tamfis-code --version` reports
  `0.4.17`.
- `tamfis-code doctor` smoke-tested for real (not just under pytest) from
  both the source tree and the reinstalled system binary, confirming the
  new local-session diagnostic lines print.

## Explicitly not attempted this pass

Repair-phase real retry loop and plan-state `/resume`; indexer incremental
re-indexing and a reference-resolution tool the model can call mid-turn;
diff preview before write/edit approval and transactional multi-file
revert; MCP server-exposure (tamfis-code driven by other agents/IDEs); REPL
tab-completion. Full detail on each in STATUS.md.
