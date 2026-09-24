# Tamfis-Code 1.8.0

## Production hardening

- Separate task kind from execution mode in delivery policy. Plain questions
  no longer fail completion because they did not produce repository evidence.
- Explicit no-edit requests no longer advertise the general shell tool.
- Stale inspect/audit turns can recognize explicit `sed -i` and `perl -pi`
  source edits and transition through the normal execute, scope, risk, and
  approval gates instead of repeatedly rejecting the call.
- Repeated `read_file` recovery now requires a materially different search or
  canonical path and cannot be misreported as a read-only environment when
  the actual cause is a duplicate-action guard or provider stall.
- Added regression coverage for the combined routing, safety, completion, and
  recovery behavior.

## Research basis

The hardening follows the provider-neutral tool contract used by modern coding
agents: tool schemas must reflect the current permission mode, tool calls must
remain structured and evidence-backed, and explicit permission escalation must
be separated from tool execution. See the OpenAI function-tool contract and
Anthropic Claude Code permission-mode documentation in the project audit notes.

## Verification

- Consolidated regression suite: 93 passed, 1 skipped.
- Prior full suite baseline: 2,660 passed, 2 skipped, 110 subtests passed;
  the five previously failing policy cases now pass in the focused rerun.
