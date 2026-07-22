# tamfis-code

A portable standalone terminal coding agent. It is designed to be installed
on any developer workstation, VPS, or server and run its agent loop, file
tools, shell, PTY/TTY sessions, approvals, plans, and state locally. It must
not depend on the machine that built it or on a co-located TamfisGPT server.

The product subscription boundary is separate from the runtime boundary:
normal users authenticate the CLI with their TamfisGPT iOS subscription/API
entitlement, and the CLI makes outbound API calls from the machine where it
is installed. Admin subscriptions are an account/billing policy, not a
requirement for installing or running the CLI.

The current standalone runtime can also call NVIDIA NIM, OpenRouter, or
Hugging Face directly. Every provider uses the same local-tool contract:
model traffic may be remote, but workspace files, commands, PTYs, approvals,
and mutation evidence remain local unless the user explicitly selects
`--remote`.

`--remote` (or a persistent `default_backend = "remote"` config setting)
switches to the original architecture: a thin client to the TamfisGPT
Remote Workspace backend, for TamfisGPT tenants using their hosted account
the same way Codex CLI uses a ChatGPT/OpenAI account, kimi-code uses a Kimi
account, or Claude Code uses a Claude account.

## Install

```
pipx install tamfis-code
```

See [USAGE_INSTALL_RELEASE.md](./USAGE_INSTALL_RELEASE.md) for installing
from source, using a TamfisGPT tenancy, and the release process.

## Quick start

```
export NVIDIA_API_KEY=...                          # or set OPENROUTER_API_KEY / HF_TOKEN
tamfis-code doctor                                  # check provider connectivity
tamfis-code ask "explain what this repo does"
tamfis-code agent "add a health-check endpoint"     # full read/write/execute loop
tamfis-code                                         # interactive REPL
```

### TamfisGPT subscription access

Create a developer API key from the TamfisGPT account connected to the iOS
subscription, then configure the machine where Tamfis-Code is installed:

```bash
export TAMFIS_API_KEY='tamfis_sk_live_...'
# Optional for a private gateway or regional deployment:
export TAMFIS_API_BASE='https://gpt.tamfitronics.com/api/v1/openai'
tamfis-code doctor
tamfis-code agent "inspect this repository and fix the failing tests"
```

The key is checked against the account's active subscription and API scopes
on every request. The CLI remains portable: the model request goes to the
TamfisGPT API, while repository files, shell commands, PTY sessions,
approvals, and mutation evidence stay on the local machine. Admin access is
not required for ordinary subscription users.

## Providers

| Provider | Env var | Notes |
|---|---|---|
| NVIDIA NIM | `NVIDIA_API_KEY` | |
| OpenRouter | `OPENROUTER_API_KEY` | |
| Hugging Face | `HF_TOKEN` | |
| TamfisGPT subscription API | `TAMFIS_API_KEY` | Key issued for an active TamfisGPT plan |

Select one explicitly with `--provider tamfis|hf|nvidia|openrouter`, or leave it
as `auto` (default). Auto routes through eligible providers in
capability-ranked order. The selected provider/model is printed at the
start of every turn.

Interrupted provider streams remain attached to the same task: clean partial
text is checkpointed to `.memory`, reconnects use visible 5/15/30-second
backoff, and continuation output is de-duplicated. If all configured routes
remain unavailable, the CLI reports an actionable error and retains the exact
checkpoint for `continue` instead of pretending the request completed.

## Local runtime storage

Tamfis Code is not tied to the machine that built or packaged it. Each
installed user gets private runtime storage in that operating system's
standard per-user location:

| Platform | Default |
|---|---|
| Linux/Unix | `$XDG_CONFIG_HOME/tamfis-code`, or `~/.config/tamfis-code` |
| macOS | `~/Library/Application Support/tamfis-code` |
| Windows | `%APPDATA%\tamfis-code` |

Set `TAMFIS_CODE_CONFIG_HOME` to use an explicit portable/container path.
Session memory is stored below that resolved directory in
`.memory/session-<id>.json`. No runtime state is written into the source
checkout or installed package directory, and no VPS/home path is compiled
into the wheel.

## Hooks

Real, user-configurable pre/post-tool-use hooks (no code changes required) —
arbitrary shell commands that observe, or for `pre_tool_use`, veto, a real
local tool call in the standalone agent loop. Configure them in
`hooks.toml`: `~/.config/tamfis-code/hooks.toml` (or the platform-native
location above) for every session, and/or `<project>/.tamfis/hooks.toml`
for one project (both apply — project hooks run after user hooks):

```toml
[[pre_tool_use]]
matcher = "write_file|edit_file"   # regex against the tool name; omit/empty matches every tool
command = "python3 my_guard.py"

[[post_tool_use]]
matcher = "execute_command"
command = "notify-send 'tamfis-code ran a command'"
```

Each hook receives a JSON event on stdin: `{"event", "tool_name",
"tool_input", "tool_output" (post_tool_use only), "session_id",
"workspace_root"}`. For `pre_tool_use`, exiting with code `2` blocks the
call — the tool is never executed, and the hook's stderr (or stdout, if
stderr is empty) becomes the denial reason the model sees. Any other exit
code doesn't block. For `post_tool_use`, the tool has already run, so no
exit code can undo it — stderr/stdout always just surfaces as additional
context appended for the model. Hooks are read fresh at the start of every
turn and time out after 30s; a hook that fails to start, errors, or times
out degrades to a visible diagnostic instead of failing the turn.

## Custom commands

Define your own `/command`, no code changes required — drop a markdown
file into a commands directory and it's available in the interactive REPL
on the next turn. `~/.config/tamfis-code/commands/<name>.md` (or the
platform-native location above) applies to every session;
`<project>/.tamfis/commands/<name>.md` applies to one project (a project
command with the same name replaces the user one). The filename becomes
the command name, an optional frontmatter block sets its description, and
the rest of the file is the prompt template sent as the objective:

```markdown
---
description: Review a diff for security issues
---
Review the following diff for security issues, focusing on injection,
auth, and secrets handling: $ARGUMENTS
```

`$ARGUMENTS` is replaced with whatever you type after the command name
(`/secreview app.py` → the template with `$ARGUMENTS` replaced by
`app.py`); a template with no `$ARGUMENTS` placeholder still gets any
typed text appended on a new line. Run `/commands` in the REPL to list
what's currently loaded. A custom command can never shadow a built-in one
(`/plan`, `/agent`, etc. always win on a name collision).

## Declarative subagent types

Give a delegated sub-task its own specialised system prompt (and,
optionally, its own model/provider) instead of every `/delegate`/`/swarm`
sub-objective sharing the same generic coding agent — Claude Code's
`.claude/agents/*.md` equivalent. `~/.config/tamfis-code/agents/<name>.md`
(every session) or `<project>/.tamfis/agents/<name>.md` (one project, and
it replaces a same-named user definition outright) becomes a `/delegate
--agent <name> ...` / `/swarm --agent <name> ...` target:

```markdown
---
description: Reviews code for bugs and security issues
model: qwen/qwen3-coder
provider: openrouter
---
You are a strict, terse code reviewer. Point out real bugs, security
issues, and correctness problems only — skip style nits.
```

`model`/`provider` are optional; omit either to use whatever the
delegating call was already using. Run `/agent-types` to list what's
loaded. The model itself can also pick a subagent type per sub-task when
it calls `delegate_parallel_tasks` (each task entry may be `{"objective":
"...", "agent_type": "reviewer"}` instead of a plain string) — this is
what actually closes the Claude Code/Kimi gap: the orchestrating model
decides which specialised subagent handles which piece of independent
work, not just a human pre-scripting it via CLI flags.

## Plan mode

`/plan <objective>` (or `--mode plan`) runs entirely read-only and
produces a real, grounded plan — no tool call in this turn can mutate
anything. When it finishes, the REPL asks **"Execute this plan now?
(y/N)"** right there — approve to switch straight into real execution
(the exact plan, not a fresh prompt), or decline to keep it saved for
later (`/plans` to list, `/execute-plan <id>` to run it whenever you're
ready, or `/plan` again to revise the objective first). This is the one
explicit approval checkpoint between "here's the plan" and anything
actually touching the workspace — Claude Code's Plan Mode equivalent.

## Safety model

Every mutating tool call (`write_file`, `edit_file`, `execute_command`) is
risk-classified locally (`tamfis_code/safety.py`), gated by
`--approval-policy` (or the `/mode` REPL command), and recorded in a local
mutation ledger — see `tamfis-code diffs` / `tamfis-code diff` /
`tamfis-code revert`. There is no sandboxing beyond risk classification;
review dangerous commands yourself before approving them.

## License

TBD — decide before the first public PyPI release (a public package still
needs a stated license; "all rights reserved"/proprietary is a valid choice
if you don't want redistribution/modification, but it should be explicit
rather than absent).

## OpenHands-Class Runtime (0.6.1)

Tamfis-Code includes an integrated event-driven agent runtime under
`tamfis_code.openhands`. It provides conversations, immutable event replay,
local/SSH/remote workspaces, terminal/file/browser/Git tools, skills, security,
secrets, snapshots, leases, multi-agent delegation, automations and a REST /
WebSocket agent server. It runs without Docker and preserves the standalone
provider order NVIDIA → OpenRouter → Hugging Face. Ollama is intentionally not
included.

Start the agent server:

```bash
tamfis-code-server --host 127.0.0.1 --port 9600
```
