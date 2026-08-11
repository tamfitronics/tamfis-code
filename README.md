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

After `tamfis-code login`, the saved TamfisGPT credential unlocks
subscription-backed inference in the standalone runtime. Model routing,
tools, approvals, shell execution, memory, and repository state stay inside
the local `tamfis-code` process. Remote Workspace is a separate legacy mode
that is selected only with `--remote` or `default_backend = "remote"`.
Login also applies a newer version already present in the configured Tamfis
Code source checkout after credentials are saved; set
`TAMFIS_CODE_AUTO_UPDATE=0` for centrally managed installations.

The standalone runtime uses TamfisGPT models. `standalone` is the default
runtime boundary; model origins and routing topology are private deployment
details and are never part of the CLI's public output contract.

## Install

```
pipx install tamfis-code
```

See [USAGE_INSTALL_RELEASE.md](./USAGE_INSTALL_RELEASE.md) for installing
from source, using a TamfisGPT tenancy, and the release process.

## Quick start

```
tamfis-code login                                   # activate TamfisGPT models
tamfis-code doctor                                  # check model-service connectivity
tamfis-code ask "explain what this repo does"
tamfis-code agent "add a health-check endpoint"     # full read/write/execute loop
tamfis-code                                         # interactive REPL
```

### TamfisGPT subscription access

Sign in with the TamfisGPT account whose subscription includes Tamfis-Code:

```bash
tamfis-code login
cd /path/to/your/repository
tamfis-code doctor
tamfis-code agent "inspect this repository and fix the failing tests"
```

The normal interactive and one-shot commands keep the local-agent connection
alive for their lifetime. To make the repository available for work started
from TamfisGPT Web without keeping an interactive chat open, run:

```bash
tamfis-code bridge
```

The bridge reconnects automatically, replays unacknowledged results after a
network interruption, and exposes only the selected workspace. Admin access
is not required for ordinary entitled subscribers.

## Running tasks in the background

Add `--bg` to `ask`/`chat`/`audit`/`agent`/`exec`, `execute-plan`, or `run`
to start work and return
immediately. The task runs in a detached process (its own OS session, not a
child of your terminal), so it keeps running to completion even after you
close the terminal or disconnect:

```bash
tamfis-code agent "add a health-check endpoint" --bg
# Started in background · job bg-a1b2c3d4 (pid 12345)

tamfis-code bg-list                     # every background job and its status
tamfis-code bg-status bg-a1b2c3d4        # one job's detail
tamfis-code bg-logs bg-a1b2c3d4 --follow # tail its output live
tamfis-code bg-stop bg-a1b2c3d4          # terminate it
```

From the interactive prompt, use `/background <objective>` or end a request
with a direct instruction such as “work in the background”. Tamfis Code starts
the same detached job and immediately returns control to the prompt. When the
job finishes, its result is injected once into the originating session; an idle
prompt wakes automatically so the agent can summarize the result or continue
unfinished work.

Use `/goal <objective>` for a persistent background objective. `/goal status`,
`/goal pause`, `/goal resume`, and `/goal cancel` control the latest goal in the
current local session.

Use `/fork` in the interactive REPL to branch the current conversation into a
new independent local session while leaving the original unchanged. The same
operation is available non-interactively with `tamfis-code fork [session_id]`;
resume the new ID printed by the command. Conversation and repository context
are copied, while in-flight tasks and queued instructions are deliberately not.

Use `tamfis-code clear-session <session_id>` to remove a stopped or stale local
session from active listings and prevent it from being reused for that
workspace. A session that still appears live is protected unless `--force` is
given. Recovery checkpoints, evidence, and the last `.memory` snapshot are
retained. Choosing **Start a new session** at the interactive startup picker
shows a separate confirmation gate naming the replaceable old session IDs. If
approved, those old sessions are erased from active listings after the new
session is created; declining the gate resumes the most recent old session.

Job status is one of `running`, `completed`, `failed`, or `stopped`. `--bg`
is not available while creating a plan (the completed plan must first be
saved locally), but a saved plan can be launched with `execute-plan --bg`.

## TamfisGPT models

Users select stable product aliases: **TamfisGPT Auto**, **TamfisGPT Fast**,
**TamfisGPT Code**, **TamfisGPT Pro**, and **TamfisGPT Vision**. Use
`/model list` in the interactive REPL, or pass `--model auto|fast|code|pro|vision`.
TamfisGPT Auto is the default and chooses the appropriate model for the task.

Interrupted model streams remain attached to the same task: clean partial
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

## Fine-grained permissions

User `config.toml` and `<project>/.tamfis/config.toml` may contain persistent
tool rules. Project values extend user values. Rules use `Tool` or
`Tool(pattern)` syntax with shell-style wildcards:

```toml
[permissions]
allow = ["execute_command(pytest *)", "read_file(*)"]
ask = ["execute_command(git push *)"]
deny = ["execute_command(* --force*)", "write_file(*.pem)"]
```

Precedence is `deny`, protected-path approval, `ask`, then `allow`. Writes to
security-sensitive paths such as `.git`, `.env`, `.tamfis`, shell startup
files, and IDE configuration always require explicit approval even in auto
mode. Run `/permissions` to inspect the effective rules.

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

## Skills

Tamfis Code discovers `SKILL.md`, `skill.toml`, `skill.json`, and Kimi-style
flat skill Markdown. It reads Tamfis personal skills plus installed user skills
from `~/.kimi-code/skills/` (or `$KIMI_CODE_HOME/skills/`),
`~/.claude/skills/`, `~/.codex/skills/`, and the shared `~/.agents/skills/`.
Project-local `.kimi-code/skills/`, `.claude/skills/`, `.codex/skills/`,
`.agents/skills/`, and `.tamfis/skills/` are layered on top. Skills whose name,
description, or tags match the objective are injected with their instructions;
project definitions override same-named personal definitions and Tamfis project
definitions have final precedence.

## Complete projects and artifacts

Large multi-file builds use the same long-running plan/checkpoint/repair loop as
ordinary coding tasks. If an objective supplies an explicit project tree,
Tamfis Code verifies that every requested file exists and is non-empty before
allowing a completion claim. ZIP/TAR projects can be safely extracted, edited,
validated, and repackaged.

The `create_artifact` and `inspect_artifact` tools create and verify real binary
DOCX, XLSX, PPTX, and PDF deliverables. XLSX cells that could trigger formula
injection are escaped unless formulas are explicitly enabled. This supports
mixed requests such as building a complete WordPress theme and delivering its
proposal, content inventory, budget workbook, launch deck, PDF manual, and
installable theme archive in one task.

## MCP, plugins, and machine output

External MCP servers stay under the local Tamfis Code runtime. Configure them
in the personal `mcp.json`, project `.mcp.json`, or `.tamfis/mcp.json` using the
standard `mcpServers` shape; discovered tools are namespaced and enter the same
approval policy as native tools. Installed Python packages can contribute tools
and skill roots through the `tamfis_code.plugins` entry-point group. Use
`tamfis-code plugins` to inspect them.

Pass global `--output-mode json` for a single machine-readable event document,
or `--output-mode jsonl` for streaming events suitable for CI and editors.

## IDE, GitHub, and scheduled automation

Run `tamfis-code --cwd <project> acp` as an ACP v1 subprocess from Zed,
JetBrains, or another ACP client. The adapter supports session creation/loading,
streamed prompt results, cancellation, and the same workspace and approval
policy used by the terminal agent.

`tamfis-code --cwd <project> github-automation install-review` creates a
least-privilege GitHub Actions workflow for automatic pull-request reviews.
Add `TAMFIS_API_KEY` as a repository Actions secret before enabling it.

Scheduled local tasks use the durable `automations` command group:

```bash
tamfis-code --cwd /path/to/project automations add nightly "Run tests and fix regressions" --every 1d
tamfis-code automations list
tamfis-code automations serve
```

The foreground service is suitable for systemd, launchd, or a container
supervisor; individual tasks retain an explicit stored approval policy.

## Declarative subagent types

Give a delegated sub-task its own specialised system prompt (and,
optionally, its own TamfisGPT model) instead of every `/delegate`/`/swarm`
sub-objective sharing the same generic coding agent — Claude Code's
`.claude/agents/*.md` equivalent. `~/.config/tamfis-code/agents/<name>.md`
(every session) or `<project>/.tamfis/agents/<name>.md` (one project, and
it replaces a same-named user definition outright) becomes a `/delegate
--agent <name> ...` / `/swarm --agent <name> ...` target:

```markdown
---
description: Reviews code for bugs and security issues
model: TamfisGPT Code
---
You are a strict, terse code reviewer. Point out real bugs, security
issues, and correctness problems only — skip style nits.
```

`model` is optional; omit it to use whatever the
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
`tamfis-code revert`. Shell commands additionally run under an OS-enforced
read-only/workspace-write sandbox (bubblewrap on Linux) with network disabled
by default. A tool may request `require_escalated`, but that request is always
classified dangerous and enters the explicit approval flow.

## License

Copyright © 2026 Tamfitronics. All rights reserved. See [LICENSE](./LICENSE).

## OpenHands-Class Runtime (0.6.1)

Tamfis-Code includes an integrated event-driven agent runtime under
`tamfis_code.openhands`. It provides conversations, immutable event replay,
local/SSH/remote workspaces, terminal/file/browser/Git tools, skills, security,
secrets, snapshots, leases, multi-agent delegation, automations and a REST /
WebSocket agent server. It runs without Docker and uses the same private
TamfisGPT model-routing layer as the terminal client.

Start the agent server:

```bash
tamfis-code-server --host 127.0.0.1 --port 9600
```
