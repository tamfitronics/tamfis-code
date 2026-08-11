# Coding-agent feature parity benchmark

Snapshot date: 2026-08-10.

This is a feature-presence benchmark, not a model-quality benchmark. A `yes`
means the feature is implemented in TamfisGPT Code or documented by the named
vendor. `partial` means the workflow exists but lacks an important native
surface or automation. The comparison uses the union of documented Kimi Code,
Claude Code, and Codex capabilities, so it is deliberately harder than matching
any single product.

Run both benchmark passes with:

```bash
python3 benchmarks/feature_parity.py --verify
```

Pass 1 scans the current source tree for implementation evidence. Pass 2 runs
focused tests plus fifteen named behavioral regression scenarios. Those scenarios
exercise persistent permission precedence and protected paths, exactly-once
background result reinjection, natural-language background and goal controls,
read-only enforcement, proportional planning, failure-triggered replanning,
semantic plan progress, paginated reads, usable read-only inspection pipelines,
session-fork isolation, detached saved-plan and shell execution, editable
queued follow-ups,
duplicate-evidence loop termination, and route-banner deduplication. The
executable matrix is the source of truth for
feature-presence scores; the behavioral pass is reported separately.

Current result: **TAMFIS-CODE** scores **25/25 (100%)** against the combined
feature union. The former material gaps now have executable surfaces:

1. **IDE integration:** `tamfis-code acp` exposes ACP v1 over stdio with
   initialize, new/load session, prompt streaming, and cancellation.
2. **GitHub workflow automation:** `tamfis-code github-automation
   install-review` installs a least-privilege automatic PR-review workflow.
3. **Scheduled automations:** `tamfis-code automations` provides add/list/run,
   enable/disable/remove, and a foreground scheduler service.
4. **Session branching:** `/fork` and `tamfis-code fork [session_id]` clone
   durable conversation/repository context into an independent local session,
   while clearing in-flight task state and preserving the original unchanged.

This score remains a feature-presence result. It does not mean every vendor's
UI or proprietary hosted service has been cloned.

The behavioral result is intentionally not converted into a competitor score:
vendor documentation can establish that a surface exists, but not that another
implementation passes the same local scenarios.

The competitor entries are grounded in vendor documentation:

- Kimi Code documents persistent sessions, MCP, skills, custom agents,
  background agents, AgentSwarm, session forks, and ACP in its [agent documentation](https://www.kimi.com/code/docs/en/kimi-code-cli/customization/agents.html), [session guide](https://www.kimi.com/code/docs/en/kimi-code-cli/guides/sessions.html), [tool reference](https://www.kimi.com/code/docs/en/kimi-code-cli/reference/tools.html), [skill reference](https://www.kimi.com/code/docs/en/kimi-code-cli/customization/skills.html), and [ACP reference](https://www.kimi.com/code/docs/en/kimi-code-cli/reference/kimi-acp).
- Claude Code documents interactive and print modes, JSON streaming, model
  aliases, permission modes, resume/continue, session branching, and MCP in its
  [CLI reference](https://code.claude.com/docs/en/cli-usage), [session guide](https://code.claude.com/docs/en/sessions), and [MCP documentation](https://docs.anthropic.com/en/docs/mcp).
- Codex documents local editing, image input, to-do tracking, web search, MCP,
  approval modes, compaction, session forks, IDE/cloud handoff, browser
  verification, and code review in its [slash-command reference](https://learn.chatgpt.com/docs/reference/slash-commands)
  and [Introducing upgrades to Codex](https://openai.com/index/introducing-upgrades-to-codex/);
  parallel agents, skills, and automations are documented in the [Codex app announcement](https://openai.com/index/introducing-the-codex-app/).

Scores measure discoverable capability only. They do not claim equivalent UX,
reliability, latency, reasoning quality, or safety strength.
