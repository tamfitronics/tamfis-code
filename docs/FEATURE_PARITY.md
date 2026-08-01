# Coding-agent feature parity benchmark

Snapshot date: 2026-08-01.

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
focused behavioral tests covering custom agents, hooks, MCP, swarms, safety,
and public rendering. The executable matrix is the source of truth for scores.

Current result: TamfisGPT Code scores **22.5/24 (93.8%)** against the combined
feature union. The material gaps are:

1. **IDE-native integration (partial):** TamfisGPT Code exposes an MCP server
   and a REST/WebSocket agent server, but does not ship first-party IDE clients
   or an ACP endpoint comparable to competitors' native integrations.
2. **GitHub workflow automation (partial):** it has a broad `gh`-compatible
   command surface, but not a first-party automatic PR-review bot.
3. **Scheduled automations (partial):** automation primitives exist in the
   integrated runtime, but there is no polished top-level scheduling UX.

The competitor entries are grounded in vendor documentation:

- Kimi Code documents persistent sessions, MCP, skills, custom agents,
  background agents, and AgentSwarm in its [agent documentation](https://www.kimi.com/code/docs/en/kimi-code-cli/customization/agents.html), [tool reference](https://www.kimi.com/code/docs/en/kimi-code-cli/reference/tools.html), and [CLI reference](https://www.kimi.com/code/docs/en/kimi-code-cli/reference/kimi-command.html).
- Claude Code documents interactive and print modes, JSON streaming, model
  aliases, permission modes, resume/continue, and MCP in its [CLI reference](https://docs.anthropic.com/en/docs/claude-code/cli-usage) and [MCP documentation](https://docs.anthropic.com/en/docs/mcp).
- Codex documents local editing, image input, to-do tracking, web search, MCP,
  approval modes, compaction, IDE/cloud handoff, browser verification, and code
  review in [Introducing upgrades to Codex](https://openai.com/index/introducing-upgrades-to-codex/); parallel agents, skills, and automations are documented in the [Codex app announcement](https://openai.com/index/introducing-the-codex-app/).

Scores measure discoverable capability only. They do not claim equivalent UX,
reliability, latency, reasoning quality, or safety strength.
