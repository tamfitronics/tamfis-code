# tamfis-code

A standalone terminal coding agent. By default it calls an LLM provider
directly — Hugging Face, NVIDIA NIM, OpenRouter, or Ollama — and runs its
own agent loop, tool execution, and local risk/approval/mutation-ledger
safety layer, with no separate backend process required.

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
export OLLAMA_BASE_URL=http://localhost:11434/v1   # or set HF_TOKEN / NVIDIA_API_KEY / OPENROUTER_API_KEY
tamfis-code doctor                                  # check provider connectivity
tamfis-code ask "explain what this repo does"
tamfis-code agent "add a health-check endpoint"     # full read/write/execute loop
tamfis-code                                         # interactive REPL
```

## Providers

| Provider | Env var | Notes |
|---|---|---|
| Ollama | `OLLAMA_BASE_URL` (default `http://localhost:11434/v1`) | Runs fully on-device, no API key |
| Hugging Face | `HF_TOKEN` | |
| NVIDIA NIM | `NVIDIA_API_KEY` | |
| OpenRouter | `OPENROUTER_API_KEY` | |

Select one explicitly with `--provider hf|nvidia|openrouter|ollama`, or
leave it as `auto` (default) to pick the best configured one.

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
