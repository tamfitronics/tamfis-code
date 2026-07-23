# Tamfis-Code 0.6.13

## Release contract

The project version is pinned to **0.6.13**. Further work must remain on this
version until the project owner explicitly authorises another version.

## Improvements

- Added the requested GitHub CLI-compatible command surface: `alias`, `api`,
  `auth`, `browse`, `cache`, `co`, `codespace`, `completion`, `config`,
  `extension`, `gist`, `gpg-key`, `issue`, `label`, `org`, `pr`, `project`,
  `release`, `repo`, `ruleset`, `run`, `search`, `secret`, `ssh-key`, `status`,
  `variable`, and `workflow`.
- Commands transparently delegate to the installed `gh` executable and retain
  native terminal I/O, current directory, authentication, and exit status.
- Reworked the persistent PTY/TTY footer into a compact Claude-style status
  line showing phase, current action, selected model, elapsed time, token use,
  approval mode, interrupt help, and mode-switch help.
- Expanded economical routing options with OpenRouter's capability-filtering
  free router, a free Qwen coding route, economical Claude/OpenAI options,
  and an additional NVIDIA Nemotron coding/reasoning route.
- Preserved capability-aware routing: tool requirements, long-context needs,
  task complexity, quality mode, provider availability, and retryable credit,
  quota, entitlement, transport, and capacity errors determine route changes.
- Removed stale wheels, build products, caches, logs, and backup files from the
  release package.
