# Install and release TamfisGPT Code

## Install on any server

Use an isolated installer so the CLI does not depend on the host Python
environment:

```bash
python3 -m pip install --user tamfis-code
# or:
pipx install tamfis-code
```

The runtime stores session state in the platform's user configuration
directory. It does not require a TamfisGPT server process on the same host.

## Configure the TamfisGPT subscription API

Create an API key from the TamfisGPT account connected to the user's active
iOS subscription, then set:

```bash
export TAMFIS_API_KEY='tamfis_sk_live_...'
export TAMFIS_API_BASE='https://gpt.tamfitronics.com/api/v1/openai'
tamfis-code doctor
```

The API key is the user's entitlement boundary. Admin credentials are not
needed for ordinary subscribers. The API handles model access and billing;
the installed CLI keeps workspace files, commands, PTYs, approvals, and the
mutation ledger on the local machine.

## Interactive queue controls

While a task is running, press `Ctrl+Y` to open `queue next>`. The submitted
line is echoed with a durable queue ID, then applied at the next safe model
round boundary. Multiple queued updates are processed in order. A second
terminal can enqueue the same session with:

```bash
tamfis-code queue "also inspect the authentication flow" --classification follow_up
```

`/queue` lists pending requests in the interactive REPL. `Ctrl+C` remains the
interrupt/exit control; ordinary keystrokes are never silently turned into a
partial instruction while streaming.

## Build and publish a release

Run the full checks, build a clean distribution, and inspect it before
publishing:

```bash
python3 -m pytest -q
python3 -m build
python3 -m twine check dist/*
```

Publishing requires an account token configured in the release environment:

```bash
python3 -m twine upload dist/*
```

The GitHub repository should receive the same source revision and release
notes as the PyPI upload. Never commit API keys, subscription tokens, local
session state, or generated build artifacts.
