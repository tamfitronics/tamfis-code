# Install and release TamfisGPT Code

## Install on any server

Use the isolated installer so the CLI does not depend on the host Python
environment or a server-specific checkout:

```bash
./install.sh
# Root installation: /usr/local/lib/tamfis-code + /usr/local/bin/tamfis-code
# User installation: ~/.local/share/tamfis-code + ~/.local/bin/tamfis-code
# Custom location:
TAMFIS_CODE_PREFIX=/opt/tamfis-code TAMFIS_CODE_BIN_DIR=/usr/local/bin ./install.sh
```

The same package is portable to macOS and Windows. On Windows, create a
virtual environment with `py -m venv`, install the project with `py -m pip
install .`, and expose the virtualenv's `Scripts` directory on PATH. No
machine-specific checkout path is compiled into the package.

When managing several local projects, configure only the approved roots:

```toml
workspace_roots = ["/path/to/project-a", "/path/to/project-b"]
```

Place that in the user's `config.toml`, or use the portable environment
variable `TAMFIS_CODE_WORKSPACE_ROOTS` (path-separated). The launch directory
is always discovered automatically; other roots are allowed only when listed
or explicitly approved with `tamfis-code workspace add <path>`.

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

## Interactive follow-up input

While a task is running, the `message>` editor remains available in the same
terminal. Type normally and press Enter; the line is echoed with a durable
queue ID, then applied at the next safe model-round boundary. Multiple queued
updates are processed in order. A second terminal can enqueue the same session
with:

```bash
tamfis-code queue "also inspect the authentication flow" --classification follow_up
```

`/queue` lists pending requests in the interactive REPL. `Ctrl+C` remains the
interrupt/exit control. No special Ctrl+Y shortcut is required.

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
