# tamfis-code: Tenancy Usage, Installation, and Release Guide

This covers three things: using a paid TamfisGPT account with `tamfis-code`
(the `--remote` path), installing `tamfis-code` on a machine that didn't
build it, and how you (the maintainer) ship an upgrade so installed copies
can update.

Context: `tamfis-code` defaults to **standalone mode** — it calls an LLM
provider directly (HF / NVIDIA NIM / OpenRouter / Ollama) using your own
provider API keys, with no TamfisGPT backend involved. `--remote` (or the
`default_backend` setting below) switches it to the original architecture:
a thin client to the TamfisGPT Remote Workspace backend, the same way
Codex CLI uses your ChatGPT/OpenAI account, kimi-code uses your Kimi
account, and Claude Code uses your Claude account/API access.

---

## 1. Using your TamfisGPT tenancy (paid access) with tamfis-code

### One-time: authenticate

```
tamfis-code login
```

Prompts for email/password (or pass `--token <token>` / set
`TAMFIS_CODE_LOGIN_TOKEN` to use an existing access token instead — prefer
the env var over typing a token on the command line, since command lines
land in shell history). This writes `~/.config/tamfis-code/credentials.json`
(mode 0600, owner-only).

If your TamfisGPT account is hosted somewhere other than this machine's
default (`http://127.0.0.1:9500`), point at your account's real API base
first — get this URL from TamfisGPT, don't guess it:

```
tamfis-code --api-base https://<your-tamfisgpt-api-host> login
```

or persist it so you don't need the flag every time (see below).

### Make `--remote` the default, so you don't type it on every command

By default, every command (`ask`, `chat`, `agent`, `exec`, `audit`, `plan`,
`execute-plan`, `init`, `doctor`, `status`, `sessions`, `diffs`, `run`,
`resume`, `retry`, and the bare interactive REPL) uses standalone mode
unless you pass `--remote`. As a paying tenant who wants TamfisGPT's hosted
backend to just be the default — the same way Claude Code defaults to your
Claude account without a flag — set this once in
`~/.config/tamfis-code/config.toml`:

```toml
api_base = "https://<your-tamfisgpt-api-host>"
default_backend = "remote"
```

(Or per-project only: the same two lines in `.tamfis/config.toml` inside
that project's directory, which overrides the user-level config.) An
environment variable also works, e.g. for a CI job or a container image:

```
export TAMFIS_CODE_DEFAULT_BACKEND=remote
export TAMFIS_CODE_API_BASE=https://<your-tamfisgpt-api-host>
```

Precedence (highest wins): `--remote` flag on a specific command > env var >
project `.tamfis/config.toml` > user `~/.config/tamfis-code/config.toml` >
built-in default (`standalone`).

### Using it day to day

Once logged in and `default_backend = "remote"` is set, every command works
exactly like the standalone ones, just backed by TamfisGPT's hosted
backend instead of your own provider key:

```
tamfis-code ask "add input validation to the signup form"
tamfis-code agent "fix the failing test in test_checkout.py"
tamfis-code            # bare invocation -- interactive REPL, remote-backed
tamfis-code status
tamfis-code sessions
```

`tamfis-code logout` clears the stored credentials. `tamfis-code doctor`
(with `default_backend = "remote"` set, or `--remote` passed) checks
connectivity/auth against the backend; without it, `doctor` checks your
local provider keys instead — useful if you're not sure which mode you're
in.

---

## 2. Installing on a machine that didn't build it

`tamfis-code` isn't published to PyPI yet — see the note at the end of this
section if you want that. Until then, install it from source using
[`pipx`](https://pipx.pypa.io) (isolates it in its own venv, exposes the
`tamfis-code`/`tamgpt-code`/`tamfis` commands globally, and — importantly —
**never install it as root**; see "why not root" below).

### Prerequisites

- Python 3.10+
- `pipx` (`sudo apt install pipx` on Debian/Ubuntu, or `pip install --user pipx`, then `pipx ensurepath`)
- `ripgrep` (`rg`) if you want `search_code` to work: `sudo apt install ripgrep`

### Install

```
pipx install "git+https://github.com/tamfitronics/tamfis-code.git"
```

...or from a local copy of the source instead:

```
pipx install /path/to/tamfis-code-source
```

This installs into `~/.local/share/pipx/venvs/tamfis-code` (owned by
whichever user ran the install) with shims at `~/.local/bin/{tamfis-code,tamgpt-code,tamfis}`.

### Making it available system-wide (in `/usr/local/bin`, like other CLI tools)

Run the install **as the user who will actually use/maintain it**, never as
root — then have root create a one-time symlink:

```
# as the real user, e.g. tamfisgpt:
pipx install /path/to/tamfis-code-source

# as root, once:
ln -s /home/<user>/.local/bin/tamfis-code /usr/local/bin/tamfis-code
ln -s /home/<user>/.local/bin/tamgpt-code /usr/local/bin/tamgpt-code
ln -s /home/<user>/.local/bin/tamfis /usr/local/bin/tamfis
```

**Why not root:** if you `pipx install`/`pip install -e` as root, the
installed package files end up root-owned. The first time anyone needs to
update or fix that code as the actual operating user, every write fails
with `Permission denied` — this exact bug is what kicked off this whole
rebuild. The split above keeps `/usr/local/bin`'s entries root-owned (which
is normal — that's true of everything else in `/usr/local/bin` too) while
the actual installed package stays owned by whoever needs to maintain it.

### Verify

```
tamfis-code --version
tamfis-code doctor        # checks local provider connectivity (HF/NVIDIA/OpenRouter/Ollama)
```

### Path to a real one-line install (`pipx install tamfis-code`, no URL)

The repo (`https://github.com/tamfitronics/tamfis-code`) and the
`.github/workflows/publish.yml` CI workflow are both set up (see section 3)
— the only remaining step is **one-time, manual, on pypi.org, and needs
your PyPI account** (nothing I can do from here):

1. Create a PyPI account if you don't have one, at pypi.org.
2. Go to `https://pypi.org/manage/account/publishing/` and register a
   "pending publisher" for a project named `tamfis-code` (doesn't need to
   exist yet): owner `tamfitronics`, repository `tamfis-code`, workflow
   `publish.yml`, environment `pypi`.
3. In the GitHub repo's settings, create an environment named `pypi`
   (Settings → Environments → New environment) — no secrets needed, PyPI
   Trusted Publishing authenticates via OIDC, not a stored API token.
4. The next `git push origin vX.Y.Z` (section 3 below) triggers the
   workflow and publishes for real. From then on, `pipx install
   tamfis-code` works with no source path or git URL, exactly like the
   competitors.

---

## 3. Shipping an upgrade so installed copies can update

Repo: **https://github.com/tamfitronics/tamfis-code** (public, created
under the `tamfitronics` org). `.github/workflows/publish.yml` runs the
test suite and publishes to PyPI (once section 2's one-time PyPI setup is
done) on every `v*` tag push.

### Step 1 — bump the version in both places (they must match)

- `pyproject.toml` → `version = "X.Y.Z"`
- `tamfis_code/__init__.py` → `__version__ = "X.Y.Z"` (this is what
  `tamfis-code --version` actually prints; `pyproject.toml`'s version is
  what packaging tools see — keeping them in sync avoids `--version` lying
  to a user about what they have installed)

Use normal semver judgment: patch for fixes, minor for new capability
(e.g. this session's whole standalone rebuild would justify a minor bump,
which is why it went `0.1.1` → `0.2.0`), major for a breaking change to the
CLI surface or config format.

### Step 2 — commit, tag, and push

```
git add pyproject.toml tamfis_code/__init__.py
git commit -m "Bump version to X.Y.Z"
git tag vX.Y.Z
git push origin main --tags
```

Pushing the tag is what triggers `publish.yml` — the test job runs first,
and only publishes to PyPI if it passes.

### Step 3 — how installed copies actually pick up the upgrade

This depends entirely on how each copy was installed:

- **Editable install pointing at this exact checkout**
  (`pipx install -e /path/to/tamfis-code-source`, what this dev machine
  uses): the new code is live immediately — there is nothing to "install",
  the running command already reads from the same files you just edited.
  Only relevant if the install and the development happen on the same
  machine.
- **Git-based install** (`pipx install "git+https://github.com/tamfitronics/tamfis-code.git"`):
  the user runs `pipx upgrade tamfis-code` (pulls the latest commit on the
  default branch) or reinstalls pinned to a tag: `pipx install --force
  "git+https://github.com/tamfitronics/tamfis-code.git@vX.Y.Z"`.
- **PyPI install** (once section 2's one-time setup is done):
  `pipx upgrade tamfis-code` (or `pip install --upgrade tamfis-code`)
  picks up whatever the workflow most recently published.

Whichever channel a given user is on, `tamfis-code --version` after
upgrading is the one honest way to confirm they actually got the new
build — that's why step 1 keeping both version strings in sync matters.
