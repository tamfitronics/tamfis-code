# Tamfis-Code installation and release

## Install for users

Use an isolated Python 3.10+ environment:

```bash
pipx install tamfis-code
tamfis-code login
cd /path/to/repository
tamfis-code doctor
tamfis-code agent "inspect the repository and fix the failing tests"
```

Login supplies TamfisGPT subscription entitlement to the local runtime. It
does not enable Remote Workspace. Users may alternatively configure
`NVIDIA_API_KEY`, `HF_TOKEN`, `OPENROUTER_API_KEY`, or a signed-in local
Ollama daemon.

## Install from a checkout

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m tamfis_code --version
.venv/bin/python -m pytest -q
```

Always invoke the module through the same interpreter used for installation.
This prevents an older wheel in `site-packages` from shadowing the checkout.

## Browser support

The Python Playwright client is installed with Tamfis-Code. It uses an
available system Chromium/Chrome executable first. On a machine without one:

```bash
playwright install chromium
```

No TamfisGPT monorepo checkout is required.

## Release gate

1. Update `pyproject.toml` and `tamfis_code/__init__.py` to the same version.
2. Run `python -m pytest -q`.
3. Run `python -m tamfis_code verify-release`.
4. Build with `python -m build`.
5. Install the wheel into a fresh temporary virtual environment.
6. Run `tamfis-code --version`, `tamfis-code --help`, `tamfis-code config`,
   and `tamfis-code doctor` from outside the source checkout.
7. Publish only the exact wheel and source archive that passed those checks.

Do not release from a development environment whose installed package
version differs from the checkout version.
