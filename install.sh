#!/usr/bin/env bash
set -euo pipefail

# Portable installer for Linux/macOS. Windows users can run the equivalent
# commands from PowerShell (see USAGE_INSTALL_RELEASE.md). The install prefix
# is deliberately configurable so this never assumes /root or a server-only
# path.
SOURCE_DIR=$(cd "$(dirname "$0")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
PREFIX=${TAMFIS_CODE_PREFIX:-}
if [[ -z "$PREFIX" ]]; then
  if [[ "$(id -u)" -eq 0 ]]; then PREFIX=/usr/local/lib/tamfis-code; else PREFIX="${HOME}/.local/share/tamfis-code"; fi
fi
BIN_DIR=${TAMFIS_CODE_BIN_DIR:-}
if [[ -z "$BIN_DIR" ]]; then
  if [[ "$(id -u)" -eq 0 ]]; then BIN_DIR=/usr/local/bin; else BIN_DIR="${HOME}/.local/bin"; fi
fi

"$PYTHON_BIN" -m py_compile "$SOURCE_DIR"/tamfis_code/*.py
mkdir -p "$PREFIX" "$BIN_DIR"
if [[ "$(id -u)" -eq 0 && -z "${TAMFIS_CODE_FORCE_VENV:-}" ]]; then
  # Server/system install: use the host's normal site-packages and expose
  # the console scripts in /usr/local/bin. This is a CLI, not a boot daemon.
  "$PYTHON_BIN" -m pip install --break-system-packages --force-reinstall "$SOURCE_DIR"
  RUNTIME_PYTHON="$PYTHON_BIN"
else
  # Non-root installs stay isolated and require no administrator access.
  "$PYTHON_BIN" -m venv "$PREFIX/venv"
  "$PREFIX/venv/bin/python" -m pip install --upgrade pip
  "$PREFIX/venv/bin/python" -m pip install --force-reinstall "$SOURCE_DIR"
  ln -sfn "$PREFIX/venv/bin/tamfis-code" "$BIN_DIR/tamfis-code"
  ln -sfn "$PREFIX/venv/bin/tamgpt-code" "$BIN_DIR/tamgpt-code"
  ln -sfn "$PREFIX/venv/bin/tamfis" "$BIN_DIR/tamfis"
  RUNTIME_PYTHON="$PREFIX/venv/bin/python"
fi

"$RUNTIME_PYTHON" - <<'PY2'
from importlib.metadata import version
from tamfis_code.providers import ProviderManager, ProviderType
from tamfis_code.render import StreamRenderer
from tamfis_code.openhands.events import EventStore
from tamfis_code.openhands.agent_server import app
m=ProviderManager()
assert ProviderType.TIER_IV not in m.routing_order
assert [p.value for p in m.routing_order] == ['nvidia','openrouter','hf']
assert hasattr(StreamRenderer,'_flush_assistant')
print('Version:',version('tamfis-code'))
print('Routing:',[p.value for p in m.routing_order])
print('OpenHands runtime: enabled')
print('Agent server:',app.title)
PY2
