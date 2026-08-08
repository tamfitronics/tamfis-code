import os

import pytest


# providers.py unconditionally loads the real /home/tamfiscode/.env into
# os.environ at import time (that's intentional for production runs -- see
# providers.py's _load_project_env). Left alone, that means routing/model
# selection tests silently pick up whatever this machine's operator has
# configured there (e.g. TAMFIS_CODE_OLLAMA_CODING_MODEL, premium/extra-usage
# opt-ins, provider fallback toggles), so results depend on who's running the
# suite and what their live .env says instead of being deterministic.
_TAMFIS_ENV_PREFIXES = ("TAMFIS_CODE_", "TAMFIS_PROVIDER_")


@pytest.fixture(autouse=True)
def _isolate_tamfis_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith(_TAMFIS_ENV_PREFIXES):
            monkeypatch.delenv(key, raising=False)
