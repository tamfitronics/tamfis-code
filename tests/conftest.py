import dataclasses
import os

import pytest

from tamfis_code import runner_local as _runner_local_module
from tamfis_code.config import Config as _RealConfig


@dataclasses.dataclass
class _TestConfig(_RealConfig):
    # Only overrides the field default; every other Config() behavior
    # (env-derived fields aside) is untouched.
    sandbox_fail_if_unavailable: bool = False


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
    # sandbox_fail_if_unavailable defaults to True on Linux as of
    # 2026-08-21 (see sandbox.py) -- correct for real usage, but several
    # tests that aren't testing sandboxing itself (test_reasoning_plan.py,
    # test_concurrent_dispatch_regressions.py, etc.) route real
    # execute_command calls through _run_local_agent_turn_impl's default
    # Config, which would now require bubblewrap to actually function, not
    # just be installed. Live-confirmed this fails in GitHub Actions'
    # runner environment specifically (bwrap's own loopback-interface setup
    # gets "Operation not permitted" there even with the binary present and
    # user namespaces nominally unrestricted -- an AppArmor/container layer
    # neither this project nor these tests should need to fight).
    #
    # An env var can't fix this: runner_local.py builds the default config
    # as `cli_config or Config()` (line ~4182), a bare dataclass
    # constructor -- only load_config() reads TAMFIS_CODE_* env vars, and
    # this call site never goes through it. So the only reliable knob is
    # the dataclass field default itself, which is baked into Config's
    # generated __init__ at class-definition time and can't be changed by
    # monkeypatching attributes after the fact. Instead we monkeypatch the
    # *name* `Config` inside runner_local's module namespace to a subclass
    # with the field default flipped -- Python resolves `Config()` against
    # that name at call time, so this reliably reaches the one call site
    # above without touching production behavior (tamfis_code.config.Config
    # itself, and every other import of it, is untouched).
    #
    # test_sandbox.py's own fail-closed tests are unaffected: they
    # construct SandboxPolicy directly with explicit kwargs, never going
    # through Config at all.
    monkeypatch.setattr(_runner_local_module, "Config", _TestConfig)
