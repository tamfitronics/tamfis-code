import dataclasses
import os
import shutil
import subprocess
import sys

import pytest

from tamfis_code import runner_local as _runner_local_module
from tamfis_code import config as _config_module
from tamfis_code.config import Config as _RealConfig


def _bwrap_actually_works() -> bool:
    """Detect whether bubblewrap can actually create a sandboxed subprocess
    on this host, not merely whether the binary is on PATH.

    Some CI/container hosts have bubblewrap installed but the kernel's
    apparmor_restrict_unprivileged_userns=1 (Ubuntu 24.04+ default) blocks
    the unprivileged user namespace bwrap needs to set up ANY sandbox,
    failing every single invocation at startup with "bwrap: setting up uid
    map: Permission denied" -- before the wrapped command ever runs, and
    regardless of network_access (that only avoids the separate
    --unshare-net loopback failure _TestConfig below already works around).
    Tests that aren't testing sandboxing itself (test_reasoning_plan.py,
    test_concurrent_dispatch_regressions.py, test_execute_command_background.py,
    test_verify_command_gate.py) route real execute_command calls through a
    real bwrap invocation and would otherwise spuriously fail every time
    just because the sandbox itself can't start here.
    """
    bwrap = shutil.which("bwrap")
    if not bwrap or not sys.platform.startswith("linux"):
        return False
    try:
        probe = subprocess.run(
            [bwrap, "--die-with-parent", "--ro-bind", "/", "/", "true"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5,
        )
        return probe.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


_BWRAP_WORKS = _bwrap_actually_works()


@dataclasses.dataclass
class _TestConfig(_RealConfig):
    # Ordinary runner integration tests exercise tool dispatch, mutation
    # accounting, validation, and recovery -- not the kernel sandbox itself.
    # Some managed CI/container hosts forbid *all* user-namespace creation,
    # so even bwrap without --unshare-net fails before the command starts.
    # Keep those tests host-independent by disabling sandbox wrapping in the
    # runner's test-only default. test_sandbox.py constructs SandboxPolicy
    # directly with explicit modes and therefore still covers the real
    # fail-closed production behavior.
    sandbox_mode: str = "danger-full-access"
    sandbox_network_access: bool = True
    sandbox_fail_if_unavailable: bool = False
    # When bwrap is installed but can't actually start a sandbox on this
    # host (see _bwrap_actually_works above), fall back to running
    # execute_command directly -- build_sandbox_command skips bwrap
    # entirely for mode="danger-full-access". None of these tests exercise
    # filesystem sandboxing itself (that's test_sandbox.py, which builds
    # SandboxPolicy directly and never goes through Config), so losing
    # that isolation here doesn't weaken what they're testing.
    sandbox_mode: str = "workspace-write" if _BWRAP_WORKS else "danger-full-access"


# providers.py unconditionally loads the real /home/tamfiscode/.env into
# os.environ at import time (that's intentional for production runs -- see
# providers.py's _load_project_env). Left alone, that means routing/model
# selection tests silently pick up whatever this machine's operator has
# configured there (e.g. TAMFIS_CODE_OLLAMA_CODING_MODEL, premium/extra-usage
# opt-ins, provider fallback toggles), so results depend on who's running the
# suite and what their live .env says instead of being deterministic.
_TAMFIS_ENV_PREFIXES = ("TAMFIS_CODE_", "TAMFIS_PROVIDER_")


@pytest.fixture(autouse=True)
def _isolate_tamfis_env(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith(_TAMFIS_ENV_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    # Unit tests must never depend on live localhost daemons. ProviderManager
    # otherwise probes Ollama and Tier IV during construction (and `/doctor`
    # legitimately constructs one), turning a pure rendering test into
    # repeated multi-second network waits on hosts without those services.
    # Tests covering these providers override the flags explicitly.
    monkeypatch.setenv("TAMFIS_PROVIDER_OLLAMA_CLOUD_ENABLED", "false")
    monkeypatch.setenv("TAMFIS_TIER_IV_ENABLED", "false")
    # Never inherit the developer/machine's real login during tests. A
    # `/doctor` rendering test previously found /root's credentials and made
    # a live Remote API request, hanging for the production timeout.
    monkeypatch.setattr(
        _config_module, "CREDENTIALS_PATH", tmp_path / "credentials.json",
    )
    # Several tests that aren't testing sandboxing itself
    # (test_reasoning_plan.py, test_concurrent_dispatch_regressions.py,
    # etc.) route real execute_command calls through
    # _run_local_agent_turn_impl's default Config, which builds a real
    # bwrap invocation. GitHub Actions' ubuntu-latest runner has bubblewrap
    # installed and user namespaces nominally unrestricted, but bwrap's own
    # loopback-interface setup inside a *new network namespace* (triggered
    # by --unshare-net, which build_sandbox_command adds whenever
    # network_access=False) fails there with "Operation not permitted" --
    # an AppArmor/container layer neither this project nor these tests need
    # to fight, since none of them are testing network isolation. See
    # _TestConfig above for the actual fix (network_access=True avoids
    # --unshare-net entirely); sandbox_fail_if_unavailable=False is a
    # separate, secondary guard for the "bwrap not installed at all" case.
    #
    # An env var can't do either override: runner_local.py builds the
    # default config
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
