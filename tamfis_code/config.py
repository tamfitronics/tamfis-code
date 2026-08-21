"""Configuration and credential storage for TamfisGPT Code.

Precedence (highest wins): CLI flag > environment variable > project-local
.tamfis/config.toml > platform-native per-user config.toml > built-in
default. Only fields the CLI actually gives distinct behaviour to are
supported here -- see docs/REMOTE_AGENT_MASTER_SPEC.md Phase 21's full
configuration wishlist for what is intentionally deferred.
"""

from __future__ import annotations

import json
import os
import sys
import stat
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

# NOTE: "never" means "never approve" (deny every action outright) --
# it is NOT the opposite-sounding-but-actually-similar "never prompt" that
# "auto"/"full-auto" already mean. Easy to reach for by name alone and get
# the exact opposite of what you wanted; see /mode's own help text and
# --approval's CLI help for the same disambiguation surfaced to the user.
APPROVAL_MODES = (
    "ask", "safe", "auto", "read-only", "plan-only",
    "suggest", "workspace", "full-auto", "never",
    "accept-edits",
)

# Short, user-facing names for /mode -- map to the policy values above.
# "ask" is this CLI's actual default/manual mode, "accept-edits" behaves
# like "safe" (auto-approve unless the server classifies the action
# dangerous, which covers ordinary file edits without covering a
# destructive shell command), "plan" is "plan-only", "auto" is itself.
MODE_ALIASES = {
    "manual": "ask",
    "plan": "plan-only",
    "accept-edits": "accept-edits",
    "auto": "auto",
}

# Reverse of MODE_ALIASES -- unambiguous since every value above is
# distinct. Used to render the raw approval_policy back into its
# short, user-facing name (e.g. for a persistent mode indicator).
_POLICY_TO_MODE_LABEL = {policy: label for label, policy in MODE_ALIASES.items()}

# Shift+Tab cycling order for the interactive REPL's mode indicator --
# mirrors Claude Code's own manual -> accept-edits -> auto cadence, with
# plan (read-only) as the fourth stop rather than folded into the cycle by
# default surprise, since switching INTO a read-only mode by accident via a
# stray keypress is worse than switching between two mutating modes.
MODE_CYCLE = ("manual", "accept-edits", "auto", "plan")


def mode_label_for_policy(policy: str) -> str:
    """The short /mode name for a raw approval_policy value, falling back
    to the raw value itself for policies with no short alias (e.g. the
    --approval-only values like "safe"/"workspace"/"never"/"suggest")."""
    return _POLICY_TO_MODE_LABEL.get(policy, policy)


def next_mode_in_cycle(policy: str) -> str:
    """The next MODE_CYCLE policy after `policy`'s label position. A
    current policy outside the named cycle (e.g. set via --approval to a
    raw value like "safe") starts the cycle fresh from the beginning,
    rather than raising or being a no-op."""
    current_label = mode_label_for_policy(policy)
    try:
        index = MODE_CYCLE.index(current_label)
    except ValueError:
        index = -1
    next_label = MODE_CYCLE[(index + 1) % len(MODE_CYCLE)]
    return MODE_ALIASES[next_label]

def resolve_config_dir(
    *, environment: Optional[Mapping[str, str]] = None,
    platform: Optional[str] = None,
    home: Optional[Path] = None,
) -> Path:
    """Resolve portable per-user storage for an installed Tamfis Code.

    Runtime data never belongs to the source checkout or site-packages:
    either can be read-only and package upgrades replace the latter. The
    explicit override supports portable/container installs; otherwise this
    follows the operating system convention of the user running the CLI.
    """
    env = os.environ if environment is None else environment
    override = str(env.get("TAMFIS_CODE_CONFIG_HOME") or "").strip()
    if override:
        return Path(override).expanduser()

    current_platform = sys.platform if platform is None else platform
    user_home = Path.home() if home is None else Path(home)
    if current_platform.startswith("win"):
        app_data = str(env.get("APPDATA") or env.get("LOCALAPPDATA") or "").strip()
        return (
            Path(app_data) / "tamfis-code"
            if app_data else user_home / "AppData" / "Roaming" / "tamfis-code"
        )

    xdg_config = str(env.get("XDG_CONFIG_HOME") or "").strip()
    if xdg_config:
        return Path(xdg_config).expanduser() / "tamfis-code"
    if current_platform == "darwin":
        return user_home / "Library" / "Application Support" / "tamfis-code"
    return user_home / ".config" / "tamfis-code"


CONFIG_DIR = resolve_config_dir()
CREDENTIALS_PATH = CONFIG_DIR / "credentials.json"
USER_CONFIG_PATH = CONFIG_DIR / "config.toml"
PROJECT_CONFIG_RELATIVE = Path(".tamfis") / "config.toml"

DEFAULT_API_BASE = "https://gpt.tamfitronics.com"


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return {}


@dataclass
class Config:
    api_base: str = DEFAULT_API_BASE
    approval_policy: str = "ask"
    colour: bool = True
    output_mode: str = "text"
    timeout_seconds: float = 120.0
    # Wall-clock budget for one agent "turn" against the runtime controller
    # (runtime/budgets.py). Previously hardcoded at 900s with no way to
    # raise it for slow cloud models; hitting it mid-generation discarded
    # the whole task instead of just ending the turn.
    turn_runtime_seconds: float = 900.0
    # Raw tool calls permitted across one complete task. The outer agent loop
    # already supports three 40-round windows for long, progressing work; the
    # old inner default of 40 contradicted that extension and terminated the
    # task first. Repetition/stall guards remain independently enforced.
    max_tool_calls: int = 120
    # How many times a turn may auto-renew its runtime budget and continue
    # (rather than the task failing outright) when it runs out of time
    # mid-execution. Purely a safety cap on the continuation loop --
    # unrelated budgets (tool-call count, repeated actions, stalls) are
    # untouched by an extension and still fail the task normally.
    max_runtime_extensions: int = 3
    # How many times the shared repair-attempt counter (runtime/budgets.py's
    # max_repair_rounds -- covers provider fallback, empty-continuation
    # recovery, and genuine fix-the-actual-failure repairs alike) may renew
    # instead of failing the task the moment it runs out.
    max_repair_extensions: int = 2
    debug: bool = False
    # Real (LLM-backed) subagent delegation is opt-in: concurrent sessions
    # against the Remote backend have open questions (rate limiting, approval
    # prompts interleaving across sessions, concurrent state.json writers)
    # that need validating against a live backend before being on by default.
    enable_subagent_delegation: bool = False
    # "auto" is retained as a compatibility spelling for the local runtime.
    # Authentication controls subscription/model entitlement only; it must
    # never transfer tool or workspace execution to a remote agent runtime.
    # "standalone": call a provider directly, no TamfisGPT backend.
    # "remote": use the TamfisGPT Remote Workspace backend for every command
    # without needing --remote on each invocation -- set this once (via
    # `tamfis-code login` writing it, or manually in config.toml) for a paid
    # TamfisGPT tenant who wants that be the default, the same way Claude
    # Code/Codex/kimi-code default to their respective hosted accounts.
    default_backend: str = "standalone"
    # Additional local roots that may be used by this installation. The
    # launch directory is always included separately; these roots make a
    # multi-project service practical without granting access to the whole
    # filesystem.
    workspace_roots: list[str] = field(default_factory=list)
    # Shell commands are kernel-isolated by default. File tools retain their
    # existing workspace boundary independently of this setting.
    sandbox_mode: str = "workspace-write"
    sandbox_network_access: bool = False
    sandbox_writable_roots: list[str] = field(default_factory=list)
    # Defaults to fail-closed on Linux as of 2026-08-21 (see sandbox.py's
    # build_sandbox_command docstring -- has no effect on macOS/Windows,
    # which have no sandbox backend to fail closed into yet). Previously
    # False, meaning a Linux host without bwrap installed silently ran
    # every "sandboxed" command with zero kernel isolation.
    sandbox_fail_if_unavailable: bool = True
    # Warns (never blocks -- see metrics.py's SessionCostGuard docstring for
    # why a CLI tool where the user pays for their own usage warrants a
    # softer rail than a hosted multi-tenant product would) once estimated
    # session spend crosses this many dollars. <= 0 disables the warning.
    session_cost_cap_usd: float = 5.0
    permission_allow: list[str] = field(default_factory=list)
    permission_ask: list[str] = field(default_factory=list)
    permission_deny: list[str] = field(default_factory=list)
    sources: dict[str, str] = field(default_factory=dict)  # field -> where it came from, for `doctor`/`config`

    def as_dict(self) -> dict[str, Any]:
        return {
            "api_base": self.api_base,
            "approval_policy": self.approval_policy,
            "colour": self.colour,
            "output_mode": self.output_mode,
            "timeout_seconds": self.timeout_seconds,
            "turn_runtime_seconds": self.turn_runtime_seconds,
            "max_tool_calls": self.max_tool_calls,
            "max_runtime_extensions": self.max_runtime_extensions,
            "max_repair_extensions": self.max_repair_extensions,
            "debug": self.debug,
            "enable_subagent_delegation": self.enable_subagent_delegation,
            "default_backend": self.default_backend,
            "workspace_roots": self.workspace_roots,
            "sandbox_mode": self.sandbox_mode,
            "sandbox_network_access": self.sandbox_network_access,
            "sandbox_writable_roots": self.sandbox_writable_roots,
            "sandbox_fail_if_unavailable": self.sandbox_fail_if_unavailable,
            "session_cost_cap_usd": self.session_cost_cap_usd,
            "permission_allow": self.permission_allow,
            "permission_ask": self.permission_ask,
            "permission_deny": self.permission_deny,
        }


def load_config(project_root: Optional[Path] = None) -> Config:
    cfg = Config()
    cfg.sources = {k: "default" for k in cfg.as_dict()}

    layers: list[tuple[str, dict[str, Any]]] = [
        ("user config", _load_toml(USER_CONFIG_PATH)),
    ]
    if project_root is not None:
        layers.append(("project config", _load_toml(project_root / PROJECT_CONFIG_RELATIVE)))

    for source_name, data in layers:
        if "api_base" in data:
            cfg.api_base = str(data["api_base"])
            cfg.sources["api_base"] = source_name
        if "approval_policy" in data and data["approval_policy"] in APPROVAL_MODES:
            cfg.approval_policy = str(data["approval_policy"])
            cfg.sources["approval_policy"] = source_name
        if "colour" in data:
            cfg.colour = bool(data["colour"])
            cfg.sources["colour"] = source_name
        if "output_mode" in data:
            cfg.output_mode = str(data["output_mode"])
            cfg.sources["output_mode"] = source_name
        if "timeout_seconds" in data:
            cfg.timeout_seconds = float(data["timeout_seconds"])
            cfg.sources["timeout_seconds"] = source_name
        if "turn_runtime_seconds" in data:
            cfg.turn_runtime_seconds = float(data["turn_runtime_seconds"])
            cfg.sources["turn_runtime_seconds"] = source_name
        if "max_tool_calls" in data:
            cfg.max_tool_calls = int(data["max_tool_calls"])
            cfg.sources["max_tool_calls"] = source_name
        if "max_runtime_extensions" in data:
            cfg.max_runtime_extensions = int(data["max_runtime_extensions"])
            cfg.sources["max_runtime_extensions"] = source_name
        if "max_repair_extensions" in data:
            cfg.max_repair_extensions = int(data["max_repair_extensions"])
            cfg.sources["max_repair_extensions"] = source_name
        if "enable_subagent_delegation" in data:
            cfg.enable_subagent_delegation = bool(data["enable_subagent_delegation"])
            cfg.sources["enable_subagent_delegation"] = source_name
        if data.get("default_backend") in ("auto", "standalone", "remote"):
            cfg.default_backend = str(data["default_backend"])
            cfg.sources["default_backend"] = source_name
        configured_roots = data.get("workspace_roots")
        if isinstance(configured_roots, list):
            cfg.workspace_roots = [
                str(Path(item).expanduser().resolve())
                for item in configured_roots
                if isinstance(item, str) and item.strip()
            ]
            cfg.sources["workspace_roots"] = source_name
        if data.get("sandbox_mode") in ("read-only", "workspace-write", "danger-full-access"):
            cfg.sandbox_mode = str(data["sandbox_mode"])
            cfg.sources["sandbox_mode"] = source_name
        if "sandbox_network_access" in data:
            cfg.sandbox_network_access = bool(data["sandbox_network_access"])
            cfg.sources["sandbox_network_access"] = source_name
        sandbox_roots = data.get("sandbox_writable_roots")
        if isinstance(sandbox_roots, list):
            cfg.sandbox_writable_roots = [
                str(Path(item).expanduser().resolve()) for item in sandbox_roots
                if isinstance(item, str) and item.strip()
            ]
            cfg.sources["sandbox_writable_roots"] = source_name
        if "sandbox_fail_if_unavailable" in data:
            cfg.sandbox_fail_if_unavailable = bool(data["sandbox_fail_if_unavailable"])
            cfg.sources["sandbox_fail_if_unavailable"] = source_name
        if "session_cost_cap_usd" in data:
            cfg.session_cost_cap_usd = float(data["session_cost_cap_usd"])
            cfg.sources["session_cost_cap_usd"] = source_name
        permissions = data.get("permissions")
        if isinstance(permissions, dict):
            for action in ("allow", "ask", "deny"):
                values = permissions.get(action)
                if isinstance(values, list):
                    field_name = f"permission_{action}"
                    combined = [*getattr(cfg, field_name), *(
                        str(value) for value in values if str(value).strip()
                    )]
                    setattr(cfg, field_name, list(dict.fromkeys(combined)))
                    cfg.sources[field_name] = source_name

    env_api_base = os.environ.get("TAMFIS_CODE_API_BASE")
    if env_api_base:
        cfg.api_base = env_api_base
        cfg.sources["api_base"] = "env TAMFIS_CODE_API_BASE"

    env_approval = os.environ.get("TAMFIS_CODE_APPROVAL_POLICY")
    if env_approval in APPROVAL_MODES:
        cfg.approval_policy = env_approval
        cfg.sources["approval_policy"] = "env TAMFIS_CODE_APPROVAL_POLICY"

    env_delegation = os.environ.get("TAMFIS_CODE_ENABLE_SUBAGENT_DELEGATION")
    if env_delegation is not None:
        cfg.enable_subagent_delegation = env_delegation.lower() in {"1", "true", "yes"}
        cfg.sources["enable_subagent_delegation"] = "env TAMFIS_CODE_ENABLE_SUBAGENT_DELEGATION"

    env_backend = os.environ.get("TAMFIS_CODE_DEFAULT_BACKEND")
    if env_backend in ("auto", "standalone", "remote"):
        cfg.default_backend = env_backend
        cfg.sources["default_backend"] = "env TAMFIS_CODE_DEFAULT_BACKEND"

    env_turn_runtime = os.environ.get("TAMFIS_CODE_TURN_RUNTIME_SECONDS")
    if env_turn_runtime:
        cfg.turn_runtime_seconds = float(env_turn_runtime)
        cfg.sources["turn_runtime_seconds"] = "env TAMFIS_CODE_TURN_RUNTIME_SECONDS"

    env_max_tool_calls = os.environ.get("TAMFIS_CODE_MAX_TOOL_CALLS")
    if env_max_tool_calls:
        cfg.max_tool_calls = int(env_max_tool_calls)
        cfg.sources["max_tool_calls"] = "env TAMFIS_CODE_MAX_TOOL_CALLS"

    env_runtime_extensions = os.environ.get("TAMFIS_CODE_MAX_RUNTIME_EXTENSIONS")
    if env_runtime_extensions:
        cfg.max_runtime_extensions = int(env_runtime_extensions)
        cfg.sources["max_runtime_extensions"] = "env TAMFIS_CODE_MAX_RUNTIME_EXTENSIONS"

    env_repair_extensions = os.environ.get("TAMFIS_CODE_MAX_REPAIR_EXTENSIONS")
    if env_repair_extensions:
        cfg.max_repair_extensions = int(env_repair_extensions)
        cfg.sources["max_repair_extensions"] = "env TAMFIS_CODE_MAX_REPAIR_EXTENSIONS"

    env_roots = os.environ.get("TAMFIS_CODE_WORKSPACE_ROOTS")
    if env_roots:
        cfg.workspace_roots = [
            str(Path(item.strip()).expanduser().resolve())
            for item in env_roots.split(os.pathsep)
            if item.strip()
        ]
        cfg.sources["workspace_roots"] = "env TAMFIS_CODE_WORKSPACE_ROOTS"

    env_sandbox = os.environ.get("TAMFIS_CODE_SANDBOX_MODE")
    if env_sandbox in ("read-only", "workspace-write", "danger-full-access"):
        cfg.sandbox_mode = env_sandbox
        cfg.sources["sandbox_mode"] = "env TAMFIS_CODE_SANDBOX_MODE"

    env_sandbox_network = os.environ.get("TAMFIS_CODE_SANDBOX_NETWORK_ACCESS")
    if env_sandbox_network is not None:
        cfg.sandbox_network_access = env_sandbox_network.lower() in {"1", "true", "yes"}
        cfg.sources["sandbox_network_access"] = "env TAMFIS_CODE_SANDBOX_NETWORK_ACCESS"

    env_sandbox_fail = os.environ.get("TAMFIS_CODE_SANDBOX_FAIL_IF_UNAVAILABLE")
    if env_sandbox_fail is not None:
        cfg.sandbox_fail_if_unavailable = env_sandbox_fail.lower() in {"1", "true", "yes"}
        cfg.sources["sandbox_fail_if_unavailable"] = "env TAMFIS_CODE_SANDBOX_FAIL_IF_UNAVAILABLE"

    env_cost_cap = os.environ.get("TAMFIS_CODE_SESSION_COST_CAP_USD")
    if env_cost_cap:
        cfg.session_cost_cap_usd = float(env_cost_cap)
        cfg.sources["session_cost_cap_usd"] = "env TAMFIS_CODE_SESSION_COST_CAP_USD"

    env_sandbox_roots = os.environ.get("TAMFIS_CODE_SANDBOX_WRITABLE_ROOTS")
    if env_sandbox_roots:
        cfg.sandbox_writable_roots = [
            str(Path(item.strip()).expanduser().resolve())
            for item in env_sandbox_roots.split(os.pathsep) if item.strip()
        ]
        cfg.sources["sandbox_writable_roots"] = "env TAMFIS_CODE_SANDBOX_WRITABLE_ROOTS"

    return cfg


@dataclass
class Credentials:
    access_token: str
    refresh_token: Optional[str] = None
    user_id: Optional[str] = None
    email: Optional[str] = None


def load_credentials() -> Optional[Credentials]:
    env_token = os.environ.get("TAMFIS_CODE_TOKEN")
    if env_token:
        return Credentials(access_token=env_token)

    if not CREDENTIALS_PATH.is_file():
        return None
    try:
        data = json.loads(CREDENTIALS_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    access_token = data.get("access_token")
    if not access_token:
        return None
    return Credentials(
        access_token=access_token,
        refresh_token=data.get("refresh_token"),
        user_id=data.get("user_id"),
        email=data.get("email"),
    )


def save_credentials(creds: Credentials) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    # Only chmod when needed -- see the identical fix in state.py's
    # _save_raw for why doing this unconditionally is a landmine (raises
    # PermissionError the moment CONFIG_DIR is ever owned by a different user
    # than the caller).
    if stat.S_IMODE(os.stat(CONFIG_DIR).st_mode) != stat.S_IRWXU:
        os.chmod(CONFIG_DIR, stat.S_IRWXU)  # 0700 -- owner-only, matches spec's "restrictive permissions"
    payload = {
        "access_token": creds.access_token,
        "refresh_token": creds.refresh_token,
        "user_id": creds.user_id,
        "email": creds.email,
    }
    CREDENTIALS_PATH.write_text(json.dumps(payload))
    os.chmod(CREDENTIALS_PATH, stat.S_IRUSR | stat.S_IWUSR)  # 0600


def clear_credentials() -> bool:
    if CREDENTIALS_PATH.is_file():
        CREDENTIALS_PATH.unlink()
        return True
    return False
