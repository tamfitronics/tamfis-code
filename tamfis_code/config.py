"""Configuration and credential storage for TamfisGPT Code.

Precedence (highest wins): CLI flag > environment variable > project-local
.tamfis/config.toml > user ~/.config/tamfis-code/config.toml > built-in
default. Only fields the CLI actually gives distinct behaviour to are
supported here -- see docs/REMOTE_AGENT_MASTER_SPEC.md Phase 21's full
configuration wishlist for what is intentionally deferred.
"""

from __future__ import annotations

import json
import os
import stat
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

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

CONFIG_DIR = Path(os.environ.get("TAMFIS_CODE_CONFIG_HOME", "") or (Path.home() / ".config" / "tamfis-code"))
CREDENTIALS_PATH = CONFIG_DIR / "credentials.json"
USER_CONFIG_PATH = CONFIG_DIR / "config.toml"
PROJECT_CONFIG_RELATIVE = Path(".tamfis") / "config.toml"

DEFAULT_API_BASE = "http://127.0.0.1:9500"


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
    debug: bool = False
    sources: dict[str, str] = field(default_factory=dict)  # field -> where it came from, for `doctor`/`config`

    def as_dict(self) -> dict[str, Any]:
        return {
            "api_base": self.api_base,
            "approval_policy": self.approval_policy,
            "colour": self.colour,
            "output_mode": self.output_mode,
            "timeout_seconds": self.timeout_seconds,
            "debug": self.debug,
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

    env_api_base = os.environ.get("TAMFIS_CODE_API_BASE")
    if env_api_base:
        cfg.api_base = env_api_base
        cfg.sources["api_base"] = "env TAMFIS_CODE_API_BASE"

    env_approval = os.environ.get("TAMFIS_CODE_APPROVAL_POLICY")
    if env_approval in APPROVAL_MODES:
        cfg.approval_policy = env_approval
        cfg.sources["approval_policy"] = "env TAMFIS_CODE_APPROVAL_POLICY"

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
