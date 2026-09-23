"""tamfis_code.self_update: local version-check against the source checkout."""
from pathlib import Path
import pytest

from tamfis_code.self_update import (
    _REQUIRED_WHEEL_MODULES,
    _parse_version,
    _pip_install_command,
    _wheel_has_required_runtime,
    check_update_available,
)


def test_parse_version_numeric_not_lexical():
    assert _parse_version("1.10.0") > _parse_version("1.9.9")
    assert _parse_version("1.4.0") > _parse_version("1.3.9")
    assert not (_parse_version("1.3.9") > _parse_version("1.4.0"))


def test_pip_command_allows_explicit_update_on_externally_managed_system_python(monkeypatch, tmp_path):
    from tamfis_code import self_update
    marker = tmp_path / "EXTERNALLY-MANAGED"
    marker.write_text("")
    monkeypatch.setattr(self_update.sysconfig, "get_path", lambda _name: str(tmp_path))
    monkeypatch.setattr(self_update.sys, "prefix", "/usr")
    monkeypatch.setattr(self_update.sys, "base_prefix", "/usr")

    command = _pip_install_command("--upgrade", "/tmp/release.whl")
    assert "--break-system-packages" in command
    assert command[-2:] == ["--upgrade", "/tmp/release.whl"]


def test_pip_command_never_uses_system_override_inside_a_venv(monkeypatch, tmp_path):
    from tamfis_code import self_update
    (tmp_path / "EXTERNALLY-MANAGED").write_text("")
    monkeypatch.setattr(self_update.sysconfig, "get_path", lambda _name: str(tmp_path))
    monkeypatch.setattr(self_update.sys, "prefix", "/venv")
    monkeypatch.setattr(self_update.sys, "base_prefix", "/usr")

    assert "--break-system-packages" not in _pip_install_command("--upgrade", "/tmp/release.whl")


def test_check_update_available_newer(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\nversion = "9.9.9"\n')
    assert check_update_available(tmp_path) == "9.9.9"


def test_check_update_available_not_newer(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\nversion = "0.0.1"\n')
    assert check_update_available(tmp_path) is None


def test_check_update_available_missing_repo(tmp_path: Path):
    assert check_update_available(tmp_path / "does-not-exist") is None


def test_published_release_without_source_checkout(monkeypatch, tmp_path):
    from tamfis_code import self_update
    monkeypatch.setattr(self_update, "DEFAULT_REPO_PATH", tmp_path)
    monkeypatch.setattr(self_update, "_remote_release", lambda: {"version": "9.9.9"})
    assert self_update.check_update_available() == "9.9.9"


def test_checksum_failure_never_runs_installer(monkeypatch, tmp_path):
    import io
    from tamfis_code import self_update
    monkeypatch.setattr(self_update, "DEFAULT_REPO_PATH", tmp_path)
    monkeypatch.setattr(self_update, "_remote_release", lambda: {
        "version": "9.9.9", "url": self_update.RELEASE_BASE + "/tamfis_code-9.9.9-py3-none-any.whl", "sha256": "0" * 64,
    })
    monkeypatch.setattr(self_update, "urlopen", lambda *a, **k: io.BytesIO(b"tampered"))
    monkeypatch.setattr(self_update.subprocess, "run", lambda *a, **k: pytest.fail("installer must not run"))
    assert self_update.apply_update()[0] is False


def test_update_rejects_wheel_missing_critical_runtime_module(tmp_path):
    import zipfile

    wheel = tmp_path / "broken.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for member in _REQUIRED_WHEEL_MODULES[:-1]:
            archive.writestr(member, "")

    valid, reason = _wheel_has_required_runtime(wheel)
    assert valid is False
    assert "openhands/tools.py" in reason


def test_update_accepts_wheel_with_critical_runtime_modules(tmp_path):
    import zipfile

    wheel = tmp_path / "complete.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for member in _REQUIRED_WHEEL_MODULES:
            archive.writestr(member, "")

    assert _wheel_has_required_runtime(wheel) == (True, "")
