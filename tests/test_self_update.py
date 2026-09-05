"""tamfis_code.self_update: local version-check against the source checkout."""
from pathlib import Path
import pytest

from tamfis_code.self_update import _parse_version, check_update_available


def test_parse_version_numeric_not_lexical():
    assert _parse_version("1.10.0") > _parse_version("1.9.9")
    assert _parse_version("1.4.0") > _parse_version("1.3.9")
    assert not (_parse_version("1.3.9") > _parse_version("1.4.0"))


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
