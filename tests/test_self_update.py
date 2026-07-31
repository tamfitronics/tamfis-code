"""tamfis_code.self_update: local version-check against the source checkout."""
from pathlib import Path

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
