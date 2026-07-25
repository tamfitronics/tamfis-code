"""Regression guard against tamfis_code.__version__ drifting from
pyproject.toml's [project].version.

Confirmed live: __init__.py's __version__ was hard-coded to "0.4.2" while
pyproject.toml (the package's actual installed/published version) had
already moved to "0.4.4" -- `tamfis-code --version` (backed by
click.version_option(__version__, ...) in cli.py) reported the stale value
even from a freshly built and installed wheel. Nothing kept the two in sync
because pyproject.toml's version isn't `dynamic` (sourced from the package),
so a manual bump to one and not the other silently drifts.
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - only exercised on Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

from tamfis_code import __version__

REPO_ROOT = Path(__file__).resolve().parent.parent


class VersionConsistencyTests(unittest.TestCase):
    def test_package_version_matches_pyproject_toml(self):
        pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        declared = pyproject["project"]["version"]
        self.assertEqual(
            __version__, declared,
            "tamfis_code.__version__ and pyproject.toml's [project].version "
            "must be bumped together -- see this test's module docstring.",
        )

    def test_version_string_is_a_plain_semver(self):
        self.assertRegex(__version__, re.compile(r"^\d+\.\d+\.\d+$"))
