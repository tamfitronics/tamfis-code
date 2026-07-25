"""Packaging/entry-point regression guards.

This project has been bitten more than once by the installed `tamfis-code`
binary silently drifting from the source tree (a stale wheel left in place
after a source fix, an undeclared dependency that only worked because a
copy happened to already be installed system-wide -- see pyproject.toml's
comments on `openai`/`mcp`). None of that is something a normal pytest run
*can* catch, since pytest runs against the source tree, not the installed
copy -- but this file at least guards the parts that live in the source
tree itself and have caused real incidents before: the [project.scripts]
entry points actually resolving to a working CLI, and every package under
tamfis_code/ being real Python packages `packages.find` can discover.
"""
from __future__ import annotations

import sys
import unittest
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - only exercised on Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

from click.testing import CliRunner

from tamfis_code import __version__

REPO_ROOT = Path(__file__).resolve().parent.parent


class EntryPointResolutionTests(unittest.TestCase):
    """[project.scripts] must resolve to a real, importable, callable target."""

    def setUp(self):
        pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.scripts = pyproject["project"]["scripts"]

    def test_at_least_one_script_is_declared(self):
        self.assertTrue(self.scripts)

    def test_every_declared_script_resolves_to_an_importable_callable(self):
        for command_name, target in self.scripts.items():
            with self.subTest(command=command_name, target=target):
                module_path, _, attr = target.partition(":")
                module = import_module(module_path)
                entry = getattr(module, attr)
                self.assertTrue(callable(entry))

    def test_every_declared_script_points_at_the_same_cli_entry(self):
        # tamfis-code / tamgpt-code / tamfis are meant to be aliases of the
        # same CLI, not accidentally-diverged separate entry points. The
        # dedicated tamfis-code-server binary is intentionally a different
        # service entry point.
        targets = {
            target for name, target in self.scripts.items()
            if name != "tamfis-code-server"
        }
        self.assertEqual(len(targets), 1, f"expected one shared entry point, found {targets}")


class CliSmokeTests(unittest.TestCase):
    """A broken import inside cli.py breaks installation silently until
    someone actually runs the installed binary -- catch that here instead.

    `main` (the [project.scripts] target, invoked by the installed binary)
    is a plain wrapper that calls the real click.group `cli` -- CliRunner
    needs the click Command object itself, not the wrapper, to invoke
    --help/--version the way an installed user would.
    """

    def setUp(self):
        self.runner = CliRunner()

    def test_help_exits_cleanly(self):
        from tamfis_code.cli import cli
        result = self.runner.invoke(cli, ["--help"])
        self.assertEqual(result.exit_code, 0, result.output)

    def test_version_flag_matches_package_version(self):
        from tamfis_code.cli import cli
        result = self.runner.invoke(cli, ["--version"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn(__version__, result.output)


class PackagingDiscoveryTests(unittest.TestCase):
    """[tool.setuptools.packages.find] only discovers directories that are
    real packages (have __init__.py) -- a subpackage missing one would be
    silently left out of the built wheel with no build-time error."""

    def test_every_directory_under_tamfis_code_with_python_files_is_a_package(self):
        package_root = REPO_ROOT / "tamfis_code"
        missing_init = []
        for path in package_root.rglob("*.py"):
            directory = path.parent
            if directory.name == "__pycache__":
                continue
            if not (directory / "__init__.py").exists():
                missing_init.append(str(directory.relative_to(REPO_ROOT)))
        self.assertEqual(sorted(set(missing_init)), [])


if __name__ == "__main__":
    unittest.main()
