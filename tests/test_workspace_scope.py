"""Multi-stack discovery and scope enforcement.

Regression coverage for: bounded multi-stack discovery in a parent folder
containing several interconnected projects (spec test #8), exclusion of
unrelated/generated/archived roots from that discovery (spec test #9), and
prevention of a broad, unscoped parent-directory scan (spec test #10).
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tamfis_code.runner_local import (
    _detect_workspace_scope,
    _excluded_root_names,
    _scope_tool_arguments,
)
from tamfis_code.workspace import classify_root


def _make_project(root: Path, name: str, *, marker: str = "pyproject.toml") -> Path:
    project = root / name
    project.mkdir()
    (project / marker).write_text("[project]\nname = 'x'\n" if marker == "pyproject.toml" else "{}")
    return project


class ClassifyRootTests(unittest.TestCase):
    def test_active_project_with_manifest(self):
        with tempfile.TemporaryDirectory() as ws:
            project = _make_project(Path(ws), "svc")
            self.assertEqual(classify_root(project), "active")

    def test_backup_directory_is_archived_even_with_a_manifest(self):
        with tempfile.TemporaryDirectory() as ws:
            project = _make_project(Path(ws), "svc_backups")
            self.assertEqual(classify_root(project), "archived")

    def test_dist_directory_is_generated(self):
        with tempfile.TemporaryDirectory() as ws:
            project = _make_project(Path(ws), "dist")
            self.assertEqual(classify_root(project), "generated")

    def test_node_modules_is_dependency(self):
        with tempfile.TemporaryDirectory() as ws:
            (Path(ws) / "node_modules").mkdir()
            self.assertEqual(classify_root(Path(ws) / "node_modules"), "dependency")

    def test_plain_directory_with_no_manifest_is_unrelated(self):
        with tempfile.TemporaryDirectory() as ws:
            plain = Path(ws) / "notes"
            plain.mkdir()
            self.assertEqual(classify_root(plain), "unrelated")


class DetectWorkspaceScopeTests(unittest.TestCase):
    def test_multi_stack_discovery_selects_only_active_project_roots(self):
        with tempfile.TemporaryDirectory() as ws:
            root = Path(ws)
            backend = _make_project(root, "backend")
            frontend = _make_project(root, "frontend", marker="package.json")
            _make_project(root, "backend_backups")  # archived -- must be excluded
            (root / ".cache").mkdir()  # no manifest -- unrelated, excluded
            (root / "notes.txt").write_text("hi")

            scope = _detect_workspace_scope(str(root), "audit this workspace end to end")

            scope_set = {str(p) for p in scope}
            self.assertIn(str(backend), scope_set)
            self.assertIn(str(frontend), scope_set)
            self.assertNotIn(str(root / "backend_backups"), scope_set)
            self.assertNotIn(str(root / ".cache"), scope_set)

    def test_single_active_project_workspace_is_its_own_scope(self):
        with tempfile.TemporaryDirectory() as ws:
            root = Path(ws)
            (root / "pyproject.toml").write_text("[project]\nname='x'\n")
            scope = _detect_workspace_scope(str(root), "fix the bug")
            self.assertEqual(scope, [root.resolve()])

    def test_explicit_absolute_project_path_expands_scope_from_admin_cwd(self):
        with tempfile.TemporaryDirectory() as parent, tempfile.TemporaryDirectory() as admin:
            project = Path(parent) / "tamfisseo"
            project.mkdir()
            (project / "package.json").write_text("{}")
            scope = _detect_workspace_scope(
                str(Path(admin)),
                f"check the project at {project / 'package.json'} and fix its bugs",
            )
            self.assertEqual(scope, [project.resolve()])

    def test_explicitly_named_root_is_honored_even_if_archived(self):
        """A user who explicitly names a backup directory is making a
        deliberate request -- classification only governs *implicit*
        (heuristic) selection, never overrides an explicit name."""
        with tempfile.TemporaryDirectory() as ws:
            root = Path(ws)
            _make_project(root, "backend")
            backups = _make_project(root, "old_backups")

            scope = _detect_workspace_scope(str(root), "look inside old_backups for the removed file")
            self.assertIn(backups.resolve(), scope)

    def test_broad_parent_directory_search_command_requires_escalation(self):
        with tempfile.TemporaryDirectory() as ws:
            root = Path(ws)
            backend = _make_project(root, "backend")
            frontend = _make_project(root, "frontend", marker="package.json")
            scope_roots = [backend.resolve(), frontend.resolve()]

            scoped, error = _scope_tool_arguments(
                "execute_command",
                {"command": "grep -r TODO .", "cwd": str(root)},
                workspace_root=str(root),
                scope_roots=scope_roots,
            )
            self.assertIsNone(error)
            self.assertIn(str(root.resolve()), scoped["_tamfis_external_scope_paths"])
            self.assertEqual(scoped["sandbox_permissions"], "require_escalated")

    def test_command_absolute_operand_outside_scope_requires_escalation(self):
        with tempfile.TemporaryDirectory() as ws, tempfile.TemporaryDirectory() as outside:
            root = Path(ws)
            project = _make_project(root, "backend")

            scoped, error = _scope_tool_arguments(
                "execute_command",
                {"command": f"find {outside} -type f", "cwd": str(project)},
                workspace_root=str(root),
                scope_roots=[project.resolve()],
            )

            self.assertIsNone(error)
            self.assertIn(str(Path(outside).resolve()), scoped["_tamfis_external_scope_paths"])
            self.assertEqual(scoped["sandbox_permissions"], "require_escalated")

    def test_read_only_postgresql_config_inspection_is_allowed(self):
        with tempfile.TemporaryDirectory() as ws:
            root = Path(ws)
            project = _make_project(root, "backend")
            scoped, error = _scope_tool_arguments(
                "execute_command",
                {
                    "command": "cat /etc/postgresql/17/main/pg_hba.conf",
                    "cwd": str(project),
                },
                workspace_root=str(root),
                scope_roots=[project.resolve()],
            )
            self.assertIsNone(error)
            self.assertEqual(scoped["cwd"], str(project.resolve()))
            self.assertIn(
                "/etc/postgresql/17/main/pg_hba.conf",
                scoped["_tamfis_external_scope_paths"],
            )
            self.assertEqual(scoped["sandbox_permissions"], "require_escalated")

    def test_read_only_postgresql_socket_inspection_is_allowed_after_symlink_resolution(self):
        with tempfile.TemporaryDirectory() as ws:
            root = Path(ws)
            project = _make_project(root, "backend")
            scoped, error = _scope_tool_arguments(
                "execute_command",
                {
                    "command": "ls -l /var/run/postgresql/.s.PGSQL.5432",
                    "cwd": str(project),
                },
                workspace_root=str(root),
                scope_roots=[project.resolve()],
            )
            self.assertIsNone(error)
            self.assertIn(
                str(Path("/var/run/postgresql/.s.PGSQL.5432").resolve()),
                scoped["_tamfis_external_scope_paths"],
            )

    def test_any_external_path_uses_the_same_explicit_escalation_flow(self):
        with tempfile.TemporaryDirectory() as ws:
            root = Path(ws)
            project = _make_project(root, "backend")
            for command in (
                "rm /run/postgresql/.s.PGSQL.5432",
                "cat /etc/shadow",
                "cat /etc/postgresql/server.key",
            ):
                with self.subTest(command=command):
                    scoped, error = _scope_tool_arguments(
                        "execute_command",
                        {"command": command, "cwd": str(project)},
                        workspace_root=str(root),
                        scope_roots=[project.resolve()],
                    )
                    self.assertIsNone(error)
                    self.assertTrue(scoped["_tamfis_external_scope_paths"])
                    self.assertEqual(scoped["sandbox_permissions"], "require_escalated")

    def test_direct_file_tool_outside_scope_requires_escalation(self):
        with tempfile.TemporaryDirectory() as ws, tempfile.TemporaryDirectory() as outside:
            root = Path(ws)
            project = _make_project(root, "backend")
            target = Path(outside) / "config.txt"
            scoped, error = _scope_tool_arguments(
                "read_file",
                {"path": str(target)},
                workspace_root=str(root),
                scope_roots=[project.resolve()],
            )
            self.assertIsNone(error)
            self.assertEqual(scoped["path"], str(target.resolve()))
            self.assertEqual(scoped["_tamfis_external_scope_paths"], [str(target.resolve())])

    def test_relative_file_tool_path_is_resolved_inside_single_nested_scope(self):
        with tempfile.TemporaryDirectory() as ws:
            root = Path(ws)
            project = _make_project(root, "backend")
            focused = project / "src" / "feature"
            focused.mkdir(parents=True)

            scoped, error = _scope_tool_arguments(
                "read_file",
                {"path": "handler.py"},
                workspace_root=str(root),
                scope_roots=[focused.resolve()],
            )

            self.assertIsNone(error)
            self.assertEqual(scoped["path"], str(focused / "handler.py"))
            self.assertNotIn("_tamfis_external_scope_paths", scoped)

    def test_heredoc_source_is_not_parsed_as_command_paths(self):
        with tempfile.TemporaryDirectory() as ws:
            root = Path(ws)
            project = _make_project(root, "tamfisseo")
            command = "cat > /tmp-placeholder/api.ts << 'EOF'\nimport { readFile } from 'node:fs/promises';\nconst route = '/';\nEOF"
            command = command.replace("/tmp-placeholder", str(project))
            scoped, error = _scope_tool_arguments(
                "execute_command",
                {"command": command, "cwd": str(root)},
                workspace_root=str(root),
                scope_roots=[project.resolve()],
            )
            self.assertIsNone(error)
            self.assertEqual(scoped["cwd"], str(project.resolve()))

    def test_do_not_touch_excludes_the_launch_directory_and_routes_to_siblings(self):
        """Confirmed live: launching tamfis-code from inside its own repo
        with the objective "audit the TamfisGPT iOS full stack. Identify it
        and do not touch tamfis-code" scoped straight to tamfis-code itself
        and started reading files there -- the exact thing it was told not
        to do. Two compounding causes: (1) nothing parsed "do not touch X"
        as an exclusion at all, and (2) the "stack" shortcut only ever
        looked at the launch directory's own children, which can never
        contain its own siblings, so it silently found nothing and fell
        back to "the workspace itself is a project root" -- tamfis-code."""
        with tempfile.TemporaryDirectory() as ws:
            root = Path(ws)
            backend = _make_project(root, "tamgpt6")
            frontend = _make_project(root, "tamfis-frontend", marker="package.json")
            cli = _make_project(root, "tamfis-code")

            scope = _detect_workspace_scope(
                str(cli), "audit the TamfisGPT iOS full stack. Identify it and do not touch tamfis-code",
            )

            scope_set = {str(p) for p in scope}
            self.assertNotIn(str(cli.resolve()), scope_set)
            self.assertIn(str(backend.resolve()), scope_set)
            self.assertIn(str(frontend.resolve()), scope_set)

    def test_excluded_root_name_is_never_selected_even_when_explicitly_named(self):
        """An exclusion instruction must beat the "explicitly named roots
        take precedence" rule -- naming something in the same sentence as
        "do not touch" is exactly what marks it excluded, not selected."""
        with tempfile.TemporaryDirectory() as ws:
            root = Path(ws)
            _make_project(root, "backend")
            forbidden = _make_project(root, "legacy_service")

            scope = _detect_workspace_scope(
                str(root), "inspect the repo, do not touch legacy_service",
            )

            self.assertNotIn(forbidden.resolve(), scope)

    def test_excluded_root_names_matches_within_a_bounded_window_of_the_trigger(self):
        excluded = _excluded_root_names(
            "audit the ios stack, do not touch tamfis-code please",
            {"tamfis-code", "tamgpt6", "tamfis-frontend"},
        )
        self.assertEqual(excluded, {"tamfis-code"})

    def test_search_code_targeting_the_common_parent_requires_escalation_for_a_stack(self):
        with tempfile.TemporaryDirectory() as ws:
            root = Path(ws)
            backend = _make_project(root, "backend")
            frontend = _make_project(root, "frontend", marker="package.json")
            scope_roots = [backend.resolve(), frontend.resolve()]

            scoped, error = _scope_tool_arguments(
                "search_code",
                {"query": "TODO", "path": str(root)},
                workspace_root=str(root),
                scope_roots=scope_roots,
            )
            self.assertIsNone(error)
            self.assertEqual(scoped["_tamfis_external_scope_paths"], [str(root.resolve())])


if __name__ == "__main__":
    unittest.main()
