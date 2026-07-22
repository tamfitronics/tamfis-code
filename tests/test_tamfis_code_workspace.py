import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tamfis_code import state as state_module
from tamfis_code.workspace import (
    _indexable_files, _project_metadata, blocking_dirty_files, build_system_prompt, classify_root,
    context_from_session, discover_local_repository, find_resumable_session,
    resolve_local_workspace, resolve_workspace,
)


class WordpressProjectMetadataTests(unittest.TestCase):
    """Confirmed live: a real WordPress site with no package.json/
    composer.json produced empty detected_languages/frameworks, so the
    model had no grounding signal at all and defaulted to Node/React
    conventions (looking for package.json) -- even when the user's own
    objective said "this is a WordPress site, not a React component"."""

    def test_wp_config_marks_php_and_wordpress_with_no_manifest_at_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "wp-config.php").write_text("<?php // config\n")
            (root / "index.php").write_text("<?php // front controller\n")
            files = [root / "wp-config.php", root / "index.php"]

            metadata = _project_metadata(root, files)

            self.assertEqual(metadata["project_manifests"], [])
            self.assertIn("PHP", metadata["detected_languages"])
            self.assertIn("WordPress", metadata["frameworks"])

    def test_theme_style_css_header_marks_wordpress(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "style.css").write_text("/*\nTheme Name: Example Theme\n*/\n")
            (root / "functions.php").write_text("<?php\n")
            files = [root / "style.css", root / "functions.php"]

            metadata = _project_metadata(root, files)

            self.assertIn("PHP", metadata["detected_languages"])
            self.assertIn("WordPress", metadata["frameworks"])

    def test_plugin_header_in_top_level_php_file_marks_wordpress(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "my-plugin.php").write_text(
                "<?php\n/**\n * Plugin Name: My Example Plugin\n */\n"
            )
            files = [root / "my-plugin.php"]

            metadata = _project_metadata(root, files)

            self.assertIn("WordPress", metadata["frameworks"])

    def test_ordinary_stylesheet_without_a_theme_header_is_not_wordpress(self):
        """A plain style.css (the overwhelming majority of them) must not
        false-positive just because the filename matches -- only a real
        WordPress "Theme Name:" header counts."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "style.css").write_text("body { margin: 0; }\n")
            (root / "package.json").write_text('{"name": "frontend"}')
            files = [root / "style.css", root / "package.json"]

            metadata = _project_metadata(root, files)

            self.assertNotIn("WordPress", metadata["frameworks"])
            # _discover_project_type distinguishes JavaScript from
            # TypeScript (via tsconfig.json, absent here) and its specific
            # label now supersedes MANIFEST_LANGUAGE_MAP's generic
            # "JavaScript/TypeScript" instead of both appearing together.
            self.assertEqual(metadata["detected_languages"], ["JavaScript"])

    def test_classify_root_treats_a_bare_wp_install_as_active_not_unrelated(self):
        """classify_root/has_project_marker previously only recognized
        MANIFEST_LANGUAGE_MAP filenames + a few infra files -- a bare
        WordPress checkout with none of those (the common case) was
        silently classified as "unrelated", not "active"."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "wp-load.php").write_text("<?php\n")

            self.assertEqual(classify_root(root), "active")


class _StatePatchMixin:
    """Redirects state.py's CONFIG_DIR/STATE_PATH to a temp dir so tests
    that create sessions (resolve_workspace/context_from_session/
    find_resumable_session all call local_state.save_session_state) never
    touch the real ~/.config/tamfis-code/state.json. Without this, tests
    using fixed ids like 10/42/1001 and workspace roots like /tmp/proj-c
    silently write those sessions into the real state file on every test
    run -- a real bug found via `tamfis-code sessions` showing stray
    test-injected sessions."""

    def setUp(self):
        self._originals = (state_module.CONFIG_DIR, state_module.STATE_PATH)
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        state_module.CONFIG_DIR = base / ".config"
        state_module.STATE_PATH = base / ".config" / "state.json"

    def tearDown(self):
        state_module.CONFIG_DIR, state_module.STATE_PATH = self._originals
        self.tmp.cleanup()


class FakeWorkspaceClient:
    def __init__(self, servers=None, sessions=None):
        self.servers = list(servers or [])
        self.sessions = list(sessions or [])
        self.created_sessions: list[dict] = []
        self.registered_servers: list[dict] = []
        self._next_session_id = 1000

    async def list_servers(self):
        return self.servers

    async def register_local_server(self, name):
        server = {"id": 100, "transport_type": "local", "name": name}
        self.servers.append(server)
        self.registered_servers.append(server)
        return {"server": server, "fingerprint": "x"}

    async def list_sessions(self):
        return self.sessions

    async def create_session(self, server_id, working_directory=None):
        self._next_session_id += 1
        session = {"id": self._next_session_id, "server_id": server_id, "working_directory": working_directory, "status": "idle"}
        self.sessions.append(session)
        self.created_sessions.append(session)
        return session

    async def get_session(self, session_id):
        for session in self.sessions:
            if session["id"] == session_id:
                return session
        raise KeyError(session_id)


class ResolveWorkspaceTests(_StatePatchMixin, unittest.TestCase):
    def test_registers_local_server_when_none_exists(self):
        client = FakeWorkspaceClient()
        ctx = asyncio.run(resolve_workspace(client, cwd=Path("/tmp/proj-a")))
        self.assertEqual(len(client.registered_servers), 1)
        self.assertEqual(ctx.server_id, 100)
        self.assertEqual(ctx.workspace_root, str(Path("/tmp/proj-a").resolve()))

    def test_reuses_existing_local_server(self):
        client = FakeWorkspaceClient(servers=[{"id": 5, "transport_type": "local"}])
        asyncio.run(resolve_workspace(client, cwd=Path("/tmp/proj-b")))
        self.assertEqual(client.registered_servers, [])  # no new registration needed

    def test_reuses_existing_session_for_same_workspace_root(self):
        root = str(Path("/tmp/proj-c").resolve())
        client = FakeWorkspaceClient(
            servers=[{"id": 5, "transport_type": "local"}],
            sessions=[{"id": 10, "server_id": 5, "working_directory": root, "status": "idle"}],
        )
        ctx = asyncio.run(resolve_workspace(client, cwd=Path("/tmp/proj-c")))
        self.assertEqual(ctx.session_id, 10)
        self.assertEqual(client.created_sessions, [])

    def test_closed_session_is_not_reused(self):
        root = str(Path("/tmp/proj-d").resolve())
        client = FakeWorkspaceClient(
            servers=[{"id": 5, "transport_type": "local"}],
            sessions=[{"id": 10, "server_id": 5, "working_directory": root, "status": "closed"}],
        )
        ctx = asyncio.run(resolve_workspace(client, cwd=Path("/tmp/proj-d")))
        self.assertNotEqual(ctx.session_id, 10)
        self.assertEqual(len(client.created_sessions), 1)

    def test_different_workspace_root_does_not_reuse_session(self):
        client = FakeWorkspaceClient(
            servers=[{"id": 5, "transport_type": "local"}],
            sessions=[{"id": 10, "server_id": 5, "working_directory": "/some/other/path", "status": "idle"}],
        )
        ctx = asyncio.run(resolve_workspace(client, cwd=Path("/tmp/proj-e")))
        self.assertNotEqual(ctx.session_id, 10)


class ContextFromSessionTests(_StatePatchMixin, unittest.TestCase):
    def test_builds_context_from_existing_session(self):
        client = FakeWorkspaceClient(sessions=[{"id": 42, "server_id": 7, "working_directory": "/srv/app"}])
        ctx = asyncio.run(context_from_session(client, 42))
        self.assertEqual(ctx.session_id, 42)
        self.assertEqual(ctx.server_id, 7)
        self.assertEqual(ctx.workspace_root, "/srv/app")

    def test_raises_for_unknown_session(self):
        client = FakeWorkspaceClient()
        with self.assertRaises(KeyError):
            asyncio.run(context_from_session(client, 999))


class FindResumableSessionTests(_StatePatchMixin, unittest.TestCase):
    def test_excludes_given_session_id(self):
        client = FakeWorkspaceClient(sessions=[{"id": 1, "status": "idle"}, {"id": 2, "status": "idle"}])
        result = asyncio.run(find_resumable_session(client, exclude_session_id=1))
        self.assertEqual(result["id"], 2)

    def test_returns_none_when_only_the_excluded_session_exists(self):
        client = FakeWorkspaceClient(sessions=[{"id": 1, "status": "idle"}])
        result = asyncio.run(find_resumable_session(client, exclude_session_id=1))
        self.assertIsNone(result)

    def test_skips_closed_sessions(self):
        client = FakeWorkspaceClient(sessions=[{"id": 1, "status": "closed"}, {"id": 2, "status": "active"}])
        result = asyncio.run(find_resumable_session(client))
        self.assertEqual(result["id"], 2)


class LocalDiscoveryCacheTests(unittest.TestCase):
    def test_disposable_untracked_cache_does_not_block_execute(self):
        self.assertEqual(
            blocking_dirty_files(["?? __pycache__/", "?? .pytest_cache/"]),
            [],
        )

    def test_tracked_or_unknown_untracked_changes_still_block_execute(self):
        lines = [" M calculator.py", "?? new_feature.py", "?? build/output.txt"]
        self.assertEqual(blocking_dirty_files(lines), lines)

    def test_unchanged_repository_reuses_cached_file_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("# Example\n")
            original_dir, original_path = state_module.CONFIG_DIR, state_module.STATE_PATH
            state_module.CONFIG_DIR, state_module.STATE_PATH = root / ".config", root / ".config" / "state.json"
            try:
                with patch("tamfis_code.workspace._git") as git, patch(
                    "tamfis_code.workspace._indexable_files", return_value=[root / "README.md"]
                ) as index:
                    git.side_effect = lambda _root, *args: {
                        ("rev-parse", "--show-toplevel"): str(root),
                        ("branch", "--show-current"): "main",
                        ("rev-parse", "HEAD"): "abc123",
                        ("status", "--short"): "",
                    }.get(args, "")
                    first = discover_local_repository(321, root)
                    second = discover_local_repository(321, root)
                self.assertEqual(first, second)
                self.assertEqual(index.call_count, 1)
                self.assertEqual(first["instruction_files"], [str(root / "README.md")])
            finally:
                state_module.CONFIG_DIR, state_module.STATE_PATH = original_dir, original_path


class ResolveLocalWorkspaceTests(unittest.TestCase):
    """resolve_local_workspace has zero RemoteAPIClient/network involvement --
    unlike ResolveWorkspaceTests above, no FakeWorkspaceClient is needed at
    all, which is itself the point: this is the Phase 1 decoupling proof."""

    def setUp(self):
        self._originals = (state_module.CONFIG_DIR, state_module.STATE_PATH)
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        state_module.CONFIG_DIR = base / ".config"
        state_module.STATE_PATH = base / ".config" / "state.json"

    def tearDown(self):
        state_module.CONFIG_DIR, state_module.STATE_PATH = self._originals
        self.tmp.cleanup()

    def test_allocates_a_local_session_id_with_no_server_id(self):
        with tempfile.TemporaryDirectory() as proj:
            ctx = resolve_local_workspace(cwd=Path(proj), discover=False)
        self.assertIsInstance(ctx.session_id, int)
        self.assertIsNone(ctx.server_id)
        self.assertEqual(ctx.workspace_root, str(Path(proj).resolve()))

    def test_reuses_session_for_same_workspace_root(self):
        with tempfile.TemporaryDirectory() as proj:
            first = resolve_local_workspace(cwd=Path(proj), discover=False)
            second = resolve_local_workspace(cwd=Path(proj), discover=False)
        self.assertEqual(first.session_id, second.session_id)

    def test_different_workspace_roots_get_different_sessions(self):
        with tempfile.TemporaryDirectory() as proj_a, tempfile.TemporaryDirectory() as proj_b:
            first = resolve_local_workspace(cwd=Path(proj_a), discover=False)
            second = resolve_local_workspace(cwd=Path(proj_b), discover=False)
        self.assertNotEqual(first.session_id, second.session_id)

    def test_session_ids_increment_from_existing_known_sessions(self):
        state_module.save_session_state(5, workspace_root="/some/other/preexisting/session")
        with tempfile.TemporaryDirectory() as proj:
            ctx = resolve_local_workspace(cwd=Path(proj), discover=False)
        self.assertEqual(ctx.session_id, 6)


class ResolveSwarmSubtaskWorkspaceTests(unittest.TestCase):
    """Unlike resolve_local_workspace, this must NEVER reuse an existing
    session for the same workspace_root -- concurrent swarm sub-tasks
    sharing one session_id would race on state.json's single-value fields
    (current_phase/running_action/active_task/...), which aren't
    merge-safe."""

    def setUp(self):
        self._originals = (state_module.CONFIG_DIR, state_module.STATE_PATH)
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        state_module.CONFIG_DIR = base / ".config"
        state_module.STATE_PATH = base / ".config" / "state.json"

    def tearDown(self):
        state_module.CONFIG_DIR, state_module.STATE_PATH = self._originals
        self.tmp.cleanup()

    def test_each_call_gets_a_distinct_session_even_for_the_same_workspace_root(self):
        from tamfis_code.workspace import resolve_swarm_subtask_workspace

        with tempfile.TemporaryDirectory() as proj:
            first = resolve_swarm_subtask_workspace(Path(proj))
            second = resolve_swarm_subtask_workspace(Path(proj))
        self.assertNotEqual(first.session_id, second.session_id)
        self.assertEqual(first.workspace_root, second.workspace_root)

    def test_records_parent_session_id_and_label(self):
        from tamfis_code.workspace import resolve_swarm_subtask_workspace

        with tempfile.TemporaryDirectory() as proj:
            ctx = resolve_swarm_subtask_workspace(Path(proj), parent_session_id=1, label="fix the bug")
        sess_state = state_module.get_session_state(ctx.session_id)
        self.assertEqual(sess_state.parent_session_id, 1)
        self.assertEqual(sess_state.swarm_label, "fix the bug")

    def test_default_parent_session_id_is_none(self):
        from tamfis_code.workspace import resolve_swarm_subtask_workspace

        with tempfile.TemporaryDirectory() as proj:
            ctx = resolve_swarm_subtask_workspace(Path(proj))
        sess_state = state_module.get_session_state(ctx.session_id)
        self.assertIsNone(sess_state.parent_session_id)

    def test_is_swarm_child_is_always_true_regardless_of_parent_session_id(self):
        # Live-caught regression: is_swarm_child, not parent_session_id, is
        # the actual hide/show marker -- a child minted with NO known
        # parent (parent_session_id=None, e.g. agent-cmd delegate, a
        # one-shot CLI invocation) must still be tagged is_swarm_child=True.
        from tamfis_code.workspace import resolve_swarm_subtask_workspace

        with tempfile.TemporaryDirectory() as proj:
            with_parent = resolve_swarm_subtask_workspace(Path(proj), parent_session_id=1)
            without_parent = resolve_swarm_subtask_workspace(Path(proj))
        self.assertTrue(state_module.get_session_state(with_parent.session_id).is_swarm_child)
        self.assertTrue(state_module.get_session_state(without_parent.session_id).is_swarm_child)
        self.assertIsNone(state_module.get_session_state(without_parent.session_id).parent_session_id)

    def test_does_not_disturb_an_ordinary_resolve_local_workspace_session(self):
        # A plain (non-swarm) session for the same workspace_root must keep
        # resolving to its own stable session_id, unaffected by any number
        # of swarm sub-task children minted afterward -- including ones
        # with no known parent_session_id at all.
        with tempfile.TemporaryDirectory() as proj:
            ordinary = resolve_local_workspace(cwd=Path(proj), discover=False)
            from tamfis_code.workspace import resolve_swarm_subtask_workspace
            resolve_swarm_subtask_workspace(Path(proj), parent_session_id=ordinary.session_id)
            resolve_swarm_subtask_workspace(Path(proj))  # no parent_session_id at all
            still_ordinary = resolve_local_workspace(cwd=Path(proj), discover=False)
        self.assertEqual(ordinary.session_id, still_ordinary.session_id)


class BuildSystemPromptTests(_StatePatchMixin, unittest.TestCase):
    def test_includes_workspace_root_and_non_git_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt = build_system_prompt(1, Path(tmp))
        self.assertIn(str(Path(tmp).resolve()), prompt)
        self.assertIn("not a Git repository", prompt)

    def test_includes_instruction_file_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "AGENTS.md").write_text("# Agent Instructions\nAlways run tests before committing.")
            prompt = build_system_prompt(1, root)
        self.assertIn("Agent Instructions", prompt)
        self.assertIn("Always run tests before committing.", prompt)
        self.assertIn("AGENTS.md", prompt)

    def test_includes_tamfis_md_instruction_file(self):
        # TAMFIS.md is this project's own documented instruction-file name
        # (see instructions.py's create_instruction_template default and
        # references.py's InstructionManager.INSTRUCTION_FILES) but was
        # missing from workspace.py's INSTRUCTION_NAMES -- the list that
        # actually feeds the live system prompt -- so a project using its
        # own tool's recommended convention was silently ignored.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "TAMFIS.md").write_text("# Project Rules\nUse tabs, not spaces.")
            prompt = build_system_prompt(1, root)
        self.assertIn("Project Rules", prompt)
        self.assertIn("Use tabs, not spaces.", prompt)
        self.assertIn("TAMFIS.md", prompt)

    def test_warns_against_repeating_identical_read_only_calls(self):
        # Regression: live-reported bug where a broad "investigate the
        # entire system for vulnerabilities" request made the model
        # re-issue an identical list_directory 3 times in a row (no other
        # action in between) until runner_local.py's loop guard killed the
        # whole task. The tool result itself was fine (a real, informative
        # 30-item listing) -- nothing steered the model toward acting on
        # what list_directory already returned instead of re-asking.
        with tempfile.TemporaryDirectory() as tmp:
            prompt = build_system_prompt(1, Path(tmp))
        self.assertIn("Never call list_directory", prompt)
        self.assertIn("stuck loop", prompt)

    def test_instruction_content_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "AGENTS.md").write_text("x" * 50_000)
            prompt = build_system_prompt(1, root)
        self.assertLess(len(prompt), 20_000)
        self.assertIn("(truncated)", prompt)

    def test_dirty_repo_is_flagged(self):
        with patch("tamfis_code.workspace._git") as git:
            git.side_effect = lambda _root, *args: {
                ("rev-parse", "--show-toplevel"): "/tmp/fake-repo",
                ("branch", "--show-current"): "main",
                ("rev-parse", "HEAD"): "abc123",
                ("status", "--short"): " M app.py",
            }.get(args, "")
            with patch("tamfis_code.workspace._indexable_files", return_value=[]):
                prompt = build_system_prompt(1, Path("/tmp/fake-repo"))
        self.assertIn("uncommitted change", prompt)
        self.assertIn("branch: main", prompt)

    def test_dependency_and_hidden_cache_readmes_are_not_project_instructions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("# Real project")
            (root / "node_modules/pkg").mkdir(parents=True)
            (root / "node_modules/pkg/README.md").write_text("# Dependency instructions")
            (root / ".cache/model").mkdir(parents=True)
            (root / ".cache/model/README.md").write_text("# Cached model instructions")
            with patch("tamfis_code.workspace._git", return_value=""):
                files = _indexable_files(root)
        self.assertIn(root / "README.md", files)
        self.assertNotIn(root / "node_modules/pkg/README.md", files)
        self.assertNotIn(root / ".cache/model/README.md", files)

    def test_nested_repository_directory_from_parent_git_is_expanded_safely(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            child = root / "backend"
            child.mkdir()
            service = child / "backend-capacity.conf"
            service.write_text("ExecStart=gunicorn app:api --bind 0.0.0.0:9500\n")
            (child / ".git").mkdir()
            (child / ".git/config").write_text("secret index metadata")
            (child / "node_modules/pkg").mkdir(parents=True)
            (child / "node_modules/pkg/index.js").write_text("dependency")

            # Parent Git repositories represent an untracked nested repo as
            # one directory line rather than returning its individual files.
            with patch("tamfis_code.workspace._git", return_value="backend/"):
                files = _indexable_files(root)

        self.assertIn(service, files)
        self.assertFalse(any(".git" in path.parts for path in files))
        self.assertFalse(any("node_modules" in path.parts for path in files))

    def test_service_port_facts_distinguish_application_binds_from_caddy_proxy_ports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = root / "tamfis-gpt-capacity.conf"
            service.write_text(
                "ExecStart=/usr/local/bin/gunicorn tier_ii_gateway.main:app "
                "--bind 0.0.0.0:9500 --workers 4\n"
            )
            caddy = root / "Caddyfile"
            caddy.write_text(":8080 {\n  reverse_proxy 127.0.0.1:9500\n}\n")
            prompt = build_system_prompt(1, root, force_discovery=True)
        self.assertIn("application_process_bind port 9500", prompt)
        self.assertIn("proxy_listener port 8080", prompt)
        self.assertIn("proxy_upstream port 9500", prompt)
        self.assertIn("proxy topology, not proof", prompt)


if __name__ == "__main__":
    unittest.main()
