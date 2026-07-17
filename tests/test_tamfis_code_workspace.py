import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tamfis_code import state as state_module
from tamfis_code.workspace import (
    blocking_dirty_files, build_system_prompt, context_from_session, discover_local_repository,
    find_resumable_session, resolve_local_workspace, resolve_workspace,
)


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


if __name__ == "__main__":
    unittest.main()
