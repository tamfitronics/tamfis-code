"""A file this session wrote and that was later deleted must not send the model into a search loop.

Live report 2026-09-21: after /home/tamgpt/check.txt was deleted, the session kept reading it, listing the
directory and reading it again, because the error only said "use list_directory or search_code to find
the right path".
"""
import asyncio
import tempfile
import unittest
from pathlib import Path

from tamfis_code import state as state_module
from tamfis_code.mcp import MCPServer
from tamfis_code.safety import record_mutation


class MissingFileHintTests(unittest.TestCase):
    def setUp(self):
        self._orig = (state_module.CONFIG_DIR, state_module.STATE_PATH)
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        state_module.CONFIG_DIR = base / ".config"
        state_module.STATE_PATH = base / ".config" / "state.json"
        self.ws = base / "ws"
        self.ws.mkdir()

    def tearDown(self):
        state_module.CONFIG_DIR, state_module.STATE_PATH = self._orig
        self.tmp.cleanup()

    def _server(self, session_id):
        server = MCPServer(workspace_root=str(self.ws), session_id=session_id)
        return server

    def _read(self, server, name):
        return asyncio.run(server._read_file(str(self.ws / name)))

    def test_a_file_written_earlier_in_this_session_gets_the_stop_searching_hint(self):
        target = self.ws / "check.txt"
        target.write_text("x\n")
        record_mutation(7, path=str(target), operation="create", original_content=None, new_content="x\n")
        target.unlink()
        message = self._read(self._server(7), "check.txt")
        self.assertIn("not found", message)
        self.assertIn("wrote that file earlier", message)
        self.assertIn("Do not keep searching", message)
        self.assertNotIn("find the right path", message)

    def test_an_ordinary_missing_file_keeps_the_generic_hint(self):
        message = self._read(self._server(8), "never-existed.txt")
        self.assertIn("find the right path", message)
        self.assertNotIn("wrote that file earlier", message)

    def test_missing_nested_path_recovers_a_unique_tree_match(self):
        target = self.ws / "src" / "deep" / "module.py"
        target.parent.mkdir(parents=True)
        target.write_text("answer = 42\n")
        message = self._read(self._server(8), "src/module.py")
        self.assertIn("Resolved requested path", message)
        self.assertIn("answer = 42", message)

    def test_missing_path_with_unique_basename_recovers_deep_file(self):
        target = self.ws / "packages" / "feature" / "handler.py"
        target.parent.mkdir(parents=True)
        target.write_text("def handle():\n    return True\n")
        message = self._read(self._server(8), "handler.py")
        self.assertIn("Resolved requested path", message)
        self.assertIn("def handle", message)

    def test_complete_tree_search_reaches_deep_file_before_typo_hint(self):
        # Reproduce the reported failure: a shallow tamgpt_init.py must not
        # hide the canonical tamgpt_api.py several levels down.
        (self.ws / "tamgpt_init.py").write_text("WRONG = True\n")
        target = self.ws
        for index in range(16):
            target /= f"layer_{index}"
        target.mkdir(parents=True)
        (target / "tamgpt_api.py").write_text("CANONICAL = True\n")

        message = self._read(self._server(8), "tamgpt_api.py")
        self.assertIn("Resolved requested path", message)
        self.assertIn("CANONICAL = True", message)
        self.assertNotIn("Did you mean", message)

    def test_live_source_wins_over_archival_duplicate(self):
        live = self.ws / "tier_iv_orchestration" / "tamgpt_api.py"
        backup = self.ws / "backups" / "old" / "tier_iv_orchestration" / "tamgpt_api.py"
        live.parent.mkdir(parents=True)
        backup.parent.mkdir(parents=True)
        live.write_text("LIVE_SOURCE = True\n")
        backup.write_text("ARCHIVAL_COPY = True\n")

        message = self._read(self._server(8), "tamgpt_api.py")
        self.assertIn("LIVE_SOURCE = True", message)
        self.assertNotIn("ARCHIVAL_COPY = True", message)

    def test_missing_prefixed_status_name_recovers_canonical_status_sibling(self):
        target = self.ws / "training_queue_state" / "status"
        target.parent.mkdir(parents=True)
        target.write_text("complete\n")
        message = self._read(self._server(8), "training_queue_state/moe_pretraining.status")
        self.assertIn("Resolved requested path", message)
        self.assertIn("complete", message)

    def test_ambiguous_tree_matches_never_choose_arbitrarily(self):
        for package in ("one", "two"):
            target = self.ws / "packages" / package / "handler.py"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(package)
        message = self._read(self._server(8), "handler.py")
        self.assertIn("multiple possible files", message)
        self.assertIn("packages/one/handler.py", message)
        self.assertIn("packages/two/handler.py", message)
        self.assertNotIn("Resolved requested path", message)

    def test_recovery_does_not_escape_an_allowed_scope(self):
        outside = self.ws.parent / "outside.py"
        outside.write_text("secret = True\n")
        scoped = MCPServer(
            workspace_root=str(self.ws),
            allowed_workspace_roots=[str(self.ws)],
        )
        message = self._read(scoped, "outside.py")
        self.assertNotIn("secret = True", message)
        self.assertIn("not found", message)

    def test_a_different_session_did_not_write_it_so_it_gets_no_such_claim(self):
        target = self.ws / "other.txt"
        target.write_text("x\n")
        record_mutation(9, path=str(target), operation="create", original_content=None, new_content="x\n")
        target.unlink()
        self.assertNotIn("wrote that file earlier", self._read(self._server(10), "other.txt"))


if __name__ == "__main__":
    unittest.main()


class TypoCorrectionTests(unittest.TestCase):
    """Live report 2026-09-21: the model read /home/tmafisseo/.../caompgns/caompgns.php (the user's typos,
    verbatim) and could not recover -- the error only said to search for the right path."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "tamfisseo" / "www" / "wp-content" / "plugins" / "campaigns").mkdir(parents=True)
        (self.root / "tamfisseo" / "www" / "wp-content" / "plugins" / "campaigns" / "campaigns.php").write_text("<?php\n")
        (self.root / "tamfitronics").mkdir()
        self.server = MCPServer(workspace_root=str(self.root))

    def tearDown(self):
        self.tmp.cleanup()

    def test_every_misspelled_component_is_corrected(self):
        wrong = self.root / "tmafisseo" / "www" / "wp-content" / "plugins" / "caompgns" / "caompgns.php"
        right = self.root / "tamfisseo" / "www" / "wp-content" / "plugins" / "campaigns" / "campaigns.php"
        self.assertEqual(self.server._correct_path_typos(str(wrong)), str(right))
        message = asyncio.run(self.server._read_file(str(wrong)))
        self.assertIn("Did you mean", message)
        self.assertIn(str(right), message)

    def test_a_missing_directory_gets_the_suggestion_too(self):
        result = asyncio.run(self.server._list_directory(str(self.root / "tmafisseo" / "www")))
        self.assertIn(str(self.root / "tamfisseo" / "www"), result[0]["error"])

    def test_relative_paths_and_an_existing_path_behave(self):
        self.assertEqual(
            self.server._correct_path_typos("tamfitroincs"), str(self.root / "tamfitronics"),
        )
        self.assertIsNone(self.server._correct_path_typos("tamfisseo/www"))          # exists: nothing to fix

    def test_it_never_guesses_wildly(self):
        self.assertIsNone(self.server._correct_path_typos(str(self.root / "zzzzzz" / "file.txt")))
        self.assertIsNone(self.server._correct_path_typos(str(self.root / "tamfisseo" / "www" / "banana.txt")))
