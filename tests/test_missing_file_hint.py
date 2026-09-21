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

    def test_a_different_session_did_not_write_it_so_it_gets_no_such_claim(self):
        target = self.ws / "other.txt"
        target.write_text("x\n")
        record_mutation(9, path=str(target), operation="create", original_content=None, new_content="x\n")
        target.unlink()
        self.assertNotIn("wrote that file earlier", self._read(self._server(10), "other.txt"))


if __name__ == "__main__":
    unittest.main()
