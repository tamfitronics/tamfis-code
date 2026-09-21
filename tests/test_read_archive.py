"""read_archive: list/read ZIP and TAR members with no size limit and no extraction, nested to any depth.

Owner report 2026-09-21: an audit hit `read_file` "looks like a binary file" on three big .zip files and
could go no further (extract_archive is a mutating tool with a 250 MB / 5,000-file ceiling and is not
offered in read-only turns).
"""
import asyncio
import io
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from tamfis_code.archive_reader import read_archive, split_chain
from tamfis_code.mcp import MCPServer
from tamfis_code.safety import READ_ONLY_TOOLS
from tamfis_code.tool_policy import READ_TOOLS


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def _tgz_bytes(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


class ReadArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        # zip -> tar.gz -> zip -> text: three levels deep
        level3 = _zip_bytes({"deep/answer.txt": b"the answer\nis 42\n"})
        level2 = _tgz_bytes({"data/level3.zip": level3, "data/notes.txt": b"tar note\n"})
        big_lines = "".join(f"row {i}\n" for i in range(5000)).encode()
        self.pack = self.root / "pack.zip"
        self.pack.write_bytes(_zip_bytes({
            "readme.txt": b"hello\nworld\n",
            "inner/level2.tar.gz": level2,
            "big.log": big_lines,
            "blob.bin": b"\x00\x01\x02binary",
        }))

    def tearDown(self):
        self.tmp.cleanup()

    def test_lists_members_and_flags_nested_archives(self):
        out = read_archive(self.pack, [])
        self.assertIn("4 files", out)
        self.assertIn("readme.txt", out)
        self.assertIn('open it with path="pack.zip!/inner/level2.tar.gz"', out)

    def test_reads_a_text_member_with_line_numbers(self):
        out = read_archive(self.pack, [], "readme.txt")
        self.assertIn("1: hello", out)
        self.assertIn("2: world", out)
        self.assertIn("End of file", out)

    def test_pages_a_large_member_with_a_continuation_offset(self):
        first = read_archive(self.pack, [], "big.log", limit=100)
        self.assertIn("lines 1-100", first)
        self.assertIn("Continue with offset=101", first)
        second = read_archive(self.pack, [], "big.log", offset=101, limit=100)
        self.assertIn("101: row 100", second)

    def test_reads_through_three_nested_levels(self):
        out = read_archive(self.pack, ["inner/level2.tar.gz", "data/level3.zip"], "deep/answer.txt")
        self.assertIn("1: the answer", out)
        self.assertIn("2: is 42", out)

    def test_lists_a_nested_archive(self):
        out = read_archive(self.pack, ["inner/level2.tar.gz"])
        self.assertIn("data/notes.txt", out)
        self.assertIn("level3.zip", out)

    def test_pattern_filters_the_listing(self):
        out = read_archive(self.pack, [], pattern="*.log")
        self.assertIn("1 files matching", out)
        self.assertNotIn("readme.txt", out)

    def test_a_gzipped_text_member_is_read_decompressed(self):
        import gzip

        z = self.root / "gz.zip"
        z.write_bytes(_zip_bytes({"data.jsonl.gz": gzip.compress(b'{"a": 1}\n{"a": 2}\n')}))
        out = read_archive(z, [], "data.jsonl.gz")
        self.assertIn('2: {"a": 2}', out)

    def test_one_enormous_line_is_clipped_not_lost(self):
        z = self.root / "long.zip"
        z.write_bytes(_zip_bytes({"min.js": b"x" * 5_000_000 + b"\nsecond\n"}))
        out = read_archive(z, [], "min.js")
        self.assertIn("line continues", out)
        self.assertIn("2: second", out)
        self.assertLess(len(out), 60_000)

    def test_binary_members_are_reported_not_dumped(self):
        self.assertIn("binary file", read_archive(self.pack, [], "blob.bin"))

    def test_naming_an_archive_as_a_member_explains_how_to_open_it(self):
        self.assertIn('path="pack.zip!/inner/level2.tar.gz"', read_archive(self.pack, [], "inner/level2.tar.gz"))

    def test_missing_member_and_non_archive_levels_are_clear_errors(self):
        self.assertIn("is not a file in", read_archive(self.pack, [], "nope.txt"))
        self.assertIn("is not a ZIP/TAR archive", read_archive(self.pack, ["readme.txt"]))
        self.assertIn("is not a file inside", read_archive(self.pack, ["ghost.zip"]))

    def test_a_corrupt_archive_is_an_error_string_not_a_crash(self):
        bad = self.root / "bad.zip"
        bad.write_bytes(b"not a zip at all")
        self.assertTrue(read_archive(bad, []).startswith("Error:"))

    def test_nothing_is_written_next_to_the_archive(self):
        before = sorted(p.name for p in self.root.iterdir())
        read_archive(self.pack, ["inner/level2.tar.gz", "data/level3.zip"], "deep/answer.txt")
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), before)

    def test_a_traversal_member_name_is_only_a_lookup_key(self):
        evil = self.root / "evil.zip"
        evil.write_bytes(_zip_bytes({"../../escape.txt": b"x\n"}))
        self.assertIn("1: x", read_archive(evil, [], "../../escape.txt"))
        self.assertFalse((self.root.parent / "escape.txt").exists())


class ToolWiringTests(unittest.TestCase):
    def test_split_chain(self):
        self.assertEqual(split_chain("a.zip!/b/c.tgz!/d.zip"), ["a.zip", "b/c.tgz", "d.zip"])

    def test_the_tool_is_read_only_and_offered_in_read_only_turns(self):
        self.assertIn("read_archive", READ_ONLY_TOOLS)
        self.assertIn("read_archive", READ_TOOLS)

    def test_the_mcp_tool_reads_a_nested_archive_inside_the_workspace(self):
        with tempfile.TemporaryDirectory() as ws:
            inner = _zip_bytes({"a.txt": b"nested ok\n"})
            (Path(ws) / "outer.zip").write_bytes(_zip_bytes({"in.zip": inner}))
            server = MCPServer(workspace_root=ws, session_id=None)
            out = asyncio.run(server._read_archive("outer.zip!/in.zip", member="a.txt"))
        self.assertIn("1: nested ok", out)

    def test_read_file_on_a_zip_points_at_read_archive(self):
        with tempfile.TemporaryDirectory() as ws:
            (Path(ws) / "x.zip").write_bytes(_zip_bytes({"a": b"b"}))
            server = MCPServer(workspace_root=ws, session_id=None)
            out = asyncio.run(server._read_file("x.zip"))
        self.assertIn("read_archive", out)


if __name__ == "__main__":
    unittest.main()
