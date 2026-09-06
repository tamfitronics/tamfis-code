import tempfile
import unittest
from pathlib import Path

from tamfis_code.references import InstructionManager


class InstructionFileDiscoveryTests(unittest.TestCase):
    def test_agents_md_is_discovered_alongside_tamfis_md(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "AGENTS.md").write_text("Follow AGENTS.md conventions.\n")
            mgr = InstructionManager(root)
            self.assertIn("AGENTS.md", mgr.instructions)
            self.assertIn("Follow AGENTS.md conventions.", mgr.get_combined_instructions())

    def test_both_tamfis_md_and_agents_md_are_combined_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "TAMFIS.md").write_text("TAMFIS-specific rule.\n")
            (root / "AGENTS.md").write_text("AGENTS-specific rule.\n")
            mgr = InstructionManager(root)
            combined = mgr.get_combined_instructions()
            self.assertIn("TAMFIS-specific rule.", combined)
            self.assertIn("AGENTS-specific rule.", combined)


class InstructionImportTests(unittest.TestCase):
    def test_import_line_is_inlined(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "style.md").write_text("Use 2-space indentation.\n")
            (root / "TAMFIS.md").write_text("# Project rules\n@import style.md\n")
            mgr = InstructionManager(root)
            content = mgr.instructions["TAMFIS.md"]
            self.assertIn("Use 2-space indentation.", content)
            self.assertNotIn("@import", content)

    def test_import_resolves_relative_to_subdirectory_for_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sub = root / "backend"
            sub.mkdir()
            (sub / "override.md").write_text("Backend-specific override.\n")
            (sub / "TAMFIS.md").write_text("@import override.md\n")
            mgr = InstructionManager(sub)
            content = mgr.instructions["TAMFIS.md"]
            self.assertIn("Backend-specific override.", content)

    def test_import_falls_back_to_workspace_root(self):
        # .tamfis/TAMFIS.md's own directory is workspace_root/.tamfis, not
        # workspace_root -- an @import there for a file that actually lives
        # at the true workspace root must still resolve, not only sibling
        # files inside .tamfis/.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "shared.md").write_text("Shared root snippet.\n")
            tamfis_dir = root / ".tamfis"
            tamfis_dir.mkdir()
            (tamfis_dir / "TAMFIS.md").write_text("@import shared.md\n")
            mgr = InstructionManager(root)
            content = mgr.instructions[".tamfis/TAMFIS.md"]
            self.assertIn("Shared root snippet.", content)

    def test_missing_import_target_degrades_to_a_comment_not_a_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "TAMFIS.md").write_text("@import does-not-exist.md\n")
            mgr = InstructionManager(root)  # must not raise
            content = mgr.instructions["TAMFIS.md"]
            self.assertIn("file not found", content)

    def test_circular_import_does_not_infinite_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.md").write_text("A content\n@import b.md\n")
            (root / "b.md").write_text("B content\n@import a.md\n")
            (root / "TAMFIS.md").write_text("@import a.md\n")
            mgr = InstructionManager(root)  # must terminate
            content = mgr.instructions["TAMFIS.md"]
            self.assertIn("A content", content)
            self.assertIn("B content", content)
            self.assertIn("circular import", content)

    def test_mid_sentence_at_import_is_not_treated_as_a_directive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "TAMFIS.md").write_text("Send questions to @import-help on Slack.\n")
            mgr = InstructionManager(root)
            content = mgr.instructions["TAMFIS.md"]
            self.assertIn("Send questions to @import-help on Slack.", content)


if __name__ == "__main__":
    unittest.main()
