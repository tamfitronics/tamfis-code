import tempfile
import unittest
from pathlib import Path

from tamfis_code.instructions import (
    create_instruction_template,
    get_instruction_context,
    parse_instruction_file,
    resolve_file_references,
)


class ParseInstructionFileTests(unittest.TestCase):
    def test_missing_file_returns_empty_dict(self):
        self.assertEqual(parse_instruction_file("/nonexistent/TAMFIS.md"), {})

    def test_parses_markdown_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "TAMFIS.md"
            path.write_text("## Overview\nThis is a project.\n\n## Standards\nUse PEP 8.\n")
            sections = parse_instruction_file(path)
            self.assertEqual(sections["Overview"], "This is a project.")
            self.assertEqual(sections["Standards"], "Use PEP 8.")

    def test_content_before_first_heading_is_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "TAMFIS.md"
            path.write_text("preamble text\n## Section\nbody\n")
            sections = parse_instruction_file(path)
            self.assertEqual(list(sections.keys()), ["Section"])

    def test_no_headings_returns_empty_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "TAMFIS.md"
            path.write_text("just plain text, no headings\n")
            self.assertEqual(parse_instruction_file(path), {})


class CreateInstructionTemplateTests(unittest.TestCase):
    def test_writes_template_and_returns_its_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "TAMFIS.md"
            result = create_instruction_template(path)
            self.assertTrue(path.exists())
            self.assertEqual(path.read_text(encoding="utf-8"), result)
            self.assertIn("# TAMFIS-CODE Instructions", result)

    def test_default_path_is_tamfis_md(self):
        with tempfile.TemporaryDirectory() as tmp:
            import os
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                create_instruction_template()
                self.assertTrue((Path(tmp) / "TAMFIS.md").exists())
            finally:
                os.chdir(cwd)


class BackwardCompatWrapperTests(unittest.TestCase):
    def test_get_instruction_context_combines_instruction_manager_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "AGENTS.md").write_text("# Agents\nFollow these rules.\n")
            context = get_instruction_context(root)
            self.assertIsInstance(context, str)

    def test_resolve_file_references_returns_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = resolve_file_references("no references here", tmp)
            self.assertIsInstance(result, dict)


if __name__ == "__main__":
    unittest.main()
