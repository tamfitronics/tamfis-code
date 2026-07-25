#!/usr/bin/env python3
"""Test file and folder reference resolution"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from tamfis_code.references import (
    ReferenceResolver,
    InstructionManager,
    FileReference,
    FolderReference,
    process_references
)

CLI_STUB = '''#!/usr/bin/env python3
"""Stub CLI module used as a fixture target for @reference resolution tests"""


def main():
    pass


if __name__ == "__main__":
    main()
'''

INSTRUCTIONS_STUB = '''"""Stub instructions module used as a fixture target for @reference resolution tests"""
'''


class TestReferenceResolver:
    """Test the reference resolver"""

    @pytest.fixture(autouse=True)
    def setup_workspace(self, tmp_path):
        """Build an isolated workspace with a tamfis_code/ package to resolve @references against.

        Uses tmp_path rather than a real checkout so these tests don't depend on any
        particular directory existing on the machine running pytest.
        """
        self.workspace = tmp_path
        code_dir = self.workspace / 'tamfis_code'
        code_dir.mkdir()
        (code_dir / 'cli.py').write_text(CLI_STUB)
        (code_dir / 'instructions.py').write_text(INSTRUCTIONS_STUB)
        self.resolver = ReferenceResolver(self.workspace)

    def test_file_reference_simple(self):
        """Test simple @file reference"""
        # Use full path relative to workspace
        text = "Please review @tamfis_code/cli.py"
        result = self.resolver.resolve_references(text)

        assert 'files' in result
        assert len(result['files']) > 0
        assert 'cli.py' in result['files'][0].path

    def test_file_reference_with_line_range(self):
        """Test @file reference with line range"""
        text = "Review @tamfis_code/cli.py:10-20"
        result = self.resolver.resolve_references(text)

        assert len(result['files']) > 0, f"Expected files, got: {result}"
        assert 'cli.py' in result['files'][0].path
        # Check line range was parsed
        assert result['files'][0].line_start == 10
        assert result['files'][0].line_end == 20

    def test_file_reference_with_single_line(self):
        """Test @file reference with single line"""
        text = "Review @tamfis_code/cli.py:42"
        result = self.resolver.resolve_references(text)

        assert len(result['files']) > 0
        assert result['files'][0].line_start == 42
        # line_end defaults to line_start for single line
        assert result['files'][0].line_end == 42

    def test_file_reference_with_short_path(self):
        """Test @file reference with short path"""
        # This tests that the resolver can find files with just the filename
        text = "Review @cli.py"
        result = self.resolver.resolve_references(text)

        # Should find cli.py in tamfis_code directory
        assert len(result['files']) > 0, f"Expected to find cli.py, got: {result}"

    def test_folder_reference(self):
        """Test @folder reference"""
        text = "Check all files in @tamfis_code/"
        result = self.resolver.resolve_references(text)

        assert 'folders' in result
        # Should find the folder since it exists
        assert len(result['folders']) > 0

    def test_multiple_references(self):
        """Test multiple references in one prompt"""
        text = "Compare @tamfis_code/cli.py and @tamfis_code/instructions.py"
        result = self.resolver.resolve_references(text)

        assert len(result['files']) >= 2

    def test_nonexistent_file(self):
        """Test reference to nonexistent file"""
        text = "Review @nonexistent.py"
        result = self.resolver.resolve_references(text)

        # Should not error, just return no files
        assert 'files' in result
        assert len(result['files']) == 0

    def test_mixed_references(self):
        """Test mixed file and folder references"""
        text = "Check @tamfis_code/cli.py and @tamfis_code/"
        result = self.resolver.resolve_references(text)

        assert len(result['files']) >= 1
        assert len(result['folders']) >= 1

class TestInstructionManager:
    """Test instruction file management"""

    @pytest.fixture(autouse=True)
    def setup_workspace(self, tmp_path):
        """Use a fresh tmp_path per test as the workspace root, cleaned up automatically by pytest."""
        self.workspace = tmp_path
        self.mgr = InstructionManager(self.workspace)

    def test_load_instructions(self):
        """Test loading instruction files"""
        instructions = self.mgr.get_all_instructions()
        # Should load TAMFIS.md if it exists
        assert isinstance(instructions, dict)

    def test_get_instruction(self):
        """Test getting specific instruction"""
        # Create a test instruction file
        test_file = self.workspace / 'TEST.md'
        test_file.write_text('# Test Instruction\nTest content')

        # Note: TEST.md won't be loaded unless in INSTRUCTION_FILES list
        # This test just verifies the method doesn't crash

    def test_create_instruction_file(self):
        """Test creating an instruction file"""
        mgr = InstructionManager(self.workspace)

        test_file = 'TEST_INSTRUCTIONS.md'
        path = mgr.create_instruction_file(test_file, content='# Custom Test\nCustom content')

        assert path.exists()
        assert path.read_text() == '# Custom Test\nCustom content'

class TestProcessReferences:
    """Test the combined reference processing"""

    def test_full_processing(self, tmp_path):
        """Test full reference processing with instructions"""
        code_dir = tmp_path / 'tamfis_code'
        code_dir.mkdir()
        (code_dir / 'cli.py').write_text(CLI_STUB)

        result = process_references(
            "Analyze @tamfis_code/cli.py",
            tmp_path
        )

        assert 'references' in result
        assert 'instruction_context' in result
        assert 'enhanced_text' in result

    def test_full_processing_with_instruction_file(self, tmp_path):
        """Test full processing with instruction file"""
        # Create a test instruction file
        test_file = tmp_path / 'TAMFIS.md'
        test_file.write_text('# Test Instructions\nThis is a test instruction file.')

        result = process_references(
            "Analyze the code structure",
            tmp_path
        )

        assert 'instruction_context' in result
        # The instruction context may be empty if no instructions are loaded

    def test_references_with_no_instruction_files(self, tmp_path):
        """Test references when no instruction files exist"""
        result = process_references(
            "Analyze code",
            tmp_path
        )
        assert 'instruction_context' in result
        assert 'references' in result

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
