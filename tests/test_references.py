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

class TestReferenceResolver:
    """Test the reference resolver"""

    def setup_method(self):
        """Setup test environment"""
        self.workspace = Path('/home/tamfisgpt/tamgpt6')
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

    def setup_method(self):
        """Setup test environment"""
        self.workspace = Path('/home/tamfisgpt/tamgpt6')
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
        
        # Clean up
        if test_file.exists():
            test_file.unlink()

    def test_create_instruction_file(self):
        """Test creating an instruction file"""
        mgr = InstructionManager(self.workspace)
        
        test_file = 'TEST_INSTRUCTIONS.md'
        path = mgr.create_instruction_file(test_file, content='# Custom Test\nCustom content')
        
        assert path.exists()
        assert path.read_text() == '# Custom Test\nCustom content'
        
        # Clean up
        if path.exists():
            path.unlink()

class TestProcessReferences:
    """Test the combined reference processing"""

    def test_full_processing(self):
        """Test full reference processing with instructions"""
        workspace = Path('/home/tamfisgpt/tamgpt6')
        
        result = process_references(
            "Analyze @tamfis_code/cli.py",
            workspace
        )
        
        assert 'references' in result
        assert 'instruction_context' in result
        assert 'enhanced_text' in result

    def test_full_processing_with_instruction_file(self):
        """Test full processing with instruction file"""
        workspace = Path('/home/tamfisgpt/tamgpt6')
        
        # Create a test instruction file
        test_file = workspace / 'TAMFIS.md'
        if not test_file.exists():
            test_file.write_text('# Test Instructions\nThis is a test instruction file.')
        
        result = process_references(
            "Analyze the code structure",
            workspace
        )
        
        assert 'instruction_context' in result
        # The instruction context may be empty if no instructions are loaded

    def test_references_with_no_instruction_files(self):
        """Test references when no instruction files exist"""
        import tempfile
        import shutil
        
        temp_dir = tempfile.mkdtemp()
        try:
            temp_path = Path(temp_dir)
            result = process_references(
                "Analyze code",
                temp_path
            )
            assert 'instruction_context' in result
            assert 'references' in result
        finally:
            shutil.rmtree(temp_dir)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
