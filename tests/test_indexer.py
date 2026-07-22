#!/usr/bin/env python3
"""Test code indexing"""

import sys
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from tamfis_code.indexer import CodeIndexer, CodeSymbol, CodeFile

class TestCodeIndexer:
    """Test code indexer"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.index_path = Path(self.temp_dir) / 'index'
        self.indexer = CodeIndexer(Path(self.temp_dir), self.index_path)

    def teardown_method(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_index_python(self):
        """Test indexing Python files"""
        test_file = Path(self.temp_dir) / 'test.py'
        test_file.write_text('''
def hello():
    """Hello function"""
    return "Hello"

class MyClass:
    def method(self):
        pass
''')
        
        self.indexer.index([str(test_file)])
        assert len(self.indexer.files) == 1
        
        code_file = self.indexer.files[str(test_file)]
        assert code_file.language == 'python'
        assert len(code_file.symbols) >= 2  # hello, MyClass

    def test_index_js(self):
        """Test indexing JavaScript files"""
        test_file = Path(self.temp_dir) / 'test.js'
        test_file.write_text('''
function hello() {
    return "Hello";
}

class MyClass {
    method() {}
}
''')
        
        self.indexer.index([str(test_file)])
        assert len(self.indexer.files) == 1
        
        code_file = self.indexer.files[str(test_file)]
        assert code_file.language == 'javascript'

    def test_search_symbol(self):
        """Test searching for symbols"""
        test_file = Path(self.temp_dir) / 'search_test.py'
        test_file.write_text('''
def hello_world():
    pass

def goodbye():
    pass
''')
        
        self.indexer.index([str(test_file)])
        results = self.indexer.search_symbol('hello')
        assert len(results) >= 1
        assert results[0].name == 'hello_world'

    def test_save_load_index(self):
        """Test saving and loading index"""
        test_file = Path(self.temp_dir) / 'test.py'
        test_file.write_text('def test_func(): pass')
        
        self.indexer.index([str(test_file)])
        self.indexer._save_index()
        
        # Create new indexer and load
        new_indexer = CodeIndexer(Path(self.temp_dir), self.index_path)
        new_indexer.load_index()
        
        assert len(new_indexer.files) == 1

    def test_unchanged_file_is_not_reparsed_on_a_second_index_call(self):
        # Before this, index() always did a full re-parse of every matching
        # file regardless of what actually changed -- `force` was accepted
        # but never read anywhere in the method.
        test_file = Path(self.temp_dir) / 'unchanged.py'
        test_file.write_text('def hello(): pass')

        self.indexer.index([str(test_file)])
        original_entry = self.indexer.files[str(test_file)]

        with patch.object(CodeIndexer, '_parse_python', wraps=self.indexer._parse_python) as spy:
            count = self.indexer.index([str(test_file)])

        spy.assert_not_called()
        assert count == 0
        assert self.indexer.last_skipped_count == 1
        assert self.indexer.files[str(test_file)] is original_entry

    def test_modified_file_is_reparsed(self):
        test_file = Path(self.temp_dir) / 'changes.py'
        test_file.write_text('def old_name(): pass')
        self.indexer.index([str(test_file)])

        # Force a distinct mtime -- some filesystems have coarse mtime
        # resolution, so back-date the original write instead of relying on
        # real wall-clock time passing between the two writes.
        os.utime(test_file, (time.time() - 5, time.time() - 5))
        self.indexer.files[str(test_file)].mtime_ns -= 10_000_000_000
        test_file.write_text('def new_name(): pass')

        count = self.indexer.index([str(test_file)])
        assert count == 1
        names = {s.name for s in self.indexer.files[str(test_file)].symbols}
        assert 'new_name' in names

    def test_force_reparses_even_when_unchanged(self):
        test_file = Path(self.temp_dir) / 'forced.py'
        test_file.write_text('def hello(): pass')
        self.indexer.index([str(test_file)])

        with patch.object(CodeIndexer, '_parse_python', wraps=self.indexer._parse_python) as spy:
            count = self.indexer.index([str(test_file)], force=True)

        spy.assert_called_once()
        assert count == 1

    def test_a_fresh_indexer_instance_loads_the_prior_index_before_incrementally_indexing(self):
        # Simulates a new CLI invocation (a new CodeIndexer object) against
        # a directory already indexed by a prior run -- incrementality must
        # hold across process/instance boundaries, not just within one.
        test_file = Path(self.temp_dir) / 'persisted.py'
        test_file.write_text('def hello(): pass')
        self.indexer.index([str(test_file)])

        fresh = CodeIndexer(Path(self.temp_dir), self.index_path)
        with patch.object(CodeIndexer, '_parse_python', wraps=fresh._parse_python) as spy:
            count = fresh.index([str(test_file)])

        spy.assert_not_called()
        assert count == 0
        assert len(fresh.files) == 1

    def test_deleted_file_is_pruned_from_a_whole_root_reindex(self):
        keep = Path(self.temp_dir) / 'keep.py'
        gone = Path(self.temp_dir) / 'gone.py'
        keep.write_text('def keep_fn(): pass')
        gone.write_text('def gone_fn(): pass')
        self.indexer.index()  # whole-root scan (no explicit paths)
        assert len(self.indexer.files) == 2

        gone.unlink()
        self.indexer.index()
        assert set(self.indexer.files.keys()) == {str(keep)}

    def test_get_stats(self):
        """Test getting index statistics"""
        test_file = Path(self.temp_dir) / 'stats_test.py'
        test_file.write_text('def func1(): pass\ndef func2(): pass')
        
        self.indexer.index([str(test_file)])
        stats = self.indexer.get_stats()
        
        assert stats['files'] >= 1
        assert stats['total_symbols'] >= 2

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
