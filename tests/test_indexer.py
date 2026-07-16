#!/usr/bin/env python3
"""Test code indexing"""

import sys
import os
import tempfile
from pathlib import Path

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
