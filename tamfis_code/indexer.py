"""Code indexing and semantic search for TAMFIS-CODE"""

import hashlib
import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field

@dataclass
class CodeSymbol:
    """A code symbol (function, class, variable)"""
    name: str
    kind: str
    file_path: str
    line_start: int
    line_end: int
    signature: Optional[str] = None
    docstring: Optional[str] = None
    parent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CodeFile:
    """Indexed code file"""
    path: str
    language: str
    symbols: List[CodeSymbol] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    size: int = 0
    lines: int = 0

class CodeIndexer:
    """Index and search code files"""
    
    SUPPORTED_LANGUAGES = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.jsx': 'javascript',
        '.go': 'go',
        '.rs': 'rust',
        '.c': 'c',
        '.cpp': 'cpp',
        '.h': 'c',
        '.hpp': 'cpp',
        '.java': 'java',
        '.rb': 'ruby',
        '.php': 'php',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.sh': 'shell',
        '.bash': 'shell',
        '.zsh': 'shell',
        '.fish': 'shell',
    }
    
    def __init__(self, root_path: Path, index_path: Optional[Path] = None):
        self.root_path = Path(root_path)
        # Workspace-scoped by default (hash of the resolved root) -- a
        # single shared ~/.tamfis/index/code_index.json used to silently
        # clobber across every project this CLI ever indexed, since no
        # caller passed an explicit index_path.
        default_index_path = Path.home() / '.tamfis' / 'index' / hashlib.sha256(str(self.root_path.resolve()).encode()).hexdigest()[:16]
        self.index_path = index_path or default_index_path
        self.files: Dict[str, CodeFile] = {}
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.ignore_patterns = [
            '*.pyc', '__pycache__', '.git', '.env', 'venv', 
            'node_modules', '.idea', '.vscode', '*.log', '*.tmp',
            '.DS_Store', 'Thumbs.db', '.pytest_cache', '.mypy_cache',
            '.coverage', 'htmlcov', '.tox', '.eggs', '*.egg-info'
        ]
    
    def index(self, paths: List[str] = None, force: bool = False):
        """Index files in the workspace"""
        if not paths:
            paths = [str(self.root_path)]
        
        indexed_count = 0
        for path_str in paths:
            path = Path(path_str)
            if path.is_file():
                if self._index_file(path):
                    indexed_count += 1
            elif path.is_dir():
                indexed_count += self._index_directory(path)
        
        self._save_index()
        return indexed_count
    
    def _index_directory(self, directory: Path) -> int:
        """Recursively index a directory"""
        count = 0
        # Check for .gitignore patterns
        ignore_patterns = self._load_gitignore(directory)
        
        for path in directory.rglob('*'):
            if path.is_file():
                # Skip ignored files
                if self._is_ignored(path, ignore_patterns):
                    continue
                ext = path.suffix.lower()
                if ext in self.SUPPORTED_LANGUAGES:
                    if self._index_file(path):
                        count += 1
        return count
    
    def _load_gitignore(self, directory: Path) -> List[str]:
        """Load .gitignore patterns"""
        gitignore = directory / '.gitignore'
        patterns = []
        if gitignore.exists():
            try:
                with open(gitignore, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            patterns.append(line)
            except Exception:
                pass
        return patterns
    
    def _is_ignored(self, path: Path, patterns: List[str]) -> bool:
        """Check if path matches ignore patterns"""
        path_str = str(path)
        # Check built-in patterns
        for pattern in self.ignore_patterns:
            if pattern in path_str or path_str.endswith(pattern):
                return True
        
        # Check .gitignore patterns
        for pattern in patterns:
            if pattern.endswith('/'):
                if pattern in path_str:
                    return True
            else:
                if pattern in path_str:
                    return True
        return False
    
    def _index_file(self, file_path: Path) -> bool:
        """Index a single file"""
        ext = file_path.suffix.lower()
        if ext not in self.SUPPORTED_LANGUAGES:
            return False
        
        language = self.SUPPORTED_LANGUAGES[ext]
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            return False
        
        # Parse symbols based on language
        if language == 'python':
            symbols = self._parse_python(content, file_path)
            imports = self._parse_python_imports(content)
        elif language in ('javascript', 'typescript'):
            symbols = self._parse_javascript(content, file_path)
            imports = self._parse_js_imports(content)
        elif language == 'go':
            symbols = self._parse_go(content, file_path)
            imports = self._parse_go_imports(content)
        else:
            symbols = self._parse_generic(content, file_path)
            imports = []
        
        code_file = CodeFile(
            path=str(file_path),
            language=language,
            symbols=symbols,
            imports=imports,
            size=file_path.stat().st_size,
            lines=len(content.split('\n'))
        )
        
        self.files[str(file_path)] = code_file
        return True
    
    def _parse_python(self, content: str, file_path: Path) -> List[CodeSymbol]:
        """Parse Python file for symbols"""
        symbols = []
        lines = content.split('\n')
        
        func_pattern = re.compile(r'^\s*def\s+(\w+)\s*\(([^)]*)\)')
        class_pattern = re.compile(r'^\s*class\s+(\w+)(?:\s*\(([^)]*)\))?')
        
        for i, line in enumerate(lines, 1):
            func_match = func_pattern.search(line)
            if func_match:
                symbols.append(CodeSymbol(
                    name=func_match.group(1),
                    kind='function',
                    file_path=str(file_path),
                    line_start=i,
                    line_end=self._find_function_end(lines, i),
                    signature=func_match.group(2),
                ))
                continue
            
            class_match = class_pattern.search(line)
            if class_match:
                symbols.append(CodeSymbol(
                    name=class_match.group(1),
                    kind='class',
                    file_path=str(file_path),
                    line_start=i,
                    line_end=self._find_class_end(lines, i),
                    signature=class_match.group(2) if class_match.group(2) else None,
                ))
                continue
        
        return symbols
    
    def _find_function_end(self, lines: List[str], start: int) -> int:
        """Find the end of a function definition"""
        if start >= len(lines):
            return start
        
        indent = len(lines[start-1]) - len(lines[start-1].lstrip())
        for i in range(start, len(lines)):
            if i == len(lines) - 1:
                return i + 1
            stripped = lines[i].strip()
            if stripped and not stripped.startswith('#'):
                current_indent = len(lines[i]) - len(lines[i].lstrip())
                if current_indent <= indent and stripped:
                    return i
        return start + 1
    
    def _find_class_end(self, lines: List[str], start: int) -> int:
        """Find the end of a class definition"""
        if start >= len(lines):
            return start
        
        indent = len(lines[start-1]) - len(lines[start-1].lstrip())
        for i in range(start, len(lines)):
            if i == len(lines) - 1:
                return i + 1
            stripped = lines[i].strip()
            if stripped and not stripped.startswith('#'):
                current_indent = len(lines[i]) - len(lines[i].lstrip())
                if current_indent <= indent and stripped and not lines[i].strip().startswith('def '):
                    return i
        return start + 1
    
    def _parse_python_imports(self, content: str) -> List[str]:
        """Parse Python imports"""
        imports = []
        for line in content.split('\n'):
            if line.startswith('import ') or line.startswith('from '):
                parts = line.split()
                if len(parts) > 1:
                    imports.append(parts[1].split('.')[0])
        return imports
    
    def _parse_javascript(self, content: str, file_path: Path) -> List[CodeSymbol]:
        """Parse JavaScript/TypeScript for symbols"""
        symbols = []
        lines = content.split('\n')
        
        func_pattern = re.compile(r'^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)')
        arrow_pattern = re.compile(r'^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>')
        class_pattern = re.compile(r'^\s*(?:export\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?')
        interface_pattern = re.compile(r'^\s*(?:export\s+)?interface\s+(\w+)')
        type_pattern = re.compile(r'^\s*(?:export\s+)?type\s+(\w+)')
        
        for i, line in enumerate(lines, 1):
            func_match = func_pattern.search(line)
            if func_match:
                symbols.append(CodeSymbol(
                    name=func_match.group(1),
                    kind='function',
                    file_path=str(file_path),
                    line_start=i,
                    line_end=i,
                    signature=func_match.group(2),
                ))
                continue
            
            arrow_match = arrow_pattern.search(line)
            if arrow_match:
                symbols.append(CodeSymbol(
                    name=arrow_match.group(1),
                    kind='function',
                    file_path=str(file_path),
                    line_start=i,
                    line_end=i,
                ))
                continue
            
            class_match = class_pattern.search(line)
            if class_match:
                symbols.append(CodeSymbol(
                    name=class_match.group(1),
                    kind='class',
                    file_path=str(file_path),
                    line_start=i,
                    line_end=self._find_js_class_end(lines, i),
                ))
                continue
            
            interface_match = interface_pattern.search(line)
            if interface_match:
                symbols.append(CodeSymbol(
                    name=interface_match.group(1),
                    kind='interface',
                    file_path=str(file_path),
                    line_start=i,
                    line_end=i,
                ))
                continue
            
            type_match = type_pattern.search(line)
            if type_match:
                symbols.append(CodeSymbol(
                    name=type_match.group(1),
                    kind='type',
                    file_path=str(file_path),
                    line_start=i,
                    line_end=i,
                ))
                continue
        
        return symbols
    
    def _find_js_class_end(self, lines: List[str], start: int) -> int:
        """Find the end of a JS class definition"""
        brace_count = 0
        for i in range(start-1, len(lines)):
            line = lines[i]
            brace_count += line.count('{') - line.count('}')
            if brace_count == 0 and i > start:
                return i + 1
        return len(lines)
    
    def _parse_js_imports(self, content: str) -> List[str]:
        """Parse JavaScript imports"""
        imports = []
        import_pattern = re.compile(r'import\s+(?:{[^}]*}\s+from\s+)?[\'"]([^\'"]+)[\'"]')
        require_pattern = re.compile(r'require\s*\(\s*[\'"]([^\'"]+)[\'"]')
        
        for line in content.split('\n'):
            for pattern in [import_pattern, require_pattern]:
                match = pattern.search(line)
                if match:
                    imports.append(match.group(1))
        return imports
    
    def _parse_go(self, content: str, file_path: Path) -> List[CodeSymbol]:
        """Parse Go file for symbols"""
        symbols = []
        lines = content.split('\n')
        
        func_pattern = re.compile(r'^\s*func\s+(\w+)\s*\(([^)]*)\)')
        type_pattern = re.compile(r'^\s*type\s+(\w+)\s+(?:struct|interface)')
        const_pattern = re.compile(r'^\s*const\s+(\w+)')
        var_pattern = re.compile(r'^\s*var\s+(\w+)')
        
        for i, line in enumerate(lines, 1):
            func_match = func_pattern.search(line)
            if func_match:
                symbols.append(CodeSymbol(
                    name=func_match.group(1),
                    kind='function',
                    file_path=str(file_path),
                    line_start=i,
                    line_end=i,
                ))
                continue
            
            type_match = type_pattern.search(line)
            if type_match:
                symbols.append(CodeSymbol(
                    name=type_match.group(1),
                    kind='type',
                    file_path=str(file_path),
                    line_start=i,
                    line_end=self._find_go_type_end(lines, i),
                ))
                continue
        
        return symbols
    
    def _find_go_type_end(self, lines: List[str], start: int) -> int:
        """Find the end of a Go type definition"""
        brace_count = 0
        for i in range(start-1, len(lines)):
            line = lines[i]
            brace_count += line.count('{') - line.count('}')
            if brace_count == 0 and i > start:
                return i + 1
        return len(lines)
    
    def _parse_go_imports(self, content: str) -> List[str]:
        """Parse Go imports"""
        imports = []
        import_pattern = re.compile(r'^\s*import\s+[\'"]?([^\'"\s]+)')
        import_block = re.compile(r'^\s*import\s*\(([^)]+)\)', re.DOTALL)
        
        # Simple import
        for line in content.split('\n'):
            match = import_pattern.search(line)
            if match:
                imports.append(match.group(1))
        
        # Import block
        block_match = import_block.search(content)
        if block_match:
            for line in block_match.group(1).split('\n'):
                if line.strip():
                    imports.append(line.strip().strip('"').strip("'"))
        
        return imports
    
    def _parse_generic(self, content: str, file_path: Path) -> List[CodeSymbol]:
        """Generic fallback parsing"""
        symbols = []
        lines = content.split('\n')
        
        patterns = [
            (r'^\s*function\s+(\w+)', 'function'),
            (r'^\s*class\s+(\w+)', 'class'),
            (r'^\s*def\s+(\w+)', 'function'),
            (r'^\s*(?:export\s+)?interface\s+(\w+)', 'interface'),
            (r'^\s*(?:export\s+)?type\s+(\w+)', 'type'),
            (r'^\s*(?:public|private|protected)?\s+(?:static\s+)?(\w+)\s*\(', 'method'),
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern, kind in patterns:
                match = re.search(pattern, line)
                if match:
                    symbols.append(CodeSymbol(
                        name=match.group(1),
                        kind=kind,
                        file_path=str(file_path),
                        line_start=i,
                        line_end=i,
                    ))
                    break
        
        return symbols
    
    def _save_index(self):
        """Save index to disk"""
        index_data = {}
        for path, code_file in self.files.items():
            index_data[path] = {
                'language': code_file.language,
                'size': code_file.size,
                'lines': code_file.lines,
                'imports': code_file.imports,
                'symbols': [
                    {
                        'name': s.name,
                        'kind': s.kind,
                        'line_start': s.line_start,
                        'line_end': s.line_end,
                        'signature': s.signature,
                        'docstring': s.docstring,
                    }
                    for s in code_file.symbols
                ]
            }
        
        index_file = self.index_path / 'code_index.json'
        with open(index_file, 'w') as f:
            json.dump(index_data, f, indent=2)
    
    def load_index(self):
        """Load index from disk"""
        index_file = self.index_path / 'code_index.json'
        if not index_file.exists():
            return
        
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        
        self.files = {}
        for path, data in index_data.items():
            symbols = [
                CodeSymbol(
                    name=s['name'],
                    kind=s['kind'],
                    file_path=path,
                    line_start=s.get('line_start', 0),
                    line_end=s.get('line_end', 0),
                    signature=s.get('signature'),
                    docstring=s.get('docstring'),
                )
                for s in data.get('symbols', [])
            ]
            self.files[path] = CodeFile(
                path=path,
                language=data.get('language', 'unknown'),
                symbols=symbols,
                imports=data.get('imports', []),
                size=data.get('size', 0),
                lines=data.get('lines', 0),
            )
    
    def search_symbol(self, query: str, kind: Optional[str] = None) -> List[CodeSymbol]:
        """Search for symbols by name"""
        results = []
        query_lower = query.lower()
        
        for file_path, code_file in self.files.items():
            for symbol in code_file.symbols:
                if query_lower in symbol.name.lower():
                    if kind is None or symbol.kind == kind:
                        results.append(symbol)
        
        return sorted(results, key=lambda x: x.file_path)
    
    def search_imports(self, module: str) -> List[CodeFile]:
        """Find files that import a module"""
        results = []
        module_lower = module.lower()
        
        for file_path, code_file in self.files.items():
            for imp in code_file.imports:
                if module_lower in imp.lower():
                    results.append(code_file)
                    break
        
        return results
    
    def get_file_summary(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get summary for a specific file"""
        if file_path not in self.files:
            return None
        
        code_file = self.files[file_path]
        return {
            'path': code_file.path,
            'language': code_file.language,
            'lines': code_file.lines,
            'size': code_file.size,
            'symbols_count': len(code_file.symbols),
            'imports_count': len(code_file.imports),
            'symbols': [
                {'name': s.name, 'kind': s.kind}
                for s in code_file.symbols
            ],
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        total_symbols = sum(len(f.symbols) for f in self.files.values())
        symbol_kinds = {}
        languages = {}
        
        for code_file in self.files.values():
            languages[code_file.language] = languages.get(code_file.language, 0) + 1
            for symbol in code_file.symbols:
                symbol_kinds[symbol.kind] = symbol_kinds.get(symbol.kind, 0) + 1
        
        return {
            'files': len(self.files),
            'total_symbols': total_symbols,
            'languages': languages,
            'symbol_kinds': symbol_kinds,
        }
