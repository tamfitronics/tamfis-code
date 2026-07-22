"""File and folder reference support for TAMFIS-CODE"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Union
from dataclasses import dataclass, field
import fnmatch

@dataclass
class FileReference:
    """A file referenced in a conversation"""
    path: str
    content: str
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    is_selected: bool = False
    
    def get_lines(self) -> List[str]:
        """Get specific lines if line range is specified"""
        if self.line_start is None:
            return self.content.split('\n')
        
        lines = self.content.split('\n')
        end = self.line_end or self.line_start  # If no end, use start (single line)
        return lines[self.line_start-1:end]
    
    def get_context(self, context_lines: int = 3) -> str:
        """Get content with context lines around selection"""
        if self.line_start is None:
            return self.content
        
        lines = self.content.split('\n')
        start = max(0, self.line_start - 1 - context_lines)
        end = min(len(lines), (self.line_end or self.line_start) + context_lines)
        
        result = []
        for i in range(start, end):
            if i >= self.line_start - 1 and i < (self.line_end or self.line_start):
                result.append(f"> {lines[i]}")
            else:
                result.append(f"  {lines[i]}")
        
        return '\n'.join(result)

@dataclass
class FolderReference:
    """A folder referenced in a conversation"""
    path: str
    files: List[FileReference] = field(default_factory=list)
    pattern: Optional[str] = None
    recursive: bool = True
    
    def get_content(self) -> str:
        """Get content of all files in the folder"""
        result = []
        for f in self.files:
            result.append(f"=== {f.path} ===\n{f.content}")
        return '\n\n'.join(result)

class ReferenceResolver:
    """Resolve @file and @folder references in prompts"""
    
    def __init__(self, workspace_root: Union[str, Path]):
        self.workspace_root = Path(workspace_root)
        self.references: Dict[str, Union[FileReference, FolderReference]] = {}
        
        # Common ignore patterns
        self.ignore_patterns = [
            '*.pyc', '__pycache__', '.git', '.env', 'venv', 
            'node_modules', '.idea', '.vscode', '*.log', '*.tmp',
            '.DS_Store', 'Thumbs.db'
        ]
    
    def resolve_references(self, text: str) -> Dict[str, Any]:
        """Find and resolve all @references in text"""
        # Pattern for @file references with optional line range
        # Matches: @path/to/file.py:10-20 or @path/to/file.py:10 or @path/to/file.py
        file_pattern = r'@([^\s@:]+(?:\.[^\s@:]+)?)(?::(\d+)(?:-(\d+))?)?'
        
        references = {
            'files': [],
            'folders': [],
            'resolved_content': text
        }
        
        # First, find folder references (ending with /)
        folder_pattern = r'@([^\s@:]+)/'
        for match in re.finditer(folder_pattern, text):
            ref_path = match.group(1)
            folder_ref = self._resolve_folder(ref_path)
            if folder_ref:
                references['folders'].append(folder_ref)
                references['resolved_content'] = references['resolved_content'].replace(
                    f'@{ref_path}/', f'[Folder: {ref_path}/]'
                )
        
        # Then find file references
        for match in re.finditer(file_pattern, text):
            # Skip if the match is part of a folder reference (ends with /)
            if match.group(0).endswith('/'):
                continue
            
            ref_path = match.group(1)
            line_start = match.group(2)
            line_end = match.group(3)
            
            # Skip if it's actually a folder (has trailing slash in original)
            full_match = match.group(0)
            end_pos = match.end()
            if end_pos < len(text) and text[end_pos] == '/':
                continue
            
            # Resolve the file
            file_ref = self._resolve_file(ref_path, line_start, line_end)
            if file_ref:
                references['files'].append(file_ref)
                references['resolved_content'] = references['resolved_content'].replace(
                    match.group(0), f'[File: {ref_path}]'
                )
        
        return references
    
    def _resolve_file(self, ref_path: str, line_start: Optional[str] = None, line_end: Optional[str] = None) -> Optional[FileReference]:
        """Resolve a file reference"""
        # Try to find the file
        full_path = self._find_file(ref_path)
        
        if not full_path or not full_path.exists() or not full_path.is_file():
            return None
        
        try:
            content = full_path.read_text(encoding='utf-8', errors='ignore')
            
            # Parse line range if specified
            ls = int(line_start) if line_start else None
            # If only line_start is specified, line_end defaults to line_start (single line)
            le = int(line_end) if line_end else ls
            
            return FileReference(
                path=str(full_path.relative_to(self.workspace_root)),
                content=content,
                line_start=ls,
                line_end=le,
            )
        except Exception:
            return None
    
    def _find_file(self, ref_path: str) -> Optional[Path]:
        """Find a file by path, trying multiple strategies"""
        # Strategy 1: Direct path
        full_path = self.workspace_root / ref_path
        if full_path.exists() and full_path.is_file():
            return full_path
        
        # Strategy 2: In tamfis_code directory
        full_path = self.workspace_root / 'tamfis_code' / Path(ref_path).name
        if full_path.exists() and full_path.is_file():
            return full_path
        
        # Strategy 3: With common extensions
        extensions = ['.py', '.js', '.ts', '.go', '.rs', '.c', '.cpp', '.h', '.java', 
                      '.rb', '.php', '.swift', '.kt', '.json', '.yaml', '.yml', '.toml',
                      '.md', '.txt', '.sh', '.bash', '.zsh', '.fish', '.html', '.css']
        for ext in extensions:
            # Try with extension in root
            full_path = self.workspace_root / f"{ref_path}{ext}"
            if full_path.exists() and full_path.is_file():
                return full_path
            # Try with extension in tamfis_code
            full_path = self.workspace_root / 'tamfis_code' / f"{Path(ref_path).name}{ext}"
            if full_path.exists() and full_path.is_file():
                return full_path
        
        # Strategy 4: Search recursively
        for path in self.workspace_root.rglob(f"*{Path(ref_path).name}"):
            if path.is_file():
                return path
        
        return None
    
    def _resolve_folder(self, ref_path: str) -> Optional[FolderReference]:
        """Resolve a folder reference"""
        full_path = self.workspace_root / ref_path
        
        if not full_path.exists() or not full_path.is_dir():
            return None
        
        files = []
        for file_path in self._get_files_in_folder(full_path):
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                files.append(FileReference(
                    path=str(file_path.relative_to(self.workspace_root)),
                    content=content,
                ))
            except Exception:
                continue
        
        return FolderReference(
            path=str(full_path.relative_to(self.workspace_root)),
            files=files,
        )
    
    def _get_files_in_folder(self, folder: Path) -> List[Path]:
        """Get all files in a folder (recursive)"""
        files = []
        for item in folder.rglob('*'):
            if item.is_file():
                # Check ignore patterns
                if not self._is_ignored(item):
                    files.append(item)
        return files
    
    def _is_ignored(self, path: Path) -> bool:
        """Check if a path should be ignored"""
        path_str = str(path)
        for pattern in self.ignore_patterns:
            if fnmatch.fnmatch(path_str, f"*{pattern}*"):
                return True
        return False
    
    def get_context_summary(self, references: Dict[str, Any]) -> str:
        """Generate a summary of resolved references for the AI context"""
        parts = []
        
        if references['files']:
            parts.append("## Referenced Files\n")
            for f in references['files']:
                parts.append(f"### {f.path}")
                parts.append(f.get_context())
                parts.append("")
        
        if references['folders']:
            parts.append("## Referenced Folders\n")
            for folder in references['folders']:
                parts.append(f"### {folder.path}")
                parts.append(f"({len(folder.files)} files)")
                parts.append("")
        
        return '\n'.join(parts) if parts else ""

class InstructionManager:
    """Manages instruction files (TAMFIS.md, etc.)"""
    
    INSTRUCTION_FILES = ['TAMFIS.md', 'CLAUDE.md', 'CODEX.md', '.tamfis', '.claude']
    
    def __init__(self, workspace_root: Union[str, Path]):
        self.workspace_root = Path(workspace_root)
        self.instructions: Dict[str, str] = {}
        self._load_instructions()
    
    def _load_instructions(self):
        """Load all instruction files from the workspace"""
        for filename in self.INSTRUCTION_FILES:
            # Check root
            root_file = self.workspace_root / filename
            if root_file.exists() and root_file.is_file():
                self.instructions[filename] = root_file.read_text(encoding='utf-8', errors='ignore')
            
            # Check .tamfis directory
            config_file = self.workspace_root / '.tamfis' / filename
            if config_file.exists() and config_file.is_file():
                self.instructions[f".tamfis/{filename}"] = config_file.read_text(encoding='utf-8', errors='ignore')
            
            # Check .claude directory
            config_file = self.workspace_root / '.claude' / filename
            if config_file.exists() and config_file.is_file():
                self.instructions[f".claude/{filename}"] = config_file.read_text(encoding='utf-8', errors='ignore')
    
    def get_instruction(self, name: str = None) -> Optional[str]:
        """Get a specific instruction file content"""
        if name:
            return self.instructions.get(name)
        
        # Return the first found instruction
        for filename in self.INSTRUCTION_FILES:
            if filename in self.instructions:
                return self.instructions[filename]
            if f".tamfis/{filename}" in self.instructions:
                return self.instructions[f".tamfis/{filename}"]
        
        return None
    
    def get_all_instructions(self) -> Dict[str, str]:
        """Get all instruction files"""
        return self.instructions
    
    def get_combined_instructions(self) -> str:
        """Get combined instruction content"""
        parts = []
        for name, content in self.instructions.items():
            parts.append(f"## {name}\n{content}")
        return '\n\n'.join(parts)
    
    def create_instruction_file(self, filename: str = 'TAMFIS.md', content: str = None) -> Path:
        """Create a new instruction file"""
        file_path = self.workspace_root / filename
        
        if content is None:
            content = """# TAMFIS-CODE Instructions

## Project Context
<!-- Describe your project context here -->

## Coding Standards
<!-- Specify coding standards and style guide -->

## Common Patterns
<!-- Document common patterns used in the project -->

## Important Notes
<!-- Any additional important information -->
"""
        
        file_path.write_text(content, encoding='utf-8')
        self.instructions[filename] = content
        return file_path

def process_references(text: str, workspace_root: Union[str, Path]) -> Dict[str, Any]:
    """Process all @references in text"""
    resolver = ReferenceResolver(workspace_root)
    references = resolver.resolve_references(text)
    
    # Get instruction context
    instruction_mgr = InstructionManager(workspace_root)
    instruction_context = instruction_mgr.get_combined_instructions()
    
    return {
        'references': references,
        'instruction_context': instruction_context,
        'enhanced_text': '\n\n'.join(filter(None, [
            instruction_context if instruction_context else '',
            references['resolved_content']
        ]))
    }
