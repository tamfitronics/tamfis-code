"""Instruction file management and reference processing"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

# Import the new reference system
from .references import (
    InstructionManager, 
    ReferenceResolver, 
    FileReference,
    FolderReference,
    process_references
)

# Re-export for backward compatibility
__all__ = [
    'InstructionManager',
    'ReferenceResolver',
    'FileReference',
    'FolderReference',
    'process_references',
    'get_instruction_context',
    'resolve_file_references',
]

def get_instruction_context(workspace_root: Union[str, Path]) -> str:
    """Get instruction context for the workspace (backward compatible)"""
    mgr = InstructionManager(workspace_root)
    return mgr.get_combined_instructions()

def resolve_file_references(text: str, workspace_root: Union[str, Path]) -> Dict[str, Any]:
    """Resolve file references in text (backward compatible)"""
    resolver = ReferenceResolver(workspace_root)
    return resolver.resolve_references(text)

# Keep the existing functionality for backward compatibility
# The original parse_instruction_file function is preserved

def parse_instruction_file(file_path: Union[str, Path]) -> Dict[str, str]:
    """Parse an instruction file into sections"""
    path = Path(file_path)
    if not path.exists():
        return {}
    
    content = path.read_text(encoding='utf-8', errors='ignore')
    sections = {}
    
    # Parse markdown sections
    section_pattern = r'^##+\s+(.+)$'
    current_section = None
    current_content = []
    
    for line in content.split('\n'):
        match = re.match(section_pattern, line)
        if match:
            if current_section:
                sections[current_section] = '\n'.join(current_content).strip()
            current_section = match.group(1).strip()
            current_content = []
        else:
            current_content.append(line)
    
    if current_section:
        sections[current_section] = '\n'.join(current_content).strip()
    
    return sections

def create_instruction_template(path: Union[str, Path] = 'TAMFIS.md') -> str:
    """Create a template instruction file"""
    template = """# TAMFIS-CODE Instructions

## Project Overview
<!-- Describe your project, its purpose, and main components -->

## Coding Standards
<!-- Define coding standards and style guide -->

### Python Standards
- Use PEP 8 for Python code
- Maximum line length: 100 characters
- Use type hints for all function definitions

### JavaScript/TypeScript Standards
- Use ESLint with standard config
- Prefer async/await over callbacks

## Project Structure
<!-- Document your project structure -->
.
├── src/
│ └── ...
├── tests/
│ └── ...
└── docs/
└── ...


## Common Patterns
<!-- Document common patterns used in the project -->

### Error Handling
<!-- How errors are handled -->

### Testing
<!-- Testing strategy and tools -->

## Important Notes
<!-- Any additional important information -->

## Dependencies
<!-- Key dependencies and their versions -->

## Environment Variables
<!-- Required environment variables -->
"""
    path = Path(path)
    path.write_text(template, encoding='utf-8')
    return template
