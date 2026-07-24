"""Small dependency-free code index used by optional public exports."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import re

@dataclass(frozen=True)
class CodeSymbol:
    name: str
    kind: str
    line: int

@dataclass
class CodeFile:
    path: str
    language: str
    symbols: list[CodeSymbol] = field(default_factory=list)

class CodeIndexer:
    def __init__(self, root: str | Path): self.root=Path(root).resolve()
    def index_file(self, path: str | Path) -> CodeFile:
        p=Path(path).resolve(); text=p.read_text(encoding='utf-8',errors='replace')
        suffix=p.suffix.lower(); language={'.py':'python','.ts':'typescript','.tsx':'typescript','.js':'javascript','.jsx':'javascript'}.get(suffix,'text')
        symbols=[]
        patterns=[('class',re.compile(r'^\s*class\s+([A-Za-z_]\w*)')),('function',re.compile(r'^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)')),('function',re.compile(r'^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)'))]
        for n,line in enumerate(text.splitlines(),1):
            for kind,pat in patterns:
                m=pat.search(line)
                if m: symbols.append(CodeSymbol(m.group(1),kind,n)); break
        return CodeFile(str(p),language,symbols)
    def index(self) -> list[CodeFile]:
        result=[]
        for p in self.root.rglob('*'):
            if p.is_file() and p.suffix.lower() in {'.py','.ts','.tsx','.js','.jsx'} and not any(part.startswith('.') for part in p.relative_to(self.root).parts):
                try: result.append(self.index_file(p))
                except OSError: pass
        return result
