"""Persistent session management for TAMFIS-CODE"""

import json
import sqlite3
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict

SESSION_DB_PATH = Path.home() / ".tamfis" / "sessions.db"

@dataclass
class Message:
    """Individual message in a session"""
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Session:
    """A complete conversation session"""
    id: str
    name: str
    created_at: datetime
    updated_at: datetime
    messages: List[Message] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    files: List[str] = field(default_factory=list)
    is_active: bool = True
    
    def add_message(self, role: str, content: str, **metadata):
        """Add a message to the session"""
        self.messages.append(Message(role, content, timestamp=datetime.now(), metadata=metadata))
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'messages': [
                {
                    'role': m.role,
                    'content': m.content,
                    'timestamp': m.timestamp.isoformat(),
                    'metadata': m.metadata
                }
                for m in self.messages
            ],
            'context': self.context,
            'files': self.files,
            'is_active': self.is_active,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Session':
        """Create Session from dictionary"""
        messages = [
            Message(
                role=m['role'],
                content=m['content'],
                timestamp=datetime.fromisoformat(m['timestamp']),
                metadata=m.get('metadata', {})
            )
            for m in data.get('messages', [])
        ]
        return cls(
            id=data['id'],
            name=data['name'],
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            messages=messages,
            context=data.get('context', {}),
            files=data.get('files', []),
            is_active=data.get('is_active', True),
        )

class SessionManager:
    """Manages persistent sessions with SQLite backend"""
    
    def __init__(self, db_path: Path = SESSION_DB_PATH):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize the database"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    messages TEXT,
                    context TEXT,
                    files TEXT,
                    is_active INTEGER
                )
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_sessions_updated 
                ON sessions(updated_at DESC)
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_sessions_active 
                ON sessions(is_active)
            ''')
    
    def save(self, session: Session) -> None:
        """Save a session to the database"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO sessions 
                (id, name, created_at, updated_at, messages, context, files, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session.id,
                session.name,
                session.created_at.isoformat(),
                session.updated_at.isoformat(),
                json.dumps([m.__dict__ if hasattr(m, '__dict__') else asdict(m) 
                           for m in session.messages], default=str),
                json.dumps(session.context),
                json.dumps(session.files),
                1 if session.is_active else 0
            ))
    
    def load(self, session_id: str) -> Optional[Session]:
        """Load a session by ID"""
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                'SELECT * FROM sessions WHERE id = ?', (session_id,)
            ).fetchone()
            
            if not row:
                return None
            
            return self._row_to_session(row)
    
    def load_active(self) -> Optional[Session]:
        """Load the most recent active session"""
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                'SELECT * FROM sessions WHERE is_active = 1 ORDER BY updated_at DESC LIMIT 1'
            ).fetchone()
            
            if not row:
                return None
            
            return self._row_to_session(row)
    
    def list_sessions(self, limit: int = 20) -> List[Session]:
        """List recent sessions"""
        with sqlite3.connect(str(self.db_path)) as conn:
            rows = conn.execute(
                'SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?',
                (limit,)
            ).fetchall()
            
            return [self._row_to_session(row) for row in rows]
    
    def delete(self, session_id: str) -> None:
        """Delete a session"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute('DELETE FROM sessions WHERE id = ?', (session_id,))
    
    def delete_old(self, days: int = 30) -> int:
        """Delete sessions older than specified days"""
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(
                'DELETE FROM sessions WHERE updated_at < ?',
                (cutoff,)
            )
            return cursor.rowcount
    
    def _row_to_session(self, row) -> Session:
        """Convert database row to Session object"""
        (id, name, created_at, updated_at, messages_json, 
         context_json, files_json, is_active) = row
        
        messages_data = json.loads(messages_json)
        messages = [
            Message(
                role=m['role'],
                content=m['content'],
                timestamp=datetime.fromisoformat(m['timestamp']),
                metadata=m.get('metadata', {})
            )
            for m in messages_data
        ]
        
        return Session(
            id=id,
            name=name,
            created_at=datetime.fromisoformat(created_at),
            updated_at=datetime.fromisoformat(updated_at),
            messages=messages,
            context=json.loads(context_json),
            files=json.loads(files_json),
            is_active=bool(is_active),
        )
    
    def create_session(self, name: str = None) -> Session:
        """Create a new session"""
        session_id = str(uuid.uuid4())[:8]
        now = datetime.now()
        
        if not name:
            name = f"session-{session_id}"
        
        session = Session(
            id=session_id,
            name=name,
            created_at=now,
            updated_at=now,
        )
        self.save(session)
        return session
    
    def archive_session(self, session_id: str) -> None:
        """Archive a session (set inactive)"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                'UPDATE sessions SET is_active = 0 WHERE id = ?',
                (session_id,)
            )
    
    def fork_session(self, session_id: str, new_name: str = None) -> Optional[Session]:
        """Fork a session (copy with new ID)"""
        original = self.load(session_id)
        if not original:
            return None
        
        new_session = Session(
            id=str(uuid.uuid4())[:8],
            name=new_name or f"{original.name}-fork",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            messages=[m for m in original.messages],
            context=dict(original.context),
            files=list(original.files),
            is_active=True,
        )
        self.save(new_session)
        return new_session
