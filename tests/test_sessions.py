#!/usr/bin/env python3
"""Test session management"""

import sys
import os
import tempfile
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from tamfis_code.sessions import SessionManager, Session, Message

class TestSessionManager:
    """Test session management"""

    def setup_method(self):
        """Setup test environment with temporary DB"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / 'test_sessions.db'
        self.manager = SessionManager(self.db_path)

    def teardown_method(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_create_session(self):
        """Test creating a session"""
        session = self.manager.create_session('Test Session')
        
        assert session.id is not None
        assert session.name == 'Test Session'
        assert len(session.messages) == 0

    def test_add_message(self):
        """Test adding messages to a session"""
        session = self.manager.create_session('Test Session')
        session.add_message('user', 'Hello, world!')
        session.add_message('assistant', 'Hi there!')
        
        assert len(session.messages) == 2
        assert session.messages[0].role == 'user'
        assert session.messages[0].content == 'Hello, world!'

    def test_save_and_load(self):
        """Test saving and loading a session"""
        session = self.manager.create_session('Test Session')
        session.add_message('user', 'Test message')
        self.manager.save(session)
        
        loaded = self.manager.load(session.id)
        assert loaded is not None
        assert loaded.id == session.id
        assert len(loaded.messages) == 1

    def test_list_sessions(self):
        """Test listing sessions"""
        self.manager.create_session('Session 1')
        self.manager.create_session('Session 2')
        
        sessions = self.manager.list_sessions()
        assert len(sessions) >= 2

    def test_fork_session(self):
        """Test forking a session"""
        original = self.manager.create_session('Original')
        original.add_message('user', 'Original message')
        self.manager.save(original)
        
        forked = self.manager.fork_session(original.id, 'Forked')
        assert forked is not None
        assert forked.id != original.id
        assert len(forked.messages) == 1

    def test_delete_session(self):
        """Test deleting a session"""
        session = self.manager.create_session('To Delete')
        self.manager.save(session)
        
        self.manager.delete(session.id)
        loaded = self.manager.load(session.id)
        assert loaded is None

    def test_delete_old(self):
        """Test deleting old sessions"""
        # Create sessions with different dates
        session1 = self.manager.create_session('Old')
        # Modify the session to have an old date
        session1.updated_at = datetime(2020, 1, 1)
        self.manager.save(session1)
        
        session2 = self.manager.create_session('New')
        self.manager.save(session2)
        
        count = self.manager.delete_old(days=10)
        assert count >= 1  # Should delete the old session

class TestSession:
    """Test the Session class"""

    def test_to_dict(self):
        """Test converting session to dict"""
        session = Session(
            id='test123',
            name='Test',
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        session.add_message('user', 'Hello')
        
        data = session.to_dict()
        assert data['id'] == 'test123'
        assert data['name'] == 'Test'
        assert len(data['messages']) == 1

    def test_from_dict(self):
        """Test creating session from dict"""
        data = {
            'id': 'test456',
            'name': 'From Dict',
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'messages': [
                {'role': 'user', 'content': 'Hello', 'timestamp': datetime.now().isoformat(), 'metadata': {}}
            ],
            'context': {},
            'files': [],
            'is_active': True
        }
        
        session = Session.from_dict(data)
        assert session.id == 'test456'
        assert session.name == 'From Dict'
        assert len(session.messages) == 1

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
