#!/usr/bin/env python3
"""Test plan and review functionality"""

import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from tamfis_code.planreview import PlanReviewer, Plan, FileChange, ChangeType

class TestPlanReviewer:
    """Test plan and review"""

    def setup_method(self):
        """Setup test environment"""
        self.reviewer = PlanReviewer()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_create_plan(self):
        """Test creating a plan"""
        changes = [
            FileChange(
                path='test.py',
                type=ChangeType.CREATE,
                content='print("Hello")',
                description='Create test file'
            )
        ]
        
        plan = self.reviewer.create_plan('Test Plan', changes)
        assert plan.id is not None
        assert plan.description == 'Test Plan'
        assert len(plan.changes) == 1

    def test_review_plan(self):
        """Test reviewing a plan"""
        changes = [
            FileChange(path='file1.py', type=ChangeType.UPDATE, description='Update file1'),
            FileChange(path='file2.py', type=ChangeType.DELETE, description='Delete file2'),
        ]
        self.reviewer.create_plan('Review Test', changes)
        
        summary = self.reviewer.review_plan()
        assert summary['total_changes'] == 2
        assert 'update' in summary['changes_by_type']
        assert 'delete' in summary['changes_by_type']

    def test_approve_plan(self):
        """Test approving a plan"""
        changes = [FileChange(path='test.py', type=ChangeType.CREATE, content='test')]
        self.reviewer.create_plan('Approve Test', changes)
        
        result = self.reviewer.approve('Looks good')
        assert result is True
        assert self.reviewer.current_plan.approved is True

    def test_reject_plan(self):
        """Test rejecting a plan"""
        changes = [FileChange(path='test.py', type=ChangeType.CREATE, content='test')]
        self.reviewer.create_plan('Reject Test', changes)
        
        result = self.reviewer.reject('Not good')
        assert result is True
        assert self.reviewer.current_plan.approved is False

    def test_apply_plan(self):
        """Test applying a plan"""
        test_file = Path(self.temp_dir) / 'test.py'
        changes = [
            FileChange(
                path=str(test_file),
                type=ChangeType.CREATE,
                content='print("Hello")',
                description='Create test file'
            )
        ]
        self.reviewer.create_plan('Apply Test', changes)
        self.reviewer.approve()
        
        results = self.reviewer.apply()
        assert results[0]['success'] is True
        assert test_file.exists()
        assert test_file.read_text() == 'print("Hello")'

    def test_apply_unapproved_plan(self):
        """Test applying an unapproved plan"""
        changes = [FileChange(path='test.py', type=ChangeType.CREATE, content='test')]
        self.reviewer.create_plan('Unapproved', changes)
        
        results = self.reviewer.apply()
        assert 'error' in results[0]

    def test_dry_run(self):
        """Test dry run mode"""
        test_file = Path(self.temp_dir) / 'test_dry_run.py'
        changes = [
            FileChange(
                path=str(test_file),
                type=ChangeType.CREATE,
                content='print("Dry run")',
                description='Dry run test'
            )
        ]
        self.reviewer.create_plan('Dry Run', changes)
        self.reviewer.approve()
        
        results = self.reviewer.apply(dry_run=True)
        assert results[0]['dry_run'] is True
        assert not test_file.exists()  # Should not have been created

    def test_undo_last_apply(self):
        """Test undoing the last applied plan"""
        test_file = Path(self.temp_dir) / 'undo_test.py'
        changes = [
            FileChange(
                path=str(test_file),
                type=ChangeType.CREATE,
                content='print("Undo test")',
                description='Undo test'
            )
        ]
        self.reviewer.create_plan('Undo Test', changes)
        self.reviewer.approve()
        self.reviewer.apply()
        
        # Now undo
        result = self.reviewer.undo_last_apply()
        assert 'undone_plan' in result
        assert not test_file.exists()  # File should be deleted

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
