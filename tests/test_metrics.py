#!/usr/bin/env python3
"""Test metrics tracking"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from tamfis_code.metrics import MetricsTracker, StreamMetrics

class TestStreamMetrics:
    """Test stream metrics"""

    def test_initialization(self):
        """Test metrics initialization"""
        metrics = StreamMetrics()
        assert metrics.tokens_used == 0
        assert metrics.tokens_per_second == 0.0

    def test_update(self):
        """Test updating metrics"""
        metrics = StreamMetrics()
        metrics.update(100, 500)  # 100 tokens in 500ms
        
        assert metrics.tokens_used == 100
        assert metrics.response_time_ms == 500

    def test_estimate_cost(self):
        """Test cost estimation"""
        metrics = StreamMetrics()
        metrics.tokens_used = 1000
        
        cost = metrics.estimate_cost('claude-3.5')
        # 1000 * 0.020 / 1000 = 0.02
        assert cost == pytest.approx(0.02, 0.001)

    def test_format_display(self):
        """Test display formatting"""
        metrics = StreamMetrics()
        metrics.tokens_used = 500
        metrics.tokens_per_second = 10.5
        metrics.estimated_cost = 0.01
        
        display = metrics.format_display()
        assert '500' in display
        assert '10.5' in display

class TestMetricsTracker:
    """Test metrics tracker"""

    def test_start_stop(self):
        """Test starting and stopping tracker"""
        tracker = MetricsTracker()
        tracker.start()
        assert tracker._active is True
        
        tracker.stop()
        assert tracker._active is False

    def test_record(self):
        """Test recording metrics"""
        tracker = MetricsTracker()
        tracker.record(100, 500, 'claude-3.5')
        
        assert tracker.metrics.tokens_used == 100
        assert tracker.metrics.model_name == 'claude-3.5'

    def test_get_summary(self):
        """Test getting metrics summary"""
        tracker = MetricsTracker()
        tracker.record(100, 500)
        
        summary = tracker.get_summary()
        assert 'tokens_used' in summary
        assert summary['tokens_used'] == 100

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
