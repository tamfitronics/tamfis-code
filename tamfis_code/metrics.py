"""Streaming metrics and performance monitoring"""

import time
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

@dataclass
class StreamMetrics:
    """Real-time streaming metrics"""
    tokens_used: int = 0
    tokens_per_second: float = 0.0
    estimated_cost: float = 0.0
    model_name: str = "default"
    context_used: int = 0
    response_time_ms: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    
    # Per-command tracking
    command_tokens: Dict[str, int] = field(default_factory=dict)
    command_times: Dict[str, float] = field(default_factory=dict)
    
    def update(self, tokens: int, elapsed_ms: float) -> None:
        """Update metrics with new data"""
        self.tokens_used += tokens
        self.response_time_ms = elapsed_ms
        self.last_update = datetime.now()
        
        delta = (self.last_update - self.start_time).total_seconds()
        if delta > 0:
            self.tokens_per_second = self.tokens_used / delta
    
    def estimate_cost(self, model: str) -> float:
        """Estimate cost based on model pricing"""
        rates = {
            'claude-3': 0.015 / 1000,
            'claude-3.5': 0.020 / 1000,
            'gpt-4': 0.030 / 1000,
            'gpt-3.5': 0.002 / 1000,
            'deepseek': 0.001 / 1000,
            'default': 0.010 / 1000,
        }
        rate = rates.get(model, rates['default'])
        return self.tokens_used * rate
    
    def format_display(self) -> str:
        """Format metrics for display"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        cost = self.estimate_cost(self.model_name)
        
        return (
            f"📊 {self.tokens_used} tokens | "
            f"{self.tokens_per_second:.1f} t/s | "
            f"${cost:.4f} | "
            f"{elapsed:.1f}s"
        )

class MetricsTracker:
    """Track metrics across a session"""
    
    def __init__(self):
        self.metrics = StreamMetrics()
        self._active = False
        self._timer: Optional[threading.Timer] = None
        self._display_callback = None
    
    def start(self, display_callback=None):
        """Start tracking metrics"""
        self._active = True
        self.metrics.start_time = datetime.now()
        self._display_callback = display_callback
        self._update_loop()
    
    def stop(self):
        """Stop tracking"""
        self._active = False
        if self._timer:
            self._timer.cancel()
    
    def _update_loop(self):
        """Periodic update loop"""
        if not self._active:
            return
        if self._display_callback:
            self._display_callback(self.metrics.format_display())
        self._timer = threading.Timer(0.5, self._update_loop)
        self._timer.daemon = True
        self._timer.start()
    
    def record(self, tokens: int, elapsed_ms: float, model: str = "default"):
        """Record a response"""
        self.metrics.update(tokens, elapsed_ms)
        self.metrics.model_name = model
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        return {
            "tokens_used": self.metrics.tokens_used,
            "tokens_per_second": self.metrics.tokens_per_second,
            "estimated_cost": self.metrics.estimated_cost,
            "model": self.metrics.model_name,
            "elapsed_seconds": (datetime.now() - self.metrics.start_time).total_seconds(),
        }
