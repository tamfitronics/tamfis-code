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
    
    # Rough blended $/token rates, matched by substring against whatever
    # model name the provider reports (real names look like
    # "gpt-5.6-sol"/"nemotron-3-nano-30b"/"grok-4.6", not the bare family
    # names below) -- this is a soft session-spend signal, not a billing
    # reconciliation tool. Checked in the order below; first match wins.
    _RATE_TABLE = (
        ('opus', 0.045 / 1000),
        ('gpt-5', 0.030 / 1000),
        ('sonnet', 0.015 / 1000),
        ('gpt-4', 0.020 / 1000),
        ('grok', 0.010 / 1000),
        ('gemini', 0.006 / 1000),
        ('nemotron', 0.002 / 1000),
        ('deepseek', 0.001 / 1000),
        ('llama', 0.0015 / 1000),
        ('qwen', 0.0015 / 1000),
    )
    _DEFAULT_RATE = 0.010 / 1000

    def estimate_cost(self, model: str) -> float:
        """Estimate cost based on model pricing"""
        model_lower = (model or "").lower()
        rate = next(
            (r for key, r in self._RATE_TABLE if key in model_lower),
            self._DEFAULT_RATE,
        )
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
    
    def __init__(self, cost_cap_usd: Optional[float] = None):
        self.metrics = StreamMetrics()
        self._active = False
        self._timer: Optional[threading.Timer] = None
        self._display_callback = None
        # Warns once per session when estimated spend crosses this many
        # dollars -- see config.py's session_cost_cap_usd doc comment for
        # why this warns rather than blocks. None/<=0 disables it.
        self._cost_cap_usd = cost_cap_usd
        self._cost_cap_warned = False
    
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

    def check_cost_cap(self) -> Optional[str]:
        """Returns a one-time warning message the first time estimated
        session spend crosses the configured cap, else None. Call after
        every record() -- safe to call unconditionally, only ever returns
        non-None once per session regardless of how many times it's
        called or how far over the cap spend goes afterward."""
        if not self._cost_cap_usd or self._cost_cap_usd <= 0 or self._cost_cap_warned:
            return None
        cost = self.metrics.estimate_cost(self.metrics.model_name)
        if cost < self._cost_cap_usd:
            return None
        self._cost_cap_warned = True
        return (
            f"⚠ Estimated session cost (~${cost:.2f}) has crossed your "
            f"${self._cost_cap_usd:.2f} cap. This is a rough estimate "
            f"(token-count-based, not real provider billing) -- just a "
            f"heads-up, not a limit; nothing is blocked. Adjust or disable "
            f"via session_cost_cap_usd in config or "
            f"TAMFIS_CODE_SESSION_COST_CAP_USD."
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        return {
            "tokens_used": self.metrics.tokens_used,
            "tokens_per_second": self.metrics.tokens_per_second,
            "estimated_cost": self.metrics.estimated_cost,
            "model": self.metrics.model_name,
            "elapsed_seconds": (datetime.now() - self.metrics.start_time).total_seconds(),
        }
