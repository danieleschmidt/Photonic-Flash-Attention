"""Performance monitoring utilities."""

import time
import threading
from typing import Dict, Any, Optional
from collections import defaultdict, deque


class PerformanceMonitor:
    """Monitor performance metrics across devices."""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics = defaultdict(lambda: deque(maxlen=history_size))
        self._lock = threading.Lock()
    
    def record_metric(self, name: str, value: float, device: str = "default") -> None:
        """Record a performance metric."""
        with self._lock:
            key = f"{device}_{name}"
            self.metrics[key].append((time.time(), value))
    
    def get_average(self, name: str, device: str = "default", window: int = 100) -> float:
        """Get average of recent metrics."""
        with self._lock:
            key = f"{device}_{name}"
            if key not in self.metrics:
                return 0.0
            
            recent = list(self.metrics[key])[-window:]
            if not recent:
                return 0.0
            
            return sum(val for _, val in recent) / len(recent)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get all statistics."""
        with self._lock:
            stats = {}
            for key, values in self.metrics.items():
                if values:
                    recent_values = [val for _, val in values]
                    stats[key] = {
                        "count": len(recent_values),
                        "average": sum(recent_values) / len(recent_values),
                        "min": min(recent_values),
                        "max": max(recent_values),
                    }
            return stats


class HealthMonitor:
    """Monitor system health."""
    
    def __init__(self):
        self.status = "healthy"
        self.issues = []
        self._lock = threading.Lock()
    
    def check_health(self) -> bool:
        """Check overall system health."""
        with self._lock:
            return self.status == "healthy"
    
    def report_issue(self, issue: str, severity: str = "warning") -> None:
        """Report a health issue."""
        with self._lock:
            self.issues.append({
                "timestamp": time.time(),
                "issue": issue,
                "severity": severity,
            })
            
            if severity == "critical":
                self.status = "critical"
            elif severity == "warning" and self.status == "healthy":
                self.status = "degraded"
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        with self._lock:
            return {
                "status": self.status,
                "issue_count": len(self.issues),
                "recent_issues": self.issues[-10:],  # Last 10 issues
            }