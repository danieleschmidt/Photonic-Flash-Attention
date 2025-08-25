"""
Simple health monitoring system without external dependencies.

Provides basic system health monitoring for photonic attention systems
using only standard library components for maximum portability.
"""

import time
import threading
import os
import gc
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

from ..utils.logging import get_logger


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class SimpleHealthMetric:
    """Simple health metric without external dependencies."""
    name: str
    value: float
    unit: str
    status: HealthStatus
    timestamp: float = field(default_factory=time.time)
    message: str = ""


class SimpleHealthMonitor:
    """Simple health monitoring system using standard library only."""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.metrics_history: deque = deque(maxlen=1000)
        self.monitoring_enabled = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self._lock = threading.RLock()
        
        # Simple counters
        self.operation_count = 0
        self.error_count = 0
        self.last_error_time = 0
        self.start_time = time.time()
        
        self.logger.info("Simple health monitor initialized")
    
    def record_operation(self, success: bool = True, operation_name: str = "unknown"):
        """Record an operation result."""
        with self._lock:
            self.operation_count += 1
            
            if not success:
                self.error_count += 1
                self.last_error_time = time.time()
                
                metric = SimpleHealthMetric(
                    name="operation_error",
                    value=1.0,
                    unit="error",
                    status=HealthStatus.WARNING,
                    message=f"Error in {operation_name}"
                )
                self.metrics_history.append(metric)
                self.logger.warning(f"Operation error recorded: {operation_name}")
    
    def check_memory_health(self) -> SimpleHealthMetric:
        """Check memory health using standard library."""
        try:
            # Force garbage collection and get counts
            collected = gc.collect()
            
            # Get object counts as a proxy for memory usage
            obj_counts = len(gc.get_objects())
            
            # Simple heuristic: if we have a lot of objects, memory might be high
            status = HealthStatus.HEALTHY
            message = "Memory usage normal"
            
            if obj_counts > 100000:  # Arbitrary threshold
                status = HealthStatus.WARNING
                message = "High object count detected"
            elif obj_counts > 500000:
                status = HealthStatus.CRITICAL
                message = "Very high object count detected"
            
            return SimpleHealthMetric(
                name="memory_objects",
                value=obj_counts,
                unit="objects",
                status=status,
                message=message
            )
            
        except Exception as e:
            return SimpleHealthMetric(
                name="memory_check",
                value=0.0,
                unit="error",
                status=HealthStatus.CRITICAL,
                message=f"Memory check failed: {e}"
            )
    
    def check_thread_health(self) -> SimpleHealthMetric:
        """Check thread health."""
        try:
            # Get active thread count
            thread_count = threading.active_count()
            
            status = HealthStatus.HEALTHY
            message = "Thread count normal"
            
            if thread_count > 50:  # Arbitrary threshold
                status = HealthStatus.WARNING
                message = "High thread count"
            elif thread_count > 100:
                status = HealthStatus.CRITICAL
                message = "Very high thread count"
            
            return SimpleHealthMetric(
                name="thread_count",
                value=thread_count,
                unit="threads",
                status=status,
                message=message
            )
            
        except Exception as e:
            return SimpleHealthMetric(
                name="thread_check",
                value=0.0,
                unit="error",
                status=HealthStatus.CRITICAL,
                message=f"Thread check failed: {e}"
            )
    
    def check_error_rate(self) -> SimpleHealthMetric:
        """Check error rate."""
        try:
            uptime = time.time() - self.start_time
            error_rate = self.error_count / max(uptime, 1.0)  # errors per second
            
            status = HealthStatus.HEALTHY
            message = "Error rate normal"
            
            if error_rate > 0.1:  # More than 1 error per 10 seconds
                status = HealthStatus.WARNING
                message = "Elevated error rate"
            elif error_rate > 1.0:  # More than 1 error per second
                status = HealthStatus.CRITICAL
                message = "High error rate"
            
            return SimpleHealthMetric(
                name="error_rate",
                value=error_rate,
                unit="errors/sec",
                status=status,
                message=message
            )
            
        except Exception as e:
            return SimpleHealthMetric(
                name="error_rate_check",
                value=0.0,
                unit="error",
                status=HealthStatus.CRITICAL,
                message=f"Error rate check failed: {e}"
            )
    
    def check_uptime(self) -> SimpleHealthMetric:
        """Check system uptime."""
        try:
            uptime = time.time() - self.start_time
            
            status = HealthStatus.HEALTHY
            message = f"Uptime: {uptime:.1f} seconds"
            
            if uptime < 60:  # Less than 1 minute - recently started
                status = HealthStatus.WARNING
                message = "Recently started"
            
            return SimpleHealthMetric(
                name="uptime",
                value=uptime,
                unit="seconds",
                status=status,
                message=message
            )
            
        except Exception as e:
            return SimpleHealthMetric(
                name="uptime_check",
                value=0.0,
                unit="error",
                status=HealthStatus.CRITICAL,
                message=f"Uptime check failed: {e}"
            )
    
    def run_all_checks(self) -> List[SimpleHealthMetric]:
        """Run all health checks and return metrics."""
        metrics = []
        
        with self._lock:
            checks = [
                self.check_memory_health,
                self.check_thread_health,
                self.check_error_rate,
                self.check_uptime
            ]
            
            for check in checks:
                try:
                    metric = check()
                    metrics.append(metric)
                    self.metrics_history.append(metric)
                except Exception as e:
                    error_metric = SimpleHealthMetric(
                        name=f"{check.__name__}_failed",
                        value=0.0,
                        unit="error",
                        status=HealthStatus.CRITICAL,
                        message=f"Check failed: {e}"
                    )
                    metrics.append(error_metric)
                    self.metrics_history.append(error_metric)
        
        return metrics
    
    def get_overall_health(self) -> Tuple[HealthStatus, Dict[str, Any]]:
        """Get overall health status."""
        metrics = self.run_all_checks()
        
        # Determine overall status
        statuses = [m.status for m in metrics]
        
        if HealthStatus.CRITICAL in statuses:
            overall_status = HealthStatus.CRITICAL
        elif HealthStatus.DEGRADED in statuses:
            overall_status = HealthStatus.DEGRADED
        elif HealthStatus.WARNING in statuses:
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.HEALTHY
        
        # Create summary
        summary = {
            "overall_status": overall_status.value,
            "uptime_seconds": time.time() - self.start_time,
            "total_operations": self.operation_count,
            "total_errors": self.error_count,
            "error_rate": self.error_count / max(time.time() - self.start_time, 1.0),
            "last_error_time": self.last_error_time,
            "metrics_count": len(metrics),
            "critical_metrics": [m.name for m in metrics if m.status == HealthStatus.CRITICAL],
            "warning_metrics": [m.name for m in metrics if m.status == HealthStatus.WARNING],
            "monitoring_enabled": self.monitoring_enabled
        }
        
        return overall_status, summary
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get detailed health report."""
        overall_status, summary = self.get_overall_health()
        
        with self._lock:
            recent_metrics = [
                m for m in self.metrics_history
                if time.time() - m.timestamp < 3600  # Last hour
            ]
            
            metric_details = []
            for metric in recent_metrics[-20:]:  # Last 20 metrics
                metric_details.append({
                    "name": metric.name,
                    "value": metric.value,
                    "unit": metric.unit,
                    "status": metric.status.value,
                    "timestamp": metric.timestamp,
                    "message": metric.message
                })
        
        return {
            "overall": summary,
            "recent_metrics": metric_details,
            "system_info": {
                "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
                "platform": os.name,
                "pid": os.getpid(),
                "thread_count": threading.active_count()
            }
        }
    
    def start_monitoring(self, check_interval: float = 30.0):
        """Start background health monitoring."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.logger.warning("Monitoring already running")
            return
        
        self.monitoring_enabled = True
        self.stop_event.clear()
        
        def monitoring_loop():
            while self.monitoring_enabled and not self.stop_event.wait(check_interval):
                try:
                    metrics = self.run_all_checks()
                    
                    # Log critical issues
                    for metric in metrics:
                        if metric.status == HealthStatus.CRITICAL:
                            self.logger.critical(f"Critical health issue: {metric.name} - {metric.message}")
                        elif metric.status == HealthStatus.WARNING:
                            self.logger.warning(f"Health warning: {metric.name} - {metric.message}")
                            
                except Exception as e:
                    self.logger.error(f"Error in monitoring loop: {e}")
        
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info(f"Started health monitoring with {check_interval}s interval")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring_enabled = False
        self.stop_event.set()
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        self.logger.info("Stopped health monitoring")
    
    def reset_stats(self):
        """Reset all statistics."""
        with self._lock:
            self.operation_count = 0
            self.error_count = 0
            self.last_error_time = 0
            self.start_time = time.time()
            self.metrics_history.clear()
            self.logger.info("Health statistics reset")


# Global simple health monitor
_simple_health_monitor: Optional[SimpleHealthMonitor] = None


def get_simple_health_monitor() -> SimpleHealthMonitor:
    """Get global simple health monitor instance."""
    global _simple_health_monitor
    if _simple_health_monitor is None:
        _simple_health_monitor = SimpleHealthMonitor()
    return _simple_health_monitor


def get_system_health_simple() -> Tuple[HealthStatus, Dict[str, Any]]:
    """Get simple system health status."""
    monitor = get_simple_health_monitor()
    return monitor.get_overall_health()


def record_operation_result(success: bool = True, operation_name: str = "unknown"):
    """Record operation result for health tracking."""
    monitor = get_simple_health_monitor()
    monitor.record_operation(success, operation_name)