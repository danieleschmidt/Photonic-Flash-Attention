"""Monitoring and observability for photonic attention system."""

# Try to import full health monitor, fallback to simple monitor
try:
    from .health_monitor import (
        HealthMonitor,
        HealthStatus,
        ComponentType,
        HealthMetric,
        ComponentHealth,
        get_health_monitor,
        start_health_monitoring,
        stop_health_monitoring,
        get_system_health
    )
except ImportError:
    # Fallback to simple health monitor
    from .simple_health import (
        SimpleHealthMonitor as HealthMonitor,
        HealthStatus,
        get_simple_health_monitor as get_health_monitor,
        get_system_health_simple as get_system_health
    )
    
    # Create dummy implementations for compatibility
    ComponentType = None
    HealthMetric = None
    ComponentHealth = None
    
    def start_health_monitoring(check_interval: float = 30.0):
        monitor = get_health_monitor()
        monitor.start_monitoring(check_interval)
    
    def stop_health_monitoring():
        monitor = get_health_monitor()
        monitor.stop_monitoring()

__all__ = [
    "HealthMonitor",
    "HealthStatus", 
    "get_health_monitor",
    "start_health_monitoring", 
    "stop_health_monitoring",
    "get_system_health"
]