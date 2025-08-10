"""Health monitoring and system diagnostics for photonic attention."""

import time
import threading
import psutil
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import logging

from ..utils.logging import get_logger
from ..utils.exceptions import PhotonicHardwareError, PhotonicThermalError
from ..photonic.hardware.detection import get_photonic_devices


logger = get_logger(__name__)


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """Types of system components."""
    HARDWARE = "hardware"
    SOFTWARE = "software"
    PHOTONIC = "photonic"
    THERMAL = "thermal"
    MEMORY = "memory"
    NETWORK = "network"
    STORAGE = "storage"


@dataclass
class HealthMetric:
    """Represents a health metric."""
    name: str
    value: float
    unit: str
    status: HealthStatus
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentHealth:
    """Health status of a system component."""
    component_type: ComponentType
    component_id: str
    status: HealthStatus
    metrics: List[HealthMetric]
    last_check: float
    error_count: int = 0
    last_error: Optional[str] = None


class HealthCheck:
    """Base class for health checks."""
    
    def __init__(self, name: str, check_interval: float = 60.0):
        """
        Initialize health check.
        
        Args:
            name: Name of the health check
            check_interval: Interval between checks in seconds
        """
        self.name = name
        self.check_interval = check_interval
        self.last_check = 0.0
        self.enabled = True
        
    def should_run(self) -> bool:
        """Check if health check should run."""
        return (
            self.enabled and
            time.time() - self.last_check >= self.check_interval
        )
    
    def run(self) -> ComponentHealth:
        """Run the health check."""
        self.last_check = time.time()
        return self.check_health()
    
    def check_health(self) -> ComponentHealth:
        """Implement health check logic."""
        raise NotImplementedError("Subclasses must implement check_health")


class PhotonicHardwareHealthCheck(HealthCheck):
    """Health check for photonic hardware."""
    
    def __init__(self):
        super().__init__("photonic_hardware", check_interval=30.0)
    
    def check_health(self) -> ComponentHealth:
        """Check photonic hardware health."""
        metrics = []
        status = HealthStatus.HEALTHY
        error_count = 0
        last_error = None
        
        try:
            devices = get_photonic_devices()
            
            if not devices:
                status = HealthStatus.CRITICAL
                last_error = "No photonic devices available"
                metrics.append(HealthMetric(
                    name="device_count",
                    value=0,
                    unit="devices",
                    status=HealthStatus.CRITICAL
                ))
            else:
                # Check each device
                for device in devices:
                    device_metrics = self._check_device_health(device)
                    metrics.extend(device_metrics)
                    
                    # Update overall status based on device health
                    for metric in device_metrics:
                        if metric.status == HealthStatus.CRITICAL:
                            status = HealthStatus.CRITICAL
                        elif metric.status == HealthStatus.DEGRADED and status != HealthStatus.CRITICAL:
                            status = HealthStatus.DEGRADED
                        elif metric.status == HealthStatus.WARNING and status == HealthStatus.HEALTHY:
                            status = HealthStatus.WARNING
                
                metrics.append(HealthMetric(
                    name="device_count",
                    value=len(devices),
                    unit="devices",
                    status=HealthStatus.HEALTHY
                ))
                
        except Exception as e:
            error_count += 1
            last_error = str(e)
            status = HealthStatus.CRITICAL
            logger.error(f"Photonic hardware health check failed: {e}")
        
        return ComponentHealth(
            component_type=ComponentType.PHOTONIC,
            component_id="photonic_system",
            status=status,
            metrics=metrics,
            last_check=time.time(),
            error_count=error_count,
            last_error=last_error
        )
    
    def _check_device_health(self, device) -> List[HealthMetric]:
        """Check health of individual photonic device."""
        metrics = []
        
        # Device availability
        availability_status = HealthStatus.HEALTHY if device.is_available else HealthStatus.CRITICAL
        metrics.append(HealthMetric(
            name=f"device_availability_{device.device_id}",
            value=1.0 if device.is_available else 0.0,
            unit="boolean",
            status=availability_status,
            metadata={"device_id": device.device_id, "device_type": device.device_type}
        ))
        
        # Temperature monitoring
        if device.temperature is not None:
            temp_status = HealthStatus.HEALTHY
            if device.temperature > 80.0:
                temp_status = HealthStatus.CRITICAL
            elif device.temperature > 70.0:
                temp_status = HealthStatus.DEGRADED
            elif device.temperature > 60.0:
                temp_status = HealthStatus.WARNING
            
            metrics.append(HealthMetric(
                name=f"temperature_{device.device_id}",
                value=device.temperature,
                unit="celsius",
                status=temp_status,
                threshold_warning=60.0,
                threshold_critical=80.0,
                metadata={"device_id": device.device_id}
            ))
        
        # Optical power
        if hasattr(device, 'current_optical_power'):
            power = getattr(device, 'current_optical_power', 0.0)
            max_power = device.max_optical_power
            power_ratio = power / max_power if max_power > 0 else 0.0
            
            power_status = HealthStatus.HEALTHY
            if power_ratio > 0.9:
                power_status = HealthStatus.CRITICAL
            elif power_ratio > 0.8:
                power_status = HealthStatus.WARNING
            
            metrics.append(HealthMetric(
                name=f"optical_power_ratio_{device.device_id}",
                value=power_ratio,
                unit="ratio",
                status=power_status,
                threshold_warning=0.8,
                threshold_critical=0.9,
                metadata={"device_id": device.device_id, "current_power": power, "max_power": max_power}
            ))
        
        # Wavelength utilization
        wavelengths_used = getattr(device, 'wavelengths_used', 0)
        wavelength_ratio = wavelengths_used / device.wavelengths if device.wavelengths > 0 else 0.0
        
        wavelength_status = HealthStatus.HEALTHY
        if wavelength_ratio > 0.95:
            wavelength_status = HealthStatus.WARNING
        
        metrics.append(HealthMetric(
            name=f"wavelength_utilization_{device.device_id}",
            value=wavelength_ratio,
            unit="ratio",
            status=wavelength_status,
            threshold_warning=0.95,
            metadata={"device_id": device.device_id, "used": wavelengths_used, "total": device.wavelengths}
        ))
        
        return metrics


class SystemResourceHealthCheck(HealthCheck):
    """Health check for system resources."""
    
    def __init__(self):
        super().__init__("system_resources", check_interval=15.0)
    
    def check_health(self) -> ComponentHealth:
        """Check system resource health."""
        metrics = []
        status = HealthStatus.HEALTHY
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_status = self._assess_usage_status(cpu_percent, 70.0, 90.0)
            metrics.append(HealthMetric(
                name="cpu_usage",
                value=cpu_percent,
                unit="percent",
                status=cpu_status,
                threshold_warning=70.0,
                threshold_critical=90.0
            ))
            
            # Memory usage
            memory = psutil.virtual_memory()
            mem_status = self._assess_usage_status(memory.percent, 75.0, 90.0)
            metrics.append(HealthMetric(
                name="memory_usage",
                value=memory.percent,
                unit="percent",
                status=mem_status,
                threshold_warning=75.0,
                threshold_critical=90.0,
                metadata={"available_gb": memory.available / (1024**3)}
            ))
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_status = self._assess_usage_status(disk_percent, 80.0, 95.0)
            metrics.append(HealthMetric(
                name="disk_usage",
                value=disk_percent,
                unit="percent",
                status=disk_status,
                threshold_warning=80.0,
                threshold_critical=95.0,
                metadata={"free_gb": disk.free / (1024**3)}
            ))
            
            # Network I/O
            net_io = psutil.net_io_counters()
            if hasattr(self, '_last_net_io'):
                bytes_sent_rate = (net_io.bytes_sent - self._last_net_io.bytes_sent) / self.check_interval
                bytes_recv_rate = (net_io.bytes_recv - self._last_net_io.bytes_recv) / self.check_interval
                
                metrics.append(HealthMetric(
                    name="network_send_rate",
                    value=bytes_sent_rate / (1024**2),  # MB/s
                    unit="MB/s",
                    status=HealthStatus.HEALTHY,
                    metadata={"bytes_per_second": bytes_sent_rate}
                ))
                
                metrics.append(HealthMetric(
                    name="network_recv_rate",
                    value=bytes_recv_rate / (1024**2),  # MB/s
                    unit="MB/s",
                    status=HealthStatus.HEALTHY,
                    metadata={"bytes_per_second": bytes_recv_rate}
                ))
            
            self._last_net_io = net_io
            
            # Determine overall status
            for metric in metrics:
                if metric.status == HealthStatus.CRITICAL:
                    status = HealthStatus.CRITICAL
                elif metric.status == HealthStatus.DEGRADED and status != HealthStatus.CRITICAL:
                    status = HealthStatus.DEGRADED
                elif metric.status == HealthStatus.WARNING and status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING
                    
        except Exception as e:
            status = HealthStatus.CRITICAL
            logger.error(f"System resource health check failed: {e}")
        
        return ComponentHealth(
            component_type=ComponentType.SOFTWARE,
            component_id="system_resources",
            status=status,
            metrics=metrics,
            last_check=time.time()
        )
    
    def _assess_usage_status(self, usage: float, warning_threshold: float, critical_threshold: float) -> HealthStatus:
        """Assess health status based on usage percentage."""
        if usage >= critical_threshold:
            return HealthStatus.CRITICAL
        elif usage >= warning_threshold:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY


class HealthMonitor:
    """Main health monitoring system."""
    
    def __init__(self):
        """Initialize health monitor."""
        self.health_checks: List[HealthCheck] = []
        self.component_health: Dict[str, ComponentHealth] = {}
        self.health_history: deque = deque(maxlen=1000)
        self.monitoring_enabled = True
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self._lock = threading.RLock()
        
        # Register default health checks
        self._register_default_checks()
        
        logger.info("Health monitor initialized")
    
    def _register_default_checks(self) -> None:
        """Register default health checks."""
        self.register_health_check(PhotonicHardwareHealthCheck())
        self.register_health_check(SystemResourceHealthCheck())
    
    def register_health_check(self, health_check: HealthCheck) -> None:
        """Register a health check."""
        with self._lock:
            self.health_checks.append(health_check)
            logger.info(f"Registered health check: {health_check.name}")
    
    def start_monitoring(self, check_interval: float = 10.0) -> None:
        """Start background health monitoring."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.warning("Health monitoring already running")
            return
        
        self.monitoring_enabled = True
        self.stop_event.clear()
        
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(check_interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info(f"Started health monitoring with {check_interval}s interval")
    
    def stop_monitoring(self) -> None:
        """Stop background health monitoring."""
        self.monitoring_enabled = False
        self.stop_event.set()
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        logger.info("Stopped health monitoring")
    
    def _monitoring_loop(self, check_interval: float) -> None:
        """Main monitoring loop."""
        while self.monitoring_enabled and not self.stop_event.wait(check_interval):
            try:
                self.run_health_checks()
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
    
    def run_health_checks(self) -> None:
        """Run all health checks."""
        with self._lock:
            for health_check in self.health_checks:
                if health_check.should_run():
                    try:
                        component_health = health_check.run()
                        self.component_health[component_health.component_id] = component_health
                        
                        # Add to history
                        self.health_history.append({
                            'timestamp': time.time(),
                            'component_id': component_health.component_id,
                            'status': component_health.status.value,
                            'metric_count': len(component_health.metrics)
                        })
                        
                        # Log critical issues
                        if component_health.status == HealthStatus.CRITICAL:
                            logger.critical(
                                f"Critical health issue in {component_health.component_id}: "
                                f"{component_health.last_error or 'Status critical'}"
                            )
                        elif component_health.status == HealthStatus.DEGRADED:
                            logger.warning(
                                f"Degraded health in {component_health.component_id}"
                            )
                            
                    except Exception as e:
                        logger.error(f"Health check '{health_check.name}' failed: {e}")
    
    def get_overall_health(self) -> Tuple[HealthStatus, Dict[str, Any]]:
        """Get overall system health status."""
        with self._lock:
            if not self.component_health:
                return HealthStatus.UNKNOWN, {"reason": "No health data available"}
            
            # Determine overall status
            statuses = [health.status for health in self.component_health.values()]
            
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
                "component_count": len(self.component_health),
                "status_breakdown": {
                    status.value: sum(1 for h in self.component_health.values() if h.status == status)
                    for status in HealthStatus
                },
                "last_check": max(
                    (h.last_check for h in self.component_health.values()), 
                    default=0
                ),
                "critical_components": [
                    comp_id for comp_id, health in self.component_health.items()
                    if health.status == HealthStatus.CRITICAL
                ],
                "total_metrics": sum(
                    len(health.metrics) for health in self.component_health.values()
                )
            }
            
            return overall_status, summary
    
    def get_component_health(self, component_id: str) -> Optional[ComponentHealth]:
        """Get health status of specific component."""
        with self._lock:
            return self.component_health.get(component_id)
    
    def get_health_metrics(
        self, 
        component_id: Optional[str] = None,
        metric_name: Optional[str] = None
    ) -> List[HealthMetric]:
        """Get health metrics with optional filtering."""
        with self._lock:
            metrics = []
            
            components_to_check = (
                [self.component_health[component_id]] if component_id and component_id in self.component_health
                else self.component_health.values()
            )
            
            for component in components_to_check:
                component_metrics = (
                    [m for m in component.metrics if m.name == metric_name] if metric_name
                    else component.metrics
                )
                metrics.extend(component_metrics)
            
            return metrics
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary."""
        overall_status, summary = self.get_overall_health()
        
        with self._lock:
            # Recent health trends
            recent_history = [
                h for h in self.health_history
                if time.time() - h['timestamp'] < 3600  # Last hour
            ]
            
            component_details = {}
            for comp_id, health in self.component_health.items():
                critical_metrics = [
                    m.name for m in health.metrics 
                    if m.status == HealthStatus.CRITICAL
                ]
                warning_metrics = [
                    m.name for m in health.metrics 
                    if m.status == HealthStatus.WARNING
                ]
                
                component_details[comp_id] = {
                    "status": health.status.value,
                    "type": health.component_type.value,
                    "last_check": health.last_check,
                    "error_count": health.error_count,
                    "last_error": health.last_error,
                    "metric_count": len(health.metrics),
                    "critical_metrics": critical_metrics,
                    "warning_metrics": warning_metrics
                }
            
            return {
                "overall": summary,
                "components": component_details,
                "monitoring_active": self.monitoring_enabled,
                "recent_checks": len(recent_history),
                "health_checks_registered": len(self.health_checks)
            }
    
    def add_health_alert(
        self, 
        component_id: str, 
        alert_message: str, 
        severity: HealthStatus = HealthStatus.WARNING
    ) -> None:
        """Add custom health alert."""
        with self._lock:
            if component_id in self.component_health:
                health = self.component_health[component_id]
                if severity == HealthStatus.CRITICAL:
                    health.status = HealthStatus.CRITICAL
                    health.last_error = alert_message
                    health.error_count += 1
                
                # Add alert as metric
                alert_metric = HealthMetric(
                    name="custom_alert",
                    value=1.0,
                    unit="alert",
                    status=severity,
                    metadata={"message": alert_message}
                )
                health.metrics.append(alert_metric)
                
                logger.log(
                    logging.CRITICAL if severity == HealthStatus.CRITICAL else logging.WARNING,
                    f"Health alert for {component_id}: {alert_message}"
                )


# Global health monitor instance
_health_monitor: Optional[HealthMonitor] = None


def get_health_monitor() -> HealthMonitor:
    """Get global health monitor instance."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    return _health_monitor


def start_health_monitoring(check_interval: float = 10.0) -> None:
    """Start global health monitoring."""
    monitor = get_health_monitor()
    monitor.start_monitoring(check_interval)


def stop_health_monitoring() -> None:
    """Stop global health monitoring."""
    monitor = get_health_monitor()
    monitor.stop_monitoring()


def get_system_health() -> Tuple[HealthStatus, Dict[str, Any]]:
    """Get overall system health status."""
    monitor = get_health_monitor()
    return monitor.get_overall_health()