"""Advanced thermal monitoring and protection for photonic devices."""

import time
import threading
import numpy as np
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import warnings

from ..utils.logging import get_logger
from ..utils.exceptions import PhotonicThermalError, PhotonicHardwareError
from ..config import get_config


class ThermalState(Enum):
    """Thermal protection states."""
    NORMAL = "normal"           # Normal operating temperature
    WARNING = "warning"         # Elevated temperature, monitoring
    THROTTLING = "throttling"   # Reducing performance to cool down
    CRITICAL = "critical"       # Emergency shutdown required
    EMERGENCY = "emergency"     # Immediate shutdown


@dataclass
class ThermalThresholds:
    """Temperature thresholds for thermal management."""
    normal_max: float = 65.0         # °C - Normal operation limit
    warning_temp: float = 75.0       # °C - Issue warning
    throttle_temp: float = 80.0      # °C - Start throttling
    critical_temp: float = 85.0      # °C - Emergency measures
    emergency_temp: float = 90.0     # °C - Immediate shutdown
    
    # Hysteresis to prevent oscillation
    hysteresis: float = 5.0          # °C - Temperature drop needed to transition down
    
    def get_state(self, temperature: float, current_state: ThermalState) -> ThermalState:
        """Determine thermal state based on temperature and current state."""
        # Emergency shutdown
        if temperature >= self.emergency_temp:
            return ThermalState.EMERGENCY
        
        # Critical state
        if temperature >= self.critical_temp:
            return ThermalState.CRITICAL
        
        # Apply hysteresis for downward transitions
        if current_state == ThermalState.CRITICAL:
            if temperature > self.critical_temp - self.hysteresis:
                return ThermalState.CRITICAL
            
        if current_state == ThermalState.THROTTLING:
            if temperature > self.throttle_temp - self.hysteresis:
                return ThermalState.THROTTLING
        
        if current_state == ThermalState.WARNING:
            if temperature > self.warning_temp - self.hysteresis:
                return ThermalState.WARNING
        
        # Normal upward transitions
        if temperature >= self.throttle_temp:
            return ThermalState.THROTTLING
        elif temperature >= self.warning_temp:
            return ThermalState.WARNING
        else:
            return ThermalState.NORMAL


@dataclass
class ThermalReading:
    """Single temperature reading with metadata."""
    timestamp: float
    temperature: float
    sensor_id: str
    device_id: str
    location: Optional[str] = None
    
    def is_valid(self) -> bool:
        """Check if reading is valid."""
        return (-40.0 <= self.temperature <= 150.0 and 
                self.timestamp > 0 and
                bool(self.sensor_id) and
                bool(self.device_id))


class ThermalSensor:
    """
    Individual thermal sensor with calibration and filtering.
    """
    
    def __init__(
        self,
        sensor_id: str,
        device_id: str,
        location: Optional[str] = None,
        calibration_offset: float = 0.0,
        calibration_gain: float = 1.0,
        filter_window: int = 5
    ):
        self.sensor_id = sensor_id
        self.device_id = device_id
        self.location = location
        self.calibration_offset = calibration_offset
        self.calibration_gain = calibration_gain
        
        # Moving average filter
        self.filter_window = filter_window
        self._recent_readings = deque(maxlen=filter_window)
        
        # Statistics
        self.total_readings = 0
        self.invalid_readings = 0
        self.last_reading_time = 0.0
        
        self.logger = get_logger(f"ThermalSensor.{sensor_id}")
    
    def add_reading(self, raw_temperature: float) -> Optional[ThermalReading]:
        """
        Add new temperature reading with calibration and filtering.
        
        Args:
            raw_temperature: Raw temperature from sensor
            
        Returns:
            Processed thermal reading or None if invalid
        """
        timestamp = time.time()
        self.total_readings += 1
        
        # Apply calibration
        calibrated_temp = (raw_temperature * self.calibration_gain) + self.calibration_offset
        
        # Create reading
        reading = ThermalReading(
            timestamp=timestamp,
            temperature=calibrated_temp,
            sensor_id=self.sensor_id,
            device_id=self.device_id,
            location=self.location
        )
        
        # Validate reading
        if not reading.is_valid():
            self.invalid_readings += 1
            self.logger.warning(f"Invalid temperature reading: {raw_temperature}°C")
            return None
        
        # Add to filter
        self._recent_readings.append(calibrated_temp)
        self.last_reading_time = timestamp
        
        # Apply moving average filter
        if len(self._recent_readings) >= 3:  # Need minimum readings for filtering
            filtered_temp = np.median(list(self._recent_readings))
            reading.temperature = filtered_temp
        
        return reading
    
    def get_filtered_temperature(self) -> Optional[float]:
        """Get current filtered temperature."""
        if not self._recent_readings:
            return None
        return np.median(list(self._recent_readings))
    
    def get_temperature_trend(self) -> Optional[float]:
        """Get temperature trend (°C/second)."""
        if len(self._recent_readings) < 3:
            return None
        
        temps = list(self._recent_readings)
        times = np.linspace(0, self.filter_window, len(temps))
        
        # Simple linear regression for trend
        if len(temps) > 1:
            coeffs = np.polyfit(times, temps, 1)
            return coeffs[0]  # Slope = °C per time unit
        
        return 0.0
    
    def is_healthy(self) -> bool:
        """Check if sensor is healthy."""
        if self.total_readings == 0:
            return True  # No readings yet
        
        # Check error rate
        error_rate = self.invalid_readings / self.total_readings
        if error_rate > 0.1:  # More than 10% invalid readings
            return False
        
        # Check if we're getting recent readings
        if time.time() - self.last_reading_time > 60.0:  # No reading in 60s
            return False
        
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get sensor status information."""
        return {
            'sensor_id': self.sensor_id,
            'device_id': self.device_id,
            'location': self.location,
            'current_temperature': self.get_filtered_temperature(),
            'temperature_trend': self.get_temperature_trend(),
            'total_readings': self.total_readings,
            'invalid_readings': self.invalid_readings,
            'error_rate': self.invalid_readings / max(1, self.total_readings),
            'last_reading_time': self.last_reading_time,
            'is_healthy': self.is_healthy(),
            'calibration': {
                'offset': self.calibration_offset,
                'gain': self.calibration_gain
            }
        }


class ThermalController:
    """
    Thermal protection controller with multiple mitigation strategies.
    """
    
    def __init__(
        self,
        device_id: str,
        thresholds: Optional[ThermalThresholds] = None,
        cooling_strategies: Optional[List[Callable]] = None
    ):
        self.device_id = device_id
        self.thresholds = thresholds or ThermalThresholds()
        self.cooling_strategies = cooling_strategies or []
        
        self.current_state = ThermalState.NORMAL
        self.last_state_change = time.time()
        self.throttle_factor = 1.0  # Performance multiplier (0-1)
        
        # Temperature history for analysis
        self.temperature_history = deque(maxlen=1000)
        self.state_history = deque(maxlen=100)
        
        # Protection actions taken
        self.protection_actions = []
        
        self.logger = get_logger(f"ThermalController.{device_id}")
        self._lock = threading.RLock()
    
    def update_temperature(self, temperature: float) -> Dict[str, Any]:
        """
        Update thermal controller with new temperature reading.
        
        Args:
            temperature: Current temperature in Celsius
            
        Returns:
            Control response with actions taken
        """
        with self._lock:
            timestamp = time.time()
            
            # Update history
            self.temperature_history.append((timestamp, temperature))
            
            # Determine new state
            new_state = self.thresholds.get_state(temperature, self.current_state)
            
            # Handle state transitions
            response = self._handle_state_transition(new_state, temperature)
            
            return response
    
    def _handle_state_transition(
        self, 
        new_state: ThermalState, 
        temperature: float
    ) -> Dict[str, Any]:
        """Handle thermal state transitions and take appropriate actions."""
        timestamp = time.time()
        actions_taken = []
        
        if new_state != self.current_state:
            # Log state change
            self.logger.info(
                f"Thermal state change: {self.current_state.value} -> {new_state.value} "
                f"at {temperature:.1f}°C"
            )
            
            self.state_history.append({
                'timestamp': timestamp,
                'from_state': self.current_state.value,
                'to_state': new_state.value,
                'temperature': temperature
            })
            
            self.current_state = new_state
            self.last_state_change = timestamp
        
        # Take actions based on current state
        if new_state == ThermalState.EMERGENCY:
            actions_taken.extend(self._handle_emergency(temperature))
        elif new_state == ThermalState.CRITICAL:
            actions_taken.extend(self._handle_critical(temperature))
        elif new_state == ThermalState.THROTTLING:
            actions_taken.extend(self._handle_throttling(temperature))
        elif new_state == ThermalState.WARNING:
            actions_taken.extend(self._handle_warning(temperature))
        elif new_state == ThermalState.NORMAL:
            actions_taken.extend(self._handle_normal(temperature))
        
        return {
            'device_id': self.device_id,
            'temperature': temperature,
            'thermal_state': new_state.value,
            'throttle_factor': self.throttle_factor,
            'actions_taken': actions_taken,
            'timestamp': timestamp
        }
    
    def _handle_emergency(self, temperature: float) -> List[str]:
        """Handle emergency thermal state."""
        actions = []
        
        # Immediate shutdown
        self.throttle_factor = 0.0
        actions.append("emergency_shutdown")
        
        # Log critical event
        self.logger.critical(f"EMERGENCY THERMAL SHUTDOWN at {temperature:.1f}°C")
        
        # Add to protection actions
        self.protection_actions.append({
            'timestamp': time.time(),
            'action': 'emergency_shutdown',
            'temperature': temperature,
            'reason': 'temperature_exceeded_emergency_threshold'
        })
        
        # Execute cooling strategies
        for strategy in self.cooling_strategies:
            try:
                strategy(temperature, ThermalState.EMERGENCY)
                actions.append(f"cooling_strategy_{strategy.__name__}")
            except Exception as e:
                self.logger.error(f"Cooling strategy failed: {e}")
        
        return actions
    
    def _handle_critical(self, temperature: float) -> List[str]:
        """Handle critical thermal state."""
        actions = []
        
        # Severe throttling
        self.throttle_factor = 0.1  # 10% performance
        actions.append("severe_throttling")
        
        self.logger.error(f"CRITICAL thermal state at {temperature:.1f}°C - severe throttling")
        
        # Emergency cooling
        for strategy in self.cooling_strategies:
            try:
                strategy(temperature, ThermalState.CRITICAL)
                actions.append(f"emergency_cooling_{strategy.__name__}")
            except Exception as e:
                self.logger.error(f"Emergency cooling failed: {e}")
        
        return actions
    
    def _handle_throttling(self, temperature: float) -> List[str]:
        """Handle throttling thermal state."""
        actions = []
        
        # Calculate throttle factor based on temperature
        temp_range = self.thresholds.critical_temp - self.thresholds.throttle_temp
        if temp_range > 0:
            temp_excess = temperature - self.thresholds.throttle_temp
            throttle_ratio = 1.0 - (temp_excess / temp_range) * 0.7  # Up to 70% reduction
            self.throttle_factor = max(0.3, min(1.0, throttle_ratio))
        else:
            self.throttle_factor = 0.5
        
        actions.append(f"throttling_to_{self.throttle_factor:.2f}")
        
        self.logger.warning(
            f"Thermal throttling at {temperature:.1f}°C - "
            f"performance reduced to {self.throttle_factor:.1%}"
        )
        
        # Activate cooling
        for strategy in self.cooling_strategies:
            try:
                strategy(temperature, ThermalState.THROTTLING)
                actions.append(f"cooling_{strategy.__name__}")
            except Exception as e:
                self.logger.warning(f"Cooling strategy failed: {e}")
        
        return actions
    
    def _handle_warning(self, temperature: float) -> List[str]:
        """Handle warning thermal state."""
        actions = []
        
        # Light throttling
        self.throttle_factor = 0.85  # 85% performance
        actions.append("light_throttling")
        
        self.logger.warning(f"Thermal warning at {temperature:.1f}°C")
        
        # Pre-emptive cooling
        for strategy in self.cooling_strategies:
            try:
                strategy(temperature, ThermalState.WARNING)
                actions.append(f"preemptive_cooling_{strategy.__name__}")
            except Exception as e:
                self.logger.warning(f"Cooling strategy failed: {e}")
        
        return actions
    
    def _handle_normal(self, temperature: float) -> List[str]:
        """Handle normal thermal state."""
        actions = []
        
        # Restore full performance
        if self.throttle_factor < 1.0:
            self.throttle_factor = 1.0
            actions.append("performance_restored")
            self.logger.info(f"Thermal state normal at {temperature:.1f}°C - performance restored")
        
        return actions
    
    def get_thermal_trend(self, window_seconds: float = 60.0) -> Optional[float]:
        """Get temperature trend over specified window."""
        if len(self.temperature_history) < 2:
            return None
        
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        # Filter recent readings
        recent_readings = [(t, temp) for t, temp in self.temperature_history 
                          if t >= cutoff_time]
        
        if len(recent_readings) < 2:
            return None
        
        # Calculate trend
        times = np.array([r[0] for r in recent_readings])
        temps = np.array([r[1] for r in recent_readings])
        
        # Linear regression
        coeffs = np.polyfit(times - times[0], temps, 1)
        return coeffs[0]  # °C per second
    
    def predict_thermal_limit(self) -> Optional[float]:
        """Predict time until thermal limit reached."""
        trend = self.get_thermal_trend()
        if not trend or trend <= 0:
            return None
        
        if not self.temperature_history:
            return None
        
        current_temp = self.temperature_history[-1][1]
        temp_to_limit = self.thresholds.critical_temp - current_temp
        
        if temp_to_limit <= 0:
            return 0.0
        
        return temp_to_limit / trend  # Seconds until limit
    
    def get_status(self) -> Dict[str, Any]:
        """Get thermal controller status."""
        with self._lock:
            current_temp = None
            if self.temperature_history:
                current_temp = self.temperature_history[-1][1]
            
            return {
                'device_id': self.device_id,
                'current_temperature': current_temp,
                'thermal_state': self.current_state.value,
                'throttle_factor': self.throttle_factor,
                'thresholds': {
                    'warning': self.thresholds.warning_temp,
                    'throttle': self.thresholds.throttle_temp,
                    'critical': self.thresholds.critical_temp,
                    'emergency': self.thresholds.emergency_temp
                },
                'thermal_trend': self.get_thermal_trend(),
                'time_to_limit': self.predict_thermal_limit(),
                'protection_actions_count': len(self.protection_actions),
                'time_in_current_state': time.time() - self.last_state_change,
                'recent_state_changes': list(self.state_history)[-5:]
            }


class ThermalMonitor:
    """
    Comprehensive thermal monitoring system for photonic devices.
    
    Manages multiple thermal sensors and controllers, provides
    system-wide thermal protection and monitoring.
    """
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.sensors: Dict[str, ThermalSensor] = {}
        self.controllers: Dict[str, ThermalController] = {}
        
        self.logger = get_logger("ThermalMonitor")
        self._lock = threading.RLock()
        self._monitoring_thread = None
        self._stop_monitoring = False
        
        # Global thermal statistics
        self.global_temp_history = deque(maxlen=1000)
        self.alert_history = deque(maxlen=100)
        
        # Callbacks for thermal events
        self.thermal_callbacks: List[Callable] = []
    
    def add_sensor(
        self,
        sensor_id: str,
        device_id: str,
        location: Optional[str] = None,
        calibration_offset: float = 0.0,
        calibration_gain: float = 1.0
    ) -> ThermalSensor:
        """Add a thermal sensor to monitoring."""
        with self._lock:
            sensor = ThermalSensor(
                sensor_id=sensor_id,
                device_id=device_id,
                location=location,
                calibration_offset=calibration_offset,
                calibration_gain=calibration_gain
            )
            
            self.sensors[sensor_id] = sensor
            self.logger.info(f"Added thermal sensor: {sensor_id} for device {device_id}")
            return sensor
    
    def add_controller(
        self,
        device_id: str,
        thresholds: Optional[ThermalThresholds] = None,
        cooling_strategies: Optional[List[Callable]] = None
    ) -> ThermalController:
        """Add a thermal controller for device protection."""
        with self._lock:
            controller = ThermalController(
                device_id=device_id,
                thresholds=thresholds,
                cooling_strategies=cooling_strategies
            )
            
            self.controllers[device_id] = controller
            self.logger.info(f"Added thermal controller for device: {device_id}")
            return controller
    
    def add_thermal_callback(self, callback: Callable) -> None:
        """Add callback for thermal events."""
        self.thermal_callbacks.append(callback)
    
    def update_temperature(self, sensor_id: str, raw_temperature: float) -> Optional[Dict[str, Any]]:
        """Update temperature reading from sensor."""
        with self._lock:
            sensor = self.sensors.get(sensor_id)
            if not sensor:
                self.logger.warning(f"Unknown sensor: {sensor_id}")
                return None
            
            # Process reading
            reading = sensor.add_reading(raw_temperature)
            if not reading:
                return None
            
            # Update global history
            self.global_temp_history.append(reading)
            
            # Update corresponding controller
            controller = self.controllers.get(sensor.device_id)
            if controller:
                response = controller.update_temperature(reading.temperature)
                
                # Check for thermal alerts
                if response['thermal_state'] in ['critical', 'emergency']:
                    self._handle_thermal_alert(response)
                
                return response
            
            return {
                'sensor_id': sensor_id,
                'device_id': sensor.device_id,
                'temperature': reading.temperature,
                'thermal_state': 'unknown',
                'timestamp': reading.timestamp
            }
    
    def _handle_thermal_alert(self, response: Dict[str, Any]) -> None:
        """Handle thermal alerts and notifications."""
        alert = {
            'timestamp': time.time(),
            'device_id': response['device_id'],
            'temperature': response['temperature'],
            'thermal_state': response['thermal_state'],
            'actions_taken': response['actions_taken']
        }
        
        self.alert_history.append(alert)
        
        # Execute callbacks
        for callback in self.thermal_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Thermal callback failed: {e}")
        
        # Log alert
        self.logger.error(
            f"THERMAL ALERT: {response['device_id']} at {response['temperature']:.1f}°C "
            f"in {response['thermal_state']} state"
        )
    
    def start_monitoring(self) -> None:
        """Start continuous thermal monitoring."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self.logger.warning("Thermal monitoring already running")
            return
        
        self._stop_monitoring = False
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="ThermalMonitor",
            daemon=True
        )
        self._monitoring_thread.start()
        self.logger.info("Started thermal monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop thermal monitoring."""
        self._stop_monitoring = True
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
        self.logger.info("Stopped thermal monitoring")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while not self._stop_monitoring:
            try:
                self._monitor_cycle()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring cycle: {e}")
                time.sleep(self.monitoring_interval)
    
    def _monitor_cycle(self) -> None:
        """Single monitoring cycle."""
        with self._lock:
            # Check sensor health
            unhealthy_sensors = []
            for sensor_id, sensor in self.sensors.items():
                if not sensor.is_healthy():
                    unhealthy_sensors.append(sensor_id)
            
            if unhealthy_sensors:
                self.logger.warning(f"Unhealthy sensors detected: {unhealthy_sensors}")
            
            # Check for thermal emergencies
            emergency_devices = []
            for device_id, controller in self.controllers.items():
                if controller.current_state == ThermalState.EMERGENCY:
                    emergency_devices.append(device_id)
            
            if emergency_devices:
                self.logger.critical(f"Devices in thermal emergency: {emergency_devices}")
    
    def get_system_thermal_status(self) -> Dict[str, Any]:
        """Get comprehensive thermal status."""
        with self._lock:
            sensor_statuses = {
                sensor_id: sensor.get_status()
                for sensor_id, sensor in self.sensors.items()
            }
            
            controller_statuses = {
                device_id: controller.get_status()
                for device_id, controller in self.controllers.items()
            }
            
            # Calculate system thermal health
            thermal_health = self._calculate_thermal_health(controller_statuses)
            
            return {
                'thermal_health_score': thermal_health,
                'sensors': sensor_statuses,
                'controllers': controller_statuses,
                'global_statistics': self._get_global_statistics(),
                'recent_alerts': list(self.alert_history)[-10:],
                'monitoring_active': not self._stop_monitoring,
                'timestamp': time.time()
            }
    
    def _calculate_thermal_health(self, controller_statuses: Dict[str, Any]) -> float:
        """Calculate overall thermal health score (0-100)."""
        if not controller_statuses:
            return 100.0
        
        health_score = 100.0
        
        for status in controller_statuses.values():
            state = status['thermal_state']
            temp = status.get('current_temperature', 0)
            
            # Penalize based on thermal state
            if state == 'emergency':
                health_score -= 50
            elif state == 'critical':
                health_score -= 30
            elif state == 'throttling':
                health_score -= 15
            elif state == 'warning':
                health_score -= 5
            
            # Additional penalty for very high temperatures
            if temp > 80:
                health_score -= (temp - 80) * 2
        
        return max(0.0, min(100.0, health_score))
    
    def _get_global_statistics(self) -> Dict[str, Any]:
        """Get global thermal statistics."""
        if not self.global_temp_history:
            return {}
        
        recent_temps = [reading.temperature for reading in self.global_temp_history 
                       if time.time() - reading.timestamp < 300]  # Last 5 minutes
        
        if not recent_temps:
            return {}
        
        return {
            'average_temperature': np.mean(recent_temps),
            'max_temperature': np.max(recent_temps),
            'min_temperature': np.min(recent_temps),
            'temperature_std': np.std(recent_temps),
            'readings_count': len(recent_temps),
            'alert_count': len(self.alert_history)
        }


# Global thermal monitor instance
_global_thermal_monitor = None

def get_thermal_monitor() -> ThermalMonitor:
    """Get global thermal monitor instance."""
    global _global_thermal_monitor
    if _global_thermal_monitor is None:
        _global_thermal_monitor = ThermalMonitor()
    return _global_thermal_monitor


def thermal_protected(thresholds: Optional[ThermalThresholds] = None):
    """
    Decorator to add thermal protection to functions.
    
    Args:
        thresholds: Custom thermal thresholds
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            monitor = get_thermal_monitor()
            
            # Check thermal state before execution
            status = monitor.get_system_thermal_status()
            for controller_status in status['controllers'].values():
                if controller_status['thermal_state'] in ['critical', 'emergency']:
                    raise PhotonicThermalError(
                        f"Operation blocked due to thermal emergency: "
                        f"{controller_status['device_id']} at {controller_status['current_temperature']:.1f}°C",
                        controller_status['current_temperature'],
                        thresholds.critical_temp if thresholds else 85.0,
                        controller_status['device_id']
                    )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator