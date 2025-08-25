"""
Simple security validation system without external dependencies.

Provides essential security features for photonic computing systems using
only standard library components for maximum portability and reliability.
"""

import hashlib
import hmac
import time
import threading
import re
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque

from .logging import get_logger
from .exceptions import PhotonicSecurityError


class SecurityLevel(Enum):
    """Security levels for different operations."""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


class ThreatType(Enum):
    """Types of security threats."""
    OPTICAL_POWER_EXCEED = auto()
    WAVELENGTH_INJECTION = auto()
    THERMAL_ATTACK = auto()
    DATA_EXFILTRATION = auto()
    TIMING_ATTACK = auto()
    SIDE_CHANNEL = auto()
    MALICIOUS_INPUT = auto()
    UNAUTHORIZED_ACCESS = auto()
    RATE_LIMIT_EXCEEDED = auto()


@dataclass
class SecurityEvent:
    """Security event for audit logging."""
    timestamp: float = field(default_factory=time.time)
    event_type: ThreatType = ThreatType.MALICIOUS_INPUT
    severity: SecurityLevel = SecurityLevel.MEDIUM
    source: str = ""
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    mitigation_applied: bool = False


@dataclass
class OpticalSafetyLimits:
    """Optical safety limits and thresholds."""
    max_optical_power_mw: float = 10.0  # Maximum safe optical power in milliwatts
    max_peak_power_mw: float = 50.0     # Peak power limit
    max_average_power_mw: float = 5.0    # Average power limit
    min_wavelength_nm: float = 1260.0    # Minimum safe wavelength
    max_wavelength_nm: float = 1650.0    # Maximum safe wavelength
    max_temperature_c: float = 85.0      # Maximum operating temperature
    
    # Forbidden wavelengths (dangerous ranges)
    forbidden_wavelengths: List[Tuple[float, float]] = field(default_factory=lambda: [
        (1064.0, 1065.0),  # Nd:YAG fundamental - avoid retinal damage
        (532.0, 533.0),    # Green lasers - high visibility risk
        (404.0, 406.0),    # Violet lasers - UV damage risk
    ])


class SimpleSecurityValidator:
    """Simple security validation system using standard library only."""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.MEDIUM):
        self.security_level = security_level
        self.safety_limits = OpticalSafetyLimits()
        self.logger = get_logger(self.__class__.__name__)
        
        # Security state
        self.security_events: deque = deque(maxlen=1000)
        self.threat_counters: Dict[ThreatType, int] = {threat: 0 for threat in ThreatType}
        self.blocked_sources: Set[str] = set()
        self.whitelisted_operations: Set[str] = set()
        
        # Rate limiting
        self.rate_limits: Dict[str, List[float]] = {}
        self.rate_limit_window = 60.0  # seconds
        self.max_requests_per_window = 100
        
        # Thread safety
        self._lock = threading.RLock()
        
        self.logger.info(f"Simple security validator initialized at {security_level.name} level")
    
    def validate_optical_power(self, power_w: float, context: str = "") -> None:
        """
        Validate optical power levels for safety.
        
        Args:
            power_w: Optical power in watts
            context: Context description for logging
            
        Raises:
            PhotonicSecurityError: If power exceeds safety limits
        """
        with self._lock:
            power_mw = power_w * 1000  # Convert to milliwatts
            
            if power_mw > self.safety_limits.max_optical_power_mw:
                self._record_security_event(
                    ThreatType.OPTICAL_POWER_EXCEED,
                    SecurityLevel.CRITICAL,
                    f"Optical power {power_mw:.2f}mW exceeds limit {self.safety_limits.max_optical_power_mw}mW",
                    f"optical_power_violation_{context}",
                    {"power_mw": power_mw, "limit_mw": self.safety_limits.max_optical_power_mw, "context": context}
                )
                raise PhotonicSecurityError(
                    f"CRITICAL: Optical power {power_mw:.2f}mW exceeds safety limit {self.safety_limits.max_optical_power_mw}mW in {context}",
                    violation_type="optical_power_exceed"
                )
            
            if power_mw > self.safety_limits.max_average_power_mw:
                self.logger.warning(f"Optical power {power_mw:.2f}mW approaching limit in {context}")
    
    def validate_wavelength(self, wavelength_m: float, context: str = "") -> None:
        """
        Validate wavelength for safety and security.
        
        Args:
            wavelength_m: Wavelength in meters
            context: Context description
            
        Raises:
            PhotonicSecurityError: If wavelength is unsafe
        """
        wavelength_nm = wavelength_m * 1e9  # Convert to nanometers
        
        # Check basic range
        if (wavelength_nm < self.safety_limits.min_wavelength_nm or 
            wavelength_nm > self.safety_limits.max_wavelength_nm):
            self._record_security_event(
                ThreatType.WAVELENGTH_INJECTION,
                SecurityLevel.HIGH,
                f"Wavelength {wavelength_nm:.1f}nm outside safe range",
                f"wavelength_violation_{context}",
                {"wavelength_nm": wavelength_nm, "context": context}
            )
            raise PhotonicSecurityError(
                f"Wavelength {wavelength_nm:.1f}nm outside safe operating range "
                f"({self.safety_limits.min_wavelength_nm}-{self.safety_limits.max_wavelength_nm}nm)",
                violation_type="unsafe_wavelength"
            )
        
        # Check forbidden wavelengths
        for min_wl, max_wl in self.safety_limits.forbidden_wavelengths:
            if min_wl <= wavelength_nm <= max_wl:
                self._record_security_event(
                    ThreatType.WAVELENGTH_INJECTION,
                    SecurityLevel.CRITICAL,
                    f"Forbidden wavelength {wavelength_nm:.1f}nm detected",
                    f"forbidden_wavelength_{context}",
                    {"wavelength_nm": wavelength_nm, "forbidden_range": (min_wl, max_wl), "context": context}
                )
                raise PhotonicSecurityError(
                    f"FORBIDDEN: Wavelength {wavelength_nm:.1f}nm is in restricted range "
                    f"({min_wl}-{max_wl}nm) - potential safety hazard",
                    violation_type="forbidden_wavelength"
                )
    
    def validate_thermal_conditions(self, temperature_c: float, component: str = "") -> None:
        """
        Validate thermal conditions for safety.
        
        Args:
            temperature_c: Temperature in Celsius
            component: Component name
            
        Raises:
            PhotonicSecurityError: If temperature is unsafe
        """
        if temperature_c > self.safety_limits.max_temperature_c:
            self._record_security_event(
                ThreatType.THERMAL_ATTACK,
                SecurityLevel.CRITICAL,
                f"Temperature {temperature_c:.1f}°C exceeds limit {self.safety_limits.max_temperature_c}°C",
                f"thermal_violation_{component}",
                {"temperature_c": temperature_c, "component": component}
            )
            raise PhotonicSecurityError(
                f"CRITICAL: {component} temperature {temperature_c:.1f}°C exceeds safety limit {self.safety_limits.max_temperature_c}°C",
                violation_type="thermal_exceed"
            )
        
        if temperature_c > self.safety_limits.max_temperature_c - 10:
            self.logger.warning(f"{component} temperature {temperature_c:.1f}°C approaching limit")
    
    def validate_input_data(self, data: Any, operation: str = "") -> Any:
        """
        Validate and sanitize input data for security.
        
        Args:
            data: Input data to validate
            operation: Operation context
            
        Returns:
            Sanitized data
            
        Raises:
            PhotonicSecurityError: If data is malicious
        """
        if self.security_level == SecurityLevel.LOW:
            return data  # Skip validation for low security
        
        # Check for obviously malicious patterns
        if isinstance(data, str):
            return self._sanitize_string_input(data, operation)
        elif isinstance(data, dict):
            return self._sanitize_dict_input(data, operation)
        elif isinstance(data, (list, tuple)):
            return self._sanitize_list_input(data, operation)
        elif isinstance(data, (int, float)):
            return self._sanitize_numeric_input(data, operation)
        else:
            # For other types, perform basic validation
            return self._validate_object_input(data, operation)
    
    def _sanitize_string_input(self, text: str, operation: str) -> str:
        """Sanitize string input for security."""
        # Check length limits
        if len(text) > 10000:  # Prevent memory exhaustion
            self._record_security_event(
                ThreatType.DATA_EXFILTRATION,
                SecurityLevel.HIGH,
                f"Excessively long string ({len(text)} chars) in {operation}",
                f"long_string_{operation}",
                {"length": len(text), "operation": operation}
            )
            raise PhotonicSecurityError(
                f"String too long ({len(text)} characters) in {operation}",
                violation_type="string_too_long"
            )
        
        # Check for injection patterns
        dangerous_patterns = [
            r'__[a-zA-Z_]+__',  # Python dunder methods
            r'eval\s*\(',       # eval() calls
            r'exec\s*\(',       # exec() calls
            r'import\s+',       # import statements
            r'\.\.\/',          # Directory traversal
            r'<script',         # Script injection
            r'javascript:',     # JavaScript URLs
            r'file://',         # File URLs
            r'data://',         # Data URLs
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                self._record_security_event(
                    ThreatType.MALICIOUS_INPUT,
                    SecurityLevel.HIGH,
                    f"Dangerous pattern '{pattern}' detected in input",
                    f"malicious_pattern_{operation}",
                    {"pattern": pattern, "operation": operation, "input_preview": text[:100]}
                )
                raise PhotonicSecurityError(
                    f"Malicious input pattern detected in {operation}: {pattern}",
                    violation_type="input_injection"
                )
        
        # Remove or escape potentially dangerous characters
        sanitized = re.sub(r'[<>"\'\\]', '', text)
        return sanitized
    
    def _sanitize_dict_input(self, data: Dict[str, Any], operation: str) -> Dict[str, Any]:
        """Sanitize dictionary input."""
        if len(data) > 1000:  # Prevent excessive data
            self._record_security_event(
                ThreatType.MALICIOUS_INPUT,
                SecurityLevel.MEDIUM,
                f"Large dictionary ({len(data)} keys) in {operation}",
                f"large_dict_{operation}",
                {"key_count": len(data), "operation": operation}
            )
            raise PhotonicSecurityError(
                f"Dictionary too large ({len(data)} keys) in {operation}",
                violation_type="dict_too_large"
            )
        
        sanitized = {}
        for key, value in data.items():
            # Validate key
            if not isinstance(key, str) or len(key) > 256:
                self._record_security_event(
                    ThreatType.MALICIOUS_INPUT,
                    SecurityLevel.MEDIUM,
                    f"Invalid dictionary key in {operation}",
                    f"invalid_key_{operation}",
                    {"key_type": type(key).__name__, "operation": operation}
                )
                continue
            
            # Recursively sanitize value
            sanitized[key] = self.validate_input_data(value, f"{operation}.{key}")
        
        return sanitized
    
    def _sanitize_list_input(self, data: Union[List, Tuple], operation: str) -> Union[List, Tuple]:
        """Sanitize list/tuple input."""
        if len(data) > 10000:  # Prevent excessive data
            self._record_security_event(
                ThreatType.MALICIOUS_INPUT,
                SecurityLevel.HIGH,
                f"Large list ({len(data)} items) in {operation}",
                f"large_list_{operation}",
                {"item_count": len(data), "operation": operation}
            )
            raise PhotonicSecurityError(
                f"List too large ({len(data)} items) in {operation}",
                violation_type="list_too_large"
            )
        
        sanitized_items = []
        for i, item in enumerate(data):
            sanitized_items.append(self.validate_input_data(item, f"{operation}[{i}]"))
        
        return type(data)(sanitized_items)  # Preserve original type (list or tuple)
    
    def _sanitize_numeric_input(self, data: Union[int, float], operation: str) -> Union[int, float]:
        """Sanitize numeric input."""
        # Check for reasonable ranges
        if isinstance(data, float):
            if abs(data) > 1e10:  # Very large numbers
                self._record_security_event(
                    ThreatType.MALICIOUS_INPUT,
                    SecurityLevel.MEDIUM,
                    f"Extremely large number ({data}) in {operation}",
                    f"large_number_{operation}",
                    {"value": data, "operation": operation}
                )
                # Clamp to reasonable range
                data = max(-1e10, min(1e10, data))
            
            # Check for NaN or infinity
            if not (data == data):  # NaN check
                self._record_security_event(
                    ThreatType.MALICIOUS_INPUT,
                    SecurityLevel.HIGH,
                    f"NaN value in {operation}",
                    f"nan_value_{operation}",
                    {"operation": operation}
                )
                raise PhotonicSecurityError(
                    f"NaN value not allowed in {operation}",
                    violation_type="invalid_number"
                )
            
            if abs(data) == float('inf'):  # Infinity check
                self._record_security_event(
                    ThreatType.MALICIOUS_INPUT,
                    SecurityLevel.HIGH,
                    f"Infinite value in {operation}",
                    f"inf_value_{operation}",
                    {"operation": operation}
                )
                raise PhotonicSecurityError(
                    f"Infinite value not allowed in {operation}",
                    violation_type="invalid_number"
                )
        
        return data
    
    def _validate_object_input(self, data: Any, operation: str) -> Any:
        """Basic validation for other object types."""
        # Check if object has suspicious attributes
        if hasattr(data, '__dict__'):
            attrs = dir(data)
            suspicious_attrs = ['__import__', '__builtins__', '__globals__', '__locals__']
            for attr in suspicious_attrs:
                if attr in attrs:
                    self._record_security_event(
                        ThreatType.MALICIOUS_INPUT,
                        SecurityLevel.HIGH,
                        f"Object with suspicious attribute '{attr}' in {operation}",
                        f"suspicious_object_{operation}",
                        {"attribute": attr, "operation": operation, "type": type(data).__name__}
                    )
                    raise PhotonicSecurityError(
                        f"Object with dangerous attribute '{attr}' not allowed in {operation}",
                        violation_type="dangerous_object"
                    )
        
        return data
    
    def validate_access_permissions(self, operation: str, user_context: Optional[str] = None) -> None:
        """
        Validate access permissions for operation.
        
        Args:
            operation: Operation being requested
            user_context: User context information
            
        Raises:
            PhotonicSecurityError: If access is denied
        """
        # Check if source is blocked
        source_key = user_context or "anonymous"
        if source_key in self.blocked_sources:
            self._record_security_event(
                ThreatType.UNAUTHORIZED_ACCESS,
                SecurityLevel.HIGH,
                f"Blocked source {source_key} attempted {operation}",
                f"blocked_access_{source_key}",
                {"operation": operation, "source": source_key}
            )
            raise PhotonicSecurityError(
                f"Access denied for blocked source: {source_key}",
                violation_type="blocked_source"
            )
        
        # Check operation whitelist for critical security level
        if self.security_level == SecurityLevel.CRITICAL:
            if operation not in self.whitelisted_operations:
                self._record_security_event(
                    ThreatType.UNAUTHORIZED_ACCESS,
                    SecurityLevel.CRITICAL,
                    f"Non-whitelisted operation {operation} attempted",
                    f"non_whitelisted_{operation}",
                    {"operation": operation, "source": source_key}
                )
                raise PhotonicSecurityError(
                    f"Operation {operation} not whitelisted for critical security level",
                    violation_type="operation_not_whitelisted"
                )
    
    def check_rate_limit(self, operation: str, source: str = "unknown") -> bool:
        """
        Check if operation is within rate limits.
        
        Args:
            operation: Operation being performed
            source: Source identifier
            
        Returns:
            True if within limits, False otherwise
        """
        with self._lock:
            current_time = time.time()
            key = f"{operation}:{source}"
            
            # Initialize if not exists
            if key not in self.rate_limits:
                self.rate_limits[key] = []
            
            # Clean old entries
            cutoff_time = current_time - self.rate_limit_window
            self.rate_limits[key] = [t for t in self.rate_limits[key] if t > cutoff_time]
            
            # Check limit
            if len(self.rate_limits[key]) >= self.max_requests_per_window:
                self._record_security_event(
                    ThreatType.RATE_LIMIT_EXCEEDED,
                    SecurityLevel.MEDIUM,
                    f"Rate limit exceeded for {operation} from {source}",
                    f"rate_limit_{operation}_{source}",
                    {"operation": operation, "source": source, "requests": len(self.rate_limits[key])}
                )
                return False
            
            # Record current request
            self.rate_limits[key].append(current_time)
            return True
    
    def _record_security_event(self, threat_type: ThreatType, severity: SecurityLevel,
                              description: str, source: str, metadata: Dict[str, Any]) -> None:
        """Record a security event."""
        event = SecurityEvent(
            event_type=threat_type,
            severity=severity,
            description=description,
            source=source,
            metadata=metadata
        )
        
        self.security_events.append(event)
        self.threat_counters[threat_type] += 1
        
        # Log based on severity
        if severity == SecurityLevel.CRITICAL:
            self.logger.critical(f"SECURITY ALERT: {description}")
        elif severity == SecurityLevel.HIGH:
            self.logger.error(f"Security warning: {description}")
        elif severity == SecurityLevel.MEDIUM:
            self.logger.warning(f"Security notice: {description}")
        else:
            self.logger.debug(f"Security info: {description}")
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        with self._lock:
            recent_events = [e for e in self.security_events 
                           if time.time() - e.timestamp < 3600]  # Last hour
            
            severity_counts = {}
            threat_counts = {}
            
            for event in recent_events:
                severity_counts[event.severity.name] = severity_counts.get(event.severity.name, 0) + 1
                threat_counts[event.event_type.name] = threat_counts.get(event.event_type.name, 0) + 1
            
            return {
                'security_level': self.security_level.name,
                'total_events': len(self.security_events),
                'recent_events_1h': len(recent_events),
                'severity_distribution': severity_counts,
                'threat_distribution': threat_counts,
                'blocked_sources': len(self.blocked_sources),
                'whitelisted_operations': len(self.whitelisted_operations),
                'rate_limit_violations': sum(1 for event in recent_events 
                                           if event.event_type == ThreatType.RATE_LIMIT_EXCEEDED),
                'safety_limits': {
                    'max_optical_power_mw': self.safety_limits.max_optical_power_mw,
                    'max_temperature_c': self.safety_limits.max_temperature_c,
                    'wavelength_range_nm': (self.safety_limits.min_wavelength_nm, 
                                          self.safety_limits.max_wavelength_nm)
                }
            }
    
    def block_source(self, source: str, reason: str = "") -> None:
        """Block a source from accessing the system."""
        with self._lock:
            self.blocked_sources.add(source)
            self.logger.warning(f"Blocked source {source}: {reason}")
    
    def unblock_source(self, source: str) -> None:
        """Unblock a previously blocked source."""
        with self._lock:
            self.blocked_sources.discard(source)
            self.logger.info(f"Unblocked source {source}")
    
    def add_whitelisted_operation(self, operation: str) -> None:
        """Add operation to whitelist."""
        with self._lock:
            self.whitelisted_operations.add(operation)
            self.logger.debug(f"Whitelisted operation: {operation}")


# Global security validator instance
_global_security_validator: Optional[SimpleSecurityValidator] = None


def get_security_validator() -> SimpleSecurityValidator:
    """Get global security validator instance."""
    global _global_security_validator
    if _global_security_validator is None:
        _global_security_validator = SimpleSecurityValidator()
    return _global_security_validator


def set_security_level(level: SecurityLevel) -> None:
    """Set global security level."""
    global _global_security_validator
    if _global_security_validator is None:
        _global_security_validator = SimpleSecurityValidator(level)
    else:
        _global_security_validator.security_level = level


# Security decorators
def require_security_validation(operation: str):
    """Decorator to require security validation for functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            validator = get_security_validator()
            
            # Check rate limits
            if not validator.check_rate_limit(operation):
                raise PhotonicSecurityError(
                    f"Rate limit exceeded for {operation}",
                    violation_type="rate_limit_exceeded"
                )
            
            # Validate input data
            validated_args = []
            for i, arg in enumerate(args):
                validated_args.append(validator.validate_input_data(arg, f"{operation}_arg{i}"))
            
            validated_kwargs = {}
            for key, value in kwargs.items():
                validated_kwargs[key] = validator.validate_input_data(value, f"{operation}.{key}")
            
            # Execute function
            return func(*validated_args, **validated_kwargs)
        
        return wrapper
    return decorator


def validate_optical_safety(max_power_mw: float = 10.0, max_temp_c: float = 85.0):
    """Decorator to validate optical safety parameters."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            validator = get_security_validator()
            
            # Check power if provided
            if 'power' in kwargs:
                validator.validate_optical_power(kwargs['power'], func.__name__)
            elif 'optical_power' in kwargs:
                validator.validate_optical_power(kwargs['optical_power'], func.__name__)
            
            # Check wavelength if provided
            if 'wavelength' in kwargs:
                validator.validate_wavelength(kwargs['wavelength'], func.__name__)
            
            # Check temperature if provided
            if 'temperature' in kwargs:
                validator.validate_thermal_conditions(kwargs['temperature'], func.__name__)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator