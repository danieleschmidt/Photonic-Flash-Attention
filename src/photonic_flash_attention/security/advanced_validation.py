"""Advanced security validation and threat detection for photonic systems."""

import re
import hashlib
import hmac
import time
import ipaddress
import secrets
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import base64
from collections import defaultdict, deque
import threading

from ..utils.logging import get_logger
from ..utils.exceptions import PhotonicSecurityError, PhotonicConfigurationError


class ThreatLevel(Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEventType(Enum):
    """Types of security events."""
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    INJECTION_ATTEMPT = "injection_attempt"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"
    MALICIOUS_INPUT = "malicious_input"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    CONFIGURATION_TAMPERING = "configuration_tampering"


@dataclass
class SecurityEvent:
    """Security event record."""
    timestamp: float
    event_type: SecurityEventType
    threat_level: ThreatLevel
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    operation: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    blocked: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            'timestamp': self.timestamp,
            'event_type': self.event_type.value,
            'threat_level': self.threat_level.value,
            'source_ip': self.source_ip,
            'user_id': self.user_id,
            'operation': self.operation,
            'details': self.details,
            'blocked': self.blocked
        }


@dataclass
class SecurityPolicy:
    """Security policy configuration."""
    # Input validation
    max_input_size: int = 10_000_000      # 10MB max input
    max_string_length: int = 1000         # Max string length
    allowed_file_extensions: Set[str] = field(default_factory=lambda: {'.json', '.yaml', '.yml', '.txt'})
    
    # Rate limiting
    max_requests_per_minute: int = 100
    max_requests_per_hour: int = 1000
    max_concurrent_requests: int = 10
    
    # Authentication
    require_authentication: bool = True
    session_timeout: int = 3600           # 1 hour
    max_failed_attempts: int = 5
    lockout_duration: int = 900           # 15 minutes
    
    # Access control
    allowed_ip_ranges: List[str] = field(default_factory=list)
    blocked_ip_ranges: List[str] = field(default_factory=list)
    require_encryption: bool = True
    
    # Content security
    scan_for_malware: bool = True
    block_executable_content: bool = True
    sanitize_inputs: bool = True


class InputValidator:
    """Advanced input validation with threat detection."""
    
    def __init__(self, policy: Optional[SecurityPolicy] = None):
        self.policy = policy or SecurityPolicy()
        self.logger = get_logger("InputValidator")
        
        # Malicious pattern detection
        self.malicious_patterns = [
            # SQL injection patterns
            r"(?i)(union|select|insert|update|delete|drop|create|alter|exec|execute)\s",
            r"(?i)(or|and)\s+\d+\s*=\s*\d+",
            r"(?i)'(\s*or\s*|\s*and\s*)\s*'",
            
            # XSS patterns
            r"<script[^>]*>.*?</script>",
            r"javascript\s*:",
            r"on\w+\s*=",
            
            # Command injection patterns
            r"[;&|`$(){}[\]\\]",
            r"(?i)(system|exec|eval|shell_exec|passthru)",
            
            # Path traversal
            r"\.\./",
            r"\.\.\\",
            
            # Python injection
            r"(?i)(__import__|exec|eval|compile|open|file)",
            r"(?i)(import\s+os|import\s+sys|import\s+subprocess)",
        ]
        
        self.compiled_patterns = [re.compile(pattern) for pattern in self.malicious_patterns]
    
    def validate_string(self, value: str, field_name: str = "input") -> str:
        """
        Validate and sanitize string input.
        
        Args:
            value: String to validate
            field_name: Name of field for error reporting
            
        Returns:
            Sanitized string
            
        Raises:
            PhotonicSecurityError: If input is malicious or invalid
        """
        if not isinstance(value, str):
            raise PhotonicSecurityError(f"Expected string for {field_name}, got {type(value)}")
        
        # Length check
        if len(value) > self.policy.max_string_length:
            raise PhotonicSecurityError(
                f"{field_name} exceeds maximum length: {len(value)} > {self.policy.max_string_length}"
            )
        
        # Malicious pattern detection
        for i, pattern in enumerate(self.compiled_patterns):
            if pattern.search(value):
                self.logger.error(f"Malicious pattern {i} detected in {field_name}: {value[:100]}...")
                raise PhotonicSecurityError(f"Malicious content detected in {field_name}")
        
        # Sanitization
        if self.policy.sanitize_inputs:
            value = self._sanitize_string(value)
        
        return value
    
    def _sanitize_string(self, value: str) -> str:
        """Sanitize string input."""
        # Remove null bytes
        value = value.replace('\x00', '')
        
        # Normalize whitespace
        value = re.sub(r'\s+', ' ', value)
        
        # Remove control characters except tab, newline, carriage return
        value = ''.join(char for char in value 
                       if ord(char) >= 32 or char in '\t\n\r')
        
        return value.strip()
    
    def validate_numeric(
        self, 
        value: Union[int, float], 
        field_name: str = "input",
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> Union[int, float]:
        """Validate numeric input."""
        if not isinstance(value, (int, float)):
            raise PhotonicSecurityError(f"Expected numeric value for {field_name}, got {type(value)}")
        
        if min_value is not None and value < min_value:
            raise PhotonicSecurityError(f"{field_name} below minimum: {value} < {min_value}")
        
        if max_value is not None and value > max_value:
            raise PhotonicSecurityError(f"{field_name} above maximum: {value} > {max_value}")
        
        return value
    
    def validate_dict(self, value: Dict[str, Any], field_name: str = "input") -> Dict[str, Any]:
        """Validate dictionary input."""
        if not isinstance(value, dict):
            raise PhotonicSecurityError(f"Expected dictionary for {field_name}, got {type(value)}")
        
        # Check size
        serialized = json.dumps(value)
        if len(serialized) > self.policy.max_input_size:
            raise PhotonicSecurityError(f"{field_name} too large: {len(serialized)} bytes")
        
        # Recursively validate all string values
        return self._validate_dict_recursive(value, field_name)
    
    def _validate_dict_recursive(self, value: Any, path: str) -> Any:
        """Recursively validate dictionary contents."""
        if isinstance(value, str):
            return self.validate_string(value, path)
        elif isinstance(value, dict):
            return {k: self._validate_dict_recursive(v, f"{path}.{k}") for k, v in value.items()}
        elif isinstance(value, list):
            return [self._validate_dict_recursive(item, f"{path}[{i}]") for i, item in enumerate(value)]
        else:
            return value
    
    def validate_file_path(self, path: str, field_name: str = "file_path") -> str:
        """Validate file path for security."""
        path = self.validate_string(path, field_name)
        
        # Check for path traversal
        if '..' in path or path.startswith('/'):
            raise PhotonicSecurityError(f"Path traversal detected in {field_name}: {path}")
        
        # Check file extension
        if '.' in path:
            extension = '.' + path.split('.')[-1].lower()
            if extension not in self.policy.allowed_file_extensions:
                raise PhotonicSecurityError(f"Disallowed file extension in {field_name}: {extension}")
        
        return path
    
    def validate_ip_address(self, ip_str: str, field_name: str = "ip_address") -> str:
        """Validate IP address."""
        ip_str = self.validate_string(ip_str, field_name)
        
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            raise PhotonicSecurityError(f"Invalid IP address in {field_name}: {ip_str}")
        
        # Check against blocked ranges
        for blocked_range in self.policy.blocked_ip_ranges:
            try:
                if ip in ipaddress.ip_network(blocked_range):
                    raise PhotonicSecurityError(f"IP address {ip_str} is in blocked range {blocked_range}")
            except ValueError:
                continue
        
        # Check against allowed ranges (if specified)
        if self.policy.allowed_ip_ranges:
            allowed = False
            for allowed_range in self.policy.allowed_ip_ranges:
                try:
                    if ip in ipaddress.ip_network(allowed_range):
                        allowed = True
                        break
                except ValueError:
                    continue
            
            if not allowed:
                raise PhotonicSecurityError(f"IP address {ip_str} not in allowed ranges")
        
        return ip_str


class RateLimiter:
    """Rate limiting with sliding window algorithm."""
    
    def __init__(self, policy: Optional[SecurityPolicy] = None):
        self.policy = policy or SecurityPolicy()
        self.logger = get_logger("RateLimiter")
        
        # Rate limiting state
        self.request_windows: Dict[str, deque] = defaultdict(lambda: deque())
        self.blocked_ips: Dict[str, float] = {}
        self._lock = threading.RLock()
    
    def check_rate_limit(self, client_id: str, operation: str = "default") -> bool:
        """
        Check if request is within rate limits.
        
        Args:
            client_id: Client identifier (IP, user ID, etc.)
            operation: Operation being performed
            
        Returns:
            True if request is allowed, False if rate limited
        """
        with self._lock:
            current_time = time.time()
            key = f"{client_id}:{operation}"
            
            # Check if client is temporarily blocked
            if client_id in self.blocked_ips:
                if current_time < self.blocked_ips[client_id]:
                    self.logger.warning(f"Request blocked - client {client_id} is rate limited")
                    return False
                else:
                    # Unblock client
                    del self.blocked_ips[client_id]
            
            # Get request window for this client
            window = self.request_windows[key]
            
            # Remove old requests (outside sliding window)
            minute_ago = current_time - 60
            while window and window[0] < minute_ago:
                window.popleft()
            
            # Check per-minute limit
            if len(window) >= self.policy.max_requests_per_minute:
                self._block_client(client_id, current_time)
                return False
            
            # Check hourly limit (approximate with longer window)
            hour_ago = current_time - 3600
            hourly_requests = sum(1 for timestamp in window if timestamp > hour_ago)
            
            if hourly_requests >= self.policy.max_requests_per_hour:
                self._block_client(client_id, current_time)
                return False
            
            # Add current request to window
            window.append(current_time)
            return True
    
    def _block_client(self, client_id: str, current_time: float) -> None:
        """Block client for rate limit violation."""
        block_until = current_time + 300  # 5 minutes
        self.blocked_ips[client_id] = block_until
        
        self.logger.warning(f"Rate limit exceeded for client {client_id}, blocked until {block_until}")
    
    def get_client_stats(self, client_id: str) -> Dict[str, Any]:
        """Get rate limiting statistics for client."""
        with self._lock:
            current_time = time.time()
            stats = {'client_id': client_id}
            
            # Check if blocked
            if client_id in self.blocked_ips:
                stats['blocked'] = True
                stats['blocked_until'] = self.blocked_ips[client_id]
            else:
                stats['blocked'] = False
            
            # Count recent requests
            minute_ago = current_time - 60
            hour_ago = current_time - 3600
            
            minute_requests = 0
            hour_requests = 0
            
            for key, window in self.request_windows.items():
                if key.startswith(f"{client_id}:"):
                    minute_requests += sum(1 for timestamp in window if timestamp > minute_ago)
                    hour_requests += sum(1 for timestamp in window if timestamp > hour_ago)
            
            stats['requests_last_minute'] = minute_requests
            stats['requests_last_hour'] = hour_requests
            stats['rate_limit_minute'] = self.policy.max_requests_per_minute
            stats['rate_limit_hour'] = self.policy.max_requests_per_hour
            
            return stats


class SecurityAuditor:
    """Security event monitoring and anomaly detection."""
    
    def __init__(self, policy: Optional[SecurityPolicy] = None):
        self.policy = policy or SecurityPolicy()
        self.logger = get_logger("SecurityAuditor")
        
        # Security event tracking
        self.security_events: deque = deque(maxlen=10000)
        self.threat_indicators: Dict[str, int] = defaultdict(int)
        self._lock = threading.RLock()
        
        # Anomaly detection thresholds
        self.anomaly_thresholds = {
            'requests_per_minute': 50,
            'failed_authentications': 10,
            'error_rate': 0.1,
            'unusual_operations': 5,
        }
    
    def log_security_event(self, event: SecurityEvent) -> None:
        """Log a security event for monitoring."""
        with self._lock:
            self.security_events.append(event)
            
            # Update threat indicators
            threat_key = f"{event.event_type.value}:{event.source_ip or 'unknown'}"
            self.threat_indicators[threat_key] += 1
            
            # Check for anomalies
            self._check_anomalies(event)
            
            # Log event
            self.logger.warning(f"Security event: {event.to_dict()}")
    
    def _check_anomalies(self, event: SecurityEvent) -> None:
        """Check for anomalous security patterns."""
        current_time = time.time()
        recent_events = [e for e in self.security_events 
                        if current_time - e.timestamp < 300]  # Last 5 minutes
        
        # Check for rapid repeated events from same source
        if event.source_ip:
            same_source_events = [e for e in recent_events 
                                if e.source_ip == event.source_ip]
            
            if len(same_source_events) > 10:  # More than 10 events in 5 minutes
                self._raise_anomaly_alert(
                    f"Rapid repeated security events from {event.source_ip}",
                    ThreatLevel.HIGH
                )
        
        # Check for multiple event types from same source
        if event.source_ip:
            event_types = set(e.event_type for e in same_source_events)
            if len(event_types) > 3:  # Multiple attack vectors
                self._raise_anomaly_alert(
                    f"Multiple attack vectors from {event.source_ip}",
                    ThreatLevel.CRITICAL
                )
    
    def _raise_anomaly_alert(self, message: str, threat_level: ThreatLevel) -> None:
        """Raise anomaly alert."""
        anomaly_event = SecurityEvent(
            timestamp=time.time(),
            event_type=SecurityEventType.ANOMALOUS_BEHAVIOR,
            threat_level=threat_level,
            details={'message': message}
        )
        
        self.security_events.append(anomaly_event)
        self.logger.critical(f"SECURITY ANOMALY: {message}")
    
    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get security summary for specified time period."""
        with self._lock:
            current_time = time.time()
            cutoff_time = current_time - (hours * 3600)
            
            recent_events = [e for e in self.security_events 
                           if e.timestamp > cutoff_time]
            
            # Count events by type and threat level
            event_counts = defaultdict(int)
            threat_counts = defaultdict(int)
            source_counts = defaultdict(int)
            
            for event in recent_events:
                event_counts[event.event_type.value] += 1
                threat_counts[event.threat_level.value] += 1
                if event.source_ip:
                    source_counts[event.source_ip] += 1
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(recent_events)
            
            return {
                'time_period_hours': hours,
                'total_events': len(recent_events),
                'event_counts': dict(event_counts),
                'threat_level_counts': dict(threat_counts),
                'top_sources': dict(sorted(source_counts.items(), 
                                         key=lambda x: x[1], reverse=True)[:10]),
                'risk_score': risk_score,
                'recent_critical_events': [
                    e.to_dict() for e in recent_events 
                    if e.threat_level == ThreatLevel.CRITICAL
                ][-10:]  # Last 10 critical events
            }
    
    def _calculate_risk_score(self, events: List[SecurityEvent]) -> float:
        """Calculate security risk score (0-100)."""
        if not events:
            return 0.0
        
        score = 0.0
        
        # Weight by threat level
        for event in events:
            if event.threat_level == ThreatLevel.LOW:
                score += 1
            elif event.threat_level == ThreatLevel.MEDIUM:
                score += 3
            elif event.threat_level == ThreatLevel.HIGH:
                score += 8
            elif event.threat_level == ThreatLevel.CRITICAL:
                score += 20
        
        # Normalize by time and cap at 100
        risk_score = min(100.0, score / max(1, len(events)) * 10)
        return risk_score


class SecureConfiguration:
    """Secure configuration management with validation."""
    
    def __init__(self):
        self.logger = get_logger("SecureConfiguration")
        self.validator = InputValidator()
        self._config_hash = None
        self._last_validation = 0
    
    def validate_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration for security issues.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Validated and sanitized configuration
            
        Raises:
            PhotonicSecurityError: If configuration has security issues
        """
        validated_config = {}
        
        for key, value in config.items():
            # Validate key
            safe_key = self.validator.validate_string(key, f"config_key_{key}")
            
            # Validate value based on type and key
            if key.endswith('_password') or key.endswith('_secret'):
                # Handle sensitive values
                validated_config[safe_key] = self._validate_sensitive_value(value, key)
            elif key.endswith('_path') or key.endswith('_file'):
                # Handle file paths
                validated_config[safe_key] = self.validator.validate_file_path(str(value), key)
            elif isinstance(value, str):
                validated_config[safe_key] = self.validator.validate_string(value, key)
            elif isinstance(value, dict):
                validated_config[safe_key] = self.validator.validate_dict(value, key)
            elif isinstance(value, (int, float)):
                validated_config[safe_key] = self.validator.validate_numeric(value, key)
            else:
                # For other types, perform basic validation
                validated_config[safe_key] = value
        
        # Calculate configuration hash for integrity checking
        self._config_hash = self._calculate_config_hash(validated_config)
        self._last_validation = time.time()
        
        return validated_config
    
    def _validate_sensitive_value(self, value: str, key: str) -> str:
        """Validate sensitive configuration values."""
        if not isinstance(value, str):
            raise PhotonicSecurityError(f"Sensitive value {key} must be string")
        
        # Check minimum length for security
        if len(value) < 8:
            raise PhotonicSecurityError(f"Sensitive value {key} too short (minimum 8 characters)")
        
        # Check for common weak values
        weak_values = ['password', '123456', 'admin', 'default', 'changeme']
        if value.lower() in weak_values:
            raise PhotonicSecurityError(f"Weak sensitive value detected for {key}")
        
        return value
    
    def _calculate_config_hash(self, config: Dict[str, Any]) -> str:
        """Calculate hash of configuration for integrity checking."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def verify_config_integrity(self, config: Dict[str, Any]) -> bool:
        """Verify configuration hasn't been tampered with."""
        if self._config_hash is None:
            return False
        
        current_hash = self._calculate_config_hash(config)
        return current_hash == self._config_hash


class SecurityManager:
    """Centralized security management system."""
    
    def __init__(self, policy: Optional[SecurityPolicy] = None):
        self.policy = policy or SecurityPolicy()
        self.validator = InputValidator(self.policy)
        self.rate_limiter = RateLimiter(self.policy)
        self.auditor = SecurityAuditor(self.policy)
        self.config_manager = SecureConfiguration()
        
        self.logger = get_logger("SecurityManager")
        self._lock = threading.RLock()
        
        # Active threats tracking
        self.active_threats: Dict[str, SecurityEvent] = {}
        self.blocked_sources: Set[str] = set()
    
    def validate_request(
        self, 
        client_id: str,
        operation: str,
        data: Any,
        source_ip: Optional[str] = None
    ) -> Any:
        """
        Comprehensive request validation.
        
        Args:
            client_id: Client identifier
            operation: Operation being performed  
            data: Request data
            source_ip: Source IP address
            
        Returns:
            Validated data
            
        Raises:
            PhotonicSecurityError: If request fails security validation
        """
        with self._lock:
            # Check if source is blocked
            if source_ip and source_ip in self.blocked_sources:
                self._log_security_event(
                    SecurityEventType.UNAUTHORIZED_ACCESS,
                    ThreatLevel.HIGH,
                    source_ip=source_ip,
                    operation=operation,
                    details={'reason': 'blocked_source'}
                )
                raise PhotonicSecurityError(f"Access denied from blocked source: {source_ip}")
            
            # Rate limiting
            if not self.rate_limiter.check_rate_limit(client_id, operation):
                self._log_security_event(
                    SecurityEventType.RATE_LIMIT_EXCEEDED,
                    ThreatLevel.MEDIUM,
                    source_ip=source_ip,
                    operation=operation,
                    details={'client_id': client_id}
                )
                raise PhotonicSecurityError("Rate limit exceeded")
            
            # Input validation
            try:
                if isinstance(data, str):
                    validated_data = self.validator.validate_string(data, f"{operation}_data")
                elif isinstance(data, dict):
                    validated_data = self.validator.validate_dict(data, f"{operation}_data")
                elif isinstance(data, (int, float)):
                    validated_data = self.validator.validate_numeric(data, f"{operation}_data")
                else:
                    validated_data = data
                
            except PhotonicSecurityError as e:
                self._log_security_event(
                    SecurityEventType.MALICIOUS_INPUT,
                    ThreatLevel.HIGH,
                    source_ip=source_ip,
                    operation=operation,
                    details={'validation_error': str(e)}
                )
                raise
            
            return validated_data
    
    def _log_security_event(
        self,
        event_type: SecurityEventType,
        threat_level: ThreatLevel,
        source_ip: Optional[str] = None,
        user_id: Optional[str] = None,
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log security event."""
        event = SecurityEvent(
            timestamp=time.time(),
            event_type=event_type,
            threat_level=threat_level,
            source_ip=source_ip,
            user_id=user_id,
            operation=operation,
            details=details or {}
        )
        
        self.auditor.log_security_event(event)
        
        # Handle high/critical threats
        if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            self._handle_high_threat(event)
    
    def _handle_high_threat(self, event: SecurityEvent) -> None:
        """Handle high-severity security threats."""
        # Block source IP for critical threats
        if (event.threat_level == ThreatLevel.CRITICAL and 
            event.source_ip and 
            event.source_ip not in self.blocked_sources):
            
            self.blocked_sources.add(event.source_ip)
            self.logger.critical(f"BLOCKED IP {event.source_ip} due to critical security threat")
        
        # Track active threats
        threat_key = f"{event.event_type.value}:{event.source_ip or 'unknown'}"
        self.active_threats[threat_key] = event
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        with self._lock:
            return {
                'policy': {
                    'max_input_size': self.policy.max_input_size,
                    'max_requests_per_minute': self.policy.max_requests_per_minute,
                    'require_authentication': self.policy.require_authentication,
                    'require_encryption': self.policy.require_encryption,
                },
                'threats': {
                    'active_count': len(self.active_threats),
                    'blocked_sources': len(self.blocked_sources),
                    'recent_summary': self.auditor.get_security_summary(hours=1),
                },
                'rate_limiting': {
                    'total_clients': len(self.rate_limiter.request_windows),
                    'blocked_clients': len(self.rate_limiter.blocked_ips),
                },
                'timestamp': time.time()
            }
    
    def emergency_lockdown(self, reason: str) -> None:
        """Activate emergency security lockdown."""
        self.logger.critical(f"EMERGENCY SECURITY LOCKDOWN: {reason}")
        
        # Block all new requests by setting very restrictive limits
        self.policy.max_requests_per_minute = 0
        self.policy.max_concurrent_requests = 0
        
        # Log emergency event
        self._log_security_event(
            SecurityEventType.ANOMALOUS_BEHAVIOR,
            ThreatLevel.CRITICAL,
            operation="emergency_lockdown",
            details={'reason': reason}
        )


# Global security manager instance
_global_security_manager = None

def get_security_manager() -> SecurityManager:
    """Get global security manager instance."""
    global _global_security_manager
    if _global_security_manager is None:
        _global_security_manager = SecurityManager()
    return _global_security_manager


def security_validated(operation_name: str):
    """
    Decorator to add security validation to functions.
    
    Args:
        operation_name: Name of operation for security logging
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            manager = get_security_manager()
            
            # Extract client information from function arguments/context
            client_id = "unknown"
            source_ip = None
            
            # Try to extract from common parameter names
            if 'client_id' in kwargs:
                client_id = kwargs['client_id']
            if 'source_ip' in kwargs:
                source_ip = kwargs['source_ip']
            
            # Validate first argument as data if present
            data = args[0] if args else None
            if data is not None:
                validated_data = manager.validate_request(
                    client_id, operation_name, data, source_ip
                )
                args = (validated_data,) + args[1:]
            
            return func(*args, **kwargs)
        return wrapper
    return decorator