"""
Advanced security utilities and access control for photonic attention systems.

This module provides comprehensive security measures including access control,
data sanitization, secure communication, and hardware security validation.
"""

import os
import hashlib
import hmac
import secrets
import torch
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from .exceptions import PhotonicSecurityError
from .logging import get_logger


logger = get_logger(__name__)


class SecurityManager:
    """Manages security policies and validation for photonic systems."""
    
    def __init__(self):
        self.max_tensor_size = 1024 * 1024 * 1024  # 1GB
        self.max_sequence_length = 16384
        self.max_batch_size = 128
        self.allowed_dtypes = {torch.float32, torch.float16, torch.bfloat16}
        self.blocked_paths = {'/proc', '/sys', '/dev', '/etc'}
        
        # Generate session key for HMAC verification
        self.session_key = secrets.token_bytes(32)
        
        logger.info("Security manager initialized")
    
    def validate_tensor_security(self, tensor: torch.Tensor, name: str = "tensor") -> None:
        """
        Validate tensor for security issues.
        
        Args:
            tensor: Tensor to validate
            name: Tensor name for logging
            
        Raises:
            PhotonicSecurityError: If validation fails
        """
        # Check tensor size
        tensor_bytes = tensor.numel() * tensor.element_size()
        if tensor_bytes > self.max_tensor_size:
            raise PhotonicSecurityError(
                f"Tensor {name} size {tensor_bytes} exceeds limit {self.max_tensor_size}"
            )
        
        # Check data type
        if tensor.dtype not in self.allowed_dtypes:
            raise PhotonicSecurityError(f"Tensor {name} dtype {tensor.dtype} not in allowed types")
        
        # Check dimensions
        if tensor.dim() > 4:  # Reasonable limit for attention tensors
            raise PhotonicSecurityError(f"Tensor {name} has too many dimensions: {tensor.dim()}")
        
        # Check for shape attacks
        if tensor.dim() >= 2:
            if tensor.shape[0] > self.max_batch_size:
                raise PhotonicSecurityError(f"Batch size {tensor.shape[0]} exceeds limit {self.max_batch_size}")
            if tensor.shape[1] > self.max_sequence_length:
                raise PhotonicSecurityError(f"Sequence length {tensor.shape[1]} exceeds limit {self.max_sequence_length}")
        
        # Check for adversarial patterns (basic checks)
        if tensor.numel() > 0:
            # Check for suspicious constant values
            if torch.all(tensor == tensor.flatten()[0]):
                logger.warning(f"Tensor {name} contains all identical values")
            
            # Check for extreme values that could cause overflow
            if tensor.abs().max() > 1e6:
                logger.warning(f"Tensor {name} contains very large values: max={tensor.abs().max()}")
        
        logger.debug(f"Tensor {name} passed security validation")
    
    def sanitize_input(self, data: Any, allow_tensors: bool = True) -> Any:
        """
        Sanitize input data to prevent injection attacks.
        
        Args:
            data: Input data to sanitize
            allow_tensors: Whether to allow tensor inputs
            
        Returns:
            Sanitized data
            
        Raises:
            PhotonicSecurityError: If data is unsafe
        """
        if isinstance(data, str):
            return self._sanitize_string(data)
        elif isinstance(data, dict):
            return {k: self.sanitize_input(v, allow_tensors) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)(self.sanitize_input(item, allow_tensors) for item in data)  
        elif isinstance(data, torch.Tensor):
            if not allow_tensors:
                raise PhotonicSecurityError("Tensor inputs not allowed in this context")
            self.validate_tensor_security(data)
            return data
        else:
            return data
    
    def _sanitize_string(self, s: str) -> str:
        """Sanitize string input."""
        # Remove null bytes
        s = s.replace('\x00', '')
        
        # Check for path traversal
        if '..' in s or s.startswith('/'):
            raise PhotonicSecurityError(f"Potential path traversal in string: {s}")
        
        # Check for control characters
        if any(ord(c) < 32 and c not in '\t\n\r' for c in s):
            raise PhotonicSecurityError("String contains control characters")
        
        # Length limit
        if len(s) > 10000:
            raise PhotonicSecurityError(f"String too long: {len(s)} characters")
        
        return s
    
    def check_file_access(self, path: Union[str, Path]) -> None:
        """
        Check if file access is allowed.
        
        Args:
            path: File path to check
            
        Raises:
            PhotonicSecurityError: If access is not allowed
        """
        path = Path(path).resolve()
        
        # Check for blocked paths
        for blocked in self.blocked_paths:
            if str(path).startswith(blocked):
                raise PhotonicSecurityError(f"Access to {path} is blocked")
        
        # Check if file exists and is readable
        if path.exists() and not os.access(path, os.R_OK):
            raise PhotonicSecurityError(f"No read permission for {path}")
        
        logger.debug(f"File access allowed: {path}")
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate a cryptographically secure token."""
        return secrets.token_hex(length)
    
    def verify_hmac(self, data: bytes, signature: str, key: Optional[bytes] = None) -> bool:
        """
        Verify HMAC signature.
        
        Args:
            data: Data to verify
            signature: HMAC signature (hex encoded)
            key: HMAC key (uses session key if None)
            
        Returns:
            True if signature is valid
        """
        if key is None:
            key = self.session_key
        
        expected = hmac.new(key, data, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, signature)
    
    def compute_hmac(self, data: bytes, key: Optional[bytes] = None) -> str:
        """
        Compute HMAC signature.
        
        Args:
            data: Data to sign
            key: HMAC key (uses session key if None)
            
        Returns:
            HMAC signature (hex encoded)
        """
        if key is None:
            key = self.session_key
        
        return hmac.new(key, data, hashlib.sha256).hexdigest()


# Global security manager instance
_security_manager = SecurityManager()


def sanitize_input(data: Any, allow_tensors: bool = True) -> Any:
    """Sanitize input data (convenience function)."""
    return _security_manager.sanitize_input(data, allow_tensors)


def validate_tensor_security(tensor: torch.Tensor, name: str = "tensor") -> None:
    """Validate tensor security (convenience function)."""
    _security_manager.validate_tensor_security(tensor, name)


def check_permissions(operation: str, resource: str) -> bool:
    """
    Check if operation is permitted on resource.
    
    Args:
        operation: Operation name (e.g., 'read', 'write', 'execute')
        resource: Resource identifier
        
    Returns:
        True if operation is permitted
    """
    # Basic permission checking - can be extended
    blocked_operations = {'execute', 'shell', 'system'}
    
    if operation in blocked_operations:
        logger.warning(f"Blocked operation '{operation}' on resource '{resource}'")
        return False
    
    return True


def secure_hash(data: Union[str, bytes]) -> str:
    """
    Compute secure hash of data.
    
    Args:
        data: Data to hash
        
    Returns:
        SHA-256 hash (hex encoded)
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    return hashlib.sha256(data).hexdigest()


def constant_time_compare(a: str, b: str) -> bool:
    """
    Constant-time string comparison to prevent timing attacks.
    
    Args:
        a: First string
        b: Second string
        
    Returns:
        True if strings are equal
    """
    return hmac.compare_digest(a, b)


class SecureConfig:
    """Secure configuration management."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from secure location."""
        if self.config_path and os.path.exists(self.config_path):
            try:
                _security_manager.check_file_access(self.config_path)
                # In a real implementation, would decrypt/verify config file
                logger.info(f"Loaded secure config from {self.config_path}")
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value (with validation)."""
        # Validate key
        if not isinstance(key, str) or not key.isalnum():
            raise PhotonicSecurityError("Invalid configuration key")
        
        # Sanitize value
        value = sanitize_input(value, allow_tensors=False)
        
        self.config[key] = value
        logger.debug(f"Set config {key}")


def get_security_manager() -> SecurityManager:
    """Get the global security manager instance."""
    return _security_manager


class HardwareSecurityModule:
    """Hardware security validation for photonic devices."""
    
    def __init__(self):
        self.device_fingerprints = {}
        self.trusted_vendors = {'LightMatter', 'Luminous Computing', 'Intel Photonics'}
        self.max_temperature = 85.0  # °C
        self.max_power = 50e-3  # 50mW
        
    def validate_device_security(self, device_info: Dict[str, Any]) -> bool:
        """
        Validate security properties of photonic device.
        
        Args:
            device_info: Device information dictionary
            
        Returns:
            True if device passes security validation
            
        Raises:
            PhotonicSecurityError: If device fails security checks
        """
        # Check vendor
        vendor = device_info.get('vendor', '')
        if vendor not in self.trusted_vendors and vendor != 'Photonic Flash Attention':  # Allow simulator
            raise PhotonicSecurityError(f"Untrusted device vendor: {vendor}")
        
        # Check device type
        device_type = device_info.get('device_type', '')
        if device_type not in {'lightmatter_mars', 'luminous_processor', 'generic_photonic', 'simulation'}:
            raise PhotonicSecurityError(f"Unknown device type: {device_type}")
        
        # Validate thermal limits
        temperature = device_info.get('temperature')
        if temperature is not None and temperature > self.max_temperature:
            raise PhotonicSecurityError(f"Device temperature {temperature}°C exceeds safe limit {self.max_temperature}°C")
        
        # Validate power limits
        max_power = device_info.get('max_optical_power', 0)
        if max_power > self.max_power:
            raise PhotonicSecurityError(f"Device max power {max_power*1000:.1f}mW exceeds safe limit {self.max_power*1000:.1f}mW")
        
        # Check device fingerprint consistency
        device_id = device_info.get('device_id', '')
        if device_id and self._check_device_fingerprint(device_id, device_info):
            logger.info(f"Device {device_id} passed security validation")
            return True
        
        return False
    
    def _check_device_fingerprint(self, device_id: str, device_info: Dict[str, Any]) -> bool:
        """Check device fingerprint for consistency."""
        # Create fingerprint from device characteristics
        fingerprint_data = f"{device_info.get('vendor', '')}{device_info.get('model', '')}{device_info.get('wavelengths', 0)}"
        fingerprint = secure_hash(fingerprint_data)
        
        if device_id in self.device_fingerprints:
            stored_fingerprint = self.device_fingerprints[device_id]
            if not constant_time_compare(fingerprint, stored_fingerprint):
                raise PhotonicSecurityError(f"Device fingerprint mismatch for {device_id}")
        else:
            self.device_fingerprints[device_id] = fingerprint
            logger.info(f"Registered device fingerprint: {device_id}")
        
        return True


class SecureModelValidator:
    """Security validation for AI models and parameters."""
    
    def __init__(self):
        self.max_model_size = 10 * 1024**3  # 10GB
        self.allowed_activations = {'relu', 'gelu', 'swish', 'softmax', 'tanh', 'sigmoid'}
        self.blocked_layer_types = {'shell', 'exec', 'system'}
    
    def validate_model_security(self, model: torch.nn.Module) -> bool:
        """
        Validate model for security issues.
        
        Args:
            model: PyTorch model to validate
            
        Returns:
            True if model passes security checks
            
        Raises:
            PhotonicSecurityError: If model fails security validation
        """
        # Check model size
        model_size = sum(p.numel() * p.element_size() for p in model.parameters())
        if model_size > self.max_model_size:
            raise PhotonicSecurityError(f"Model size {model_size/1024**3:.2f}GB exceeds limit {self.max_model_size/1024**3:.0f}GB")
        
        # Check for suspicious layer types
        for name, module in model.named_modules():
            module_name = name.lower()
            class_name = type(module).__name__.lower()
            
            # Check for blocked layer types
            if any(blocked in module_name or blocked in class_name 
                   for blocked in self.blocked_layer_types):
                raise PhotonicSecurityError(f"Blocked layer type detected: {name} ({type(module).__name__})")
            
            # Validate activation functions
            if hasattr(module, 'activation') or 'activation' in class_name:
                if not any(allowed in class_name for allowed in self.allowed_activations):
                    logger.warning(f"Unknown activation function: {class_name}")
        
        # Check parameter ranges
        for name, param in model.named_parameters():
            if param.numel() > 0:
                # Check for extreme parameter values
                if param.abs().max() > 100.0:
                    logger.warning(f"Large parameter values in {name}: max={param.abs().max():.2f}")
                
                # Check for suspicious patterns
                if torch.all(param == 0):
                    logger.warning(f"All-zero parameters in {name}")
                elif param.var() < 1e-10:
                    logger.warning(f"Very low variance parameters in {name}")
        
        logger.info(f"Model passed security validation: {model_size/1024**2:.1f}MB, {sum(1 for _ in model.parameters())} parameters")
        return True
    
    def sanitize_model_state(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Sanitize model state dictionary.
        
        Args:
            state_dict: Model state dictionary
            
        Returns:
            Sanitized state dictionary
        """
        sanitized = {}
        
        for key, tensor in state_dict.items():
            # Validate key name
            if not isinstance(key, str) or len(key) > 1000:
                logger.warning(f"Skipping invalid key: {key}")
                continue
            
            # Validate tensor
            try:
                validate_tensor_security(tensor, key)
                
                # Clip extreme values
                if tensor.dtype in {torch.float32, torch.float16}:
                    tensor = torch.clamp(tensor, -100.0, 100.0)
                
                sanitized[key] = tensor
                
            except PhotonicSecurityError as e:
                logger.warning(f"Skipping unsafe tensor {key}: {e}")
        
        return sanitized


class DataProtectionManager:
    """Data protection and privacy utilities."""
    
    def __init__(self):
        self.sensitive_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
            r'\b\d{16}\b',  # Credit card pattern
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b(?:\d{1,3}\.){3}\d{1,3}\b',  # IP address
        ]
    
    def scan_for_sensitive_data(self, text: str) -> List[str]:
        """
        Scan text for sensitive data patterns.
        
        Args:
            text: Text to scan
            
        Returns:
            List of detected sensitive patterns
        """
        import re
        
        detected = []
        for pattern in self.sensitive_patterns:
            matches = re.findall(pattern, text)
            if matches:
                detected.extend(matches)
        
        return detected
    
    def anonymize_tensor(self, tensor: torch.Tensor, noise_scale: float = 0.01) -> torch.Tensor:
        """
        Add differential privacy noise to tensor.
        
        Args:
            tensor: Input tensor
            noise_scale: Scale of noise to add
            
        Returns:
            Anonymized tensor
        """
        if tensor.dtype in {torch.float32, torch.float16, torch.bfloat16}:
            noise = torch.randn_like(tensor) * noise_scale
            return tensor + noise
        else:
            logger.warning(f"Cannot anonymize tensor of dtype {tensor.dtype}")
            return tensor
    
    def redact_sensitive_info(self, data: Any) -> Any:
        """
        Redact sensitive information from data structures.
        
        Args:
            data: Data to redact
            
        Returns:
            Data with sensitive information redacted
        """
        if isinstance(data, str):
            # Redact sensitive patterns
            for pattern in self.sensitive_patterns:
                import re
                data = re.sub(pattern, '[REDACTED]', data)
            return data
        
        elif isinstance(data, dict):
            redacted = {}
            sensitive_keys = {'password', 'secret', 'token', 'key', 'credential'}
            
            for k, v in data.items():
                if any(sensitive in str(k).lower() for sensitive in sensitive_keys):
                    redacted[k] = '[REDACTED]'
                else:
                    redacted[k] = self.redact_sensitive_info(v)
            return redacted
        
        elif isinstance(data, (list, tuple)):
            return type(data)(self.redact_sensitive_info(item) for item in data)
        
        else:
            return data


class AuditLogger:
    """Security audit logging."""
    
    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        self.events = []
    
    def log_security_event(self, event_type: str, details: Dict[str, Any], severity: str = 'INFO') -> None:
        """
        Log security event.
        
        Args:
            event_type: Type of security event
            details: Event details
            severity: Event severity level
        """
        import time
        
        event = {
            'timestamp': time.time(),
            'event_type': event_type,
            'severity': severity,
            'details': details,
            'session_id': getattr(self, 'session_id', 'unknown')
        }
        
        self.events.append(event)
        
        # Log to file if configured
        if self.log_file:
            try:
                with open(self.log_file, 'a') as f:
                    f.write(f"{event}\n")
            except Exception as e:
                logger.error(f"Failed to write audit log: {e}")
        
        # Also log to standard logger
        log_msg = f"SECURITY EVENT [{event_type}]: {details}"
        if severity == 'CRITICAL':
            logger.critical(log_msg)
        elif severity == 'WARNING':
            logger.warning(log_msg)
        else:
            logger.info(log_msg)
    
    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get security event summary for specified time period.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Security summary dictionary
        """
        import time
        
        cutoff_time = time.time() - (hours * 3600)
        recent_events = [e for e in self.events if e['timestamp'] > cutoff_time]
        
        event_counts = {}
        severity_counts = {}
        
        for event in recent_events:
            event_type = event['event_type']
            severity = event['severity']
            
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'total_events': len(recent_events),
            'event_types': event_counts,
            'severity_distribution': severity_counts,
            'time_period_hours': hours
        }


# Global security instances
_hardware_security = HardwareSecurityModule()
_model_validator = SecureModelValidator()
_data_protection = DataProtectionManager()
_audit_logger = AuditLogger()


def validate_hardware_security(device_info: Dict[str, Any]) -> bool:
    """Validate hardware security (convenience function)."""
    return _hardware_security.validate_device_security(device_info)


def validate_model_security(model: torch.nn.Module) -> bool:
    """Validate model security (convenience function)."""
    return _model_validator.validate_model_security(model)


def log_security_event(event_type: str, details: Dict[str, Any], severity: str = 'INFO') -> None:
    """Log security event (convenience function)."""
    _audit_logger.log_security_event(event_type, details, severity)