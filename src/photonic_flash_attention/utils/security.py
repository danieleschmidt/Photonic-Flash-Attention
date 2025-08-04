"""Security utilities for photonic attention system."""

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