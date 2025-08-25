"""Utility modules for photonic attention."""

from .logging import get_logger, setup_logging
from .exceptions import PhotonicHardwareError, PhotonicComputationError

# Import PyTorch-dependent modules conditionally with fallbacks
try:
    from .validation import validate_tensor_shape, validate_attention_inputs
except ImportError:
    def validate_tensor_shape(*args, **kwargs):
        pass
    def validate_attention_inputs(*args, **kwargs):
        pass

try:
    from .security import sanitize_input, check_permissions
    from .monitoring import PerformanceMonitor, HealthMonitor
except ImportError:
    # Fallback to simple implementations
    from .simple_security import get_security_validator
    
    def sanitize_input(data, operation="unknown"):
        validator = get_security_validator()
        return validator.validate_input_data(data, operation)
    
    def check_permissions(operation, user_context=None):
        validator = get_security_validator()
        try:
            validator.validate_access_permissions(operation, user_context)
            return True
        except:
            return False
    
    PerformanceMonitor = None
    HealthMonitor = None

__all__ = [
    "get_logger",
    "setup_logging",
    "validate_tensor_shape",
    "validate_attention_inputs",
    "PhotonicHardwareError",
    "PhotonicComputationError",
    "sanitize_input",
    "check_permissions",
    "PerformanceMonitor",
    "HealthMonitor",
]