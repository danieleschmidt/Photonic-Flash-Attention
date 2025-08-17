"""Utility modules for photonic attention."""

from .logging import get_logger, setup_logging
from .exceptions import PhotonicHardwareError, PhotonicComputationError

# Import PyTorch-dependent modules conditionally
try:
    from .validation import validate_tensor_shape, validate_attention_inputs
    from .security import sanitize_input, check_permissions
    from .monitoring import PerformanceMonitor, HealthMonitor
except ImportError:
    # PyTorch not available - provide fallback functions
    def validate_tensor_shape(*args, **kwargs):
        pass
    def validate_attention_inputs(*args, **kwargs):
        pass
    def sanitize_input(*args, **kwargs):
        return args[0] if args else None
    def check_permissions(*args, **kwargs):
        return True
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