"""Security modules for photonic attention system."""

# Import PyTorch-dependent modules conditionally
try:
    from .advanced_validation import (
        SecurityManager, get_security_manager, security_validated,
        InputValidator, RateLimiter, SecurityAuditor, SecureConfiguration,
        ThreatLevel, SecurityEventType, SecurityPolicy
    )
except ImportError:
    # PyTorch not available
    SecurityManager = None
    get_security_manager = None
    security_validated = None
    InputValidator = None
    RateLimiter = None
    SecurityAuditor = None
    SecureConfiguration = None
    ThreatLevel = None
    SecurityEventType = None
    SecurityPolicy = None

__all__ = [
    "SecurityManager",
    "get_security_manager", 
    "security_validated",
    "InputValidator",
    "RateLimiter",
    "SecurityAuditor",
    "SecureConfiguration",
    "ThreatLevel",
    "SecurityEventType",
    "SecurityPolicy",
]