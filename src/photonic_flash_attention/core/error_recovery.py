"""Advanced error recovery and fault tolerance for photonic attention."""

import time
import threading
from typing import Dict, Any, Optional, Callable, List, Union, Tuple
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque

from ..utils.exceptions import (
    PhotonicHardwareError, PhotonicComputationError, PhotonicTimeoutError,
    PhotonicThermalError, PhotonicCalibrationError
)
from ..utils.logging import get_logger


logger = get_logger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Error recovery strategies."""
    RETRY = "retry"
    FALLBACK = "fallback"
    DEGRADE = "degrade"
    SHUTDOWN = "shutdown"
    RECALIBRATE = "recalibrate"


@dataclass
class ErrorEvent:
    """Represents an error event."""
    timestamp: float
    error_type: str
    severity: ErrorSeverity
    message: str
    operation: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None


@dataclass 
class RecoveryPolicy:
    """Defines recovery policy for error types."""
    error_pattern: str
    max_retries: int
    retry_delay: float
    backoff_multiplier: float
    max_delay: float
    strategy: RecoveryStrategy
    severity_threshold: ErrorSeverity


class CircuitBreaker:
    """Circuit breaker pattern for photonic operations."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception,
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time before attempting to close circuit
            expected_exception: Type of exception to catch
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self._failure_count = 0
        self._last_failure_time = None
        self._state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.RLock()
        
        logger.info(f"Circuit breaker initialized: threshold={failure_threshold}")
    
    @contextmanager
    def __call__(self, operation_name: str = "unknown"):
        """Context manager for circuit breaker."""
        with self._lock:
            if self._state == 'OPEN':
                if self._should_attempt_reset():
                    self._state = 'HALF_OPEN'
                    logger.info(f"Circuit breaker half-open for {operation_name}")
                else:
                    raise PhotonicComputationError(
                        f"Circuit breaker open for {operation_name}",
                        operation=operation_name
                    )
        
        try:
            yield
            self._on_success()
            
        except self.expected_exception as e:
            self._on_failure(operation_name, e)
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        return (
            self._last_failure_time is not None and
            time.time() - self._last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self) -> None:
        """Handle successful operation."""
        with self._lock:
            self._failure_count = 0
            if self._state == 'HALF_OPEN':
                self._state = 'CLOSED'
                logger.info("Circuit breaker closed after successful recovery")
    
    def _on_failure(self, operation_name: str, exception: Exception) -> None:
        """Handle failed operation."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._failure_count >= self.failure_threshold:
                self._state = 'OPEN'
                logger.error(
                    f"Circuit breaker opened for {operation_name} "
                    f"after {self._failure_count} failures: {exception}"
                )
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state."""
        with self._lock:
            return {
                'state': self._state,
                'failure_count': self._failure_count,
                'last_failure_time': self._last_failure_time,
                'time_until_retry': (
                    self.recovery_timeout - (time.time() - (self._last_failure_time or 0))
                    if self._last_failure_time else 0
                )
            }


class ErrorRecoveryManager:
    """Manages error recovery and fault tolerance."""
    
    DEFAULT_POLICIES = [
        RecoveryPolicy(
            error_pattern="timeout",
            max_retries=3,
            retry_delay=1.0,
            backoff_multiplier=2.0,
            max_delay=10.0,
            strategy=RecoveryStrategy.RETRY,
            severity_threshold=ErrorSeverity.MEDIUM
        ),
        RecoveryPolicy(
            error_pattern="hardware",
            max_retries=2,
            retry_delay=5.0,
            backoff_multiplier=1.5,
            max_delay=30.0,
            strategy=RecoveryStrategy.RECALIBRATE,
            severity_threshold=ErrorSeverity.HIGH
        ),
        RecoveryPolicy(
            error_pattern="thermal",
            max_retries=1,
            retry_delay=30.0,
            backoff_multiplier=1.0,
            max_delay=30.0,
            strategy=RecoveryStrategy.DEGRADE,
            severity_threshold=ErrorSeverity.CRITICAL
        ),
        RecoveryPolicy(
            error_pattern="computation",
            max_retries=2,
            retry_delay=0.5,
            backoff_multiplier=2.0,
            max_delay=5.0,
            strategy=RecoveryStrategy.FALLBACK,
            severity_threshold=ErrorSeverity.MEDIUM
        )
    ]
    
    def __init__(self, policies: Optional[List[RecoveryPolicy]] = None):
        """Initialize error recovery manager."""
        self.policies = policies or self.DEFAULT_POLICIES
        self.error_history: deque = deque(maxlen=1000)
        self.recovery_stats = {
            'total_errors': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'strategy_usage': {}
        }
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()
        
        logger.info(f"Error recovery manager initialized with {len(self.policies)} policies")
    
    def handle_error(
        self,
        error: Exception,
        operation: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """
        Handle error with appropriate recovery strategy.
        
        Args:
            error: The exception that occurred
            operation: Operation being performed
            context: Additional context information
            
        Returns:
            Recovery result or None if recovery failed
        """
        with self._lock:
            # Create error event
            error_event = self._create_error_event(error, operation, context)
            self.error_history.append(error_event)
            self.recovery_stats['total_errors'] += 1
            
            # Find matching policy
            policy = self._find_matching_policy(error, operation)
            if not policy:
                logger.error(f"No recovery policy found for error: {error}")
                return None
            
            logger.info(f"Applying recovery strategy '{policy.strategy.value}' for {operation}")
            
            # Apply recovery strategy
            try:
                result = self._apply_recovery_strategy(error_event, policy, context)
                if result is not None:
                    error_event.recovery_successful = True
                    self.recovery_stats['successful_recoveries'] += 1
                    logger.info(f"Recovery successful for {operation}")
                else:
                    error_event.recovery_successful = False
                    self.recovery_stats['failed_recoveries'] += 1
                    logger.error(f"Recovery failed for {operation}")
                
                # Update strategy usage stats
                strategy_name = policy.strategy.value
                self.recovery_stats['strategy_usage'][strategy_name] = (
                    self.recovery_stats['strategy_usage'].get(strategy_name, 0) + 1
                )
                
                return result
                
            except Exception as recovery_error:
                logger.error(f"Recovery strategy failed: {recovery_error}")
                error_event.recovery_successful = False
                self.recovery_stats['failed_recoveries'] += 1
                return None
    
    def _create_error_event(
        self,
        error: Exception,
        operation: str,
        context: Optional[Dict[str, Any]]
    ) -> ErrorEvent:
        """Create error event from exception."""
        severity = self._assess_error_severity(error, operation)
        error_type = type(error).__name__
        
        return ErrorEvent(
            timestamp=time.time(),
            error_type=error_type,
            severity=severity,
            message=str(error),
            operation=operation,
            metadata=context or {},
            recovery_attempted=True
        )
    
    def _assess_error_severity(self, error: Exception, operation: str) -> ErrorSeverity:
        """Assess error severity."""
        if isinstance(error, PhotonicThermalError):
            return ErrorSeverity.CRITICAL
        elif isinstance(error, PhotonicHardwareError):
            return ErrorSeverity.HIGH
        elif isinstance(error, PhotonicTimeoutError):
            return ErrorSeverity.MEDIUM
        elif isinstance(error, PhotonicComputationError):
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _find_matching_policy(
        self, 
        error: Exception, 
        operation: str
    ) -> Optional[RecoveryPolicy]:
        """Find matching recovery policy for error."""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        operation_lower = operation.lower()
        
        for policy in self.policies:
            pattern = policy.error_pattern.lower()
            if (pattern in error_str or 
                pattern in error_type or 
                pattern in operation_lower):
                return policy
        
        return None
    
    def _apply_recovery_strategy(
        self,
        error_event: ErrorEvent,
        policy: RecoveryPolicy,
        context: Optional[Dict[str, Any]]
    ) -> Optional[Any]:
        """Apply specific recovery strategy."""
        error_event.recovery_strategy = policy.strategy
        
        if policy.strategy == RecoveryStrategy.RETRY:
            return self._retry_operation(error_event, policy, context)
        elif policy.strategy == RecoveryStrategy.FALLBACK:
            return self._fallback_operation(error_event, context)
        elif policy.strategy == RecoveryStrategy.DEGRADE:
            return self._degrade_operation(error_event, context)
        elif policy.strategy == RecoveryStrategy.RECALIBRATE:
            return self._recalibrate_operation(error_event, context)
        elif policy.strategy == RecoveryStrategy.SHUTDOWN:
            return self._shutdown_operation(error_event, context)
        else:
            logger.warning(f"Unknown recovery strategy: {policy.strategy}")
            return None
    
    def _retry_operation(
        self,
        error_event: ErrorEvent,
        policy: RecoveryPolicy,
        context: Optional[Dict[str, Any]]
    ) -> Optional[Any]:
        """Retry operation with backoff."""
        operation_func = context.get('operation_func') if context else None
        operation_args = context.get('operation_args', []) if context else []
        operation_kwargs = context.get('operation_kwargs', {}) if context else {}
        
        if not operation_func:
            logger.warning("No operation function provided for retry")
            return None
        
        delay = policy.retry_delay
        
        for attempt in range(policy.max_retries):
            try:
                time.sleep(delay)
                logger.info(f"Retry attempt {attempt + 1}/{policy.max_retries} for {error_event.operation}")
                
                result = operation_func(*operation_args, **operation_kwargs)
                logger.info(f"Retry successful after {attempt + 1} attempts")
                return result
                
            except Exception as retry_error:
                logger.warning(f"Retry attempt {attempt + 1} failed: {retry_error}")
                delay = min(delay * policy.backoff_multiplier, policy.max_delay)
        
        logger.error(f"All retry attempts failed for {error_event.operation}")
        return None
    
    def _fallback_operation(
        self,
        error_event: ErrorEvent,
        context: Optional[Dict[str, Any]]
    ) -> Optional[Any]:
        """Execute fallback operation."""
        fallback_func = context.get('fallback_func') if context else None
        fallback_args = context.get('fallback_args', []) if context else []
        fallback_kwargs = context.get('fallback_kwargs', {}) if context else {}
        
        if not fallback_func:
            logger.warning("No fallback function provided")
            return None
        
        try:
            logger.info(f"Executing fallback for {error_event.operation}")
            result = fallback_func(*fallback_args, **fallback_kwargs)
            logger.info("Fallback operation successful")
            return result
            
        except Exception as fallback_error:
            logger.error(f"Fallback operation failed: {fallback_error}")
            return None
    
    def _degrade_operation(
        self,
        error_event: ErrorEvent,
        context: Optional[Dict[str, Any]]
    ) -> Optional[Any]:
        """Degrade operation performance."""
        degrade_func = context.get('degrade_func') if context else None
        
        if not degrade_func:
            logger.warning("No degraded mode function provided")
            return None
        
        try:
            logger.info(f"Executing degraded mode for {error_event.operation}")
            result = degrade_func()
            logger.info("Degraded mode operation successful")
            return result
            
        except Exception as degrade_error:
            logger.error(f"Degraded mode operation failed: {degrade_error}")
            return None
    
    def _recalibrate_operation(
        self,
        error_event: ErrorEvent,
        context: Optional[Dict[str, Any]]
    ) -> Optional[Any]:
        """Recalibrate system and retry."""
        calibrate_func = context.get('calibrate_func') if context else None
        operation_func = context.get('operation_func') if context else None
        
        if not calibrate_func:
            logger.warning("No calibration function provided")
            return None
        
        try:
            logger.info(f"Recalibrating system for {error_event.operation}")
            calibrate_func()
            
            if operation_func:
                operation_args = context.get('operation_args', [])
                operation_kwargs = context.get('operation_kwargs', {})
                result = operation_func(*operation_args, **operation_kwargs)
                logger.info("Operation successful after recalibration")
                return result
            else:
                logger.info("Recalibration completed")
                return True
                
        except Exception as calibrate_error:
            logger.error(f"Recalibration failed: {calibrate_error}")
            return None
    
    def _shutdown_operation(
        self,
        error_event: ErrorEvent,
        context: Optional[Dict[str, Any]]
    ) -> Optional[Any]:
        """Shutdown system safely."""
        shutdown_func = context.get('shutdown_func') if context else None
        
        logger.critical(f"Initiating safe shutdown due to {error_event.operation} error")
        
        if shutdown_func:
            try:
                shutdown_func()
                logger.info("Safe shutdown completed")
            except Exception as shutdown_error:
                logger.error(f"Shutdown failed: {shutdown_error}")
        
        return None
    
    def get_circuit_breaker(
        self,
        operation: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0
    ) -> CircuitBreaker:
        """Get or create circuit breaker for operation."""
        if operation not in self.circuit_breakers:
            self.circuit_breakers[operation] = CircuitBreaker(
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout
            )
        
        return self.circuit_breakers[operation]
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get recovery statistics."""
        with self._lock:
            recent_errors = [
                e for e in self.error_history
                if time.time() - e.timestamp < 3600  # Last hour
            ]
            
            error_types = {}
            severity_counts = {}
            
            for error in recent_errors:
                error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
                severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
            
            circuit_breaker_states = {
                name: breaker.get_state()
                for name, breaker in self.circuit_breakers.items()
            }
            
            return {
                'total_stats': self.recovery_stats.copy(),
                'recent_errors': len(recent_errors),
                'error_types_1h': error_types,
                'severity_distribution_1h': severity_counts,
                'circuit_breakers': circuit_breaker_states,
                'recovery_success_rate': (
                    self.recovery_stats['successful_recoveries'] /
                    max(self.recovery_stats['total_errors'], 1)
                )
            }
    
    def reset_stats(self) -> None:
        """Reset recovery statistics."""
        with self._lock:
            self.recovery_stats = {
                'total_errors': 0,
                'successful_recoveries': 0,
                'failed_recoveries': 0,
                'strategy_usage': {}
            }
            self.error_history.clear()
            logger.info("Recovery statistics reset")


# Convenience decorators
def with_error_recovery(
    operation_name: str,
    fallback_func: Optional[Callable] = None,
    max_retries: int = 3
):
    """Decorator for automatic error recovery."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            recovery_manager = get_error_recovery_manager()
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    'operation_func': func,
                    'operation_args': args,
                    'operation_kwargs': kwargs,
                    'fallback_func': fallback_func
                }
                
                return recovery_manager.handle_error(e, operation_name, context)
        
        return wrapper
    return decorator


def with_circuit_breaker(
    operation_name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0
):
    """Decorator for circuit breaker pattern."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            recovery_manager = get_error_recovery_manager()
            breaker = recovery_manager.get_circuit_breaker(
                operation_name, failure_threshold, recovery_timeout
            )
            
            with breaker(operation_name):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Global error recovery manager
_error_recovery_manager: Optional[ErrorRecoveryManager] = None


def get_error_recovery_manager() -> ErrorRecoveryManager:
    """Get global error recovery manager."""
    global _error_recovery_manager
    if _error_recovery_manager is None:
        _error_recovery_manager = ErrorRecoveryManager()
    return _error_recovery_manager


def set_error_recovery_manager(manager: ErrorRecoveryManager) -> None:
    """Set global error recovery manager."""
    global _error_recovery_manager
    _error_recovery_manager = manager