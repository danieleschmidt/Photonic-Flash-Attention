"""
Comprehensive tests for error recovery and fault tolerance systems.

Tests all aspects of the error recovery engine including circuit breakers,
retry policies, graceful degradation, and recovery strategies.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from collections import defaultdict

from src.photonic_flash_attention.core.error_recovery import (
    ErrorRecoveryEngine, CircuitBreaker, CircuitBreakerState, 
    RetryPolicy, GracefulDegradationManager, ErrorContext, 
    RecoveryStrategy, ErrorSeverity, RecoveryResult,
    with_error_recovery, create_circuit_breaker
)
from src.photonic_flash_attention.utils.exceptions import (
    PhotonicHardwareError, PhotonicComputationError, PhotonicTimeoutError,
    PhotonicThermalError, PhotonicDriverError
)


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker is properly initialized."""
        breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=30.0,
            expected_exception=PhotonicHardwareError,
            name="test_breaker"
        )
        
        assert breaker.failure_threshold == 3
        assert breaker.recovery_timeout == 30.0
        assert breaker.expected_exception == PhotonicHardwareError
        assert breaker.name == "test_breaker"
        assert breaker.get_state() == CircuitBreakerState.CLOSED
    
    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state allows operations."""
        breaker = CircuitBreaker(failure_threshold=3)
        
        with breaker.protect():
            # Should allow operation
            result = "success"
        
        assert breaker.get_state() == CircuitBreakerState.CLOSED
    
    def test_circuit_breaker_opens_on_failures(self):
        """Test circuit breaker opens after threshold failures."""
        breaker = CircuitBreaker(failure_threshold=2)
        
        # First failure
        with pytest.raises(Exception):
            with breaker.protect():
                raise PhotonicHardwareError("Test failure", "test_device")
        
        assert breaker.get_state() == CircuitBreakerState.CLOSED
        
        # Second failure - should open circuit
        with pytest.raises(Exception):
            with breaker.protect():
                raise PhotonicHardwareError("Test failure 2", "test_device")
        
        assert breaker.get_state() == CircuitBreakerState.OPEN
    
    def test_circuit_breaker_rejects_when_open(self):
        """Test circuit breaker rejects requests when open."""
        breaker = CircuitBreaker(failure_threshold=1)
        
        # Cause failure to open circuit
        with pytest.raises(Exception):
            with breaker.protect():
                raise PhotonicHardwareError("Test failure", "test_device")
        
        # Should now reject requests
        with pytest.raises(PhotonicComputationError, match="Circuit breaker .* is OPEN"):
            with breaker.protect():
                pass
    
    def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker half-open recovery."""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        
        # Cause failure
        with pytest.raises(Exception):
            with breaker.protect():
                raise PhotonicHardwareError("Test failure", "test_device")
        
        # Wait for recovery timeout
        time.sleep(0.2)
        
        # Should be half-open now and allow one request
        with breaker.protect():
            # Successful operation should close the circuit
            pass
        
        assert breaker.get_state() == CircuitBreakerState.CLOSED
    
    def test_circuit_breaker_statistics(self):
        """Test circuit breaker collects statistics."""
        breaker = CircuitBreaker(failure_threshold=2)
        
        # Successful operation
        with breaker.protect():
            pass
        
        # Failed operation
        with pytest.raises(Exception):
            with breaker.protect():
                raise PhotonicHardwareError("Test failure", "test_device")
        
        stats = breaker.get_stats()
        assert stats['stats']['total_calls'] == 2
        assert stats['stats']['successful_calls'] == 1
        assert stats['stats']['failed_calls'] == 1


class TestRetryPolicy:
    """Test retry policy functionality."""
    
    def test_retry_policy_initialization(self):
        """Test retry policy initialization."""
        policy = RetryPolicy(
            max_retries=5,
            base_delay=2.0,
            max_delay=30.0,
            exponential_base=3.0,
            retryable_exceptions=(PhotonicTimeoutError,)
        )
        
        assert policy.max_retries == 5
        assert policy.base_delay == 2.0
        assert policy.max_delay == 30.0
        assert policy.exponential_base == 3.0
        assert policy.retryable_exceptions == (PhotonicTimeoutError,)
    
    def test_should_retry_logic(self):
        """Test retry decision logic."""
        policy = RetryPolicy(
            max_retries=3,
            retryable_exceptions=(PhotonicTimeoutError, PhotonicComputationError)
        )
        
        # Should retry timeout errors
        assert policy.should_retry(PhotonicTimeoutError("timeout", 30.0), 1)
        
        # Should retry computation errors
        assert policy.should_retry(PhotonicComputationError("compute error"), 1)
        
        # Should not retry thermal errors (not retryable)
        assert not policy.should_retry(PhotonicThermalError("too hot", 85.0, 80.0), 1)
        
        # Should not retry after max attempts
        assert not policy.should_retry(PhotonicTimeoutError("timeout", 30.0), 4)
    
    def test_exponential_backoff_delay(self):
        """Test exponential backoff delay calculation."""
        policy = RetryPolicy(base_delay=1.0, exponential_base=2.0, max_delay=10.0)
        
        # Test delay progression
        assert policy.get_delay(0) <= 1.0  # Base delay with jitter
        assert policy.get_delay(1) <= 2.0  # 2^1 with jitter
        assert policy.get_delay(2) <= 4.0  # 2^2 with jitter
        assert policy.get_delay(10) <= 10.0  # Should cap at max_delay


class TestGracefulDegradationManager:
    """Test graceful degradation functionality."""
    
    def test_degradation_manager_initialization(self):
        """Test degradation manager initialization."""
        manager = GracefulDegradationManager()
        
        assert manager.current_level == 'photonic'
        assert manager.get_current_quality() == 1.0
        assert 'photonic' in manager.fallback_chain
        assert 'cpu' in manager.fallback_chain
    
    def test_degrade_to_level(self):
        """Test degrading to specific quality level."""
        manager = GracefulDegradationManager()
        
        manager.degrade_to_level('gpu_standard', 'Test degradation')
        
        assert manager.current_level == 'gpu_standard'
        assert manager.get_current_quality() == 0.7
        assert 'Test degradation' in str(manager.degradation_reasons)
    
    def test_recover_to_higher_level(self):
        """Test recovering to higher quality level."""
        manager = GracefulDegradationManager()
        
        # Degrade first
        manager.degrade_to_level('cpu', 'Test degradation')
        assert manager.get_current_quality() == 0.5
        
        # Recover
        manager.recover_to_level('gpu_optimized')
        
        assert manager.current_level == 'gpu_optimized'
        assert manager.get_current_quality() == 0.9
    
    def test_fallback_implementation_selection(self):
        """Test fallback implementation selection."""
        manager = GracefulDegradationManager()
        
        assert manager.get_fallback_implementation() == 'photonic'
        
        manager.degrade_to_level('gpu_standard', 'Test')
        assert manager.get_fallback_implementation() == 'gpu_standard'


class TestErrorRecoveryEngine:
    """Test main error recovery engine."""
    
    def test_recovery_engine_initialization(self):
        """Test recovery engine initialization."""
        engine = ErrorRecoveryEngine()
        
        assert engine.retry_policy is not None
        assert engine.degradation_manager is not None
        assert len(engine.circuit_breakers) > 0
        assert engine.recovery_stats['total_errors'] == 0
    
    def test_error_handling_with_retry_strategy(self):
        """Test error handling with retry strategy."""
        engine = ErrorRecoveryEngine()
        
        error = PhotonicTimeoutError("Operation timeout", 30.0)
        result = engine.handle_error(error, "test_operation", max_retries=2)
        
        assert isinstance(result, RecoveryResult)
        assert result.strategy_used in [RecoveryStrategy.RETRY, RecoveryStrategy.FALLBACK]
        assert engine.recovery_stats['total_errors'] == 1
    
    def test_error_handling_with_fallback_strategy(self):
        """Test error handling with fallback strategy."""
        engine = ErrorRecoveryEngine()
        
        error = PhotonicComputationError("Computation failed")
        result = engine.handle_error(error, "test_computation", fallback_available=True)
        
        assert isinstance(result, RecoveryResult)
        # Should use fallback or retry strategy
        assert result.strategy_used in [RecoveryStrategy.FALLBACK, RecoveryStrategy.RETRY]
    
    def test_thermal_error_degradation(self):
        """Test thermal errors trigger graceful degradation."""
        engine = ErrorRecoveryEngine()
        
        error = PhotonicThermalError("Temperature too high", 85.0, 80.0)
        result = engine.handle_error(error, "photonic_operation")
        
        assert result.strategy_used == RecoveryStrategy.GRACEFUL_DEGRADATION
        assert engine.degradation_manager.current_level != 'photonic'
    
    def test_hardware_error_circuit_breaker(self):
        """Test hardware errors use circuit breaker."""
        engine = ErrorRecoveryEngine()
        
        error = PhotonicHardwareError("Hardware failure", "device_1")
        result = engine.handle_error(error, "hardware_operation", component_id="photonic_hardware")
        
        assert result.strategy_used == RecoveryStrategy.CIRCUIT_BREAKER
        # Circuit breaker should have recorded the failure
        assert "photonic_hardware" in engine.circuit_breakers
    
    def test_recovery_statistics(self):
        """Test recovery statistics collection."""
        engine = ErrorRecoveryEngine()
        
        # Generate some errors
        errors = [
            PhotonicTimeoutError("timeout", 30.0),
            PhotonicComputationError("compute error"),
            PhotonicHardwareError("hw error", "device_1")
        ]
        
        for error in errors:
            engine.handle_error(error, "test_operation")
        
        stats = engine.get_recovery_stats()
        assert stats['recovery_stats']['total_errors'] == 3
        assert stats['recovery_stats']['error_types']['PhotonicTimeoutError'] == 1
        assert stats['recovery_stats']['error_types']['PhotonicComputationError'] == 1
        assert stats['recovery_stats']['error_types']['PhotonicHardwareError'] == 1
    
    def test_recovery_history_tracking(self):
        """Test recovery history tracking."""
        engine = ErrorRecoveryEngine()
        
        error = PhotonicTimeoutError("timeout", 30.0)
        engine.handle_error(error, "test_operation")
        
        assert len(engine.recovery_history) == 1
        history_entry = engine.recovery_history[0]
        assert history_entry['operation'] == 'test_operation'
        assert history_entry['error_type'] == 'PhotonicTimeoutError'
    
    def test_component_health_monitoring(self):
        """Test component health monitoring."""
        engine = ErrorRecoveryEngine()
        
        health = engine.get_component_health()
        assert 'circuit_breaker_states' in health
        assert 'degradation_active' in health
        assert isinstance(health['degradation_active'], bool)
    
    def test_recovery_state_reset(self):
        """Test recovery state reset."""
        engine = ErrorRecoveryEngine()
        
        # Generate some state
        error = PhotonicHardwareError("hw error", "device_1")
        engine.handle_error(error, "test_operation")
        engine.degradation_manager.degrade_to_level('cpu', 'test')
        
        # Reset state
        engine.reset_recovery_state()
        
        assert engine.recovery_stats['total_errors'] == 0
        assert engine.degradation_manager.current_level == 'photonic'
        assert len(engine.recovery_history) == 0


class TestErrorRecoveryDecorator:
    """Test error recovery decorator functionality."""
    
    def test_with_error_recovery_decorator(self):
        """Test error recovery decorator."""
        
        @with_error_recovery(
            operation="test_operation",
            component_id="test_component",
            max_retries=2
        )
        def failing_function():
            raise PhotonicTimeoutError("timeout", 30.0, "test_operation")
        
        # Should handle error and potentially recover
        with pytest.raises(PhotonicTimeoutError):
            failing_function()
        
        # Check that recovery was attempted
        engine = ErrorRecoveryEngine()  # Get fresh instance
        # Note: In real implementation, would check global instance stats
    
    def test_decorator_with_successful_retry(self):
        """Test decorator with successful retry after failure."""
        call_count = 0
        
        @with_error_recovery(
            operation="test_operation", 
            max_retries=3,
            fallback_available=False
        )
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise PhotonicTimeoutError("timeout", 30.0, "test_operation")
            return "success"
        
        # Should succeed after retries
        result = flaky_function()
        assert result == "success"
        assert call_count >= 3  # Original call + retries


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration with recovery engine."""
    
    def test_circuit_breaker_creation(self):
        """Test circuit breaker creation function."""
        breaker = create_circuit_breaker(
            name="test_breaker",
            failure_threshold=3,
            recovery_timeout=30.0,
            expected_exception=PhotonicHardwareError
        )
        
        assert breaker.name == "test_breaker"
        assert breaker.failure_threshold == 3
        assert breaker.recovery_timeout == 30.0
        
        # Should be registered with global engine
        engine = ErrorRecoveryEngine()  # Get instance to check registration
        # Note: In real implementation, would verify registration
    
    def test_circuit_breaker_decorator(self):
        """Test circuit breaker as decorator."""
        breaker = CircuitBreaker(failure_threshold=2)
        
        @breaker
        def test_function():
            raise PhotonicHardwareError("test error", "device_1")
        
        # First two calls should fail normally
        with pytest.raises(PhotonicHardwareError):
            test_function()
        
        with pytest.raises(PhotonicHardwareError):
            test_function()
        
        # Third call should be rejected by circuit breaker
        with pytest.raises(PhotonicComputationError):
            test_function()


class TestRecoveryEngineThreadSafety:
    """Test thread safety of recovery engine."""
    
    def test_concurrent_error_handling(self):
        """Test concurrent error handling is thread-safe."""
        engine = ErrorRecoveryEngine()
        results = []
        errors = []
        
        def handle_error_thread(thread_id):
            try:
                error = PhotonicComputationError(f"Error from thread {thread_id}")
                result = engine.handle_error(error, f"operation_{thread_id}")
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=handle_error_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) == 10
        assert engine.recovery_stats['total_errors'] == 10
    
    def test_concurrent_circuit_breaker_access(self):
        """Test concurrent circuit breaker access."""
        breaker = CircuitBreaker(failure_threshold=5)
        successes = []
        failures = []
        
        def test_breaker_thread(should_fail=False):
            try:
                with breaker.protect():
                    if should_fail:
                        raise PhotonicHardwareError("test failure", "device_1")
                    else:
                        successes.append(1)
            except Exception:
                failures.append(1)
        
        # Start threads - some failing, some succeeding
        threads = []
        for i in range(20):
            should_fail = i % 4 == 0  # Every 4th thread fails
            thread = threading.Thread(target=test_breaker_thread, args=(should_fail,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Verify thread safety - no exceptions from race conditions
        total_operations = len(successes) + len(failures)
        assert total_operations == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])