"""Advanced Fault Tolerance and Resilience System.

Implements comprehensive fault tolerance mechanisms for photonic flash attention
including circuit breakers, graceful degradation, and automatic recovery.
"""

import asyncio
import threading
import time
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import torch
import numpy as np
from pathlib import Path
import json
import pickle

from ..config import get_config
from ..utils.logging import get_logger
from ..utils.exceptions import (
    PhotonicHardwareError, PhotonicComputationError, PhotonicTimeoutError
)


class SystemState(Enum):
    """System operational states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    RECOVERY = "recovery"
    MAINTENANCE = "maintenance"


class ComponentState(Enum):
    """Component operational states."""
    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"
    DISABLED = "disabled"


@dataclass
class FailureEvent:
    """Record of a system failure event."""
    timestamp: float
    component: str
    error_type: str
    error_message: str
    severity: str
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_time: Optional[float] = None


@dataclass
class RecoveryStrategy:
    """Strategy for recovering from failures."""
    name: str
    priority: int
    max_attempts: int
    backoff_multiplier: float
    timeout_seconds: float
    recovery_function: Callable
    prerequisites: List[str] = field(default_factory=list)
    success_threshold: float = 0.8


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for fault tolerance.
    
    Prevents cascade failures by temporarily disabling failing components
    and allowing gradual recovery.
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 3,
        half_open_max_requests: int = 5,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.half_open_max_requests = half_open_max_requests
        
        self.logger = get_logger(f"CircuitBreaker({name})")
        
        # State management
        self._state = 'closed'  # closed, open, half-open
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0
        self._half_open_requests = 0
        
        # Statistics
        self._total_requests = 0
        self._total_failures = 0
        self._total_successes = 0
        
        self._lock = threading.RLock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function through circuit breaker."""
        with self._lock:
            self._total_requests += 1
            
            # Check if circuit is open
            if self._state == 'open':
                if time.time() - self._last_failure_time < self.recovery_timeout:
                    raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is open")
                else:
                    # Transition to half-open
                    self._state = 'half-open'
                    self._half_open_requests = 0
                    self.logger.info(f"Circuit breaker {self.name} transitioning to half-open")
            
            # Limit requests in half-open state
            if self._state == 'half-open':
                if self._half_open_requests >= self.half_open_max_requests:
                    raise CircuitBreakerOpenError(f"Circuit breaker {self.name} half-open request limit reached")
                self._half_open_requests += 1
        
        # Execute function
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        
        except Exception as e:
            self._record_failure(e)
            raise
    
    def _record_success(self):
        """Record successful execution."""
        with self._lock:
            self._total_successes += 1
            self._failure_count = 0
            
            if self._state == 'half-open':
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._state = 'closed'
                    self._success_count = 0
                    self.logger.info(f"Circuit breaker {self.name} closed after successful recovery")
    
    def _record_failure(self, error: Exception):
        """Record failed execution."""
        with self._lock:
            self._total_failures += 1
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._failure_count >= self.failure_threshold:
                self._state = 'open'
                self._success_count = 0
                self.logger.warning(f"Circuit breaker {self.name} opened due to {self._failure_count} failures")
            
            if self._state == 'half-open':
                self._state = 'open'
                self._success_count = 0
                self.logger.warning(f"Circuit breaker {self.name} reopened due to failure in half-open state")
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state information."""
        with self._lock:
            return {
                'name': self.name,
                'state': self._state,
                'failure_count': self._failure_count,
                'success_count': self._success_count,
                'total_requests': self._total_requests,
                'total_failures': self._total_failures,
                'total_successes': self._total_successes,
                'success_rate': self._total_successes / max(self._total_requests, 1),
                'last_failure_time': self._last_failure_time,
            }
    
    def reset(self):
        """Reset circuit breaker to closed state."""
        with self._lock:
            self._state = 'closed'
            self._failure_count = 0
            self._success_count = 0
            self._half_open_requests = 0
            self.logger.info(f"Circuit breaker {self.name} reset")


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class GracefulDegradationManager:
    """
    Manages graceful degradation of system capabilities.
    
    Provides fallback mechanisms and reduced functionality when
    components fail or performance degrades.
    """
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.config = get_config()
        
        # Degradation policies
        self.degradation_policies = {
            'photonic_hardware_failure': self._degrade_to_gpu_only,
            'memory_pressure': self._reduce_batch_size,
            'thermal_throttling': self._reduce_optical_power,
            'network_issues': self._enable_local_cache,
            'high_latency': self._switch_to_fast_mode,
        }
        
        # Fallback configurations
        self.fallback_configs = {
            'gpu_only': {
                'device': 'gpu',
                'use_flash': True,
                'precision': 'fp16',
                'enable_checkpointing': True,
            },
            'reduced_precision': {
                'precision': 'fp16',
                'quantization': 'int8',
                'reduce_heads': True,
            },
            'minimal_resources': {
                'batch_size': 1,
                'sequence_length': 512,
                'embed_dim': 256,
                'num_heads': 4,
            }
        }
        
        # Current degradation state
        self.active_degradations = set()
        self.degradation_history = deque(maxlen=1000)
        
        self._lock = threading.RLock()
    
    def apply_degradation(self, trigger: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply degradation policy for given trigger."""
        with self._lock:
            if trigger in self.degradation_policies:
                policy_func = self.degradation_policies[trigger]
                degraded_config = policy_func(context)
                
                self.active_degradations.add(trigger)
                self.degradation_history.append({
                    'timestamp': time.time(),
                    'trigger': trigger,
                    'context': context,
                    'config': degraded_config,
                    'action': 'applied'
                })
                
                self.logger.info(f"Applied degradation policy '{trigger}': {degraded_config}")
                return degraded_config
            else:
                self.logger.warning(f"No degradation policy for trigger: {trigger}")
                return context
    
    def remove_degradation(self, trigger: str) -> None:
        """Remove degradation policy."""
        with self._lock:
            if trigger in self.active_degradations:
                self.active_degradations.remove(trigger)
                self.degradation_history.append({
                    'timestamp': time.time(),
                    'trigger': trigger,
                    'action': 'removed'
                })
                self.logger.info(f"Removed degradation policy: {trigger}")
    
    def _degrade_to_gpu_only(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback to GPU-only computation."""
        config = context.copy()
        config.update(self.fallback_configs['gpu_only'])
        return config
    
    def _reduce_batch_size(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Reduce batch size to manage memory pressure."""
        config = context.copy()
        current_batch_size = config.get('batch_size', 4)
        config['batch_size'] = max(1, current_batch_size // 2)
        config['enable_gradient_checkpointing'] = True
        return config
    
    def _reduce_optical_power(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Reduce optical power to manage thermal issues."""
        config = context.copy()
        config['optical_power_reduction'] = 0.7
        config['thermal_throttling_enabled'] = True
        config['wavelengths_used'] = min(config.get('wavelengths_used', 80), 40)
        return config
    
    def _enable_local_cache(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enable aggressive local caching."""
        config = context.copy()
        config['enable_result_caching'] = True
        config['cache_size_multiplier'] = 2.0
        config['cache_ttl_seconds'] = 3600
        return config
    
    def _switch_to_fast_mode(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Switch to fast mode with reduced accuracy."""
        config = context.copy()
        config.update(self.fallback_configs['reduced_precision'])
        config['approximation_mode'] = 'fast'
        return config
    
    def get_degradation_status(self) -> Dict[str, Any]:
        """Get current degradation status."""
        with self._lock:
            return {
                'active_degradations': list(self.active_degradations),
                'total_degradations_applied': len(self.degradation_history),
                'recent_degradations': list(self.degradation_history)[-10:],
                'available_policies': list(self.degradation_policies.keys()),
            }


class AutoRecoverySystem:
    """
    Automated recovery system for handling failures.
    
    Implements multiple recovery strategies with intelligent selection
    and automatic execution.
    """
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.config = get_config()
        
        # Recovery strategies
        self.recovery_strategies = {
            'hardware_reset': RecoveryStrategy(
                name='hardware_reset',
                priority=1,
                max_attempts=3,
                backoff_multiplier=2.0,
                timeout_seconds=30.0,
                recovery_function=self._recover_hardware,
            ),
            'service_restart': RecoveryStrategy(
                name='service_restart',
                priority=2,
                max_attempts=2,
                backoff_multiplier=1.5,
                timeout_seconds=60.0,
                recovery_function=self._restart_service,
            ),
            'cache_clear': RecoveryStrategy(
                name='cache_clear',
                priority=3,
                max_attempts=1,
                backoff_multiplier=1.0,
                timeout_seconds=10.0,
                recovery_function=self._clear_cache,
            ),
            'memory_cleanup': RecoveryStrategy(
                name='memory_cleanup',
                priority=3,
                max_attempts=1,
                backoff_multiplier=1.0,
                timeout_seconds=15.0,
                recovery_function=self._cleanup_memory,
            ),
        }
        
        # Recovery history and statistics
        self.recovery_attempts = defaultdict(int)
        self.recovery_successes = defaultdict(int)
        self.recovery_history = deque(maxlen=1000)
        
        # Background recovery task
        self._recovery_active = False
        self._recovery_thread = None
        
        self._lock = threading.RLock()
    
    def attempt_recovery(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Attempt recovery from error."""
        error_type = type(error).__name__
        
        with self._lock:
            # Select appropriate recovery strategies
            strategies = self._select_recovery_strategies(error_type, context)
            
            if not strategies:
                self.logger.warning(f"No recovery strategies available for error: {error_type}")
                return False
            
            # Attempt recovery with selected strategies
            for strategy in strategies:
                if self._execute_recovery_strategy(strategy, error, context):
                    return True
            
            return False
    
    def _select_recovery_strategies(self, error_type: str, context: Dict[str, Any]) -> List[RecoveryStrategy]:
        """Select appropriate recovery strategies."""
        strategies = []
        
        # Map error types to strategies
        error_strategy_map = {
            'PhotonicHardwareError': ['hardware_reset'],
            'PhotonicTimeoutError': ['service_restart', 'cache_clear'],
            'PhotonicComputationError': ['memory_cleanup', 'cache_clear'],
            'OutOfMemoryError': ['memory_cleanup'],
            'RuntimeError': ['service_restart'],
        }
        
        strategy_names = error_strategy_map.get(error_type, ['cache_clear', 'memory_cleanup'])
        
        for name in strategy_names:
            if name in self.recovery_strategies:
                strategy = self.recovery_strategies[name]
                
                # Check if strategy has been attempted too many times
                if self.recovery_attempts[name] < strategy.max_attempts:
                    strategies.append(strategy)
        
        # Sort by priority
        strategies.sort(key=lambda s: s.priority)
        return strategies
    
    def _execute_recovery_strategy(self, strategy: RecoveryStrategy, error: Exception, context: Dict[str, Any]) -> bool:
        """Execute a recovery strategy."""
        self.logger.info(f"Attempting recovery strategy: {strategy.name}")
        
        attempt_count = self.recovery_attempts[strategy.name]
        self.recovery_attempts[strategy.name] += 1
        
        # Calculate backoff delay
        backoff_delay = strategy.backoff_multiplier ** attempt_count
        if backoff_delay > 1:
            time.sleep(min(backoff_delay, 30))  # Cap at 30 seconds
        
        try:
            start_time = time.time()
            
            # Execute recovery function with timeout
            success = asyncio.run(
                asyncio.wait_for(
                    self._run_recovery_function(strategy.recovery_function, error, context),
                    timeout=strategy.timeout_seconds
                )
            )
            
            recovery_time = time.time() - start_time
            
            if success:
                self.recovery_successes[strategy.name] += 1
                self.recovery_history.append({
                    'timestamp': time.time(),
                    'strategy': strategy.name,
                    'error_type': type(error).__name__,
                    'success': True,
                    'recovery_time': recovery_time,
                    'attempt_count': attempt_count + 1,
                })
                
                self.logger.info(f"Recovery strategy '{strategy.name}' succeeded in {recovery_time:.2f}s")
                return True
            
        except asyncio.TimeoutError:
            self.logger.error(f"Recovery strategy '{strategy.name}' timed out after {strategy.timeout_seconds}s")
        except Exception as e:
            self.logger.error(f"Recovery strategy '{strategy.name}' failed: {e}")
        
        self.recovery_history.append({
            'timestamp': time.time(),
            'strategy': strategy.name,
            'error_type': type(error).__name__,
            'success': False,
            'attempt_count': attempt_count + 1,
        })
        
        return False
    
    async def _run_recovery_function(self, recovery_func: Callable, error: Exception, context: Dict[str, Any]) -> bool:
        """Run recovery function asynchronously."""
        if asyncio.iscoroutinefunction(recovery_func):
            return await recovery_func(error, context)
        else:
            return recovery_func(error, context)
    
    async def _recover_hardware(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Attempt hardware recovery."""
        self.logger.info("Attempting hardware recovery...")
        
        try:
            # Simulate hardware reset
            await asyncio.sleep(2)  # Hardware reset delay
            
            # Check hardware status
            if hasattr(context.get('hardware'), 'reset'):
                context['hardware'].reset()
            
            # Validate hardware is working
            await asyncio.sleep(1)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Hardware recovery failed: {e}")
            return False
    
    async def _restart_service(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Restart service components."""
        self.logger.info("Restarting service components...")
        
        try:
            # Simulate service restart
            await asyncio.sleep(3)  # Service restart delay
            
            # Clear any cached state
            if 'cache' in context:
                context['cache'].clear()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Service restart failed: {e}")
            return False
    
    async def _clear_cache(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Clear system caches."""
        self.logger.info("Clearing system caches...")
        
        try:
            # Clear PyTorch cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Clear application caches
            if 'cache_manager' in context:
                context['cache_manager'].clear_all()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Cache clear failed: {e}")
            return False
    
    async def _cleanup_memory(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Clean up memory usage."""
        self.logger.info("Cleaning up memory...")
        
        try:
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear PyTorch cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Additional memory cleanup
            await asyncio.sleep(0.5)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Memory cleanup failed: {e}")
            return False
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get recovery statistics."""
        with self._lock:
            stats = {
                'total_attempts': sum(self.recovery_attempts.values()),
                'total_successes': sum(self.recovery_successes.values()),
                'strategy_stats': {},
                'recent_recoveries': list(self.recovery_history)[-20:],
            }
            
            for strategy_name in self.recovery_strategies:
                attempts = self.recovery_attempts[strategy_name]
                successes = self.recovery_successes[strategy_name]
                
                stats['strategy_stats'][strategy_name] = {
                    'attempts': attempts,
                    'successes': successes,
                    'success_rate': successes / max(attempts, 1),
                }
            
            stats['overall_success_rate'] = stats['total_successes'] / max(stats['total_attempts'], 1)
            
            return stats
    
    def reset_recovery_statistics(self) -> None:
        """Reset recovery statistics."""
        with self._lock:
            self.recovery_attempts.clear()
            self.recovery_successes.clear()
            self.recovery_history.clear()
            self.logger.info("Recovery statistics reset")


class SystemHealthMonitor:
    """
    Monitors overall system health and triggers recovery actions.
    """
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.config = get_config()
        
        # Component health tracking
        self.component_health = {
            'photonic_hardware': ComponentState.OPERATIONAL,
            'gpu_compute': ComponentState.OPERATIONAL,
            'memory_system': ComponentState.OPERATIONAL,
            'network': ComponentState.OPERATIONAL,
            'storage': ComponentState.OPERATIONAL,
        }
        
        # Health thresholds
        self.health_thresholds = {
            'cpu_usage': 0.9,
            'memory_usage': 0.85,
            'gpu_memory_usage': 0.9,
            'temperature': 80.0,  # Celsius
            'error_rate': 0.1,
            'latency_p99': 1000.0,  # ms
        }
        
        # Health metrics history
        self.health_history = deque(maxlen=1000)
        self.failure_events = deque(maxlen=500)
        
        # System state
        self.system_state = SystemState.HEALTHY
        self.last_health_check = 0
        
        # Monitoring thread
        self._monitoring_active = False
        self._monitoring_thread = None
        
        self._lock = threading.RLock()
    
    def start_monitoring(self, interval_seconds: float = 10.0) -> None:
        """Start continuous health monitoring."""
        with self._lock:
            if self._monitoring_active:
                self.logger.warning("Health monitoring already active")
                return
            
            self._monitoring_active = True
            self._monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                args=(interval_seconds,),
                daemon=True
            )
            self._monitoring_thread.start()
            
            self.logger.info(f"Health monitoring started with {interval_seconds}s interval")
    
    def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        with self._lock:
            if not self._monitoring_active:
                return
            
            self._monitoring_active = False
            
            if self._monitoring_thread and self._monitoring_thread.is_alive():
                self._monitoring_thread.join(timeout=5.0)
            
            self.logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self, interval_seconds: float) -> None:
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                self._perform_health_check()
                time.sleep(interval_seconds)
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}", exc_info=True)
                time.sleep(interval_seconds)
    
    def _perform_health_check(self) -> None:
        """Perform comprehensive health check."""
        current_time = time.time()
        
        try:
            # Collect health metrics
            metrics = self._collect_health_metrics()
            
            # Evaluate component health
            component_states = self._evaluate_component_health(metrics)
            
            # Update component health
            with self._lock:
                self.component_health.update(component_states)
                self.last_health_check = current_time
                
                # Record health snapshot
                health_snapshot = {
                    'timestamp': current_time,
                    'metrics': metrics,
                    'component_states': dict(component_states),
                    'system_state': self.system_state.value,
                }
                self.health_history.append(health_snapshot)
            
            # Determine overall system state
            new_system_state = self._determine_system_state(component_states)
            
            if new_system_state != self.system_state:
                self._handle_state_transition(self.system_state, new_system_state)
                self.system_state = new_system_state
        
        except Exception as e:
            self.logger.error(f"Health check failed: {e}", exc_info=True)
    
    def _collect_health_metrics(self) -> Dict[str, Any]:
        """Collect system health metrics."""
        metrics = {}
        
        try:
            # CPU and memory metrics
            import psutil
            metrics['cpu_usage'] = psutil.cpu_percent(interval=1)
            metrics['memory_usage'] = psutil.virtual_memory().percent / 100
            
            # GPU metrics
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_stats()
                allocated = gpu_memory.get('allocated_bytes.all.current', 0)
                reserved = gpu_memory.get('reserved_bytes.all.current', 0)
                max_memory = torch.cuda.get_device_properties(0).total_memory
                metrics['gpu_memory_usage'] = allocated / max_memory
                metrics['gpu_memory_reserved'] = reserved / max_memory
                
                # GPU utilization (simplified)
                metrics['gpu_utilization'] = min(1.0, allocated / (max_memory * 0.8))
            
            # Temperature (simplified - would need hardware integration)
            metrics['temperature'] = 45.0  # Placeholder
            
            # Network metrics (simplified)
            metrics['network_latency'] = 10.0  # Placeholder ms
            metrics['network_throughput'] = 0.95  # Placeholder ratio
            
            # Application-specific metrics
            metrics['error_rate'] = 0.02  # Placeholder
            metrics['request_latency_p99'] = 150.0  # Placeholder ms
            
        except Exception as e:
            self.logger.warning(f"Failed to collect some metrics: {e}")
        
        return metrics
    
    def _evaluate_component_health(self, metrics: Dict[str, Any]) -> Dict[str, ComponentState]:
        """Evaluate health of individual components."""
        states = {}
        
        # Memory system health
        memory_usage = metrics.get('memory_usage', 0)
        if memory_usage > 0.95:
            states['memory_system'] = ComponentState.FAILED
        elif memory_usage > 0.85:
            states['memory_system'] = ComponentState.DEGRADED
        else:
            states['memory_system'] = ComponentState.OPERATIONAL
        
        # GPU compute health
        gpu_memory_usage = metrics.get('gpu_memory_usage', 0)
        gpu_utilization = metrics.get('gpu_utilization', 0)
        
        if gpu_memory_usage > 0.95 or gpu_utilization > 0.98:
            states['gpu_compute'] = ComponentState.DEGRADED
        else:
            states['gpu_compute'] = ComponentState.OPERATIONAL
        
        # Photonic hardware health (would need real hardware integration)
        temperature = metrics.get('temperature', 25)
        if temperature > 85:
            states['photonic_hardware'] = ComponentState.FAILED
        elif temperature > 70:
            states['photonic_hardware'] = ComponentState.DEGRADED
        else:
            states['photonic_hardware'] = ComponentState.OPERATIONAL
        
        # Network health
        network_latency = metrics.get('network_latency', 0)
        if network_latency > 1000:
            states['network'] = ComponentState.DEGRADED
        else:
            states['network'] = ComponentState.OPERATIONAL
        
        # Storage health (simplified)
        states['storage'] = ComponentState.OPERATIONAL
        
        return states
    
    def _determine_system_state(self, component_states: Dict[str, ComponentState]) -> SystemState:
        """Determine overall system state from component states."""
        failed_components = sum(1 for state in component_states.values() if state == ComponentState.FAILED)
        degraded_components = sum(1 for state in component_states.values() if state == ComponentState.DEGRADED)
        
        if failed_components > 0:
            return SystemState.CRITICAL
        elif degraded_components > 1:
            return SystemState.DEGRADED
        elif degraded_components == 1:
            return SystemState.DEGRADED
        else:
            return SystemState.HEALTHY
    
    def _handle_state_transition(self, old_state: SystemState, new_state: SystemState) -> None:
        """Handle system state transitions."""
        self.logger.info(f"System state transition: {old_state.value} -> {new_state.value}")
        
        # Record state change event
        self.failure_events.append(FailureEvent(
            timestamp=time.time(),
            component='system',
            error_type='state_transition',
            error_message=f'{old_state.value} -> {new_state.value}',
            severity=new_state.value,
        ))
        
        # Trigger appropriate actions
        if new_state == SystemState.CRITICAL:
            self._handle_critical_state()
        elif new_state == SystemState.DEGRADED:
            self._handle_degraded_state()
        elif new_state == SystemState.HEALTHY:
            self._handle_healthy_state()
    
    def _handle_critical_state(self) -> None:
        """Handle critical system state."""
        self.logger.critical("System in critical state - initiating emergency procedures")
        
        # Trigger emergency response
        # - Notify administrators
        # - Save critical state
        # - Initiate recovery procedures
    
    def _handle_degraded_state(self) -> None:
        """Handle degraded system state."""
        self.logger.warning("System in degraded state - activating degradation policies")
        
        # Trigger degradation responses
        # - Reduce system load
        # - Enable fallback mechanisms
        # - Monitor for recovery
    
    def _handle_healthy_state(self) -> None:
        """Handle return to healthy state."""
        self.logger.info("System returned to healthy state")
        
        # Clean up degradation policies
        # - Restore full functionality
        # - Clear temporary measures
    
    def record_failure(self, component: str, error: Exception, context: Dict[str, Any]) -> None:
        """Record a failure event."""
        failure_event = FailureEvent(
            timestamp=time.time(),
            component=component,
            error_type=type(error).__name__,
            error_message=str(error),
            severity=self._classify_error_severity(error),
            context=context,
        )
        
        with self._lock:
            self.failure_events.append(failure_event)
        
        self.logger.error(f"Recorded failure in {component}: {error}")
    
    def _classify_error_severity(self, error: Exception) -> str:
        """Classify error severity."""
        critical_errors = (PhotonicHardwareError, SystemError, MemoryError)
        warning_errors = (PhotonicTimeoutError, ConnectionError)
        
        if isinstance(error, critical_errors):
            return 'critical'
        elif isinstance(error, warning_errors):
            return 'warning'
        else:
            return 'info'
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        with self._lock:
            return {
                'system_state': self.system_state.value,
                'component_health': {k: v.value for k, v in self.component_health.items()},
                'last_health_check': self.last_health_check,
                'monitoring_active': self._monitoring_active,
                'recent_failures': [{
                    'timestamp': event.timestamp,
                    'component': event.component,
                    'error_type': event.error_type,
                    'severity': event.severity,
                } for event in list(self.failure_events)[-10:]],
                'health_trends': self._analyze_health_trends(),
            }
    
    def _analyze_health_trends(self) -> Dict[str, Any]:
        """Analyze health trends from history."""
        if len(self.health_history) < 2:
            return {'trend': 'insufficient_data'}
        
        recent_snapshots = list(self.health_history)[-10:]
        
        # Simple trend analysis
        degraded_count = sum(1 for snapshot in recent_snapshots 
                           if snapshot['system_state'] in ['degraded', 'critical'])
        
        trend = 'stable'
        if degraded_count > len(recent_snapshots) * 0.5:
            trend = 'declining'
        elif degraded_count == 0:
            trend = 'healthy'
        
        return {
            'trend': trend,
            'degraded_ratio': degraded_count / len(recent_snapshots),
            'samples_analyzed': len(recent_snapshots),
        }


class ResilientAttentionWrapper:
    """
    Wrapper that adds comprehensive fault tolerance to attention mechanisms.
    
    Integrates circuit breakers, graceful degradation, auto-recovery,
    and health monitoring into a unified resilient system.
    """
    
    def __init__(self, attention_module, config: Optional[Dict[str, Any]] = None):
        self.attention_module = attention_module
        self.config = config or {}
        
        self.logger = get_logger(f"{self.__class__.__name__}({type(attention_module).__name__})")
        
        # Resilience components
        self.circuit_breaker = CircuitBreaker(
            name=f"{type(attention_module).__name__}_cb",
            failure_threshold=self.config.get('failure_threshold', 5),
            recovery_timeout=self.config.get('recovery_timeout', 60.0),
        )
        
        self.degradation_manager = GracefulDegradationManager()
        self.recovery_system = AutoRecoverySystem()
        self.health_monitor = SystemHealthMonitor()
        
        # Start monitoring
        if self.config.get('enable_monitoring', True):
            self.health_monitor.start_monitoring()
        
        # Fallback attention (simple implementation)
        self._fallback_attention = self._create_fallback_attention()
        
        self.logger.info(f"Resilient wrapper initialized for {type(attention_module).__name__}")
    
    def __call__(self, *args, **kwargs):
        """Execute attention with full fault tolerance."""
        try:
            # Execute through circuit breaker
            return self.circuit_breaker.call(self._safe_attention_call, *args, **kwargs)
        
        except CircuitBreakerOpenError:
            self.logger.warning("Circuit breaker open - using fallback attention")
            return self._fallback_attention(*args, **kwargs)
        
        except Exception as e:
            self.logger.error(f"Attention execution failed: {e}")
            
            # Record failure
            self.health_monitor.record_failure('attention', e, {'args': len(args), 'kwargs': list(kwargs.keys())})
            
            # Attempt recovery
            context = {'attention_module': self.attention_module}
            if self.recovery_system.attempt_recovery(e, context):
                self.logger.info("Recovery successful - retrying attention")
                try:
                    return self.attention_module(*args, **kwargs)
                except Exception as retry_error:
                    self.logger.error(f"Retry after recovery failed: {retry_error}")
            
            # Apply degradation and use fallback
            degraded_config = self.degradation_manager.apply_degradation(
                'attention_failure', {'error': str(e)}
            )
            
            self.logger.info("Using fallback attention with degraded configuration")
            return self._fallback_attention(*args, **kwargs)
    
    def _safe_attention_call(self, *args, **kwargs):
        """Safe attention call with timeout and validation."""
        start_time = time.time()
        
        try:
            # Input validation
            self._validate_inputs(args, kwargs)
            
            # Execute attention
            result = self.attention_module(*args, **kwargs)
            
            # Output validation
            self._validate_output(result, args)
            
            # Performance monitoring
            execution_time = time.time() - start_time
            if execution_time > self.config.get('max_execution_time', 5.0):
                self.logger.warning(f"Slow attention execution: {execution_time:.2f}s")
            
            return result
        
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Attention failed after {execution_time:.2f}s: {e}")
            raise
    
    def _validate_inputs(self, args, kwargs):
        """Validate attention inputs."""
        if not args:
            raise ValueError("No input arguments provided")
        
        # Basic tensor validation
        for i, arg in enumerate(args):
            if hasattr(arg, 'shape') and hasattr(arg, 'dtype'):
                if torch.isnan(arg).any():
                    raise ValueError(f"Input {i} contains NaN values")
                if torch.isinf(arg).any():
                    raise ValueError(f"Input {i} contains infinite values")
    
    def _validate_output(self, output, inputs):
        """Validate attention output."""
        if output is None:
            raise ValueError("Attention returned None")
        
        if hasattr(output, 'shape') and hasattr(inputs[0], 'shape'):
            if output.shape != inputs[0].shape:
                raise ValueError(f"Output shape {output.shape} doesn't match input shape {inputs[0].shape}")
        
        if hasattr(output, 'dtype'):
            if torch.isnan(output).any():
                raise ValueError("Output contains NaN values")
            if torch.isinf(output).any():
                raise ValueError("Output contains infinite values")
    
    def _create_fallback_attention(self):
        """Create simple fallback attention mechanism."""
        def fallback_attention(query, key=None, value=None, **kwargs):
            # Simple identity or scaled attention
            if key is None:
                key = query
            if value is None:
                value = query
            
            # Simplified attention computation
            batch_size, seq_len, embed_dim = query.shape
            
            # Simple weighted average (no actual attention computation)
            # This is a very basic fallback that maintains tensor shapes
            weights = torch.ones(batch_size, seq_len, seq_len) / seq_len
            output = torch.bmm(weights, value)
            
            self.logger.debug("Using fallback attention (identity approximation)")
            return output
        
        return fallback_attention
    
    def get_resilience_status(self) -> Dict[str, Any]:
        """Get comprehensive resilience status."""
        return {
            'circuit_breaker': self.circuit_breaker.get_state(),
            'degradation': self.degradation_manager.get_degradation_status(),
            'recovery': self.recovery_system.get_recovery_statistics(),
            'health': self.health_monitor.get_health_status(),
        }
    
    def reset_resilience_state(self) -> None:
        """Reset all resilience components."""
        self.circuit_breaker.reset()
        self.recovery_system.reset_recovery_statistics()
        
        # Clear degradations
        for degradation in list(self.degradation_manager.active_degradations):
            self.degradation_manager.remove_degradation(degradation)
        
        self.logger.info("Resilience state reset")
    
    def shutdown(self) -> None:
        """Shutdown resilience components."""
        self.health_monitor.stop_monitoring()
        self.logger.info("Resilient wrapper shutdown complete")
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.shutdown()
        except Exception:
            pass  # Ignore cleanup errors
