"""
Performance optimization modules for photonic flash attention.

This package provides comprehensive performance optimization capabilities
including adaptive optimization, caching, parallelization, and workload analysis.
"""

from .performance_optimizer import (
    AdaptiveOptimizer,
    WorkloadProfiler,
    CacheManager,
    OptimizationLevel,
    OptimizationTarget,
    WorkloadType,
    get_performance_optimizer,
    optimize_function
)

__all__ = [
    'AdaptiveOptimizer',
    'WorkloadProfiler', 
    'CacheManager',
    'OptimizationLevel',
    'OptimizationTarget',
    'WorkloadType',
    'get_performance_optimizer',
    'optimize_function'
]