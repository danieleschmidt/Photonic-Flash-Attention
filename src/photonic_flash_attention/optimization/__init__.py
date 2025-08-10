"""Performance optimization and scaling for photonic attention."""

from .performance_optimizer import (
    PerformanceOptimizer,
    OptimizationConfig,
    PerformanceProfile,
    MemoryPool,
    BatchProcessor,
    PerformanceCache,
    AutoTuner,
    get_performance_optimizer,
    optimize_attention_call,
    get_optimization_stats
)

__all__ = [
    "PerformanceOptimizer",
    "OptimizationConfig", 
    "PerformanceProfile",
    "MemoryPool",
    "BatchProcessor",
    "PerformanceCache",
    "AutoTuner",
    "get_performance_optimizer",
    "optimize_attention_call", 
    "get_optimization_stats"
]