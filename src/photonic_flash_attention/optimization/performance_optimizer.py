"""Performance optimization and auto-tuning for photonic attention."""

import time
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict
import pickle
import json
from pathlib import Path
import logging

from ..utils.logging import get_logger
from ..config import get_config


logger = get_logger(__name__)


@dataclass
class PerformanceProfile:
    """Performance profile for workload characteristics."""
    batch_size: int
    seq_length: int
    embed_dim: int
    num_heads: int
    device_type: str
    avg_latency_ms: float
    avg_throughput_ops_per_sec: float
    memory_usage_mb: float
    energy_consumption_mj: float
    sample_count: int = 0
    last_updated: float = field(default_factory=time.time)


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization."""
    enable_caching: bool = True
    cache_size_mb: int = 512
    enable_batching: bool = True
    max_batch_size: int = 32
    batch_timeout_ms: float = 10.0
    enable_prefetching: bool = True
    prefetch_queue_size: int = 16
    enable_parallelization: bool = True
    max_workers: int = 4
    enable_autotuning: bool = True
    autotuning_samples: int = 100
    performance_cache_file: str = "photonic_performance_cache.json"


class MemoryPool:
    """Memory pool for efficient tensor allocation."""
    
    def __init__(self, max_size_mb: int = 512):
        """Initialize memory pool."""
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.pools: Dict[Tuple[int, ...], deque] = defaultdict(deque)
        self.allocated_size = 0
        self._lock = threading.RLock()
        
        logger.info(f"Memory pool initialized: {max_size_mb} MB limit")
    
    def get_tensor(self, shape: Tuple[int, ...], dtype_size: int = 4) -> Optional[Any]:
        """Get tensor from pool or None if not available."""
        try:
            # Try to import torch dynamically
            import torch
            
            with self._lock:
                pool = self.pools[shape]
                if pool:
                    tensor = pool.popleft()
                    logger.debug(f"Retrieved tensor from pool: {shape}")
                    return tensor
            
            return None
            
        except ImportError:
            # Torch not available, return None
            return None
    
    def return_tensor(self, tensor: Any, shape: Tuple[int, ...]) -> None:
        """Return tensor to pool."""
        try:
            import torch
            
            if not isinstance(tensor, torch.Tensor):
                return
            
            tensor_size = tensor.numel() * tensor.element_size()
            
            with self._lock:
                if self.allocated_size + tensor_size <= self.max_size_bytes:
                    # Clear tensor data for security
                    tensor.zero_()
                    
                    pool = self.pools[shape]
                    pool.append(tensor)
                    self.allocated_size += tensor_size
                    
                    logger.debug(f"Returned tensor to pool: {shape}")
                else:
                    logger.debug(f"Pool full, discarding tensor: {shape}")
                    
        except ImportError:
            # Torch not available, ignore
            pass
    
    def clear(self) -> None:
        """Clear all pools."""
        with self._lock:
            self.pools.clear()
            self.allocated_size = 0
            logger.info("Memory pool cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self._lock:
            total_tensors = sum(len(pool) for pool in self.pools.values())
            return {
                "total_pools": len(self.pools),
                "total_tensors": total_tensors,
                "allocated_size_mb": self.allocated_size / (1024 * 1024),
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                "utilization": self.allocated_size / self.max_size_bytes
            }


class BatchProcessor:
    """Batches requests for efficient processing."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize batch processor."""
        self.config = config
        self.batch_queue: deque = deque()
        self.batch_futures: List = []
        self.processing = False
        self._lock = threading.Lock()
        self._batch_ready = threading.Event()
        
        if config.enable_batching:
            self._start_batch_processor()
        
        logger.info(f"Batch processor initialized: max_batch={config.max_batch_size}")
    
    def _start_batch_processor(self) -> None:
        """Start batch processing thread."""
        self.processing = True
        self.batch_thread = threading.Thread(target=self._batch_processing_loop, daemon=True)
        self.batch_thread.start()
    
    def _batch_processing_loop(self) -> None:
        """Main batch processing loop."""
        while self.processing:
            try:
                # Wait for batch to be ready or timeout
                batch_ready = self._batch_ready.wait(self.config.batch_timeout_ms / 1000.0)
                
                if batch_ready or len(self.batch_queue) >= self.config.max_batch_size:
                    self._process_current_batch()
                
            except Exception as e:
                logger.error(f"Error in batch processing loop: {e}")
    
    def _process_current_batch(self) -> None:
        """Process current batch of requests."""
        with self._lock:
            if not self.batch_queue:
                return
            
            # Extract batch
            batch = []
            while self.batch_queue and len(batch) < self.config.max_batch_size:
                batch.append(self.batch_queue.popleft())
            
            self._batch_ready.clear()
        
        if batch:
            logger.debug(f"Processing batch of {len(batch)} requests")
            # Process batch (implementation specific)
            self._execute_batch(batch)
    
    def _execute_batch(self, batch: List[Any]) -> None:
        """Execute batch processing (to be implemented by subclasses)."""
        # This would be implemented by specific attention modules
        pass
    
    def add_to_batch(self, request: Any) -> None:
        """Add request to batch queue."""
        if not self.config.enable_batching:
            return
        
        with self._lock:
            self.batch_queue.append(request)
            
            if len(self.batch_queue) >= self.config.max_batch_size:
                self._batch_ready.set()
    
    def stop(self) -> None:
        """Stop batch processing."""
        self.processing = False
        self._batch_ready.set()
        
        if hasattr(self, 'batch_thread'):
            self.batch_thread.join(timeout=5.0)


class PerformanceCache:
    """Cache for performance optimization data."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize performance cache."""
        self.config = config
        self.profiles: Dict[str, PerformanceProfile] = {}
        self.cache_file = Path(config.performance_cache_file)
        self._lock = threading.RLock()
        
        self._load_cache()
        logger.info(f"Performance cache initialized: {len(self.profiles)} profiles loaded")
    
    def _load_cache(self) -> None:
        """Load performance profiles from cache file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    
                for key, profile_data in data.items():
                    profile = PerformanceProfile(**profile_data)
                    self.profiles[key] = profile
                    
                logger.info(f"Loaded {len(self.profiles)} performance profiles from cache")
                
            except Exception as e:
                logger.warning(f"Failed to load performance cache: {e}")
    
    def _save_cache(self) -> None:
        """Save performance profiles to cache file."""
        try:
            # Create directory if needed
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert profiles to dict for JSON serialization
            data = {}
            for key, profile in self.profiles.items():
                data[key] = {
                    'batch_size': profile.batch_size,
                    'seq_length': profile.seq_length,
                    'embed_dim': profile.embed_dim,
                    'num_heads': profile.num_heads,
                    'device_type': profile.device_type,
                    'avg_latency_ms': profile.avg_latency_ms,
                    'avg_throughput_ops_per_sec': profile.avg_throughput_ops_per_sec,
                    'memory_usage_mb': profile.memory_usage_mb,
                    'energy_consumption_mj': profile.energy_consumption_mj,
                    'sample_count': profile.sample_count,
                    'last_updated': profile.last_updated
                }
            
            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.debug(f"Saved {len(self.profiles)} performance profiles to cache")
            
        except Exception as e:
            logger.error(f"Failed to save performance cache: {e}")
    
    def get_profile_key(
        self, 
        batch_size: int, 
        seq_length: int, 
        embed_dim: int, 
        num_heads: int,
        device_type: str
    ) -> str:
        """Generate key for performance profile."""
        # Round values to reduce fragmentation
        batch_size = ((batch_size - 1) // 4 + 1) * 4  # Round to nearest 4
        seq_length = ((seq_length - 1) // 64 + 1) * 64  # Round to nearest 64
        
        return f"{device_type}_{batch_size}_{seq_length}_{embed_dim}_{num_heads}"
    
    def get_performance_profile(
        self, 
        batch_size: int, 
        seq_length: int, 
        embed_dim: int, 
        num_heads: int,
        device_type: str
    ) -> Optional[PerformanceProfile]:
        """Get performance profile for workload."""
        key = self.get_profile_key(batch_size, seq_length, embed_dim, num_heads, device_type)
        
        with self._lock:
            return self.profiles.get(key)
    
    def update_performance_profile(
        self,
        batch_size: int,
        seq_length: int, 
        embed_dim: int,
        num_heads: int,
        device_type: str,
        latency_ms: float,
        throughput_ops_per_sec: float,
        memory_usage_mb: float,
        energy_consumption_mj: float = 0.0
    ) -> None:
        """Update performance profile with new measurement."""
        key = self.get_profile_key(batch_size, seq_length, embed_dim, num_heads, device_type)
        
        with self._lock:
            if key in self.profiles:
                # Update existing profile with exponential moving average
                profile = self.profiles[key]
                alpha = 0.1  # Learning rate
                
                profile.avg_latency_ms = (
                    (1 - alpha) * profile.avg_latency_ms + alpha * latency_ms
                )
                profile.avg_throughput_ops_per_sec = (
                    (1 - alpha) * profile.avg_throughput_ops_per_sec + alpha * throughput_ops_per_sec
                )
                profile.memory_usage_mb = (
                    (1 - alpha) * profile.memory_usage_mb + alpha * memory_usage_mb
                )
                profile.energy_consumption_mj = (
                    (1 - alpha) * profile.energy_consumption_mj + alpha * energy_consumption_mj
                )
                profile.sample_count += 1
                profile.last_updated = time.time()
                
            else:
                # Create new profile
                profile = PerformanceProfile(
                    batch_size=batch_size,
                    seq_length=seq_length,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    device_type=device_type,
                    avg_latency_ms=latency_ms,
                    avg_throughput_ops_per_sec=throughput_ops_per_sec,
                    memory_usage_mb=memory_usage_mb,
                    energy_consumption_mj=energy_consumption_mj,
                    sample_count=1
                )
                self.profiles[key] = profile
            
            # Periodically save cache
            if profile.sample_count % 10 == 0:
                self._save_cache()
    
    def get_best_device_for_workload(
        self, 
        batch_size: int, 
        seq_length: int, 
        embed_dim: int, 
        num_heads: int
    ) -> str:
        """Get best device type for given workload."""
        devices = ['photonic', 'gpu', 'cpu']
        best_device = 'gpu'  # Default
        best_score = float('inf')
        
        with self._lock:
            for device in devices:
                profile = self.get_performance_profile(
                    batch_size, seq_length, embed_dim, num_heads, device
                )
                
                if profile and profile.sample_count >= 3:
                    # Weighted score: latency * energy
                    score = profile.avg_latency_ms * (1 + profile.energy_consumption_mj)
                    
                    if score < best_score:
                        best_score = score
                        best_device = device
        
        return best_device


class AutoTuner:
    """Automatic performance tuning system."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize auto-tuner."""
        self.config = config
        self.tuning_active = config.enable_autotuning
        self.performance_cache = PerformanceCache(config)
        self.tuning_jobs: List = []
        self.executor: Optional[ThreadPoolExecutor] = None
        
        if self.tuning_active:
            self._start_autotuner()
        
        logger.info(f"Auto-tuner initialized: active={self.tuning_active}")
    
    def _start_autotuner(self) -> None:
        """Start auto-tuning background process."""
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="autotuner")
    
    def suggest_optimal_device(
        self, 
        batch_size: int, 
        seq_length: int, 
        embed_dim: int, 
        num_heads: int
    ) -> str:
        """Suggest optimal device for workload."""
        return self.performance_cache.get_best_device_for_workload(
            batch_size, seq_length, embed_dim, num_heads
        )
    
    def record_performance(
        self,
        batch_size: int,
        seq_length: int,
        embed_dim: int, 
        num_heads: int,
        device_type: str,
        latency_ms: float,
        memory_usage_mb: float = 0.0,
        energy_mj: float = 0.0
    ) -> None:
        """Record performance measurement."""
        # Calculate throughput
        ops_count = batch_size * seq_length * seq_length * embed_dim
        throughput = ops_count / (latency_ms / 1000.0) if latency_ms > 0 else 0.0
        
        self.performance_cache.update_performance_profile(
            batch_size=batch_size,
            seq_length=seq_length,
            embed_dim=embed_dim,
            num_heads=num_heads,
            device_type=device_type,
            latency_ms=latency_ms,
            throughput_ops_per_sec=throughput,
            memory_usage_mb=memory_usage_mb,
            energy_consumption_mj=energy_mj
        )
    
    def benchmark_workload(
        self,
        batch_size: int,
        seq_length: int, 
        embed_dim: int,
        num_heads: int,
        benchmark_func: Callable,
        devices: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """Benchmark workload across different devices."""
        if devices is None:
            devices = ['gpu', 'photonic']
        
        results = {}
        
        for device in devices:
            try:
                # Run benchmark
                start_time = time.perf_counter()
                benchmark_func(device)
                end_time = time.perf_counter()
                
                latency_ms = (end_time - start_time) * 1000
                
                # Record result
                self.record_performance(
                    batch_size, seq_length, embed_dim, num_heads,
                    device, latency_ms
                )
                
                results[device] = {
                    'latency_ms': latency_ms,
                    'device': device
                }
                
                logger.debug(f"Benchmarked {device}: {latency_ms:.2f}ms")
                
            except Exception as e:
                logger.warning(f"Benchmark failed for {device}: {e}")
                results[device] = {
                    'error': str(e),
                    'device': device
                }
        
        return results
    
    def get_optimization_recommendations(
        self, 
        batch_size: int, 
        seq_length: int, 
        embed_dim: int, 
        num_heads: int
    ) -> Dict[str, Any]:
        """Get optimization recommendations for workload."""
        optimal_device = self.suggest_optimal_device(batch_size, seq_length, embed_dim, num_heads)
        
        profile = self.performance_cache.get_performance_profile(
            batch_size, seq_length, embed_dim, num_heads, optimal_device
        )
        
        recommendations = {
            'optimal_device': optimal_device,
            'confidence': 'high' if profile and profile.sample_count >= 10 else 'low',
            'expected_latency_ms': profile.avg_latency_ms if profile else None,
            'expected_memory_mb': profile.memory_usage_mb if profile else None,
        }
        
        # Add specific recommendations
        if seq_length > 2048:
            recommendations['suggestions'] = [
                'Consider using photonic device for long sequences',
                'Enable gradient checkpointing to reduce memory usage'
            ]
        elif batch_size > 16:
            recommendations['suggestions'] = [
                'Consider sequence parallelization for large batches',
                'Enable memory pooling for efficient allocation'
            ]
        else:
            recommendations['suggestions'] = [
                'GPU device recommended for small workloads'
            ]
        
        return recommendations
    
    def stop(self) -> None:
        """Stop auto-tuner."""
        if self.executor:
            self.executor.shutdown(wait=True)
        
        logger.info("Auto-tuner stopped")


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        """Initialize performance optimizer."""
        self.config = config or OptimizationConfig()
        self.memory_pool = MemoryPool(self.config.cache_size_mb) if self.config.enable_caching else None
        self.batch_processor = BatchProcessor(self.config) if self.config.enable_batching else None
        self.autotuner = AutoTuner(self.config) if self.config.enable_autotuning else None
        
        # Performance tracking
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.avg_latency_ms = 0.0
        self._lock = threading.RLock()
        
        logger.info("Performance optimizer initialized")
    
    def optimize_attention_call(
        self,
        batch_size: int,
        seq_length: int,
        embed_dim: int,
        num_heads: int,
        attention_func: Callable,
        **kwargs
    ) -> Any:
        """Optimize attention function call."""
        with self._lock:
            self.total_requests += 1
        
        # Get optimization recommendations
        if self.autotuner:
            recommendations = self.autotuner.get_optimization_recommendations(
                batch_size, seq_length, embed_dim, num_heads
            )
            optimal_device = recommendations['optimal_device']
        else:
            optimal_device = 'gpu'  # Default
        
        # Execute optimized call
        start_time = time.perf_counter()
        
        try:
            result = attention_func(device=optimal_device, **kwargs)
            
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            
            # Record performance
            if self.autotuner:
                self.autotuner.record_performance(
                    batch_size, seq_length, embed_dim, num_heads,
                    optimal_device, latency_ms
                )
            
            # Update average latency
            with self._lock:
                alpha = 0.1
                self.avg_latency_ms = (
                    (1 - alpha) * self.avg_latency_ms + alpha * latency_ms
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Optimized attention call failed: {e}")
            raise
    
    def get_memory_tensor(self, shape: Tuple[int, ...]) -> Optional[Any]:
        """Get tensor from memory pool."""
        if self.memory_pool:
            tensor = self.memory_pool.get_tensor(shape)
            if tensor is not None:
                with self._lock:
                    self.cache_hits += 1
                return tensor
        
        with self._lock:
            self.cache_misses += 1
        return None
    
    def return_memory_tensor(self, tensor: Any, shape: Tuple[int, ...]) -> None:
        """Return tensor to memory pool."""
        if self.memory_pool:
            self.memory_pool.return_tensor(tensor, shape)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        with self._lock:
            stats = {
                'total_requests': self.total_requests,
                'avg_latency_ms': self.avg_latency_ms,
                'cache_stats': {
                    'hits': self.cache_hits,
                    'misses': self.cache_misses,
                    'hit_rate': self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
                }
            }
            
            if self.memory_pool:
                stats['memory_pool'] = self.memory_pool.get_stats()
            
            if self.autotuner:
                stats['autotuning_active'] = self.autotuner.tuning_active
                stats['performance_profiles'] = len(self.autotuner.performance_cache.profiles)
            
            return stats
    
    def shutdown(self) -> None:
        """Shutdown optimizer and cleanup resources."""
        if self.batch_processor:
            self.batch_processor.stop()
        
        if self.autotuner:
            self.autotuner.stop()
        
        if self.memory_pool:
            self.memory_pool.clear()
        
        logger.info("Performance optimizer shutdown complete")


# Global optimizer instance
_performance_optimizer: Optional[PerformanceOptimizer] = None


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer."""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer()
    return _performance_optimizer


def optimize_attention_call(
    batch_size: int,
    seq_length: int, 
    embed_dim: int,
    num_heads: int,
    attention_func: Callable,
    **kwargs
) -> Any:
    """Optimize attention function call (convenience function)."""
    optimizer = get_performance_optimizer()
    return optimizer.optimize_attention_call(
        batch_size, seq_length, embed_dim, num_heads, attention_func, **kwargs
    )


def get_optimization_stats() -> Dict[str, Any]:
    """Get optimization statistics (convenience function)."""
    optimizer = get_performance_optimizer()
    return optimizer.get_performance_stats()