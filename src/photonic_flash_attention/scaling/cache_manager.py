"""
Intelligent caching and memory optimization for photonic attention.

This module provides adaptive caching, memory pooling, and tensor optimization
for high-performance photonic attention computations.
"""

import time
import threading
import weakref
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict, defaultdict
import hashlib
import pickle
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..utils.logging import get_logger
from ..utils.exceptions import PhotonicComputationError


logger = get_logger(__name__)


class CachePolicy(Enum):
    """Cache replacement policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    ADAPTIVE = "adaptive"  # Adaptive Replacement Cache
    TTL = "ttl"  # Time To Live


class ComputationCacheEntry:
    """Entry in the computation cache."""
    
    def __init__(self, key: str, result: Any, cost: float = 1.0):
        self.key = key
        self.result = result
        self.cost = cost
        self.access_count = 1
        self.creation_time = time.time()
        self.last_access_time = time.time()
        self.size_bytes = self._estimate_size(result)
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate memory size of object."""
        if TORCH_AVAILABLE and isinstance(obj, torch.Tensor):
            return obj.numel() * obj.element_size()
        
        try:
            return len(pickle.dumps(obj))
        except Exception:
            # Fallback estimate
            return 1024  # 1KB default


@dataclass
class CacheConfig:
    """Configuration for cache manager."""
    max_memory_bytes: int = 2 * 1024**3  # 2GB
    max_entries: int = 10000
    policy: CachePolicy = CachePolicy.ADAPTIVE
    ttl_seconds: float = 3600.0  # 1 hour
    cleanup_interval: float = 300.0  # 5 minutes
    enable_compression: bool = True
    compression_threshold: int = 1024  # Compress objects > 1KB
    enable_prefetching: bool = True
    prefetch_window: int = 3  # Prefetch next 3 likely items
    memory_pressure_threshold: float = 0.8  # Start eviction at 80% memory


class AdaptiveCache:
    """Adaptive cache with multiple replacement policies."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        
        # Cache storage
        self.cache: Dict[str, ComputationCacheEntry] = {}
        self.access_order = OrderedDict()  # For LRU
        self.access_counts = defaultdict(int)  # For LFU
        self.insertion_order = OrderedDict()  # For FIFO
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.current_memory = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background cleanup
        self._start_cleanup_thread()
        
        logger.info(f"Adaptive cache initialized: {config.max_memory_bytes / 1024**2:.0f}MB capacity")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            entry = self.cache[key]
            
            # Check TTL
            if (self.config.policy == CachePolicy.TTL and 
                time.time() - entry.creation_time > self.config.ttl_seconds):
                self._remove_entry(key)
                self.misses += 1
                return None
            
            # Update access patterns
            self._update_access_patterns(key)
            self.hits += 1
            
            logger.debug(f"Cache hit: {key}")
            return entry.result
    
    def put(self, key: str, value: Any, cost: float = 1.0) -> bool:
        """Put item in cache."""
        with self._lock:
            # Create entry
            entry = ComputationCacheEntry(key, value, cost)
            
            # Check if we need to evict
            if not self._has_capacity(entry):
                if not self._make_space(entry.size_bytes):
                    logger.warning(f"Could not cache item {key}: insufficient space")
                    return False
            
            # Remove existing entry if present
            if key in self.cache:
                self._remove_entry(key)
            
            # Add new entry
            self.cache[key] = entry
            self.current_memory += entry.size_bytes
            
            # Update tracking structures
            self.access_order[key] = time.time()
            self.access_counts[key] = 1
            self.insertion_order[key] = time.time()
            
            logger.debug(f"Cached item: {key} ({entry.size_bytes} bytes)")
            return True
    
    def _has_capacity(self, entry: ComputationCacheEntry) -> bool:
        """Check if cache has capacity for entry."""
        return (
            len(self.cache) < self.config.max_entries and
            self.current_memory + entry.size_bytes <= self.config.max_memory_bytes
        )
    
    def _make_space(self, required_bytes: int) -> bool:
        """Make space in cache by evicting items."""
        if self.config.policy == CachePolicy.LRU:
            return self._evict_lru(required_bytes)
        elif self.config.policy == CachePolicy.LFU:
            return self._evict_lfu(required_bytes)
        elif self.config.policy == CachePolicy.FIFO:
            return self._evict_fifo(required_bytes)
        elif self.config.policy == CachePolicy.ADAPTIVE:
            return self._evict_adaptive(required_bytes)
        else:
            return self._evict_lru(required_bytes)  # Default
    
    def _evict_lru(self, required_bytes: int) -> bool:
        """Evict least recently used items."""
        freed_bytes = 0
        
        # Sort by access time (oldest first)
        lru_keys = sorted(self.access_order.keys(), key=lambda k: self.access_order[k])
        
        for key in lru_keys:
            if freed_bytes >= required_bytes:
                break
            
            entry = self.cache[key]
            freed_bytes += entry.size_bytes
            self._remove_entry(key)
            self.evictions += 1
        
        return freed_bytes >= required_bytes
    
    def _evict_lfu(self, required_bytes: int) -> bool:
        """Evict least frequently used items."""
        freed_bytes = 0
        
        # Sort by access count (lowest first)
        lfu_keys = sorted(self.access_counts.keys(), key=lambda k: self.access_counts[k])
        
        for key in lfu_keys:
            if key not in self.cache:
                continue
            if freed_bytes >= required_bytes:
                break
            
            entry = self.cache[key]
            freed_bytes += entry.size_bytes
            self._remove_entry(key)
            self.evictions += 1
        
        return freed_bytes >= required_bytes
    
    def _evict_fifo(self, required_bytes: int) -> bool:
        """Evict first-in-first-out items."""
        freed_bytes = 0
        
        # Sort by insertion time (oldest first)
        fifo_keys = sorted(self.insertion_order.keys(), key=lambda k: self.insertion_order[k])
        
        for key in fifo_keys:
            if freed_bytes >= required_bytes:
                break
            
            entry = self.cache[key]
            freed_bytes += entry.size_bytes
            self._remove_entry(key)
            self.evictions += 1
        
        return freed_bytes >= required_bytes
    
    def _evict_adaptive(self, required_bytes: int) -> bool:
        """Adaptive replacement cache (ARC) policy."""
        freed_bytes = 0
        current_time = time.time()
        
        # Score items based on multiple factors
        scored_items = []
        for key, entry in self.cache.items():
            recency_score = 1.0 / (current_time - entry.last_access_time + 1)
            frequency_score = self.access_counts[key]
            cost_score = 1.0 / entry.cost
            
            # Weighted combination
            total_score = recency_score * 0.4 + frequency_score * 0.4 + cost_score * 0.2
            scored_items.append((total_score, key))
        
        # Sort by score (lowest first for eviction)
        scored_items.sort()
        
        for score, key in scored_items:
            if freed_bytes >= required_bytes:
                break
            
            entry = self.cache[key]
            freed_bytes += entry.size_bytes
            self._remove_entry(key)
            self.evictions += 1
        
        return freed_bytes >= required_bytes
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry from all tracking structures."""
        if key in self.cache:
            entry = self.cache[key]
            self.current_memory -= entry.size_bytes
            del self.cache[key]
        
        self.access_order.pop(key, None)
        self.access_counts.pop(key, None)
        self.insertion_order.pop(key, None)
    
    def _update_access_patterns(self, key: str) -> None:
        """Update access patterns for cache policies."""
        current_time = time.time()
        
        self.access_order[key] = current_time
        self.access_counts[key] += 1
        
        if key in self.cache:
            self.cache[key].last_access_time = current_time
            self.cache[key].access_count += 1
    
    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread."""
        def cleanup_loop():
            while True:
                try:
                    time.sleep(self.config.cleanup_interval)
                    self._cleanup_expired_entries()
                    self._check_memory_pressure()
                except Exception as e:
                    logger.error(f"Cache cleanup error: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        cleanup_thread.start()
    
    def _cleanup_expired_entries(self) -> None:
        """Clean up expired TTL entries."""
        if self.config.policy != CachePolicy.TTL:
            return
        
        current_time = time.time()
        expired_keys = []
        
        with self._lock:
            for key, entry in self.cache.items():
                if current_time - entry.creation_time > self.config.ttl_seconds:
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_entry(key)
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _check_memory_pressure(self) -> None:
        """Check and handle memory pressure."""
        memory_usage = self.current_memory / self.config.max_memory_bytes
        
        if memory_usage > self.config.memory_pressure_threshold:
            # Aggressively evict items
            target_memory = int(self.config.max_memory_bytes * 0.7)  # Target 70%
            required_bytes = self.current_memory - target_memory
            
            with self._lock:
                self._make_space(required_bytes)
                logger.info(f"Memory pressure cleanup: freed {required_bytes} bytes")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / max(total_requests, 1)
            
            return {
                'entries': len(self.cache),
                'memory_bytes': self.current_memory,
                'memory_mb': self.current_memory / 1024**2,
                'memory_usage': self.current_memory / self.config.max_memory_bytes,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'evictions': self.evictions,
                'policy': self.config.policy.value
            }
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()
            self.access_order.clear()
            self.access_counts.clear()
            self.insertion_order.clear()
            self.current_memory = 0
            logger.info("Cache cleared")


class TensorMemoryPool:
    """Memory pool for tensor reuse and allocation optimization."""
    
    def __init__(self, max_pool_size: int = 100):
        self.max_pool_size = max_pool_size
        self.pools: Dict[Tuple[torch.Size, torch.dtype, str], List[torch.Tensor]] = defaultdict(list)
        self.allocation_count = 0
        self.reuse_count = 0
        self._lock = threading.RLock()
        
        logger.info(f"Tensor memory pool initialized: max_size={max_pool_size}")
    
    def get_tensor(self, shape: torch.Size, dtype: torch.dtype, device: str = 'cpu') -> torch.Tensor:
        """Get tensor from pool or allocate new one."""
        if not TORCH_AVAILABLE:
            raise PhotonicComputationError("PyTorch not available for tensor operations")
        
        pool_key = (shape, dtype, device)
        
        with self._lock:
            pool = self.pools[pool_key]
            
            if pool:
                tensor = pool.pop()
                tensor.zero_()  # Clear data
                self.reuse_count += 1
                logger.debug(f"Reused tensor from pool: {shape}")
                return tensor
            else:
                # Allocate new tensor
                device_obj = torch.device(device)
                tensor = torch.zeros(shape, dtype=dtype, device=device_obj)
                self.allocation_count += 1
                logger.debug(f"Allocated new tensor: {shape}")
                return tensor
    
    def return_tensor(self, tensor: torch.Tensor) -> None:
        """Return tensor to pool for reuse."""
        if not TORCH_AVAILABLE:
            return
        
        pool_key = (tensor.shape, tensor.dtype, str(tensor.device))
        
        with self._lock:
            pool = self.pools[pool_key]
            
            if len(pool) < self.max_pool_size:
                # Detach from computation graph
                if tensor.requires_grad:
                    tensor = tensor.detach()
                
                pool.append(tensor)
                logger.debug(f"Returned tensor to pool: {tensor.shape}")
            else:
                # Pool is full, let tensor be garbage collected
                pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self._lock:
            total_tensors = sum(len(pool) for pool in self.pools.values())
            total_requests = self.allocation_count + self.reuse_count
            reuse_rate = self.reuse_count / max(total_requests, 1)
            
            return {
                'pool_types': len(self.pools),
                'total_pooled_tensors': total_tensors,
                'allocations': self.allocation_count,
                'reuses': self.reuse_count,
                'reuse_rate': reuse_rate
            }


class ComputationCacheManager:
    """High-level cache manager for photonic attention computations."""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.computation_cache = AdaptiveCache(self.config)
        self.tensor_pool = TensorMemoryPool() if TORCH_AVAILABLE else None
        
        # Cache key generation
        self._key_cache = {}
        self._key_cache_lock = threading.RLock()
        
        logger.info("Computation cache manager initialized")
    
    def cached_computation(
        self,
        func: Callable,
        *args,
        cache_key_prefix: str = "",
        cache_ttl: Optional[float] = None,
        **kwargs
    ) -> Any:
        """Execute function with caching."""
        # Generate cache key
        cache_key = self._generate_cache_key(func, args, kwargs, cache_key_prefix)
        
        # Try to get from cache
        result = self.computation_cache.get(cache_key)
        if result is not None:
            return result
        
        # Compute result
        start_time = time.time()
        result = func(*args, **kwargs)
        computation_time = time.time() - start_time
        
        # Cache result (with cost based on computation time)
        cost = computation_time
        self.computation_cache.put(cache_key, result, cost)
        
        return result
    
    def _generate_cache_key(
        self,
        func: Callable,
        args: Tuple,
        kwargs: Dict[str, Any],
        prefix: str = ""
    ) -> str:
        """Generate cache key for function call."""
        # Create deterministic key from function and arguments
        func_name = func.__name__ if hasattr(func, '__name__') else str(func)
        
        # Serialize arguments (handle special types)
        serializable_args = []
        for arg in args:
            if TORCH_AVAILABLE and isinstance(arg, torch.Tensor):
                # Use tensor shape, dtype, and hash of first few elements
                tensor_sig = (tuple(arg.shape), str(arg.dtype), str(arg.device))
                if arg.numel() > 0:
                    # Sample a few elements for hash
                    sample_size = min(10, arg.numel())
                    sample = arg.flatten()[:sample_size]
                    tensor_hash = hash(tuple(sample.tolist()))
                    tensor_sig = (tensor_sig, tensor_hash)
                serializable_args.append(tensor_sig)
            else:
                try:
                    serializable_args.append(arg)
                except Exception:
                    serializable_args.append(str(arg))
        
        # Create hash
        key_data = {
            'prefix': prefix,
            'function': func_name,
            'args': serializable_args,
            'kwargs': kwargs
        }
        
        key_str = pickle.dumps(key_data, protocol=pickle.HIGHEST_PROTOCOL)
        key_hash = hashlib.sha256(key_str).hexdigest()
        
        return f"{prefix}_{func_name}_{key_hash[:16]}"
    
    def get_tensor(self, shape: torch.Size, dtype: torch.dtype, device: str = 'cpu') -> Optional[torch.Tensor]:
        """Get tensor from memory pool."""
        if self.tensor_pool:
            return self.tensor_pool.get_tensor(shape, dtype, device)
        return None
    
    def return_tensor(self, tensor: torch.Tensor) -> None:
        """Return tensor to memory pool."""
        if self.tensor_pool:
            self.tensor_pool.return_tensor(tensor)
    
    def prefetch_patterns(self, access_patterns: List[str]) -> None:
        """Prefetch cache entries based on access patterns."""
        if not self.config.enable_prefetching:
            return
        
        # Simple prefetching based on common patterns
        for pattern in access_patterns[-self.config.prefetch_window:]:
            # This could be enhanced with ML-based prediction
            cache_result = self.computation_cache.get(pattern)
            if cache_result:
                logger.debug(f"Prefetched cache entry: {pattern}")
    
    def warm_cache(self, warm_up_functions: List[Tuple[Callable, Tuple, Dict]]) -> None:
        """Warm up cache with common computations."""
        logger.info(f"Warming cache with {len(warm_up_functions)} functions")
        
        for func, args, kwargs in warm_up_functions:
            try:
                self.cached_computation(func, *args, **kwargs)
            except Exception as e:
                logger.warning(f"Cache warm-up failed for {func}: {e}")
        
        logger.info("Cache warm-up completed")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache and memory statistics."""
        stats = {
            'cache': self.computation_cache.get_stats(),
            'config': {
                'max_memory_mb': self.config.max_memory_bytes / 1024**2,
                'max_entries': self.config.max_entries,
                'policy': self.config.policy.value,
                'ttl_seconds': self.config.ttl_seconds
            }
        }
        
        if self.tensor_pool:
            stats['tensor_pool'] = self.tensor_pool.get_stats()
        
        return stats
    
    def optimize_memory(self) -> None:
        """Perform memory optimization."""
        logger.info("Performing memory optimization")
        
        # Force cleanup
        self.computation_cache._cleanup_expired_entries()
        self.computation_cache._check_memory_pressure()
        
        # Clear tensor pools if needed
        if self.tensor_pool and TORCH_AVAILABLE:
            # Clear large unused tensors
            with self.tensor_pool._lock:
                cleared_count = 0
                for pool_key, pool in list(self.tensor_pool.pools.items()):
                    if len(pool) > 10:  # Keep only 10 of each type
                        removed = pool[10:]
                        self.tensor_pool.pools[pool_key] = pool[:10]
                        cleared_count += len(removed)
                
                if cleared_count > 0:
                    logger.info(f"Cleared {cleared_count} excess pooled tensors")
        
        # Force garbage collection if available
        try:
            import gc
            gc.collect()
        except ImportError:
            pass


# Global cache manager instance
_cache_manager: Optional[ComputationCacheManager] = None


def get_cache_manager() -> ComputationCacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = ComputationCacheManager()
    return _cache_manager


def cached_computation(func: Callable, *args, **kwargs) -> Any:
    """Convenience function for cached computation."""
    cache_manager = get_cache_manager()
    return cache_manager.cached_computation(func, *args, **kwargs)


# Decorator for automatic caching
def cache_computation(cache_key_prefix: str = "", ttl: Optional[float] = None):
    """Decorator for automatic computation caching."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            cache_manager = get_cache_manager()
            return cache_manager.cached_computation(
                func, *args, cache_key_prefix=cache_key_prefix, cache_ttl=ttl, **kwargs
            )
        return wrapper
    return decorator