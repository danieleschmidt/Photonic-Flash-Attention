"""Advanced caching and memory pooling for high-performance photonic operations."""

import time
import threading
import weakref
import hashlib
import pickle
from typing import Dict, List, Optional, Any, Tuple, Union, Callable, Set
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
from enum import Enum
import numpy as np
import gc

from ..utils.logging import get_logger, get_performance_logger
from ..config import get_config

# Import torch conditionally for better compatibility
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None


class CacheLevel(Enum):
    """Cache levels with different characteristics."""
    L1_FAST = "l1_fast"         # Ultra-fast cache, small size
    L2_BALANCED = "l2_balanced" # Balanced speed/size cache
    L3_LARGE = "l3_large"       # Large cache, slower access
    PERSISTENT = "persistent"   # Disk-backed persistent cache


class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"                 # Least Recently Used
    LFU = "lfu"                 # Least Frequently Used  
    FIFO = "fifo"               # First In, First Out
    TTL = "ttl"                 # Time To Live
    ADAPTIVE = "adaptive"       # Adaptive based on access patterns


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    avg_access_time: float = 0.0
    hit_rate: float = 0.0
    
    def update_hit(self, access_time: float) -> None:
        """Update statistics for cache hit."""
        self.hits += 1
        self._update_avg_access_time(access_time)
        self._update_hit_rate()
    
    def update_miss(self, access_time: float) -> None:
        """Update statistics for cache miss."""
        self.misses += 1
        self._update_avg_access_time(access_time)
        self._update_hit_rate()
    
    def update_eviction(self) -> None:
        """Update statistics for eviction."""
        self.evictions += 1
    
    def _update_avg_access_time(self, access_time: float) -> None:
        """Update average access time with exponential moving average."""
        alpha = 0.1
        if self.avg_access_time == 0:
            self.avg_access_time = access_time
        else:
            self.avg_access_time = (1 - alpha) * self.avg_access_time + alpha * access_time
    
    def _update_hit_rate(self) -> None:
        """Update hit rate."""
        total = self.hits + self.misses
        self.hit_rate = self.hits / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'size_bytes': self.size_bytes,
            'avg_access_time': self.avg_access_time,
            'hit_rate': self.hit_rate,
            'total_requests': self.hits + self.misses
        }


@dataclass
class CacheEntry:
    """Individual cache entry with metadata."""
    key: str
    value: Any
    size_bytes: int
    created_time: float
    last_access_time: float
    access_count: int = 0
    ttl: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        return time.time() > self.created_time + self.ttl
    
    def touch(self) -> None:
        """Update access information."""
        self.last_access_time = time.time()
        self.access_count += 1
    
    def age(self) -> float:
        """Get age of entry in seconds."""
        return time.time() - self.created_time
    
    def idle_time(self) -> float:
        """Get time since last access."""
        return time.time() - self.last_access_time


class TensorMemoryPool:
    """Memory pool for efficient tensor allocation and reuse."""
    
    def __init__(self, max_pool_size: int = 1024 * 1024 * 1024):  # 1GB default
        self.max_pool_size = max_pool_size
        self.logger = get_logger("TensorMemoryPool")
        
        # Memory pools by size and dtype
        self.pools: Dict[Tuple[torch.dtype, torch.device], Dict[Tuple, List]] = defaultdict(
            lambda: defaultdict(list)
        )
        
        # Pool statistics
        self.allocations = 0
        self.deallocations = 0
        self.pool_hits = 0
        self.current_size = 0
        
        self._lock = threading.RLock()
    
    def allocate(
        self, 
        shape: Tuple[int, ...], 
        dtype: 'torch.dtype' = None, 
        device: 'torch.device' = None
    ) -> Optional['torch.Tensor']:
        """
        Allocate tensor from pool or create new one.
        
        Args:
            shape: Tensor shape
            dtype: Tensor data type
            device: Device to allocate on
            
        Returns:
            Allocated tensor or None if PyTorch not available
        """
        if not _TORCH_AVAILABLE:
            return None
        
        dtype = dtype or torch.float32
        device = device or torch.device('cpu')
        
        with self._lock:
            pool_key = (dtype, device)
            shape_key = tuple(shape)
            
            # Try to get from pool
            if pool_key in self.pools and shape_key in self.pools[pool_key]:
                pool = self.pools[pool_key][shape_key]
                if pool:
                    tensor = pool.pop()
                    tensor.zero_()  # Clear the tensor
                    self.pool_hits += 1
                    self.logger.debug(f"Tensor allocated from pool: {shape} {dtype}")
                    return tensor
            
            # Create new tensor
            try:
                tensor = torch.zeros(shape, dtype=dtype, device=device)
                self.allocations += 1
                self.current_size += tensor.numel() * tensor.element_size()
                self.logger.debug(f"New tensor allocated: {shape} {dtype}")
                return tensor
            except Exception as e:
                self.logger.error(f"Failed to allocate tensor {shape} {dtype}: {e}")
                return None
    
    def deallocate(self, tensor: 'torch.Tensor') -> None:
        """
        Return tensor to pool for reuse.
        
        Args:
            tensor: Tensor to return to pool
        """
        if not _TORCH_AVAILABLE or tensor is None:
            return
        
        with self._lock:
            pool_key = (tensor.dtype, tensor.device)
            shape_key = tuple(tensor.shape)
            
            # Check pool size limits
            tensor_size = tensor.numel() * tensor.element_size()
            if self.current_size + tensor_size > self.max_pool_size:
                self._cleanup_pools()
            
            # Add to pool if there's space
            if self.current_size + tensor_size <= self.max_pool_size:
                self.pools[pool_key][shape_key].append(tensor)
                self.deallocations += 1
                self.logger.debug(f"Tensor returned to pool: {tensor.shape} {tensor.dtype}")
            else:
                # Pool is full, let tensor be garbage collected
                self.logger.debug(f"Pool full, tensor not cached: {tensor.shape}")
    
    def _cleanup_pools(self) -> None:
        """Clean up pools to free memory."""
        # Remove least recently used tensors
        total_freed = 0
        target_free = self.max_pool_size * 0.2  # Free 20% of pool
        
        for pool_key in list(self.pools.keys()):
            for shape_key in list(self.pools[pool_key].keys()):
                pool = self.pools[pool_key][shape_key]
                while pool and total_freed < target_free:
                    tensor = pool.pop(0)  # Remove oldest
                    total_freed += tensor.numel() * tensor.element_size()
                
                # Remove empty shape pools
                if not pool:
                    del self.pools[pool_key][shape_key]
            
            # Remove empty dtype pools
            if not self.pools[pool_key]:
                del self.pools[pool_key]
        
        self.current_size -= total_freed
        self.logger.info(f"Freed {total_freed / (1024**2):.1f} MB from tensor pools")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self._lock:
            pool_sizes = {}
            for pool_key, shape_pools in self.pools.items():
                dtype_str = str(pool_key[0])
                device_str = str(pool_key[1])
                key = f"{dtype_str}_{device_str}"
                pool_sizes[key] = sum(len(pool) for pool in shape_pools.values())
            
            return {
                'allocations': self.allocations,
                'deallocations': self.deallocations,
                'pool_hits': self.pool_hits,
                'hit_rate': self.pool_hits / max(1, self.allocations),
                'current_size_mb': self.current_size / (1024**2),
                'max_size_mb': self.max_pool_size / (1024**2),
                'utilization': self.current_size / self.max_pool_size,
                'pool_sizes': pool_sizes
            }


class AdaptiveCache:
    """
    Adaptive multi-level cache with intelligent eviction policies.
    
    Automatically adjusts caching strategies based on access patterns,
    memory pressure, and performance characteristics.
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        max_memory: int = 512 * 1024 * 1024,  # 512MB
        eviction_policy: EvictionPolicy = EvictionPolicy.ADAPTIVE,
        enable_compression: bool = True
    ):
        self.max_size = max_size
        self.max_memory = max_memory
        self.eviction_policy = eviction_policy
        self.enable_compression = enable_compression
        
        self.logger = get_logger("AdaptiveCache")
        self.perf_logger = get_performance_logger("AdaptiveCache")
        
        # Cache storage
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.access_frequencies: Dict[str, int] = defaultdict(int)
        self.access_times: Dict[str, List[float]] = defaultdict(list)
        
        # Statistics
        self.stats = CacheStats()
        
        # Adaptive parameters
        self.adaptive_params = {
            'lru_weight': 0.4,
            'lfu_weight': 0.3,
            'ttl_weight': 0.2,
            'size_weight': 0.1
        }
        
        self._lock = threading.RLock()
        
        # Background cleanup thread
        self._cleanup_thread = None
        self._stop_cleanup = False
        self._start_cleanup_thread()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        start_time = time.time()
        
        with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check if expired
                if entry.is_expired():
                    del self.cache[key]
                    self.stats.update_miss(time.time() - start_time)
                    return None
                
                # Update access information
                entry.touch()
                self.access_frequencies[key] += 1
                self.access_times[key].append(time.time())
                
                # Move to end for LRU
                self.cache.move_to_end(key)
                
                # Decompress if needed
                value = self._decompress_value(entry.value)
                
                self.stats.update_hit(time.time() - start_time)
                self.logger.debug(f"Cache hit for key: {key}")
                return value
            
            else:
                self.stats.update_miss(time.time() - start_time)
                self.logger.debug(f"Cache miss for key: {key}")
                return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Put value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        with self._lock:
            # Compress value if enabled
            compressed_value = self._compress_value(value)
            
            # Calculate size
            size_bytes = self._calculate_size(compressed_value)
            
            # Check if we need to make space
            self._ensure_space(size_bytes)
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=compressed_value,
                size_bytes=size_bytes,
                created_time=time.time(),
                last_access_time=time.time(),
                ttl=ttl
            )
            
            # Remove existing entry if present
            if key in self.cache:
                old_entry = self.cache[key]
                self.stats.size_bytes -= old_entry.size_bytes
            
            # Add new entry
            self.cache[key] = entry
            self.stats.size_bytes += size_bytes
            
            # Initialize access tracking
            self.access_frequencies[key] = 1
            self.access_times[key] = [time.time()]
            
            self.logger.debug(f"Cached value for key: {key}, size: {size_bytes} bytes")
    
    def invalidate(self, key: str) -> bool:
        """
        Invalidate cache entry.
        
        Args:
            key: Cache key to invalidate
            
        Returns:
            True if key was found and removed
        """
        with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                self.stats.size_bytes -= entry.size_bytes
                del self.cache[key]
                
                # Clean up access tracking
                if key in self.access_frequencies:
                    del self.access_frequencies[key]
                if key in self.access_times:
                    del self.access_times[key]
                
                self.logger.debug(f"Invalidated cache key: {key}")
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()
            self.access_frequencies.clear()
            self.access_times.clear()
            self.stats = CacheStats()
            self.logger.info("Cache cleared")
    
    def _ensure_space(self, needed_bytes: int) -> None:
        """Ensure sufficient space in cache."""
        # Check size limit
        while len(self.cache) >= self.max_size:
            self._evict_entry()
        
        # Check memory limit
        while self.stats.size_bytes + needed_bytes > self.max_memory:
            self._evict_entry()
    
    def _evict_entry(self) -> None:
        """Evict entry based on eviction policy."""
        if not self.cache:
            return
        
        if self.eviction_policy == EvictionPolicy.ADAPTIVE:
            key_to_evict = self._adaptive_eviction()
        elif self.eviction_policy == EvictionPolicy.LRU:
            key_to_evict = next(iter(self.cache))  # First (oldest) key
        elif self.eviction_policy == EvictionPolicy.LFU:
            key_to_evict = min(self.access_frequencies.keys(), 
                             key=lambda k: self.access_frequencies[k])
        elif self.eviction_policy == EvictionPolicy.TTL:
            key_to_evict = self._find_earliest_expiry()
        else:  # FIFO
            key_to_evict = next(iter(self.cache))
        
        if key_to_evict:
            entry = self.cache[key_to_evict]
            self.stats.size_bytes -= entry.size_bytes
            del self.cache[key_to_evict]
            self.stats.update_eviction()
            
            self.logger.debug(f"Evicted cache key: {key_to_evict}")
    
    def _adaptive_eviction(self) -> Optional[str]:
        """Adaptive eviction based on multiple factors."""
        if not self.cache:
            return None
        
        scores = {}
        current_time = time.time()
        
        for key, entry in self.cache.items():
            # LRU component (higher idle time = higher score)
            lru_score = entry.idle_time()
            
            # LFU component (lower frequency = higher score)
            frequency = self.access_frequencies.get(key, 1)
            lfu_score = 1.0 / frequency
            
            # TTL component (closer to expiry = higher score)
            if entry.ttl:
                time_to_expiry = (entry.created_time + entry.ttl) - current_time
                ttl_score = max(0, 1.0 - (time_to_expiry / entry.ttl))
            else:
                ttl_score = 0
            
            # Size component (larger entries = higher score)
            max_size = max(e.size_bytes for e in self.cache.values())
            size_score = entry.size_bytes / max_size if max_size > 0 else 0
            
            # Weighted combination
            combined_score = (
                self.adaptive_params['lru_weight'] * lru_score +
                self.adaptive_params['lfu_weight'] * lfu_score +
                self.adaptive_params['ttl_weight'] * ttl_score +
                self.adaptive_params['size_weight'] * size_score
            )
            
            scores[key] = combined_score
        
        # Return key with highest eviction score
        return max(scores.keys(), key=lambda k: scores[k])
    
    def _find_earliest_expiry(self) -> Optional[str]:
        """Find entry with earliest expiry time."""
        earliest_key = None
        earliest_expiry = float('inf')
        
        for key, entry in self.cache.items():
            if entry.ttl:
                expiry_time = entry.created_time + entry.ttl
                if expiry_time < earliest_expiry:
                    earliest_expiry = expiry_time
                    earliest_key = key
        
        return earliest_key or next(iter(self.cache))
    
    def _compress_value(self, value: Any) -> Any:
        """Compress value if compression is enabled."""
        if not self.enable_compression:
            return value
        
        try:
            # Try to compress serializable objects
            if _TORCH_AVAILABLE and isinstance(value, torch.Tensor):
                # Don't compress tensors as they're already efficient
                return value
            elif isinstance(value, (str, bytes, list, dict, tuple)):
                # Compress serializable objects
                serialized = pickle.dumps(value)
                if len(serialized) > 1024:  # Only compress if > 1KB
                    import zlib
                    compressed = zlib.compress(serialized)
                    if len(compressed) < len(serialized) * 0.8:  # 20% compression ratio
                        return ('compressed', compressed)
            
            return value
            
        except Exception:
            # Compression failed, return original
            return value
    
    def _decompress_value(self, value: Any) -> Any:
        """Decompress value if it was compressed."""
        if isinstance(value, tuple) and len(value) == 2 and value[0] == 'compressed':
            try:
                import zlib
                serialized = zlib.decompress(value[1])
                return pickle.loads(serialized)
            except Exception:
                # Decompression failed
                return value
        
        return value
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes."""
        try:
            if _TORCH_AVAILABLE and isinstance(value, torch.Tensor):
                return value.numel() * value.element_size()
            elif isinstance(value, (str, bytes)):
                return len(value)
            elif isinstance(value, tuple) and len(value) == 2 and value[0] == 'compressed':
                return len(value[1])
            else:
                # Approximate size using pickle
                return len(pickle.dumps(value))
        except Exception:
            # Fallback size estimation
            return 1024
    
    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread."""
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            name="CacheCleanup",
            daemon=True
        )
        self._cleanup_thread.start()
    
    def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while not self._stop_cleanup:
            try:
                time.sleep(60)  # Cleanup every minute
                self._cleanup_expired()
                self._update_adaptive_params()
            except Exception as e:
                self.logger.error(f"Error in cache cleanup: {e}")
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        with self._lock:
            expired_keys = []
            current_time = time.time()
            
            for key, entry in self.cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                entry = self.cache[key]
                self.stats.size_bytes -= entry.size_bytes
                del self.cache[key]
                
                # Clean up access tracking
                if key in self.access_frequencies:
                    del self.access_frequencies[key]
                if key in self.access_times:
                    del self.access_times[key]
            
            if expired_keys:
                self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _update_adaptive_params(self) -> None:
        """Update adaptive eviction parameters based on access patterns."""
        with self._lock:
            if len(self.cache) < 10:  # Need sufficient data
                return
            
            # Analyze access patterns
            recent_accesses = []
            current_time = time.time()
            
            for key, times in self.access_times.items():
                recent = [t for t in times if current_time - t < 3600]  # Last hour
                if recent:
                    recent_accesses.append(len(recent))
            
            if not recent_accesses:
                return
            
            # Adjust weights based on access patterns
            avg_frequency = np.mean(recent_accesses)
            
            if avg_frequency > 10:  # High frequency access
                # Favor LFU
                self.adaptive_params['lfu_weight'] = 0.5
                self.adaptive_params['lru_weight'] = 0.2
            else:  # Low frequency access
                # Favor LRU
                self.adaptive_params['lru_weight'] = 0.5
                self.adaptive_params['lfu_weight'] = 0.2
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'cache_stats': self.stats.to_dict(),
                'cache_size': len(self.cache),
                'max_size': self.max_size,
                'memory_usage_mb': self.stats.size_bytes / (1024**2),
                'max_memory_mb': self.max_memory / (1024**2),
                'eviction_policy': self.eviction_policy.value,
                'adaptive_params': self.adaptive_params.copy(),
                'compression_enabled': self.enable_compression
            }
    
    def stop(self) -> None:
        """Stop background processes."""
        self._stop_cleanup = True
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5.0)


class CacheManager:
    """
    Centralized cache management system with multiple cache levels.
    
    Manages L1/L2/L3 caches with different characteristics and
    automatic promotion/demotion based on access patterns.
    """
    
    def __init__(self):
        self.logger = get_logger("CacheManager")
        
        # Multi-level caches
        self.l1_cache = AdaptiveCache(
            max_size=100,
            max_memory=64 * 1024 * 1024,  # 64MB
            eviction_policy=EvictionPolicy.LRU
        )
        
        self.l2_cache = AdaptiveCache(
            max_size=500,
            max_memory=256 * 1024 * 1024,  # 256MB
            eviction_policy=EvictionPolicy.ADAPTIVE
        )
        
        self.l3_cache = AdaptiveCache(
            max_size=2000,
            max_memory=1024 * 1024 * 1024,  # 1GB
            eviction_policy=EvictionPolicy.LFU,
            enable_compression=True
        )
        
        # Memory pool for tensors
        self.tensor_pool = TensorMemoryPool()
        
        # Cache hierarchy management
        self.promotion_threshold = 3  # Promote after 3 accesses
        self.access_counters: Dict[str, int] = defaultdict(int)
        
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache hierarchy.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        start_time = time.time()
        
        with self._lock:
            # Try L1 first
            value = self.l1_cache.get(key)
            if value is not None:
                self.access_counters[key] += 1
                self.logger.debug(f"L1 cache hit for key: {key}")
                return value
            
            # Try L2
            value = self.l2_cache.get(key)
            if value is not None:
                self.access_counters[key] += 1
                # Promote to L1 if frequently accessed
                if self.access_counters[key] >= self.promotion_threshold:
                    self.l1_cache.put(key, value)
                    self.logger.debug(f"Promoted {key} from L2 to L1")
                return value
            
            # Try L3
            value = self.l3_cache.get(key)
            if value is not None:
                self.access_counters[key] += 1
                # Promote to L2 if frequently accessed
                if self.access_counters[key] >= self.promotion_threshold:
                    self.l2_cache.put(key, value)
                    self.logger.debug(f"Promoted {key} from L3 to L2")
                return value
            
            return None
    
    def put(self, key: str, value: Any, level: CacheLevel = CacheLevel.L3_LARGE, 
            ttl: Optional[float] = None) -> None:
        """
        Put value in specified cache level.
        
        Args:
            key: Cache key
            value: Value to cache
            level: Cache level to store in
            ttl: Time to live in seconds
        """
        with self._lock:
            if level == CacheLevel.L1_FAST:
                self.l1_cache.put(key, value, ttl)
            elif level == CacheLevel.L2_BALANCED:
                self.l2_cache.put(key, value, ttl)
            elif level == CacheLevel.L3_LARGE:
                self.l3_cache.put(key, value, ttl)
            
            self.access_counters[key] = 1
            self.logger.debug(f"Cached {key} in {level.value}")
    
    def invalidate(self, key: str) -> None:
        """Invalidate key from all cache levels."""
        with self._lock:
            self.l1_cache.invalidate(key)
            self.l2_cache.invalidate(key)
            self.l3_cache.invalidate(key)
            
            if key in self.access_counters:
                del self.access_counters[key]
            
            self.logger.debug(f"Invalidated {key} from all cache levels")
    
    def clear_all(self) -> None:
        """Clear all caches."""
        with self._lock:
            self.l1_cache.clear()
            self.l2_cache.clear()
            self.l3_cache.clear()
            self.access_counters.clear()
            self.logger.info("Cleared all caches")
    
    def get_tensor(self, shape: Tuple[int, ...], dtype=None, device=None) -> Optional[Any]:
        """Get tensor from memory pool."""
        return self.tensor_pool.allocate(shape, dtype, device)
    
    def return_tensor(self, tensor) -> None:
        """Return tensor to memory pool."""
        self.tensor_pool.deallocate(tensor)
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._lock:
            return {
                'l1_cache': self.l1_cache.get_stats(),
                'l2_cache': self.l2_cache.get_stats(),
                'l3_cache': self.l3_cache.get_stats(),
                'tensor_pool': self.tensor_pool.get_stats(),
                'total_keys': len(self.access_counters),
                'promotion_threshold': self.promotion_threshold,
                'memory_usage': {
                    'l1_mb': self.l1_cache.stats.size_bytes / (1024**2),
                    'l2_mb': self.l2_cache.stats.size_bytes / (1024**2),
                    'l3_mb': self.l3_cache.stats.size_bytes / (1024**2),
                    'tensor_pool_mb': self.tensor_pool.current_size / (1024**2)
                }
            }
    
    def stop(self) -> None:
        """Stop all cache background processes."""
        self.l1_cache.stop()
        self.l2_cache.stop()
        self.l3_cache.stop()


# Global cache manager instance
_global_cache_manager = None

def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = CacheManager()
    return _global_cache_manager


def cached(
    key_func: Optional[Callable] = None,
    ttl: Optional[float] = None,
    level: CacheLevel = CacheLevel.L2_BALANCED
):
    """
    Decorator for caching function results.
    
    Args:
        key_func: Function to generate cache key from arguments
        ttl: Time to live in seconds
        level: Cache level to use
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache_manager = get_cache_manager()
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = hashlib.md5("|".join(key_parts).encode()).hexdigest()
            
            # Try to get from cache
            result = cache_manager.get(cache_key)
            if result is not None:
                return result
            
            # Compute result and cache it
            result = func(*args, **kwargs)
            cache_manager.put(cache_key, result, level, ttl)
            
            return result
        return wrapper
    return decorator