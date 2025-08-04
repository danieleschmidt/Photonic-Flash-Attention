"""Unified memory management for photonic and GPU computation."""

import torch
import threading
import gc
import weakref
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from contextlib import contextmanager

from ..utils.logging import get_logger
from ..utils.exceptions import PhotonicMemoryError
from ..config import get_config


@dataclass
class MemoryBlock:
    """Represents a memory block in the pool."""
    tensor: torch.Tensor
    size_bytes: int
    last_used: float
    ref_count: int
    device: str
    dtype: torch.dtype
    
    @property
    def shape(self) -> torch.Size:
        return self.tensor.shape


class UnifiedMemoryManager:
    """
    Unified memory manager for efficient memory allocation and reuse
    across GPU and photonic devices.
    
    Features:
    - Memory pooling and reuse
    - Cross-device memory management
    - Automatic garbage collection
    - Memory usage monitoring
    - OOM prevention
    """
    
    def __init__(
        self,
        max_memory_gb: float = 16.0,
        pool_size_mb: int = 512,
        gc_threshold: float = 0.8,
        enable_profiling: bool = False,
    ):
        self.max_memory_bytes = int(max_memory_gb * 1024 ** 3)
        self.pool_size_bytes = int(pool_size_mb * 1024 ** 2)
        self.gc_threshold = gc_threshold
        self.enable_profiling = enable_profiling
        
        self.logger = get_logger(self.__class__.__name__)
        
        # Memory pools by device and shape
        self.memory_pools: Dict[str, Dict[Tuple, List[MemoryBlock]]] = defaultdict(lambda: defaultdict(list))
        
        # Active allocations tracking
        self.active_allocations: Dict[id, MemoryBlock] = {}
        self.allocation_history: List[Dict[str, Any]] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self.total_allocated = 0
        self.total_freed = 0
        self.pool_hits = 0
        self.pool_misses = 0
        self.oom_events = 0
        
        # Weak references for automatic cleanup
        self.tensor_refs: Dict[int, weakref.ref] = {}
        
        self.logger.info(f"Memory manager initialized: max={max_memory_gb}GB, pool={pool_size_mb}MB")
    
    def allocate(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = 'cuda',
        requires_grad: bool = False,
        pin_memory: bool = False,
    ) -> torch.Tensor:
        """
        Allocate tensor with memory pooling.
        
        Args:
            shape: Tensor shape
            dtype: Data type
            device: Target device
            requires_grad: Whether gradient is required
            pin_memory: Whether to pin memory for faster transfer
            
        Returns:
            Allocated tensor
            
        Raises:
            PhotonicMemoryError: If allocation fails
        """
        with self._lock:
            device_str = str(device)
            size_bytes = self._calculate_size(shape, dtype)
            
            # Check memory limits
            if self._would_exceed_limits(size_bytes):
                self._trigger_gc()
                if self._would_exceed_limits(size_bytes):
                    self.oom_events += 1
                    raise PhotonicMemoryError(
                        f"Would exceed memory limit: {size_bytes} bytes",
                        size_bytes,
                        self.max_memory_bytes - self._get_current_usage()
                    )
            
            # Try to reuse from pool
            tensor = self._try_pool_allocation(shape, dtype, device_str)
            
            if tensor is not None:
                self.pool_hits += 1
                self.logger.debug(f"Pool hit: {shape} on {device_str}")
            else:
                self.pool_misses += 1
                # Allocate new tensor
                try:
                    if device_str.startswith('cuda'):
                        tensor = torch.empty(shape, dtype=dtype, device=device, pin_memory=pin_memory)
                    else:
                        tensor = torch.empty(shape, dtype=dtype, device=device)
                    
                    self.logger.debug(f"New allocation: {shape} on {device_str}")
                
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        self.oom_events += 1
                        self._emergency_cleanup()
                        # Try again after cleanup
                        try:
                            tensor = torch.empty(shape, dtype=dtype, device=device)
                        except RuntimeError:
                            raise PhotonicMemoryError(f"Failed to allocate after cleanup: {e}")
                    else:
                        raise PhotonicMemoryError(f"Allocation failed: {e}")
            
            # Set gradient requirement
            if requires_grad:
                tensor.requires_grad_(True)
            
            # Track allocation
            self._track_allocation(tensor, device_str)
            
            self.total_allocated += size_bytes
            
            if self.enable_profiling:
                self._record_allocation(shape, dtype, device_str, size_bytes)
            
            return tensor
    
    def deallocate(self, tensor: torch.Tensor) -> None:
        """
        Deallocate tensor and return to pool if possible.
        
        Args:
            tensor: Tensor to deallocate
        """
        with self._lock:
            tensor_id = id(tensor)
            
            if tensor_id not in self.active_allocations:
                self.logger.warning(f"Attempting to deallocate untracked tensor: {tensor.shape}")
                return
            
            block = self.active_allocations[tensor_id]
            device_str = block.device
            
            # Return to pool if size is reasonable
            if block.size_bytes <= self.pool_size_bytes:
                shape_key = tuple(tensor.shape)
                
                # Clear tensor data (security)
                tensor.zero_()
                
                # Update block
                block.last_used = torch.cuda.Event(enable_timing=True).query() if torch.cuda.is_available() else 0
                block.ref_count = 0
                
                # Add to pool
                self.memory_pools[device_str][shape_key].append(block)
                
                # Limit pool size
                pool = self.memory_pools[device_str][shape_key]
                if len(pool) > 10:  # Max 10 tensors per shape
                    oldest = min(pool, key=lambda b: b.last_used)
                    pool.remove(oldest)
                    del oldest.tensor
                
                self.logger.debug(f"Returned to pool: {tensor.shape} on {device_str}")
            
            else:
                # Too large for pool, just delete
                del tensor
                self.logger.debug(f"Deleted large tensor: {block.tensor.shape}")
            
            # Remove from active tracking
            del self.active_allocations[tensor_id]
            if tensor_id in self.tensor_refs:
                del self.tensor_refs[tensor_id]
            
            self.total_freed += block.size_bytes
    
    def _try_pool_allocation(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: str,
    ) -> Optional[torch.Tensor]:
        """Try to allocate from memory pool."""
        shape_key = tuple(shape)
        
        if device not in self.memory_pools:
            return None
        
        if shape_key not in self.memory_pools[device]:
            return None
        
        pool = self.memory_pools[device][shape_key]
        if not pool:
            return None
        
        # Find suitable block
        for i, block in enumerate(pool):
            if block.dtype == dtype and block.ref_count == 0:
                # Remove from pool
                pool.pop(i)
                return block.tensor
        
        return None
    
    def _track_allocation(self, tensor: torch.Tensor, device: str) -> None:
        """Track tensor allocation."""
        tensor_id = id(tensor)
        size_bytes = tensor.numel() * tensor.element_size()
        
        block = MemoryBlock(
            tensor=tensor,
            size_bytes=size_bytes,
            last_used=torch.cuda.Event(enable_timing=True).query() if torch.cuda.is_available() else 0,
            ref_count=1,
            device=device,
            dtype=tensor.dtype,
        )
        
        self.active_allocations[tensor_id] = block
        
        # Create weak reference for automatic cleanup
        def cleanup_callback(ref):
            if tensor_id in self.active_allocations:
                self.deallocate(self.active_allocations[tensor_id].tensor)
        
        self.tensor_refs[tensor_id] = weakref.ref(tensor, cleanup_callback)
    
    def _calculate_size(self, shape: Tuple[int, ...], dtype: torch.dtype) -> int:
        """Calculate tensor size in bytes."""
        numel = 1
        for dim in shape:
            numel *= dim
        
        if dtype == torch.float32:
            return numel * 4
        elif dtype == torch.float16 or dtype == torch.bfloat16:
            return numel * 2
        elif dtype == torch.int64:
            return numel * 8
        elif dtype == torch.int32:
            return numel * 4
        else:
            return numel * 4  # Default estimate
    
    def _would_exceed_limits(self, additional_bytes: int) -> bool:
        """Check if allocation would exceed memory limits."""
        current_usage = self._get_current_usage()
        return (current_usage + additional_bytes) > self.max_memory_bytes
    
    def _get_current_usage(self) -> int:
        """Get current memory usage in bytes."""
        return sum(block.size_bytes for block in self.active_allocations.values())
    
    def _trigger_gc(self) -> None:
        """Trigger garbage collection and cleanup."""
        self.logger.info("Triggering memory cleanup")
        
        # Python garbage collection
        gc.collect()
        
        # PyTorch cache cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clean up old pool entries
        self._cleanup_pools()
        
        # Remove stale weak references
        stale_refs = []
        for tensor_id, ref in self.tensor_refs.items():
            if ref() is None:
                stale_refs.append(tensor_id)
        
        for tensor_id in stale_refs:
            del self.tensor_refs[tensor_id]
            if tensor_id in self.active_allocations:
                del self.active_allocations[tensor_id]
    
    def _cleanup_pools(self) -> None:
        """Clean up memory pools."""
        current_time = torch.cuda.Event(enable_timing=True).query() if torch.cuda.is_available() else 0
        cleanup_threshold = 300.0  # 5 minutes
        
        for device_pools in self.memory_pools.values():
            for shape_key, pool in list(device_pools.items()):
                # Remove old entries
                pool[:] = [
                    block for block in pool
                    if (current_time - block.last_used) < cleanup_threshold
                ]
                
                # Remove empty pools
                if not pool:
                    del device_pools[shape_key]
    
    def _emergency_cleanup(self) -> None:
        """Emergency cleanup when OOM occurs."""
        self.logger.warning("Performing emergency memory cleanup")
        
        # Clear all pools
        for device_pools in self.memory_pools.values():
            for pool in device_pools.values():
                for block in pool:
                    del block.tensor
                pool.clear()
            device_pools.clear()
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _record_allocation(self, shape: Tuple[int, ...], dtype: torch.dtype, device: str, size_bytes: int) -> None:
        """Record allocation for profiling."""
        record = {
            'timestamp': torch.cuda.Event(enable_timing=True).query() if torch.cuda.is_available() else 0,
            'shape': shape,
            'dtype': str(dtype),
            'device': device,
            'size_bytes': size_bytes,
            'total_allocated': self.total_allocated,
        }
        
        self.allocation_history.append(record)
        
        # Limit history size
        if len(self.allocation_history) > 1000:
            self.allocation_history = self.allocation_history[-500:]
    
    @contextmanager
    def temporary_allocation(self, *args, **kwargs):
        """Context manager for temporary tensor allocation."""
        tensor = self.allocate(*args, **kwargs)
        try:
            yield tensor
        finally:
            self.deallocate(tensor)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        with self._lock:
            current_usage = self._get_current_usage()
            
            # Pool statistics
            pool_stats = {}
            total_pooled = 0
            for device, device_pools in self.memory_pools.items():
                device_pooled = 0
                device_blocks = 0
                for pool in device_pools.values():
                    device_blocks += len(pool)
                    device_pooled += sum(block.size_bytes for block in pool)
                
                pool_stats[device] = {
                    'blocks': device_blocks,
                    'bytes': device_pooled,
                    'mb': device_pooled / (1024 ** 2),
                }
                total_pooled += device_pooled
            
            stats = {
                'current_usage_bytes': current_usage,
                'current_usage_mb': current_usage / (1024 ** 2),
                'current_usage_gb': current_usage / (1024 ** 3),
                'max_memory_gb': self.max_memory_bytes / (1024 ** 3),
                'memory_utilization': current_usage / self.max_memory_bytes,
                'total_allocated_bytes': self.total_allocated,
                'total_freed_bytes': self.total_freed,
                'net_allocated_bytes': self.total_allocated - self.total_freed,
                'active_allocations': len(self.active_allocations),
                'pool_stats': pool_stats,
                'total_pooled_mb': total_pooled / (1024 ** 2),
                'pool_hit_rate': self.pool_hits / (self.pool_hits + self.pool_misses) if (self.pool_hits + self.pool_misses) > 0 else 0,
                'oom_events': self.oom_events,
            }
            
            # Add GPU memory info if available
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i)
                    reserved = torch.cuda.memory_reserved(i)
                    stats[f'gpu_{i}_allocated_mb'] = allocated / (1024 ** 2)
                    stats[f'gpu_{i}_reserved_mb'] = reserved / (1024 ** 2)
            
            return stats
    
    def optimize_memory_layout(self) -> None:
        """Optimize memory layout and defragment pools."""
        with self._lock:
            self.logger.info("Optimizing memory layout")
            
            # Defragment pools by device
            for device, device_pools in self.memory_pools.items():
                for shape_key, pool in device_pools.items():
                    # Sort by last used time (LRU)
                    pool.sort(key=lambda b: b.last_used, reverse=True)
                    
                    # Keep only most recent entries
                    if len(pool) > 5:
                        for block in pool[5:]:
                            del block.tensor
                        pool[:] = pool[:5]
            
            # Trigger cleanup
            self._trigger_gc()
            
            self.logger.info("Memory optimization complete")
    
    def clear_pools(self) -> None:
        """Clear all memory pools."""
        with self._lock:
            for device_pools in self.memory_pools.values():
                for pool in device_pools.values():
                    for block in pool:
                        del block.tensor
                    pool.clear()
                device_pools.clear()
            
            self.logger.info("Cleared all memory pools")
    
    def set_memory_limit(self, limit_gb: float) -> None:
        """Update memory limit."""
        with self._lock:
            old_limit = self.max_memory_bytes / (1024 ** 3)
            self.max_memory_bytes = int(limit_gb * 1024 ** 3)
            
            self.logger.info(f"Memory limit updated: {old_limit:.2f}GB -> {limit_gb:.2f}GB")
            
            # Trigger cleanup if current usage exceeds new limit
            if self._get_current_usage() > self.max_memory_bytes:
                self._trigger_gc()


# Global memory manager instance
_memory_manager = UnifiedMemoryManager()


def get_memory_manager() -> UnifiedMemoryManager:
    """Get the global memory manager instance."""
    return _memory_manager


def allocate_tensor(*args, **kwargs) -> torch.Tensor:
    """Convenience function for tensor allocation."""
    return _memory_manager.allocate(*args, **kwargs)


def deallocate_tensor(tensor: torch.Tensor) -> None:
    """Convenience function for tensor deallocation."""
    _memory_manager.deallocate(tensor)


@contextmanager
def temporary_tensor(*args, **kwargs):
    """Context manager for temporary tensor allocation."""
    with _memory_manager.temporary_allocation(*args, **kwargs) as tensor:
        yield tensor