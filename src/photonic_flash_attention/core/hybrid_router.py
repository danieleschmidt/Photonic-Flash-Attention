"""Intelligent hybrid routing with performance optimization and scaling."""

import torch
import torch.nn as nn
import numpy as np
import threading
import time
from typing import Dict, Any, Optional, Tuple, List, Union
from collections import deque
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..config import get_config
from ..utils.logging import get_logger
from ..utils.exceptions import PhotonicComputationError, PhotonicTimeoutError
from .flash_attention_3 import FlashAttention3
from .photonic_attention import PhotonicAttention


@dataclass
class PerformanceMetrics:
    """Performance metrics for routing decisions."""
    latency_ms: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    energy_mj: float = 0.0
    memory_mb: float = 0.0
    temperature_c: float = 0.0
    accuracy_score: float = 1.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class WorkloadCharacteristics:
    """Characteristics of the current workload."""
    batch_size: int
    seq_length: int
    embed_dim: int
    num_heads: int
    is_training: bool = False
    has_mask: bool = False
    dtype: torch.dtype = torch.float32
    
    def to_features(self) -> np.ndarray:
        """Convert to feature vector for ML models."""
        return np.array([
            self.batch_size,
            self.seq_length,
            self.embed_dim,
            self.num_heads,
            float(self.is_training),
            float(self.has_mask),
            1.0 if self.dtype == torch.float32 else 0.5 if self.dtype == torch.float16 else 0.25,
        ])


class AdaptiveRouter:
    """
    Intelligent router that learns optimal device selection.
    
    Uses machine learning to predict the best computation device
    based on workload characteristics and performance history.
    """
    
    def __init__(
        self,
        history_size: int = 1000,
        learning_rate: float = 0.01,
        exploration_rate: float = 0.1,
        min_samples_for_prediction: int = 50,
    ):
        self.history_size = history_size
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.min_samples_for_prediction = min_samples_for_prediction
        
        self.logger = get_logger(self.__class__.__name__)
        
        # Performance history
        self.gpu_history: deque = deque(maxlen=history_size)
        self.photonic_history: deque = deque(maxlen=history_size)
        
        # Simple linear model for performance prediction
        self.gpu_weights = np.random.normal(0, 0.1, 7)  # 7 features
        self.photonic_weights = np.random.normal(0, 0.1, 7)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Caching for frequent patterns
        self._prediction_cache: Dict[str, Tuple[str, float]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        self.logger.info("Adaptive router initialized")
    
    def select_device(self, workload: WorkloadCharacteristics) -> str:
        """
        Select optimal device for the given workload.
        
        Args:
            workload: Workload characteristics
            
        Returns:
            Device selection ('gpu' or 'photonic')
        """
        with self._lock:
            # Check cache first
            cache_key = self._get_cache_key(workload)
            if cache_key in self._prediction_cache:
                device, confidence = self._prediction_cache[cache_key]
                self._cache_hits += 1
                self.logger.debug(f"Cache hit: {device} (confidence: {confidence:.3f})")
                return device
            
            self._cache_misses += 1
            
            # Make prediction
            device = self._predict_optimal_device(workload)
            
            # Add to cache with confidence
            confidence = self._get_prediction_confidence(workload)
            self._prediction_cache[cache_key] = (device, confidence)
            
            # Limit cache size
            if len(self._prediction_cache) > 1000:
                # Remove oldest entries (simple FIFO)
                oldest_key = next(iter(self._prediction_cache))
                del self._prediction_cache[oldest_key]
            
            return device
    
    def _get_cache_key(self, workload: WorkloadCharacteristics) -> str:
        """Generate cache key for workload."""
        # Round values to reduce cache fragmentation
        return f"{workload.batch_size}_{workload.seq_length//32*32}_{workload.embed_dim}_{workload.num_heads}_{workload.is_training}_{workload.has_mask}"
    
    def _predict_optimal_device(self, workload: WorkloadCharacteristics) -> str:
        """Predict optimal device using learned model."""
        features = workload.to_features()
        
        # Check if we have enough data for ML prediction
        if (len(self.gpu_history) < self.min_samples_for_prediction or 
            len(self.photonic_history) < self.min_samples_for_prediction):
            return self._heuristic_selection(workload)
        
        # Predict performance for each device
        gpu_predicted_latency = np.dot(self.gpu_weights, features)
        photonic_predicted_latency = np.dot(self.photonic_weights, features)
        
        # Add exploration noise
        if np.random.random() < self.exploration_rate:
            return np.random.choice(['gpu', 'photonic'])
        
        # Select device with better predicted performance
        device = 'gpu' if gpu_predicted_latency < photonic_predicted_latency else 'photonic'
        
        self.logger.debug(f"ML prediction: GPU={gpu_predicted_latency:.2f}ms, Photonic={photonic_predicted_latency:.2f}ms -> {device}")
        return device
    
    def _heuristic_selection(self, workload: WorkloadCharacteristics) -> str:
        """Fallback heuristic selection when insufficient data."""
        config = get_config()
        
        # Use sequence length threshold as primary heuristic
        if workload.seq_length >= config.photonic_threshold:
            return 'photonic'
        
        # Consider batch size and computation complexity
        total_ops = workload.batch_size * workload.seq_length * workload.seq_length
        if total_ops > 1e6:  # Large computation
            return 'photonic'
        
        return 'gpu'
    
    def _get_prediction_confidence(self, workload: WorkloadCharacteristics) -> float:
        """Get confidence score for the prediction."""
        # Simple confidence based on amount of historical data
        total_samples = len(self.gpu_history) + len(self.photonic_history)
        return min(1.0, total_samples / (2 * self.min_samples_for_prediction))
    
    def update_performance(
        self,
        device: str,
        workload: WorkloadCharacteristics,
        metrics: PerformanceMetrics,
    ) -> None:
        """
        Update performance history and retrain model.
        
        Args:
            device: Device that was used
            workload: Workload characteristics
            metrics: Performance metrics
        """
        with self._lock:
            # Add to appropriate history
            sample = (workload.to_features(), metrics.latency_ms)
            
            if device == 'gpu':
                self.gpu_history.append(sample)
            elif device == 'photonic':
                self.photonic_history.append(sample)
            
            # Periodically retrain models
            if (len(self.gpu_history) + len(self.photonic_history)) % 10 == 0:
                self._update_models()
            
            # Clear cache periodically to allow adaptation
            if len(self._prediction_cache) > 0 and np.random.random() < 0.01:
                self._prediction_cache.clear()
                self.logger.debug("Cleared prediction cache for adaptation")
    
    def _update_models(self) -> None:
        """Update performance prediction models using gradient descent."""
        if len(self.gpu_history) >= 10:
            self._update_single_model(self.gpu_history, self.gpu_weights)
        
        if len(self.photonic_history) >= 10:
            self._update_single_model(self.photonic_history, self.photonic_weights)
    
    def _update_single_model(self, history: deque, weights: np.ndarray) -> None:
        """Update a single performance model."""
        # Convert history to training data
        X = np.array([sample[0] for sample in history])
        y = np.array([sample[1] for sample in history])
        
        # Normalize features
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0) + 1e-8
        X_norm = (X - X_mean) / X_std
        
        # Gradient descent step
        predictions = np.dot(X_norm, weights)
        errors = predictions - y
        gradient = np.dot(X_norm.T, errors) / len(history)
        
        # Update weights
        weights -= self.learning_rate * gradient
        
        # Compute and log training error
        mse = np.mean(errors ** 2)
        self.logger.debug(f"Model update: MSE = {mse:.2f}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics."""
        with self._lock:
            cache_total = self._cache_hits + self._cache_misses
            cache_hit_rate = self._cache_hits / cache_total if cache_total > 0 else 0.0
            
            return {
                'gpu_samples': len(self.gpu_history),
                'photonic_samples': len(self.photonic_history),
                'total_samples': len(self.gpu_history) + len(self.photonic_history),
                'cache_size': len(self._prediction_cache),
                'cache_hit_rate': cache_hit_rate,
                'exploration_rate': self.exploration_rate,
                'min_samples_for_ml': self.min_samples_for_prediction,
                'using_ml_prediction': len(self.gpu_history) >= self.min_samples_for_prediction,
            }


class HybridFlashAttention(nn.Module):
    """
    High-performance hybrid attention with intelligent routing and scaling.
    
    Features:
    - Adaptive device selection using ML
    - Concurrent multi-device processing
    - Performance optimization and caching
    - Auto-scaling based on load
    - Resource pooling and management
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        device: Union[str, torch.device] = 'auto',
        dtype: Optional[torch.dtype] = None,
        enable_scaling: bool = True,
        max_concurrent_requests: int = 4,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.enable_scaling = enable_scaling
        self.max_concurrent_requests = max_concurrent_requests
        
        self.logger = get_logger(self.__class__.__name__)
        self.config = get_config()
        
        # Initialize individual attention implementations
        self.gpu_attention = FlashAttention3(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            device=device if device != 'auto' else None,
            dtype=dtype,
        )
        
        self.photonic_attention = None
        try:
            self.photonic_attention = PhotonicAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                bias=bias,
                device=device if device != 'auto' else None,
                dtype=dtype,
            )
        except Exception as e:
            self.logger.warning(f"Photonic attention not available: {e}")
        
        # Adaptive router
        self.router = AdaptiveRouter()
        
        # Scaling and load balancing
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_requests) if enable_scaling else None
        self.active_requests = 0
        self.request_queue = deque()
        self._scaling_lock = threading.Lock()
        
        # Performance optimization
        self.warmup_complete = False
        self.memory_pool = {}  # Simple memory pool for frequently used tensors
        
        # Statistics
        self.total_requests = 0
        self.concurrent_requests = 0
        self.peak_concurrent = 0
        
        self.logger.info(f"Hybrid attention initialized: scaling={enable_scaling}, max_concurrent={max_concurrent_requests}")
    
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with intelligent routing and scaling.
        
        Args:
            query: Query tensor [batch, seq_len, embed_dim]
            key: Key tensor (optional)
            value: Value tensor (optional)
            attention_mask: Attention mask (optional)
            need_weights: Whether to return attention weights
            
        Returns:
            output: Attention output
            weights: Attention weights (if requested)
        """
        # Update request statistics
        with self._scaling_lock:
            self.total_requests += 1
            self.active_requests += 1
            self.concurrent_requests = self.active_requests
            self.peak_concurrent = max(self.peak_concurrent, self.concurrent_requests)
        
        try:
            # Handle high load with scaling
            if self.enable_scaling and self.active_requests > self.max_concurrent_requests:
                return self._scaled_forward(query, key, value, attention_mask, need_weights)
            else:
                return self._standard_forward(query, key, value, attention_mask, need_weights)
        
        finally:
            with self._scaling_lock:
                self.active_requests -= 1
    
    def _standard_forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        need_weights: bool,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Standard forward pass with device selection."""
        # Characterize workload
        workload = WorkloadCharacteristics(
            batch_size=query.shape[0],
            seq_length=query.shape[1],
            embed_dim=query.shape[2],
            num_heads=self.num_heads,
            is_training=self.training,
            has_mask=attention_mask is not None,
            dtype=query.dtype,
        )
        
        # Select optimal device
        selected_device = self.router.select_device(workload)
        
        # Warmup phase - run both devices for comparison
        if not self.warmup_complete and self.total_requests < 10:
            return self._warmup_forward(query, key, value, attention_mask, need_weights, workload)
        
        # Execute on selected device
        start_time = time.perf_counter()
        
        try:
            if selected_device == 'photonic' and self.photonic_attention is not None:
                result = self.photonic_attention(query, key, value, attention_mask, need_weights)
                device_used = 'photonic'
            else:
                result = self.gpu_attention(query, key, value, attention_mask, need_weights)
                device_used = 'gpu'
            
            # Record performance
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            
            metrics = PerformanceMetrics(
                latency_ms=latency_ms,
                throughput_tokens_per_sec=workload.batch_size * workload.seq_length / (end_time - start_time),
                energy_mj=self._estimate_energy(device_used, workload),
                memory_mb=self._estimate_memory(workload),
            )
            
            self.router.update_performance(device_used, workload, metrics)
            
            return result
        
        except Exception as e:
            self.logger.error(f"Attention computation failed on {selected_device}: {e}")
            # Fallback to GPU if photonic fails
            if selected_device == 'photonic':
                return self.gpu_attention(query, key, value, attention_mask, need_weights)
            else:
                raise
    
    def _scaled_forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        need_weights: bool,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Scaled forward pass with load balancing and parallel processing.
        
        Splits large batches across multiple devices for parallel processing.
        """
        batch_size = query.shape[0]
        
        # For very large batches, split across devices
        if batch_size >= 8 and self.photonic_attention is not None:
            return self._parallel_batch_forward(query, key, value, attention_mask, need_weights)
        
        # For high concurrency, queue the request
        if self.executor is not None:
            future = self.executor.submit(
                self._standard_forward, query, key, value, attention_mask, need_weights
            )
            try:
                return future.result(timeout=30.0)  # 30 second timeout
            except TimeoutError:
                raise PhotonicTimeoutError("Request timeout in scaled processing", 30.0, "scaled_forward")
        else:
            return self._standard_forward(query, key, value, attention_mask, need_weights)
    
    def _parallel_batch_forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        need_weights: bool,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Process large batches in parallel across multiple devices."""
        batch_size = query.shape[0]
        split_size = batch_size // 2
        
        # Split inputs
        q1, q2 = query[:split_size], query[split_size:]
        k1 = key[:split_size] if key is not None else None
        k2 = key[split_size:] if key is not None else None
        v1 = value[:split_size] if value is not None else None
        v2 = value[split_size:] if value is not None else None
        mask1 = attention_mask[:split_size] if attention_mask is not None else None
        mask2 = attention_mask[split_size:] if attention_mask is not None else None
        
        # Submit parallel tasks
        futures = []
        if self.executor is not None:
            # GPU task
            gpu_future = self.executor.submit(
                self.gpu_attention, q1, k1, v1, mask1, need_weights
            )
            futures.append(('gpu', gpu_future))
            
            # Photonic task  
            if self.photonic_attention is not None:
                photonic_future = self.executor.submit(
                    self.photonic_attention, q2, k2, v2, mask2, need_weights
                )
                futures.append(('photonic', photonic_future))
            else:
                # Fallback to GPU for second half
                gpu_future2 = self.executor.submit(
                    self.gpu_attention, q2, k2, v2, mask2, need_weights
                )
                futures.append(('gpu', gpu_future2))
            
            # Collect results
            results = []
            for device, future in futures:
                try:
                    result = future.result(timeout=30.0)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Parallel task failed on {device}: {e}")
                    # Fallback computation
                    if device == 'photonic':
                        result = self.gpu_attention(q2, k2, v2, mask2, need_weights)
                    else:
                        result = self.gpu_attention(q1, k1, v1, mask1, need_weights)
                    results.append(result)
            
            # Combine results
            if need_weights:
                outputs, weights = zip(*results)
                combined_output = torch.cat(outputs, dim=0)
                combined_weights = torch.cat(weights, dim=0) if weights[0] is not None else None
                return combined_output, combined_weights
            else:
                outputs = results
                return torch.cat(outputs, dim=0)
        
        else:
            # Sequential fallback
            return self._standard_forward(query, key, value, attention_mask, need_weights)
    
    def _warmup_forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        need_weights: bool,
        workload: WorkloadCharacteristics,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Warmup phase - benchmark both devices for learning.
        """
        results = {}
        
        # Test GPU
        start_time = time.perf_counter()
        gpu_result = self.gpu_attention(query, key, value, attention_mask, need_weights)
        gpu_time = time.perf_counter() - start_time
        
        gpu_metrics = PerformanceMetrics(
            latency_ms=gpu_time * 1000,
            throughput_tokens_per_sec=workload.batch_size * workload.seq_length / gpu_time,
            energy_mj=self._estimate_energy('gpu', workload),
        )
        self.router.update_performance('gpu', workload, gpu_metrics)
        results['gpu'] = (gpu_result, gpu_time)
        
        # Test photonic if available
        if self.photonic_attention is not None:
            try:
                start_time = time.perf_counter()
                photonic_result = self.photonic_attention(query, key, value, attention_mask, need_weights)
                photonic_time = time.perf_counter() - start_time
                
                photonic_metrics = PerformanceMetrics(
                    latency_ms=photonic_time * 1000,
                    throughput_tokens_per_sec=workload.batch_size * workload.seq_length / photonic_time,
                    energy_mj=self._estimate_energy('photonic', workload),
                )
                self.router.update_performance('photonic', workload, photonic_metrics)
                results['photonic'] = (photonic_result, photonic_time)
                
            except Exception as e:
                self.logger.warning(f"Photonic warmup failed: {e}")
        
        # Mark warmup complete after sufficient samples
        if self.total_requests >= 10:
            self.warmup_complete = True
            self.logger.info("Warmup phase completed")
        
        # Return the faster result
        if 'photonic' in results and results['photonic'][1] < results['gpu'][1]:
            return results['photonic'][0]
        else:
            return results['gpu'][0]
    
    def _estimate_energy(self, device: str, workload: WorkloadCharacteristics) -> float:
        """Estimate energy consumption in millijoules."""
        # Simplified energy model
        ops = workload.batch_size * workload.seq_length * workload.seq_length * workload.embed_dim
        
        if device == 'gpu':
            # GPU: ~300W TDP, ~50 TOPS
            energy_per_op = 300 / 50e12 * 1000  # mJ per operation
        else:
            # Photonic: ~10W, ~10 TOPS (estimated)
            energy_per_op = 10 / 10e12 * 1000  # mJ per operation
        
        return ops * energy_per_op
    
    def _estimate_memory(self, workload: WorkloadCharacteristics) -> float:
        """Estimate memory usage in MB."""
        elements = workload.batch_size * workload.seq_length * workload.embed_dim
        bytes_per_element = 4 if workload.dtype == torch.float32 else 2
        return elements * bytes_per_element / (1024 * 1024)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            'total_requests': self.total_requests,
            'concurrent_requests': self.concurrent_requests,
            'peak_concurrent': self.peak_concurrent,
            'warmup_complete': self.warmup_complete,
            'scaling_enabled': self.enable_scaling,
            'max_concurrent': self.max_concurrent_requests,
        }
        
        # Add router stats
        stats.update(self.router.get_stats())
        
        # Add device-specific stats
        if self.photonic_attention:
            stats['photonic_stats'] = self.photonic_attention.get_performance_stats()
        stats['gpu_stats'] = self.gpu_attention.get_performance_stats()
        
        return stats
    
    def enable_auto_scaling(self, enabled: bool = True, max_concurrent: Optional[int] = None) -> None:
        """Enable or disable auto-scaling."""
        self.enable_scaling = enabled
        if max_concurrent is not None:
            self.max_concurrent_requests = max_concurrent
        
        if enabled and self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent_requests)
        elif not enabled and self.executor is not None:
            self.executor.shutdown(wait=False)
            self.executor = None
        
        self.logger.info(f"Auto-scaling {'enabled' if enabled else 'disabled'}")
    
    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self.total_requests = 0
        self.concurrent_requests = 0
        self.peak_concurrent = 0
        self.warmup_complete = False
        
        # Clear router cache and history
        self.router = AdaptiveRouter()
        
        self.logger.info("Performance statistics reset")
    
    def __del__(self):
        """Cleanup resources."""
        if self.executor is not None:
            self.executor.shutdown(wait=True)