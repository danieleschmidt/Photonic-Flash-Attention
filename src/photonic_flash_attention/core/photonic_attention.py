"""Photonic attention implementation with comprehensive error handling."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any, Union
from ..config import get_config
from ..photonic.hardware.detection import get_best_photonic_device, PhotonicDevice
from ..photonic.optical_kernels.matrix_mult import OpticalMatMul
from ..photonic.optical_kernels.nonlinearity import OpticalSoftmax
from ..utils.logging import get_logger
from ..utils.validation import validate_tensor_shape, validate_attention_inputs
from ..utils.exceptions import PhotonicHardwareError, PhotonicComputationError


class PhotonicAttention(nn.Module):
    """
    Photonic attention implementation with full error handling and validation.
    
    Uses silicon photonic hardware for matrix operations and attention computation.
    Falls back to GPU if photonic hardware fails or is unavailable.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        safety_checks: bool = True,
    ):
        super().__init__()
        
        # Validate inputs
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError(f"dropout must be between 0 and 1, got {dropout}")
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.safety_checks = safety_checks
        
        # Logger
        self.logger = get_logger(f"{self.__class__.__name__}")
        
        # Configuration
        self.config = get_config()
        
        # Thermal protection (set before hardware initialization)
        self.thermal_shutdown_temp = self.config.thermal_shutdown_temp
        self.thermal_warning_temp = self.thermal_shutdown_temp - 10.0
        
        # Performance and safety tracking
        self.last_latency_ms = 0.0
        self.last_energy_mj = 0.0
        self.last_temperature_c = 0.0
        self.failure_count = 0
        self.max_failures = 3
        self.is_degraded = False
        
        # Hardware detection and validation
        self.photonic_device = None
        self.device_validated = False
        self._initialize_photonic_hardware()
        
        # Handle device specification
        actual_device = None if device == 'auto' else device
        
        # Neural network layers
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, device=actual_device, dtype=dtype)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, device=actual_device, dtype=dtype)
        
        # Dropout
        self.dropout_module = nn.Dropout(dropout) if dropout > 0 else None
        
        # Photonic kernels
        self.optical_matmul = None
        self.optical_softmax = None
        self._initialize_optical_kernels()
        
        self.logger.info(f"Initialized PhotonicAttention: {embed_dim}d, {num_heads} heads, device={self.photonic_device.device_id if self.photonic_device else 'None'}")
    
    def _initialize_photonic_hardware(self) -> None:
        """Initialize and validate photonic hardware."""
        try:
            self.photonic_device = get_best_photonic_device()
            
            if self.photonic_device is None:
                self.logger.warning("No photonic hardware detected - using simulation mode")
                return
            
            # Validate device capabilities
            if self.photonic_device.wavelengths < self.num_heads:
                self.logger.error(f"Insufficient wavelengths: need {self.num_heads}, have {self.photonic_device.wavelengths}")
                raise PhotonicHardwareError(f"Device has insufficient wavelengths for {self.num_heads} heads")
            
            # Check temperature if available
            if self.photonic_device.temperature is not None:
                if self.photonic_device.temperature > self.thermal_shutdown_temp:
                    raise PhotonicHardwareError(f"Device temperature too high: {self.photonic_device.temperature}°C > {self.thermal_shutdown_temp}°C")
            
            self.device_validated = True
            self.logger.info(f"Validated photonic device: {self.photonic_device.device_type}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize photonic hardware: {e}")
            self.photonic_device = None
            if self.safety_checks:
                raise PhotonicHardwareError(f"Hardware initialization failed: {e}")
    
    def _initialize_optical_kernels(self) -> None:
        """Initialize optical computation kernels."""
        if self.photonic_device is None:
            return
        
        try:
            from ..photonic.optical_kernels.matrix_mult import OpticalMatMulConfig
            from ..photonic.optical_kernels.nonlinearity import OpticalNonlinearityConfig
            
            # Create optical matrix multiply configuration
            matmul_config = OpticalMatMulConfig(
                n_wavelengths=min(self.photonic_device.wavelengths, self.num_heads * 2),
                modulator_resolution=self.config.modulator_resolution,
            )
            self.optical_matmul = OpticalMatMul(matmul_config)
            
            # Create optical softmax
            softmax_config = OpticalNonlinearityConfig(
                n_wavelengths=min(self.photonic_device.wavelengths, self.num_heads * 2),
            )
            self.optical_softmax = OpticalSoftmax(softmax_config)
            
            self.logger.info("Initialized optical kernels successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize optical kernels: {e}")
            self.optical_matmul = None
            self.optical_softmax = None
            
            if self.safety_checks:
                raise PhotonicComputationError(f"Kernel initialization failed: {e}")
    
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with comprehensive error handling and validation.
        
        Args:
            query: Query tensor [batch, seq_len, embed_dim]
            key: Key tensor (optional)
            value: Value tensor (optional)
            attention_mask: Attention mask (optional)
            need_weights: Whether to return attention weights
            
        Returns:
            output: Attention output [batch, seq_len, embed_dim]
            weights: Attention weights if requested
            
        Raises:
            PhotonicComputationError: If photonic computation fails
            ValueError: If inputs are invalid
        """
        # Input validation
        if self.safety_checks:
            self._validate_inputs(query, key, value, attention_mask)
        
        # Thermal safety check
        if not self._check_thermal_safety():
            self.logger.warning("Thermal safety check failed - falling back to CPU/GPU")
            return self._fallback_forward(query, key, value, attention_mask, need_weights)
        
        # Performance timing
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time:
            start_time.record()
        
        try:
            # Main computation
            output, weights = self._photonic_forward(query, key, value, attention_mask, need_weights)
            
            # Reset failure count on success
            if self.failure_count > 0:
                self.logger.info(f"Photonic computation recovered after {self.failure_count} failures")
                self.failure_count = 0
                self.is_degraded = False
            
        except Exception as e:
            self.logger.error(f"Photonic computation failed: {e}")
            self.failure_count += 1
            
            if self.failure_count >= self.max_failures:
                self.logger.critical(f"Too many failures ({self.failure_count}), marking as degraded")
                self.is_degraded = True
            
            # Fallback to CPU/GPU
            output, weights = self._fallback_forward(query, key, value, attention_mask, need_weights)
        
        # Performance tracking
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            self.last_latency_ms = start_time.elapsed_time(end_time)
        
        # Post-computation validation
        if self.safety_checks:
            self._validate_outputs(output, weights, query.shape)
        
        return output, weights if need_weights else None
    
    def _validate_inputs(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> None:
        """Validate input tensors."""
        # Basic tensor validation
        validate_attention_inputs(query, key, value, attention_mask)
        
        # Shape validation
        batch_size, seq_len, embed_dim = query.shape
        if embed_dim != self.embed_dim:
            raise ValueError(f"Query embed_dim {embed_dim} doesn't match expected {self.embed_dim}")
        
        # Device validation
        if not query.is_cuda and self.photonic_device and self.photonic_device.device_type != "simulation":
            self.logger.warning("Input tensor not on CUDA - this may hurt performance")
        
        # Sequence length limits
        max_seq_len = getattr(self.config, 'max_sequence_length', 8192)
        if seq_len > max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {max_seq_len}")
        
        # Memory usage estimation
        estimated_memory = batch_size * seq_len * seq_len * 4  # bytes
        if estimated_memory > 8e9:  # 8GB limit
            self.logger.warning(f"Large memory usage estimated: {estimated_memory/1e9:.1f}GB")
    
    def _validate_outputs(
        self,
        output: torch.Tensor,
        weights: Optional[torch.Tensor],
        input_shape: torch.Size,
    ) -> None:
        """Validate output tensors."""
        # Output shape validation
        if output.shape != input_shape:
            raise PhotonicComputationError(f"Output shape {output.shape} doesn't match input {input_shape}")
        
        # NaN/Inf detection
        if torch.isnan(output).any():
            raise PhotonicComputationError("NaN detected in attention output")
        if torch.isinf(output).any():
            raise PhotonicComputationError("Inf detected in attention output")
        
        # Attention weights validation
        if weights is not None:
            if torch.isnan(weights).any():
                raise PhotonicComputationError("NaN detected in attention weights")
            
            # Check if weights sum to 1
            weight_sums = weights.sum(dim=-1)
            if not torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-3):
                self.logger.warning("Attention weights don't sum to 1 - possible numerical issues")
    
    def _check_thermal_safety(self) -> bool:
        """Check thermal safety conditions."""
        if not self.config.temperature_monitoring:
            return True
        
        if self.photonic_device is None or self.photonic_device.temperature is None:
            return True  # No temperature monitoring available
        
        temp = self.photonic_device.temperature
        self.last_temperature_c = temp
        
        if temp > self.thermal_shutdown_temp:
            self.logger.critical(f"THERMAL SHUTDOWN: {temp}°C > {self.thermal_shutdown_temp}°C")
            return False
        
        if temp > self.thermal_warning_temp:
            self.logger.warning(f"High temperature warning: {temp}°C > {self.thermal_warning_temp}°C")
        
        return True
    
    def _photonic_forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        need_weights: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Main photonic computation path."""
        if self.is_degraded or self.optical_matmul is None:
            raise PhotonicComputationError("Photonic hardware is degraded or unavailable")
        
        batch_size, seq_len, _ = query.shape
        
        # Handle self-attention
        if key is None:
            key = query
        if value is None:
            value = query
        
        # QKV projection using optical matrix multiplication
        if torch.equal(query, key) and torch.equal(query, value):
            # Self-attention: compute QKV in one operation
            qkv_weights = self.qkv_proj.weight.T  # [embed_dim, 3*embed_dim] -> [3*embed_dim, embed_dim]
            qkv = self.optical_matmul.forward(query, qkv_weights)
            if self.qkv_proj.bias is not None:
                qkv = qkv + self.qkv_proj.bias
            q, k, v = qkv.chunk(3, dim=-1)
        else:
            # Cross-attention: separate projections
            q_weight, k_weight, v_weight = self.qkv_proj.weight.chunk(3, dim=0)
            q_bias, k_bias, v_bias = self.qkv_proj.bias.chunk(3, dim=0) if self.qkv_proj.bias is not None else (None, None, None)
            
            q = self.optical_matmul.forward(query, q_weight.T)
            if q_bias is not None:
                q = q + q_bias
            k = self.optical_matmul.forward(key, k_weight.T)
            if k_bias is not None:
                k = k + k_bias
            v = self.optical_matmul.forward(value, v_weight.T)
            if v_bias is not None:
                v = v + v_bias
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scale queries
        q = q * self.scaling
        
        # Optical attention computation
        attn_scores = self.optical_matmul.forward(q, k.transpose(-2, -1))
        
        # Apply attention mask
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Optical softmax
        attn_weights = self.optical_softmax.forward(attn_scores, dim=-1)
        
        # Apply dropout
        if self.dropout_module is not None:
            attn_weights = self.dropout_module(attn_weights)
        
        # Compute attention output
        attn_output = self.optical_matmul.forward(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.optical_matmul.forward(attn_output, self.out_proj.weight.T)
        if self.out_proj.bias is not None:
            output = output + self.out_proj.bias
        
        return output, attn_weights if need_weights else None
    
    def _fallback_forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        need_weights: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Fallback to standard GPU/CPU attention."""
        from .flash_attention_3 import FlashAttention3
        
        # Create temporary GPU attention module
        if not hasattr(self, '_fallback_attention'):
            self._fallback_attention = FlashAttention3(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                bias=self.qkv_proj.bias is not None,
                device=query.device,
                dtype=query.dtype,
            )
            
            # Copy weights
            self._fallback_attention.qkv_proj.weight.data = self.qkv_proj.weight.data
            self._fallback_attention.out_proj.weight.data = self.out_proj.weight.data
            if self.qkv_proj.bias is not None:
                self._fallback_attention.qkv_proj.bias.data = self.qkv_proj.bias.data
            if self.out_proj.bias is not None:
                self._fallback_attention.out_proj.bias.data = self.out_proj.bias.data
        
        return self._fallback_attention(query, key, value, attention_mask, need_weights)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance and health statistics."""
        stats = {
            "device": "photonic",
            "implementation": "photonic_attention",
            "latency_ms": self.last_latency_ms,
            "energy_mj": self.last_energy_mj,
            "temperature_c": self.last_temperature_c,
            "failure_count": self.failure_count,
            "is_degraded": self.is_degraded,
            "device_validated": self.device_validated,
        }
        
        if self.photonic_device:
            stats.update({
                "device_id": self.photonic_device.device_id,
                "device_type": self.photonic_device.device_type,
                "vendor": self.photonic_device.vendor,
                "wavelengths": self.photonic_device.wavelengths,
                "max_optical_power_mw": self.photonic_device.max_optical_power * 1000,
            })
        
        return stats
    
    def reset_error_state(self) -> None:
        """Reset error state and attempt recovery."""
        self.logger.info("Resetting error state and attempting recovery")
        self.failure_count = 0
        self.is_degraded = False
        
        # Re-initialize hardware
        try:
            self._initialize_photonic_hardware()
            self._initialize_optical_kernels()
            self.logger.info("Recovery successful")
        except Exception as e:
            self.logger.error(f"Recovery failed: {e}")
    
    def enable_safety_checks(self, enabled: bool = True) -> None:
        """Enable or disable safety checks (for performance)."""
        self.safety_checks = enabled
        self.logger.info(f"Safety checks {'enabled' if enabled else 'disabled'}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get detailed health status of the photonic system."""
        return {
            "overall_health": "healthy" if not self.is_degraded else "degraded",
            "hardware_available": self.photonic_device is not None,
            "device_validated": self.device_validated,
            "failure_count": self.failure_count,
            "max_failures": self.max_failures,
            "thermal_status": "ok" if self.last_temperature_c < self.thermal_warning_temp else "warning",
            "last_temperature_c": self.last_temperature_c,
            "thermal_limits": {
                "warning": self.thermal_warning_temp,
                "shutdown": self.thermal_shutdown_temp,
            },
            "optical_kernels_available": {
                "matrix_multiply": self.optical_matmul is not None,
                "softmax": self.optical_softmax is not None,
            },
        }