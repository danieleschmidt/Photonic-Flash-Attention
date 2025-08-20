"""
Optical matrix multiplication kernels for photonic computing.

This module implements high-performance matrix multiplication using silicon photonic
devices including wavelength division multiplexing (WDM), optical crossbars,
and coherent optical computing primitives.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
import logging
from enum import Enum

from ...utils.exceptions import PhotonicComputeError, HardwareNotAvailableError
from ...utils.validation import validate_optical_tensor, validate_matrix_dimensions
from ...config import get_config

logger = logging.getLogger(__name__)


class OpticalPrecision(Enum):
    """Optical computation precision levels."""
    FP16 = "fp16"
    FP32 = "fp32" 
    MIXED = "mixed"
    ANALOG = "analog"


@dataclass
class OpticalMatMulConfig:
    """Configuration for optical matrix multiplication."""
    n_wavelengths: int = 80
    modulator_resolution: int = 6  # bits
    extinction_ratio: float = 20.0  # dB
    insertion_loss: float = 0.5  # dB
    crosstalk_suppression: float = -30.0  # dB
    detector_responsivity: float = 1.0  # A/W
    optical_power_budget: float = 10.0  # W (simulation mode - no real power constraints)
    wavelength_spacing: float = 100e9  # Hz
    temperature_sensitivity: float = 0.1  # nm/K
    precision: OpticalPrecision = OpticalPrecision.FP16


class WavelengthBank:
    """Manages wavelength allocation for optical computing."""
    
    def __init__(self, n_wavelengths: int = 80):
        self.n_wavelengths = n_wavelengths
        self.allocated_channels = set()
        self.channel_map = {}
        self._base_wavelength = 1550e-9  # m
        self._channel_spacing = 100e9  # Hz
        
    def allocate_channels(self, n_channels: int) -> List[int]:
        """Allocate optical channels for computation."""
        if len(self.allocated_channels) + n_channels > self.n_wavelengths:
            raise PhotonicComputeError(
                f"Insufficient wavelengths: need {n_channels}, "
                f"have {self.n_wavelengths - len(self.allocated_channels)} available"
            )
        
        channels = []
        for i in range(self.n_wavelengths):
            if i not in self.allocated_channels and len(channels) < n_channels:
                channels.append(i)
                self.allocated_channels.add(i)
        
        return channels
    
    def release_channels(self, channels: List[int]) -> None:
        """Release optical channels."""
        for channel in channels:
            self.allocated_channels.discard(channel)
    
    def get_wavelength(self, channel: int) -> float:
        """Get wavelength for channel in meters."""
        c = 299792458  # m/s
        freq = c / self._base_wavelength + channel * self._channel_spacing
        return c / freq


class OpticalCrossbar:
    """Silicon photonic crossbar switch for matrix operations."""
    
    def __init__(self, config: OpticalMatMulConfig):
        self.config = config
        self.crossbar_size = min(128, config.n_wavelengths)  # Realistic limit
        self.switching_matrix = torch.eye(self.crossbar_size)
        self.insertion_loss_matrix = torch.full(
            (self.crossbar_size, self.crossbar_size), 
            config.insertion_loss
        )
        self.crosstalk_matrix = self._generate_crosstalk_matrix()
        
    def _generate_crosstalk_matrix(self) -> torch.Tensor:
        """Generate realistic crosstalk matrix for photonic crossbar."""
        size = self.crossbar_size
        crosstalk = torch.zeros(size, size)
        
        # Adjacent channel crosstalk
        for i in range(size - 1):
            crosstalk[i, i + 1] = 10 ** (self.config.crosstalk_suppression / 10)
            crosstalk[i + 1, i] = 10 ** (self.config.crosstalk_suppression / 10)
        
        # Diagonal elements (self-coupling)
        crosstalk.fill_diagonal_(1.0)
        
        return crosstalk
    
    def configure_routing(self, input_channels: List[int], 
                         output_channels: List[int]) -> torch.Tensor:
        """Configure crossbar routing matrix."""
        routing_matrix = torch.zeros(self.crossbar_size, self.crossbar_size)
        
        for i, (inp, out) in enumerate(zip(input_channels, output_channels)):
            if inp < self.crossbar_size and out < self.crossbar_size:
                routing_matrix[inp, out] = 1.0
        
        # Apply crosstalk and insertion loss
        routing_matrix = routing_matrix * self.crosstalk_matrix
        routing_matrix = routing_matrix * torch.exp(-self.insertion_loss_matrix / 10)
        
        return routing_matrix


class OpticalMatMul:
    """High-performance optical matrix multiplication kernel."""
    
    def __init__(self, config: Optional[OpticalMatMulConfig] = None):
        self.config = config or OpticalMatMulConfig()
        self.wavelength_bank = WavelengthBank(self.config.n_wavelengths)
        self.crossbar = OpticalCrossbar(self.config)
        self.device_cache = {}
        self.performance_stats = {
            "operations": 0,
            "total_latency": 0.0,
            "energy_consumed": 0.0,
        }
        
        logger.info(f"Initialized OpticalMatMul with {self.config.n_wavelengths} wavelengths")
    
    def validate_inputs(self, a: torch.Tensor, b: torch.Tensor) -> None:
        """Validate input tensors for optical computation."""
        validate_optical_tensor(a)
        validate_optical_tensor(b)
        validate_matrix_dimensions(a, b)
        
        if a.device != b.device:
            raise PhotonicComputeError("Input tensors must be on same device")
        
        # Check optical power constraints
        max_power = max(a.abs().max().item(), b.abs().max().item())
        if max_power > self.config.optical_power_budget:
            raise PhotonicComputeError(
                f"Input power {max_power:.3e} W exceeds budget "
                f"{self.config.optical_power_budget:.3e} W"
            )
    
    def encode_to_optical(self, tensor: torch.Tensor, 
                         channels: List[int]) -> torch.Tensor:
        """Encode electronic signal to optical domain."""
        batch_size = tensor.shape[0] if tensor.dim() > 2 else 1
        
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        
        # Quantize to modulator resolution
        if self.config.precision != OpticalPrecision.ANALOG:
            n_levels = 2 ** self.config.modulator_resolution
            tensor = torch.round(tensor * n_levels) / n_levels
        
        # Wavelength division encoding
        optical_tensor = torch.zeros(
            batch_size, len(channels), *tensor.shape[-2:],
            dtype=tensor.dtype, device=tensor.device
        )
        
        # Distribute data across wavelength channels
        for i, channel in enumerate(channels):
            if i < tensor.shape[-2]:  # Handle dimension mismatch gracefully
                # Simple WDM encoding - each wavelength carries a slice
                optical_tensor[:, i] = tensor[:, i % tensor.shape[-2]]
        
        # Apply modulator transfer function
        optical_tensor = self._apply_modulator_response(optical_tensor)
        
        return optical_tensor
    
    def _apply_modulator_response(self, optical_signal: torch.Tensor) -> torch.Tensor:
        """Apply electro-optic modulator transfer function."""
        # Mach-Zehnder modulator response
        extinction_ratio_linear = 10 ** (self.config.extinction_ratio / 10)
        
        # Sinusoidal transfer function with extinction ratio
        modulated = (1 + extinction_ratio_linear * torch.cos(np.pi * optical_signal)) / 2
        modulated = torch.clamp(modulated, 0.0, 1.0)
        
        return modulated
    
    def optical_matrix_multiply(self, a_optical: torch.Tensor, 
                              b_optical: torch.Tensor,
                              a_channels: List[int],
                              b_channels: List[int]) -> torch.Tensor:
        """Perform optical matrix multiplication using photonic crossbar."""
        batch_size = a_optical.shape[0]
        
        # Configure crossbar for matrix multiplication
        routing_matrix = self.crossbar.configure_routing(a_channels, b_channels)
        
        # Optical MAC (Multiply-Accumulate) operations
        result_optical = torch.zeros(
            batch_size, len(b_channels), 
            a_optical.shape[-2], b_optical.shape[-1],
            dtype=a_optical.dtype, device=a_optical.device
        )
        
        # Parallel wavelength-domain computation
        for batch_idx in range(batch_size):
            # Extract batch slices
            a_batch = a_optical[batch_idx]  # [n_channels_a, M, K]
            b_batch = b_optical[batch_idx]  # [n_channels_b, K, N] 
            
            # Optical matrix multiplication via crossbar routing
            for i, ch_a in enumerate(a_channels):
                for j, ch_b in enumerate(b_channels):
                    if i < a_batch.shape[0] and j < b_batch.shape[0]:
                        # Apply routing loss and crosstalk
                        coupling_efficiency = routing_matrix[ch_a % routing_matrix.shape[0], 
                                                           ch_b % routing_matrix.shape[1]]
                        
                        # Optical multiplication (intensity modulation)
                        optical_product = coupling_efficiency * torch.matmul(
                            a_batch[i], b_batch[j]
                        )
                        
                        result_optical[batch_idx, j] += optical_product
        
        return result_optical
    
    def decode_from_optical(self, optical_tensor: torch.Tensor,
                           channels: List[int]) -> torch.Tensor:
        """Decode optical signal back to electronic domain."""
        # Apply photodetector response
        electronic_signal = self._apply_photodetector_response(optical_tensor)
        
        # Wavelength division demultiplexing
        batch_size = electronic_signal.shape[0]
        channel_data = electronic_signal.shape[1]
        
        # Coherent summation across wavelength channels
        if channel_data > 1:
            result = torch.sum(electronic_signal, dim=1)
        else:
            result = electronic_signal.squeeze(1)
        
        # Remove batch dimension if originally 2D
        if len(result.shape) == 3 and result.shape[0] == 1:
            result = result.squeeze(0)
        
        return result
    
    def _apply_photodetector_response(self, optical_signal: torch.Tensor) -> torch.Tensor:
        """Apply photodetector responsivity and noise."""
        # Apply responsivity
        electronic_signal = self.config.detector_responsivity * optical_signal
        
        # Add shot noise (Poisson noise)
        if self.training and optical_signal.requires_grad:
            # Only add noise during training to maintain differentiability
            noise_std = torch.sqrt(electronic_signal.abs() * 1.602e-19 / (2 * 50))  # Shot noise
            noise = torch.randn_like(electronic_signal) * noise_std
            electronic_signal = electronic_signal + 0.1 * noise  # Scaled down for stability
        
        return electronic_signal
    
    @property
    def training(self) -> bool:
        """Check if we're in training mode (heuristic)."""
        return torch.is_grad_enabled()
    
    def forward(self, a: torch.Tensor, b: torch.Tensor, 
                mode: str = "auto") -> torch.Tensor:
        """
        Perform optical matrix multiplication.
        
        Args:
            a: Input tensor A [M, K] or [batch, M, K]
            b: Input tensor B [K, N] or [batch, K, N] 
            mode: Computation mode ('auto', 'wavelength_parallel', 'spatial_parallel')
            
        Returns:
            Result tensor [M, N] or [batch, M, N]
        """
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time:
            start_time.record()
        
        try:
            # Validate inputs
            self.validate_inputs(a, b)
            
            # Determine optimal computation strategy
            if mode == "auto":
                mode = self._select_optimal_mode(a, b)
            
            # Allocate wavelength channels
            channels_needed = max(a.shape[-2], b.shape[-1])
            channels = self.wavelength_bank.allocate_channels(
                min(channels_needed, self.config.n_wavelengths)
            )
            
            try:
                # Encode to optical domain
                a_optical = self.encode_to_optical(a, channels[:len(channels)//2])
                b_optical = self.encode_to_optical(b, channels[len(channels)//2:])
                
                # Optical computation
                result_optical = self.optical_matrix_multiply(
                    a_optical, b_optical,
                    channels[:len(channels)//2],
                    channels[len(channels)//2:]
                )
                
                # Decode back to electronic domain
                result = self.decode_from_optical(result_optical, channels)
                
            finally:
                # Always release channels
                self.wavelength_bank.release_channels(channels)
            
            # Record performance metrics
            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                latency = start_time.elapsed_time(end_time)
                self._update_performance_stats(latency, a, b)
            
            logger.debug(f"Optical matmul completed: {a.shape} @ {b.shape} -> {result.shape}")
            
            return result
            
        except Exception as e:
            logger.error(f"Optical matrix multiplication failed: {e}")
            raise PhotonicComputeError(f"Optical computation failed: {e}") from e
    
    def _select_optimal_mode(self, a: torch.Tensor, b: torch.Tensor) -> str:
        """Select optimal computation mode based on tensor dimensions."""
        total_ops = a.shape[-2] * a.shape[-1] * b.shape[-1]
        
        if total_ops > 1e6:  # Large computation
            return "wavelength_parallel"
        else:
            return "spatial_parallel"
    
    def _update_performance_stats(self, latency_ms: float, 
                                 a: torch.Tensor, b: torch.Tensor) -> None:
        """Update performance statistics."""
        self.performance_stats["operations"] += 1
        self.performance_stats["total_latency"] += latency_ms
        
        # Estimate energy consumption (rough approximation)
        ops = a.shape[-2] * a.shape[-1] * b.shape[-1]
        energy_per_op = 1e-12  # J per operation (very rough estimate)
        energy = ops * energy_per_op
        self.performance_stats["energy_consumed"] += energy
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        stats = self.performance_stats.copy()
        if stats["operations"] > 0:
            stats["avg_latency_ms"] = stats["total_latency"] / stats["operations"]
            stats["avg_energy_per_op"] = stats["energy_consumed"] / stats["operations"]
        return stats
    
    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self.performance_stats = {
            "operations": 0,
            "total_latency": 0.0,
            "energy_consumed": 0.0,
        }


def optical_matmul(a: torch.Tensor, b: torch.Tensor, 
                   config: Optional[OpticalMatMulConfig] = None) -> torch.Tensor:
    """
    Convenience function for optical matrix multiplication.
    
    Args:
        a: Input tensor A
        b: Input tensor B  
        config: Optional configuration
        
    Returns:
        Matrix multiplication result using optical computation
    """
    optical_mm = OpticalMatMul(config)
    return optical_mm.forward(a, b)


# Global optical matrix multiplier for efficiency
_global_optical_matmul = None

def get_global_optical_matmul() -> OpticalMatMul:
    """Get global optical matrix multiplier instance."""
    global _global_optical_matmul
    if _global_optical_matmul is None:
        config = OpticalMatMulConfig(
            n_wavelengths=get_config().photonic_wavelengths,
            modulator_resolution=get_config().modulator_resolution
        )
        _global_optical_matmul = OpticalMatMul(config)
    return _global_optical_matmul