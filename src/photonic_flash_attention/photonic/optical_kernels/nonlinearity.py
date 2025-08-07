"""
Optical nonlinearity operations for photonic computing.

This module implements all-optical activation functions, softmax operations,
and other nonlinear transformations using silicon photonic devices and
nonlinear optical effects.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict, Any, List, Union
from dataclasses import dataclass
import logging
from enum import Enum

from ...utils.exceptions import PhotonicComputeError
from ...utils.validation import validate_optical_tensor
from ...config import get_config
from .matrix_mult import OpticalPrecision, WavelengthBank

logger = logging.getLogger(__name__)


class NonlinearityType(Enum):
    """Types of optical nonlinearities."""
    SOFTMAX = "softmax"
    RELU = "relu"
    GELU = "gelu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    SWISH = "swish"
    LAYERNORM = "layernorm"


@dataclass
class OpticalNonlinearityConfig:
    """Configuration for optical nonlinearity operations."""
    n_wavelengths: int = 80
    power_threshold: float = 1e-3  # W
    saturation_power: float = 10e-3  # W
    response_time: float = 1e-12  # s (picosecond response)
    chi3_coefficient: float = 2.5e-22  # m²/W (silicon)
    channel_spacing: float = 100e9  # Hz
    temperature_coefficient: float = 0.1  # nm/K
    precision: OpticalPrecision = OpticalPrecision.FP16
    enable_approximations: bool = True  # Use approximations for complex functions


class OpticalSoftmax:
    """All-optical softmax implementation using nonlinear optical effects."""
    
    def __init__(self, config: Optional[OpticalNonlinearityConfig] = None):
        self.config = config or OpticalNonlinearityConfig()
        self.wavelength_bank = WavelengthBank(self.config.n_wavelengths)
        self.performance_stats = {
            "operations": 0,
            "total_latency": 0.0,
            "energy_consumed": 0.0,
        }
        
    def optical_exp_approximation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Approximate exponential function using optical nonlinearity.
        
        Uses Kerr effect and saturable absorption to approximate exp(x).
        For practical implementation, we use polynomial approximation
        in the optical domain.
        """
        # Clamp input to prevent optical damage
        x_clamped = torch.clamp(x, -10.0, 10.0)
        
        if self.config.enable_approximations:
            # Use polynomial approximation suitable for optical implementation
            # exp(x) ≈ 1 + x + x²/2 + x³/6 for small x
            # For larger x, use piecewise approximation
            
            # Optical polynomial computation using intensity modulation
            x2 = x_clamped * x_clamped  # Optical squaring via intensity
            x3 = x2 * x_clamped        # Third-order term
            
            # Approximate exponential with optical-friendly coefficients
            result = 1.0 + x_clamped + 0.5 * x2 + 0.167 * x3
            
            # Apply optical saturation characteristic
            result = self._apply_optical_saturation(result)
        else:
            # Use PyTorch exp for comparison/fallback
            result = torch.exp(x_clamped)
        
        return result
    
    def _apply_optical_saturation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply optical power saturation effects."""
        # Saturable absorption model
        sat_power = self.config.saturation_power
        return x / (1.0 + x / sat_power)
    
    def optical_sum_reduction(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        All-optical summation using wavelength division multiplexing.
        
        Different wavelength channels carry different elements,
        then coherently combined for summation.
        """
        # Allocate wavelength channels for parallel computation
        n_channels = min(x.shape[dim], self.config.n_wavelengths)
        channels = self.wavelength_bank.allocate_channels(n_channels)
        
        try:
            # Encode elements across wavelength channels
            # In practice, this would involve electro-optic modulators
            wdm_encoded = self._wavelength_encode(x, channels, dim)
            
            # Coherent summation via optical combiners
            optical_sum = self._coherent_combine(wdm_encoded, channels)
            
            # Photodetection to convert back to electronic domain
            result = self._photodetect(optical_sum)
            
            return result
        
        finally:
            self.wavelength_bank.release_channels(channels)
    
    def _wavelength_encode(self, x: torch.Tensor, channels: List[int], 
                          dim: int) -> torch.Tensor:
        """Encode tensor elements across wavelength channels."""
        # Split tensor along specified dimension
        split_tensors = torch.split(x, 1, dim=dim)
        
        # Encode each split on a different wavelength
        encoded = []
        for i, tensor_slice in enumerate(split_tensors):
            if i < len(channels):
                # Apply wavelength-dependent phase/amplitude encoding
                wavelength = self._get_channel_wavelength(channels[i])
                phase_factor = torch.exp(1j * 2 * np.pi * wavelength * 1e15)  # Optical phase
                
                # For simplicity, use amplitude encoding
                encoded_slice = tensor_slice * phase_factor.real
                encoded.append(encoded_slice)
        
        return torch.stack(encoded, dim=0)
    
    def _coherent_combine(self, wdm_tensor: torch.Tensor, 
                         channels: List[int]) -> torch.Tensor:
        """Coherent combination of wavelength-multiplexed signals."""
        # Sum across wavelength dimension (coherent combining)
        combined = torch.sum(wdm_tensor, dim=0)
        
        # Apply combining losses and phase noise
        combining_efficiency = 0.9 ** len(channels)  # Loss scales with channel count
        return combined * combining_efficiency
    
    def _photodetect(self, optical_signal: torch.Tensor) -> torch.Tensor:
        """Convert optical signal to electronic via photodetection."""
        # Square-law detection (intensity)
        intensity = optical_signal.real ** 2 + optical_signal.imag ** 2
        
        # Apply detector responsivity and noise
        responsivity = 1.0  # A/W
        electronic_signal = responsivity * intensity
        
        # Add shot noise in training mode
        if torch.is_grad_enabled():
            shot_noise_std = torch.sqrt(electronic_signal.abs() * 1.602e-19)
            noise = torch.randn_like(electronic_signal) * shot_noise_std * 0.01
            electronic_signal = electronic_signal + noise
        
        return electronic_signal
    
    def _get_channel_wavelength(self, channel: int) -> float:
        """Get wavelength for specific channel."""
        base_wavelength = 1550e-9  # m
        spacing = 0.8e-9  # m (100 GHz)
        return base_wavelength + channel * spacing
    
    def forward(self, x: torch.Tensor, dim: int = -1, 
               temperature: float = 1.0) -> torch.Tensor:
        """
        All-optical softmax computation.
        
        Args:
            x: Input tensor
            dim: Dimension to apply softmax
            temperature: Temperature scaling parameter
            
        Returns:
            Softmax probabilities
        """
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time:
            start_time.record()
        
        try:
            # Validate input
            validate_optical_tensor(x)
            
            # Temperature scaling
            if temperature != 1.0:
                x = x / temperature
            
            # Numerical stability: subtract max
            x_max = torch.max(x, dim=dim, keepdim=True)[0]
            x_shifted = x - x_max
            
            # Optical exponential approximation
            exp_x = self.optical_exp_approximation(x_shifted)
            
            # All-optical summation
            sum_exp = self.optical_sum_reduction(exp_x, dim=dim)
            sum_exp = sum_exp.unsqueeze(dim) if sum_exp.dim() < x.dim() else sum_exp
            
            # Division (implemented as multiplication with optical inverse)
            softmax_probs = exp_x / (sum_exp + 1e-8)  # Add small epsilon for stability
            
            # Apply optical power normalization
            softmax_probs = torch.clamp(softmax_probs, 0.0, 1.0)
            
            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                latency = start_time.elapsed_time(end_time)
                self._update_performance_stats(latency, x)
            
            return softmax_probs
            
        except Exception as e:
            logger.error(f"Optical softmax failed: {e}")
            # Fallback to electronic computation
            logger.warning("Falling back to electronic softmax")
            return torch.softmax(x / temperature, dim=dim)


class OpticalActivations:
    """Collection of all-optical activation functions."""
    
    def __init__(self, config: Optional[OpticalNonlinearityConfig] = None):
        self.config = config or OpticalNonlinearityConfig()
    
    def optical_relu(self, x: torch.Tensor) -> torch.Tensor:
        """
        All-optical ReLU using electro-optic switches.
        
        Uses Mach-Zehnder interferometer with electronic control
        to create optical switching based on sign detection.
        """
        # In practice, this would use optical bistability or switching
        # For simulation, we model the optical-electronic-optical conversion
        
        # Electronic sign detection (would be done optically)
        sign_mask = (x > 0).float()
        
        # Optical switching based on sign
        # Models electro-optic switch response
        switch_response = self._optical_switch_response(sign_mask)
        
        # Apply switching to input signal
        result = x * switch_response
        
        return result
    
    def _optical_switch_response(self, control: torch.Tensor) -> torch.Tensor:
        """Model electro-optic switch response."""
        # Mach-Zehnder modulator transfer function
        # V_π is the voltage for π phase shift
        v_pi = 3.0  # Typical value in volts
        
        # Convert control signal to optical transmission
        phase_shift = np.pi * control / v_pi
        transmission = torch.cos(phase_shift) ** 2
        
        return transmission
    
    def optical_gelu(self, x: torch.Tensor) -> torch.Tensor:
        """
        Approximate GELU using optical polynomial approximation.
        
        GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
        """
        # Optical polynomial computation
        x3 = x * x * x  # Cubic term via optical multiplication
        
        # Polynomial approximation coefficients optimized for optical implementation
        sqrt_2_pi = np.sqrt(2.0 / np.pi)
        inner = sqrt_2_pi * (x + 0.044715 * x3)
        
        # Optical tanh approximation
        tanh_approx = self.optical_tanh(inner)
        
        # Final GELU computation
        result = 0.5 * x * (1.0 + tanh_approx)
        
        return result
    
    def optical_tanh(self, x: torch.Tensor) -> torch.Tensor:
        """
        Approximate tanh using optical saturation.
        
        Uses the natural saturation characteristics of optical amplifiers
        to approximate hyperbolic tangent function.
        """
        # Clamp to prevent optical damage
        x_clamped = torch.clamp(x, -5.0, 5.0)
        
        # Use optical saturation to approximate tanh
        # tanh(x) ≈ x / (1 + |x|) for moderate range
        saturation_approx = x_clamped / (1.0 + torch.abs(x_clamped))
        
        # Apply optical nonlinearity correction
        result = self._apply_kerr_nonlinearity(saturation_approx)
        
        return result
    
    def _apply_kerr_nonlinearity(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Kerr nonlinearity for improved approximation."""
        # Kerr effect introduces intensity-dependent refractive index
        # n = n₀ + n₂I, where I is optical intensity
        
        chi3 = self.config.chi3_coefficient
        intensity = x ** 2  # Optical intensity
        
        # Phase modulation due to Kerr effect
        phase_modulation = chi3 * intensity
        
        # Convert phase modulation back to amplitude
        kerr_response = x * torch.cos(phase_modulation)
        
        return kerr_response


class OpticalLayerNorm:
    """All-optical layer normalization using coherent optical processing."""
    
    def __init__(self, normalized_shape: Union[int, List[int]], 
                 config: Optional[OpticalNonlinearityConfig] = None):
        self.normalized_shape = normalized_shape if isinstance(normalized_shape, (list, tuple)) else [normalized_shape]
        self.config = config or OpticalNonlinearityConfig()
        
        # Optical parameters that would be learned
        self.optical_weight = torch.ones(self.normalized_shape)
        self.optical_bias = torch.zeros(self.normalized_shape)
    
    def forward(self, x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """
        All-optical layer normalization.
        
        Uses coherent optical processing to compute mean and variance,
        then applies optical scaling and shifting.
        """
        # Compute dimensions for normalization
        dims = list(range(len(x.shape) - len(self.normalized_shape), len(x.shape)))
        
        # All-optical mean computation
        mean = self._optical_mean(x, dims)
        
        # All-optical variance computation  
        var = self._optical_variance(x, mean, dims)
        
        # Optical normalization
        x_normalized = (x - mean) / torch.sqrt(var + eps)
        
        # Apply learned optical scaling and bias
        result = x_normalized * self.optical_weight + self.optical_bias
        
        return result
    
    def _optical_mean(self, x: torch.Tensor, dims: List[int]) -> torch.Tensor:
        """Compute mean using optical averaging."""
        # Would use optical power splitters and combiners
        return torch.mean(x, dim=dims, keepdim=True)
    
    def _optical_variance(self, x: torch.Tensor, mean: torch.Tensor, 
                         dims: List[int]) -> torch.Tensor:
        """Compute variance using optical intensity measurements."""
        # Optical variance computation via intensity detection
        centered = x - mean
        variance = torch.mean(centered * centered, dim=dims, keepdim=True)
        
        return variance


class OpticalNonlinearityKernel:
    """Unified kernel for all optical nonlinearity operations."""
    
    def __init__(self, config: Optional[OpticalNonlinearityConfig] = None):
        self.config = config or OpticalNonlinearityConfig()
        self.softmax = OpticalSoftmax(config)
        self.activations = OpticalActivations(config)
    
    def apply_nonlinearity(self, x: torch.Tensor, 
                          nonlinearity_type: NonlinearityType,
                          **kwargs) -> torch.Tensor:
        """Apply specified nonlinearity using optical implementation."""
        
        if nonlinearity_type == NonlinearityType.SOFTMAX:
            dim = kwargs.get('dim', -1)
            temperature = kwargs.get('temperature', 1.0)
            return self.softmax.forward(x, dim, temperature)
        
        elif nonlinearity_type == NonlinearityType.RELU:
            return self.activations.optical_relu(x)
        
        elif nonlinearity_type == NonlinearityType.GELU:
            return self.activations.optical_gelu(x)
        
        elif nonlinearity_type == NonlinearityType.TANH:
            return self.activations.optical_tanh(x)
        
        elif nonlinearity_type == NonlinearityType.SIGMOID:
            # Sigmoid as scaled/shifted tanh
            tanh_result = self.activations.optical_tanh(x * 0.5)
            return 0.5 * (tanh_result + 1.0)
        
        elif nonlinearity_type == NonlinearityType.SWISH:
            # Swish = x * sigmoid(x)
            sigmoid_x = self.apply_nonlinearity(x, NonlinearityType.SIGMOID)
            return x * sigmoid_x
        
        else:
            raise PhotonicComputeError(f"Unsupported nonlinearity: {nonlinearity_type}")


# Convenience functions
def optical_softmax(x: torch.Tensor, dim: int = -1, 
                   temperature: float = 1.0) -> torch.Tensor:
    """Convenience function for optical softmax."""
    softmax_op = OpticalSoftmax()
    return softmax_op.forward(x, dim, temperature)


def optical_relu(x: torch.Tensor) -> torch.Tensor:
    """Convenience function for optical ReLU."""
    activations = OpticalActivations()
    return activations.optical_relu(x)


def optical_gelu(x: torch.Tensor) -> torch.Tensor:
    """Convenience function for optical GELU."""
    activations = OpticalActivations()
    return activations.optical_gelu(x)


# Global optical nonlinearity kernel
_global_optical_nonlinearity = None

def get_global_optical_nonlinearity() -> OpticalNonlinearityKernel:
    """Get global optical nonlinearity kernel instance."""
    global _global_optical_nonlinearity
    if _global_optical_nonlinearity is None:
        config = OpticalNonlinearityConfig(
            n_wavelengths=get_config().photonic_wavelengths
        )
        _global_optical_nonlinearity = OpticalNonlinearityKernel(config)
    return _global_optical_nonlinearity