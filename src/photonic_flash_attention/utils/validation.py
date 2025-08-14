"""
Comprehensive input/output validation and sanitization for photonic attention.

This module provides robust validation, sanitization, and security checks
for all data flowing through the photonic attention system.
"""

import torch
import numpy as np
from typing import Optional, Tuple, Union, List
from .exceptions import PhotonicConfigurationError, PhotonicComputationError


def validate_tensor_shape(
    tensor: torch.Tensor,
    expected_dims: int,
    expected_shape: Optional[Tuple[int, ...]] = None,
    name: str = "tensor"
) -> None:
    """
    Validate tensor shape and dimensions.
    
    Args:
        tensor: Input tensor to validate
        expected_dims: Expected number of dimensions
        expected_shape: Expected shape (None means any size for that dimension)
        name: Name of the tensor for error messages
        
    Raises:
        PhotonicComputationError: If validation fails
    """
    if not isinstance(tensor, torch.Tensor):
        raise PhotonicComputationError(f"{name} must be a torch.Tensor, got {type(tensor)}")
    
    if tensor.dim() != expected_dims:
        raise PhotonicComputationError(
            f"{name} must have {expected_dims} dimensions, got {tensor.dim()}"
        )
    
    if expected_shape is not None:
        actual_shape = tensor.shape
        if len(actual_shape) != len(expected_shape):
            raise PhotonicComputationError(
                f"{name} shape mismatch: expected {len(expected_shape)} dims, got {len(actual_shape)}"
            )
        
        for i, (actual, expected) in enumerate(zip(actual_shape, expected_shape)):
            if expected is not None and actual != expected:
                raise PhotonicComputationError(
                    f"{name} dimension {i} mismatch: expected {expected}, got {actual}"
                )


def validate_attention_inputs(
    query: torch.Tensor,
    key: Optional[torch.Tensor] = None,
    value: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> None:
    """
    Validate attention input tensors.
    
    Args:
        query: Query tensor [batch, seq_len, embed_dim]
        key: Key tensor (optional)
        value: Value tensor (optional)
        attention_mask: Attention mask (optional)
        
    Raises:
        PhotonicComputationError: If validation fails
    """
    # Validate query tensor
    validate_tensor_shape(query, 3, name="query")
    batch_size, seq_len_q, embed_dim = query.shape
    
    if batch_size <= 0 or seq_len_q <= 0 or embed_dim <= 0:
        raise PhotonicComputationError(f"Invalid query shape: {query.shape}")
    
    # Validate key tensor
    if key is not None:
        validate_tensor_shape(key, 3, name="key")
        key_batch, seq_len_k, key_embed_dim = key.shape
        
        if key_batch != batch_size:
            raise PhotonicComputationError(f"Key batch size {key_batch} doesn't match query {batch_size}")
        if key_embed_dim != embed_dim:
            raise PhotonicComputationError(f"Key embed_dim {key_embed_dim} doesn't match query {embed_dim}")
    else:
        seq_len_k = seq_len_q
    
    # Validate value tensor
    if value is not None:
        validate_tensor_shape(value, 3, name="value")
        value_batch, value_seq_len, value_embed_dim = value.shape
        
        if value_batch != batch_size:
            raise PhotonicComputationError(f"Value batch size {value_batch} doesn't match query {batch_size}")
        if value_seq_len != seq_len_k:
            raise PhotonicComputationError(f"Value seq_len {value_seq_len} doesn't match key {seq_len_k}")
        if value_embed_dim != embed_dim:
            raise PhotonicComputationError(f"Value embed_dim {value_embed_dim} doesn't match query {embed_dim}")
    
    # Validate attention mask
    if attention_mask is not None:
        if attention_mask.dim() == 2:
            # [batch, seq_len] padding mask
            mask_batch, mask_seq_len = attention_mask.shape
            if mask_batch != batch_size:
                raise PhotonicComputationError(f"Mask batch size {mask_batch} doesn't match query {batch_size}")
            if mask_seq_len != seq_len_k:
                raise PhotonicComputationError(f"Mask seq_len {mask_seq_len} doesn't match key {seq_len_k}")
        elif attention_mask.dim() == 3:
            # [batch, seq_len_q, seq_len_k] attention mask
            mask_batch, mask_seq_q, mask_seq_k = attention_mask.shape
            if mask_batch != batch_size:
                raise PhotonicComputationError(f"Mask batch size {mask_batch} doesn't match query {batch_size}")
            if mask_seq_q != seq_len_q:
                raise PhotonicComputationError(f"Mask seq_len_q {mask_seq_q} doesn't match query {seq_len_q}")
            if mask_seq_k != seq_len_k:
                raise PhotonicComputationError(f"Mask seq_len_k {mask_seq_k} doesn't match key {seq_len_k}")
        elif attention_mask.dim() == 4:
            # [batch, num_heads, seq_len_q, seq_len_k] full attention mask
            mask_batch, mask_heads, mask_seq_q, mask_seq_k = attention_mask.shape
            if mask_batch != batch_size:
                raise PhotonicComputationError(f"Mask batch size {mask_batch} doesn't match query {batch_size}")
            if mask_seq_q != seq_len_q:
                raise PhotonicComputationError(f"Mask seq_len_q {mask_seq_q} doesn't match query {seq_len_q}")
            if mask_seq_k != seq_len_k:
                raise PhotonicComputationError(f"Mask seq_len_k {mask_seq_k} doesn't match key {seq_len_k}")
        else:
            raise PhotonicComputationError(f"Attention mask must have 2, 3, or 4 dimensions, got {attention_mask.dim()}")


def validate_photonic_config(config: dict) -> None:
    """
    Validate photonic hardware configuration.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        PhotonicConfigurationError: If configuration is invalid
    """
    required_fields = ['wavelengths', 'modulator_resolution', 'max_optical_power']
    
    for field in required_fields:
        if field not in config:
            raise PhotonicConfigurationError(f"Missing required config field: {field}")
    
    # Validate wavelengths
    wavelengths = config['wavelengths']
    if not isinstance(wavelengths, int) or wavelengths <= 0:
        raise PhotonicConfigurationError(f"wavelengths must be positive integer, got {wavelengths}")
    if wavelengths > 256:  # Reasonable upper limit
        raise PhotonicConfigurationError(f"wavelengths too high: {wavelengths} > 256")
    
    # Validate modulator resolution
    resolution = config['modulator_resolution']
    if not isinstance(resolution, int) or resolution < 1 or resolution > 16:
        raise PhotonicConfigurationError(f"modulator_resolution must be 1-16 bits, got {resolution}")
    
    # Validate optical power
    power = config['max_optical_power']
    if not isinstance(power, (int, float)) or power <= 0:
        raise PhotonicConfigurationError(f"max_optical_power must be positive, got {power}")
    if power > 0.1:  # 100mW safety limit
        raise PhotonicConfigurationError(f"max_optical_power too high: {power}W > 0.1W")
    
    # Validate optional fields
    if 'temperature_limit' in config:
        temp_limit = config['temperature_limit']
        if not isinstance(temp_limit, (int, float)) or temp_limit < 0 or temp_limit > 150:
            raise PhotonicConfigurationError(f"temperature_limit must be 0-150°C, got {temp_limit}")
    
    if 'wavelength_range' in config:
        wl_range = config['wavelength_range']
        if not isinstance(wl_range, (list, tuple)) or len(wl_range) != 2:
            raise PhotonicConfigurationError(f"wavelength_range must be [min, max], got {wl_range}")
        if wl_range[0] >= wl_range[1]:
            raise PhotonicConfigurationError(f"Invalid wavelength range: {wl_range[0]} >= {wl_range[1]}")


def validate_sequence_length(seq_len: int, max_len: int = 16384) -> None:
    """
    Validate sequence length limits.
    
    Args:
        seq_len: Sequence length to validate
        max_len: Maximum allowed sequence length
        
    Raises:
        PhotonicComputationError: If sequence is too long
    """
    if seq_len <= 0:
        raise PhotonicComputationError(f"Sequence length must be positive, got {seq_len}")
    
    if seq_len > max_len:
        raise PhotonicComputationError(f"Sequence length {seq_len} exceeds maximum {max_len}")


def validate_batch_size(batch_size: int, max_batch: int = 128) -> None:
    """
    Validate batch size limits.
    
    Args:
        batch_size: Batch size to validate
        max_batch: Maximum allowed batch size
        
    Raises:
        PhotonicComputationError: If batch is too large
    """
    if batch_size <= 0:
        raise PhotonicComputationError(f"Batch size must be positive, got {batch_size}")
    
    if batch_size > max_batch:
        raise PhotonicComputationError(f"Batch size {batch_size} exceeds maximum {max_batch}")


def validate_memory_usage(tensors: List[torch.Tensor], max_memory_gb: float = 16.0) -> None:
    """
    Validate memory usage of tensors.
    
    Args:
        tensors: List of tensors to check
        max_memory_gb: Maximum allowed memory in GB
        
    Raises:
        PhotonicComputationError: If memory usage is too high
    """
    total_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
    total_gb = total_bytes / (1024 ** 3)
    
    if total_gb > max_memory_gb:
        raise PhotonicComputationError(
            f"Memory usage {total_gb:.2f}GB exceeds limit {max_memory_gb}GB"
        )


def validate_optical_tensor(tensor: torch.Tensor, name: str = "tensor") -> None:
    """
    Validate tensor for optical computation.
    
    Args:
        tensor: Tensor to validate
        name: Name for error messages
        
    Raises:
        PhotonicComputationError: If validation fails
    """
    check_tensor_finite(tensor, name)
    
    # Check data type
    if tensor.dtype not in (torch.float16, torch.float32, torch.complex64, torch.complex128):
        raise PhotonicComputationError(
            f"{name} has unsupported dtype {tensor.dtype} for optical computation"
        )
    
    # Check tensor size limits
    max_elements = 100_000_000  # 100M elements
    if tensor.numel() > max_elements:
        raise PhotonicComputationError(
            f"{name} too large: {tensor.numel()} elements > {max_elements}"
        )


def validate_matrix_dimensions(a: torch.Tensor, b: torch.Tensor) -> None:
    """
    Validate matrix multiplication dimensions.
    
    Args:
        a: First matrix
        b: Second matrix
        
    Raises:
        PhotonicComputationError: If dimensions are incompatible
    """
    if a.dim() < 2 or b.dim() < 2:
        raise PhotonicComputationError(
            f"Matrices must be at least 2D: got {a.dim()}D and {b.dim()}D"
        )
    
    # Check inner dimensions for matrix multiplication
    a_inner = a.shape[-1]
    b_inner = b.shape[-2]
    
    if a_inner != b_inner:
        raise PhotonicComputationError(
            f"Matrix inner dimensions don't match: {a_inner} != {b_inner}"
        )


def check_tensor_finite(tensor: torch.Tensor, name: str = "tensor") -> None:
    """
    Check that tensor contains only finite values.
    
    Args:
        tensor: Tensor to check
        name: Name for error messages
        
    Raises:
        PhotonicComputationError: If tensor contains NaN or Inf
    """
    if torch.isnan(tensor).any():
        raise PhotonicComputationError(f"{name} contains NaN values")
    
    if torch.isinf(tensor).any():
        raise PhotonicComputationError(f"{name} contains infinite values")


def check_tensor_range(
    tensor: torch.Tensor,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    name: str = "tensor"
) -> None:
    """
    Check that tensor values are within specified range.
    
    Args:
        tensor: Tensor to check
        min_val: Minimum allowed value (optional)
        max_val: Maximum allowed value (optional)
        name: Name for error messages
        
    Raises:
        PhotonicComputationError: If values are out of range
    """
    if min_val is not None:
        if tensor.min().item() < min_val:
            raise PhotonicComputationError(f"{name} contains values below {min_val}")
    
    if max_val is not None:
        if tensor.max().item() > max_val:
            raise PhotonicComputationError(f"{name} contains values above {max_val}")


def sanitize_tensor_input(tensor: torch.Tensor, clip_range: Optional[Tuple[float, float]] = None) -> torch.Tensor:
    """
    Sanitize tensor input by clipping and checking for issues.
    
    Args:
        tensor: Input tensor to sanitize
        clip_range: Optional (min, max) range to clip values
        
    Returns:
        Sanitized tensor
        
    Raises:
        PhotonicComputationError: If tensor is invalid
    """
    # Check for NaN/Inf
    if torch.isnan(tensor).any():
        raise PhotonicComputationError("Input tensor contains NaN values")
    
    if torch.isinf(tensor).any():
        # Replace infinite values with large but finite values
        tensor = torch.where(torch.isinf(tensor), torch.sign(tensor) * 1e6, tensor)
    
    # Clip values if range specified
    if clip_range is not None:
        tensor = torch.clamp(tensor, clip_range[0], clip_range[1])
    
    return tensor


def validate_model_structure(model: torch.nn.Module) -> None:
    """
    Validate model structure for photonic conversion.
    
    Args:
        model: PyTorch model to validate
        
    Raises:
        PhotonicComputationError: If model structure is invalid
    """
    if not isinstance(model, torch.nn.Module):
        raise PhotonicComputationError("Model must be a PyTorch nn.Module")
    
    # Check if model has parameters
    total_params = sum(p.numel() for p in model.parameters())
    if total_params == 0:
        raise PhotonicComputationError("Model has no trainable parameters")
    
    # Check model can be set to different modes
    try:
        model.eval()
        model.train()
    except Exception as e:
        raise PhotonicComputationError(f"Model cannot switch between train/eval modes: {e}")
    
    # Check for attention patterns
    attention_found = False
    for name, module in model.named_modules():
        module_name = name.lower()
        class_name = type(module).__name__.lower()
        if any(pattern in module_name or pattern in class_name 
               for pattern in ['attention', 'attn', 'multihead']):
            attention_found = True
            break
    
    if not attention_found:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("No attention layers detected in model - photonic conversion may have no effect")


def validate_device_compatibility(tensor: torch.Tensor, target_device: str) -> None:
    """
    Validate tensor device compatibility for photonic operations.
    
    Args:
        tensor: Tensor to check
        target_device: Target device ('cuda', 'cpu', 'photonic')
        
    Raises:
        PhotonicComputationError: If device is incompatible
    """
    current_device = str(tensor.device)
    
    # Check CUDA compatibility for photonic operations
    if target_device == 'photonic' and not current_device.startswith('cuda'):
        raise PhotonicComputationError(
            f"Photonic operations require CUDA tensors, got {current_device}"
        )
    
    # Check tensor is contiguous
    if not tensor.is_contiguous():
        raise PhotonicComputationError("Tensor must be contiguous for photonic operations")


def validate_numerical_stability(
    tensor: torch.Tensor, 
    operation: str,
    epsilon: float = 1e-8
) -> None:
    """
    Validate numerical stability for photonic operations.
    
    Args:
        tensor: Tensor to check
        operation: Operation name for context
        epsilon: Minimum value for stability checks
        
    Raises:
        PhotonicComputationError: If numerical issues detected
    """
    # Check for very small values that could cause instability
    if operation in ['division', 'softmax', 'log']:
        if (tensor.abs() < epsilon).any():
            raise PhotonicComputationError(
                f"Tensor contains values too small for {operation}: min={tensor.abs().min():.2e}"
            )
    
    # Check gradient flow for training
    if tensor.requires_grad and operation in ['forward_pass']:
        if tensor.grad is not None and torch.isnan(tensor.grad).any():
            raise PhotonicComputationError(f"Gradient contains NaN values in {operation}")


def validate_optical_power_budget(
    tensor: torch.Tensor, 
    max_power: float = 10e-3,  # 10mW
    name: str = "tensor"
) -> None:
    """
    Validate optical power budget for photonic operations.
    
    Args:
        tensor: Tensor representing optical signals
        max_power: Maximum allowed optical power in Watts
        name: Tensor name for error messages
        
    Raises:
        PhotonicComputationError: If power budget exceeded
    """
    # Estimate optical power as sum of absolute values
    power_estimate = tensor.abs().sum().item() / tensor.numel()
    
    if power_estimate > max_power:
        raise PhotonicComputationError(
            f"{name} exceeds optical power budget: {power_estimate:.2e}W > {max_power:.2e}W"
        )


def validate_wavelength_allocation(
    required_channels: int, 
    available_channels: int,
    operation: str = "optical_operation"
) -> None:
    """
    Validate wavelength channel allocation.
    
    Args:
        required_channels: Number of channels needed
        available_channels: Number of channels available
        operation: Operation name for context
        
    Raises:
        PhotonicComputationError: If insufficient channels
    """
    if required_channels > available_channels:
        raise PhotonicComputationError(
            f"{operation} requires {required_channels} wavelength channels, "
            f"only {available_channels} available"
        )
    
    if required_channels <= 0:
        raise PhotonicComputationError(f"Invalid channel requirement: {required_channels}")


def validate_thermal_conditions(
    temperature: float,
    max_temp: float = 85.0,
    min_temp: float = -40.0,
    component: str = "photonic_device"
) -> None:
    """
    Validate thermal operating conditions.
    
    Args:
        temperature: Current temperature in Celsius
        max_temp: Maximum safe temperature
        min_temp: Minimum safe temperature
        component: Component name for error messages
        
    Raises:
        PhotonicComputationError: If temperature out of range
    """
    if temperature > max_temp:
        raise PhotonicComputationError(
            f"{component} temperature {temperature:.1f}°C exceeds maximum {max_temp:.1f}°C"
        )
    
    if temperature < min_temp:
        raise PhotonicComputationError(
            f"{component} temperature {temperature:.1f}°C below minimum {min_temp:.1f}°C"
        )


def sanitize_config_dict(config: dict, allowed_keys: List[str]) -> dict:
    """
    Sanitize configuration dictionary by removing unknown keys.
    
    Args:
        config: Configuration dictionary
        allowed_keys: List of allowed configuration keys
        
    Returns:
        Sanitized configuration dictionary
    """
    sanitized = {}
    
    for key, value in config.items():
        if key in allowed_keys:
            sanitized[key] = value
        else:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Ignoring unknown configuration key: {key}")
    
    return sanitized


def validate_attention_pattern(
    attention_weights: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    tolerance: float = 1e-3
) -> None:
    """
    Validate attention weight patterns for correctness.
    
    Args:
        attention_weights: Attention weight matrix
        mask: Optional attention mask
        tolerance: Tolerance for sum-to-one check
        
    Raises:
        PhotonicComputationError: If attention pattern is invalid
    """
    # Check dimensions
    if attention_weights.dim() < 2:
        raise PhotonicComputationError(
            f"Attention weights must be at least 2D, got {attention_weights.dim()}D"
        )
    
    # Check for negative weights
    if (attention_weights < 0).any():
        raise PhotonicComputationError("Attention weights cannot be negative")
    
    # Check sum-to-one property (last dimension)
    if mask is not None:
        # Apply mask before checking sums
        masked_weights = attention_weights.masked_fill(mask == 0, 0.0)
        weight_sums = masked_weights.sum(dim=-1)
    else:
        weight_sums = attention_weights.sum(dim=-1)
    
    # Allow for small numerical errors
    target_sum = 1.0
    if not torch.allclose(weight_sums, torch.ones_like(weight_sums) * target_sum, atol=tolerance):
        max_error = (weight_sums - target_sum).abs().max().item()
        raise PhotonicComputationError(
            f"Attention weights don't sum to 1.0, max error: {max_error:.6f}"
        )


def validate_quantum_coherence(
    optical_state: torch.Tensor,
    coherence_threshold: float = 0.8,
    name: str = "optical_state"
) -> None:
    """
    Validate quantum coherence properties of optical states.
    
    Args:
        optical_state: Complex-valued optical state tensor
        coherence_threshold: Minimum coherence required (0-1)
        name: State name for error messages
        
    Raises:
        PhotonicComputationError: If coherence is insufficient
    """
    if not torch.is_complex(optical_state):
        raise PhotonicComputationError(f"{name} must be complex-valued for coherence validation")
    
    # Calculate coherence as normalized correlation
    amplitude = optical_state.abs()
    phase = optical_state.angle()
    
    # Measure phase coherence across spatial/temporal dimensions
    if optical_state.numel() > 1:
        phase_variance = phase.var().item()
        coherence = torch.exp(-phase_variance).item()
        
        if coherence < coherence_threshold:
            raise PhotonicComputationError(
                f"{name} coherence {coherence:.3f} below threshold {coherence_threshold:.3f}"
            )


def validate_modulation_depth(
    modulated_signal: torch.Tensor,
    min_depth: float = 0.1,
    max_depth: float = 1.0,
    name: str = "modulated_signal"
) -> None:
    """
    Validate optical modulation depth.
    
    Args:
        modulated_signal: Modulated optical signal
        min_depth: Minimum acceptable modulation depth
        max_depth: Maximum acceptable modulation depth  
        name: Signal name for error messages
        
    Raises:
        PhotonicComputationError: If modulation depth is invalid
    """
    # Calculate modulation depth as (max - min) / (max + min)
    signal_max = modulated_signal.max().item()
    signal_min = modulated_signal.min().item()
    
    if signal_max == signal_min:
        modulation_depth = 0.0
    else:
        modulation_depth = (signal_max - signal_min) / (signal_max + signal_min)
    
    if modulation_depth < min_depth:
        raise PhotonicComputationError(
            f"{name} modulation depth {modulation_depth:.3f} below minimum {min_depth:.3f}"
        )
    
    if modulation_depth > max_depth:
        raise PhotonicComputationError(
            f"{name} modulation depth {modulation_depth:.3f} above maximum {max_depth:.3f}"
        )