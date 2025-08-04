"""Input validation and sanitization utilities."""

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