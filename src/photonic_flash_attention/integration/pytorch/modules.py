"""PyTorch modules for photonic attention."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from ...config import get_config
from ...core.flash_attention_3 import FlashAttention3
from ...photonic.hardware.detection import is_photonic_available


class PhotonicFlashAttention(nn.Module):
    """
    Drop-in replacement for standard attention with photonic acceleration.
    
    Automatically switches between photonic and GPU computation based on:
    - Sequence length thresholds
    - Hardware availability  
    - Performance characteristics
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        photonic_threshold: Optional[int] = None,
        device: Union[str, torch.device] = 'auto',
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Configuration
        config = get_config()
        self.photonic_threshold = photonic_threshold or config.photonic_threshold
        self.auto_device_selection = device == 'auto' and config.auto_device_selection
        
        # Initialize GPU fallback (always available)
        self.gpu_attention = FlashAttention3(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            device=device if device != 'auto' else None,
            dtype=dtype,
        )
        
        # Initialize photonic attention if available
        self.photonic_attention = None
        self.photonic_available = is_photonic_available()
        
        if self.photonic_available:
            from ...core.photonic_attention import PhotonicAttention
            self.photonic_attention = PhotonicAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                bias=bias,
                device=device,
                dtype=dtype,
            )
        
        # Performance tracking
        self.last_device_used = "gpu"
        self.last_latency_ms = 0.0
        self.last_energy_mj = 0.0
        self._performance_history = []
        
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with automatic device selection.
        
        Args:
            query: Query tensor [batch, seq_len, embed_dim]
            key: Key tensor (optional, defaults to query)
            value: Value tensor (optional, defaults to query)
            attention_mask: Attention mask (optional)
            need_weights: Whether to return attention weights
            
        Returns:
            output: Attention output [batch, seq_len, embed_dim]
            weights: Attention weights (if need_weights=True)
        """
        batch_size, seq_len, _ = query.shape
        
        # Determine optimal device
        use_photonic = self._should_use_photonic(batch_size, seq_len)
        
        if use_photonic and self.photonic_attention is not None:
            # Use photonic computation
            output, weights = self._forward_photonic(query, key, value, attention_mask, need_weights)
            self.last_device_used = "photonic"
        else:
            # Use GPU computation
            output, weights = self._forward_gpu(query, key, value, attention_mask, need_weights)
            self.last_device_used = "gpu"
        
        # Update performance history
        self._update_performance_stats()
        
        return (output, weights) if need_weights else output
    
    def _should_use_photonic(self, batch_size: int, seq_len: int) -> bool:
        """Determine whether to use photonic computation."""
        if not self.photonic_available or self.photonic_attention is None:
            return False
        
        if not self.auto_device_selection:
            return True  # Use photonic if explicitly requested
        
        # Use photonic for longer sequences
        if seq_len >= self.photonic_threshold:
            return True
        
        # Use performance history for smarter decisions
        if len(self._performance_history) > 10:
            recent_photonic = [h for h in self._performance_history[-10:] if h['device'] == 'photonic']
            recent_gpu = [h for h in self._performance_history[-10:] if h['device'] == 'gpu']
            
            if recent_photonic and recent_gpu:
                avg_photonic_latency = sum(h['latency_ms'] for h in recent_photonic) / len(recent_photonic)
                avg_gpu_latency = sum(h['latency_ms'] for h in recent_gpu) / len(recent_gpu)
                
                # Use photonic if it's faster for similar workloads
                if avg_photonic_latency < avg_gpu_latency * 0.9:  # 10% margin
                    return True
        
        return False
    
    def _forward_gpu(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        need_weights: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass using GPU computation."""
        return self.gpu_attention(query, key, value, attention_mask, need_weights)
    
    def _forward_photonic(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        need_weights: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass using photonic computation."""
        return self.photonic_attention(query, key, value, attention_mask, need_weights)
    
    def _update_performance_stats(self) -> None:
        """Update performance statistics."""
        if self.last_device_used == "photonic" and self.photonic_attention:
            stats = self.photonic_attention.get_performance_stats()
        else:
            stats = self.gpu_attention.get_performance_stats()
        
        self.last_latency_ms = stats.get('latency_ms', 0.0)
        self.last_energy_mj = stats.get('energy_mj', 0.0)
        
        # Add to history
        self._performance_history.append({
            'device': self.last_device_used,
            'latency_ms': self.last_latency_ms,
            'energy_mj': self.last_energy_mj,
            'timestamp': torch.cuda.Event(enable_timing=True).query() if torch.cuda.is_available() else 0,
        })
        
        # Keep history bounded
        if len(self._performance_history) > 100:
            self._performance_history = self._performance_history[-100:]
    
    def get_performance_stats(self) -> dict:
        """Get comprehensive performance statistics."""
        stats = {
            'last_device_used': self.last_device_used,
            'last_latency_ms': self.last_latency_ms,
            'last_energy_mj': self.last_energy_mj,
            'photonic_available': self.photonic_available,
            'photonic_threshold': self.photonic_threshold,
        }
        
        if self._performance_history:
            photonic_calls = [h for h in self._performance_history if h['device'] == 'photonic']
            gpu_calls = [h for h in self._performance_history if h['device'] == 'gpu']
            
            stats.update({
                'total_calls': len(self._performance_history),
                'photonic_calls': len(photonic_calls),
                'gpu_calls': len(gpu_calls),
                'photonic_usage_ratio': len(photonic_calls) / len(self._performance_history),
            })
            
            if photonic_calls:
                stats['avg_photonic_latency_ms'] = sum(h['latency_ms'] for h in photonic_calls) / len(photonic_calls)
                stats['avg_photonic_energy_mj'] = sum(h['energy_mj'] for h in photonic_calls) / len(photonic_calls)
            
            if gpu_calls:
                stats['avg_gpu_latency_ms'] = sum(h['latency_ms'] for h in gpu_calls) / len(gpu_calls)
                stats['avg_gpu_energy_mj'] = sum(h['energy_mj'] for h in gpu_calls) / len(gpu_calls)
        
        return stats
    
    def set_photonic_threshold(self, threshold: int) -> None:
        """Update the photonic threshold."""
        self.photonic_threshold = threshold
    
    def enable_photonic(self, enabled: bool = True) -> None:
        """Enable or disable photonic computation."""
        if enabled and not self.photonic_available:
            print("Warning: Photonic hardware not available")
        self.auto_device_selection = enabled
    
    def reset_performance_history(self) -> None:
        """Reset performance tracking history."""
        self._performance_history.clear()


class PhotonicMultiHeadAttention(PhotonicFlashAttention):
    """
    Alias for PhotonicFlashAttention with MultiHeadAttention-compatible interface.
    
    Provides compatibility with torch.nn.MultiHeadAttention API.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = False,
        photonic_threshold: Optional[int] = None,
        device: Union[str, torch.device] = 'auto',
        dtype: Optional[torch.dtype] = None,
    ):
        # Handle additional MultiHeadAttention parameters
        if add_bias_kv or add_zero_attn:
            raise NotImplementedError("add_bias_kv and add_zero_attn not yet supported")
        
        if kdim is not None or vdim is not None:
            raise NotImplementedError("Different key/value dimensions not yet supported")
        
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            photonic_threshold=photonic_threshold,
            device=device,
            dtype=dtype,
        )
        
        self.batch_first = batch_first
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with MultiHeadAttention-compatible interface.
        
        Args:
            query: Query tensor
            key: Key tensor  
            value: Value tensor
            key_padding_mask: Key padding mask
            need_weights: Whether to return attention weights
            attn_mask: Attention mask
            average_attn_weights: Whether to average attention weights across heads
            
        Returns:
            output: Attention output
            weights: Attention weights (if need_weights=True)
        """
        # Handle batch_first format
        if not self.batch_first:
            # Convert from (seq_len, batch, embed_dim) to (batch, seq_len, embed_dim)
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        # Combine masks
        attention_mask = attn_mask
        if key_padding_mask is not None:
            if attention_mask is not None:
                attention_mask = attention_mask + key_padding_mask.unsqueeze(1)
            else:
                attention_mask = key_padding_mask.unsqueeze(1)
        
        # Call parent forward
        result = super().forward(query, key, value, attention_mask, need_weights)
        
        if need_weights:
            output, weights = result
            
            # Average weights across heads if requested
            if weights is not None and average_attn_weights:
                weights = weights.mean(dim=1)  # Average over head dimension
            
            # Convert back to original format
            if not self.batch_first:
                output = output.transpose(0, 1)
            
            return output, weights
        else:
            output = result
            if not self.batch_first:
                output = output.transpose(0, 1)
            return output, None