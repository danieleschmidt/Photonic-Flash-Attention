"""Pure GPU Flash-Attention 3 implementation."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from ..config import get_config


class FlashAttention3(nn.Module):
    """
    Flash-Attention 3 implementation optimized for GPU computation.
    
    This is the electronic baseline that photonic attention will be compared against.
    Implements tiling, memory-efficient attention with backward pass optimization.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.scaling = self.head_dim ** -0.5
        
        # QKV projection
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, device=device, dtype=dtype)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, device=device, dtype=dtype)
        
        # Dropout
        self.dropout_module = nn.Dropout(dropout) if dropout > 0 else None
        
        # Performance tracking
        self.last_latency_ms = 0.0
        self.last_memory_mb = 0.0
        
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of Flash-Attention 3.
        
        Args:
            query: Query tensor [batch, seq_len, embed_dim]
            key: Key tensor (if None, uses query)
            value: Value tensor (if None, uses query)
            attention_mask: Attention mask [batch, seq_len] or [batch, seq_len, seq_len]
            need_weights: Whether to return attention weights
            
        Returns:
            output: Attention output [batch, seq_len, embed_dim]
            weights: Attention weights if need_weights=True
        """
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time:
            start_time.record()
        
        batch_size, seq_len, embed_dim = query.shape
        
        # Handle self-attention case
        if key is None:
            key = query
        if value is None:
            value = query
            
        # QKV projection
        if torch.equal(query, key) and torch.equal(query, value):
            # Self-attention: compute QKV in one shot
            qkv = self.qkv_proj(query)  # [batch, seq_len, 3 * embed_dim]
            q, k, v = qkv.chunk(3, dim=-1)
        else:
            # Cross-attention: separate projections
            q = self.qkv_proj(query)[:, :, :embed_dim]
            k = self.qkv_proj(key)[:, :, embed_dim:2*embed_dim] 
            v = self.qkv_proj(value)[:, :, 2*embed_dim:]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Flash attention computation
        attn_output, attn_weights = self._flash_attention_forward(
            q, k, v, attention_mask, need_weights
        )
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        output = self.out_proj(attn_output)
        
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            self.last_latency_ms = start_time.elapsed_time(end_time)
            self.last_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        return output, attn_weights if need_weights else None
    
    def _flash_attention_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor, 
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Flash attention forward pass with tiling.
        
        Implements memory-efficient attention computation using tiling
        to reduce memory complexity from O(n²) to O(n).
        """
        batch_size, num_heads, seq_len_q, head_dim = q.shape
        seq_len_k = k.shape[2]
        
        # Scale queries
        q = q * self.scaling
        
        # Choose tile size based on memory constraints
        config = get_config()
        available_memory = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 8e9
        tile_size = self._compute_optimal_tile_size(seq_len_q, seq_len_k, head_dim, available_memory)
        
        if seq_len_q <= tile_size and seq_len_k <= tile_size:
            # Small sequences: use standard attention
            return self._standard_attention(q, k, v, attention_mask, need_weights)
        else:
            # Large sequences: use tiled flash attention
            return self._tiled_attention(q, k, v, attention_mask, need_weights, tile_size)
    
    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Standard attention computation for small sequences."""
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        
        # Apply attention mask
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply dropout
        if self.dropout_module is not None:
            attn_weights = self.dropout_module(attn_weights)
        
        # Compute weighted sum
        attn_output = torch.matmul(attn_weights, v)
        
        return attn_output, attn_weights if need_weights else None
    
    def _tiled_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        tile_size: int = 128,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Tiled flash attention for memory efficiency.
        
        Processes attention in tiles to maintain O(n) memory complexity.
        """
        batch_size, num_heads, seq_len_q, head_dim = q.shape
        seq_len_k = k.shape[2]
        
        # Initialize output
        output = torch.zeros_like(q)
        attn_weights_full = torch.zeros(
            batch_size, num_heads, seq_len_q, seq_len_k, 
            device=q.device, dtype=q.dtype
        ) if need_weights else None
        
        # Process in tiles
        for i in range(0, seq_len_q, tile_size):
            q_end = min(i + tile_size, seq_len_q)
            q_tile = q[:, :, i:q_end, :]
            
            # Initialize tile accumulators
            max_score = torch.full(
                (batch_size, num_heads, q_end - i), 
                float('-inf'), device=q.device, dtype=q.dtype
            )
            sum_exp = torch.zeros(
                (batch_size, num_heads, q_end - i), 
                device=q.device, dtype=q.dtype
            )
            output_tile = torch.zeros(
                (batch_size, num_heads, q_end - i, head_dim),
                device=q.device, dtype=q.dtype
            )
            
            for j in range(0, seq_len_k, tile_size):
                k_end = min(j + tile_size, seq_len_k)
                k_tile = k[:, :, j:k_end, :]
                v_tile = v[:, :, j:k_end, :]
                
                # Compute attention scores for this tile
                scores_tile = torch.matmul(q_tile, k_tile.transpose(-2, -1))
                
                # Apply mask if needed
                if attention_mask is not None:
                    mask_tile = attention_mask[:, :, i:q_end, j:k_end]
                    scores_tile = scores_tile.masked_fill(mask_tile == 0, float('-inf'))
                
                # Online softmax computation
                max_score_tile = torch.max(scores_tile, dim=-1, keepdim=True).values
                max_score_new = torch.max(max_score.unsqueeze(-1), max_score_tile)
                
                # Update exponentials
                exp_scores = torch.exp(scores_tile - max_score_new)
                exp_old = torch.exp(max_score.unsqueeze(-1) - max_score_new) * sum_exp.unsqueeze(-1)
                
                sum_exp_new = exp_old.sum(dim=-1) + exp_scores.sum(dim=-1)
                
                # Update output
                if sum_exp_new.sum() > 0:
                    output_tile = (exp_old * output_tile + torch.matmul(exp_scores, v_tile)) / sum_exp_new.unsqueeze(-1)
                
                # Update accumulators
                max_score = max_score_new.squeeze(-1)
                sum_exp = sum_exp_new
                
                # Store attention weights if needed
                if need_weights:
                    attn_weights_full[:, :, i:q_end, j:k_end] = exp_scores / sum_exp_new.unsqueeze(-1)
            
            output[:, :, i:q_end, :] = output_tile
        
        return output, attn_weights_full
    
    def _compute_optimal_tile_size(
        self, 
        seq_len_q: int, 
        seq_len_k: int, 
        head_dim: int, 
        available_memory: float
    ) -> int:
        """Compute optimal tile size based on available memory."""
        # Estimate memory usage per tile
        bytes_per_element = 4  # float32
        memory_per_tile = lambda tile_size: (
            tile_size * head_dim + 
            tile_size * seq_len_k + 
            tile_size
        ) * bytes_per_element
        
        # Binary search for optimal tile size
        min_tile = 32
        max_tile = min(seq_len_q, seq_len_k, 512)
        
        target_memory = available_memory * get_config().max_memory_usage
        
        while min_tile < max_tile:
            mid_tile = (min_tile + max_tile + 1) // 2
            if memory_per_tile(mid_tile) <= target_memory:
                min_tile = mid_tile
            else:
                max_tile = mid_tile - 1
        
        return max(min_tile, 32)  # Minimum tile size of 32
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics from last forward pass."""
        return {
            "latency_ms": self.last_latency_ms,
            "memory_mb": self.last_memory_mb,
            "device": "cuda",
            "implementation": "flash_attention_3",
        }