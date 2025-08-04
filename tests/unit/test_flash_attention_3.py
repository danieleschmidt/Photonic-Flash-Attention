"""Unit tests for Flash Attention 3 implementation."""

import pytest
import torch
import torch.nn.functional as F
import numpy as np

from photonic_flash_attention.core.flash_attention_3 import FlashAttention3


class TestFlashAttention3:
    """Test the GPU Flash Attention 3 implementation."""
    
    def test_initialization(self, attention_config, device, dtype):
        """Test attention module initialization."""
        batch_size, seq_len, embed_dim, num_heads = attention_config
        
        attention = FlashAttention3(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.1,
            bias=True,
            device=device,
            dtype=dtype,
        )
        
        assert attention.embed_dim == embed_dim
        assert attention.num_heads == num_heads
        assert attention.head_dim == embed_dim // num_heads
        assert attention.dropout == 0.1
        
        # Check parameter shapes
        assert attention.qkv_proj.weight.shape == (3 * embed_dim, embed_dim)
        assert attention.out_proj.weight.shape == (embed_dim, embed_dim)
    
    def test_invalid_initialization(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(AssertionError):
            FlashAttention3(embed_dim=64, num_heads=5)  # Not divisible
        
        with pytest.raises(ValueError):
            FlashAttention3(embed_dim=-64, num_heads=8)  # Negative embed_dim
    
    def test_forward_pass(self, gpu_attention, sample_tensors, assert_tensors_close):
        """Test basic forward pass."""
        query, key, value, attention_mask = sample_tensors
        
        # Forward pass without attention weights
        output = gpu_attention(query, key, value, attention_mask, need_weights=False)
        
        assert output.shape == query.shape
        assert output.dtype == query.dtype
        assert output.device == query.device
        
        # Check that output is not all zeros
        assert not torch.allclose(output, torch.zeros_like(output))
        
        # Check for NaN or Inf
        assert torch.isfinite(output).all()
    
    def test_forward_with_weights(self, gpu_attention, sample_tensors):
        """Test forward pass with attention weights."""
        query, key, value, attention_mask = sample_tensors
        
        output, weights = gpu_attention(query, key, value, attention_mask, need_weights=True)
        
        assert output.shape == query.shape
        assert weights is not None
        
        batch_size, seq_len = query.shape[:2]
        expected_weight_shape = (batch_size, gpu_attention.num_heads, seq_len, seq_len)
        assert weights.shape == expected_weight_shape
        
        # Attention weights should sum to approximately 1
        weight_sums = weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-3)
        
        # Attention weights should be non-negative
        assert (weights >= 0).all()
    
    def test_self_attention(self, gpu_attention, sample_tensors):
        """Test self-attention (query=key=value)."""
        query, _, _, attention_mask = sample_tensors
        
        # Self-attention: pass only query
        output = gpu_attention(query, attention_mask=attention_mask, need_weights=False)
        
        assert output.shape == query.shape
        assert torch.isfinite(output).all()
    
    def test_cross_attention(self, gpu_attention, sample_tensors):
        """Test cross-attention with different key/value."""
        query, key, value, attention_mask = sample_tensors
        
        # Modify key and value to be different from query
        key = key + 0.1
        value = value + 0.2
        
        output = gpu_attention(query, key, value, attention_mask, need_weights=False)
        
        assert output.shape == query.shape
        assert torch.isfinite(output).all()
    
    def test_attention_mask(self, gpu_attention, sample_tensors):
        """Test attention mask functionality."""
        query, key, value, attention_mask = sample_tensors
        
        # Test with mask
        output_with_mask = gpu_attention(query, key, value, attention_mask, need_weights=False)
        
        # Test without mask
        output_without_mask = gpu_attention(query, key, value, None, need_weights=False)
        
        # Outputs should be different when mask is applied
        assert not torch.allclose(output_with_mask, output_without_mask, atol=1e-3)
    
    def test_different_sequence_lengths(self, attention_config, device, dtype):
        """Test with different sequence lengths for key/value."""
        batch_size, seq_len_q, embed_dim, num_heads = attention_config
        seq_len_kv = seq_len_q // 2 + 10  # Different length
        
        attention = FlashAttention3(
            embed_dim=embed_dim,
            num_heads=num_heads,
            device=device,
            dtype=dtype,
        )
        
        query = torch.randn(batch_size, seq_len_q, embed_dim, dtype=dtype, device=device)
        key = torch.randn(batch_size, seq_len_kv, embed_dim, dtype=dtype, device=device)
        value = torch.randn(batch_size, seq_len_kv, embed_dim, dtype=dtype, device=device)
        
        output = attention(query, key, value, need_weights=False)
        
        assert output.shape == (batch_size, seq_len_q, embed_dim)
    
    def test_gradient_computation(self, gpu_attention, sample_tensors):
        """Test that gradients are computed correctly."""
        query, key, value, attention_mask = sample_tensors
        
        # Enable gradients
        query.requires_grad_(True)
        key.requires_grad_(True)
        value.requires_grad_(True)
        
        output = gpu_attention(query, key, value, attention_mask, need_weights=False)
        
        # Compute loss and backward pass
        loss = output.sum()
        loss.backward()
        
        # Check that gradients were computed
        assert query.grad is not None
        assert key.grad is not None
        assert value.grad is not None
        
        # Check gradient shapes
        assert query.grad.shape == query.shape
        assert key.grad.shape == key.shape
        assert value.grad.shape == value.shape
    
    def test_dropout(self, attention_config, device, dtype, sample_tensors):
        """Test dropout functionality."""
        batch_size, seq_len, embed_dim, num_heads = attention_config
        
        # Create attention module with dropout
        attention = FlashAttention3(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.5,
            device=device,
            dtype=dtype,
        )
        
        query, key, value, attention_mask = sample_tensors
        
        # Training mode (dropout active)
        attention.train()
        output1 = attention(query, key, value, attention_mask, need_weights=False)
        output2 = attention(query, key, value, attention_mask, need_weights=False)
        
        # Outputs should be different due to dropout
        assert not torch.allclose(output1, output2, atol=1e-3)
        
        # Eval mode (dropout inactive)
        attention.eval()
        output3 = attention(query, key, value, attention_mask, need_weights=False)
        output4 = attention(query, key, value, attention_mask, need_weights=False)
        
        # Outputs should be identical in eval mode
        assert torch.allclose(output3, output4, atol=1e-6)
    
    def test_memory_efficiency(self, device):
        """Test memory efficiency for large sequences."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for memory test")
        
        # Large sequence that would cause OOM with naive attention
        batch_size, seq_len, embed_dim, num_heads = 1, 2048, 512, 8
        
        attention = FlashAttention3(
            embed_dim=embed_dim,
            num_heads=num_heads,
            device=device,
        )
        
        query = torch.randn(batch_size, seq_len, embed_dim, device=device)
        
        # Should not run out of memory
        output = attention(query, need_weights=False)
        assert output.shape == query.shape
    
    def test_performance_stats(self, gpu_attention, sample_tensors):
        """Test performance statistics collection."""
        query, key, value, attention_mask = sample_tensors
        
        # Perform forward pass
        gpu_attention(query, key, value, attention_mask, need_weights=False)
        
        # Check performance stats
        stats = gpu_attention.get_performance_stats()
        
        assert 'latency_ms' in stats
        assert 'memory_mb' in stats
        assert 'device' in stats
        assert 'implementation' in stats
        
        assert stats['device'] == 'cuda'
        assert stats['implementation'] == 'flash_attention_3'
        assert stats['latency_ms'] >= 0
    
    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_batch_sizes(self, batch_size, device, dtype):
        """Test different batch sizes."""
        seq_len, embed_dim, num_heads = 128, 512, 8
        
        attention = FlashAttention3(
            embed_dim=embed_dim,
            num_heads=num_heads,
            device=device,
            dtype=dtype,
        )
        
        query = torch.randn(batch_size, seq_len, embed_dim, dtype=dtype, device=device)
        
        output = attention(query, need_weights=False)
        assert output.shape == (batch_size, seq_len, embed_dim)
    
    def test_numerical_stability(self, gpu_attention, device, dtype):
        """Test numerical stability with extreme values."""
        batch_size, seq_len, embed_dim = 2, 64, 256
        
        # Create tensors with large values
        scale = 10.0 if dtype == torch.float32 else 5.0  # Adjust for precision
        query = torch.randn(batch_size, seq_len, embed_dim, dtype=dtype, device=device) * scale
        key = torch.randn(batch_size, seq_len, embed_dim, dtype=dtype, device=device) * scale
        value = torch.randn(batch_size, seq_len, embed_dim, dtype=dtype, device=device) * scale
        
        output = gpu_attention(query, key, value, need_weights=False)
        
        # Should not produce NaN or Inf
        assert torch.isfinite(output).all()
    
    def test_different_head_dimensions(self, device, dtype):
        """Test different head dimensions."""
        configs = [
            (512, 8),   # head_dim = 64
            (768, 12),  # head_dim = 64
            (1024, 16), # head_dim = 64
            (384, 6),   # head_dim = 64
        ]
        
        for embed_dim, num_heads in configs:
            attention = FlashAttention3(
                embed_dim=embed_dim,
                num_heads=num_heads,
                device=device,
                dtype=dtype,
            )
            
            query = torch.randn(2, 128, embed_dim, dtype=dtype, device=device)
            output = attention(query, need_weights=False)
            
            assert output.shape == query.shape
            assert torch.isfinite(output).all()