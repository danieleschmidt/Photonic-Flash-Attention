"""Pytest configuration and shared fixtures."""

import pytest
import torch
import numpy as np
from typing import Generator, Tuple
import os
import tempfile

# Enable simulation mode for testing
os.environ['PHOTONIC_SIMULATION'] = '1'

from photonic_flash_attention import PhotonicFlashAttention
from photonic_flash_attention.core.flash_attention_3 import FlashAttention3
from photonic_flash_attention.core.photonic_attention import PhotonicAttention
from photonic_flash_attention.core.hybrid_router import HybridFlashAttention


@pytest.fixture(scope="session")
def device():
    """Test device (CUDA if available, CPU otherwise)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture(params=[torch.float32, torch.float16])
def dtype(request):
    """Test with different dtypes."""
    return request.param


@pytest.fixture(params=[
    (2, 128, 512, 8),   # Small
    (4, 256, 768, 12),  # Medium  
    (1, 512, 1024, 16), # Large
])
def attention_config(request):
    """Different attention configurations (batch, seq_len, embed_dim, num_heads)."""
    return request.param


@pytest.fixture
def sample_tensors(attention_config, dtype, device):
    """Generate sample query, key, value tensors."""
    batch_size, seq_len, embed_dim, num_heads = attention_config
    
    torch.manual_seed(42)  # Reproducible tests
    
    query = torch.randn(batch_size, seq_len, embed_dim, dtype=dtype, device=device)
    key = torch.randn(batch_size, seq_len, embed_dim, dtype=dtype, device=device)
    value = torch.randn(batch_size, seq_len, embed_dim, dtype=dtype, device=device)
    
    # Optional attention mask
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    # Mask out last 10% of sequence
    mask_len = seq_len // 10
    if mask_len > 0:
        attention_mask[:, -mask_len:] = False
    
    return query, key, value, attention_mask


@pytest.fixture
def gpu_attention(attention_config, device, dtype):
    """GPU Flash Attention module."""
    batch_size, seq_len, embed_dim, num_heads = attention_config
    
    attention = FlashAttention3(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0.0,
        device=device,
        dtype=dtype,
    )
    
    return attention


@pytest.fixture
def photonic_attention(attention_config, device, dtype):
    """Photonic attention module."""
    batch_size, seq_len, embed_dim, num_heads = attention_config
    
    attention = PhotonicAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0.0,
        device=device,
        dtype=dtype,
        safety_checks=True,
    )
    
    return attention


@pytest.fixture
def hybrid_attention(attention_config, device, dtype):
    """Hybrid attention module."""
    batch_size, seq_len, embed_dim, num_heads = attention_config
    
    attention = HybridFlashAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0.0,
        device=device,
        dtype=dtype,
        enable_scaling=True,
    )
    
    return attention


@pytest.fixture
def photonic_flash_attention(attention_config, device, dtype):
    """Main PhotonicFlashAttention module."""
    batch_size, seq_len, embed_dim, num_heads = attention_config
    
    attention = PhotonicFlashAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0.0,
        device='auto',
        dtype=dtype,
    )
    
    return attention


@pytest.fixture
def temp_dir():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture(scope="session")
def benchmark_results():
    """Store benchmark results across tests."""
    return {}


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "photonic: marks tests that require photonic hardware"
    )
    config.addinivalue_line(
        "markers", "security: marks security-related tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks performance benchmark tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names/locations."""
    for item in items:
        # Add slow marker to performance tests
        if "performance" in str(item.fspath) or "benchmark" in item.name:
            item.add_marker(pytest.mark.slow)
            item.add_marker(pytest.mark.performance)
        
        # Add GPU marker to tests that need GPU
        if "gpu" in item.name or torch.cuda.is_available():
            item.add_marker(pytest.mark.gpu)
        
        # Add photonic marker to photonic tests
        if "photonic" in str(item.fspath) or "photonic" in item.name:
            item.add_marker(pytest.mark.photonic)
        
        # Add security marker to security tests
        if "security" in str(item.fspath):
            item.add_marker(pytest.mark.security)


@pytest.fixture
def assert_tensors_close():
    """Helper fixture for tensor comparison."""
    def _assert_close(tensor1, tensor2, rtol=1e-5, atol=1e-8, msg=""):
        """Assert that two tensors are close."""
        if tensor1.dtype != tensor2.dtype:
            # Convert to common dtype for comparison
            common_dtype = torch.float32
            tensor1 = tensor1.to(common_dtype)
            tensor2 = tensor2.to(common_dtype)
        
        try:
            torch.testing.assert_close(tensor1, tensor2, rtol=rtol, atol=atol)
        except AssertionError as e:
            pytest.fail(f"Tensors not close {msg}: {e}")
    
    return _assert_close


@pytest.fixture
def performance_timer():
    """Timer for performance measurements."""
    import time
    
    class Timer:
        def __init__(self):
            self.times = {}
        
        def start(self, name):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.times[name] = time.perf_counter()
        
        def end(self, name):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if name in self.times:
                elapsed = time.perf_counter() - self.times[name]
                del self.times[name]
                return elapsed
            return 0.0
    
    return Timer()