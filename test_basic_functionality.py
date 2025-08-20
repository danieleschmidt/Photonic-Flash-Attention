#!/usr/bin/env python3
"""Basic functionality tests for Photonic Flash Attention."""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_import():
    """Test basic imports work."""
    print("Testing imports...")
    try:
        import photonic_flash_attention
        from photonic_flash_attention import PhotonicFlashAttention, get_device_info, get_version
        from photonic_flash_attention.core.flash_attention_3 import FlashAttention3
        from photonic_flash_attention.core.hybrid_router import HybridFlashAttention
        from photonic_flash_attention.photonic.hardware.detection import detect_photonic_hardware
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_basic_attention():
    """Test basic attention functionality."""
    print("\nTesting basic attention...")
    try:
        # Set simulation mode for testing
        os.environ['PHOTONIC_SIMULATION'] = 'true'
        
        from photonic_flash_attention.core.flash_attention_3 import FlashAttention3
        
        # Create attention module
        attention = FlashAttention3(
            embed_dim=64,
            num_heads=4,
            dropout=0.0,
        )
        
        # Create test input
        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, 64)
        
        # Forward pass
        output, weights = attention(x, need_weights=True)
        
        # Check output shape
        assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"
        
        # Check attention weights shape
        expected_weights_shape = (batch_size, 4, seq_len, seq_len)
        if weights is not None:
            assert weights.shape == expected_weights_shape, f"Weights shape mismatch: {weights.shape} vs {expected_weights_shape}"
        
        print("✓ Basic attention test passed")
        return True
        
    except Exception as e:
        print(f"✗ Basic attention test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_photonic_attention():
    """Test photonic attention with simulation."""
    print("\nTesting photonic attention...")
    try:
        os.environ['PHOTONIC_SIMULATION'] = 'true'
        
        from photonic_flash_attention.core.photonic_attention import PhotonicAttention
        
        # Create photonic attention module
        attention = PhotonicAttention(
            embed_dim=64,
            num_heads=4,
            dropout=0.0,
            safety_checks=False,  # Disable safety checks for testing
        )
        
        # Create test input
        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, 64)
        
        # Forward pass (should fallback to GPU since simulation mode)
        output, weights = attention(x, need_weights=False)
        
        # Check output shape
        assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"
        
        print("✓ Photonic attention test passed")
        return True
        
    except Exception as e:
        print(f"✗ Photonic attention test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hybrid_attention():
    """Test hybrid attention router."""
    print("\nTesting hybrid attention...")
    try:
        os.environ['PHOTONIC_SIMULATION'] = 'true'
        
        from photonic_flash_attention.core.hybrid_router import HybridFlashAttention
        
        # Create hybrid attention module
        attention = HybridFlashAttention(
            embed_dim=64,
            num_heads=4,
            dropout=0.0,
            enable_scaling=False,  # Disable scaling for testing
        )
        
        # Create test input
        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, 64)
        
        # Forward pass
        result = attention(x)
        
        # Handle tuple return (output, weights) vs single output
        if isinstance(result, tuple):
            output, weights = result
        else:
            output = result
        
        # Check output shape
        assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"
        
        # Get performance stats
        stats = attention.get_performance_stats()
        assert isinstance(stats, dict), "Performance stats should be a dictionary"
        
        print("✓ Hybrid attention test passed")
        return True
        
    except Exception as e:
        print(f"✗ Hybrid attention test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_optical_kernels():
    """Test optical computation kernels."""
    print("\nTesting optical kernels...")
    try:
        from photonic_flash_attention.photonic.optical_kernels.matrix_mult import OpticalMatMul
        from photonic_flash_attention.photonic.optical_kernels.nonlinearity import OpticalSoftmax
        
        # Test optical matrix multiplication
        matmul = OpticalMatMul()
        A = torch.randn(4, 8) * 0.01  # Very small values for power budget
        B = torch.randn(8, 6) * 0.01
        
        result = matmul.forward(A, B)
        expected_shape = (4, 6)
        assert result.shape == expected_shape, f"MatMul shape mismatch: {result.shape} vs {expected_shape}"
        
        # Test optical softmax
        softmax = OpticalSoftmax()
        x = torch.randn(2, 4, 8) * 0.01  # Small values for power budget
        result = softmax.forward(x, dim=-1)
        
        assert result.shape == x.shape, f"Softmax shape mismatch: {result.shape} vs {x.shape}"
        
        # Check softmax properties (approximately)
        sums = result.sum(dim=-1)
        expected_sums = torch.ones_like(sums)
        assert torch.allclose(sums, expected_sums, atol=0.1), "Softmax doesn't sum to 1"
        
        print("✓ Optical kernels test passed")
        return True
        
    except Exception as e:
        print(f"✗ Optical kernels test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_device_detection():
    """Test device detection."""
    print("\nTesting device detection...")
    try:
        from photonic_flash_attention.photonic.hardware.detection import (
            detect_photonic_hardware, get_photonic_devices, get_device_info
        )
        
        # Should detect simulation device with env var set
        os.environ['PHOTONIC_SIMULATION'] = 'true'
        
        has_devices = detect_photonic_hardware()
        assert has_devices, "Should detect simulation device"
        
        devices = get_photonic_devices()
        assert len(devices) > 0, "Should have at least one device"
        
        device_info = get_device_info()
        assert isinstance(device_info, dict), "Device info should be a dictionary"
        
        print("✓ Device detection test passed")
        return True
        
    except Exception as e:
        print(f"✗ Device detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_configuration():
    """Test configuration system."""
    print("\nTesting configuration...")
    try:
        from photonic_flash_attention.config import get_config, GlobalConfig
        
        config = get_config()
        assert isinstance(config, GlobalConfig), "Config should be GlobalConfig instance"
        
        # Test setting configuration
        original_threshold = config.photonic_threshold
        GlobalConfig.update(photonic_threshold=1024)
        
        updated_config = get_config()
        assert updated_config.photonic_threshold == 1024, "Config update failed"
        
        # Reset to original
        GlobalConfig.update(photonic_threshold=original_threshold)
        
        print("✓ Configuration test passed")
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test integration with PyTorch modules."""
    print("\nTesting PyTorch integration...")
    try:
        from photonic_flash_attention.integration.pytorch.modules import PhotonicFlashAttention
        
        # Create module
        attention = PhotonicFlashAttention(
            embed_dim=64,
            num_heads=4,
            dropout=0.0,
        )
        
        # Create test input
        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, 64)
        
        # Forward pass
        output = attention(x)
        
        # Check output shape
        assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"
        
        # Test that module can be put in different modes
        attention.train()
        attention.eval()
        
        print("✓ PyTorch integration test passed")
        return True
        
    except Exception as e:
        print(f"✗ PyTorch integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling():
    """Test error handling and validation."""
    print("\nTesting error handling...")
    try:
        from photonic_flash_attention.utils.validation import (
            validate_attention_inputs, validate_tensor_shape
        )
        from photonic_flash_attention.utils.exceptions import PhotonicComputationError
        
        # Test valid inputs
        q = torch.randn(2, 8, 64)
        k = torch.randn(2, 8, 64)
        v = torch.randn(2, 8, 64)
        
        validate_attention_inputs(q, k, v)  # Should not raise
        
        # Test invalid inputs
        try:
            invalid_k = torch.randn(2, 8, 32)  # Wrong embed_dim
            validate_attention_inputs(q, invalid_k, v)
            assert False, "Should have raised exception"
        except PhotonicComputationError:
            pass  # Expected
        
        print("✓ Error handling test passed")
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=== Photonic Flash Attention Basic Functionality Tests ===\n")
    
    # Set environment for testing
    os.environ['PHOTONIC_SIMULATION'] = 'true'
    os.environ['PHOTONIC_LOG_LEVEL'] = 'WARNING'  # Reduce log noise
    
    tests = [
        test_import,
        test_configuration,
        test_device_detection,
        test_basic_attention,
        test_photonic_attention,
        test_hybrid_attention,
        test_optical_kernels,
        test_integration,
        test_error_handling,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            failed += 1
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed > 0:
        print("\n❌ Some tests failed!")
        return 1
    else:
        print("\n✅ All tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())