#!/usr/bin/env python3
"""
Comprehensive test for the Photonic Flash Attention implementation.
Tests all major components and verifies the system works end-to-end.
"""

import os
import sys
import traceback
import time
from typing import Dict, Any, List

import torch
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported successfully."""
    print("🧪 Testing imports...")
    
    try:
        # Core imports
        from photonic_flash_attention import (
            PhotonicFlashAttention, 
            get_device_info, 
            get_version,
            set_global_config
        )
        
        # Component imports
        from photonic_flash_attention.core.flash_attention_3 import FlashAttention3
        from photonic_flash_attention.core.photonic_attention import PhotonicAttention
        from photonic_flash_attention.core.hybrid_router import HybridFlashAttention
        
        # Hardware detection
        from photonic_flash_attention.photonic.hardware.detection import (
            detect_photonic_hardware,
            get_best_photonic_device,
            is_photonic_available
        )
        
        # Configuration
        from photonic_flash_attention.config import get_config
        
        # Utilities
        from photonic_flash_attention.utils.logging import get_logger
        from photonic_flash_attention.utils.exceptions import PhotonicFlashAttentionError
        
        print("✅ All imports successful")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False


def test_basic_functionality():
    """Test basic photonic attention functionality."""
    print("\n🧪 Testing basic functionality...")
    
    try:
        from photonic_flash_attention import PhotonicFlashAttention, get_device_info
        
        # Get device info
        device_info = get_device_info()
        print(f"   Device info: {device_info}")
        
        # Create attention module
        attention = PhotonicFlashAttention(
            embed_dim=256,
            num_heads=8,
            dropout=0.1,
            photonic_threshold=128,
        )
        
        # Test with different sequence lengths
        batch_size = 2
        seq_lens = [64, 128, 256]
        
        for seq_len in seq_lens:
            print(f"   Testing seq_len={seq_len}")
            
            # Create test data
            query = torch.randn(batch_size, seq_len, 256)
            key = torch.randn(batch_size, seq_len, 256)  
            value = torch.randn(batch_size, seq_len, 256)
            attention_mask = torch.ones(batch_size, seq_len)
            
            # Forward pass
            start_time = time.perf_counter()
            output = attention(query, key, value, attention_mask)
            end_time = time.perf_counter()
            
            # Verify output
            expected_shape = (batch_size, seq_len, 256)
            assert output.shape == expected_shape, f"Shape mismatch: {output.shape} != {expected_shape}"
            assert torch.isfinite(output).all(), "Output contains non-finite values"
            
            # Check performance stats
            stats = attention.get_performance_stats()
            device_used = stats['last_device_used']
            latency = (end_time - start_time) * 1000
            
            print(f"     Device: {device_used}, Latency: {latency:.2f}ms")
        
        print("✅ Basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False


def test_hardware_detection():
    """Test photonic hardware detection."""
    print("\n🧪 Testing hardware detection...")
    
    try:
        from photonic_flash_attention.photonic.hardware.detection import (
            detect_photonic_hardware,
            get_best_photonic_device,
            list_photonic_devices,
            is_photonic_available
        )
        
        # Test hardware detection
        is_available = detect_photonic_hardware()
        print(f"   Photonic hardware available: {is_available}")
        
        # Test device listing
        devices = list_photonic_devices()
        print(f"   Found {len(devices)} photonic device(s)")
        
        for device in devices:
            print(f"     {device['id']}: {device['vendor']} {device['type']}")
            print(f"       Wavelengths: {device['wavelengths']}")
            print(f"       Max power: {device['max_power_mw']:.1f} mW")
            print(f"       Available: {device['available']}")
        
        # Test best device selection
        best_device = get_best_photonic_device()
        if best_device:
            print(f"   Best device: {best_device.device_id}")
        else:
            print("   No best device found")
        
        print("✅ Hardware detection test passed")
        return True
        
    except Exception as e:
        print(f"❌ Hardware detection test failed: {e}")
        traceback.print_exc()
        return False


def test_hybrid_router():
    """Test the hybrid routing system."""
    print("\n🧪 Testing hybrid router...")
    
    try:
        from photonic_flash_attention.core.hybrid_router import HybridFlashAttention
        
        # Create hybrid attention
        hybrid_attention = HybridFlashAttention(
            embed_dim=384,
            num_heads=6,
            enable_scaling=True,
            max_concurrent_requests=2,
        )
        
        # Test with various workloads
        test_cases = [
            (1, 128),   # Small: should use GPU
            (1, 512),   # Medium: may use photonic
            (4, 1024),  # Large: should use photonic if available
        ]
        
        for batch_size, seq_len in test_cases:
            print(f"   Testing batch_size={batch_size}, seq_len={seq_len}")
            
            query = torch.randn(batch_size, seq_len, 384)
            
            # Multiple iterations to test adaptive routing
            for i in range(3):
                output = hybrid_attention(query)
                stats = hybrid_attention.get_performance_stats()
                
                print(f"     Iter {i+1}: device={stats.get('gpu_stats', {}).get('device', 'unknown')}")
            
            # Test performance stats
            perf_stats = hybrid_attention.get_performance_stats()
            print(f"     Total requests: {perf_stats['total_requests']}")
            print(f"     Warmup complete: {perf_stats['warmup_complete']}")
        
        print("✅ Hybrid router test passed")
        return True
        
    except Exception as e:
        print(f"❌ Hybrid router test failed: {e}")
        traceback.print_exc()
        return False


def test_configuration():
    """Test configuration system."""
    print("\n🧪 Testing configuration...")
    
    try:
        from photonic_flash_attention.config import get_config
        from photonic_flash_attention import set_global_config
        
        # Test default config
        config = get_config()
        print(f"   Default photonic threshold: {config.photonic_threshold}")
        print(f"   Default wavelengths: {config.photonic_wavelengths}")
        print(f"   Auto device selection: {config.auto_device_selection}")
        
        # Test config update
        original_threshold = config.photonic_threshold
        set_global_config(
            photonic_threshold=1024,
            enable_profiling=True,
            log_level="DEBUG"
        )
        
        updated_config = get_config()
        assert updated_config.photonic_threshold == 1024
        assert updated_config.enable_profiling == True
        assert updated_config.log_level == "DEBUG"
        
        print("   ✓ Configuration update successful")
        
        # Test environment variable loading
        os.environ['PHOTONIC_THRESHOLD'] = '2048'
        os.environ['PHOTONIC_WAVELENGTHS'] = '128'
        
        # Would need to recreate config instance to test env loading
        print("   ✓ Environment variables set")
        
        # Reset config
        set_global_config(photonic_threshold=original_threshold)
        
        print("✅ Configuration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        traceback.print_exc()
        return False


def test_error_handling():
    """Test error handling and recovery."""
    print("\n🧪 Testing error handling...")
    
    try:
        from photonic_flash_attention import PhotonicFlashAttention
        from photonic_flash_attention.utils.exceptions import PhotonicFlashAttentionError
        
        # Test invalid configuration
        try:
            attention = PhotonicFlashAttention(
                embed_dim=-1,  # Invalid
                num_heads=8,
            )
            print("❌ Should have raised error for negative embed_dim")
            return False
        except (ValueError, PhotonicFlashAttentionError):
            print("   ✓ Invalid embed_dim caught")
        
        # Test mismatched dimensions
        attention = PhotonicFlashAttention(embed_dim=256, num_heads=8)
        
        try:
            query = torch.randn(2, 64, 128)  # Wrong embed_dim
            output = attention(query)
            print("❌ Should have raised error for dimension mismatch")
            return False
        except (ValueError, RuntimeError):
            print("   ✓ Dimension mismatch caught")
        
        # Test NaN input handling
        try:
            query = torch.full((2, 64, 256), float('nan'))
            output = attention(query)
            # Should either handle gracefully or raise error
            if torch.isnan(output).any():
                print("   ⚠️ NaN input produced NaN output (acceptable)")
            else:
                print("   ✓ NaN input handled gracefully")
        except (ValueError, RuntimeError):
            print("   ✓ NaN input error caught")
        
        print("✅ Error handling test passed")
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        traceback.print_exc()
        return False


def test_cli():
    """Test CLI functionality."""
    print("\n🧪 Testing CLI...")
    
    try:
        from photonic_flash_attention.cli import device_info, benchmark
        
        # Test device info
        print("   Testing device info...")
        
        # Mock args for device info
        class MockArgs:
            def __init__(self):
                self.json = False
                self.refresh = True
        
        args = MockArgs()
        device_info(args)
        
        print("   ✓ Device info command works")
        
        # Test benchmark with minimal settings
        print("   Testing benchmark (minimal)...")
        
        class BenchmarkArgs:
            def __init__(self):
                self.seq_lengths = [64, 128]
                self.batch_sizes = [1, 2]
                self.embed_dim = 128
                self.num_heads = 4
                self.num_iterations = 2
                self.output = None
                self.verbose = False
        
        benchmark_args = BenchmarkArgs()
        results = benchmark(benchmark_args)
        
        assert len(results) == 4  # 2 seq_lens × 2 batch_sizes
        print(f"   ✓ Benchmark completed: {len(results)} results")
        
        print("✅ CLI test passed")
        return True
        
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        traceback.print_exc()
        return False


def test_examples():
    """Test that examples run without errors."""
    print("\n🧪 Testing examples...")
    
    try:
        # Test basic usage example
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'examples'))
        
        # Import and run basic example functions
        import basic_usage
        
        # Test basic attention computation
        print("   Running basic attention example...")
        output, stats = basic_usage.basic_attention_example()
        
        assert output is not None
        assert stats is not None
        print("   ✓ Basic example completed")
        
        # Test implementation comparison
        print("   Running implementation comparison...")
        comparison_results = basic_usage.compare_implementations()
        
        assert len(comparison_results) > 0
        print(f"   ✓ Comparison completed: {len(comparison_results)} results")
        
        print("✅ Examples test passed")
        return True
        
    except Exception as e:
        print(f"❌ Examples test failed: {e}")
        traceback.print_exc()
        return False


def performance_benchmark():
    """Run performance benchmarks."""
    print("\n🏃‍♂️ Running performance benchmarks...")
    
    try:
        from photonic_flash_attention import PhotonicFlashAttention
        
        # Configuration
        embed_dim = 768
        num_heads = 12
        batch_size = 4
        seq_lengths = [128, 256, 512, 1024]
        
        attention = PhotonicFlashAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            photonic_threshold=256,
        )
        
        print(f"   Configuration: embed_dim={embed_dim}, num_heads={num_heads}")
        print("   " + "="*60)
        print(f"   {'Seq Len':>8} | {'Device':>10} | {'Latency':>10} | {'Throughput':>12}")
        print("   " + "-"*60)
        
        results = []
        
        for seq_len in seq_lengths:
            # Create test data
            query = torch.randn(batch_size, seq_len, embed_dim)
            
            # Warmup
            for _ in range(2):
                _ = attention(query)
            
            # Benchmark
            times = []
            for _ in range(5):
                start = time.perf_counter()
                output = attention(query)
                end = time.perf_counter()
                times.append(end - start)
            
            # Calculate metrics
            avg_time = np.mean(times[1:])  # Skip first measurement
            latency_ms = avg_time * 1000
            tokens_per_sec = (batch_size * seq_len) / avg_time
            
            stats = attention.get_performance_stats()
            device_used = stats['last_device_used']
            
            print(f"   {seq_len:>8} | {device_used:>10} | {latency_ms:>8.2f}ms | {tokens_per_sec:>8.0f} tok/s")
            
            results.append({
                'seq_len': seq_len,
                'device': device_used,
                'latency_ms': latency_ms,
                'throughput': tokens_per_sec,
            })
        
        print("   " + "="*60)
        
        # Analyze results
        gpu_results = [r for r in results if r['device'] == 'gpu']
        photonic_results = [r for r in results if r['device'] == 'photonic']
        
        if gpu_results and photonic_results:
            print(f"   GPU sequences: {[r['seq_len'] for r in gpu_results]}")
            print(f"   Photonic sequences: {[r['seq_len'] for r in photonic_results]}")
            
            # Calculate speedups
            for p_result in photonic_results:
                seq_len = p_result['seq_len']
                g_result = next((r for r in gpu_results if r['seq_len'] == seq_len), None)
                if g_result:
                    speedup = g_result['latency_ms'] / p_result['latency_ms']
                    print(f"   Speedup at {seq_len}: {speedup:.2f}x")
        
        print("✅ Performance benchmark completed")
        return True
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        traceback.print_exc()
        return False


def run_quality_gates():
    """Run quality gates and checks."""
    print("\n🛡️ Running quality gates...")
    
    quality_checks = {
        'imports': False,
        'basic_functionality': False,
        'hardware_detection': False,
        'hybrid_router': False,
        'configuration': False,
        'error_handling': False,
        'cli': False,
        'examples': False,
    }
    
    # Run all tests
    test_functions = [
        ('imports', test_imports),
        ('basic_functionality', test_basic_functionality),
        ('hardware_detection', test_hardware_detection),
        ('hybrid_router', test_hybrid_router),
        ('configuration', test_configuration),
        ('error_handling', test_error_handling),
        ('cli', test_cli),
        ('examples', test_examples),
    ]
    
    for test_name, test_func in test_functions:
        try:
            quality_checks[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            quality_checks[test_name] = False
    
    # Run performance benchmark
    try:
        perf_success = performance_benchmark()
    except Exception as e:
        print(f"❌ Performance benchmark crashed: {e}")
        perf_success = False
    
    # Summary
    print("\n" + "="*70)
    print("📊 QUALITY GATES SUMMARY")
    print("="*70)
    
    passed_tests = sum(quality_checks.values())
    total_tests = len(quality_checks)
    
    for test_name, passed in quality_checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20s}: {status}")
    
    print(f"\nPerformance Benchmark: {'✅ PASS' if perf_success else '❌ FAIL'}")
    print("-"*70)
    print(f"Overall Score: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests:.1%})")
    
    if passed_tests == total_tests and perf_success:
        print("\n🎉 ALL QUALITY GATES PASSED! System is ready for deployment.")
        return True
    else:
        print(f"\n⚠️ {total_tests - passed_tests} quality gate(s) failed. Review and fix issues before deployment.")
        return False


def main():
    """Main test runner."""
    print("🚀 Photonic Flash Attention - Comprehensive System Test")
    print("="*70)
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Working directory: {os.getcwd()}")
    print()
    
    # Enable simulation mode for testing
    os.environ['PHOTONIC_SIMULATION'] = '1'
    print("🔧 Simulation mode enabled for testing")
    
    success = run_quality_gates()
    
    print("\n" + "="*70)
    if success:
        print("🎊 TEST SUITE COMPLETED SUCCESSFULLY!")
        print("   The Photonic Flash Attention system is working correctly.")
        print("   All major components have been validated.")
        exit_code = 0
    else:
        print("💥 TEST SUITE FAILED!")
        print("   Some components are not working correctly.")
        print("   Please review the failed tests and fix issues.")
        exit_code = 1
    
    print("="*70)
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())