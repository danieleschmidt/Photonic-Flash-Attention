#!/usr/bin/env python3
"""Minimal functionality tests without PyTorch dependencies."""

import os
import sys
import numpy as np
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_basic_imports():
    """Test basic imports without PyTorch."""
    print("Testing basic imports...")
    try:
        # Test config imports
        from photonic_flash_attention.config import get_config, GlobalConfig
        
        config = get_config()
        print(f"  Config loaded: {type(config)}")
        
        # Test logging imports
        from photonic_flash_attention.utils.logging import get_logger, setup_logging
        
        logger = get_logger('test')
        print(f"  Logger created: {type(logger)}")
        
        # Test exceptions
        from photonic_flash_attention.utils.exceptions import (
            PhotonicFlashAttentionError, PhotonicHardwareError
        )
        
        print("  Exception classes imported")
        
        # Test version info
        try:
            from photonic_flash_attention import get_version
            version = get_version()
            print(f"  Version: {version}")
        except Exception as e:
            print(f"  Version import failed: {e}")
        
        print("✓ Basic imports successful")
        return True
        
    except Exception as e:
        print(f"✗ Basic imports failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_device_detection():
    """Test device detection without PyTorch."""
    print("\nTesting device detection...")
    try:
        os.environ['PHOTONIC_SIMULATION'] = 'true'
        
        from photonic_flash_attention.photonic.hardware.detection import (
            detect_photonic_hardware, get_photonic_devices, PhotonicDevice
        )
        
        # Test device detection
        has_devices = detect_photonic_hardware()
        print(f"  Devices detected: {has_devices}")
        
        devices = get_photonic_devices()
        print(f"  Found {len(devices)} devices")
        
        if devices:
            device = devices[0]
            print(f"  First device: {device.device_id}, {device.vendor}")
            
        print("✓ Device detection successful")
        return True
        
    except Exception as e:
        print(f"✗ Device detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_validation_functions():
    """Test validation functions without PyTorch tensors."""
    print("\nTesting validation functions...")
    try:
        from photonic_flash_attention.utils.validation import (
            validate_photonic_config, validate_sequence_length, validate_batch_size
        )
        
        # Test config validation
        valid_config = {
            'wavelengths': 80,
            'modulator_resolution': 6,
            'max_optical_power': 0.01
        }
        
        validate_photonic_config(valid_config)  # Should not raise
        print("  Valid config accepted")
        
        # Test sequence length validation
        validate_sequence_length(512)  # Should not raise
        print("  Valid sequence length accepted")
        
        # Test batch size validation
        validate_batch_size(8)  # Should not raise
        print("  Valid batch size accepted")
        
        # Test invalid config
        try:
            invalid_config = {'wavelengths': -1}
            validate_photonic_config(invalid_config)
            print("✗ Should have rejected invalid config")
            return False
        except Exception:
            print("  Invalid config properly rejected")
        
        print("✓ Validation functions successful")
        return True
        
    except Exception as e:
        print(f"✗ Validation functions failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_system():
    """Test configuration system."""
    print("\nTesting configuration system...")
    try:
        from photonic_flash_attention.config import get_config, GlobalConfig
        
        # Get initial config
        config = get_config()
        original_threshold = config.photonic_threshold
        print(f"  Original threshold: {original_threshold}")
        
        # Update config
        new_threshold = 1024
        GlobalConfig.update(photonic_threshold=new_threshold)
        
        # Verify update
        updated_config = get_config()
        if updated_config.photonic_threshold != new_threshold:
            print(f"✗ Config update failed: {updated_config.photonic_threshold} != {new_threshold}")
            return False
        
        print(f"  Updated threshold: {updated_config.photonic_threshold}")
        
        # Test config dict conversion
        config_dict = config.to_dict()
        print(f"  Config dict keys: {len(config_dict)}")
        
        # Reset to original
        GlobalConfig.update(photonic_threshold=original_threshold)
        
        print("✓ Configuration system successful")
        return True
        
    except Exception as e:
        print(f"✗ Configuration system failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_logging_system():
    """Test logging system."""
    print("\nTesting logging system...")
    try:
        from photonic_flash_attention.utils.logging import (
            setup_logging, get_logger, get_performance_logger
        )
        
        # Setup logging
        setup_logging(level='WARNING')  # Reduce noise
        
        # Get logger
        logger = get_logger('test_module')
        print(f"  Logger created: {logger.name}")
        
        # Test logging
        logger.info("Test info message")
        logger.warning("Test warning message")
        
        # Get performance logger
        perf_logger = get_performance_logger('test_perf')
        print(f"  Performance logger: {type(perf_logger)}")
        
        # Test performance timing
        perf_logger.start_timer('test_operation')
        import time
        time.sleep(0.01)  # Small delay
        duration = perf_logger.end_timer('test_operation')
        print(f"  Timed operation: {duration:.3f}s")
        
        print("✓ Logging system successful")
        return True
        
    except Exception as e:
        print(f"✗ Logging system failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cli_imports():
    """Test CLI module imports."""
    print("\nTesting CLI imports...")
    try:
        # This will fail without torch, but we can test the import structure
        try:
            from photonic_flash_attention import cli
            print("  CLI module imported successfully")
        except ImportError as e:
            if "torch" in str(e):
                print("  CLI module requires PyTorch (expected)")
            else:
                raise
        
        print("✓ CLI imports successful")
        return True
        
    except Exception as e:
        print(f"✗ CLI imports failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_package_structure():
    """Test overall package structure."""
    print("\nTesting package structure...")
    try:
        # Test that we can import the main package
        import photonic_flash_attention
        
        # Test core modules exist
        from photonic_flash_attention import core
        from photonic_flash_attention import photonic
        from photonic_flash_attention import utils
        from photonic_flash_attention import integration
        from photonic_flash_attention import config
        
        print("  All main modules found")
        
        # Test submodules exist
        from photonic_flash_attention.photonic import hardware
        from photonic_flash_attention.photonic import optical_kernels
        from photonic_flash_attention.utils import logging
        from photonic_flash_attention.utils import validation
        from photonic_flash_attention.utils import exceptions
        
        print("  All submodules found")
        
        print("✓ Package structure successful")
        return True
        
    except Exception as e:
        print(f"✗ Package structure failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run minimal tests."""
    print("=== Photonic Flash Attention Minimal Tests (No PyTorch) ===\n")
    
    # Set environment for testing
    os.environ['PHOTONIC_SIMULATION'] = 'true'
    os.environ['PHOTONIC_LOG_LEVEL'] = 'WARNING'
    
    tests = [
        test_basic_imports,
        test_package_structure,
        test_config_system,
        test_logging_system,
        test_device_detection,
        test_validation_functions,
        test_cli_imports,
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
        print("\n✅ All minimal tests passed!")
        print("\nNote: Full tests require PyTorch installation")
        return 0


if __name__ == "__main__":
    sys.exit(main())