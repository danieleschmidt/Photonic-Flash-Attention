#!/usr/bin/env python3
"""Test validation functions without PyTorch."""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_non_torch_validation():
    """Test validation functions that don't require PyTorch."""
    print("Testing non-PyTorch validation functions...")
    try:
        from photonic_flash_attention.utils.validation import (
            validate_photonic_config, validate_sequence_length, validate_batch_size,
            sanitize_config_dict, validate_thermal_conditions
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
        
        # Test thermal validation
        validate_thermal_conditions(25.0)  # Should not raise
        print("  Valid temperature accepted")
        
        # Test config sanitization
        config = {'valid_key': 1, 'unknown_key': 2}
        sanitized = sanitize_config_dict(config, ['valid_key'])
        assert 'valid_key' in sanitized
        assert 'unknown_key' not in sanitized
        print("  Config sanitization works")
        
        print("✓ Non-PyTorch validation functions successful")
        return True
        
    except Exception as e:
        print(f"✗ Non-PyTorch validation functions failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_non_torch_validation()
    sys.exit(0 if success else 1)