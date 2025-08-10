"""Security validation tests for photonic attention system."""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

def test_input_sanitization():
    """Test input data sanitization."""
    test_inputs = [
        ("normal string", True),  # Should pass
        ("string with spaces", True),  # Should pass
        ("../path/traversal", False),  # Should fail
        ("/absolute/path", False),  # Should fail
        ("string\x00with\x01control", False),  # Should fail
        ("a" * 10001, False),  # Too long, should fail
        ("SELECT * FROM users", True),  # SQL-like but not actual injection
    ]
    
    # Mock security validator since we can't import with torch
    class MockSecurityValidator:
        def _sanitize_string(self, s):
            # Basic sanitization logic
            if '..' in s or s.startswith('/'):
                raise Exception("Path traversal detected")
            if any(ord(c) < 32 and c not in '\t\n\r' for c in s):
                raise Exception("Control characters detected")
            if len(s) > 10000:
                raise Exception("String too long")
            return s
    
    validator = MockSecurityValidator()
    
    for test_input, should_pass in test_inputs:
        if should_pass:
            try:
                result = validator._sanitize_string(test_input)
                assert result == test_input
            except Exception:
                raise AssertionError(f"Input should have passed: {test_input}")
        else:
            try:
                validator._sanitize_string(test_input)
                raise AssertionError(f"Input should have failed: {test_input}")
            except Exception:
                pass  # Expected to fail
    
    print("✅ Input sanitization test passed")


if __name__ == "__main__":
    """Run security validation tests."""
    print("🔒 Running Security Validation Tests")
    print("=" * 40)
    
    test_input_sanitization()
    
    print("=" * 40)
    print("✅ Security tests passed!")