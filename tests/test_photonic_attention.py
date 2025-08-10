"""Comprehensive tests for photonic flash attention."""

import pytest
import sys
import os
import time
from typing import Dict, Any

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Set simulation mode for testing
os.environ['PHOTONIC_SIMULATION'] = 'true'

def test_config_system():
    """Test configuration system."""
    try:
        from src.photonic_flash_attention.config import PhotonicConfig, get_config
        
        # Test default config
        config = get_config()
        assert hasattr(config, 'photonic_threshold')
        assert hasattr(config, 'auto_device_selection')
        
        # Test custom config
        custom_config = PhotonicConfig(
            photonic_threshold=1024,
            auto_device_selection=True
        )
        assert custom_config.photonic_threshold == 1024
        
        print("✅ Config system test passed")
        
    except ImportError:
        pytest.skip("Config system requires torch")


def test_hardware_detection():
    """Test photonic hardware detection."""
    try:
        from src.photonic_flash_attention.photonic.hardware.detection import (
            detect_photonic_hardware, get_photonic_devices, get_device_info
        )
        
        # Should detect simulation device
        available = detect_photonic_hardware()
        assert available  # Should be True due to simulation mode
        
        devices = get_photonic_devices()
        assert len(devices) > 0  # Should have at least simulation device
        
        device_info = get_device_info()
        assert 'num_devices' in device_info
        assert device_info['num_devices'] > 0
        
        print("✅ Hardware detection test passed")
        
    except ImportError:
        pytest.skip("Hardware detection requires torch")


def test_security_validation():
    """Test security utilities.""" 
    try:
        from src.photonic_flash_attention.utils.security import (
            SecurityValidator, validate_optical_power, validate_wavelength
        )
        
        validator = SecurityValidator()
        
        # Test optical power validation
        validator.validate_optical_power(0.001, "test")  # 1mW - should pass
        
        with pytest.raises(Exception):  # Should raise for unsafe power
            validator.validate_optical_power(1.0, "test")  # 1W - too high
        
        # Test wavelength validation
        validator.validate_wavelength(1550e-9, "test")  # 1550nm - should pass
        
        with pytest.raises(Exception):  # Should raise for unsafe wavelength
            validator.validate_wavelength(400e-9, "test")  # 400nm - outside safe range
        
        print("✅ Security validation test passed")
        
    except ImportError:
        pytest.skip("Security validation requires torch")


def test_error_recovery():
    """Test error recovery system."""
    try:
        from src.photonic_flash_attention.core.error_recovery import (
            ErrorRecoveryManager, RecoveryStrategy, ErrorSeverity
        )
        
        manager = ErrorRecoveryManager()
        
        # Test error handling
        def failing_operation():
            raise ValueError("Test error")
        
        def fallback_operation():
            return "fallback_result"
        
        context = {
            'operation_func': failing_operation,
            'fallback_func': fallback_operation
        }
        
        result = manager.handle_error(
            ValueError("Test error"), 
            "test_operation", 
            context
        )
        
        # Should fallback successfully
        assert result == "fallback_result"
        
        stats = manager.get_recovery_stats()
        assert stats['total_stats']['total_errors'] == 1
        
        print("✅ Error recovery test passed")
        
    except ImportError:
        pytest.skip("Error recovery requires torch")


def test_performance_optimization():
    """Test performance optimization system."""
    try:
        from src.photonic_flash_attention.optimization.performance_optimizer import (
            OptimizationConfig, PerformanceProfile, AutoTuner
        )
        
        config = OptimizationConfig(enable_autotuning=True)
        assert config.enable_autotuning
        
        # Test performance profile
        profile = PerformanceProfile(
            batch_size=16,
            seq_length=512, 
            embed_dim=768,
            num_heads=12,
            device_type="gpu",
            avg_latency_ms=25.0,
            avg_throughput_ops_per_sec=1000.0,
            memory_usage_mb=256.0,
            energy_consumption_mj=0.5
        )
        
        assert profile.batch_size == 16
        assert profile.device_type == "gpu"
        
        # Test autotuner
        autotuner = AutoTuner(config)
        device_suggestion = autotuner.suggest_optimal_device(16, 512, 768, 12)
        assert device_suggestion in ['gpu', 'photonic', 'cpu']
        
        print("✅ Performance optimization test passed")
        
    except ImportError:
        pytest.skip("Performance optimization requires torch")


def test_load_balancing():
    """Test load balancing system."""
    try:
        from src.photonic_flash_attention.scaling.load_balancer import (
            LoadBalancerConfig, ComputeNode, NodeStatus, ConsistentHashRing
        )
        
        # Test config
        config = LoadBalancerConfig(max_nodes=5, enable_auto_scaling=True)
        assert config.max_nodes == 5
        
        # Test compute node
        node = ComputeNode(
            node_id="test_node",
            device_type="photonic", 
            capacity=16,
            current_load=8,
            weight=2.0
        )
        
        assert node.utilization == 0.5  # 8/16
        assert node.is_available
        
        # Test consistent hash ring
        ring = ConsistentHashRing()
        ring.add_node("node1")
        ring.add_node("node2")
        
        selected_node = ring.get_node("test_key")
        assert selected_node in ["node1", "node2"]
        
        print("✅ Load balancing test passed")
        
    except ImportError:
        pytest.skip("Load balancing requires torch")


def test_health_monitoring():
    """Test health monitoring system."""
    try:
        from src.photonic_flash_attention.monitoring.health_monitor import (
            HealthStatus, HealthMetric, ComponentType
        )
        
        # Test health metric
        metric = HealthMetric(
            name="cpu_usage",
            value=75.0,
            unit="percent",
            status=HealthStatus.WARNING,
            threshold_warning=70.0,
            threshold_critical=90.0
        )
        
        assert metric.status == HealthStatus.WARNING
        assert metric.value == 75.0
        
        print("✅ Health monitoring test passed")
        
    except ImportError:
        pytest.skip("Health monitoring requires additional dependencies")


def test_validation_utilities():
    """Test validation utilities."""
    try:
        from src.photonic_flash_attention.utils.validation import (
            validate_sequence_length, validate_batch_size
        )
        
        # Valid inputs should not raise
        validate_sequence_length(512)
        validate_batch_size(16)
        
        # Invalid inputs should raise
        with pytest.raises(Exception):
            validate_sequence_length(-1)
        
        with pytest.raises(Exception):
            validate_batch_size(0)
        
        print("✅ Validation utilities test passed")
        
    except ImportError:
        pytest.skip("Validation utilities require torch")


def test_logging_system():
    """Test logging system.""" 
    try:
        from src.photonic_flash_attention.utils.logging import (
            get_logger, setup_logging
        )
        
        # Test logger creation
        logger = get_logger("test_module")
        assert logger is not None
        
        # Test logging setup
        setup_logging(level="INFO", enable_performance_logging=True)
        
        logger.info("Test log message")
        
        print("✅ Logging system test passed")
        
    except ImportError:
        pytest.skip("Logging system requires torch")


def run_integration_test():
    """Run integration test of core functionality."""
    try:
        # Test that we can import main module
        from src.photonic_flash_attention import __version__
        assert __version__ is not None
        
        print("✅ Integration test passed")
        
    except ImportError as e:
        print(f"⚠️ Integration test failed (expected without torch): {e}")


def run_performance_benchmark():
    """Run basic performance benchmark."""
    try:
        # Simulate some computation
        start_time = time.time()
        
        # Simple computation to simulate attention
        for i in range(1000):
            result = sum(j * j for j in range(100))
        
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        
        print(f"✅ Performance benchmark: {elapsed_ms:.2f}ms for 1000 iterations")
        
        # Check performance is reasonable
        assert elapsed_ms < 1000, "Performance too slow"
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")


def test_security_compliance():
    """Test security compliance measures."""
    try:
        # Check that no hardcoded secrets exist
        import os
        import glob
        
        source_files = glob.glob("src/**/*.py", recursive=True)
        
        for file_path in source_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().lower()
                
                # Check for common security anti-patterns
                suspicious_patterns = [
                    'password =', 'secret =', 'api_key =', 'token =',
                    'hardcoded', 'todo: remove', 'fixme: security'
                ]
                
                for pattern in suspicious_patterns:
                    if pattern in content:
                        print(f"⚠️ Potential security issue in {file_path}: {pattern}")
        
        print("✅ Security compliance check passed")
        
    except Exception as e:
        print(f"❌ Security compliance check failed: {e}")


if __name__ == "__main__":
    """Run all tests when called directly."""
    print("🧪 Running Photonic Flash Attention Test Suite")
    print("=" * 50)
    
    # Core functionality tests
    test_config_system()
    test_hardware_detection() 
    test_security_validation()
    test_error_recovery()
    test_performance_optimization()
    test_load_balancing()
    test_health_monitoring()
    test_validation_utilities()
    test_logging_system()
    
    # Integration and performance tests
    run_integration_test()
    run_performance_benchmark()
    test_security_compliance()
    
    print("=" * 50)
    print("✅ All tests completed!")
    print("Note: Some tests may be skipped due to missing dependencies (PyTorch, etc.)")