#!/usr/bin/env python3
"""
Basic integration test without external dependencies.

Tests core functionality of the photonic attention system using
built-in Python modules only.
"""

import sys
import os
import unittest
from unittest.mock import Mock, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Mock torch completely
torch_mock = MagicMock()
torch_mock.Tensor = MagicMock
torch_mock.nn = MagicMock()
torch_mock.nn.Module = object
torch_mock.nn.Linear = MagicMock
torch_mock.nn.Dropout = MagicMock
torch_mock.cuda = MagicMock()
torch_mock.cuda.is_available.return_value = False
torch_mock.__version__ = "2.0.0"

# Mock numpy
numpy_mock = MagicMock()
numpy_mock.random = MagicMock()
numpy_mock.random.randn = lambda *args: [[1.0] * args[-1] for _ in range(args[0])]
numpy_mock.pi = 3.14159
numpy_mock.sqrt = lambda x: x ** 0.5
numpy_mock.array = lambda x: x

# Mock psutil for monitoring
psutil_mock = MagicMock()
psutil_mock.cpu_count.return_value = 4
psutil_mock.virtual_memory.return_value = MagicMock(total=16*1024**3, available=8*1024**3)
psutil_mock.disk_usage.return_value = MagicMock(total=100*1024**3, free=50*1024**3)

# Apply mocks
sys.modules['torch'] = torch_mock
sys.modules['torch.nn'] = torch_mock.nn  
sys.modules['torch.nn.functional'] = MagicMock()
sys.modules['numpy'] = numpy_mock
sys.modules['np'] = numpy_mock
sys.modules['psutil'] = psutil_mock

class TestBasicIntegration(unittest.TestCase):
    """Basic integration tests."""
    
    def test_hardware_detection_import(self):
        """Test hardware detection can be imported."""
        try:
            from photonic_flash_attention.photonic.hardware.detection import (
                PhotonicDevice, detect_photonic_hardware
            )
            self.assertTrue(True)  # Import successful
        except ImportError as e:
            self.fail(f"Hardware detection import failed: {e}")
    
    def test_device_creation(self):
        """Test PhotonicDevice can be created."""
        from photonic_flash_attention.photonic.hardware.detection import PhotonicDevice
        
        device = PhotonicDevice(
            device_id="test:0",
            device_type="simulation", 
            vendor="Test",
            model="Test Model",
            wavelengths=80,
            max_optical_power=0.01
        )
        
        self.assertEqual(device.device_id, "test:0")
        self.assertEqual(device.wavelengths, 80)
        self.assertTrue(device.is_available)
    
    def test_simulation_mode_detection(self):
        """Test simulation mode device detection."""
        os.environ["PHOTONIC_SIMULATION"] = "true"
        
        try:
            from photonic_flash_attention.photonic.hardware.detection import (
                detect_photonic_hardware, get_best_photonic_device
            )
            
            # Should detect at least simulation device
            has_devices = detect_photonic_hardware()
            self.assertTrue(has_devices)
            
            # Should get simulation device
            device = get_best_photonic_device()
            self.assertIsNotNone(device)
            self.assertEqual(device.device_type, "simulation")
            
        finally:
            os.environ.pop("PHOTONIC_SIMULATION", None)
    
    def test_config_import(self):
        """Test configuration can be imported."""
        try:
            from photonic_flash_attention.config import get_config
            config = get_config()
            self.assertIsNotNone(config)
        except ImportError as e:
            self.fail(f"Config import failed: {e}")
    
    def test_exceptions_import(self):
        """Test exceptions can be imported and used."""
        try:
            from photonic_flash_attention.utils.exceptions import (
                PhotonicHardwareError, PhotonicComputationError
            )
            
            # Test exception creation
            error = PhotonicHardwareError("Test error")
            self.assertIn("Test error", str(error))
            
            comp_error = PhotonicComputationError("Computation failed")
            self.assertIn("Computation failed", str(comp_error))
            
        except ImportError as e:
            self.fail(f"Exceptions import failed: {e}")
    
    def test_main_package_import(self):
        """Test main package can be imported."""
        try:
            import photonic_flash_attention
            self.assertTrue(hasattr(photonic_flash_attention, '__version__'))
        except ImportError as e:
            self.fail(f"Main package import failed: {e}")
    
    def test_optical_kernel_imports(self):
        """Test optical kernels can be imported."""
        try:
            from photonic_flash_attention.photonic.optical_kernels.matrix_mult import (
                OpticalMatMulConfig, WavelengthBank
            )
            from photonic_flash_attention.photonic.optical_kernels.nonlinearity import (
                OpticalNonlinearityConfig, NonlinearityType
            )
            
            # Test config creation
            config = OpticalMatMulConfig(n_wavelengths=64)
            self.assertEqual(config.n_wavelengths, 64)
            
            # Test enum values
            self.assertEqual(NonlinearityType.SOFTMAX.value, "softmax")
            
        except ImportError as e:
            self.fail(f"Optical kernel imports failed: {e}")
    
    def test_wavelength_bank(self):
        """Test wavelength bank functionality.""" 
        from photonic_flash_attention.photonic.optical_kernels.matrix_mult import WavelengthBank
        
        bank = WavelengthBank(n_wavelengths=10)
        self.assertEqual(bank.n_wavelengths, 10)
        
        # Test channel allocation
        channels = bank.allocate_channels(3)
        self.assertEqual(len(channels), 3)
        self.assertTrue(all(isinstance(ch, int) for ch in channels))
        
        # Test channel release
        bank.release_channels(channels)
        self.assertEqual(len(bank.allocated_channels), 0)
    
    def test_monitoring_import(self):
        """Test monitoring systems can be imported."""
        try:
            from photonic_flash_attention.monitoring.health_monitor import HealthMonitor
            from photonic_flash_attention.optimization.performance_optimizer import PerformanceOptimizer
            
            # Should be importable
            self.assertTrue(True)
            
        except ImportError as e:
            self.fail(f"Monitoring imports failed: {e}")


def run_tests():
    """Run basic integration tests."""
    print("Running Basic Integration Tests...")
    print("=" * 50)
    
    # Enable simulation mode
    os.environ["PHOTONIC_SIMULATION"] = "true"
    
    try:
        # Create test suite
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(TestBasicIntegration)
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Print summary
        print("\n" + "=" * 50)
        if result.wasSuccessful():
            print(f"✅ All {result.testsRun} tests passed!")
            return True
        else:
            print(f"❌ {len(result.failures)} failures, {len(result.errors)} errors")
            
            # Print failures
            for test, traceback in result.failures:
                print(f"\nFAILURE: {test}")
                print(traceback)
            
            # Print errors
            for test, traceback in result.errors:
                print(f"\nERROR: {test}")
                print(traceback)
            
            return False
    
    finally:
        os.environ.pop("PHOTONIC_SIMULATION", None)


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)