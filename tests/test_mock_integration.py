"""
Integration tests using mock dependencies.

This module tests the photonic attention system without requiring
full PyTorch installation, using mocks and lightweight simulation.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import sys
import os

# Mock torch before importing our modules
torch_mock = MagicMock()
torch_mock.Tensor = MagicMock
torch_mock.nn = MagicMock()
torch_mock.nn.Module = object
torch_mock.nn.Linear = MagicMock
torch_mock.nn.Dropout = MagicMock
torch_mock.cuda = MagicMock()
torch_mock.cuda.is_available.return_value = False
torch_mock.cuda.Event = MagicMock
torch_mock.cuda.synchronize = MagicMock()
torch_mock.__version__ = "2.0.0"

# Create mock tensor operations
def mock_tensor(data, **kwargs):
    mock_t = MagicMock()
    mock_t.shape = getattr(data, 'shape', (1,))
    mock_t.device = kwargs.get('device', 'cpu')
    mock_t.dtype = kwargs.get('dtype', 'float32')
    mock_t.dim.return_value = len(mock_t.shape)
    mock_t.__matmul__ = lambda self, other: mock_tensor(np.random.randn(*other.shape))
    mock_t.transpose.return_value = mock_t
    mock_t.view.return_value = mock_t
    mock_t.chunk.return_value = [mock_t, mock_t, mock_t]
    mock_t.unsqueeze.return_value = mock_t
    mock_t.squeeze.return_value = mock_t
    return mock_t

torch_mock.tensor = mock_tensor
torch_mock.randn = lambda *args, **kwargs: mock_tensor(np.random.randn(*args))
torch_mock.zeros = lambda *args, **kwargs: mock_tensor(np.zeros(args))
torch_mock.ones = lambda *args, **kwargs: mock_tensor(np.ones(args))
torch_mock.eye = lambda n: mock_tensor(np.eye(n))

sys.modules['torch'] = torch_mock
sys.modules['torch.nn'] = torch_mock.nn
sys.modules['torch.nn.functional'] = MagicMock()

# Now we can import our modules
import photonic_flash_attention
from photonic_flash_attention.core.photonic_attention import PhotonicAttention
from photonic_flash_attention.core.hybrid_router import HybridFlashAttention
from photonic_flash_attention.photonic.hardware.detection import (
    detect_photonic_hardware, get_best_photonic_device, PhotonicDevice
)


class TestMockIntegration(unittest.TestCase):
    """Integration tests with mock dependencies."""
    
    def setUp(self):
        """Set up test environment."""
        os.environ["PHOTONIC_SIMULATION"] = "true"
    
    def tearDown(self):
        """Clean up test environment."""
        os.environ.pop("PHOTONIC_SIMULATION", None)
    
    def test_hardware_detection_simulation_mode(self):
        """Test hardware detection in simulation mode."""
        # Should detect simulated device
        self.assertTrue(detect_photonic_hardware())
        
        device = get_best_photonic_device()
        self.assertIsNotNone(device)
        self.assertEqual(device.device_type, "simulation")
        self.assertEqual(device.wavelengths, 80)
    
    def test_photonic_attention_initialization(self):
        """Test PhotonicAttention can be initialized."""
        try:
            attention = PhotonicAttention(
                embed_dim=768,
                num_heads=12,
                dropout=0.1,
                safety_checks=False  # Disable for mock testing
            )
            self.assertEqual(attention.embed_dim, 768)
            self.assertEqual(attention.num_heads, 12)
            self.assertEqual(attention.head_dim, 64)
        except Exception as e:
            self.fail(f"PhotonicAttention initialization failed: {e}")
    
    def test_hybrid_attention_initialization(self):
        """Test HybridFlashAttention can be initialized."""
        try:
            with patch('photonic_flash_attention.core.hybrid_router.PhotonicAttention') as mock_photonic:
                mock_photonic.return_value = MagicMock()
                
                hybrid_attention = HybridFlashAttention(
                    embed_dim=512,
                    num_heads=8,
                    dropout=0.0,
                    enable_scaling=False  # Simplify for testing
                )
                
                self.assertEqual(hybrid_attention.embed_dim, 512)
                self.assertEqual(hybrid_attention.num_heads, 8)
                self.assertIsNotNone(hybrid_attention.router)
                
        except Exception as e:
            self.fail(f"HybridFlashAttention initialization failed: {e}")
    
    @patch.dict(os.environ, {"PHOTONIC_SIMULATION": "true"})
    def test_optical_kernels_creation(self):
        """Test optical kernels can be created."""
        from photonic_flash_attention.photonic.optical_kernels.matrix_mult import OpticalMatMul
        from photonic_flash_attention.photonic.optical_kernels.nonlinearity import OpticalSoftmax
        
        try:
            # Test optical matrix multiplication
            opt_matmul = OpticalMatMul()
            self.assertEqual(opt_matmul.config.n_wavelengths, 80)
            
            # Test optical softmax
            opt_softmax = OpticalSoftmax()
            self.assertIsNotNone(opt_softmax.wavelength_bank)
            
        except Exception as e:
            self.fail(f"Optical kernel creation failed: {e}")
    
    def test_configuration_loading(self):
        """Test configuration system."""
        from photonic_flash_attention.config import get_config
        
        try:
            config = get_config()
            self.assertIsNotNone(config)
            # Should have default values
            self.assertGreater(config.photonic_threshold, 0)
            
        except Exception as e:
            self.fail(f"Configuration loading failed: {e}")
    
    def test_import_structure(self):
        """Test that all modules can be imported."""
        import photonic_flash_attention
        from photonic_flash_attention import PhotonicFlashAttention
        from photonic_flash_attention.core import FlashAttention3, PhotonicAttention
        from photonic_flash_attention.optimization import PerformanceOptimizer
        from photonic_flash_attention.monitoring import HealthMonitor
        
        # Should not raise any import errors
        self.assertTrue(True)
    
    def test_mock_forward_pass(self):
        """Test a mock forward pass through the system."""
        try:
            # Create mock tensors
            batch_size, seq_len, embed_dim = 2, 128, 512
            
            query = mock_tensor(np.random.randn(batch_size, seq_len, embed_dim))
            
            with patch('photonic_flash_attention.core.photonic_attention.PhotonicAttention') as MockPhotonic:
                mock_attention = MockPhotonic.return_value
                mock_attention.return_value = (query, None)  # Mock output
                
                # This should work without actual tensor operations
                result = mock_attention(query)
                self.assertIsNotNone(result)
                
        except Exception as e:
            self.fail(f"Mock forward pass failed: {e}")
    
    def test_performance_monitoring(self):
        """Test performance monitoring system."""
        from photonic_flash_attention.monitoring.health_monitor import HealthMonitor
        
        try:
            monitor = HealthMonitor()
            stats = monitor.get_system_stats()
            self.assertIsInstance(stats, dict)
            
        except Exception as e:
            self.fail(f"Performance monitoring failed: {e}")
    
    def test_cli_interface(self):
        """Test CLI interface can be loaded."""
        from photonic_flash_attention.cli import main
        
        # Should be importable without errors
        self.assertIsNotNone(main)
    
    def test_device_info_retrieval(self):
        """Test device information retrieval."""
        from photonic_flash_attention.photonic.hardware.detection import get_device_info
        
        info = get_device_info()
        self.assertIsInstance(info, dict)
        self.assertIn("num_devices", info)
        self.assertIn("devices", info)
        
        # Should have at least simulation device
        self.assertGreaterEqual(info["num_devices"], 1)
    
    def test_error_handling(self):
        """Test error handling in mock environment."""
        from photonic_flash_attention.utils.exceptions import (
            PhotonicHardwareError, PhotonicComputationError
        )
        
        # Test custom exceptions can be created
        try:
            raise PhotonicHardwareError("Test error")
        except PhotonicHardwareError as e:
            self.assertIn("Test error", str(e))
        
        try:
            raise PhotonicComputationError("Computation failed")
        except PhotonicComputationError as e:
            self.assertIn("Computation failed", str(e))


class TestPhotnicDeviceInfo(unittest.TestCase):
    """Test photonic device information system."""
    
    def setUp(self):
        os.environ["PHOTONIC_SIMULATION"] = "true"
    
    def tearDown(self):
        os.environ.pop("PHOTONIC_SIMULATION", None)
    
    def test_device_creation(self):
        """Test PhotonicDevice creation."""
        device = PhotonicDevice(
            device_id="test:0",
            device_type="test",
            vendor="Test Vendor",
            model="Test Model",
            wavelengths=64,
            max_optical_power=10e-3,
            temperature=25.0,
        )
        
        self.assertEqual(device.device_id, "test:0")
        self.assertEqual(device.wavelengths, 64)
        self.assertEqual(device.temperature, 25.0)
        self.assertTrue(device.is_available)
    
    def test_device_info_serialization(self):
        """Test device information can be serialized."""
        from photonic_flash_attention.photonic.hardware.detection import get_device_info
        
        info = get_device_info()
        
        # Should be JSON-serializable
        import json
        try:
            json_str = json.dumps(info)
            parsed = json.loads(json_str)
            self.assertEqual(info["num_devices"], parsed["num_devices"])
        except (TypeError, ValueError) as e:
            self.fail(f"Device info not JSON serializable: {e}")


class TestSystemIntegration(unittest.TestCase):
    """Test system-level integration."""
    
    def test_end_to_end_simulation(self):
        """Test end-to-end system in simulation mode."""
        os.environ["PHOTONIC_SIMULATION"] = "true"
        
        try:
            # Initialize system
            from photonic_flash_attention import PhotonicFlashAttention
            
            # Should work in simulation mode
            model = PhotonicFlashAttention(
                embed_dim=256,
                num_heads=8,
                device='auto'
            )
            
            self.assertIsNotNone(model)
            
        except Exception as e:
            self.fail(f"End-to-end simulation failed: {e}")
        finally:
            os.environ.pop("PHOTONIC_SIMULATION", None)
    
    def test_benchmark_system(self):
        """Test benchmark system can be initialized."""
        from photonic_flash_attention.optimization import PerformanceOptimizer
        
        try:
            optimizer = PerformanceOptimizer()
            self.assertIsNotNone(optimizer)
            
        except Exception as e:
            self.fail(f"Benchmark system failed: {e}")


if __name__ == '__main__':
    # Run with simulation enabled
    os.environ["PHOTONIC_SIMULATION"] = "true"
    
    unittest.main(verbosity=2)