"""
Photonic Flash Attention - Hybrid photonic-electronic attention implementation.

This library provides a seamless drop-in replacement for Flash-Attention that
automatically switches between optical and electronic computation based on
sequence length and hardware availability.
"""

# Core imports that require PyTorch - import conditionally
try:
    from .core.flash_attention_3 import FlashAttention3
    from .core.photonic_attention import PhotonicAttention
    from .core.hybrid_router import HybridFlashAttention
    from .integration.pytorch.modules import PhotonicFlashAttention
    from .integration.pytorch.convert import convert_to_photonic
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    # Define placeholder classes/functions for when PyTorch is not available
    FlashAttention3 = None
    PhotonicAttention = None
    HybridFlashAttention = None
    PhotonicFlashAttention = None
    convert_to_photonic = None

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.ai"

__all__ = [
    "PhotonicFlashAttention",
    "FlashAttention3", 
    "PhotonicAttention",
    "HybridFlashAttention",
    "convert_to_photonic",
]


def get_version() -> str:
    """Get the current version of the library."""
    return __version__


def get_device_info() -> dict:
    """Get information about available computation devices."""
    from .photonic.hardware.detection import detect_photonic_hardware
    
    info = {
        "photonic_available": detect_photonic_hardware(),
        "version": __version__,
    }
    
    # Add CUDA info if PyTorch is available
    if _TORCH_AVAILABLE:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    else:
        info["cuda_available"] = False
        info["cuda_device_count"] = 0
    
    return info


def set_global_config(**kwargs) -> None:
    """Set global configuration options."""
    from .config import GlobalConfig
    GlobalConfig.update(**kwargs)