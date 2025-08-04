"""
Photonic Flash Attention - Hybrid photonic-electronic attention implementation.

This library provides a seamless drop-in replacement for Flash-Attention that
automatically switches between optical and electronic computation based on
sequence length and hardware availability.
"""

from .core.flash_attention_3 import FlashAttention3
from .core.photonic_attention import PhotonicAttention
from .core.hybrid_router import HybridFlashAttention
from .integration.pytorch.modules import PhotonicFlashAttention
from .integration.pytorch.convert import convert_to_photonic

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
    import torch
    from .photonic.hardware.detection import detect_photonic_hardware
    
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "photonic_available": detect_photonic_hardware(),
        "version": __version__,
    }
    
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    
    return info


def set_global_config(**kwargs) -> None:
    """Set global configuration options."""
    from .config import GlobalConfig
    GlobalConfig.update(**kwargs)