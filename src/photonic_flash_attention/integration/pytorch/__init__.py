"""PyTorch integration for Photonic Flash Attention."""

from .modules import PhotonicFlashAttention, PhotonicMultiHeadAttention
from .convert import convert_to_photonic, ModelConverter

# Alias for backward compatibility
PhotonicModelConverter = ModelConverter

__all__ = [
    "PhotonicFlashAttention",
    "PhotonicMultiHeadAttention",
    "convert_to_photonic", 
    "PhotonicModelConverter",
    "ModelConverter",
]