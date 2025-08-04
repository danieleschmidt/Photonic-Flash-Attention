"""PyTorch integration for Photonic Flash Attention."""

from .modules import PhotonicFlashAttention, PhotonicMultiHeadAttention
from .convert import convert_to_photonic, PhotonicModelConverter

__all__ = [
    "PhotonicFlashAttention",
    "PhotonicMultiHeadAttention",
    "convert_to_photonic", 
    "PhotonicModelConverter",
]