"""Framework integration modules."""

from .pytorch.modules import PhotonicFlashAttention
from .pytorch.convert import convert_to_photonic

__all__ = [
    "PhotonicFlashAttention",
    "convert_to_photonic",
]