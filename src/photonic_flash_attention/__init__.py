"""
Photonic Flash Attention
========================

A Flash Attention implementation with photonic hardware stubs.

Classes
-------
FlashAttentionChunk
    Tiled scaled dot-product attention with O(N) memory via chunked computation.

PhotonicMatmul
    Matrix multiply with Mach-Zehnder Interferometer (MZI) mesh hardware stubs.

BenchmarkSuite
    Compare standard vs Flash Attention across sequence lengths.
"""

from .flash_attention import FlashAttentionChunk
from .photonic_matmul import PhotonicMatmul
from .benchmark import BenchmarkSuite

__all__ = ["FlashAttentionChunk", "PhotonicMatmul", "BenchmarkSuite"]
__version__ = "0.1.0"
