"""Core attention implementations."""

# Import PyTorch-dependent modules conditionally
try:
    from .flash_attention_3 import FlashAttention3
    from .photonic_attention import PhotonicAttention
    from .hybrid_router import HybridFlashAttention, AdaptiveRouter
    from .memory_manager import UnifiedMemoryManager
except ImportError:
    # PyTorch not available
    FlashAttention3 = None
    PhotonicAttention = None
    HybridFlashAttention = None
    AdaptiveRouter = None
    UnifiedMemoryManager = None

__all__ = [
    "FlashAttention3",
    "PhotonicAttention", 
    "HybridFlashAttention",
    "AdaptiveRouter",
    "UnifiedMemoryManager",
]