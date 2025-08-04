"""Optical computing kernels for photonic attention."""

from .matrix_mult import OpticalMatrixMultiply
from .nonlinearity import OpticalSoftmax, OpticalActivation
from .interconnect import PhotonicInterconnect

__all__ = [
    "OpticalMatrixMultiply",
    "OpticalSoftmax",
    "OpticalActivation", 
    "PhotonicInterconnect",
]