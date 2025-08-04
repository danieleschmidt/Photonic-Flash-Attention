"""Photonic computing components."""

from .optical_kernels import OpticalMatrixMultiply, OpticalSoftmax
from .hardware.detection import detect_photonic_hardware
from .simulation.circuit import PhotonicCircuitSimulator

__all__ = [
    "OpticalMatrixMultiply",
    "OpticalSoftmax", 
    "detect_photonic_hardware",
    "PhotonicCircuitSimulator",
]