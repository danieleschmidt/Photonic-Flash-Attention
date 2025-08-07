"""
Photonic simulation package for circuit-level modeling and analysis.

This package provides comprehensive simulation tools for silicon photonic
devices and circuits, including wave propagation, device modeling, and
system-level performance analysis.
"""

from .circuit import (
    PhotonicCircuit,
    SimulationConfig,
    DeviceType,
    MaterialProperties,
    PhotonicDevice,
    Waveguide,
    DirectionalCoupler,
    RingResonator,
    MZIModulator,
    Photodetector,
    create_mzi_circuit,
    create_ring_filter,
    get_global_circuit_simulator
)

__all__ = [
    "PhotonicCircuit",
    "SimulationConfig", 
    "DeviceType",
    "MaterialProperties",
    "PhotonicDevice",
    "Waveguide",
    "DirectionalCoupler", 
    "RingResonator",
    "MZIModulator",
    "Photodetector",
    "create_mzi_circuit",
    "create_ring_filter",
    "get_global_circuit_simulator"
]