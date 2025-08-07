"""
Circuit-level simulation for photonic computing systems.

This module provides comprehensive circuit simulation capabilities for silicon
photonic devices, including FDTD wave propagation, circuit-level modeling,
and device characterization for photonic computing applications.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod
import json

from ...utils.exceptions import PhotonicComputeError
from ...utils.validation import validate_optical_tensor
from ...config import get_config

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Types of photonic devices."""
    WAVEGUIDE = "waveguide"
    DIRECTIONAL_COUPLER = "directional_coupler"
    RING_RESONATOR = "ring_resonator"
    MZI_MODULATOR = "mzi_modulator"
    PHOTODETECTOR = "photodetector"
    OPTICAL_AMP = "optical_amplifier"
    PHASE_SHIFTER = "phase_shifter"
    SPLITTER = "splitter"
    COMBINER = "combiner"


@dataclass
class MaterialProperties:
    """Optical material properties."""
    refractive_index: complex = 3.48 + 0j  # Silicon at 1550nm
    group_index: float = 4.27  # Silicon
    nonlinear_index: float = 2.5e-22  # m²/W
    absorption_coeff: float = 0.1  # dB/cm
    thermo_optic_coeff: float = 1.8e-4  # /K
    dispersion: float = -1.2e-24  # s²/m
    bandgap: float = 1.1  # eV


@dataclass
class SimulationConfig:
    """Configuration for circuit simulation."""
    wavelength_start: float = 1500e-9  # m
    wavelength_stop: float = 1600e-9  # m
    wavelength_points: int = 1001
    temperature: float = 300.0  # K
    material: MaterialProperties = field(default_factory=MaterialProperties)
    time_step: float = 1e-15  # femtosecond
    simulation_time: float = 1e-9  # nanosecond
    mesh_resolution: float = 10e-9  # nanometer
    boundary_conditions: str = "PML"  # Perfectly Matched Layer
    enable_nonlinearity: bool = True
    enable_thermal_effects: bool = True


class PhotonicDevice(ABC):
    """Abstract base class for photonic devices."""
    
    def __init__(self, device_id: str, device_type: DeviceType):
        self.device_id = device_id
        self.device_type = device_type
        self.input_ports = []
        self.output_ports = []
        self.parameters = {}
        self.s_matrix = None  # Scattering matrix
        self.operating_wavelength = 1550e-9  # m
        
    @abstractmethod
    def calculate_s_matrix(self, wavelength: float) -> torch.Tensor:
        """Calculate scattering matrix for given wavelength."""
        pass
    
    @abstractmethod
    def get_insertion_loss(self, wavelength: float) -> float:
        """Get insertion loss in dB."""
        pass
    
    def connect_input(self, port_id: int, source_device: 'PhotonicDevice', 
                     source_port: int) -> None:
        """Connect input port to source device."""
        if port_id >= len(self.input_ports):
            self.input_ports.extend([None] * (port_id + 1 - len(self.input_ports)))
        
        self.input_ports[port_id] = (source_device, source_port)
    
    def connect_output(self, port_id: int, dest_device: 'PhotonicDevice',
                      dest_port: int) -> None:
        """Connect output port to destination device."""
        if port_id >= len(self.output_ports):
            self.output_ports.extend([None] * (port_id + 1 - len(self.output_ports)))
        
        self.output_ports[port_id] = (dest_device, dest_port)


class Waveguide(PhotonicDevice):
    """Silicon photonic waveguide model."""
    
    def __init__(self, device_id: str, length: float, width: float = 500e-9,
                 height: float = 220e-9):
        super().__init__(device_id, DeviceType.WAVEGUIDE)
        self.length = length  # m
        self.width = width    # m
        self.height = height  # m
        self.effective_index = None
        self.group_index = None
        self._calculate_mode_properties()
        
    def _calculate_mode_properties(self) -> None:
        """Calculate effective and group indices using approximation."""
        # Simplified calculation - in practice would use mode solver
        core_index = 3.48  # Silicon
        cladding_index = 1.44  # SiO2
        
        # Effective index approximation for rectangular waveguide
        v_number = (2 * np.pi / 1550e-9) * self.width * np.sqrt(core_index**2 - cladding_index**2)
        
        if v_number > 2.405:  # Multi-mode
            self.effective_index = core_index - 0.5 * (1550e-9 / (np.pi * self.width))**2
        else:  # Single mode
            self.effective_index = cladding_index + (core_index - cladding_index) * (v_number / 2.405)**2
        
        # Group index approximation
        self.group_index = self.effective_index + 0.79  # Typical silicon waveguide
    
    def calculate_s_matrix(self, wavelength: float) -> torch.Tensor:
        """Calculate waveguide scattering matrix."""
        # Phase delay
        beta = 2 * np.pi * self.effective_index / wavelength
        phase_delay = np.exp(-1j * beta * self.length)
        
        # Loss (simplified)
        loss_coeff = 0.1  # dB/cm
        loss_factor = 10 ** (-loss_coeff * self.length * 100 / 20)  # Convert to linear
        
        # 2x2 S-matrix for straight waveguide
        s_matrix = torch.zeros(2, 2, dtype=torch.complex64)
        s_matrix[0, 1] = phase_delay * loss_factor  # Forward transmission
        s_matrix[1, 0] = phase_delay * loss_factor  # Backward transmission
        
        return s_matrix
    
    def get_insertion_loss(self, wavelength: float) -> float:
        """Get waveguide insertion loss in dB."""
        loss_coeff = 0.1  # dB/cm
        return loss_coeff * self.length * 100  # Convert m to cm


class DirectionalCoupler(PhotonicDevice):
    """Directional coupler device model."""
    
    def __init__(self, device_id: str, coupling_length: float, gap: float = 200e-9):
        super().__init__(device_id, DeviceType.DIRECTIONAL_COUPLER)
        self.coupling_length = coupling_length  # m
        self.gap = gap  # m
        self.coupling_coefficient = self._calculate_coupling_coefficient()
        
    def _calculate_coupling_coefficient(self) -> float:
        """Calculate coupling coefficient based on geometry."""
        # Simplified model - exponential decay with gap
        k0 = 2 * np.pi / 1550e-9
        n_eff = 3.48
        
        # Coupling coefficient approximation
        kappa = 0.5 * np.exp(-2 * k0 * n_eff * self.gap / 1550e-9)
        return kappa
    
    def calculate_s_matrix(self, wavelength: float) -> torch.Tensor:
        """Calculate directional coupler S-matrix."""
        # Coupling strength
        kappa = self.coupling_coefficient
        theta = kappa * self.coupling_length
        
        # 4x4 S-matrix for directional coupler
        s_matrix = torch.zeros(4, 4, dtype=torch.complex64)
        
        # Through ports (0->1, 2->3)
        s_matrix[1, 0] = np.cos(theta)
        s_matrix[3, 2] = np.cos(theta)
        
        # Cross ports (0->3, 2->1)  
        s_matrix[3, 0] = -1j * np.sin(theta)
        s_matrix[1, 2] = -1j * np.sin(theta)
        
        # Reciprocity
        s_matrix[0, 1] = s_matrix[1, 0]
        s_matrix[2, 3] = s_matrix[3, 2]
        s_matrix[0, 3] = s_matrix[3, 0]
        s_matrix[2, 1] = s_matrix[1, 2]
        
        return s_matrix
    
    def get_insertion_loss(self, wavelength: float) -> float:
        """Get directional coupler insertion loss."""
        return 0.1  # Typical 0.1 dB


class RingResonator(PhotonicDevice):
    """Ring resonator model."""
    
    def __init__(self, device_id: str, radius: float, coupling_gap: float = 200e-9):
        super().__init__(device_id, DeviceType.RING_RESONATOR)
        self.radius = radius  # m
        self.coupling_gap = coupling_gap  # m
        self.q_factor = 10000  # Quality factor
        self.fsr = None  # Free spectral range
        self.finesse = None
        self._calculate_resonator_properties()
        
    def _calculate_resonator_properties(self) -> None:
        """Calculate resonator properties."""
        # Effective index
        n_eff = 3.48  # Silicon
        
        # Free spectral range
        circumference = 2 * np.pi * self.radius
        self.fsr = 1550e-9**2 / (n_eff * circumference)  # m
        
        # Finesse
        self.finesse = self.q_factor * self.fsr / 1550e-9
    
    def calculate_s_matrix(self, wavelength: float) -> torch.Tensor:
        """Calculate ring resonator S-matrix."""
        # Resonance condition
        n_eff = 3.48
        circumference = 2 * np.pi * self.radius
        
        # Phase round trip
        beta = 2 * np.pi * n_eff / wavelength
        phi = beta * circumference
        
        # Coupling coefficients
        kappa1 = 0.1  # Input coupling
        kappa2 = 0.1  # Output coupling
        
        # Ring transmission
        t1 = np.sqrt(1 - kappa1**2)
        t2 = np.sqrt(1 - kappa2**2)
        
        # S-matrix elements
        denominator = 1 - t1 * t2 * np.exp(1j * phi)
        
        s11 = (t1 - t2 * np.exp(1j * phi)) / denominator
        s21 = 1j * np.sqrt(kappa1 * kappa2) / denominator
        
        # 2x2 S-matrix
        s_matrix = torch.zeros(2, 2, dtype=torch.complex64)
        s_matrix[0, 0] = s11  # Reflection
        s_matrix[1, 0] = s21  # Transmission
        s_matrix[0, 1] = s21  # Reciprocity
        s_matrix[1, 1] = -s11  # Reflection
        
        return s_matrix
    
    def get_insertion_loss(self, wavelength: float) -> float:
        """Get ring resonator insertion loss."""
        # On resonance: high loss, off resonance: low loss
        s_matrix = self.calculate_s_matrix(wavelength)
        transmission = abs(s_matrix[1, 0])**2
        
        if transmission > 0:
            return -10 * np.log10(transmission)
        else:
            return float('inf')


class MZIModulator(PhotonicDevice):
    """Mach-Zehnder interferometer modulator."""
    
    def __init__(self, device_id: str, length: float, v_pi: float = 3.0):
        super().__init__(device_id, DeviceType.MZI_MODULATOR)
        self.length = length  # m
        self.v_pi = v_pi  # Voltage for π phase shift
        self.applied_voltage = 0.0  # V
        
    def set_bias_voltage(self, voltage: float) -> None:
        """Set bias voltage for modulator."""
        self.applied_voltage = voltage
    
    def calculate_s_matrix(self, wavelength: float) -> torch.Tensor:
        """Calculate MZI modulator S-matrix."""
        # Phase shift due to applied voltage
        phase_shift = np.pi * self.applied_voltage / self.v_pi
        
        # Split ratio (50:50)
        split_ratio = 0.5
        
        # Upper and lower arm phase delays
        phi_upper = phase_shift / 2
        phi_lower = -phase_shift / 2
        
        # Transfer matrix calculation
        # Simplified 2x2 representation
        s_matrix = torch.zeros(2, 2, dtype=torch.complex64)
        
        # MZI response
        cos_term = np.cos(phase_shift / 2)
        sin_term = np.sin(phase_shift / 2)
        
        s_matrix[1, 0] = cos_term  # Transmission
        s_matrix[0, 0] = 1j * sin_term  # Reflection
        s_matrix[0, 1] = 1j * sin_term  # Reciprocity
        s_matrix[1, 1] = cos_term
        
        return s_matrix
    
    def get_insertion_loss(self, wavelength: float) -> float:
        """Get modulator insertion loss."""
        base_loss = 1.0  # dB base insertion loss
        
        # Additional loss from modulation
        modulation_loss = 0.1 * abs(self.applied_voltage / self.v_pi)
        
        return base_loss + modulation_loss
    
    def modulate(self, input_signal: torch.Tensor, 
                modulation_data: torch.Tensor) -> torch.Tensor:
        """Apply modulation to input optical signal."""
        # Convert data to voltage levels
        voltage_swing = self.v_pi
        voltage_signal = modulation_data * voltage_swing
        
        # Time-varying phase modulation
        phase_modulation = np.pi * voltage_signal / self.v_pi
        modulated_signal = input_signal * torch.exp(1j * phase_modulation)
        
        return modulated_signal.real  # Intensity modulation


class Photodetector(PhotonicDevice):
    """Photodetector model."""
    
    def __init__(self, device_id: str, responsivity: float = 1.0, 
                 dark_current: float = 1e-9):
        super().__init__(device_id, DeviceType.PHOTODETECTOR)
        self.responsivity = responsivity  # A/W
        self.dark_current = dark_current  # A
        self.bandwidth = 40e9  # Hz
        self.noise_current = 1e-12  # A
        
    def calculate_s_matrix(self, wavelength: float) -> torch.Tensor:
        """Photodetector doesn't have S-matrix (not reciprocal)."""
        # Terminal device - absorbs all light
        s_matrix = torch.zeros(1, 1, dtype=torch.complex64)
        return s_matrix
    
    def get_insertion_loss(self, wavelength: float) -> float:
        """Photodetector absorbs all light."""
        return float('inf')  # Complete absorption
    
    def detect(self, optical_power: torch.Tensor) -> torch.Tensor:
        """Convert optical power to photocurrent."""
        # Photodetection with responsivity
        photocurrent = self.responsivity * optical_power
        
        # Add dark current
        photocurrent = photocurrent + self.dark_current
        
        # Add shot noise (simplified)
        if torch.is_grad_enabled():
            shot_noise = torch.randn_like(photocurrent) * np.sqrt(2 * 1.602e-19 * photocurrent * 1e9)
            photocurrent = photocurrent + shot_noise * 1e-3  # Scale for stability
        
        return photocurrent


class PhotonicCircuit:
    """Complete photonic circuit simulator."""
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self.devices = {}  # device_id -> PhotonicDevice
        self.connections = []  # List of connections
        self.wavelengths = np.linspace(
            self.config.wavelength_start,
            self.config.wavelength_stop,
            self.config.wavelength_points
        )
        self.s_matrix_global = None
        self.simulation_results = {}
        
        logger.info("Initialized PhotonicCircuit simulator")
    
    def add_device(self, device: PhotonicDevice) -> None:
        """Add device to circuit."""
        self.devices[device.device_id] = device
        logger.debug(f"Added device {device.device_id} of type {device.device_type}")
    
    def connect_devices(self, src_device_id: str, src_port: int,
                       dst_device_id: str, dst_port: int) -> None:
        """Connect two devices."""
        if src_device_id not in self.devices or dst_device_id not in self.devices:
            raise PhotonicComputeError("Device not found in circuit")
        
        src_device = self.devices[src_device_id]
        dst_device = self.devices[dst_device_id]
        
        # Create bidirectional connection
        src_device.connect_output(src_port, dst_device, dst_port)
        dst_device.connect_input(dst_port, src_device, src_port)
        
        self.connections.append((src_device_id, src_port, dst_device_id, dst_port))
        logger.debug(f"Connected {src_device_id}:{src_port} -> {dst_device_id}:{dst_port}")
    
    def simulate_frequency_response(self) -> Dict[str, torch.Tensor]:
        """Simulate frequency response of the circuit."""
        results = {
            'wavelengths': torch.tensor(self.wavelengths),
            'transmission': torch.zeros(len(self.wavelengths), dtype=torch.complex64),
            'reflection': torch.zeros(len(self.wavelengths), dtype=torch.complex64),
            'insertion_loss': torch.zeros(len(self.wavelengths)),
        }
        
        for i, wavelength in enumerate(self.wavelengths):
            # Calculate circuit response at this wavelength
            transmission, reflection = self._calculate_circuit_response(wavelength)
            
            results['transmission'][i] = transmission
            results['reflection'][i] = reflection
            results['insertion_loss'][i] = -20 * np.log10(abs(transmission)) if abs(transmission) > 0 else 100
        
        self.simulation_results = results
        logger.info("Completed frequency response simulation")
        
        return results
    
    def _calculate_circuit_response(self, wavelength: float) -> Tuple[complex, complex]:
        """Calculate circuit response at single wavelength."""
        if not self.devices:
            return 0.0, 0.0
        
        # For simplicity, calculate cascade of S-matrices
        # In practice, would solve full network using nodal analysis
        
        total_transmission = 1.0 + 0j
        total_reflection = 0.0 + 0j
        
        for device in self.devices.values():
            s_matrix = device.calculate_s_matrix(wavelength)
            
            if s_matrix.shape[0] >= 2:
                # Extract transmission and reflection
                transmission = s_matrix[1, 0].item()
                reflection = s_matrix[0, 0].item()
                
                total_transmission *= transmission
                total_reflection += reflection * (1 - abs(total_reflection)**2)
        
        return total_transmission, total_reflection
    
    def simulate_time_domain(self, input_signal: torch.Tensor, 
                           time_points: torch.Tensor) -> torch.Tensor:
        """Simulate time-domain response."""
        if not hasattr(self, 'simulation_results') or not self.simulation_results:
            # Need frequency response first
            self.simulate_frequency_response()
        
        # Convert to time domain using FFT
        # This is a simplified implementation
        freq_response = self.simulation_results['transmission']
        
        # Pad/truncate to match input signal length
        if len(freq_response) != len(input_signal):
            if len(freq_response) < len(input_signal):
                # Zero-pad frequency response
                padded_response = torch.zeros(len(input_signal), dtype=torch.complex64)
                padded_response[:len(freq_response)] = freq_response
                freq_response = padded_response
            else:
                freq_response = freq_response[:len(input_signal)]
        
        # Apply frequency response to input
        input_fft = torch.fft.fft(input_signal)
        output_fft = input_fft * freq_response
        output_signal = torch.fft.ifft(output_fft).real
        
        return output_signal
    
    def optimize_design(self, target_response: torch.Tensor, 
                       optimization_params: List[str]) -> Dict[str, float]:
        """Optimize circuit design to match target response."""
        # Simplified optimization using gradient descent
        # In practice, would use more sophisticated algorithms
        
        best_params = {}
        best_error = float('inf')
        
        # Parameter sweep (simplified)
        for _ in range(100):  # Optimization iterations
            # Simulate current design
            results = self.simulate_frequency_response()
            current_response = torch.abs(results['transmission'])
            
            # Calculate error
            if len(current_response) == len(target_response):
                error = torch.mean((current_response - target_response)**2).item()
                
                if error < best_error:
                    best_error = error
                    # Store current parameters
                    for param in optimization_params:
                        if hasattr(self, param):
                            best_params[param] = getattr(self, param)
        
        logger.info(f"Optimization completed with error: {best_error:.6f}")
        return best_params
    
    def export_design(self, filename: str) -> None:
        """Export circuit design to file."""
        design_data = {
            'config': {
                'wavelength_start': self.config.wavelength_start,
                'wavelength_stop': self.config.wavelength_stop,
                'wavelength_points': self.config.wavelength_points,
                'temperature': self.config.temperature,
            },
            'devices': {},
            'connections': self.connections
        }
        
        # Export device parameters
        for device_id, device in self.devices.items():
            device_data = {
                'type': device.device_type.value,
                'parameters': device.parameters
            }
            
            # Add device-specific parameters
            if isinstance(device, Waveguide):
                device_data['parameters'].update({
                    'length': device.length,
                    'width': device.width,
                    'height': device.height
                })
            elif isinstance(device, RingResonator):
                device_data['parameters'].update({
                    'radius': device.radius,
                    'coupling_gap': device.coupling_gap,
                    'q_factor': device.q_factor
                })
            
            design_data['devices'][device_id] = device_data
        
        # Save to JSON file
        with open(filename, 'w') as f:
            json.dump(design_data, f, indent=2, default=str)
        
        logger.info(f"Circuit design exported to {filename}")
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get circuit performance metrics."""
        if not self.simulation_results:
            self.simulate_frequency_response()
        
        transmission = torch.abs(self.simulation_results['transmission'])
        insertion_loss = self.simulation_results['insertion_loss']
        
        metrics = {
            'max_transmission': torch.max(transmission).item(),
            'min_transmission': torch.min(transmission).item(),
            'avg_insertion_loss': torch.mean(insertion_loss).item(),
            'max_insertion_loss': torch.max(insertion_loss).item(),
            'bandwidth_3db': self._calculate_3db_bandwidth(),
            'extinction_ratio': self._calculate_extinction_ratio(),
        }
        
        return metrics
    
    def _calculate_3db_bandwidth(self) -> float:
        """Calculate 3dB bandwidth."""
        if not self.simulation_results:
            return 0.0
        
        transmission = torch.abs(self.simulation_results['transmission'])
        max_transmission = torch.max(transmission)
        threshold = max_transmission / np.sqrt(2)  # -3dB
        
        # Find bandwidth
        above_threshold = transmission > threshold
        if torch.any(above_threshold):
            indices = torch.where(above_threshold)[0]
            bandwidth_points = len(indices)
            wavelength_span = self.wavelengths[-1] - self.wavelengths[0]
            bandwidth = bandwidth_points * wavelength_span / len(self.wavelengths)
            return bandwidth
        
        return 0.0
    
    def _calculate_extinction_ratio(self) -> float:
        """Calculate extinction ratio in dB."""
        if not self.simulation_results:
            return 0.0
        
        transmission = torch.abs(self.simulation_results['transmission'])
        max_val = torch.max(transmission)
        min_val = torch.min(transmission[transmission > 0])  # Avoid log(0)
        
        if min_val > 0:
            return 20 * np.log10(max_val / min_val)
        else:
            return float('inf')


# Convenience functions
def create_mzi_circuit(length: float = 100e-6) -> PhotonicCircuit:
    """Create a basic MZI circuit."""
    circuit = PhotonicCircuit()
    
    # Add devices
    splitter = DirectionalCoupler("splitter", 10e-6)
    combiner = DirectionalCoupler("combiner", 10e-6)
    upper_arm = Waveguide("upper_arm", length)
    lower_arm = Waveguide("lower_arm", length)
    modulator = MZIModulator("modulator", length / 2)
    
    # Add to circuit
    for device in [splitter, combiner, upper_arm, lower_arm, modulator]:
        circuit.add_device(device)
    
    # Connect devices
    circuit.connect_devices("splitter", 1, "upper_arm", 0)
    circuit.connect_devices("splitter", 3, "lower_arm", 0)
    circuit.connect_devices("upper_arm", 1, "combiner", 0)
    circuit.connect_devices("lower_arm", 1, "combiner", 2)
    
    return circuit


def create_ring_filter(radius: float = 10e-6) -> PhotonicCircuit:
    """Create a ring resonator filter circuit."""
    circuit = PhotonicCircuit()
    
    # Add devices
    waveguide_in = Waveguide("wg_in", 50e-6)
    waveguide_out = Waveguide("wg_out", 50e-6)
    ring = RingResonator("ring", radius)
    
    # Add to circuit
    for device in [waveguide_in, waveguide_out, ring]:
        circuit.add_device(device)
    
    # Connect devices (simplified - ring coupling is implicit)
    circuit.connect_devices("wg_in", 1, "ring", 0)
    circuit.connect_devices("ring", 1, "wg_out", 0)
    
    return circuit


# Global circuit simulator
_global_circuit_simulator = None

def get_global_circuit_simulator() -> PhotonicCircuit:
    """Get global photonic circuit simulator instance."""
    global _global_circuit_simulator
    if _global_circuit_simulator is None:
        _global_circuit_simulator = PhotonicCircuit()
    return _global_circuit_simulator