"""Photonic hardware detection and management."""

import os
import sys
import subprocess
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class PhotonicDevice:
    """Information about a detected photonic device."""
    device_id: str
    device_type: str
    vendor: str
    model: str
    wavelengths: int
    max_optical_power: float  # Watts
    temperature: Optional[float] = None  # Celsius
    is_available: bool = True
    driver_version: Optional[str] = None


class PhotonicHardwareDetector:
    """Detects and manages photonic computing hardware."""
    
    def __init__(self):
        self._devices: List[PhotonicDevice] = []
        self._detection_methods = [
            self._detect_lightmatter,
            self._detect_luminous,
            self._detect_generic_pcie,
            self._detect_simulation_mode,
        ]
    
    def detect_all_devices(self) -> List[PhotonicDevice]:
        """Detect all available photonic devices."""
        self._devices.clear()
        
        for detection_method in self._detection_methods:
            try:
                devices = detection_method()
                self._devices.extend(devices)
            except Exception as e:
                # Continue with other detection methods
                pass
        
        return self._devices.copy()
    
    def _detect_lightmatter(self) -> List[PhotonicDevice]:
        """Detect LightMatter Mars photonic accelerators."""
        devices = []
        
        # Check for LightMatter driver
        try:
            result = subprocess.run(
                ["lspci", "-d", "1234:5678"],  # LightMatter vendor:device ID (example)
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode == 0 and result.stdout.strip():
                # Parse lspci output for LightMatter devices
                lines = result.stdout.strip().split('\n')
                for i, line in enumerate(lines):
                    device = PhotonicDevice(
                        device_id=f"lightmatter:{i}",
                        device_type="lightmatter_mars",
                        vendor="LightMatter",
                        model="Mars",
                        wavelengths=64,
                        max_optical_power=15e-3,  # 15 mW
                        driver_version=self._get_lightmatter_driver_version(),
                    )
                    devices.append(device)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        return devices
    
    def _detect_luminous(self) -> List[PhotonicDevice]:
        """Detect Luminous Computing photonic processors."""
        devices = []
        
        # Check for Luminous device files
        luminous_device_path = "/dev/luminous"
        if os.path.exists(luminous_device_path):
            try:
                # Query device info
                with open(f"{luminous_device_path}/info", "r") as f:
                    info = f.read().strip()
                
                device = PhotonicDevice(
                    device_id="luminous:0",
                    device_type="luminous_processor",
                    vendor="Luminous Computing",
                    model="Photonic Processor",
                    wavelengths=80,
                    max_optical_power=20e-3,  # 20 mW
                )
                devices.append(device)
            except (IOError, OSError):
                pass
        
        return devices
    
    def _detect_generic_pcie(self) -> List[PhotonicDevice]:
        """Detect generic PCIe photonic accelerators."""
        devices = []
        
        # Known photonic vendor IDs (examples)
        photonic_vendors = {
            "8086": "Intel Photonics",
            "10de": "NVIDIA Photonic",
            "1002": "AMD Photonic",
        }
        
        try:
            result = subprocess.run(
                ["lspci", "-n"], capture_output=True, text=True, timeout=5
            )
            
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    for vendor_id, vendor_name in photonic_vendors.items():
                        if vendor_id in line and "photonic" in line.lower():
                            device = PhotonicDevice(
                                device_id=f"generic:{len(devices)}",
                                device_type="generic_photonic",
                                vendor=vendor_name,
                                model="Generic Photonic Accelerator",
                                wavelengths=32,
                                max_optical_power=10e-3,  # 10 mW
                            )
                            devices.append(device)
                            break
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        return devices
    
    def _detect_simulation_mode(self) -> List[PhotonicDevice]:
        """Create simulated photonic device for development."""
        devices = []
        
        # Check if simulation mode is enabled
        if (os.getenv("PHOTONIC_SIMULATION", "false").lower() in ("true", "1") or
            "--photonic-sim" in sys.argv):
            
            device = PhotonicDevice(
                device_id="simulator:0",
                device_type="simulation",
                vendor="Photonic Flash Attention",
                model="Software Simulator",
                wavelengths=80,
                max_optical_power=100e-3,  # No power limit in simulation
                temperature=25.0,  # Fixed temperature
                driver_version="sim-0.1.0",
            )
            devices.append(device)
        
        return devices
    
    def _get_lightmatter_driver_version(self) -> Optional[str]:
        """Get LightMatter driver version."""
        try:
            result = subprocess.run(
                ["modinfo", "lightmatter"], 
                capture_output=True, text=True, timeout=5
            )
            
            for line in result.stdout.split('\n'):
                if line.startswith("version:"):
                    return line.split(":", 1)[1].strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        return None
    
    def get_device_by_id(self, device_id: str) -> Optional[PhotonicDevice]:
        """Get device by ID."""
        for device in self._devices:
            if device.device_id == device_id:
                return device
        return None
    
    def get_best_device(self) -> Optional[PhotonicDevice]:
        """Get the best available photonic device."""
        if not self._devices:
            return None
        
        # Prioritize by device type
        priority_order = [
            "lightmatter_mars",
            "luminous_processor", 
            "generic_photonic",
            "simulation",
        ]
        
        for device_type in priority_order:
            for device in self._devices:
                if device.device_type == device_type and device.is_available:
                    return device
        
        # Fallback to first available device
        return next((d for d in self._devices if d.is_available), None)


# Global detector instance
_detector = PhotonicHardwareDetector()


def detect_photonic_hardware() -> bool:
    """Check if any photonic hardware is available."""
    global _detector
    devices = _detector.detect_all_devices()
    return len(devices) > 0


def get_photonic_devices() -> List[PhotonicDevice]:
    """Get list of all detected photonic devices."""
    global _detector
    return _detector.detect_all_devices()


def get_best_photonic_device() -> Optional[PhotonicDevice]:
    """Get the best available photonic device."""
    global _detector
    _detector.detect_all_devices()
    return _detector.get_best_device()


def is_photonic_available() -> bool:
    """Check if photonic computation is available."""
    return detect_photonic_hardware()


def get_device_info() -> Dict[str, Any]:
    """Get detailed information about photonic devices."""
    devices = get_photonic_devices()
    
    return {
        "num_devices": len(devices),
        "devices": [
            {
                "id": device.device_id,
                "type": device.device_type,
                "vendor": device.vendor,
                "model": device.model,
                "wavelengths": device.wavelengths,
                "max_power_mw": device.max_optical_power * 1000,
                "temperature_c": device.temperature,
                "available": device.is_available,
                "driver_version": device.driver_version,
            }
            for device in devices
        ],
        "best_device": _detector.get_best_device().device_id if _detector.get_best_device() else None,
    }