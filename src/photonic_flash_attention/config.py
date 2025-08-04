"""Global configuration for Photonic Flash Attention."""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class GlobalConfig:
    """Global configuration settings."""
    
    # Device settings
    device_priority: list = field(default_factory=lambda: ["photonic", "cuda"])
    photonic_threshold: int = 512  # Switch to photonic for seq_len > threshold
    auto_device_selection: bool = True
    
    # Memory settings
    max_memory_usage: float = 0.8  # Fraction of available memory to use
    memory_pool_enabled: bool = True
    
    # Performance settings
    enable_profiling: bool = False
    benchmark_mode: bool = False
    cache_kernel_selections: bool = True
    
    # Photonic hardware settings
    photonic_wavelengths: int = 80
    modulator_resolution: int = 6  # bits
    detector_noise_floor: float = 1e-12  # W
    
    # Safety settings
    max_optical_power: float = 10e-3  # W
    temperature_monitoring: bool = True
    thermal_shutdown_temp: float = 85.0  # °C
    
    # Logging settings
    log_level: str = "INFO"
    log_device_switches: bool = True
    log_performance_metrics: bool = False
    
    _instance: Optional['GlobalConfig'] = None
    
    @classmethod
    def get_instance(cls) -> 'GlobalConfig':
        """Get singleton instance of global config."""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load_from_env()
        return cls._instance
    
    @classmethod
    def update(cls, **kwargs) -> None:
        """Update global configuration."""
        instance = cls.get_instance()
        for key, value in kwargs.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
            else:
                raise ValueError(f"Unknown config key: {key}")
    
    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        env_mappings = {
            "PHOTONIC_THRESHOLD": ("photonic_threshold", int),
            "PHOTONIC_WAVELENGTHS": ("photonic_wavelengths", int),
            "MAX_OPTICAL_POWER": ("max_optical_power", float),
            "LOG_LEVEL": ("log_level", str),
            "ENABLE_PROFILING": ("enable_profiling", self._str_to_bool),
            "AUTO_DEVICE_SELECTION": ("auto_device_selection", self._str_to_bool),
        }
        
        for env_var, (attr_name, converter) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    setattr(self, attr_name, converter(value))
                except (ValueError, TypeError) as e:
                    print(f"Warning: Invalid value for {env_var}: {value}. Error: {e}")
    
    @staticmethod
    def _str_to_bool(value: str) -> bool:
        """Convert string to boolean."""
        return value.lower() in ("true", "1", "yes", "on")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            key: value for key, value in self.__dict__.items() 
            if not key.startswith('_')
        }
    
    def __repr__(self) -> str:
        """String representation of config."""
        config_items = [f"{k}={v}" for k, v in self.to_dict().items()]
        return f"GlobalConfig({', '.join(config_items)})"


# Convenience function to get global config
def get_config() -> GlobalConfig:
    """Get the global configuration instance."""
    return GlobalConfig.get_instance()