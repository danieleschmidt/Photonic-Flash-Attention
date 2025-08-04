"""Custom exceptions for photonic attention system."""


class PhotonicFlashAttentionError(Exception):
    """Base exception for all photonic flash attention errors."""
    pass


class PhotonicHardwareError(PhotonicFlashAttentionError):
    """Raised when photonic hardware encounters an error."""
    
    def __init__(self, message: str, device_id: str = None, error_code: str = None):
        super().__init__(message)
        self.device_id = device_id
        self.error_code = error_code
        self.message = message
    
    def __str__(self) -> str:
        parts = [self.message]
        if self.device_id:
            parts.append(f"Device: {self.device_id}")
        if self.error_code:
            parts.append(f"Error Code: {self.error_code}")
        return " | ".join(parts)


class PhotonicComputationError(PhotonicFlashAttentionError):
    """Raised when photonic computation fails."""
    
    def __init__(self, message: str, operation: str = None, input_shapes: tuple = None):
        super().__init__(message)
        self.operation = operation
        self.input_shapes = input_shapes
        self.message = message
    
    def __str__(self) -> str:
        parts = [self.message]
        if self.operation:
            parts.append(f"Operation: {self.operation}")
        if self.input_shapes:
            parts.append(f"Input shapes: {self.input_shapes}")
        return " | ".join(parts)


class PhotonicConfigurationError(PhotonicFlashAttentionError):
    """Raised when configuration is invalid."""
    pass


class PhotonicSecurityError(PhotonicFlashAttentionError):
    """Raised when security validation fails."""
    pass


class PhotonicThermalError(PhotonicHardwareError):
    """Raised when thermal limits are exceeded."""
    
    def __init__(self, message: str, temperature: float, limit: float, device_id: str = None):
        super().__init__(message, device_id, "THERMAL_LIMIT_EXCEEDED")
        self.temperature = temperature
        self.limit = limit
    
    def __str__(self) -> str:
        return f"{self.message} | Temperature: {self.temperature}°C > {self.limit}°C | Device: {self.device_id or 'Unknown'}"


class PhotonicCalibrationError(PhotonicHardwareError):
    """Raised when device calibration fails."""
    pass


class PhotonicMemoryError(PhotonicFlashAttentionError):
    """Raised when memory allocation or management fails."""
    
    def __init__(self, message: str, requested_bytes: int = None, available_bytes: int = None):
        super().__init__(message)
        self.requested_bytes = requested_bytes
        self.available_bytes = available_bytes
    
    def __str__(self) -> str:
        parts = [self.message]
        if self.requested_bytes is not None:
            parts.append(f"Requested: {self.requested_bytes / 1024 / 1024:.1f} MB")
        if self.available_bytes is not None:
            parts.append(f"Available: {self.available_bytes / 1024 / 1024:.1f} MB")
        return " | ".join(parts)


class PhotonicDriverError(PhotonicHardwareError):
    """Raised when photonic device driver encounters an error."""
    
    def __init__(self, message: str, driver_version: str = None, device_id: str = None):
        super().__init__(message, device_id, "DRIVER_ERROR")
        self.driver_version = driver_version
    
    def __str__(self) -> str:
        parts = [self.message]
        if self.driver_version:
            parts.append(f"Driver: {self.driver_version}")
        if self.device_id:
            parts.append(f"Device: {self.device_id}")
        return " | ".join(parts)


class PhotonicTimeoutError(PhotonicFlashAttentionError):
    """Raised when operations timeout."""
    
    def __init__(self, message: str, timeout_seconds: float, operation: str = None):
        super().__init__(message)
        self.timeout_seconds = timeout_seconds
        self.operation = operation
    
    def __str__(self) -> str:
        parts = [self.message]
        if self.operation:
            parts.append(f"Operation: {self.operation}")
        parts.append(f"Timeout: {self.timeout_seconds}s")
        return " | ".join(parts)