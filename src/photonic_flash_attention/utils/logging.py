"""Comprehensive logging system for photonic attention."""

import logging
import logging.handlers
import os
import sys
import json
import time
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime, timezone


class PhotonicFormatter(logging.Formatter):
    """Custom formatter for photonic attention logs."""
    
    def __init__(self, include_extra: bool = True, json_format: bool = False):
        self.include_extra = include_extra
        self.json_format = json_format
        
        if json_format:
            super().__init__()
        else:
            super().__init__(
                fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
    
    def format(self, record: logging.LogRecord) -> str:
        if self.json_format:
            return self._format_json(record)
        else:
            return self._format_text(record)
    
    def _format_json(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add extra fields
        if self.include_extra:
            for key, value in record.__dict__.items():
                if key not in ('name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                              'filename', 'module', 'lineno', 'funcName', 'created', 
                              'msecs', 'relativeCreated', 'thread', 'threadName',
                              'processName', 'process', 'getMessage', 'stack_info',
                              'exc_info', 'exc_text'):
                    try:
                        json.dumps(value)  # Test JSON serializability
                        log_entry[key] = value
                    except (TypeError, ValueError):
                        log_entry[key] = str(value)
        
        # Add exception info
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)
    
    def _format_text(self, record: logging.LogRecord) -> str:
        """Format log record as text."""
        formatted = super().format(record)
        
        # Add extra fields if present
        if self.include_extra:
            extra_fields = []
            for key, value in record.__dict__.items():
                if key not in ('name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                              'filename', 'module', 'lineno', 'funcName', 'created', 
                              'msecs', 'relativeCreated', 'thread', 'threadName',
                              'processName', 'process', 'getMessage', 'stack_info',
                              'exc_info', 'exc_text', 'asctime'):
                    extra_fields.append(f"{key}={value}")
            
            if extra_fields:
                formatted += f" | {' | '.join(extra_fields)}"
        
        return formatted


class PerformanceLogger:
    """Logger for performance metrics."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._timers: Dict[str, float] = {}
    
    def start_timer(self, name: str) -> None:
        """Start a performance timer."""
        self._timers[name] = time.perf_counter()
    
    def end_timer(self, name: str, **extra_fields) -> float:
        """End a performance timer and log the duration."""
        if name not in self._timers:
            self.logger.warning(f"Timer '{name}' was not started")
            return 0.0
        
        duration = time.perf_counter() - self._timers[name]
        del self._timers[name]
        
        self.logger.info(
            f"Performance: {name} completed in {duration*1000:.2f}ms",
            extra={
                'performance_metric': name,
                'duration_ms': duration * 1000,
                'duration_s': duration,
                **extra_fields
            }
        )
        
        return duration
    
    def log_metric(self, name: str, value: float, unit: str = "", **extra_fields) -> None:
        """Log a performance metric."""
        self.logger.info(
            f"Metric: {name} = {value}{unit}",
            extra={
                'metric_name': name,
                'metric_value': value,
                'metric_unit': unit,
                **extra_fields
            }
        )


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False,
    max_file_size: int = 50 * 1024 * 1024,  # 50MB
    backup_count: int = 5,
) -> None:
    """
    Set up comprehensive logging for the photonic attention system.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (if None, logs to stdout)
        json_format: Use JSON format for structured logging
        max_file_size: Maximum log file size before rotation
        backup_count: Number of backup files to keep
    """
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = PhotonicFormatter(include_extra=True, json_format=json_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (with rotation)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)  # File gets all messages
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger('photonic_flash_attention').setLevel(getattr(logging, level.upper()))
    
    # Reduce noise from external libraries
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('numpy').setLevel(logging.WARNING)
    
    # Log setup completion
    logger = logging.getLogger('photonic_flash_attention.logging')
    logger.info(f"Logging initialized: level={level}, file={log_file}, json={json_format}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name (typically module name)
        
    Returns:
        Configured logger instance
    """
    # Ensure the name is under our namespace
    if not name.startswith('photonic_flash_attention'):
        name = f'photonic_flash_attention.{name}'
    
    logger = logging.getLogger(name)
    
    # Add performance logging capability
    if not hasattr(logger, 'perf'):
        logger.perf = PerformanceLogger(logger)
    
    return logger


def get_performance_logger(name: str) -> PerformanceLogger:
    """Get a performance logger for the specified module."""
    logger = get_logger(name)
    return logger.perf


class LogContext:
    """Context manager for adding extra fields to log messages."""
    
    def __init__(self, logger: logging.Logger, **extra_fields):
        self.logger = logger
        self.extra_fields = extra_fields
        self.old_factory = None
    
    def __enter__(self):
        self.old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.extra_fields.items():
                setattr(record, key, value)
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.old_factory)


# Initialize logging from environment variables
def _init_from_env():
    """Initialize logging from environment variables."""
    log_level = os.getenv('PHOTONIC_LOG_LEVEL', 'INFO')
    log_file = os.getenv('PHOTONIC_LOG_FILE')
    json_format = os.getenv('PHOTONIC_LOG_JSON', 'false').lower() in ('true', '1')
    
    if not logging.getLogger().handlers:  # Only setup if not already configured
        setup_logging(level=log_level, log_file=log_file, json_format=json_format)


# Auto-initialize
_init_from_env()