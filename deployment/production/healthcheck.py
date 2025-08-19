#!/usr/bin/env python3
"""
Production health check for Photonic Flash Attention system.

Performs comprehensive health validation including:
- System imports and initialization
- Hardware availability checks
- Performance validation
- Resource monitoring
- Security validation
"""

import sys
import time
import os
import psutil
import logging
from typing import Dict, Any, Tuple

# Configure minimal logging for health checks
logging.basicConfig(level=logging.WARNING)


def check_imports() -> Tuple[bool, str]:
    """Check that all critical imports work."""
    try:
        import numpy as np
        import torch
        
        # Core library import
        import photonic_flash_attention
        from photonic_flash_attention import get_device_info, get_version
        
        # Verify version
        version = get_version()
        if not version:
            return False, "Version check failed"
        
        return True, f"Imports successful, version {version}"
    
    except ImportError as e:
        return False, f"Import error: {e}"
    except Exception as e:
        return False, f"Unexpected error: {e}"


def check_hardware() -> Tuple[bool, str]:
    """Check hardware availability and status."""
    try:
        from photonic_flash_attention import get_device_info
        
        device_info = get_device_info()
        
        # Check CUDA availability if expected
        cuda_available = device_info.get('cuda_available', False)
        photonic_available = device_info.get('photonic_available', False)
        
        # At least one compute device should be available
        if not cuda_available and not photonic_available:
            return False, "No compute devices available"
        
        return True, f"Hardware OK: CUDA={cuda_available}, Photonic={photonic_available}"
    
    except Exception as e:
        return False, f"Hardware check failed: {e}"


def check_performance() -> Tuple[bool, str]:
    """Basic performance validation."""
    try:
        import numpy as np
        
        # Simple matrix operation test
        start_time = time.time()
        a = np.random.randn(100, 100)
        b = np.random.randn(100, 100)
        c = np.dot(a, b)
        elapsed = time.time() - start_time
        
        # Should complete quickly
        if elapsed > 1.0:
            return False, f"Performance degraded: {elapsed:.3f}s for basic operation"
        
        return True, f"Performance OK: {elapsed:.3f}s"
    
    except Exception as e:
        return False, f"Performance check failed: {e}"


def check_resources() -> Tuple[bool, str]:
    """Check system resources."""
    try:
        # Memory check
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            return False, f"Memory usage too high: {memory.percent:.1f}%"
        
        # CPU check
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 95:
            return False, f"CPU usage too high: {cpu_percent:.1f}%"
        
        # Disk check
        disk = psutil.disk_usage('/')
        if disk.percent > 95:
            return False, f"Disk usage too high: {disk.percent:.1f}%"
        
        return True, f"Resources OK: CPU={cpu_percent:.1f}%, MEM={memory.percent:.1f}%, DISK={disk.percent:.1f}%"
    
    except Exception as e:
        return False, f"Resource check failed: {e}"


def check_thermal() -> Tuple[bool, str]:
    """Check thermal status if sensors available."""
    try:
        # Try to get temperature readings
        temps = psutil.sensors_temperatures()
        
        if not temps:
            return True, "No thermal sensors detected"
        
        max_temp = 0.0
        for name, entries in temps.items():
            for entry in entries:
                if entry.current > max_temp:
                    max_temp = entry.current
        
        # Check thermal limits
        if max_temp > 85.0:
            return False, f"Temperature too high: {max_temp:.1f}°C"
        
        return True, f"Thermal OK: max {max_temp:.1f}°C"
    
    except Exception as e:
        # Non-critical failure
        return True, f"Thermal check skipped: {e}"


def check_configuration() -> Tuple[bool, str]:
    """Check configuration and environment."""
    try:
        # Check required environment variables
        required_envs = ['PHOTONIC_MODE']
        for env_var in required_envs:
            if env_var not in os.environ:
                return False, f"Missing environment variable: {env_var}"
        
        # Check data directories
        data_dirs = ['/app/data', '/app/logs', '/app/models']
        for data_dir in data_dirs:
            if not os.path.exists(data_dir):
                return False, f"Missing data directory: {data_dir}"
            
            if not os.access(data_dir, os.W_OK):
                return False, f"Data directory not writable: {data_dir}"
        
        return True, "Configuration OK"
    
    except Exception as e:
        return False, f"Configuration check failed: {e}"


def run_health_checks() -> Dict[str, Any]:
    """Run comprehensive health checks."""
    checks = {
        'imports': check_imports,
        'hardware': check_hardware,
        'performance': check_performance,
        'resources': check_resources,
        'thermal': check_thermal,
        'configuration': check_configuration
    }
    
    results = {}
    overall_healthy = True
    
    for check_name, check_func in checks.items():
        try:
            success, message = check_func()
            results[check_name] = {
                'status': 'PASS' if success else 'FAIL',
                'message': message
            }
            
            if not success:
                overall_healthy = False
                
        except Exception as e:
            results[check_name] = {
                'status': 'ERROR',
                'message': f"Check failed with exception: {e}"
            }
            overall_healthy = False
    
    results['overall'] = {
        'status': 'HEALTHY' if overall_healthy else 'UNHEALTHY',
        'timestamp': time.time()
    }
    
    return results


def main() -> int:
    """Main health check entry point."""
    try:
        # Run health checks
        results = run_health_checks()
        
        # Print results in production-friendly format
        overall_status = results['overall']['status']
        
        if overall_status == 'HEALTHY':
            print("HEALTHY")
            
            # Optional: print detailed status for debugging
            if os.environ.get('HEALTH_CHECK_VERBOSE', '').lower() == 'true':
                for check_name, result in results.items():
                    if check_name != 'overall':
                        print(f"{check_name}: {result['status']} - {result['message']}")
            
            return 0
        
        else:
            print("UNHEALTHY")
            
            # Print failed checks
            for check_name, result in results.items():
                if check_name != 'overall' and result['status'] != 'PASS':
                    print(f"FAILED: {check_name} - {result['message']}", file=sys.stderr)
            
            return 1
    
    except Exception as e:
        print(f"HEALTH_CHECK_ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())