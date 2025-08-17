"""Command-line interface for Photonic Flash Attention."""

import argparse
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import numpy as np

from . import get_device_info, get_version
from .core.hybrid_router import HybridFlashAttention
from .photonic.hardware.detection import get_photonic_devices, get_device_info as get_photonic_device_info
from .utils.logging import setup_logging, get_logger
from .config import get_config


def benchmark(args=None):
    """Run performance benchmarks."""
    if args is None:
        parser = argparse.ArgumentParser(description='Benchmark photonic attention performance')
        parser.add_argument('--seq-lengths', nargs='+', type=int, 
                          default=[128, 256, 512, 1024, 2048, 4096],
                          help='Sequence lengths to benchmark')
        parser.add_argument('--batch-sizes', nargs='+', type=int,
                          default=[1, 2, 4, 8],
                          help='Batch sizes to benchmark')
        parser.add_argument('--embed-dim', type=int, default=768,
                          help='Embedding dimension')
        parser.add_argument('--num-heads', type=int, default=12,
                          help='Number of attention heads')
        parser.add_argument('--num-iterations', type=int, default=10,
                          help='Number of iterations per benchmark')
        parser.add_argument('--output', type=str,
                          help='Output file for results (JSON)')
        parser.add_argument('--verbose', '-v', action='store_true',
                          help='Verbose output')
        args = parser.parse_args()
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logging(level=log_level)
    logger = get_logger('benchmark')
    
    logger.info("Starting Photonic Flash Attention benchmarks")
    logger.info(f"Device info: {get_device_info()}")
    
    # Initialize attention module
    attention = HybridFlashAttention(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        enable_scaling=True,
    )
    
    results = []
    
    for batch_size in args.batch_sizes:
        for seq_len in args.seq_lengths:
            logger.info(f"Benchmarking batch_size={batch_size}, seq_len={seq_len}")
            
            # Generate test data
            query = torch.randn(batch_size, seq_len, args.embed_dim)
            
            # Warmup
            for _ in range(3):
                _ = attention(query)
            
            # Benchmark
            latencies = []
            for i in range(args.num_iterations):
                start_time = time.perf_counter()
                output = attention(query)
                end_time = time.perf_counter()
                
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
                
                if args.verbose:
                    logger.debug(f"  Iteration {i+1}: {latency_ms:.2f}ms")
            
            # Calculate statistics
            avg_latency = np.mean(latencies)
            std_latency = np.std(latencies)
            min_latency = np.min(latencies)
            max_latency = np.max(latencies)
            
            tokens_per_sec = (batch_size * seq_len) / (avg_latency / 1000)
            
            result = {
                'batch_size': batch_size,
                'seq_length': seq_len,
                'embed_dim': args.embed_dim,
                'num_heads': args.num_heads,
                'avg_latency_ms': avg_latency,
                'std_latency_ms': std_latency,
                'min_latency_ms': min_latency,
                'max_latency_ms': max_latency,
                'tokens_per_sec': tokens_per_sec,
                'last_device_used': getattr(attention, 'last_device_used', 'unknown'),
            }
            
            # Add device-specific stats
            stats = attention.get_performance_stats()
            result.update({
                'gpu_calls': stats.get('gpu_calls', 0),
                'photonic_calls': stats.get('photonic_calls', 0),
                'photonic_usage_ratio': stats.get('photonic_usage_ratio', 0.0),
            })
            
            results.append(result)
            
            logger.info(f"  Average: {avg_latency:.2f} ± {std_latency:.2f}ms, "
                       f"{tokens_per_sec:.0f} tokens/sec, "
                       f"device: {result['last_device_used']}")
    
    # Print summary
    logger.info("\n=== Benchmark Results ===")
    for result in results:
        logger.info(
            f"Batch {result['batch_size']:2d}, Seq {result['seq_length']:4d}: "
            f"{result['avg_latency_ms']:6.2f}ms, "
            f"{result['tokens_per_sec']:8.0f} tok/s, "
            f"{result['last_device_used']}"
        )
    
    # Save results
    if args.output:
        output_data = {
            'benchmark_info': {
                'version': get_version(),
                'timestamp': time.time(),
                'device_info': get_device_info(),
                'config': get_config().to_dict(),
            },
            'results': results,
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Results saved to {args.output}")
    
    return results


def calibrate(args=None):
    """Calibrate photonic hardware."""
    if args is None:
        parser = argparse.ArgumentParser(description='Calibrate photonic hardware')
        parser.add_argument('--device-id', type=str,
                          help='Specific device ID to calibrate (default: all)')
        parser.add_argument('--test-patterns', type=int, default=100,
                          help='Number of test patterns for calibration')
        parser.add_argument('--save-calibration', type=str,
                          help='Save calibration data to file')
        parser.add_argument('--load-calibration', type=str,
                          help='Load calibration data from file')
        parser.add_argument('--verbose', '-v', action='store_true',
                          help='Verbose output')
        args = parser.parse_args()
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logging(level=log_level)
    logger = get_logger('calibration')
    
    logger.info("Starting photonic hardware calibration")
    
    # Get photonic devices
    devices = get_photonic_devices()
    
    if not devices:
        logger.error("No photonic devices found")
        return False
    
    logger.info(f"Found {len(devices)} photonic device(s)")
    
    # Filter devices if specific ID requested
    if args.device_id:
        devices = [d for d in devices if d.device_id == args.device_id]
        if not devices:
            logger.error(f"Device {args.device_id} not found")
            return False
    
    calibration_results = {}
    
    for device_info in devices:
        device_id = device_info.device_id
        logger.info(f"Calibrating device: {device_id}")
        
        try:
            # Load existing calibration if requested
            if args.load_calibration:
                with open(args.load_calibration, 'r') as f:
                    cal_data = json.load(f)
                    if device_id in cal_data:
                        logger.info(f"Loaded calibration for {device_id}")
                        calibration_results[device_id] = cal_data[device_id]
                        continue
            
            # Perform calibration
            cal_result = _perform_device_calibration(
                device_info, 
                args.test_patterns,
                logger
            )
            
            calibration_results[device_id] = cal_result
            
            # Log results
            accuracy = cal_result.get('accuracy', 0.0)
            latency = cal_result.get('avg_latency_ms', 0.0)
            
            logger.info(f"  Accuracy: {accuracy:.3f}")
            logger.info(f"  Latency: {latency:.2f}ms")
            
            if accuracy < 0.9:
                logger.warning(f"  Low accuracy for {device_id}: {accuracy:.3f}")
        
        except Exception as e:
            logger.error(f"Calibration failed for {device_id}: {e}")
            calibration_results[device_id] = {'error': str(e)}
    
    # Save calibration results
    if args.save_calibration:
        with open(args.save_calibration, 'w') as f:
            json.dump(calibration_results, f, indent=2)
        logger.info(f"Calibration results saved to {args.save_calibration}")
    
    # Summary
    successful_calibrations = sum(1 for r in calibration_results.values() if 'error' not in r)
    logger.info(f"\nCalibration complete: {successful_calibrations}/{len(devices)} devices successful")
    
    return successful_calibrations == len(devices)


def _perform_device_calibration(
    device_info: Any, 
    num_patterns: int,
    logger
) -> Dict[str, Any]:
    """Perform calibration for a single device."""
    from .photonic.optical_kernels.matrix_mult import OpticalMatMul
    
    # Create optical matrix multiply kernel for this device
    # Note: In a real implementation, this would interface with actual hardware
    optical_kernel = OpticalMatMul()
    
    wavelengths = device_info.wavelengths if hasattr(device_info, 'wavelengths') else 64
    
    # Generate test patterns
    patterns = []
    responses = []
    latencies = []
    
    for i in range(num_patterns):
        # Create random test pattern
        size = min(64, wavelengths)
        A = torch.randn(size, size) * 0.5  # Keep values reasonable
        B = torch.randn(size, size) * 0.5
        
        # Measure response
        start_time = time.perf_counter()
        try:
            result = optical_kernel.forward(A, B)
            end_time = time.perf_counter()
            
            latency = (end_time - start_time) * 1000
            latencies.append(latency)
            
            # Store pattern and response
            patterns.append((A, B))
            responses.append(result)
            
            if (i + 1) % 10 == 0:
                logger.debug(f"  Completed {i+1}/{num_patterns} patterns")
        
        except Exception as e:
            logger.warning(f"  Pattern {i} failed: {e}")
    
    if not responses:
        raise RuntimeError("No successful calibration patterns")
    
    # Analyze calibration quality
    errors = []
    for (A, B), result in zip(patterns, responses):
        expected = torch.matmul(A, B)
        error = torch.abs(result - expected).mean().item()
        errors.append(error)
    
    avg_error = np.mean(errors)
    accuracy = max(0.0, 1.0 - avg_error)
    
    return {
        'num_patterns': len(responses),
        'avg_error': avg_error,
        'accuracy': accuracy,
        'avg_latency_ms': np.mean(latencies) if latencies else 0.0,
        'std_latency_ms': np.std(latencies) if latencies else 0.0,
        'timestamp': time.time(),
    }


def device_info(args=None):
    """Display device information."""
    if args is None:
        parser = argparse.ArgumentParser(description='Display photonic device information')
        parser.add_argument('--json', action='store_true',
                          help='Output in JSON format')
        parser.add_argument('--refresh', action='store_true',
                          help='Refresh device cache')
        args = parser.parse_args()
    
    # Get device information
    info = get_device_info()
    devices = get_photonic_devices()
    
    if args.json:
        # Convert devices to serializable format
        device_list = []
        for device in devices:
            device_list.append({
                'id': device.device_id,
                'type': device.device_type,
                'vendor': device.vendor,
                'model': device.model,
                'wavelengths': device.wavelengths,
                'max_power_mw': device.max_optical_power * 1000,
                'temperature_c': device.temperature,
                'available': device.is_available,
                'driver_version': device.driver_version,
            })
        
        output = {
            'version': get_version(),
            'system_info': info,
            'devices': device_list,
        }
        print(json.dumps(output, indent=2))
    else:
        # Human-readable output
        print(f"Photonic Flash Attention v{get_version()}")
        print(f"CUDA available: {info['cuda_available']}")
        if info['cuda_available']:
            print(f"CUDA devices: {info['cuda_device_count']}")
            for i, name in enumerate(info.get('gpu_names', [])):
                print(f"  GPU {i}: {name}")
        
        print(f"\nPhotonic devices: {len(devices)}")
        if devices:
            for device in devices:
                print(f"  {device.device_id}: {device.vendor} {device.device_type}")
                print(f"    Wavelengths: {device.wavelengths}")
                print(f"    Max power: {device.max_optical_power * 1000:.1f} mW")
                if device.temperature:
                    print(f"    Temperature: {device.temperature:.1f}°C")
                print(f"    Available: {device.is_available}")
                print()
        else:
            print("  No photonic devices detected")
            print("  Try setting PHOTONIC_SIMULATION=true for simulation mode")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Photonic Flash Attention CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  benchmark     Run performance benchmarks
  calibrate     Calibrate photonic hardware  
  device-info   Display device information
  
Examples:
  photonic-benchmark --seq-lengths 512 1024 2048 --output results.json
  photonic-calibrate --device-id lightmatter:0 --save-calibration cal.json
  photonic-device-info --json
"""
    )
    
    parser.add_argument('--version', action='version', version=f'%(prog)s {get_version()}')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run benchmarks')
    benchmark_parser.set_defaults(func=benchmark)
    
    # Calibrate command
    calibrate_parser = subparsers.add_parser('calibrate', help='Calibrate hardware')
    calibrate_parser.set_defaults(func=calibrate)
    
    # Device info command
    device_parser = subparsers.add_parser('device-info', help='Display device info')
    device_parser.set_defaults(func=device_info)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    try:
        result = args.func(args)
        return 0 if result else 1
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
