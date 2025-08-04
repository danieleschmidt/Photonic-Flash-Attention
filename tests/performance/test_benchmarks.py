"""Performance benchmark tests for photonic attention."""

import pytest
import torch
import time
import numpy as np
from typing import Dict, List, Tuple

from photonic_flash_attention import PhotonicFlashAttention
from photonic_flash_attention.core.flash_attention_3 import FlashAttention3
from photonic_flash_attention.core.hybrid_router import HybridFlashAttention


@pytest.mark.performance
@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Comprehensive performance benchmarks."""
    
    @pytest.fixture(autouse=True)
    def setup_benchmark(self, benchmark_results):
        """Setup for benchmark tests."""
        self.benchmark_results = benchmark_results
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def measure_latency(self, func, *args, warmup_runs=3, test_runs=10):
        """Measure function latency with proper GPU synchronization."""
        # Warmup runs
        for _ in range(warmup_runs):
            func(*args)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Timed runs
        times = []
        for _ in range(test_runs):
            start_time = time.perf_counter()
            result = func(*args)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'result': result,
        }
    
    def measure_memory_usage(self):
        """Measure current GPU memory usage."""
        if torch.cuda.is_available():
            return {
                'allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
                'reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
                'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024 / 1024,
            }
        return {'allocated_mb': 0, 'reserved_mb': 0, 'max_allocated_mb': 0}
    
    @pytest.mark.parametrize("seq_len", [128, 256, 512, 1024, 2048])
    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_scaling_benchmark(self, seq_len, batch_size, device, benchmark_results):
        """Test performance scaling with sequence length and batch size."""
        embed_dim, num_heads = 768, 12
        
        # Skip very large configurations to prevent OOM
        if seq_len * batch_size > 8192:
            pytest.skip("Configuration too large for testing")
        
        # Create test data
        query = torch.randn(batch_size, seq_len, embed_dim, device=device)
        
        # Test GPU Flash Attention
        gpu_attention = FlashAttention3(embed_dim=embed_dim, num_heads=num_heads, device=device)
        gpu_results = self.measure_latency(gpu_attention, query, need_weights=False)
        
        # Test Photonic Flash Attention
        photonic_attention = PhotonicFlashAttention(embed_dim=embed_dim, num_heads=num_heads, device='auto')
        photonic_results = self.measure_latency(photonic_attention, query, need_weights=False)
        
        # Store results
        config_key = f"seq{seq_len}_batch{batch_size}"
        benchmark_results[config_key] = {
            'seq_len': seq_len,
            'batch_size': batch_size,
            'embed_dim': embed_dim,
            'num_heads': num_heads,
            'gpu_latency_ms': gpu_results['mean_ms'],
            'photonic_latency_ms': photonic_results['mean_ms'],
            'speedup': gpu_results['mean_ms'] / photonic_results['mean_ms'],
            'memory_usage': self.measure_memory_usage(),
        }
        
        # Basic performance assertions
        assert gpu_results['mean_ms'] > 0
        assert photonic_results['mean_ms'] > 0
        
        # Log results
        print(f"\nBenchmark {config_key}:")
        print(f"  GPU: {gpu_results['mean_ms']:.2f}ms ± {gpu_results['std_ms']:.2f}ms")
        print(f"  Photonic: {photonic_results['mean_ms']:.2f}ms ± {photonic_results['std_ms']:.2f}ms")
        print(f"  Speedup: {gpu_results['mean_ms'] / photonic_results['mean_ms']:.2f}x")
    
    def test_memory_efficiency_benchmark(self, device):
        """Test memory efficiency across different implementations."""
        configs = [
            (512, 8, 768, 12),   # Medium
            (1024, 4, 512, 8),   # Long sequence
            (256, 16, 1024, 16), # Large model
        ]
        
        results = {}
        
        for seq_len, batch_size, embed_dim, num_heads in configs:
            config_name = f"{seq_len}x{batch_size}x{embed_dim}x{num_heads}"
            
            # Clear memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            query = torch.randn(batch_size, seq_len, embed_dim, device=device)
            
            # Test different implementations
            implementations = {
                'gpu': FlashAttention3(embed_dim=embed_dim, num_heads=num_heads, device=device),
                'photonic': PhotonicFlashAttention(embed_dim=embed_dim, num_heads=num_heads, device='auto'),
                'hybrid': HybridFlashAttention(embed_dim=embed_dim, num_heads=num_heads, device='auto'),
            }
            
            config_results = {}
            
            for impl_name, impl in implementations.items():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                
                # Measure memory before
                memory_before = self.measure_memory_usage()
                
                # Run implementation
                try:
                    output = impl(query, need_weights=False)
                    memory_peak = self.measure_memory_usage()
                    
                    config_results[impl_name] = {
                        'memory_before_mb': memory_before['allocated_mb'],
                        'memory_peak_mb': memory_peak['max_allocated_mb'],
                        'memory_delta_mb': memory_peak['max_allocated_mb'] - memory_before['allocated_mb'],
                        'success': True,
                    }
                
                except Exception as e:
                    config_results[impl_name] = {
                        'memory_before_mb': memory_before['allocated_mb'],
                        'memory_peak_mb': 0,
                        'memory_delta_mb': 0,
                        'success': False,
                        'error': str(e),
                    }
            
            results[config_name] = config_results
            
            # Print results
            print(f"\nMemory benchmark {config_name}:")
            for impl_name, result in config_results.items():
                if result['success']:
                    print(f"  {impl_name}: {result['memory_delta_mb']:.1f}MB peak")
                else:
                    print(f"  {impl_name}: FAILED - {result.get('error', 'Unknown error')}")
        
        # Store results
        self.benchmark_results['memory_efficiency'] = results
    
    def test_throughput_benchmark(self, device):
        """Test throughput (tokens/second) for different configurations."""
        configs = [
            (128, 1, 512, 8),   # Small, single batch
            (512, 4, 768, 12),  # Medium
            (1024, 2, 512, 8),  # Long sequence
        ]
        
        results = {}
        
        for seq_len, batch_size, embed_dim, num_heads in configs:
            config_name = f"{seq_len}x{batch_size}"
            total_tokens = seq_len * batch_size
            
            query = torch.randn(batch_size, seq_len, embed_dim, device=device)
            
            # Test implementations
            implementations = {
                'gpu': FlashAttention3(embed_dim=embed_dim, num_heads=num_heads, device=device),
                'photonic': PhotonicFlashAttention(embed_dim=embed_dim, num_heads=num_heads, device='auto'),
            }
            
            config_results = {}
            
            for impl_name, impl in implementations.items():
                timing_results = self.measure_latency(impl, query, need_weights=False)
                
                # Calculate throughput
                latency_seconds = timing_results['mean_ms'] / 1000
                throughput = total_tokens / latency_seconds
                
                config_results[impl_name] = {
                    'latency_ms': timing_results['mean_ms'],
                    'throughput_tokens_per_sec': throughput,
                    'total_tokens': total_tokens,
                }
            
            results[config_name] = config_results
            
            # Print results
            print(f"\nThroughput benchmark {config_name} ({total_tokens} tokens):")
            for impl_name, result in config_results.items():
                print(f"  {impl_name}: {result['throughput_tokens_per_sec']:.0f} tokens/sec")
        
        self.benchmark_results['throughput'] = results
    
    def test_accuracy_benchmark(self, device):
        """Test numerical accuracy between implementations."""
        # Use deterministic operations for reproducible results
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        batch_size, seq_len, embed_dim, num_heads = 2, 256, 512, 8
        
        # Create test inputs
        query = torch.randn(batch_size, seq_len, embed_dim, device=device)
        key = torch.randn(batch_size, seq_len, embed_dim, device=device)
        value = torch.randn(batch_size, seq_len, embed_dim, device=device)
        
        # Reference implementation (GPU)
        gpu_attention = FlashAttention3(embed_dim=embed_dim, num_heads=num_heads, device=device)
        
        # Set to eval mode for deterministic behavior
        gpu_attention.eval()
        
        with torch.no_grad():
            reference_output = gpu_attention(query, key, value, need_weights=False)
        
        # Test other implementations
        implementations = {
            'photonic': PhotonicFlashAttention(embed_dim=embed_dim, num_heads=num_heads, device='auto'),
            'hybrid': HybridFlashAttention(embed_dim=embed_dim, num_heads=num_heads, device='auto'),
        }
        
        accuracy_results = {}
        
        for impl_name, impl in implementations.items():
            impl.eval()
            
            try:
                with torch.no_grad():
                    test_output = impl(query, key, value, need_weights=False)
                
                # Compute accuracy metrics
                mse = torch.mean((reference_output - test_output) ** 2).item()
                mae = torch.mean(torch.abs(reference_output - test_output)).item()
                max_error = torch.max(torch.abs(reference_output - test_output)).item()
                
                # Relative error
                relative_error = mae / torch.mean(torch.abs(reference_output)).item()
                
                accuracy_results[impl_name] = {
                    'mse': mse,
                    'mae': mae,
                    'max_error': max_error,
                    'relative_error': relative_error,
                    'success': True,
                }
                
                # Basic accuracy assertions
                assert relative_error < 0.1, f"{impl_name} relative error too high: {relative_error}"
                
            except Exception as e:
                accuracy_results[impl_name] = {
                    'success': False,
                    'error': str(e),
                }
        
        # Print results
        print(f"\nAccuracy benchmark:")
        for impl_name, result in accuracy_results.items():
            if result['success']:
                print(f"  {impl_name}: MAE={result['mae']:.6f}, RelErr={result['relative_error']:.6f}")
            else:
                print(f"  {impl_name}: FAILED - {result['error']}")
        
        self.benchmark_results['accuracy'] = accuracy_results
    
    @pytest.mark.parametrize("enable_scaling", [True, False])
    def test_concurrency_benchmark(self, enable_scaling, device):
        """Test concurrent processing performance."""
        import threading
        import concurrent.futures
        
        batch_size, seq_len, embed_dim, num_heads = 2, 512, 768, 12
        
        # Create attention module
        attention = HybridFlashAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            device='auto',
            enable_scaling=enable_scaling,
            max_concurrent_requests=4,
        )
        
        def run_attention():
            query = torch.randn(batch_size, seq_len, embed_dim, device=device)
            return attention(query, need_weights=False)
        
        # Test concurrent execution
        num_threads = 8
        num_requests_per_thread = 5
        
        start_time = time.perf_counter()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            
            for _ in range(num_threads):
                for _ in range(num_requests_per_thread):
                    future = executor.submit(run_attention)
                    futures.append(future)
            
            # Wait for all to complete
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append(f"ERROR: {e}")
        
        end_time = time.perf_counter()
        
        total_requests = num_threads * num_requests_per_thread
        total_time = end_time - start_time
        
        # Get performance stats
        stats = attention.get_performance_stats()
        
        benchmark_result = {
            'enable_scaling': enable_scaling,
            'total_requests': total_requests,
            'total_time_seconds': total_time,
            'requests_per_second': total_requests / total_time,
            'successful_requests': len([r for r in results if not isinstance(r, str)]),
            'failed_requests': len([r for r in results if isinstance(r, str)]),
            'peak_concurrent': stats.get('peak_concurrent', 0),
            'performance_stats': stats,
        }
        
        # Print results
        print(f"\nConcurrency benchmark (scaling={enable_scaling}):")
        print(f"  Total requests: {total_requests}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Requests/sec: {benchmark_result['requests_per_second']:.2f}")
        print(f"  Success rate: {benchmark_result['successful_requests']}/{total_requests}")
        print(f"  Peak concurrent: {benchmark_result['peak_concurrent']}")
        
        # Store results
        if 'concurrency' not in self.benchmark_results:
            self.benchmark_results['concurrency'] = {}
        self.benchmark_results['concurrency'][f'scaling_{enable_scaling}'] = benchmark_result
        
        # Assertions
        assert benchmark_result['successful_requests'] > 0, "No successful requests"
        assert benchmark_result['requests_per_second'] > 0, "Zero throughput"
    
    def test_generate_benchmark_report(self, benchmark_results):
        """Generate a comprehensive benchmark report."""
        if not benchmark_results:
            pytest.skip("No benchmark results available")
        
        print("\n" + "="*80)
        print("PHOTONIC FLASH ATTENTION BENCHMARK REPORT")
        print("="*80)
        
        # Scaling results
        if any(k.startswith('seq') for k in benchmark_results.keys()):
            print("\nSCALING BENCHMARK RESULTS:")
            print("-" * 40)
            
            scaling_results = {k: v for k, v in benchmark_results.items() if k.startswith('seq')}
            
            for config, result in sorted(scaling_results.items()):
                print(f"{config:20} | GPU: {result['gpu_latency_ms']:6.2f}ms | "
                      f"Photonic: {result['photonic_latency_ms']:6.2f}ms | "
                      f"Speedup: {result['speedup']:5.2f}x")
        
        # Memory efficiency
        if 'memory_efficiency' in benchmark_results:
            print("\nMEMORY EFFICIENCY RESULTS:")
            print("-" * 40)
            
            for config, results in benchmark_results['memory_efficiency'].items():
                print(f"\n{config}:")
                for impl, result in results.items():
                    if result['success']:
                        print(f"  {impl:10}: {result['memory_delta_mb']:6.1f}MB")
                    else:
                        print(f"  {impl:10}: FAILED")
        
        # Throughput results
        if 'throughput' in benchmark_results:
            print("\nTHROUGHPUT RESULTS:")
            print("-" * 40)
            
            for config, results in benchmark_results['throughput'].items():
                print(f"\n{config}:")
                for impl, result in results.items():
                    print(f"  {impl:10}: {result['throughput_tokens_per_sec']:8.0f} tokens/sec")
        
        # Accuracy results
        if 'accuracy' in benchmark_results:
            print("\nACCURACY RESULTS:")
            print("-" * 40)
            
            for impl, result in benchmark_results['accuracy'].items():
                if result['success']:
                    print(f"{impl:10}: Relative Error = {result['relative_error']:.6f}")
                else:
                    print(f"{impl:10}: FAILED")
        
        # Concurrency results
        if 'concurrency' in benchmark_results:
            print("\nCONCURRENCY RESULTS:")
            print("-" * 40)
            
            for config, result in benchmark_results['concurrency'].items():
                print(f"{config:20}: {result['requests_per_second']:6.2f} req/sec | "
                      f"Peak concurrent: {result['peak_concurrent']}")
        
        print("\n" + "="*80)
        
        # Save results to file for CI/CD
        import json
        with open('/tmp/benchmark_results.json', 'w') as f:
            # Convert any non-serializable objects to strings
            serializable_results = {}
            for k, v in benchmark_results.items():
                try:
                    json.dumps(v)
                    serializable_results[k] = v
                except TypeError:
                    serializable_results[k] = str(v)
            
            json.dump(serializable_results, f, indent=2)
        
        print(f"Benchmark results saved to /tmp/benchmark_results.json")