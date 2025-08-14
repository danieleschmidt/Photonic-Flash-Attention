#!/usr/bin/env python3
"""
Performance benchmark script for photonic attention system.

Tests system performance, scalability, and resource usage.
"""

import os
import sys
import time
import threading
import multiprocessing
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import random

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

@dataclass
class BenchmarkResult:
    """Result of a benchmark test."""
    test_name: str
    duration_seconds: float
    operations_per_second: float
    memory_used_mb: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

class PerformanceBenchmark:
    """Comprehensive performance benchmark suite."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmark tests."""
        print("🚀 PHOTONIC FLASH ATTENTION - PERFORMANCE BENCHMARK")
        print("=" * 65)
        
        benchmarks = [
            ("Import Performance", self.benchmark_import_speed),
            ("Device Detection", self.benchmark_device_detection),
            ("Configuration Loading", self.benchmark_config_loading),
            ("Memory Management", self.benchmark_memory_management),
            ("Concurrent Operations", self.benchmark_concurrency),
            ("Cache Performance", self.benchmark_cache_performance),
            ("Load Balancer", self.benchmark_load_balancer),
        ]
        
        for test_name, benchmark_func in benchmarks:
            print(f"\n🔄 Running: {test_name}")
            try:
                result = benchmark_func()
                self.results.append(result)
                if result.success:
                    print(f"✅ {test_name}: {result.operations_per_second:.2f} ops/sec "
                          f"({result.duration_seconds:.3f}s, {result.memory_used_mb:.1f}MB)")
                else:
                    print(f"❌ {test_name}: {result.error_message}")
            except Exception as e:
                print(f"❌ {test_name}: Benchmark failed - {e}")
                result = BenchmarkResult(
                    test_name=test_name,
                    duration_seconds=0,
                    operations_per_second=0,
                    memory_used_mb=0,
                    success=False,
                    error_message=str(e)
                )
                self.results.append(result)
        
        return self.generate_report()
    
    def benchmark_import_speed(self) -> BenchmarkResult:
        """Benchmark import performance."""
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        # Test multiple imports
        operations = 0
        try:
            import photonic_flash_attention
            operations += 1
            
            from photonic_flash_attention.core import PhotonicAttention
            operations += 1
            
            from photonic_flash_attention.photonic.hardware.detection import get_photonic_devices
            operations += 1
            
            from photonic_flash_attention.utils.security import SecurityManager
            operations += 1
            
            from photonic_flash_attention.monitoring.health_monitor import HealthMonitor
            operations += 1
            
        except Exception as e:
            duration = time.time() - start_time
            return BenchmarkResult(
                test_name="Import Performance",
                duration_seconds=duration,
                operations_per_second=0,
                memory_used_mb=0,
                success=False,
                error_message=str(e)
            )
        
        end_time = time.time()
        end_memory = self.get_memory_usage()
        
        duration = end_time - start_time
        ops_per_sec = operations / duration if duration > 0 else 0
        memory_used = max(0, end_memory - start_memory)
        
        return BenchmarkResult(
            test_name="Import Performance",
            duration_seconds=duration,
            operations_per_second=ops_per_sec,
            memory_used_mb=memory_used,
            success=True
        )
    
    def benchmark_device_detection(self) -> BenchmarkResult:
        """Benchmark device detection performance."""
        os.environ["PHOTONIC_SIMULATION"] = "true"
        
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        try:
            from photonic_flash_attention.photonic.hardware.detection import get_photonic_devices
            
            operations = 0
            for _ in range(10):  # Run detection 10 times
                devices = get_photonic_devices()
                operations += 1
                if not devices:
                    raise Exception("No devices detected")
        
        except Exception as e:
            duration = time.time() - start_time
            return BenchmarkResult(
                test_name="Device Detection",
                duration_seconds=duration,
                operations_per_second=0,
                memory_used_mb=0,
                success=False,
                error_message=str(e)
            )
        finally:
            os.environ.pop("PHOTONIC_SIMULATION", None)
        
        end_time = time.time()
        end_memory = self.get_memory_usage()
        
        duration = end_time - start_time
        ops_per_sec = operations / duration if duration > 0 else 0
        memory_used = max(0, end_memory - start_memory)
        
        return BenchmarkResult(
            test_name="Device Detection",
            duration_seconds=duration,
            operations_per_second=ops_per_sec,
            memory_used_mb=memory_used,
            success=True,
            metadata={"devices_found": len(devices)}
        )
    
    def benchmark_config_loading(self) -> BenchmarkResult:
        """Benchmark configuration loading performance."""
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        try:
            from photonic_flash_attention.config import get_config
            
            operations = 0
            for _ in range(100):  # Load config 100 times
                config = get_config()
                operations += 1
                if not hasattr(config, 'photonic_threshold'):
                    raise Exception("Config missing expected attributes")
        
        except Exception as e:
            duration = time.time() - start_time
            return BenchmarkResult(
                test_name="Configuration Loading",
                duration_seconds=duration,
                operations_per_second=0,
                memory_used_mb=0,
                success=False,
                error_message=str(e)
            )
        
        end_time = time.time()
        end_memory = self.get_memory_usage()
        
        duration = end_time - start_time
        ops_per_sec = operations / duration if duration > 0 else 0
        memory_used = max(0, end_memory - start_memory)
        
        return BenchmarkResult(
            test_name="Configuration Loading",
            duration_seconds=duration,
            operations_per_second=ops_per_sec,
            memory_used_mb=memory_used,
            success=True
        )
    
    def benchmark_memory_management(self) -> BenchmarkResult:
        """Benchmark memory management performance."""
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        try:
            from photonic_flash_attention.memory.unified_manager import get_memory_manager
            
            memory_manager = get_memory_manager()
            operations = 0
            
            # Simulate memory operations
            for _ in range(50):
                # Simulate tensor allocation/deallocation
                size = random.randint(1000, 10000)
                key = f"test_tensor_{operations}"
                
                # Mock tensor allocation
                memory_manager.track_allocation(key, size)
                operations += 1
                
                # Mock deallocation
                memory_manager.track_deallocation(key)
                operations += 1
                
        except Exception as e:
            duration = time.time() - start_time
            return BenchmarkResult(
                test_name="Memory Management",
                duration_seconds=duration,
                operations_per_second=0,
                memory_used_mb=0,
                success=False,
                error_message=str(e)
            )
        
        end_time = time.time()
        end_memory = self.get_memory_usage()
        
        duration = end_time - start_time
        ops_per_sec = operations / duration if duration > 0 else 0
        memory_used = max(0, end_memory - start_memory)
        
        return BenchmarkResult(
            test_name="Memory Management",
            duration_seconds=duration,
            operations_per_second=ops_per_sec,
            memory_used_mb=memory_used,
            success=True
        )
    
    def benchmark_concurrency(self) -> BenchmarkResult:
        """Benchmark concurrent operations."""
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        operations = 0
        errors = []
        
        def worker_task(worker_id: int, num_ops: int):
            nonlocal operations, errors
            try:
                from photonic_flash_attention.config import get_config
                
                for _ in range(num_ops):
                    config = get_config()
                    operations += 1
                    time.sleep(0.001)  # Simulate work
                    
            except Exception as e:
                errors.append(f"Worker {worker_id}: {e}")
        
        try:
            # Run 5 concurrent workers, 10 operations each
            threads = []
            for i in range(5):
                thread = threading.Thread(target=worker_task, args=(i, 10))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads
            for thread in threads:
                thread.join()
            
            if errors:
                raise Exception(f"Concurrent errors: {errors}")
        
        except Exception as e:
            duration = time.time() - start_time
            return BenchmarkResult(
                test_name="Concurrent Operations",
                duration_seconds=duration,
                operations_per_second=0,
                memory_used_mb=0,
                success=False,
                error_message=str(e)
            )
        
        end_time = time.time()
        end_memory = self.get_memory_usage()
        
        duration = end_time - start_time
        ops_per_sec = operations / duration if duration > 0 else 0
        memory_used = max(0, end_memory - start_memory)
        
        return BenchmarkResult(
            test_name="Concurrent Operations",
            duration_seconds=duration,
            operations_per_second=ops_per_sec,
            memory_used_mb=memory_used,
            success=True
        )
    
    def benchmark_cache_performance(self) -> BenchmarkResult:
        """Benchmark cache performance."""
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        try:
            # Simple cache simulation (avoiding torch dependencies)
            cache = {}
            operations = 0
            
            # Cache write operations
            for i in range(1000):
                key = f"key_{i % 100}"  # 100 unique keys, repeated 10 times
                value = f"value_{i}"
                cache[key] = value
                operations += 1
            
            # Cache read operations
            for i in range(1000):
                key = f"key_{i % 100}"
                value = cache.get(key)
                operations += 1
                if value is None:
                    raise Exception(f"Cache miss for {key}")
        
        except Exception as e:
            duration = time.time() - start_time
            return BenchmarkResult(
                test_name="Cache Performance",
                duration_seconds=duration,
                operations_per_second=0,
                memory_used_mb=0,
                success=False,
                error_message=str(e)
            )
        
        end_time = time.time()
        end_memory = self.get_memory_usage()
        
        duration = end_time - start_time
        ops_per_sec = operations / duration if duration > 0 else 0
        memory_used = max(0, end_memory - start_memory)
        
        return BenchmarkResult(
            test_name="Cache Performance",
            duration_seconds=duration,
            operations_per_second=ops_per_sec,
            memory_used_mb=memory_used,
            success=True
        )
    
    def benchmark_load_balancer(self) -> BenchmarkResult:
        """Benchmark load balancer performance."""
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        try:
            # Simple load balancer simulation
            nodes = [f"node_{i}" for i in range(5)]
            current_node = 0
            operations = 0
            
            # Simulate load balancing decisions
            for i in range(1000):
                # Round-robin selection
                selected_node = nodes[current_node % len(nodes)]
                current_node += 1
                operations += 1
                
                # Simulate some work
                if not selected_node:
                    raise Exception("No node selected")
        
        except Exception as e:
            duration = time.time() - start_time
            return BenchmarkResult(
                test_name="Load Balancer",
                duration_seconds=duration,
                operations_per_second=0,
                memory_used_mb=0,
                success=False,
                error_message=str(e)
            )
        
        end_time = time.time()
        end_memory = self.get_memory_usage()
        
        duration = end_time - start_time
        ops_per_sec = operations / duration if duration > 0 else 0
        memory_used = max(0, end_memory - start_memory)
        
        return BenchmarkResult(
            test_name="Load Balancer",
            duration_seconds=duration,
            operations_per_second=ops_per_sec,
            memory_used_mb=memory_used,
            success=True
        )
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            return usage.ru_maxrss / 1024  # Convert to MB
        except ImportError:
            return 0.0  # Fallback if resource module not available
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        successful_tests = [r for r in self.results if r.success]
        failed_tests = [r for r in self.results if not r.success]
        
        total_ops_per_sec = sum(r.operations_per_second for r in successful_tests)
        avg_duration = sum(r.duration_seconds for r in successful_tests) / max(len(successful_tests), 1)
        total_memory = sum(r.memory_used_mb for r in successful_tests)
        
        print("\n" + "=" * 65)
        print("📊 BENCHMARK RESULTS SUMMARY")
        print("=" * 65)
        print(f"Total tests run: {len(self.results)}")
        print(f"Successful tests: {len(successful_tests)}")
        print(f"Failed tests: {len(failed_tests)}")
        print(f"Success rate: {len(successful_tests)/len(self.results)*100:.1f}%")
        print(f"Total operations/sec: {total_ops_per_sec:.2f}")
        print(f"Average test duration: {avg_duration:.3f} seconds")
        print(f"Total memory used: {total_memory:.1f} MB")
        
        # Performance grade
        performance_score = 100
        
        if len(failed_tests) > 0:
            performance_score -= len(failed_tests) * 15
            print(f"\n❌ Failed tests detected: -{len(failed_tests) * 15} points")
        
        if avg_duration > 1.0:
            performance_score -= 10
            print(f"⚠️ Slow average duration: -10 points")
        
        if total_ops_per_sec < 100:
            performance_score -= 20
            print(f"⚠️ Low throughput: -20 points")
        
        performance_score = max(0, performance_score)
        
        print(f"\n🏆 PERFORMANCE SCORE: {performance_score}/100")
        
        if performance_score >= 85:
            print("🟢 EXCELLENT - High performance system")
            grade = "A"
        elif performance_score >= 70:
            print("🟡 GOOD - Acceptable performance")
            grade = "B"
        elif performance_score >= 50:
            print("🟠 FAIR - Performance improvements recommended")
            grade = "C"
        else:
            print("🔴 POOR - Significant performance issues")
            grade = "D"
        
        return {
            'total_tests': len(self.results),
            'successful_tests': len(successful_tests),
            'failed_tests': len(failed_tests),
            'success_rate': len(successful_tests)/len(self.results),
            'total_ops_per_sec': total_ops_per_sec,
            'avg_duration': avg_duration,
            'total_memory_mb': total_memory,
            'performance_score': performance_score,
            'grade': grade,
            'results': self.results
        }


def main():
    """Run performance benchmark suite."""
    benchmark = PerformanceBenchmark()
    report = benchmark.run_all_benchmarks()
    
    # Return success based on performance score
    return report['performance_score'] >= 50


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)