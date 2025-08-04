#!/usr/bin/env python3
"""
Basic usage example for Photonic Flash Attention.

This example demonstrates:
1. Basic attention computation
2. Performance monitoring
3. Device selection
4. Configuration options
"""

import torch
import time
import matplotlib.pyplot as plt
from photonic_flash_attention import PhotonicFlashAttention, get_device_info


def main():
    print("🚀 Photonic Flash Attention - Basic Usage Example")
    print("=" * 50)
    
    # Check available devices
    device_info = get_device_info()
    print(f"CUDA available: {device_info['cuda_available']}")
    print(f"Photonic available: {device_info['photonic_available']}")
    print()
    
    # Create attention module
    embed_dim = 768
    num_heads = 12
    
    attention = PhotonicFlashAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0.1,
        photonic_threshold=512,  # Use photonics for sequences > 512 tokens
        device='auto',
    )
    
    print(f"Created attention module: {embed_dim}d, {num_heads} heads")
    print()
    
    # Test different sequence lengths
    sequence_lengths = [128, 256, 512, 1024, 2048]
    batch_size = 4
    
    results = []
    
    print("Testing different sequence lengths:")
    print("-" * 70)
    print(f"{'Seq Len':>8} | {'Device':>10} | {'Latency (ms)':>12} | {'Energy (mJ)':>11}")
    print("-" * 70)
    
    for seq_len in sequence_lengths:
        # Create test data
        query = torch.randn(batch_size, seq_len, embed_dim)
        key = torch.randn(batch_size, seq_len, embed_dim)
        value = torch.randn(batch_size, seq_len, embed_dim)
        
        # Warm-up run
        _ = attention(query, key, value)
        
        # Timed run
        start_time = time.perf_counter()
        output = attention(query, key, value)
        end_time = time.perf_counter()
        
        # Get performance stats
        stats = attention.get_performance_stats()
        
        latency_ms = (end_time - start_time) * 1000
        device_used = stats['last_device_used']
        energy_mj = stats.get('last_energy_mj', 0.0)
        
        print(f"{seq_len:>8} | {device_used:>10} | {latency_ms:>12.2f} | {energy_mj:>11.2f}")
        
        results.append({
            'seq_len': seq_len,
            'device': device_used,
            'latency_ms': latency_ms,
            'energy_mj': energy_mj,
        })
        
        # Verify output shape
        assert output.shape == (batch_size, seq_len, embed_dim)
        assert torch.isfinite(output).all()
    
    print("-" * 70)
    print()
    
    # Performance analysis
    analyze_performance(results)
    
    # Device switching demonstration
    demonstrate_device_switching(attention)
    
    # Configuration examples
    demonstrate_configuration()
    
    print("✅ Basic usage example completed successfully!")


def analyze_performance(results):
    """Analyze and visualize performance results."""
    print("📊 Performance Analysis:")
    print("-" * 30)
    
    # Find crossover point
    gpu_results = [r for r in results if r['device'] == 'gpu']
    photonic_results = [r for r in results if r['device'] == 'photonic']
    
    if gpu_results and photonic_results:
        min_photonic_seq = min(r['seq_len'] for r in photonic_results)
        print(f"Photonic acceleration starts at sequence length: {min_photonic_seq}")
        
        # Calculate speedup for photonic sequences
        for result in photonic_results:
            seq_len = result['seq_len']
            gpu_result = next((r for r in gpu_results if r['seq_len'] == seq_len), None)
            if gpu_result:
                speedup = gpu_result['latency_ms'] / result['latency_ms']
                energy_saving = 1 - (result['energy_mj'] / gpu_result['energy_mj']) if gpu_result['energy_mj'] > 0 else 0
                print(f"  Seq {seq_len}: {speedup:.2f}x speedup, {energy_saving:.1%} energy savings")
    
    # Plot results if matplotlib is available
    try:
        plot_performance(results)
    except ImportError:
        print("Matplotlib not available - skipping plot")
    
    print()


def plot_performance(results):
    """Plot performance results."""
    seq_lens = [r['seq_len'] for r in results]
    latencies = [r['latency_ms'] for r in results]
    devices = [r['device'] for r in results]
    
    plt.figure(figsize=(10, 6))
    
    # Color by device
    colors = ['blue' if d == 'gpu' else 'red' for d in devices]
    plt.scatter(seq_lens, latencies, c=colors, alpha=0.7, s=100)
    
    # Add legend
    gpu_line = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='GPU')
    photonic_line = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Photonic')
    plt.legend(handles=[gpu_line, photonic_line])
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Latency (ms)')
    plt.title('Photonic Flash Attention Performance')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.xscale('log')
    
    # Save plot
    plt.tight_layout()
    plt.savefig('performance_plot.png', dpi=150)
    print("📈 Performance plot saved as 'performance_plot.png'")


def demonstrate_device_switching(attention):
    """Demonstrate manual device switching."""
    print("🔄 Device Switching Demonstration:")
    print("-" * 40)
    
    # Create test data
    query = torch.randn(2, 1024, 768)
    
    # Force GPU computation
    attention.enable_photonic(False)
    output_gpu = attention(query)
    stats_gpu = attention.get_performance_stats()
    print(f"Forced GPU: {stats_gpu['last_device_used']} ({stats_gpu['last_latency_ms']:.2f}ms)")
    
    # Force photonic computation (if available)
    attention.enable_photonic(True)
    output_photonic = attention(query)
    stats_photonic = attention.get_performance_stats()
    print(f"Forced Photonic: {stats_photonic['last_device_used']} ({stats_photonic['last_latency_ms']:.2f}ms)")
    
    # Re-enable automatic selection
    attention.enable_photonic(True)  # This enables auto-selection
    output_auto = attention(query)
    stats_auto = attention.get_performance_stats()
    print(f"Auto selection: {stats_auto['last_device_used']} ({stats_auto['last_latency_ms']:.2f}ms)")
    
    # Verify outputs are close (within numerical precision)
    if torch.allclose(output_gpu, output_photonic, atol=1e-3):
        print("✅ GPU and photonic outputs match within tolerance")
    else:
        print("⚠️  GPU and photonic outputs differ (expected for different implementations)")
    
    print()


def demonstrate_configuration():
    """Demonstrate different configuration options."""
    print("⚙️  Configuration Examples:")
    print("-" * 30)
    
    # High-performance configuration
    hp_attention = PhotonicFlashAttention(
        embed_dim=1024,
        num_heads=16,
        dropout=0.0,  # No dropout for inference
        photonic_threshold=256,  # Aggressive photonic usage
        device='auto',
    )
    print("High-performance config: embed_dim=1024, num_heads=16, threshold=256")
    
    # Memory-efficient configuration
    me_attention = PhotonicFlashAttention(
        embed_dim=512,
        num_heads=8,
        dropout=0.1,
        photonic_threshold=1024,  # Conservative photonic usage
        device='auto',
    )
    print("Memory-efficient config: embed_dim=512, num_heads=8, threshold=1024")
    
    # Development configuration (simulation mode)
    import os
    os.environ['PHOTONIC_SIMULATION'] = '1'
    
    dev_attention = PhotonicFlashAttention(
        embed_dim=768,
        num_heads=12,
        device='auto',
    )
    print("Development config: simulation mode enabled")
    
    # Global configuration
    from photonic_flash_attention import set_global_config
    
    set_global_config(
        photonic_threshold=512,
        enable_profiling=True,
        log_level="INFO",
        max_memory_usage=0.8,
    )
    print("Global config: threshold=512, profiling=True, log_level=INFO")
    
    print()


if __name__ == "__main__":
    main()