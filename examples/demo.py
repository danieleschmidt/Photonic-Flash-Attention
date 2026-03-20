"""
Demo: Photonic Flash Attention on seq_len=1024
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from photonic_flash_attention import FlashAttentionChunk, PhotonicMatmul, BenchmarkSuite


def main():
    print("=" * 60)
    print("Photonic Flash Attention Demo")
    print("=" * 60)

    # Config
    seq_len = 1024
    d_model = 64
    d_head = 64
    num_heads = 1

    print(f"\nConfig: seq_len={seq_len}, d_model={d_model}, d_head={d_head}, num_heads={num_heads}")

    # Generate random Q, K, V
    rng = np.random.default_rng(42)
    Q = rng.standard_normal((seq_len, d_head))
    K = rng.standard_normal((seq_len, d_head))
    V = rng.standard_normal((seq_len, d_head))

    print(f"\nQ shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}")

    # -----------------------------------------------------------------------
    # Standard attention (O(N²) memory)
    # -----------------------------------------------------------------------
    fa = FlashAttentionChunk(block_size=64)

    print("\n[1] Standard Attention (O(N²) memory)...")
    out_std = fa.attention(Q, K, V)
    print(f"    Output shape: {out_std.shape}")
    memory_std = seq_len * seq_len * 8  # float64, N×N matrix in bytes
    print(f"    Attention matrix memory: {memory_std / 1024:.1f} KB  ({seq_len}×{seq_len} float64)")

    # -----------------------------------------------------------------------
    # Flash Attention (O(N) memory)
    # -----------------------------------------------------------------------
    print("\n[2] Flash Attention (O(N) memory, block_size=64)...")
    out_flash = fa.flash_attention(Q, K, V)
    print(f"    Output shape: {out_flash.shape}")
    memory_flash = fa.block_size * d_head * 8 * 3  # Q/K/V chunk in bytes
    print(f"    Peak chunk memory: ~{memory_flash / 1024:.1f} KB  ({fa.block_size}×{d_head} float64)")

    # -----------------------------------------------------------------------
    # Verify equivalence
    # -----------------------------------------------------------------------
    print("\n[3] Equivalence check...")
    max_diff = np.abs(out_std - out_flash).max()
    is_close = np.allclose(out_std, out_flash, atol=1e-5)
    print(f"    Max absolute diff: {max_diff:.2e}")
    print(f"    Within atol=1e-5:  {'✓ PASS' if is_close else '✗ FAIL'}")

    # -----------------------------------------------------------------------
    # Photonic Matmul demo
    # -----------------------------------------------------------------------
    print("\n[4] PhotonicMatmul demo (CPU simulation mode)...")
    pm = PhotonicMatmul(use_photonic=False)
    A = rng.standard_normal((64, 64))
    B = rng.standard_normal((64, 64))
    C = pm.matmul(A, B)
    expected = A @ B
    print(f"    A @ B shape: {C.shape}")
    print(f"    Matches numpy: {'✓' if np.allclose(C, expected) else '✗'}")

    # Batch matmul
    Ab = rng.standard_normal((4, 32, 64))
    Bb = rng.standard_normal((4, 64, 32))
    Cb = pm.batch_matmul(Ab, Bb)
    print(f"    Batch matmul shape: {Cb.shape}")

    # -----------------------------------------------------------------------
    # Benchmark suite
    # -----------------------------------------------------------------------
    print("\n[5] Benchmark suite...")
    suite = BenchmarkSuite(seq_lengths=[64, 128, 256, 512, 1024], d_models=[64])
    results = suite.run_suite()
    suite.print_report(results)

    print("Demo complete.")


if __name__ == "__main__":
    main()
