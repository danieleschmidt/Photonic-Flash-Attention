"""
Benchmark suite comparing standard vs Flash Attention.

Measures wall-clock time and verifies numerical equivalence.
"""

import time
import numpy as np
from typing import Dict, List, Optional

from .flash_attention import FlashAttentionChunk


class BenchmarkSuite:
    """Benchmark standard attention vs Flash Attention.

    Parameters
    ----------
    seq_lengths : list of int, optional
        Sequence lengths to benchmark. Default: [64, 128, 256, 512, 1024]
    d_models : list of int, optional
        Model dimensions to benchmark. Default: [64]
    """

    def __init__(self, seq_lengths: Optional[List[int]] = None,
                 d_models: Optional[List[int]] = None):
        self.seq_lengths = seq_lengths or [64, 128, 256, 512, 1024]
        self.d_models = d_models or [64]

    def benchmark_attention(self, seq_len: int, d_model: int, d_head: int,
                            num_heads: int = 1) -> Dict:
        """Time both standard and flash attention for given config.

        Parameters
        ----------
        seq_len  : int — sequence length N
        d_model  : int — model dimension
        d_head   : int — per-head dimension
        num_heads: int — number of attention heads

        Returns
        -------
        dict with keys: seq_len, d_model, d_head, num_heads,
                        standard_time_ms, flash_time_ms, speedup, equivalent
        """
        rng = np.random.default_rng(42)
        Q = rng.standard_normal((seq_len, d_head))
        K = rng.standard_normal((seq_len, d_head))
        V = rng.standard_normal((seq_len, d_head))

        fa = FlashAttentionChunk(block_size=64)

        # Warm up
        _ = fa.attention(Q, K, V)
        _ = fa.flash_attention(Q, K, V)

        # Time standard attention
        n_runs = 3
        t0 = time.perf_counter()
        for _ in range(n_runs):
            out_std = fa.attention(Q, K, V)
        standard_time_ms = (time.perf_counter() - t0) / n_runs * 1000

        # Time flash attention
        t0 = time.perf_counter()
        for _ in range(n_runs):
            out_flash = fa.flash_attention(Q, K, V)
        flash_time_ms = (time.perf_counter() - t0) / n_runs * 1000

        equivalent = np.allclose(out_std, out_flash, atol=1e-5)
        speedup = standard_time_ms / flash_time_ms if flash_time_ms > 0 else float('nan')

        return {
            "seq_len": seq_len,
            "d_model": d_model,
            "d_head": d_head,
            "num_heads": num_heads,
            "standard_time_ms": standard_time_ms,
            "flash_time_ms": flash_time_ms,
            "speedup": speedup,
            "equivalent": equivalent,
        }

    def run_suite(self) -> Dict:
        """Run full benchmark suite over all seq_lengths x d_models.

        Returns
        -------
        dict with keys: results (list), all_equivalent (bool)
        """
        results = []
        for d_model in self.d_models:
            for seq_len in self.seq_lengths:
                result = self.benchmark_attention(
                    seq_len=seq_len,
                    d_model=d_model,
                    d_head=d_model,
                )
                results.append(result)

        all_equivalent = all(r["equivalent"] for r in results)
        return {"results": results, "all_equivalent": all_equivalent}

    def print_report(self, results: Dict) -> None:
        """Print a formatted benchmark comparison table.

        Parameters
        ----------
        results : dict, output of run_suite()
        """
        rows = results["results"]
        all_eq = results["all_equivalent"]

        header = f"{'seq_len':>8} {'d_head':>6} {'std(ms)':>10} {'flash(ms)':>10} {'speedup':>8} {'equiv':>6}"
        sep = "-" * len(header)

        print("\n=== Flash Attention Benchmark ===")
        print(header)
        print(sep)
        for r in rows:
            eq = "✓" if r["equivalent"] else "✗"
            print(
                f"{r['seq_len']:>8} {r['d_head']:>6} "
                f"{r['standard_time_ms']:>10.3f} {r['flash_time_ms']:>10.3f} "
                f"{r['speedup']:>8.2f}x {eq:>6}"
            )
        print(sep)
        print(f"All outputs equivalent (atol=1e-5): {'YES' if all_eq else 'NO'}\n")

    def verify_equivalence(self, seq_len: int, d_model: int) -> bool:
        """Check that flash attention == standard attention for given config.

        Parameters
        ----------
        seq_len : int
        d_model : int

        Returns
        -------
        bool — True if outputs match within atol=1e-5
        """
        rng = np.random.default_rng(0)
        Q = rng.standard_normal((seq_len, d_model))
        K = rng.standard_normal((seq_len, d_model))
        V = rng.standard_normal((seq_len, d_model))

        fa = FlashAttentionChunk(block_size=64)
        out_std = fa.attention(Q, K, V)
        out_flash = fa.flash_attention(Q, K, V)

        return bool(np.allclose(out_std, out_flash, atol=1e-5))
