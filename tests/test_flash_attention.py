"""
Tests for photonic_flash_attention.

Covers:
- FlashAttentionChunk (flash vs standard equivalence, shapes, masks, numerics)
- PhotonicMatmul (shapes, correctness, batched)
- BenchmarkSuite (equivalence, run)
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from photonic_flash_attention import FlashAttentionChunk, PhotonicMatmul, BenchmarkSuite


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_qkv(seq_len, d_head, seed=42):
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((seq_len, d_head))
    K = rng.standard_normal((seq_len, d_head))
    V = rng.standard_normal((seq_len, d_head))
    return Q, K, V


def causal_mask(N):
    """Lower-triangular boolean mask (True = keep)."""
    return np.tril(np.ones((N, N), dtype=bool))


# ---------------------------------------------------------------------------
# FlashAttentionChunk tests
# ---------------------------------------------------------------------------

class TestFlashAttention:

    def test_flash_attention_output_shape(self):
        Q, K, V = make_qkv(128, 64)
        fa = FlashAttentionChunk()
        out = fa.flash_attention(Q, K, V)
        assert out.shape == (128, 64), f"Expected (128, 64), got {out.shape}"

    def test_flash_attention_matches_standard(self):
        Q, K, V = make_qkv(64, 32)
        fa = FlashAttentionChunk(block_size=16)
        out_std = fa.attention(Q, K, V)
        out_flash = fa.flash_attention(Q, K, V)
        assert np.allclose(out_std, out_flash, atol=1e-5), \
            f"Max diff: {np.abs(out_std - out_flash).max():.2e}"

    def test_flash_attention_seq_len_1024(self):
        Q, K, V = make_qkv(1024, 64)
        fa = FlashAttentionChunk(block_size=64)
        out_std = fa.attention(Q, K, V)
        out_flash = fa.flash_attention(Q, K, V)
        assert np.allclose(out_std, out_flash, atol=1e-5), \
            f"seq_len=1024 max diff: {np.abs(out_std - out_flash).max():.2e}"

    def test_flash_attention_with_mask(self):
        N, d = 64, 32
        Q, K, V = make_qkv(N, d)
        mask = causal_mask(N)
        fa = FlashAttentionChunk(block_size=16)
        out_std = fa.attention(Q, K, V, mask=mask)
        out_flash = fa.flash_attention(Q, K, V, mask=mask)
        assert np.allclose(out_std, out_flash, atol=1e-5), \
            f"masked max diff: {np.abs(out_std - out_flash).max():.2e}"

    def test_flash_attention_different_block_sizes(self):
        Q, K, V = make_qkv(128, 32)
        fa_ref = FlashAttentionChunk(block_size=128)  # full sequence = one block
        out_ref = fa_ref.flash_attention(Q, K, V)

        for bs in [1, 8, 16, 32, 64]:
            fa = FlashAttentionChunk(block_size=bs)
            out = fa.flash_attention(Q, K, V)
            assert np.allclose(out_ref, out, atol=1e-5), \
                f"block_size={bs} max diff: {np.abs(out_ref - out).max():.2e}"

    def test_attention_numerics(self):
        """Softmax stability: large scores should not produce NaN/Inf."""
        rng = np.random.default_rng(7)
        Q = rng.standard_normal((32, 16)) * 100  # large values
        K = rng.standard_normal((32, 16)) * 100
        V = rng.standard_normal((32, 16))
        fa = FlashAttentionChunk(block_size=8)
        out_std = fa.attention(Q, K, V)
        out_flash = fa.flash_attention(Q, K, V)
        assert not np.any(np.isnan(out_std)), "NaN in standard attention"
        assert not np.any(np.isnan(out_flash)), "NaN in flash attention"
        assert np.allclose(out_std, out_flash, atol=1e-4), \
            f"large-score max diff: {np.abs(out_std - out_flash).max():.2e}"

    def test_flash_causal_mask(self):
        """Causal mask: each position only attends to itself and earlier."""
        N, d = 32, 16
        Q, K, V = make_qkv(N, d)
        mask = causal_mask(N)
        fa = FlashAttentionChunk(block_size=8)
        out_std = fa.attention(Q, K, V, mask=mask)
        out_flash = fa.flash_attention(Q, K, V, mask=mask)
        assert np.allclose(out_std, out_flash, atol=1e-5), \
            f"causal max diff: {np.abs(out_std - out_flash).max():.2e}"

    def test_large_sequence(self):
        """seq_len=512 should produce matching results."""
        Q, K, V = make_qkv(512, 64, seed=99)
        fa = FlashAttentionChunk(block_size=64)
        out_std = fa.attention(Q, K, V)
        out_flash = fa.flash_attention(Q, K, V)
        assert np.allclose(out_std, out_flash, atol=1e-5), \
            f"seq_len=512 max diff: {np.abs(out_std - out_flash).max():.2e}"


# ---------------------------------------------------------------------------
# PhotonicMatmul tests
# ---------------------------------------------------------------------------

class TestPhotonicMatmul:

    def test_photonic_matmul_shape(self):
        pm = PhotonicMatmul()
        A = np.random.randn(32, 64)
        B = np.random.randn(64, 16)
        C = pm.matmul(A, B)
        assert C.shape == (32, 16), f"Expected (32, 16), got {C.shape}"

    def test_photonic_matmul_correct(self):
        rng = np.random.default_rng(1)
        A = rng.standard_normal((16, 32))
        B = rng.standard_normal((32, 8))
        pm = PhotonicMatmul(use_photonic=False)
        C = pm.matmul(A, B)
        assert np.allclose(C, A @ B, atol=1e-10), "CPU matmul incorrect"

    def test_photonic_batch_matmul(self):
        rng = np.random.default_rng(2)
        A = rng.standard_normal((4, 16, 32))
        B = rng.standard_normal((4, 32, 8))
        pm = PhotonicMatmul()
        C = pm.batch_matmul(A, B)
        assert C.shape == (4, 16, 8), f"Expected (4, 16, 8), got {C.shape}"
        assert np.allclose(C, np.matmul(A, B), atol=1e-10), "Batch matmul incorrect"


# ---------------------------------------------------------------------------
# BenchmarkSuite tests
# ---------------------------------------------------------------------------

class TestBenchmarkSuite:

    def test_benchmark_equivalence(self):
        suite = BenchmarkSuite()
        equiv = suite.verify_equivalence(seq_len=64, d_model=32)
        assert equiv, "Flash attention not equivalent to standard for seq_len=64"

    def test_benchmark_run(self):
        suite = BenchmarkSuite(seq_lengths=[32, 64], d_models=[16])
        results = suite.run_suite()
        assert "results" in results
        assert "all_equivalent" in results
        assert results["all_equivalent"], "Not all benchmark runs produced equivalent outputs"
        assert len(results["results"]) == 2  # 2 seq lengths × 1 d_model

    def test_benchmark_attention_returns_dict(self):
        suite = BenchmarkSuite()
        r = suite.benchmark_attention(seq_len=32, d_model=16, d_head=16)
        for key in ("seq_len", "d_model", "d_head", "standard_time_ms",
                    "flash_time_ms", "speedup", "equivalent"):
            assert key in r, f"Missing key: {key}"
        assert r["equivalent"], "benchmark_attention result not equivalent"
