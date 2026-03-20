# Photonic Flash Attention

A Python implementation of **Flash Attention** with **photonic hardware stubs** for Mach-Zehnder Interferometer (MZI) mesh acceleration.

## What is Flash Attention?

Flash Attention ([Dao et al. 2022](https://arxiv.org/abs/2205.14135)) computes scaled dot-product attention with **O(N) memory** instead of O(N²) by never materializing the full attention matrix.

**Standard attention:**
```
attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) @ V
```
Memory: O(N²) — stores the full N×N score matrix.

**Flash attention:**  
Chunks Q into blocks, iterates K/V, accumulates using the **log-sum-exp trick** for numerical stability. Memory: O(block_size × d).

## What is Photonic Computing?

Photonic processors use **Mach-Zehnder Interferometer (MZI) meshes** — optical waveguide arrays — to perform matrix-vector multiplications at the speed of light. The [Clements decomposition](https://optica.org/abstract.cfm?uri=optica-3-12-1460) programs an arbitrary unitary on the MZI mesh.

This repo provides:
- A real Flash Attention CPU implementation (numpy, passes all tests)
- `PhotonicMatmul`: matrix multiply with documented stubs showing where a real MZI mesh would execute

## Installation

```bash
pip install -e .
```

## Usage

```python
import numpy as np
from photonic_flash_attention import FlashAttentionChunk, PhotonicMatmul, BenchmarkSuite

# Flash Attention
fa = FlashAttentionChunk(block_size=64)
Q = np.random.randn(1024, 64)
K = np.random.randn(1024, 64)
V = np.random.randn(1024, 64)

out_standard = fa.attention(Q, K, V)        # O(N²) memory
out_flash    = fa.flash_attention(Q, K, V)  # O(N) memory

import numpy as np
assert np.allclose(out_standard, out_flash, atol=1e-5)  # numerically equivalent

# Photonic Matmul (CPU simulation)
pm = PhotonicMatmul(use_photonic=False)
C = pm.matmul(A, B)

# Benchmark
suite = BenchmarkSuite(seq_lengths=[64, 128, 256, 512, 1024])
results = suite.run_suite()
suite.print_report(results)
```

## Demo

```bash
python examples/demo.py
```

## Tests

```bash
pytest tests/ -v
```

13 tests covering:
- Output shapes
- Flash == standard attention (atol=1e-5)
- seq_len=1024 equivalence  
- Causal masking
- Multiple block sizes
- Softmax numerical stability (large scores)
- Photonic matmul shapes and correctness
- Benchmark suite

## Architecture

```
src/photonic_flash_attention/
├── __init__.py          # exports
├── flash_attention.py   # FlashAttentionChunk
├── photonic_matmul.py   # PhotonicMatmul with MZI stubs
└── benchmark.py         # BenchmarkSuite
examples/
└── demo.py              # seq_len=1024 demo
tests/
└── test_flash_attention.py
```

## Flash Attention Algorithm

```
for q_block in chunks(Q, block_size):
    scores = q_block @ K.T / sqrt(d_k)        # (block_size, N)
    m = max(scores, axis=-1)                    # running max for stability
    exp_scores = exp(scores - m)               # numerically stable
    l = sum(exp_scores, axis=-1)               # normalizer
    output[block] = (exp_scores / l) @ V       # weighted sum
```

## References

- Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (2022)
- Clements et al., "An optimal design for universal multiport interferometers" (2016)
- Shen et al., "Deep learning with coherent nanophotonic circuits" (2017)
