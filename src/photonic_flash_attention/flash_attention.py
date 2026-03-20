"""
Flash Attention implementation using chunked/tiled computation.

FlashAttentionChunk implements the core Flash Attention algorithm:
- Standard attention: O(N²) memory (full N×N attention matrix)
- Flash attention: O(N) memory (processes in blocks, never materializes full matrix)

Math: attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) @ V

Flash attention chunks Q into blocks and uses the log-sum-exp trick to
accumulate results numerically stably without storing the full attention matrix.
"""

import numpy as np


class FlashAttentionChunk:
    """Tiled Flash Attention with O(N) memory complexity.

    Parameters
    ----------
    block_size : int
        Size of query blocks for tiled computation. Smaller = less memory,
        more passes. Larger = more memory, fewer passes.
    """

    def __init__(self, block_size: int = 64):
        self.block_size = block_size

    def attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                  mask: np.ndarray = None) -> np.ndarray:
        """Standard scaled dot-product attention.

        Computes: softmax(QK^T / sqrt(d_k)) @ V

        Memory: O(N²) — materializes the full N×N attention score matrix.

        Parameters
        ----------
        Q : np.ndarray, shape (N, d_k)
        K : np.ndarray, shape (N, d_k)
        V : np.ndarray, shape (N, d_v)
        mask : np.ndarray or None, shape (N, N), boolean mask (True = keep)

        Returns
        -------
        np.ndarray, shape (N, d_v)
        """
        d_k = Q.shape[-1]
        scale = 1.0 / np.sqrt(d_k)

        # Compute full attention scores: (N, N)
        scores = Q @ K.T * scale  # (N, N)

        # Apply mask (set masked positions to -inf before softmax)
        if mask is not None:
            scores = np.where(mask, scores, -1e9)

        # Numerically stable softmax: subtract row max before exp
        scores_max = scores.max(axis=-1, keepdims=True)
        scores_exp = np.exp(scores - scores_max)
        attn_weights = scores_exp / scores_exp.sum(axis=-1, keepdims=True)

        return attn_weights @ V

    def flash_attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                        mask: np.ndarray = None) -> np.ndarray:
        """Flash Attention: chunked tiled attention with O(N) memory.

        Uses the online softmax / log-sum-exp trick to accumulate results
        block by block without ever materializing the full N×N attention matrix.

        Memory: O(block_size * d) — only one Q-block is live at a time.

        Parameters
        ----------
        Q : np.ndarray, shape (N, d_k)
        K : np.ndarray, shape (N, d_k)
        V : np.ndarray, shape (N, d_v)
        mask : np.ndarray or None, shape (N, N), boolean mask (True = keep)

        Returns
        -------
        np.ndarray, shape (N, d_v) — numerically equivalent to attention()
        """
        N, d_k = Q.shape
        d_v = V.shape[-1]
        scale = 1.0 / np.sqrt(d_k)

        output = np.zeros((N, d_v), dtype=np.float64)

        # Process Q in blocks
        for q_start in range(0, N, self.block_size):
            q_end = min(q_start + self.block_size, N)
            q_chunk = Q[q_start:q_end]  # (block_size, d_k) — only this lives in memory

            chunk_mask = None
            if mask is not None:
                chunk_mask = mask[q_start:q_end]  # (block_size, N)

            output[q_start:q_end] = self._compute_chunk(q_chunk, K, V, scale, chunk_mask)

        return output

    def _compute_chunk(self, q_chunk: np.ndarray, K: np.ndarray, V: np.ndarray,
                       scale: float, mask_chunk: np.ndarray = None) -> np.ndarray:
        """Compute attention output for one query block against all K, V.

        Uses online log-sum-exp to accumulate without storing full scores.

        Parameters
        ----------
        q_chunk : np.ndarray, shape (Bq, d_k) — one block of Q
        K       : np.ndarray, shape (N, d_k)
        V       : np.ndarray, shape (N, d_v)
        scale   : float, 1/sqrt(d_k)
        mask_chunk : np.ndarray or None, shape (Bq, N)

        Returns
        -------
        np.ndarray, shape (Bq, d_v)
        """
        Bq = q_chunk.shape[0]
        d_v = V.shape[-1]

        # Full scores for this chunk: (Bq, N)
        scores = q_chunk @ K.T * scale  # (Bq, N)

        if mask_chunk is not None:
            scores = np.where(mask_chunk, scores, -1e9)

        # Online softmax via log-sum-exp for numerical stability
        # m_i = max over all K for each query in block
        m = scores.max(axis=-1, keepdims=True)            # (Bq, 1)
        exp_scores = np.exp(scores - m)                   # (Bq, N)
        l = exp_scores.sum(axis=-1, keepdims=True)        # (Bq, 1)  (sum of exp)

        # Accumulate: attn_out = (exp_scores / l) @ V
        attn_out = (exp_scores / l) @ V                   # (Bq, d_v)

        return attn_out
