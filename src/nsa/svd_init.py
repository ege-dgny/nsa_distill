"""Truncated SVD initialization for NSA low-rank linear factors.

Per CLAUDE.md Phase 2 / NSA-Net §III.A: factorize a teacher weight
W in R^{m x n} as U V^T with U in R^{m x r} and V in R^{n x r}.
Sigma is absorbed into U so only U, V (and the layer bias) remain trainable.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def svd_init(W: torch.Tensor, r: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Truncated-SVD initializer for an NSA low-rank factorization.

    Args:
        W: teacher weight, shape (m, n) (== nn.Linear.weight in standard layout).
        r: target rank (must satisfy 0 < r <= min(m, n)).

    Returns:
        U: (m, r) factor with Sigma already absorbed into the columns.
        V: (n, r) factor.

    The reconstruction is W_hat = U @ V.T (approximates W in Frobenius norm
    by the Eckart-Young theorem).
    """
    if W.dim() != 2:
        raise ValueError(f"svd_init expects a 2D weight, got shape {tuple(W.shape)}")
    m, n = W.shape
    if r <= 0 or r > min(m, n):
        raise ValueError(f"rank {r} out of range for weight shape {(m, n)}")

    W32 = W.detach().to(torch.float32)
    U, S, Vh = torch.linalg.svd(W32, full_matrices=False)
    # Keep U orthonormal (U.T @ U == I) and absorb singular values into V.
    # This matches the NSA-Net reference implementation (low_rank_decom) and
    # makes the orthonormality regularizer ||U.T U - I||_F^2 near zero at init.
    # W_hat = U @ V.T  ==  U @ (Vh.T * S).T  ==  U @ (S @ Vh)  ==  original U S Vh.
    U_r = U[:, :r].contiguous()
    V_r = (Vh[:r, :].T * S[:r]).contiguous()
    return U_r, V_r
