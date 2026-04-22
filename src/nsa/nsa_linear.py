"""NSALinear: nn.Linear replacement with U V^T factorization for NSA distillation.

Per CLAUDE.md Phase 2: forward is y = (x @ V) @ U.T + b, equivalent to
y = x @ (U V.T).T + b = x @ W_hat.T + b, with Sigma absorbed into U at init.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .svd_init import svd_init


class NSALinear(nn.Module):
    """Low-rank linear layer with explicit U, V factors for null-space distillation.

    Trainable parameters: U (out_features, rank), V (in_features, rank), bias.
    `effective_weight()` returns U @ V.T for use in null_loss.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if rank <= 0 or rank > min(in_features, out_features):
            raise ValueError(
                f"rank {rank} out of range for in/out features {(in_features, out_features)}"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        self.U = nn.Parameter(torch.empty(out_features, rank))
        self.V = nn.Parameter(torch.empty(in_features, rank))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.U, a=5**0.5)
        nn.init.kaiming_uniform_(self.V, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x @ self.V, self.U, self.bias)

    def effective_weight(self) -> torch.Tensor:
        """Return W_hat = U @ V.T (shape: out_features x in_features)."""
        return self.U @ self.V.T

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int) -> "NSALinear":
        """Build an NSALinear initialized via truncated SVD of `linear.weight`.

        The new module inherits the original linear's dtype/device.
        """
        m, n = linear.out_features, linear.in_features
        bias = linear.bias is not None
        target_dtype = linear.weight.dtype
        target_device = linear.weight.device

        module = cls(in_features=n, out_features=m, rank=rank, bias=bias)
        module.to(device=target_device, dtype=target_dtype)
        U, V = svd_init(linear.weight.data, rank)
        with torch.no_grad():
            module.U.data.copy_(U.to(dtype=target_dtype, device=target_device))
            module.V.data.copy_(V.to(dtype=target_dtype, device=target_device))
            if bias:
                module.bias.data.copy_(linear.bias.data.to(dtype=target_dtype, device=target_device))
        return module

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, bias={self.bias is not None}"
        )
