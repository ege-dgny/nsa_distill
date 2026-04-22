"""NSA distillation losses for OpenVLA-OFT action head.

Definitions (CLAUDE.md §0, NSA-Net Eqs. 11/13):
  L_null  = sum_i || W_hat_i (h_i^T - h_hat_i^S) ||_2^2     (per-sample mean)
  L_orth  = sum_i || U_i^T U_i - I ||_F^2
  L_KD    = || a_teacher - a_student ||_1                  (mean over elems)
  L_sup   = || a_gt      - a_student ||_1                  (mean over elems)
"""

from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn as nn

from .nsa_linear import NSALinear


def _resolve(model: nn.Module, dotted: str) -> nn.Module:
    m = model
    for p in dotted.split("."):
        if p.isdigit():
            m = m[int(p)]
        else:
            m = getattr(m, p)
    return m


def null_loss(
    teacher_acts: Dict[str, torch.Tensor],
    student_acts: Dict[str, torch.Tensor],
    student: nn.Module,
    target_names: Iterable[str],
) -> torch.Tensor:
    """Sum over targeted layers of mean-squared W_hat (h_t - h_s).

    Both teacher_acts and student_acts hold the input to the layer named by `name`.
    student_acts must retain its grad graph so backprop reaches NSALinear.{U,V,bias}.
    """
    device = next(student.parameters()).device
    total = torch.zeros((), device=device, dtype=torch.float32)
    for name in target_names:
        h_t = teacher_acts[name].to(torch.float32)
        h_s = student_acts[name].to(torch.float32)
        mod = _resolve(student, name)
        if not isinstance(mod, NSALinear):
            raise TypeError(f"null_loss target '{name}' is {type(mod).__name__}, expected NSALinear")
        W_hat = mod.effective_weight().to(torch.float32)
        diff = h_t - h_s
        proj = diff @ W_hat.T
        total = total + proj.pow(2).mean()
    return total


def orth_loss(student: nn.Module, target_names: Iterable[str]) -> torch.Tensor:
    """Orthonormality regularizer on the U columns of every NSALinear in target_names."""
    device = next(student.parameters()).device
    total = torch.zeros((), device=device, dtype=torch.float32)
    for name in target_names:
        mod = _resolve(student, name)
        if not isinstance(mod, NSALinear):
            raise TypeError(f"orth_loss target '{name}' is {type(mod).__name__}, expected NSALinear")
        U = mod.U.to(torch.float32)
        UtU = U.T @ U
        I = torch.eye(U.shape[1], device=U.device, dtype=torch.float32)
        total = total + (UtU - I).pow(2).sum()
    return total


def action_l1_loss(a_pred: torch.Tensor, a_target: torch.Tensor) -> torch.Tensor:
    """Mean L1 over all action elements; works on continuous action chunks."""
    return (a_pred.to(torch.float32) - a_target.to(torch.float32)).abs().mean()
