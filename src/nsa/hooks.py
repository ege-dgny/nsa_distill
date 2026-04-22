"""Forward-pre-hook activation capture for NSA distillation.

`L_null` requires the *input* h_i to each targeted linear (NSA-Net Eq. 11).
Use forward-pre-hooks (not forward-hooks) so we capture the input, not the
output. Teacher captures with detached tensors; student keeps the autograd
graph so grads flow back through U/V/bias of the NSALinear that produced
the downstream activation difference.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn


class ActivationCapture:
    """Context-manager that registers forward-pre-hooks on the named submodules.

    Usage:
        cap = ActivationCapture(model, target_names, detach=True)
        with cap:
            _ = model(batch)
        h = cap.acts['model.fc1']  # shape: input to that linear
    """

    def __init__(
        self,
        model: nn.Module,
        target_names: Iterable[str],
        detach: bool,
    ) -> None:
        self.model = model
        self.target_names = list(target_names)
        self.detach = detach
        self.acts: Dict[str, torch.Tensor] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, name: str):
        def hook(_module: nn.Module, inputs: tuple) -> None:
            if not inputs:
                return
            x = inputs[0]
            if self.detach:
                x = x.detach()
            self.acts[name] = x
        return hook

    def _resolve(self, name: str) -> nn.Module:
        m = self.model
        for p in name.split("."):
            if p.isdigit():
                m = m[int(p)]
            else:
                m = getattr(m, p)
        return m

    def __enter__(self) -> "ActivationCapture":
        self.acts.clear()
        for name in self.target_names:
            mod = self._resolve(name)
            handle = mod.register_forward_pre_hook(self._make_hook(name))
            self._handles.append(handle)
        return self

    def __exit__(self, exc_type, exc, tb) -> Optional[bool]:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False
