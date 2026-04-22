"""In-place replacement of nn.Linear submodules with NSALinear factors."""

from __future__ import annotations

from typing import Iterable, List

import torch.nn as nn

from .nsa_linear import NSALinear


def _resolve_parent(model: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    """Return (parent_module, leaf_attr) for a dotted name like 'a.b.c'."""
    parts = dotted_name.split(".")
    parent = model
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    leaf = parts[-1]
    return parent, leaf


def _set_child(parent: nn.Module, leaf: str, new_module: nn.Module) -> None:
    if leaf.isdigit():
        parent[int(leaf)] = new_module
    else:
        setattr(parent, leaf, new_module)


def _get_child(parent: nn.Module, leaf: str) -> nn.Module:
    if leaf.isdigit():
        return parent[int(leaf)]
    return getattr(parent, leaf)


def replace_linears(
    model: nn.Module,
    target_names: Iterable[str],
    rank: int,
) -> List[str]:
    """Swap each named nn.Linear submodule for an NSALinear at the given rank.

    Args:
        model: container module whose submodules will be edited in place.
        target_names: dotted module paths (e.g. 'model.fc1') pointing at nn.Linear leaves.
        rank: NSA rank for every replacement.

    Returns:
        List of names that were successfully replaced (all of `target_names` if no errors).
    """
    replaced: List[str] = []
    for name in target_names:
        parent, leaf = _resolve_parent(model, name)
        original = _get_child(parent, leaf)
        if not isinstance(original, nn.Linear):
            raise TypeError(
                f"Target '{name}' is {type(original).__name__}, expected nn.Linear"
            )
        new_module = NSALinear.from_linear(original, rank=rank)
        _set_child(parent, leaf, new_module)
        replaced.append(name)
    return replaced
