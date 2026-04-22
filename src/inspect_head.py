"""Phase 1: enumerate the OpenVLA-OFT L1 regression action head and decide
NSA-compression targets.

Loads the teacher model, prints the action_head module tree, sums parameters
per nn.Linear, computes the rank-128 NSA compression ratio for the chosen
targets, and writes results/action_head_shapes.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENVLA_OFT = REPO_ROOT / "openvla-oft"
sys.path.insert(0, str(OPENVLA_OFT))

from prismatic.models.action_heads import L1RegressionActionHead  # noqa: E402

# NSA-compression targets (decided in CLAUDE.md §3 / plan §1.2)
NSA_TARGETS = (
    "model.fc1",
    "model.mlp_resnet_blocks.0.ffn.1",
    "model.mlp_resnet_blocks.1.ffn.1",
)


def collect_linear_shapes(action_head: nn.Module) -> List[Dict]:
    rows = []
    for name, mod in action_head.named_modules():
        if isinstance(mod, nn.Linear):
            params = mod.in_features * mod.out_features + (
                mod.out_features if mod.bias is not None else 0
            )
            rows.append(
                {
                    "name": name,
                    "in_features": mod.in_features,
                    "out_features": mod.out_features,
                    "bias": mod.bias is not None,
                    "params": params,
                    "is_nsa_target": name in NSA_TARGETS,
                }
            )
    return rows


def nsa_param_count(in_f: int, out_f: int, rank: int, has_bias: bool) -> int:
    return rank * (in_f + out_f) + (out_f if has_bias else 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument(
        "--out",
        type=str,
        default=str(REPO_ROOT / "results" / "action_head_shapes.json"),
    )
    parser.add_argument(
        "--llm_dim",
        type=int,
        default=4096,
        help="LLM hidden dim (Llama-2-7B = 4096)",
    )
    args = parser.parse_args()

    head = L1RegressionActionHead(input_dim=args.llm_dim, hidden_dim=args.llm_dim, action_dim=7)
    print("=== L1RegressionActionHead module tree ===")
    print(head)

    rows = collect_linear_shapes(head)
    print("\n=== nn.Linear inventory ===")
    print(f"{'name':45s} {'in':>8s} {'out':>8s} {'bias':>5s} {'params':>14s} {'NSA?':>5s}")
    for r in rows:
        print(
            f"{r['name']:45s} {r['in_features']:>8d} {r['out_features']:>8d} "
            f"{int(r['bias']):>5d} {r['params']:>14,d} {int(r['is_nsa_target']):>5d}"
        )

    teacher_target_params = sum(r["params"] for r in rows if r["is_nsa_target"])
    student_target_params = sum(
        nsa_param_count(r["in_features"], r["out_features"], args.rank, r["bias"])
        for r in rows if r["is_nsa_target"]
    )
    teacher_total = sum(r["params"] for r in rows)

    summary = {
        "rank": args.rank,
        "nsa_targets": list(NSA_TARGETS),
        "teacher_action_head_params": teacher_total,
        "teacher_target_params": teacher_target_params,
        "student_target_params": student_target_params,
        "compression_ratio": (
            teacher_target_params / student_target_params if student_target_params else None
        ),
        "linears": rows,
    }

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2, default=str))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
