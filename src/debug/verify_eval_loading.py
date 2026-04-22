"""Validate that a saved student checkpoint will load cleanly into a
freshly-initialized teacher action_head (what eval does).

Run: CUDA_VISIBLE_DEVICES='' python -m src.debug.verify_eval_loading <ckpt.pt>
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
OPENVLA_OFT = REPO_ROOT / "openvla-oft"
sys.path.insert(0, str(OPENVLA_OFT))
sys.path.insert(0, str(REPO_ROOT))

from prismatic.models.action_heads import L1RegressionActionHead  # noqa: E402
from prismatic.vla.constants import ACTION_DIM  # noqa: E402

from src.nsa.replace import replace_linears  # noqa: E402
from src.build_student import NSA_TARGETS  # noqa: E402


def main() -> None:
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    if ckpt is None:
        print("usage: verify_eval_loading.py <student.pt>")
        sys.exit(2)

    print(f"[verify] building fresh L1RegressionActionHead + replace_linears (rank=128)")
    head = L1RegressionActionHead(input_dim=4096, hidden_dim=4096, action_dim=ACTION_DIM)
    head = head.to(torch.bfloat16)
    replace_linears(head, NSA_TARGETS, rank=128)

    print(f"[verify] loading {ckpt}")
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    state = {k.removeprefix("module."): v for k, v in state.items()}

    missing, unexpected = head.load_state_dict(state, strict=False)
    print(f"[verify] missing keys   ({len(missing)}): {missing[:5]}")
    print(f"[verify] unexpected keys({len(unexpected)}): {unexpected[:5]}")
    if not missing and not unexpected:
        print("[verify] PERFECT MATCH — eval loading will succeed.")
    else:
        print("[verify] keys differ; double-check NSA_TARGETS and save/load paths.")

    head_params = sum(p.numel() for p in head.parameters())
    print(f"[verify] student head params: {head_params:,}")


if __name__ == "__main__":
    main()
