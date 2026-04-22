"""Phase 2: deep-copy the teacher action head, replace 3 linears with rank-r
NSALinear modules (SVD-init), and verify teacher vs student action L1 on a
random batch of action_hidden_states.

Saves checkpoints/student_r{r}_init.pt with the student state_dict + the list
of NSA-target names + rank.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENVLA_OFT = REPO_ROOT / "openvla-oft"
sys.path.insert(0, str(OPENVLA_OFT))
sys.path.insert(0, str(REPO_ROOT))

from prismatic.models.action_heads import L1RegressionActionHead  # noqa: E402
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK  # noqa: E402

from src.nsa.replace import replace_linears  # noqa: E402

NSA_TARGETS = (
    "model.fc1",
    "model.mlp_resnet_blocks.0.ffn.1",
    "model.mlp_resnet_blocks.1.ffn.1",
)


def load_teacher_action_head(
    teacher_dir: Path,
    llm_dim: int,
    dtype: torch.dtype = torch.bfloat16,
) -> L1RegressionActionHead:
    """Load the L1RegressionActionHead with the teacher's action_head--*_checkpoint.pt weights."""
    head = L1RegressionActionHead(input_dim=llm_dim, hidden_dim=llm_dim, action_dim=ACTION_DIM)
    ckpt_files = sorted(teacher_dir.glob("action_head--*_checkpoint.pt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No action_head--*_checkpoint.pt in {teacher_dir}")
    ckpt_path = ckpt_files[-1]
    print(f"Loading teacher action_head weights from {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if isinstance(state, dict) and "action_head" in state and isinstance(state["action_head"], dict):
        state = state["action_head"]
    state = {k.removeprefix("module."): v for k, v in state.items()}
    missing, unexpected = head.load_state_dict(state, strict=False)
    if missing:
        print(f"  WARN missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  WARN unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    head = head.to(dtype=dtype)
    head.eval()
    return head


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--teacher_dir",
        type=str,
        default=str(REPO_ROOT / "checkpoints" / "teacher_libero_spatial"),
    )
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--llm_dim", type=int, default=4096)
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if not args.out:
        args.out = str(REPO_ROOT / "checkpoints" / f"student_r{args.rank}_init.pt")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    teacher = load_teacher_action_head(Path(args.teacher_dir), args.llm_dim)
    student = copy.deepcopy(teacher)
    replaced = replace_linears(student, NSA_TARGETS, rank=args.rank)
    print(f"Replaced {len(replaced)} linears with NSALinear@rank={args.rank}: {replaced}")

    teacher = teacher.to(device)
    student = student.to(device)

    B = 2
    chunk_x_action = NUM_ACTIONS_CHUNK * ACTION_DIM
    actions_hidden = torch.randn(
        B, chunk_x_action, args.llm_dim, device=device, dtype=torch.bfloat16
    )

    with torch.no_grad():
        a_t = teacher.predict_action(actions_hidden)
        a_s = student.predict_action(actions_hidden)
    diff = (a_t.float() - a_s.float()).abs()
    print(f"\nTeacher action shape: {tuple(a_t.shape)}, student: {tuple(a_s.shape)}")
    print(f"Action L1 (mean abs diff): {diff.mean().item():.6f}")
    print(f"Action L1 (max abs diff):  {diff.max().item():.6f}")
    print(f"Teacher action range:      [{a_t.min().item():.4f}, {a_t.max().item():.4f}]")

    teacher_param_total = sum(p.numel() for p in teacher.parameters())
    student_param_total = sum(p.numel() for p in student.parameters())
    print(f"\nTeacher action_head params: {teacher_param_total:,}")
    print(f"Student action_head params: {student_param_total:,}")
    print(f"Compression ratio (whole head): {teacher_param_total / student_param_total:.2f}x")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    payload = {
        "state_dict": student.state_dict(),
        "rank": args.rank,
        "nsa_targets": list(NSA_TARGETS),
        "llm_dim": args.llm_dim,
        "action_l1_init": diff.mean().item(),
    }
    torch.save(payload, args.out)
    print(f"\nSaved {args.out}")

    sidecar = args.out + ".json"
    with open(sidecar, "w") as f:
        json.dump({k: v for k, v in payload.items() if k != "state_dict"}, f, indent=2)
    print(f"Saved {sidecar}")


if __name__ == "__main__":
    main()
