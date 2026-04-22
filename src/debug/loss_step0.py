"""Phase 3 HUMAN CHECKPOINT 3: dump all four losses on a single random batch
of action-token hidden states, before any training.

Acceptance:
  L_sup, L_KD finite, sensible scale (action L1 ~ a fraction of typical |action|).
  L_null small (good SVD init).
  L_orth nonzero but finite (Sigma is absorbed into U so U.T@U != I).

This script does NOT require LIBERO observations; it draws random Llama-style
hidden states with similar magnitude. The numbers are scale-only sanity checks,
not held-out validation losses.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
OPENVLA_OFT = REPO_ROOT / "openvla-oft"
sys.path.insert(0, str(OPENVLA_OFT))
sys.path.insert(0, str(REPO_ROOT))

from prismatic.models.action_heads import L1RegressionActionHead  # noqa: E402
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK  # noqa: E402

from src.build_student import NSA_TARGETS, load_teacher_action_head  # noqa: E402
from src.nsa.hooks import ActivationCapture  # noqa: E402
from src.nsa.losses import action_l1_loss, null_loss, orth_loss  # noqa: E402
from src.nsa.replace import replace_linears  # noqa: E402


def build_student(teacher: L1RegressionActionHead, rank: int) -> L1RegressionActionHead:
    student = copy.deepcopy(teacher)
    replace_linears(student, NSA_TARGETS, rank=rank)
    return student


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--teacher_dir",
        type=str,
        default=str(REPO_ROOT / "checkpoints" / "teacher_libero_spatial"),
    )
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--llm_dim", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    teacher = load_teacher_action_head(Path(args.teacher_dir), args.llm_dim).to(device)
    student = build_student(teacher, rank=args.rank).to(device)

    for p in teacher.parameters():
        p.requires_grad_(False)

    chunk_x_action = NUM_ACTIONS_CHUNK * ACTION_DIM
    actions_hidden = torch.randn(
        args.batch, chunk_x_action, args.llm_dim, device=device, dtype=torch.bfloat16
    )

    a_gt = torch.randn(args.batch, NUM_ACTIONS_CHUNK, ACTION_DIM, device=device, dtype=torch.bfloat16) * 0.5

    teacher_cap = ActivationCapture(teacher, NSA_TARGETS, detach=True)
    student_cap = ActivationCapture(student, NSA_TARGETS, detach=False)

    with teacher_cap:
        with torch.no_grad():
            a_t = teacher.predict_action(actions_hidden)
    with student_cap:
        a_s = student.predict_action(actions_hidden)

    l_null = null_loss(teacher_cap.acts, student_cap.acts, student, NSA_TARGETS)
    l_orth = orth_loss(student, NSA_TARGETS)
    l_sup = action_l1_loss(a_s, a_gt)
    l_kd = action_l1_loss(a_s, a_t)

    print("\n=== Step-0 losses (random batch, rank=%d) ===" % args.rank)
    print(f"  L_sup  (action L1 vs random GT)   = {l_sup.item():.6f}")
    print(f"  L_KD   (action L1 vs teacher)     = {l_kd.item():.6f}")
    print(f"  L_null (Sum_i ||W_hat (h_t-h_s)||^2 over targets) = {l_null.item():.6f}")
    print(f"  L_orth (Sum_i ||U.T U - I||_F^2)  = {l_orth.item():.6f}")
    print()
    print("Per-target diagnostics:")
    for name in NSA_TARGETS:
        h_t = teacher_cap.acts[name]
        h_s = student_cap.acts[name].detach()
        diff = (h_t.float() - h_s.float()).abs().mean().item()
        norm = h_t.float().abs().mean().item()
        print(f"  {name}: mean|h_t|={norm:.4f}  mean|h_t - h_s|={diff:.4f}")

    print("\nGradient sanity: backwarding sum of (l_sup + l_kd + l_null + l_orth)...")
    total = l_sup + l_kd + l_null + l_orth
    total.backward()
    grad_norm = 0.0
    n_grads = 0
    for n, p in student.named_parameters():
        if p.grad is not None:
            grad_norm += p.grad.detach().float().pow(2).sum().item()
            n_grads += 1
    grad_norm = grad_norm**0.5
    print(f"Student params with grads: {n_grads}, total grad L2 norm = {grad_norm:.4f}")


if __name__ == "__main__":
    main()
