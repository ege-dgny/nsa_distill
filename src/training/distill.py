"""NSA-Distill training loop.

Shared frozen backbone (OpenVLA-OFT VLA + proprio projector).
Teacher action head: frozen.
Student action head: NSALinear rank-r replacements on top of the teacher head,
only the three NSA targets' (U, V, bias) parameters are trainable (but we also
train the remaining full-rank layers of the student head — fc2 and the per-block
residual/norm linears — so it can reconcile any SVD residual).

Loss: L_sup + beta * L_KD + alpha * L_null + lambda * L_orth
  L_sup  = L1(gt_actions, student_actions)
  L_KD   = L1(teacher_actions, student_actions)
  L_null = sum_i || W_hat_i (h_t_i - h_s_i) ||_2^2
  L_orth = sum_i || U_i^T U_i - I ||_F^2

The VLA backbone is run once per batch (shared between teacher and student).
Proprio projector is shared and frozen.

Single GPU (GPU 0), AdamW + cosine decay with warmup, BF16 autocast, FP32 loss.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
OPENVLA_OFT = REPO_ROOT / "openvla-oft"
sys.path.insert(0, str(OPENVLA_OFT))
sys.path.insert(0, str(REPO_ROOT))

# torch.load monkey-patch for LIBERO numpy pickles (same as sanity_check.py)
_orig_torch_load = torch.load


def _torch_load_compat(*a, **kw):
    if "weights_only" not in kw:
        kw["weights_only"] = False
    return _orig_torch_load(*a, **kw)


torch.load = _torch_load_compat

from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor  # noqa: E402

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig  # noqa: E402
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction  # noqa: E402
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor  # noqa: E402
from prismatic.models.action_heads import L1RegressionActionHead  # noqa: E402
from prismatic.models.projectors import ProprioProjector  # noqa: E402
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM  # noqa: E402

from src.nsa.hooks import ActivationCapture  # noqa: E402
from src.nsa.losses import action_l1_loss, null_loss, orth_loss  # noqa: E402
from src.nsa.nsa_linear import NSALinear  # noqa: E402
from src.nsa.replace import replace_linears  # noqa: E402
from src.training.config import TrainConfig  # noqa: E402
from src.training.data import make_libero_dataloader  # noqa: E402
from src.build_student import NSA_TARGETS, load_teacher_action_head  # noqa: E402

try:
    import wandb
except ImportError:
    wandb = None


# ---------------------------------------------------------------------------
# Model assembly
# ---------------------------------------------------------------------------


def register_openvla_autoclasses() -> None:
    """OpenVLA-OFT saves with `trust_remote_code=True` classes but we want to load
    via AutoConfig/AutoModel without re-downloading."""
    try:
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    except ValueError:
        pass


def load_frozen_vla(teacher_dir: str, device: torch.device) -> OpenVLAForActionPrediction:
    register_openvla_autoclasses()
    vla = AutoModelForVision2Seq.from_pretrained(
        teacher_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    vla.vision_backbone.set_num_images_in_input(2)
    vla.eval()
    for p in vla.parameters():
        p.requires_grad_(False)
    return vla


def load_frozen_proprio_projector(teacher_dir: str, llm_dim: int, device: torch.device) -> ProprioProjector:
    ckpt_files = sorted(Path(teacher_dir).glob("proprio_projector--*_checkpoint.pt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No proprio_projector--*_checkpoint.pt in {teacher_dir}")
    ckpt_path = ckpt_files[-1]
    state = torch.load(ckpt_path, map_location="cpu")
    state = {k.removeprefix("module."): v for k, v in state.items()}
    proj = ProprioProjector(llm_dim=llm_dim, proprio_dim=PROPRIO_DIM)
    proj.load_state_dict(state, strict=True)
    proj = proj.to(device=device, dtype=torch.bfloat16)
    proj.eval()
    for p in proj.parameters():
        p.requires_grad_(False)
    return proj


def build_student_head(teacher_head: L1RegressionActionHead, rank: int) -> L1RegressionActionHead:
    student = copy.deepcopy(teacher_head)
    replace_linears(student, NSA_TARGETS, rank=rank)
    return student


def student_trainable_params(student: L1RegressionActionHead, target_names: Tuple[str, ...]):
    """We train the full student head (not just the NSA targets) so that the
    downstream full-rank layers can adapt to the low-rank factorization.
    NSA regularizers are still applied only to the NSA target layers."""
    for p in student.parameters():
        p.requires_grad_(True)
    return [p for p in student.parameters() if p.requires_grad]


# ---------------------------------------------------------------------------
# Backbone forward helpers (adapted from openvla-oft/vla-scripts/finetune.py)
# ---------------------------------------------------------------------------


from prismatic.training.train_utils import (  # noqa: E402
    get_current_action_mask,
    get_next_actions_mask,
)


def compute_num_patches(vla: OpenVLAForActionPrediction, use_proprio: bool) -> int:
    n = vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input()
    if use_proprio:
        n += 1
    return n


def run_backbone_forward(
    vla: OpenVLAForActionPrediction,
    proprio_projector: ProprioProjector,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    use_proprio: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Runs the VLA backbone and returns (actions_hidden_states, ground_truth_actions).

    actions_hidden_states: (B, NUM_ACTIONS_CHUNK*ACTION_DIM, D), bf16.
    ground_truth_actions:  (B, NUM_ACTIONS_CHUNK, ACTION_DIM), bf16.
    """
    gt_actions = batch["actions"].to(device).to(torch.bfloat16)
    proprio = batch["proprio"].to(device).to(torch.bfloat16) if use_proprio else None
    labels = batch["labels"].to(device)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = vla(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device),
            labels=labels,
            output_hidden_states=True,
            proprio=proprio,
            proprio_projector=proprio_projector if use_proprio else None,
        )

    gt_token_ids = labels[:, 1:]
    curr_mask = get_current_action_mask(gt_token_ids)
    next_mask = get_next_actions_mask(gt_token_ids)

    num_patches = compute_num_patches(vla, use_proprio=use_proprio)
    last_hidden = output.hidden_states[-1]           # (B, L, D)
    text_hidden = last_hidden[:, num_patches:-1]     # slice off vision + final eos
    B = batch["input_ids"].shape[0]
    actions_hidden_states = (
        text_hidden[curr_mask | next_mask]
        .reshape(B, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
        .to(torch.bfloat16)
    )
    return actions_hidden_states, gt_actions


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


def nsa_step(
    actions_hidden_states: torch.Tensor,
    gt_actions: torch.Tensor,
    teacher_head: L1RegressionActionHead,
    student_head: L1RegressionActionHead,
    cfg: TrainConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """One loss computation. Returns total loss and per-loss scalar metrics."""
    teacher_cap = ActivationCapture(teacher_head, NSA_TARGETS, detach=True)
    student_cap = ActivationCapture(student_head, NSA_TARGETS, detach=False)

    with teacher_cap:
        with torch.no_grad():
            a_t = teacher_head.predict_action(actions_hidden_states)
    with student_cap:
        a_s = student_head.predict_action(actions_hidden_states)

    l_sup = action_l1_loss(a_s, gt_actions)
    l_kd = action_l1_loss(a_s, a_t.detach())
    l_null = null_loss(teacher_cap.acts, student_cap.acts, student_head, NSA_TARGETS)
    l_orth = orth_loss(student_head, NSA_TARGETS)

    total = l_sup + cfg.beta * l_kd + cfg.alpha * l_null + cfg.lam * l_orth
    metrics = {
        "loss": float(total.detach().item()),
        "L_sup": float(l_sup.detach().item()),
        "L_KD": float(l_kd.detach().item()),
        "L_null": float(l_null.detach().item()),
        "L_orth": float(l_orth.detach().item()),
    }
    return total, metrics


def cosine_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    progress = min(progress, 1.0)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = cfg.lr * cfg.cosine_min_lr_ratio
    return min_lr + (cfg.lr - min_lr) * cos


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def save_student(path: Path, student: L1RegressionActionHead, cfg: TrainConfig, step: int, metrics: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": student.state_dict(),
        "rank": cfg.rank,
        "nsa_targets": list(NSA_TARGETS),
        "alpha": cfg.alpha,
        "beta": cfg.beta,
        "lambda": cfg.lam,
        "step": step,
        "metrics": metrics,
    }
    torch.save(payload, path)
    sidecar = path.with_suffix(path.suffix + ".json")
    with open(sidecar, "w") as f:
        json.dump({k: v for k, v in payload.items() if k != "state_dict"}, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.1, help="null-loss weight")
    parser.add_argument("--beta", type=float, default=0.1, help="L_KD weight")
    parser.add_argument("--lam", type=float, default=0.1, help="orth-loss weight")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--run_name", type=str, default="nsa_r128_libero_spatial")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--teacher_dir", type=str, default="")
    parser.add_argument("--data_root_dir", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="libero_spatial_no_noops")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    cfg = TrainConfig()
    cfg.rank = args.rank
    cfg.alpha = args.alpha
    cfg.beta = args.beta
    cfg.lam = args.lam
    cfg.lr = args.lr
    cfg.max_steps = args.max_steps
    cfg.per_device_batch_size = args.per_device_batch_size
    cfg.grad_accum_steps = args.grad_accum_steps
    cfg.warmup_steps = args.warmup_steps
    cfg.ckpt_every = args.ckpt_every
    cfg.log_every = args.log_every
    cfg.run_name = args.run_name
    cfg.use_wandb = args.use_wandb
    cfg.seed = args.seed
    if args.out_dir:
        cfg.out_dir = args.out_dir
    if args.teacher_dir:
        cfg.teacher_dir = args.teacher_dir
    if args.data_root_dir:
        cfg.data_root_dir = args.data_root_dir
    if args.dataset_name:
        cfg.dataset_name = args.dataset_name

    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    run_dir = Path(cfg.out_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2, default=str)

    print("=" * 80)
    print(f"NSA-Distill run: {cfg.run_name}")
    print(f"  rank={cfg.rank}, alpha={cfg.alpha}, beta={cfg.beta}, lambda={cfg.lam}")
    print(f"  lr={cfg.lr}, max_steps={cfg.max_steps}, per-device bs={cfg.per_device_batch_size}, grad_accum={cfg.grad_accum_steps}")
    print(f"  out_dir={run_dir}")
    print("=" * 80)

    if cfg.use_wandb and wandb is not None:
        wandb.init(project=cfg.wandb_project, name=cfg.run_name, config=asdict(cfg))

    print("[load] VLA backbone (frozen)")
    vla = load_frozen_vla(cfg.teacher_dir, device)
    print("[load] proprio_projector (frozen)")
    proprio_projector = load_frozen_proprio_projector(cfg.teacher_dir, vla.llm_dim, device)
    print("[load] teacher action_head (frozen)")
    teacher_head = load_teacher_action_head(Path(cfg.teacher_dir), vla.llm_dim).to(device)
    for p in teacher_head.parameters():
        p.requires_grad_(False)

    print(f"[build] student action_head (rank={cfg.rank})")
    student_head = build_student_head(teacher_head, rank=cfg.rank).to(device)

    # Diagnostic: verify SVD init produces near-orthonormal U at rank=r.
    _orth_init = 0.0
    for _n, _m in student_head.named_modules():
        if isinstance(_m, NSALinear):
            _U = _m.U.float()
            _UtU = _U.T @ _U
            _I = torch.eye(_U.shape[1], device=_U.device)
            _l = (_UtU - _I).pow(2).sum().item()
            _orth_init += _l
            print(f"  [init check] {_n}: U.shape={tuple(_U.shape)} dtype={_m.U.dtype} ||UtU-I||_F^2={_l:.6f}")
    print(f"  [init check] total L_orth @ init = {_orth_init:.6f}  (should be << 1)")

    trainable = student_trainable_params(student_head, NSA_TARGETS)
    n_trainable = sum(p.numel() for p in trainable)
    n_teacher = sum(p.numel() for p in teacher_head.parameters())
    print(f"  student trainable params: {n_trainable:,}")
    print(f"  teacher total params     : {n_teacher:,}")
    print(f"  compression ratio         : {n_teacher / n_trainable:.2f}x")

    optimizer = torch.optim.AdamW(
        trainable,
        lr=cfg.lr,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
    )

    print("[data] building RLDS loader")
    dataloader, _, _, dataset_statistics = make_libero_dataloader(
        data_root_dir=cfg.data_root_dir,
        dataset_name=cfg.dataset_name,
        teacher_dir=cfg.teacher_dir,
        batch_size=cfg.per_device_batch_size,
        image_sizes=(args.image_size, args.image_size),
        use_wrist_image=True,
        use_proprio=cfg.use_proprio,
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )
    def _jsonify(obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_jsonify(v) for v in obj]
        return obj

    with open(run_dir / "dataset_statistics.json", "w") as f:
        json.dump(_jsonify(dataset_statistics), f, indent=2)

    data_iter = iter(dataloader)
    print("[train] starting loop")
    optimizer.zero_grad(set_to_none=True)
    running: Dict[str, float] = {}
    t_start = time.time()

    for step in range(cfg.max_steps):
        lr = cosine_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        micro_metrics = {"loss": 0.0, "L_sup": 0.0, "L_KD": 0.0, "L_null": 0.0, "L_orth": 0.0}

        for _ in range(cfg.grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            actions_hidden_states, gt_actions = run_backbone_forward(
                vla=vla,
                proprio_projector=proprio_projector,
                batch=batch,
                device=device,
                use_proprio=cfg.use_proprio,
            )
            actions_hidden_states = actions_hidden_states.detach()

            total_loss, metrics = nsa_step(
                actions_hidden_states=actions_hidden_states,
                gt_actions=gt_actions,
                teacher_head=teacher_head,
                student_head=student_head,
                cfg=cfg,
            )
            (total_loss / cfg.grad_accum_steps).backward()
            for k in micro_metrics:
                micro_metrics[k] += metrics[k] / cfg.grad_accum_steps

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip).item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        micro_metrics["lr"] = lr
        micro_metrics["grad_norm"] = grad_norm
        for k, v in micro_metrics.items():
            running[k] = running.get(k, 0.0) + v

        if (step + 1) % cfg.log_every == 0:
            elapsed = time.time() - t_start
            sps = (step + 1) / elapsed if elapsed > 0 else 0.0
            avg = {k: v / cfg.log_every for k, v in running.items()}
            msg = (
                f"step {step + 1:>6d}/{cfg.max_steps} | "
                f"loss {avg['loss']:.4f} sup {avg['L_sup']:.4f} kd {avg['L_KD']:.4f} "
                f"null {avg['L_null']:.2e} orth {avg['L_orth']:.2e} | "
                f"lr {avg['lr']:.2e} gn {avg['grad_norm']:.2f} | {sps:.2f} st/s"
            )
            print(msg)
            if cfg.use_wandb and wandb is not None:
                wandb.log({**avg, "step": step + 1, "steps_per_sec": sps})
            running = {}

        if (step + 1) % cfg.ckpt_every == 0 or (step + 1) == cfg.max_steps:
            ck_path = run_dir / f"student_r{cfg.rank}_step{step + 1}.pt"
            save_student(ck_path, student_head, cfg, step + 1, micro_metrics)
            print(f"  [ckpt] {ck_path}")

    final_path = run_dir / f"student_r{cfg.rank}_final.pt"
    save_student(final_path, student_head, cfg, cfg.max_steps, micro_metrics)
    print(f"\n[done] final checkpoint: {final_path}")
    if cfg.use_wandb and wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
