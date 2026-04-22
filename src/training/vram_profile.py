"""VRAM profiler: doubling-search the largest per-device batch size that fits on GPU 0.

We run one forward + backward step per candidate batch size, measuring peak allocated
memory. Once we hit OOM, back off by 20% (rounded down to nearest multiple of 2) and
report the final (batch_size, peak_MB).

Usage:
    python -m src.training.vram_profile [--rank 128] [--max_bs 64]
"""

from __future__ import annotations

import argparse
import gc
import sys
import traceback
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "openvla-oft"))

from src.training.config import TrainConfig  # noqa: E402
from src.training.distill import (  # noqa: E402
    NSA_TARGETS,
    build_student_head,
    load_frozen_proprio_projector,
    load_frozen_vla,
    nsa_step,
    run_backbone_forward,
    student_trainable_params,
)
from src.build_student import load_teacher_action_head  # noqa: E402
from src.training.data import make_libero_dataloader  # noqa: E402


def try_batch_size(
    bs: int,
    cfg: TrainConfig,
    device: torch.device,
    vla,
    proprio_projector,
    teacher_head,
    student_head,
    trainable,
    optimizer,
) -> float:
    """Attempt forward+backward at batch size `bs`; return peak-MB or float('inf') on OOM."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()

    loader, _, _, _ = make_libero_dataloader(
        data_root_dir=cfg.data_root_dir,
        dataset_name=cfg.dataset_name,
        teacher_dir=cfg.teacher_dir,
        batch_size=bs,
        image_sizes=(224, 224),
        use_wrist_image=True,
        use_proprio=cfg.use_proprio,
        shuffle_buffer_size=1024,
        image_aug=False,
    )
    it = iter(loader)
    try:
        batch = next(it)
        actions_hidden_states, gt_actions = run_backbone_forward(
            vla=vla,
            proprio_projector=proprio_projector,
            batch=batch,
            device=device,
            use_proprio=cfg.use_proprio,
        )
        actions_hidden_states = actions_hidden_states.detach()

        total_loss, _ = nsa_step(
            actions_hidden_states=actions_hidden_states,
            gt_actions=gt_actions,
            teacher_head=teacher_head,
            student_head=student_head,
            cfg=cfg,
        )
        total_loss.backward()
        optimizer.zero_grad(set_to_none=True)
        peak = torch.cuda.max_memory_allocated() / 1024**2
        return peak
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        return float("inf")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            gc.collect()
            return float("inf")
        raise
    finally:
        del loader
        del it


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--max_bs", type=int, default=64, help="stop searching beyond this")
    parser.add_argument("--start_bs", type=int, default=2)
    args = parser.parse_args()

    cfg = TrainConfig()
    cfg.rank = args.rank
    device = torch.device("cuda")

    print("[profile] loading VLA + projectors + teacher + student (once)")
    vla = load_frozen_vla(cfg.teacher_dir, device)
    proprio_projector = load_frozen_proprio_projector(cfg.teacher_dir, vla.llm_dim, device)
    teacher_head = load_teacher_action_head(Path(cfg.teacher_dir), vla.llm_dim).to(device)
    for p in teacher_head.parameters():
        p.requires_grad_(False)
    student_head = build_student_head(teacher_head, rank=cfg.rank).to(device)
    trainable = student_trainable_params(student_head, NSA_TARGETS)
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr)

    total_gpu_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
    print(f"[profile] total GPU memory: {total_gpu_mb:.0f} MB")

    results = []
    bs = args.start_bs
    last_ok = None
    while bs <= args.max_bs:
        print(f"[profile] trying batch_size={bs} ...", flush=True)
        try:
            peak = try_batch_size(
                bs, cfg, device, vla, proprio_projector,
                teacher_head, student_head, trainable, optimizer,
            )
        except Exception as e:
            print(f"  ERROR (non-OOM): {e}")
            traceback.print_exc()
            break
        if peak == float("inf"):
            print(f"  bs={bs} OOM")
            break
        else:
            print(f"  bs={bs} peak={peak:.0f} MB  ({peak/total_gpu_mb*100:.1f}% of total)")
            results.append((bs, peak))
            last_ok = bs
            bs *= 2

    if last_ok is None:
        print("[profile] FAIL: even smallest batch size OOMed")
        return

    final_bs = max(2, int(last_ok * 0.8))
    final_bs -= final_bs % 2  # even
    print()
    print("=" * 60)
    print("[profile] RESULTS")
    print("=" * 60)
    for bs, peak in results:
        print(f"  bs={bs:3d}  peak={peak:.0f} MB")
    print()
    print(f"[profile] largest OK batch: {last_ok}")
    print(f"[profile] RECOMMENDED per_device_batch_size={final_bs} (80% of largest OK)")

    grad_accum = max(1, 32 // final_bs)
    print(f"[profile] RECOMMENDED grad_accum_steps={grad_accum} (effective batch = {final_bs * grad_accum})")


if __name__ == "__main__":
    main()
