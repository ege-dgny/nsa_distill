"""NSA-Distill training configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TrainConfig:
    # Where the teacher OpenVLA-OFT checkpoint lives
    teacher_dir: str = str(REPO_ROOT / "checkpoints" / "teacher_libero_spatial")

    # RLDS dataset root + name
    data_root_dir: str = str(REPO_ROOT / "datasets" / "rlds")
    dataset_name: str = "libero_spatial_no_noops"

    # NSA hyperparameters (CLAUDE.md §0 / §4)
    rank: int = 128
    alpha: float = 0.1   # null-space loss weight
    beta: float = 0.1    # action-L1 KD weight
    lam: float = 0.1     # orthonormality weight
    nsa_targets: Tuple[str, ...] = (
        "model.fc1",
        "model.mlp_resnet_blocks.0.ffn.1",
        "model.mlp_resnet_blocks.1.ffn.1",
    )

    # Optimization
    lr: float = 1e-4
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.999)
    grad_clip: float = 1.0

    # Schedule
    # 1-hour training budget at ~0.29 st/s -> ~1040 steps; 900 leaves margin for init/ckpt.
    max_steps: int = 900
    warmup_steps: int = 100
    cosine_min_lr_ratio: float = 0.1

    # Batch / accumulation (VRAM profile: bs=64, single-step -> 58 GB peak on 96 GB GPU 0).
    per_device_batch_size: int = 64
    grad_accum_steps: int = 1

    # Data pipeline
    shuffle_buffer_size: int = 100_000
    image_aug: bool = True
    num_images_in_input: int = 2
    use_proprio: bool = True

    # Logging / checkpointing
    log_every: int = 10
    ckpt_every: int = 500
    out_dir: str = str(REPO_ROOT / "checkpoints" / "student_runs")
    run_name: str = "nsa_r128_libero_spatial"

    use_wandb: bool = False
    wandb_entity: str = ""
    wandb_project: str = "nsa-distill"

    seed: int = 7
    device: str = "cuda"

    # Misc
    log_keys: List[str] = field(
        default_factory=lambda: [
            "loss",
            "L_sup",
            "L_KD",
            "L_null",
            "L_orth",
            "lr",
            "grad_norm",
        ]
    )
