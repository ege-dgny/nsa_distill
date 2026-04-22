"""Phase 0 sanity check: run 3 LIBERO-Spatial rollouts on the teacher model.

Wraps OpenVLA-OFT's run_libero_eval helpers (initialize_model, run_task) to
avoid the full 50x10 rollout sweep. Just enough to confirm the pipeline runs.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENVLA_OFT = REPO_ROOT / "openvla-oft"
sys.path.insert(0, str(OPENVLA_OFT))
sys.path.insert(0, str(OPENVLA_OFT / "experiments" / "robot" / "libero"))

# torch>=2.6 defaults torch.load(weights_only=True), but LIBERO's init-state files
# are numpy pickles. Force the legacy behaviour for this trusted local data.
import torch as _torch  # noqa: E402

_orig_torch_load = _torch.load


def _torch_load_compat(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_torch_load(*a, **kw)


_torch.load = _torch_load_compat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--teacher_dir",
        type=str,
        default=str(REPO_ROOT / "checkpoints" / "teacher_libero_spatial"),
    )
    parser.add_argument("--num_tasks", type=int, default=1)
    parser.add_argument("--num_trials_per_task", type=int, default=3)
    parser.add_argument("--task_suite", type=str, default="libero_spatial")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.chdir(str(OPENVLA_OFT))

    from libero.libero import benchmark

    from experiments.robot.libero.run_libero_eval import (
        GenerateConfig,
        initialize_model,
        run_task,
        setup_logging,
        validate_config,
    )
    from experiments.robot.robot_utils import (
        get_image_resize_size,
        set_seed_everywhere,
    )

    cfg = GenerateConfig(
        pretrained_checkpoint=args.teacher_dir,
        task_suite_name=args.task_suite,
        num_trials_per_task=args.num_trials_per_task,
        seed=args.seed,
        run_id_note="sanity",
        use_wandb=False,
    )
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)

    print("Loading teacher VLA + action head...")
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)

    log_file, log_path, run_id = setup_logging(cfg)
    print(f"Logging to {log_path}")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    print(f"Task suite: {cfg.task_suite_name}, {task_suite.n_tasks} tasks total")
    n_tasks = min(args.num_tasks, task_suite.n_tasks)

    total_episodes, total_successes = 0, 0
    for task_id in range(n_tasks):
        total_episodes, total_successes = run_task(
            cfg,
            task_suite,
            task_id,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            total_episodes,
            total_successes,
            log_file,
        )

    sr = total_successes / max(total_episodes, 1)
    print(
        f"\n=== SANITY RESULT === episodes={total_episodes}, successes={total_successes}, "
        f"success_rate={sr:.3f}"
    )
    log_file.close()


if __name__ == "__main__":
    main()
