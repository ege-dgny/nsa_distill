"""Phase 5: full LIBERO-Spatial evaluation for teacher and NSA-Distill students.

Wraps OpenVLA-OFT's `initialize_model` and `run_task` to support 3-/100-/500-rollout
modes. When a student checkpoint is provided, the action head is first switched to the
NSA rank-r structure (NSALinear for the targeted layers) and then loaded.

Writes a structured JSON to `results/<out_name>.json` with per-task success rate,
overall success rate, parameter counts, and inference Hz.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
OPENVLA_OFT = REPO_ROOT / "openvla-oft"
sys.path.insert(0, str(OPENVLA_OFT))
sys.path.insert(0, str(OPENVLA_OFT / "experiments" / "robot" / "libero"))
sys.path.insert(0, str(REPO_ROOT))

import torch as _torch  # noqa: E402

_orig_torch_load = _torch.load


def _torch_load_compat(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_torch_load(*a, **kw)


_torch.load = _torch_load_compat


from src.nsa.replace import replace_linears  # noqa: E402
from src.nsa.nsa_linear import NSALinear  # noqa: E402
from src.build_student import NSA_TARGETS  # noqa: E402


def attach_student_head(action_head, student_ckpt_path: str, rank: int, device) -> None:
    """In-place: replace target linears in action_head with NSALinear@rank,
    then load the student state_dict."""
    replace_linears(action_head, NSA_TARGETS, rank=rank)
    action_head = action_head.to(torch.bfloat16).to(device)
    payload = torch.load(student_ckpt_path, map_location="cpu")
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    state = {k.removeprefix("module."): v for k, v in state.items()}
    missing, unexpected = action_head.load_state_dict(state, strict=False)
    if missing:
        print(f"  WARN missing keys: {missing[:5]}")
    if unexpected:
        print(f"  WARN unexpected keys: {unexpected[:5]}")
    action_head.eval()


def count_action_head_params(action_head) -> int:
    return sum(p.numel() for p in action_head.parameters())


def measure_inference_hz(model, action_head, proprio_projector, processor, cfg, resize_size, n_warmup=5, n_trials=20) -> float:
    """Run the model end-to-end on a dummy observation and report average inference Hz."""
    from experiments.robot.openvla_utils import get_vla_action
    import numpy as np

    device = next(model.parameters()).device
    H = W = resize_size if isinstance(resize_size, int) else resize_size[0]
    img = np.zeros((H, W, 3), dtype=np.uint8)
    wrist_img = np.zeros((H, W, 3), dtype=np.uint8)
    proprio = np.zeros(8, dtype=np.float32)
    obs = {
        "full_image": img,
        "wrist_image": wrist_img,
        "state": proprio,
    }
    task_label = "place the object on the plate"

    for _ in range(n_warmup):
        try:
            _ = get_vla_action(
                cfg, model, processor, obs, task_label, action_head,
                proprio_projector, None, use_film=False,
            )
        except Exception as e:
            print(f"[timer] warmup failed: {e}; falling back to default")
            return 0.0

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_trials):
        _ = get_vla_action(
            cfg, model, processor, obs, task_label, action_head,
            proprio_projector, None, use_film=False,
        )
    torch.cuda.synchronize()
    dt = time.time() - t0
    hz = n_trials / dt if dt > 0 else 0.0
    return hz


def run_eval(
    teacher_dir: str,
    num_trials_per_task: int,
    num_tasks: Optional[int],
    task_suite: str,
    run_id_note: str,
    seed: int,
    student_ckpt: Optional[str],
    student_rank: Optional[int],
    out_json: str,
    measure_hz: bool,
) -> Dict:
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
        pretrained_checkpoint=teacher_dir,
        task_suite_name=task_suite,
        num_trials_per_task=num_trials_per_task,
        seed=seed,
        run_id_note=run_id_note,
        use_wandb=False,
    )
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)

    print(f"[eval] loading model from {teacher_dir}")
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)

    teacher_head_params = count_action_head_params(action_head)

    if student_ckpt is not None:
        assert student_rank is not None, "--student_rank is required with --student_ckpt"
        print(f"[eval] attaching student: {student_ckpt} @ rank={student_rank}")
        device = next(action_head.parameters()).device
        attach_student_head(action_head, student_ckpt, student_rank, device)
        student_head_params = count_action_head_params(action_head)
    else:
        student_head_params = None

    log_file, log_path, run_id = setup_logging(cfg)
    print(f"[eval] log file: {log_path}")

    benchmark_dict = benchmark.get_benchmark_dict()
    ts = benchmark_dict[cfg.task_suite_name]()
    n_tasks = ts.n_tasks if num_tasks is None else min(num_tasks, ts.n_tasks)
    print(f"[eval] task_suite={cfg.task_suite_name}: {n_tasks} tasks x {num_trials_per_task} rollouts")

    inference_hz = 0.0
    if measure_hz:
        try:
            print("[eval] measuring inference Hz on dummy observation...")
            inference_hz = measure_inference_hz(
                model, action_head, proprio_projector, processor, cfg, resize_size
            )
            print(f"[eval] inference Hz (warm)  : {inference_hz:.2f}")
        except Exception as e:
            print(f"[eval] hz measurement failed: {e}")

    per_task_sr: List[Dict] = []
    total_episodes, total_successes = 0, 0
    for task_id in range(n_tasks):
        prev_ep, prev_succ = total_episodes, total_successes
        total_episodes, total_successes = run_task(
            cfg, ts, task_id, model, resize_size, processor,
            action_head, proprio_projector, noisy_action_projector,
            total_episodes, total_successes, log_file,
        )
        this_ep = total_episodes - prev_ep
        this_succ = total_successes - prev_succ
        sr = this_succ / max(this_ep, 1)
        per_task_sr.append({
            "task_id": task_id,
            "task_description": ts.get_task(task_id).language,
            "episodes": this_ep,
            "successes": this_succ,
            "success_rate": sr,
        })
        print(f"[eval] task {task_id}: {this_succ}/{this_ep} = {sr:.2%}  (cumul {total_successes}/{total_episodes} = {total_successes/total_episodes:.2%})")

    overall = total_successes / max(total_episodes, 1)
    log_file.close()

    result = {
        "teacher_dir": teacher_dir,
        "student_ckpt": student_ckpt,
        "student_rank": student_rank,
        "task_suite": task_suite,
        "num_tasks": n_tasks,
        "num_trials_per_task": num_trials_per_task,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "success_rate": overall,
        "per_task": per_task_sr,
        "teacher_head_params": teacher_head_params,
        "student_head_params": student_head_params,
        "compression_ratio": teacher_head_params / student_head_params if student_head_params else None,
        "inference_hz": inference_hz,
        "seed": seed,
    }

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[eval] results saved -> {out_json}")
    print(f"[eval] overall success rate: {overall:.2%}  ({total_successes}/{total_episodes})")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_dir", type=str, default=str(REPO_ROOT / "checkpoints" / "teacher_libero_spatial"))
    parser.add_argument("--num_trials_per_task", type=int, default=50)
    parser.add_argument("--num_tasks", type=int, default=None, help="None = all tasks in suite")
    parser.add_argument("--task_suite", type=str, default="libero_spatial")
    parser.add_argument("--run_id_note", type=str, default="eval")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--student_ckpt", type=str, default=None)
    parser.add_argument("--student_rank", type=int, default=None)
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument("--measure_hz", action="store_true")
    args = parser.parse_args()

    run_eval(
        teacher_dir=args.teacher_dir,
        num_trials_per_task=args.num_trials_per_task,
        num_tasks=args.num_tasks,
        task_suite=args.task_suite,
        run_id_note=args.run_id_note,
        seed=args.seed,
        student_ckpt=args.student_ckpt,
        student_rank=args.student_rank,
        out_json=args.out_json,
        measure_hz=args.measure_hz,
    )


if __name__ == "__main__":
    main()
