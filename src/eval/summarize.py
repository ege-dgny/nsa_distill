"""Combine teacher and student eval JSONs into a unified summary for the paper."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict


def load(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_json", type=str, required=True)
    parser.add_argument("--student_json", type=str, required=True)
    parser.add_argument("--ablation_json", type=str, default=None)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    teacher = load(args.teacher_json)
    student = load(args.student_json)

    summary = {
        "teacher": {
            "action_head_params": teacher.get("teacher_head_params"),
            "success_rate": teacher.get("success_rate"),
            "per_task": teacher.get("per_task"),
            "inference_hz": teacher.get("inference_hz"),
            "num_rollouts": teacher.get("total_episodes"),
        },
        "student_r128": {
            "action_head_params": student.get("student_head_params"),
            "success_rate": student.get("success_rate"),
            "per_task": student.get("per_task"),
            "inference_hz": student.get("inference_hz"),
            "num_rollouts": student.get("total_episodes"),
            "compression_ratio": student.get("compression_ratio"),
            "student_ckpt": student.get("student_ckpt"),
        },
        "delta_success_rate_pp": (
            (student.get("success_rate") - teacher.get("success_rate")) * 100
            if teacher.get("success_rate") is not None and student.get("success_rate") is not None
            else None
        ),
    }

    if args.ablation_json and os.path.exists(args.ablation_json):
        abl = load(args.ablation_json)
        summary["student_r128_no_null"] = {
            "action_head_params": abl.get("student_head_params"),
            "success_rate": abl.get("success_rate"),
            "per_task": abl.get("per_task"),
            "inference_hz": abl.get("inference_hz"),
            "num_rollouts": abl.get("total_episodes"),
            "student_ckpt": abl.get("student_ckpt"),
        }
        if summary["student_r128_no_null"]["success_rate"] is not None:
            summary["ablation_delta_pp_wrt_with_null"] = (
                summary["student_r128_no_null"]["success_rate"] - summary["student_r128"]["success_rate"]
            ) * 100

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[summarize] -> {args.out}")
    if "delta_success_rate_pp" in summary:
        print(f"  Teacher SR : {summary['teacher']['success_rate']}")
        print(f"  Student SR : {summary['student_r128']['success_rate']}")
        print(f"  Delta (pp) : {summary['delta_success_rate_pp']}")
    if "student_r128_no_null" in summary:
        print(f"  Abl SR     : {summary['student_r128_no_null']['success_rate']}")


if __name__ == "__main__":
    main()
