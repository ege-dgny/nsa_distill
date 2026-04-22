#!/usr/bin/env bash
# Phase 5: LIBERO-Spatial eval (500 rollouts = 10 tasks x 50 trials).
# Usage:
#   scripts/03_eval.sh teacher            # teacher, 500 rollouts
#   scripts/03_eval.sh student <ckpt>     # student @ rank 128, 500 rollouts
#   N_TRIALS=20 N_TASKS=5 scripts/03_eval.sh teacher   # 100-rollout quick mode
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONUNBUFFERED=1

REPO=/home/ege/icra/nsa-distill
cd "$REPO"

source /home/ege/miniconda3/etc/profile.d/conda.sh
conda activate nsa-distill

TEACHER_DIR="${TEACHER_DIR:-$REPO/checkpoints/teacher_libero_spatial}"
N_TRIALS="${N_TRIALS:-50}"
N_TASKS="${N_TASKS:-}"
MODE="${1:-teacher}"
RANK="${RANK:-128}"
NOTE="${NOTE:-$MODE}"

TASKS_FLAG=""
if [[ -n "$N_TASKS" ]]; then
    TASKS_FLAG="--num_tasks $N_TASKS"
fi

if [[ "$MODE" == "teacher" ]]; then
    OUT_JSON="${OUT_JSON:-$REPO/results/libero_spatial_teacher.json}"
    python -m src.eval.libero_eval \
        --teacher_dir "$TEACHER_DIR" \
        --num_trials_per_task "$N_TRIALS" \
        --task_suite libero_spatial \
        --run_id_note "$NOTE" \
        --out_json "$OUT_JSON" \
        --measure_hz \
        $TASKS_FLAG
elif [[ "$MODE" == "student" ]]; then
    CKPT="${2:-}"
    if [[ -z "$CKPT" ]]; then
        echo "Usage: $0 student <checkpoint.pt>" >&2
        exit 1
    fi
    OUT_JSON="${OUT_JSON:-$REPO/results/libero_spatial_student_r${RANK}.json}"
    python -m src.eval.libero_eval \
        --teacher_dir "$TEACHER_DIR" \
        --student_ckpt "$CKPT" \
        --student_rank "$RANK" \
        --num_trials_per_task "$N_TRIALS" \
        --task_suite libero_spatial \
        --run_id_note "$NOTE" \
        --out_json "$OUT_JSON" \
        --measure_hz \
        $TASKS_FLAG
else
    echo "Unknown mode: $MODE (expected teacher|student)" >&2
    exit 1
fi
