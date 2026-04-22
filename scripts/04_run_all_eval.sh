#!/usr/bin/env bash
# Phase 5 orchestration: once main training completes, run teacher eval and
# student eval sequentially on GPU 0. Produces:
#   results/libero_spatial_teacher.json
#   results/libero_spatial_student_r128.json
#   results/libero_spatial_main.json  (combined summary for the paper)
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONUNBUFFERED=1

REPO=/home/ege/icra/nsa-distill
cd "$REPO"

source /home/ege/miniconda3/etc/profile.d/conda.sh
conda activate nsa-distill

STUDENT_CKPT="${STUDENT_CKPT:-$REPO/checkpoints/student_runs/nsa_r128_a0.1_libero_spatial/student_r128_final.pt}"
# 1-hour-budget pipeline default: 5 tasks x 20 trials = 100 rollouts per eval.
# Override with N_TRIALS=50 N_TASKS= (empty) for the full 500-rollout run.
N_TRIALS="${N_TRIALS:-20}"
N_TASKS="${N_TASKS:-5}"

if [[ ! -f "$STUDENT_CKPT" ]]; then
    echo "Missing student checkpoint: $STUDENT_CKPT" >&2
    exit 1
fi

# 1) Teacher baseline
echo "[1/3] TEACHER eval  ($N_TRIALS trials/task)"
N_TRIALS=$N_TRIALS N_TASKS=$N_TASKS NOTE=teacher_main \
    OUT_JSON=$REPO/results/libero_spatial_teacher.json \
    bash "$REPO/scripts/03_eval.sh" teacher

# 2) Student rank-128
echo "[2/3] STUDENT (r=128) eval  ($N_TRIALS trials/task)"
N_TRIALS=$N_TRIALS N_TASKS=$N_TASKS RANK=128 NOTE=student_r128_main \
    OUT_JSON=$REPO/results/libero_spatial_student_r128.json \
    bash "$REPO/scripts/03_eval.sh" student "$STUDENT_CKPT"

# 3) Combine into main.json
echo "[3/3] Combining results -> results/libero_spatial_main.json"
python -m src.eval.summarize \
    --teacher_json $REPO/results/libero_spatial_teacher.json \
    --student_json $REPO/results/libero_spatial_student_r128.json \
    --out $REPO/results/libero_spatial_main.json

echo "[done] Main eval complete."
