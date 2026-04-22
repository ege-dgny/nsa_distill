#!/usr/bin/env bash
# Phase 6: alpha=0 ablation. Train an NSA student WITHOUT the null-space loss
# (same schedule as the main run: 900 steps, bs=64, grad_accum=1 = 1-hour cap),
# then eval it with the same rollout budget as 04_run_all_eval.sh so results
# are directly comparable.
#
# Outputs:
#   checkpoints/student_runs/nsa_r128_a0.0_libero_spatial/student_r128_final.pt
#   results/libero_spatial_ablation.json
#
# Composes with 04_run_all_eval.sh to refresh results/libero_spatial_main.json
# with an extra `student_r128_no_null` block via src/eval/summarize.py.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONUNBUFFERED=1

REPO=/home/ege/icra/nsa-distill
cd "$REPO"

source /home/ege/miniconda3/etc/profile.d/conda.sh
conda activate nsa-distill

N_TRIALS="${N_TRIALS:-20}"
N_TASKS="${N_TASKS:-5}"
MAX_STEPS="${MAX_STEPS:-900}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-64}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
RANK="${RANK:-128}"
RUN_NAME="nsa_r${RANK}_a0.0_libero_spatial"
ABL_CKPT="$REPO/checkpoints/student_runs/$RUN_NAME/student_r${RANK}_final.pt"

TEACHER_JSON="$REPO/results/libero_spatial_teacher.json"
STUDENT_JSON="$REPO/results/libero_spatial_student_r${RANK}.json"
ABL_JSON="$REPO/results/libero_spatial_ablation.json"

echo "[1/3] TRAIN  alpha=0 ablation  ($MAX_STEPS steps, bs=$PER_DEVICE_BATCH)"
MAX_STEPS=$MAX_STEPS \
PER_DEVICE_BATCH=$PER_DEVICE_BATCH \
GRAD_ACCUM=$GRAD_ACCUM \
ALPHA=0.0 \
RANK=$RANK \
RUN_NAME=$RUN_NAME \
    bash "$REPO/scripts/02_train.sh" \
        > "$REPO/logs/train_r${RANK}_ablation.log" 2>&1

if [[ ! -f "$ABL_CKPT" ]]; then
    echo "Ablation training failed: $ABL_CKPT not found" >&2
    tail -n 40 "$REPO/logs/train_r${RANK}_ablation.log"
    exit 1
fi

echo "[2/3] EVAL   alpha=0 ablation  ($N_TRIALS trials/task, $N_TASKS tasks)"
N_TRIALS=$N_TRIALS N_TASKS=$N_TASKS RANK=$RANK NOTE=student_r${RANK}_no_null \
    OUT_JSON=$ABL_JSON \
    bash "$REPO/scripts/03_eval.sh" student "$ABL_CKPT"

echo "[3/3] SUMMARIZE  teacher + student + ablation -> results/libero_spatial_main.json"
if [[ -f "$TEACHER_JSON" && -f "$STUDENT_JSON" ]]; then
    python -m src.eval.summarize \
        --teacher_json "$TEACHER_JSON" \
        --student_json "$STUDENT_JSON" \
        --ablation_json "$ABL_JSON" \
        --out "$REPO/results/libero_spatial_main.json"
else
    echo "NOTE: teacher/student JSONs missing; run scripts/04_run_all_eval.sh first then re-summarize." >&2
fi

echo "[done] Ablation pipeline complete."
