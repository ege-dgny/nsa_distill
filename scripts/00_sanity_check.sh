#!/usr/bin/env bash
# Phase 0 HUMAN CHECKPOINT: 3-rollout teacher sanity eval on LIBERO-Spatial.
# Pinned to GPU 0, single-process.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONUNBUFFERED=1

REPO=/home/ege/icra/nsa-distill
cd "$REPO"

source /home/ege/miniconda3/etc/profile.d/conda.sh
conda activate nsa-distill

python "$REPO/src/sanity_check.py" \
    --teacher_dir "$REPO/checkpoints/teacher_libero_spatial" \
    --num_tasks 1 \
    --num_trials_per_task 3 \
    --task_suite libero_spatial
