#!/usr/bin/env bash
# Phase 4: train NSA-Distill student on LIBERO-Spatial (single GPU, GPU 0).
# CLAUDE.md §4 defaults: alpha=beta=lambda=0.1, lr=1e-4, rank=128.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONUNBUFFERED=1

REPO=/home/ege/icra/nsa-distill
cd "$REPO"

source /home/ege/miniconda3/etc/profile.d/conda.sh
conda activate nsa-distill

PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-64}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
# Budget: 1 hour wall-clock per run. At ~0.29 st/s this is ~1040 steps; 900 gives margin.
MAX_STEPS="${MAX_STEPS:-900}"
RANK="${RANK:-128}"
ALPHA="${ALPHA:-0.1}"
RUN_NAME="${RUN_NAME:-nsa_r${RANK}_a${ALPHA}_libero_spatial}"
USE_WANDB="${USE_WANDB:-0}"

WANDB_FLAG=""
if [[ "$USE_WANDB" == "1" ]]; then
    WANDB_FLAG="--use_wandb"
fi

python -m src.training.distill \
    --rank "$RANK" \
    --alpha "$ALPHA" \
    --beta 0.1 \
    --lam 0.1 \
    --lr 1e-4 \
    --max_steps "$MAX_STEPS" \
    --per_device_batch_size "$PER_DEVICE_BATCH" \
    --grad_accum_steps "$GRAD_ACCUM" \
    --warmup_steps 100 \
    --ckpt_every 450 \
    --log_every 10 \
    --run_name "$RUN_NAME" \
    $WANDB_FLAG \
    "$@"
