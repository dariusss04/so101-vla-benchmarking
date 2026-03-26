#!/usr/bin/env bash
# Train SmolVLA via lerobot-train (default pipeline). Pushes model to HF repo.
# Note: lerobot-train does not support true resume (optimizer/scheduler state).
# More flags are available; check with:
#   lerobot-train --help

set -euo pipefail

DATASET_REPO_ID=${DATASET_REPO_ID:-your-username/your-dataset}
DATASET_ROOT=${DATASET_ROOT:-./data/your-dataset}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/smolvla_run}
JOB_NAME=${JOB_NAME:-smolvla_run}
HF_REPO_ID=${HF_REPO_ID:-your-username/smolvla-model}

BATCH_SIZE=${BATCH_SIZE:-4}
STEPS=${STEPS:-60000}
LOG_FREQ=${LOG_FREQ:-50}
SAVE_FREQ=${SAVE_FREQ:-2000}
RESUME=${RESUME:-false}
USE_AMP=${USE_AMP:-true}

lerobot-train \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --policy.type=smolvla \
  --policy.use_amp="${USE_AMP}" \
  --batch_size="${BATCH_SIZE}" \
  --steps="${STEPS}" \
  --log_freq="${LOG_FREQ}" \
  --save_checkpoint=true \
  --save_freq="${SAVE_FREQ}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${JOB_NAME}" \
  --policy.push_to_hub=true \
  --policy.repo_id="${HF_REPO_ID}" \
  --resume="${RESUME}" \
  --policy.pad_language_to=max_length \
  --policy.tokenizer_max_length=48 \
  --policy.n_action_steps=50 \
  --policy.chunk_size=50 \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.train_state_proj=true \
  --policy.load_vlm_weights=true
