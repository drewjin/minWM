#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.."; pwd)"
cd "$PROJECT_ROOT"

CONFIG_PATH="${CONFIG_PATH:-Wan21/configs/causal_forcing_dmd_camera.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-./ckpts/Wan21/Action2V/dmd/model.pt}"
DATA_PATH="${DATA_PATH:-Wan21/prompts/demos.txt}"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-output/pipe_forcing/baselines}"
NUM_OUTPUT_FRAMES="${NUM_OUTPUT_FRAMES:-20}"
SEED="${SEED:-0}"
TRAJECTORY="${TRAJECTORY:-w*19}"
TRAJECTORY_PATH="${TRAJECTORY_PATH:-}"
MAX_PROMPTS="${MAX_PROMPTS:--1}"
NO_DECODE="${NO_DECODE:-0}"
OVERWRITE="${OVERWRITE:-0}"

ARGS=()
if [ -n "$TRAJECTORY_PATH" ]; then
  ARGS+=(--trajectory_path "$TRAJECTORY_PATH")
else
  ARGS+=(--trajectory "$TRAJECTORY")
fi
if [ "$NO_DECODE" = "1" ]; then
  ARGS+=(--no_decode)
fi
if [ "$OVERWRITE" = "1" ]; then
  ARGS+=(--overwrite)
fi

python pipe_forcing/wan_dmd_baselines.py \
  --mode append \
  --sp_size 1 \
  --config_path "$CONFIG_PATH" \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --data_path "$DATA_PATH" \
  --output_folder "$OUTPUT_FOLDER" \
  --num_output_frames "$NUM_OUTPUT_FRAMES" \
  --seed "$SEED" \
  --max_prompts "$MAX_PROMPTS" \
  "${ARGS[@]}"

