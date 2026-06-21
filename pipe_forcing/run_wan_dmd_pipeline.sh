#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.."; pwd)"
cd "$PROJECT_ROOT"

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

CONFIG_PATH="${CONFIG_PATH:-Wan21/configs/causal_forcing_dmd_camera.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-./ckpts/Wan21/Action2V/dmd/model.pt}"
DATA_PATH="${DATA_PATH:-Wan21/prompts/demos.txt}"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-output/pipe_forcing/wan_dmd_camera}"
NUM_OUTPUT_FRAMES="${NUM_OUTPUT_FRAMES:-20}"
SEED="${SEED:-0}"
TRAJECTORY="${TRAJECTORY:-w*19}"
TRAJECTORY_PATH="${TRAJECTORY_PATH:-}"
MAX_PROMPTS="${MAX_PROMPTS:--1}"
OVERWRITE="${OVERWRITE:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29633}"

TRAJ_ARGS=()
if [ -n "$TRAJECTORY_PATH" ]; then
  TRAJ_ARGS+=(--trajectory_path "$TRAJECTORY_PATH")
else
  TRAJ_ARGS+=(--trajectory "$TRAJECTORY")
fi

OVERWRITE_ARGS=()
if [ "$OVERWRITE" = "1" ]; then
  OVERWRITE_ARGS+=(--overwrite)
fi

echo "=== Pipe Forcing: Wan DMD 4-stage causal camera inference ==="
echo "  Config:     $CONFIG_PATH"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Output:     $OUTPUT_FOLDER"
echo "  Frames:     $NUM_OUTPUT_FRAMES"

torchrun \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  --nproc_per_node=4 \
  pipe_forcing/wan_dmd_pipeline.py \
  --config_path "$CONFIG_PATH" \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --data_path "$DATA_PATH" \
  --output_folder "$OUTPUT_FOLDER" \
  --num_output_frames "$NUM_OUTPUT_FRAMES" \
  --seed "$SEED" \
  --max_prompts "$MAX_PROMPTS" \
  "${OVERWRITE_ARGS[@]}" \
  "${TRAJ_ARGS[@]}"
