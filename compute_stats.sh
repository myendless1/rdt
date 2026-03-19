#!/usr/bin/env bash
set -euo pipefail

SAVE_PATH=${SAVE_PATH:-configs/astribot_stats_eef_pose_delta.json}
DATASET_PATH=${DATASET_PATH:-/media/damoxing/datasets/myendless/pick_up_board_black_bg}
ACTION_MODE=${ACTION_MODE:-eef_pose}
ACTION_TARGET=${ACTION_TARGET:-delta}

python data/compute_dataset_stat_hdf5.py \
  --save_path "$SAVE_PATH" \
  --dataset_path "$DATASET_PATH" \
  --action_mode "$ACTION_MODE" \
  --action_target "$ACTION_TARGET" \

# SAVE_PATH=configs/astribot_stats_joint_delta.json ACTION_MODE=joint ACTION_TARGET=delta bash compute_stats.sh  