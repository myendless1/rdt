#!/usr/bin/env bash
set -euo pipefail

source /mnt/dataset/tyqlsq/miniconda3/bin/activate rdt

cd /media/damoxing/fileset/vla_test/RoboticsDiffusionTransformer
mkdir -p logs

if ! lsof -iTCP:7890 -sTCP:LISTEN >/dev/null 2>&1; then
  /media/damoxing/fileset/clash/clash -d /media/damoxing/fileset/clash/ &
fi
export http_proxy=http://localhost:7890 
export https_proxy=http://localhost:7890 
export no_proxy=localhost,127.0.0.1

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-18081}
DATASET_PATH=${DATASET_PATH:-/media/damoxing/datasets/myendless/pick_up_board_black_bg}
PRETRAINED_MODEL=${PRETRAINED_MODEL:-checkpoints/rdt-1b-finetune-astribot-eef/checkpoint-5000}
TEXT_ENCODER_NAME=${TEXT_ENCODER_NAME:-google/t5-v1_1-xxl}
VISION_ENCODER_NAME=${VISION_ENCODER_NAME:-google/siglip-so400m-patch14-384}
NUM_SAMPLES=${NUM_SAMPLES:-8}
DATASET_STAT_PATH=${DATASET_STAT_PATH:-configs/astribot_stats_eef_pose_delta.json}


python scripts/rdt_http_server.py \
  --host "$HOST" \
  --port "$PORT" \
  --config_path configs/base.yaml \
  --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
  --pretrained_text_encoder_name_or_path "$TEXT_ENCODER_NAME" \
  --pretrained_vision_encoder_name_or_path "$VISION_ENCODER_NAME" \
  --dataset_stat_path "$DATASET_STAT_PATH" \
  --dataset_name astribot \
  --action_mode eef_pose \
  --control_frequency 25 \
  --device cuda \
  --dtype bf16
