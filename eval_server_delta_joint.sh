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

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-18080}
DATASET_PATH=${DATASET_PATH:-/media/damoxing/datasets/myendless/random-rack2}
PRETRAINED_MODEL=${PRETRAINED_MODEL:-checkpoints/rdt-finetune-astribot-1b/checkpoint-1000}
TEXT_ENCODER_NAME=${TEXT_ENCODER_NAME:-google/t5-v1_1-xxl}
VISION_ENCODER_NAME=${VISION_ENCODER_NAME:-google/siglip-so400m-patch14-384}
NUM_SAMPLES=${NUM_SAMPLES:-100}

SERVER_LOG=logs/rdt_server_delta_joint.log
CLIENT_JSON=logs/rdt_eval_delta_joint.json

python scripts/rdt_http_server.py \
  --host "$HOST" \
  --port "$PORT" \
  --config_path configs/base.yaml \
  --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
  --pretrained_text_encoder_name_or_path "$TEXT_ENCODER_NAME" \
  --pretrained_vision_encoder_name_or_path "$VISION_ENCODER_NAME" \
  --dataset_stat_path configs/dataset_stat.json \
  --dataset_name astribot \
  --action_mode delta_joint \
  --control_frequency 25 \
  --device cuda \
  --dtype bf16 > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
  if kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

for _ in $(seq 1 60); do
  if curl -s "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

python scripts/rdt_mock_client.py \
  --server_url "http://$HOST:$PORT" \
  --dataset_path "$DATASET_PATH" \
  --action_mode delta_joint \
  --num_samples "$NUM_SAMPLES" \
  --timeout_sec 120 \
  --output_json "$CLIENT_JSON"

cat "$CLIENT_JSON"
