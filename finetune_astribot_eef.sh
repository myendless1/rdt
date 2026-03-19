#!/usr/bin/env bash
set -euo pipefail

# Activate the requested conda environment.
source /mnt/dataset/tyqlsq/miniconda3/bin/activate rdt

# Keep cache on the larger disk and avoid deleting it on each run.
mkdir -p /media/damoxing/ckp/.cache
ln -sfn /media/damoxing/ckp/.cache ~/.cache

# Ensure .netrc is in place for any authenticated downloads (e.g., Hugging Face)
cp /media/damoxing/fileset/.netrc ~/

# # Start clash only if local proxy is not already listening.
# if ! lsof -iTCP:7890 -sTCP:LISTEN >/dev/null 2>&1; then
#   /media/damoxing/fileset/clash/clash -d /media/damoxing/fileset/clash/ &
# fi
# export http_proxy=http://localhost:7890 
# export https_proxy=http://localhost:7890 
# export no_proxy=localhost,127.0.0.1

# Prefer local cache to avoid flaky TLS calls when model files are already present.
# Force deterministic cache paths to avoid inherited broken values like HF_HUB_CACHE=/hub.
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

cd /media/damoxing/fileset/vla_test/RoboticsDiffusionTransformer

export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}

# Overridable training arguments.
export NUM_GPUS=${NUM_GPUS:-1}
export TEXT_ENCODER_NAME=${TEXT_ENCODER_NAME:-google/t5-v1_1-xxl}
export VISION_ENCODER_NAME=${VISION_ENCODER_NAME:-google/siglip-so400m-patch14-384}
export PRETRAINED_MODEL=${PRETRAINED_MODEL:-checkpoint/rdt-1b}
export OUTPUT_DIR=${OUTPUT_DIR:-./checkpoints/rdt-finetune-astribot-1b-eef}
export REPORT_TO=${REPORT_TO:-wandb}
export HDF5_ACTION_MODE=${HDF5_ACTION_MODE:-delta_eef_pose}

# Keep defaults practical for first run; can be overridden externally.
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
export SAMPLE_BATCH_SIZE=${SAMPLE_BATCH_SIZE:-16}
export MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-200000}
export CHECKPOINTING_PERIOD=${CHECKPOINTING_PERIOD:-500}
export CHECKPOINTS_TOTAL_LIMIT=${CHECKPOINTS_TOTAL_LIMIT:-4}
export SAMPLE_PERIOD=${SAMPLE_PERIOD:-100}
export LR=${LR:-1e-4}
export NUM_WORKERS=${NUM_WORKERS:-8}
export TARGET_GPU=${TARGET_GPU:-1}

mkdir -p "$OUTPUT_DIR"
mkdir -p logs

export MASTER_PORT=${MASTER_PORT:-29211}
if lsof -iTCP:"$MASTER_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  for port in $(seq 29211 29350); do
    if ! lsof -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
      MASTER_PORT="$port"
      break
    fi
  done
fi

# Pin to a physical GPU explicitly. Do not combine with --num_gpus, otherwise
# DeepSpeed launcher may override CUDA_VISIBLE_DEVICES and fall back to GPU 0.
deepspeed --include "localhost:${TARGET_GPU}" --master_port="$MASTER_PORT" main.py \
  --deepspeed="./configs/zero2.json" \
  --pretrained_model_name_or_path="$PRETRAINED_MODEL" \
  --pretrained_text_encoder_name_or_path="$TEXT_ENCODER_NAME" \
  --pretrained_vision_encoder_name_or_path="$VISION_ENCODER_NAME" \
  --output_dir="$OUTPUT_DIR" \
  --train_batch_size="$TRAIN_BATCH_SIZE" \
  --sample_batch_size="$SAMPLE_BATCH_SIZE" \
  --max_train_steps="$MAX_TRAIN_STEPS" \
  --checkpointing_period="$CHECKPOINTING_PERIOD" \
  --sample_period="$SAMPLE_PERIOD" \
  --checkpoints_total_limit="$CHECKPOINTS_TOTAL_LIMIT" \
  --lr_scheduler="constant" \
  --learning_rate="$LR" \
  --mixed_precision="bf16" \
  --dataloader_num_workers="$NUM_WORKERS" \
  --image_aug \
  --dataset_type="finetune" \
  --state_noise_snr=40 \
  --load_from_hdf5 \
  --hdf5_action_mode="$HDF5_ACTION_MODE" \
  --report_to="$REPORT_TO" \
  --log_name="rdt_finetune_astribot_eef" \
  --hdf5_action_target delta
