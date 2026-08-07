#!/usr/bin/env bash
# AnyWord RGB fine-tuning launcher.
# Edit this parameter block. The script intentionally does not activate or
# validate a Python environment; use the environment selected by the caller.

# ----------------------------- parameters -----------------------------
GPU_IDS="0,1"
NUM_GPUS=2
MASTER_PORT=29521

CONFIG_FILE="configs/PixelDiT_512px_anyword_rgb_finetune.yaml"
DATA_DIR="../dataset/pixeldit_wids_text"
HOLDOUT_FILE="configs/anyword_rgb_512_holdout.json"
OUTPUT_DIR="output/anyword_rgb_512"
RUN_NAME="anyword-rgb-512"

# Set RESUME=1 to restore the latest checkpoint and optimizer state from
# OUTPUT_DIR/checkpoints/latest.pth. With RESUME=0, start from BASE_MODEL.
RESUME="${RESUME:-0}"
BASE_MODEL="ckpts/PixelDiT-1300M-1024px/pixeldit_t2i_v1.pth"
GEMMA_DIR="ckpts/gemma-2-2b-it"

TRAIN_EPOCHS=10
TRAIN_BATCH_SIZE=3
NUM_WORKERS=10
LEARNING_RATE=2e-5
SAVE_MODEL_STEPS=1000
EVAL_SAMPLING_STEPS=500
KEEP_RECOVERY_CHECKPOINTS=1
# -----------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SCRIPT_DIR}"

CONFIG_PATH="${SCRIPT_DIR}/${CONFIG_FILE}"
DATA_PATH="${SCRIPT_DIR}/${DATA_DIR}"
HOLDOUT_PATH="${SCRIPT_DIR}/${HOLDOUT_FILE}"
WORK_PATH="${SCRIPT_DIR}/${OUTPUT_DIR}"
MODEL_PATH="${PROJECT_ROOT}/${BASE_MODEL}"
GEMMA_PATH="${PROJECT_ROOT}/${GEMMA_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PIXDIT_GEMMA_PATH="${GEMMA_PATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WIDS_CACHE="${WORK_PATH}/wids_cache"
export WIDS_BATCHSAMPLER_CACHE="${WORK_PATH}/batchsampler_cache"

TRAIN_ARGS=(
    "--config_path=${CONFIG_PATH}"
    "--work_dir=${WORK_PATH}"
    "--name=${RUN_NAME}"
    "--data.data_dir=[${DATA_PATH}]"
    "--data.exclude_indices_path=${HOLDOUT_PATH}"
    "--train.num_epochs=${TRAIN_EPOCHS}"
    "--train.train_batch_size=${TRAIN_BATCH_SIZE}"
    "--train.num_workers=${NUM_WORKERS}"
    "--learning_rate=${LEARNING_RATE}"
    "--train.save_model_steps=${SAVE_MODEL_STEPS}"
    "--train.eval_sampling_steps=${EVAL_SAMPLING_STEPS}"
    "--train.keep_recovery_checkpoints=${KEEP_RECOVERY_CHECKPOINTS}"
    "--report_to=tensorboard"
    "--tracker_project_name=anyword-rgb-512"
)

if [[ "${RESUME}" == "1" ]]; then
    TRAIN_ARGS+=("--resume_from=latest")
else
    TRAIN_ARGS+=("--load_from=${MODEL_PATH}")
fi

python -m torch.distributed.run \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    train.py "${TRAIN_ARGS[@]}"
