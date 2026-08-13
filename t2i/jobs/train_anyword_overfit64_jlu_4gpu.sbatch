#!/usr/bin/env bash
#SBATCH --job-name=pixdit-overfit64
#SBATCH --partition=L40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/home/wangye/projects/PixelDiT-master/t2i/output/anyword_overfit64_jlu/slurm/%x-%j.out
#SBATCH --error=/home/wangye/projects/PixelDiT-master/t2i/output/anyword_overfit64_jlu/slurm/%x-%j.err

set -euo pipefail

PROJECT_DIR=/home/wangye/projects/PixelDiT-master/t2i
WORK_DIR=/home/wangye/projects/PixelDiT-master/t2i/output/anyword_overfit64_jlu
CONFIG_PATH=${PROJECT_DIR}/configs/PixelDiT_512px_anyword_overfit64_jlu.yaml

mkdir -p "${WORK_DIR}/slurm"
cd "${PROJECT_DIR}"
source /home/wangye/anaconda3/etc/profile.d/conda.sh
conda activate ad

export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/wangye/projects/PixelDiT-master${PYTHONPATH:+:${PYTHONPATH}}
export PIXDIT_GPU_IDS=0,1,2,3
export PIXDIT_ENV_PREFIX=/home/wangye/anaconda3/envs/ad
export PIXDIT_GEMMA_PATH=/home/wangye/huggingface_ckpts/gemma-2-2b-it
export PIXDIT_MODEL_PATH=/home/wangye/projects/PixelDiT-1300M-1024px/pixeldit_t2i_v1.pth
export PIXDIT_CONFIG_PATH=${CONFIG_PATH}
export PIXDIT_WORK_DIR=${WORK_DIR}
export PIXDIT_RESUME=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WIDS_CACHE=${WORK_DIR}/wids_cache
export WIDS_BATCHSAMPLER_CACHE=${WORK_DIR}/batchsampler_cache
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES before launcher: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Start: $(date --iso-8601=seconds)"
nvidia-smi

exec bash "${PROJECT_DIR}/run_anyword_rgb_finetune.sh"
