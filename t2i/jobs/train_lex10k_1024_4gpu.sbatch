#!/usr/bin/env bash
#SBATCH --job-name=pixeldit-lex10k-1024
#SBATCH --partition=L40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/home/wangye/projects/PixelDiT-master/t2i/output/lex10k_1024/slurm/%x-%j.out
#SBATCH --error=/home/wangye/projects/PixelDiT-master/t2i/output/lex10k_1024/slurm/%x-%j.err

set -euo pipefail

PROJECT_DIR=/home/wangye/projects/PixelDiT-master/t2i
WORK_DIR=${PROJECT_DIR}/output/lex10k_1024
CONFIG_PATH=${PROJECT_DIR}/configs/PixelDiT_1024px_lex10k_finetune.yaml

mkdir -p "${WORK_DIR}/slurm"
cd "${PROJECT_DIR}"

source /home/wangye/anaconda3/etc/profile.d/conda.sh
conda activate ad

export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/wangye/projects/PixelDiT-master${PYTHONPATH:+:${PYTHONPATH}}
export LD_LIBRARY_PATH=/home/wangye/anaconda3/envs/ad/lib:/home/wangye/anaconda3/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export PIXDIT_GEMMA_PATH=/home/wangye/huggingface_ckpts/gemma-2-2b-it
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WIDS_CACHE=${WORK_DIR}/wids_cache
export WIDS_BATCHSAMPLER_CACHE=${WORK_DIR}/batchsampler_cache
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Start: $(date --iso-8601=seconds)"
nvidia-smi

exec /home/wangye/anaconda3/envs/ad/bin/python -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_port=29531 \
    train.py \
    --config_path="${CONFIG_PATH}" \
    --work_dir="${WORK_DIR}" \
    --name=lex10k-1024 \
    --report_to=tensorboard \
    --tracker_project_name=lex10k-1024
