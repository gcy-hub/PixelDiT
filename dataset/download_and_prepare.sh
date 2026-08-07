#!/usr/bin/env bash
# Download the two AnyWord parquet files and convert them to PixelDiT WIDS.
# Run from any directory after logging in to the Hugging Face CLI.

# ----------------------------- parameters -----------------------------
DATASET_REPO="tyxsspa/AnyWord-3M"
DATASET_REMOTE_PREFIX="laion"
DATASET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARQUET_DIR="${DATASET_ROOT}/parquet"
WIDS_DIR="${DATASET_ROOT}/pixeldit_wids_text"
BATCH_SIZE=256
VERIFY_WORKERS=10
HF_ENDPOINT="https://huggingface.co"
# -----------------------------------------------------------------------

set -euo pipefail
export HF_ENDPOINT

if command -v hf >/dev/null 2>&1; then
    HF_COMMAND=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_COMMAND=(huggingface-cli download)
else
    echo "Install the Hugging Face CLI (hf or huggingface-cli) before running this script." >&2
    exit 2
fi

mkdir -p "${PARQUET_DIR}"

for parquet_name in train_3.parquet train_4.parquet; do
    remote_name="${parquet_name}"
    if [[ -n "${DATASET_REMOTE_PREFIX}" ]]; then
        remote_name="${DATASET_REMOTE_PREFIX}/${remote_name}"
    fi
    echo "Downloading ${DATASET_REPO}/${remote_name}"
    "${HF_COMMAND[@]}" "${DATASET_REPO}" "${remote_name}" \
        --repo-type dataset \
        --local-dir "${PARQUET_DIR}/.hf"
    if [[ -f "${PARQUET_DIR}/.hf/${remote_name}" ]]; then
        mv "${PARQUET_DIR}/.hf/${remote_name}" "${PARQUET_DIR}/${parquet_name}"
    elif [[ -f "${PARQUET_DIR}/.hf/${parquet_name}" ]]; then
        mv "${PARQUET_DIR}/.hf/${parquet_name}" "${PARQUET_DIR}/${parquet_name}"
    else
        echo "Downloaded file was not found under ${PARQUET_DIR}/.hf: ${remote_name}" >&2
        exit 1
    fi
done

PROJECT_ROOT="$(cd "${DATASET_ROOT}/.." && pwd)"
CONVERTER="${PROJECT_ROOT}/t2i/tools/convert_anyword_parquet_to_wids.py"
python "${CONVERTER}" \
    --input_dir "${PARQUET_DIR}" \
    --output_dir "${WIDS_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --verify_workers "${VERIFY_WORKERS}"

echo "Prepared PixelDiT WIDS data under ${WIDS_DIR}."
