#!/usr/bin/env bash
# PixelDiT dataset download scripts.
#
# Usage:
#   bash download_and_prepare.sh [anyword|imagenet1k|all]
#
#   anyword    (default) download the two AnyWord-3M parquet files and convert
#              them to PixelDiT WIDS shards.
#   imagenet1k download the ImageNet-1K dataset (official ILSVRC Hugging Face
#              mirror) as parquet files.
#   all        run both of the above.
#
# Run from any directory after logging in to the Hugging Face CLI.

# ----------------------------- parameters -----------------------------
# AnyWord-3M
DATASET_REPO="tyxsspa/AnyWord-3M"
DATASET_REMOTE_PREFIX="laion"
BATCH_SIZE=256
VERIFY_WORKERS=10

# ImageNet-1K (official ILSVRC mirror on Hugging Face; gated: accept the
# terms at https://huggingface.co/datasets/ILSVRC/imagenet-1k before running)
IMAGENET_REPO="ILSVRC/imagenet-1k"
IMAGENET_INCLUDE_TEST=0   # set to 1 to also download data/test-* (13.6 GB)

DATASET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARQUET_DIR="${DATASET_ROOT}/parquet"
WIDS_DIR="${DATASET_ROOT}/pixeldit_wids_text"
IMAGENET_DIR="${DATASET_ROOT}/imagenet1k"
HF_ENDPOINT="https://hf-mirror.com"   # 国内镜像；境外可直接用 https://huggingface.co
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

usage() {
    echo "Usage: bash $0 [anyword|imagenet1k|all]" >&2
    echo "  anyword    download AnyWord-3M parquet files and convert them to PixelDiT WIDS (default)" >&2
    echo "  imagenet1k download the ImageNet-1K parquet files from ${IMAGENET_REPO}" >&2
    echo "  all        run both" >&2
}

download_anyword() {
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
}

download_imagenet1k() {
    echo "Downloading ImageNet-1K from ${IMAGENET_REPO} ..."
    local extra_args=()
    if [[ "${IMAGENET_INCLUDE_TEST}" != "1" ]]; then
        # --exclude requires huggingface_hub >= 0.23.
        extra_args+=(--exclude "data/test-*")
    fi
    "${HF_COMMAND[@]}" "${IMAGENET_REPO}" \
        --repo-type dataset \
        --local-dir "${IMAGENET_DIR}" \
        "${extra_args[@]+"${extra_args[@]}"}"
    echo "ImageNet-1K parquet files are under ${IMAGENET_DIR}/data/."
}

MODE="${1:-anyword}"
case "${MODE}" in
    anyword)
        download_anyword
        ;;
    imagenet1k)
        download_imagenet1k
        ;;
    all)
        download_anyword
        download_imagenet1k
        ;;
    *)
        usage
        exit 2
        ;;
esac
