#!/usr/bin/env bash
# Download all model assets used by the AnyWord RGB training/inference setup.
# Run this script from any directory. It uses the caller's existing HF CLI and
# authentication; it does not create or activate a Python environment.

# ----------------------------- parameters -----------------------------
HF_GEMMA_REPO="google/gemma-2-2b-it"
HF_PIXELDIT_REPO="nvidia/PixelDiT-1300M-1024px"
PIXELDIT_FILE="pixeldit_t2i_v1.pth"
CHECKPOINT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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

mkdir -p "${CHECKPOINT_ROOT}/gemma-2-2b-it" "${CHECKPOINT_ROOT}/PixelDiT-1300M-1024px"

echo "Downloading ${HF_GEMMA_REPO} into ${CHECKPOINT_ROOT}/gemma-2-2b-it"
"${HF_COMMAND[@]}" "${HF_GEMMA_REPO}" \
    --repo-type model \
    --local-dir "${CHECKPOINT_ROOT}/gemma-2-2b-it"

echo "Downloading ${HF_PIXELDIT_REPO}/${PIXELDIT_FILE}"
"${HF_COMMAND[@]}" "${HF_PIXELDIT_REPO}" "${PIXELDIT_FILE}" \
    --repo-type model \
    --local-dir "${CHECKPOINT_ROOT}/PixelDiT-1300M-1024px"

echo "Model assets are ready under ${CHECKPOINT_ROOT}."
