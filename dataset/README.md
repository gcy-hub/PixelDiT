# Dataset preparation (AnyWord-3M / ImageNet-1K)

`download_and_prepare.sh` handles both training datasets in this repository:

- `anyword` (default): downloads the two AnyWord-3M source parquet files and
  converts them to PixelDiT WIDS shards, which the T2I training job consumes
  directly.
- `imagenet1k`: downloads the ImageNet-1K (ILSVRC 2012) dataset as parquet
  files from the official Hugging Face mirror.

## Prerequisites

1. Install the Hugging Face CLI (`hf` or `huggingface-cli`) in the environment
   you intend to use. The script does not activate a Python environment.
2. Accept the AnyWord-3M dataset access terms, if the dataset repository is
   gated, and run `hf auth login`.
3. Make sure there is enough disk space for both parquet files and the WIDS
   output. The original parquet files are retained; conversion is read-only.

## One-command preparation

From the project root:

```bash
# AnyWord-3M -> PixelDiT WIDS (default)
bash dataset/download_and_prepare.sh

# ImageNet-1K parquet download
bash dataset/download_and_prepare.sh imagenet1k

# Both
bash dataset/download_and_prepare.sh all
```

The script has an editable parameter block at the top:

```bash
DATASET_REPO="tyxsspa/AnyWord-3M"
DATASET_REMOTE_PREFIX="laion"
DATASET_ROOT=".../dataset"
```

If the hosting mirror stores the files at the repository root, set
`DATASET_REMOTE_PREFIX=""`. The downloaded files are placed at:

```text
dataset/parquet/train_3.parquet
dataset/parquet/train_4.parquet
```

## ImageNet-1K download

```bash
bash dataset/download_and_prepare.sh imagenet1k
```

The ImageNet-1K files come from the official ILSVRC Hugging Face mirror
[ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k).
Notes:

1. The repository is gated: accept the access terms on the dataset page
   ([hf-mirror](https://hf-mirror.com/datasets/ILSVRC/imagenet-1k) /
   [huggingface.co](https://huggingface.co/datasets/ILSVRC/imagenet-1k)) and
   run `hf auth login` before downloading.
2. Reserve ~167 GB of disk space: the training split is ~146.5 GB (294 parquet
   files) and the validation split ~6.7 GB. The ~13.6 GB test split is skipped
   unless you set `IMAGENET_INCLUDE_TEST=1` in the parameter block.
   Skipping it needs `--exclude`, which requires `huggingface_hub >= 0.23`; on
   older CLI versions remove the flag or upgrade.
3. Downloads default to the China mirror
   `HF_ENDPOINT="https://hf-mirror.com"`; switch it back to
   `https://huggingface.co` in the parameter block if you are outside China.
4. The downloaded files land at `dataset/imagenet1k/data/`:

```text
dataset/imagenet1k/
├── classes.py
└── data/
    ├── train-00000-of-00294.parquet
    ├── ...
    ├── validation-00000-of-00014.parquet
    └── ...
```

Each parquet row stores the image bytes, label, and file name, so the splits
can be reconstructed into the per-class folder layout that the
[c2i ImageNet pipeline](../c2i/README.md) expects (REPA-E preprocessing).
If you prefer the original tarballs, download `ILSVRC2012_img_train.tar`,
`ILSVRC2012_img_val.tar`, and `ILSVRC2012_devkit_t12.tar.gz` from
[image-net.org](https://image-net.org/) after logging in.

## Conversion output

The converter is [../t2i/tools/convert_anyword_parquet_to_wids.py](../t2i/tools/convert_anyword_parquet_to_wids.py).
It reads parquet batches, validates the image dimensions, and writes:

```text
dataset/pixeldit_wids_text/
├── conversion-manifest.json
├── wids-meta.json
└── shards/
    ├── train_3-00000.tar
    └── ...
```

Each tar sample is a pair of members with the same key:

```text
train_3-000000000.jpg
train_3-000000000.json
```

The image bytes are preserved without transcoding. JSON keeps the original
caption, OCR annotations, extracted `texts`, image dimensions, source parquet
and row, and `wm_score`. The `prompt` field is the caption used by PixelDiT:
valid OCR strings are made explicit with text such as `with the visible words
"PRODUCT" and "NEW" clearly rendered in the image.`

No rows are filtered by `wm_score`, and the source parquet files are never
deleted, moved, renamed, overwritten, or updated. The conversion manifest
records size, mtime, and SHA-256 before and after conversion. Re-running the
converter skips inputs already recorded as complete and verifies their shards.

To verify an existing conversion without reading rows again:

```bash
python t2i/tools/convert_anyword_parquet_to_wids.py \
  --input_dir dataset/parquet \
  --output_dir dataset/pixeldit_wids_text \
  --verify_only
```

The training split contains 215,552 source rows. The training config excludes
five fixed held-out rows for validation, leaving 215,547 training samples.
