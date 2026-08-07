# AnyWord-3M dataset preparation

The training job consumes local PixelDiT WIDS shards, not parquet files
directly. `download_and_prepare.sh` downloads the two source parquet files and
then runs the repository converter.

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
bash dataset/download_and_prepare.sh
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
