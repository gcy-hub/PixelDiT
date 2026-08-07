# PixelDiT AnyWord RGB fine-tuning

This directory is a self-contained packaging of the PixelDiT T2I code for
fine-tuning on AnyWord-3M. It is a copy of the upstream `PixelDiT-master`
tree; all shared-data, checkpoint, and launcher changes described here are
made only in this copy.

## Upstream attribution

This project is based on the official [NVlabs/PixelDiT](https://github.com/NVlabs/PixelDiT)
repository. It is distributed as an independent repository rather than as a
GitHub fork. The upstream license and security files are retained. The custom
work in this repository adds the AnyWord-3M RGB data conversion pipeline, local
checkpoint/data download scripts, portable training launcher, and latest-
checkpoint inference wrapper.

PixelDiT operates directly in RGB pixel space. This fine-tuning setup does not
use a VAE, OCR network, ControlNet, CLIP, or DINOv2. The prompt contains the
caption and the OCR strings, so the model receives the text to render through
the normal Gemma text-conditioning path.

## Directory layout

```text
PixelDiT/
├── ckpts/
│   └── download.sh                         # model downloader
├── dataset/
│   ├── download_and_prepare.sh              # parquet download + conversion
│   ├── README.md
│   ├── parquet/                             # downloaded train_3/train_4
│   └── pixeldit_wids_text/                  # generated WIDS shards
├── t2i/
│   ├── configs/PixelDiT_512px_anyword_rgb_finetune.yaml
│   ├── run_anyword_rgb_finetune.sh          # edit top parameter block
│   ├── infer_latest.py                      # prompt-file inference entrypoint
│   └── tools/convert_anyword_parquet_to_wids.py
└── pixdit_core/
```

The `ckpts/`, `dataset/parquet/`, `dataset/pixeldit_wids_text/`, and
`t2i/output/` directories are runtime data. They are intentionally not part of
the source checkout and can be mounted or generated separately.

## Requirements

Use any Python environment that has a compatible PyTorch/CUDA installation.
The launcher does not activate a conda environment and does not assume a
particular Python path.

```bash
python -m pip install -r requirements.txt
```

The requirements include `pyarrow` for parquet conversion and `tensorboard`
for training logs. The GPU count, CUDA visibility, and distributed launcher
are controlled by the top parameter block in
`t2i/run_anyword_rgb_finetune.sh`.

The download scripts use the Hugging Face CLI (`hf` or `huggingface-cli`). For
the gated Gemma repository, accept its terms and authenticate first:

```bash
hf auth login
```

## 1. Download model checkpoints

Run:

```bash
bash ckpts/download.sh
```

The script downloads:

| Local path | Source |
|---|---|
| `ckpts/gemma-2-2b-it/` | `google/gemma-2-2b-it` |
| `ckpts/PixelDiT-1300M-1024px/pixeldit_t2i_v1.pth` | `nvidia/PixelDiT-1300M-1024px` |

The script does not activate an environment. It uses the caller's existing HF
CLI and token. If a mirror is required, edit the repository variables at the
top of `ckpts/download.sh`.

## 2. Download and prepare AnyWord-3M

Run:

```bash
bash dataset/download_and_prepare.sh
```

This downloads `train_3.parquet` and `train_4.parquet` into
`dataset/parquet/`, then runs the converter to create
`dataset/pixeldit_wids_text/`. The converter writes 10,000 samples per tar
shard, preserves the original image bytes, and stores `key.jpg` plus
`key.json` for every sample.

The JSON prompt is built from the original caption and valid OCR annotations,
for example:

```text
a red new product button with the word new product, with the visible words "PRODUCT" and "NEW" clearly rendered in the image.
```

The source parquet files are read-only inputs. The converter records and
rechecks their size, mtime, and SHA-256 in `conversion-manifest.json`; it does
not delete, move, rename, overwrite, or update them. See `dataset/README.md`
for the exact WIDS schema and verification command.

## 3. Configure and start training

Edit only the parameter block at the top of
`t2i/run_anyword_rgb_finetune.sh`:

```bash
GPU_IDS="0,1"
NUM_GPUS=2
RESUME=0
TRAIN_EPOCHS=10
TRAIN_BATCH_SIZE=3
NUM_WORKERS=10
LEARNING_RATE=2e-5
SAVE_MODEL_STEPS=1000
EVAL_SAMPLING_STEPS=500
```

Then run from the project root:

```bash
bash t2i/run_anyword_rgb_finetune.sh
```

The script sets `CUDA_VISIBLE_DEVICES` and the local Gemma path, then invokes
the distributed launcher from the caller's `python`. It does not validate GPU
IDs, activate a Python environment, or download anything during training.

The default fine-tuning configuration is 512px RGB PixelDiT with:

- `PixDiTTrainer`, 1.3B-parameter architecture, bf16
- local Gemma-2-2B-it text encoder, maximum length 300
- Flow matching with `flow_shift=4.0`
- per-GPU batch size 3, 10 epochs, 10 data workers
- learning rate `2e-5`, constant schedule with 2,000 warmup steps
- no REPA (`repa_loss_weight=0`), no VAE
- validation every 500 steps
- recovery checkpoint every 1,000 steps, retaining one recovery checkpoint

Checkpoints and logs are written below `t2i/output/<run_name>/`. The launcher
uses `--load_from` with the PixelDiT base weights when `RESUME=0`. To resume
the latest checkpoint and optimizer/scheduler state instead:

```bash
RESUME=1 bash t2i/run_anyword_rgb_finetune.sh
```

The output directory must already contain
`checkpoints/latest.pth` for this mode.

## 4. Run T2I inference

Place one prompt per line in `t2i/prompts.txt`, then pass the output directory
name (not an arbitrary path):

```bash
python t2i/infer_latest.py anyword_rgb_512 --gpu 0
```

The entrypoint requires `t2i/output/anyword_rgb_512/checkpoints/latest.pth`
and fails instead of silently selecting another checkpoint. It reads the run's
`config.yaml`, uses `ckpts/gemma-2-2b-it`, and writes images under a new
`t2i/output/<run_name>/inference_<timestamp>/vis/` directory.

Useful options include `--steps 50`, `--cfg_scale 3.5`, `--seed 0`, and
`--batch_size 1`.

## What is loaded during training?

1. `ckpts/PixelDiT-1300M-1024px/pixeldit_t2i_v1.pth` initializes the trainable
   PixelDiT RGB model.
2. `ckpts/gemma-2-2b-it/` provides frozen/no-gradient prompt embeddings.
3. The WIDS tar shards provide RGB bytes and JSON prompt metadata.

There is no VAE checkpoint in this pipeline: `vae_type` is `none` and the
model input is three-channel RGB. The `vae` section visible in the fully
serialized YAML is a generic PixelDiT configuration section, not an active
model.

## Troubleshooting

- `No parquet files found`: run `bash dataset/download_and_prepare.sh`, or
  edit its repository and remote-prefix parameters.
- `PIXDIT_GEMMA_PATH is required`: verify that
  `ckpts/gemma-2-2b-it/config.json` exists and that the caller exported no
  incorrect `PIXDIT_GEMMA_PATH`.
- `latest.pth does not exist`: use `RESUME=0` for a fresh run, or point the
  launcher at an output directory with a recovery checkpoint.
- `Weights only load failed`: the inference loader explicitly uses
  `weights_only=False` for these trusted local recovery checkpoints. Do not
  use that mode with untrusted pickle files.
