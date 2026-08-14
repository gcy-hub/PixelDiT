## Requirements

```bash
python -m pip install -r requirements.txt
```

## 1. Download model checkpoints

Run:

```bash
bash ckpts/download.sh
```

## 2. Download and prepare AnyWord-3M and ImageNet-1K

Run:

```bash
bash dataset/download_and_prepare.sh all
```

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

The output directory must already contain
`checkpoints/latest.pth` for this mode.

## 4. Run T2I inference

Place one prompt per line in `t2i/prompts.txt`, then pass the output directory
name (not an arbitrary path):

```bash
python t2i/infer_latest.py anyword_rgb_512 --gpu 0
```
