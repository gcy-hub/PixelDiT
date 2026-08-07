# AnyWord T2I workspace

The project-level workflow is documented in [../README.md](../README.md).
This directory contains the PixelDiT T2I implementation, the 512px AnyWord
configuration, the training launcher, and the inference entrypoint.

## Training

Prepare checkpoints and data from the project root, then edit the parameter
block at the top of `run_anyword_rgb_finetune.sh`:

```bash
bash ../ckpts/download.sh
bash ../dataset/download_and_prepare.sh
bash run_anyword_rgb_finetune.sh
```

Set `RESUME=1` in the launcher only when
`output/<run_name>/checkpoints/latest.pth` exists and optimizer/scheduler state
should be restored.

## Inference

The latest-checkpoint wrapper reads one prompt per line and refuses to fall
back to another checkpoint:

```bash
python infer_latest.py anyword_rgb_512 --gpu 0
```

Images are saved under a timestamped directory inside the selected output run.
