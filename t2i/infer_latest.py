#!/usr/bin/env python3
"""Generate T2I images from the latest checkpoint of a PixelDiT output run.

The output run is intentionally addressed by name rather than by an arbitrary
path.  This keeps both the checkpoint lookup and generated images inside the
repository's ``t2i/output`` directory.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


T2I_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = T2I_ROOT.parent
OUTPUT_ROOT = T2I_ROOT / "output"
DEFAULT_PROMPTS = T2I_ROOT / "prompts.txt"
DEFAULT_GEMMA_PATH = PROJECT_ROOT / "ckpts" / "gemma-2-2b-it"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PixelDiT T2I inference from output/<run_name>/checkpoints/latest.pth."
    )
    parser.add_argument(
        "run_name",
        help="Directory name directly under t2i/output (for example, anyword_rgb_512_holdout_lr2e-5).",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=DEFAULT_PROMPTS,
        help=f"One prompt per line (default: {DEFAULT_PROMPTS}).",
    )
    parser.add_argument("--gpu", type=int, default=0, help="Physical CUDA device index to use.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50, help="Flow DPM-Solver sampling steps.")
    parser.add_argument("--cfg_scale", type=float, default=3.5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--output_name",
        default=None,
        help="Optional name for the new inference directory. The default includes a timestamp.",
    )
    return parser.parse_args()


def validate_run_name(run_name: str) -> str:
    candidate = Path(run_name)
    if not run_name or candidate.name != run_name or candidate.is_absolute() or run_name in {".", ".."}:
        raise ValueError("run_name must be one directory name directly under t2i/output.")
    return run_name


def read_prompt_count(prompts_path: Path) -> int:
    if not prompts_path.is_file():
        raise FileNotFoundError(f"Prompt file does not exist: {prompts_path}")
    with prompts_path.open("r", encoding="utf-8") as handle:
        prompt_count = sum(1 for _ in handle)
    if prompt_count == 0:
        raise ValueError(f"Prompt file is empty: {prompts_path}")
    return prompt_count


def main() -> None:
    args = parse_args()
    run_name = validate_run_name(args.run_name)
    if args.gpu < 0:
        raise ValueError("--gpu must be non-negative.")
    if args.steps <= 0 or args.batch_size <= 0:
        raise ValueError("--steps and --batch_size must be positive.")

    run_dir = OUTPUT_ROOT / run_name
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Output run directory does not exist: {run_dir}")

    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Run config does not exist: {config_path}")

    checkpoint_path = run_dir / "checkpoints" / "latest.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Latest checkpoint does not exist: {checkpoint_path}. "
            "This script intentionally will not choose another checkpoint."
        )

    prompts_path = args.prompts.expanduser().resolve()
    prompt_count = read_prompt_count(prompts_path)
    inference_name = args.output_name or f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if Path(inference_name).name != inference_name or inference_name in {".", ".."}:
        raise ValueError("--output_name must be a single directory name.")

    inference_dir = run_dir / inference_name
    if inference_dir.exists():
        raise FileExistsError(f"Inference output already exists: {inference_dir}")
    inference_dir.mkdir(parents=False)

    gemma_path = Path(os.environ.get("PIXDIT_GEMMA_PATH", DEFAULT_GEMMA_PATH))
    if not gemma_path.is_dir():
        raise FileNotFoundError(f"Local Gemma directory does not exist: {gemma_path}")

    command = [
        sys.executable,
        str(T2I_ROOT / "inference.py"),
        "--config",
        str(config_path),
        "--model_path",
        str(checkpoint_path),
        "--work_dir",
        str(inference_dir),
        "--txt_file",
        str(prompts_path),
        "--sample_nums",
        str(prompt_count),
        "--end_index",
        str(prompt_count),
        "--bs",
        str(args.batch_size),
        "--cfg_scale",
        str(args.cfg_scale),
        "--step",
        str(args.steps),
        "--seed",
        str(args.seed),
        "--dataset",
        "prompt_file",
        "--gpu_id",
        "0",
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    environment["PIXDIT_GEMMA_PATH"] = str(gemma_path)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Prompts: {prompts_path} ({prompt_count} lines)")
    print(f"Images will be saved under: {inference_dir / 'vis'}")
    subprocess.run(command, cwd=T2I_ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
