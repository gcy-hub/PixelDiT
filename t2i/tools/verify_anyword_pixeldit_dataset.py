#!/usr/bin/env python3
"""Load an AnyWord WIDS output through PixelDiT's RGB multiscale dataset."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    args = parser.parse_args()

    t2i_dir = Path(__file__).resolve().parents[1]
    os.environ.setdefault("WIDS_CACHE", str(t2i_dir / "output" / "wids_smoke_cache"))
    sys.path.insert(0, str(t2i_dir))
    from diffusion.data.builder import build_dataset

    dataset = build_dataset(
        {
            "type": "PixelDatasetMS",
            "data_dir": [str(args.data_dir)],
            "caption_proportion": {"prompt": 1},
            "external_caption_suffixes": [],
            "external_clipscore_suffixes": [],
            "clip_thr": 0.0,
            "clip_thr_temperature": 1.0,
            "load_text_feat": False,
            "sort_dataset": False,
            "num_replicas": 1,
            "aspect_ratio_type": "ASPECT_RATIO_512",
        },
        resolution=512,
    )
    image, prompt, _, data_info, _, caption_type, _, _ = dataset[0]
    if image.shape[0] != 3 or not isinstance(prompt, str) or caption_type != "prompt":
        raise RuntimeError("PixelDatasetMS did not yield the expected RGB image and prompt.")
    print(
        f"PixelDatasetMS OK: samples={len(dataset)}, image_shape={tuple(image.shape)}, "
        f"bucket={data_info['aspect_ratio']}, prompt={prompt!r}"
    )


if __name__ == "__main__":
    main()
