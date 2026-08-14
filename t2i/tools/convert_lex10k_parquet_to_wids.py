#!/usr/bin/env python3
"""Convert LeX-10K parquet shards to PixelDiT WIDS tar shards.

The source parquet files are read-only.  Each WIDS sample contains the
original PNG bytes as ``key.png`` and a JSON sidecar with the training prompt,
OCR text, dimensions, and source provenance.  Thirty deterministic samples
are written to a holdout manifest and are excluded by PixelDatasetMS through
their global WIDS indices.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import tarfile
import tempfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from PIL import Image


DEFAULT_INPUT = Path("/home/wangye/projects/LeX-10K/data")
DEFAULT_OUTPUT = Path("/home/wangye/projects/PixelDiT-master/t2i/data/lex10k_wids")
SHARD_SIZE = 1000
HOLDOUT_SIZE = 30
HOLDOUT_SEED = 20260813


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"name": path.name, "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": sha256(path)}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, ensure_ascii=True, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def image_bytes(value: Any, source: str) -> bytes:
    if not isinstance(value, dict) or "bytes" not in value:
        raise ValueError(f"{source}: image is not a struct with bytes")
    payload = value["bytes"]
    if isinstance(payload, memoryview):
        payload = payload.tobytes()
    if not isinstance(payload, (bytes, bytearray)):
        raise ValueError(f"{source}: image bytes have type {type(payload).__name__}")
    return bytes(payload)


def parse_ocr(value: Any) -> tuple[list[str], Any]:
    if value is None or value == "":
        return [], []
    parsed = json.loads(value) if isinstance(value, str) else value
    texts: list[str] = []
    if isinstance(parsed, list):
        for item in parsed:
            try:
                text = item[1][0]
            except (IndexError, KeyError, TypeError):
                continue
            if isinstance(text, str) and text.strip():
                texts.append(" ".join(text.split()))
    return texts, parsed


def make_prompt(row: dict[str, Any], texts: list[str]) -> tuple[str, str]:
    caption = str(row.get("post_aligned_caption") or row.get("caption") or "").strip()
    caption = " ".join(caption.replace("*", " ").split())
    if not texts:
        return caption, caption
    quoted = ", ".join(f'"{text}"' for text in texts)
    # Gemma receives at most 300 tokens.  Put the string(s) that must be
    # rendered first so a long visual description cannot truncate them.
    prompt = f"Render the exact visible text {quoted} clearly in the image. {caption.rstrip(' .')}."
    return prompt, caption


def add_member(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mtime = 0
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(payload))


def write_shard(output: Path, name: str, rows: list[tuple[str, bytes, dict[str, Any]]]) -> dict[str, Any]:
    shard_dir = output / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    final = shard_dir / name
    temporary = final.with_suffix(".tar.tmp")
    with tarfile.open(temporary, "w") as archive:
        for key, payload, metadata in rows:
            add_member(archive, f"{key}.png", payload)
            add_member(archive, f"{key}.json", json.dumps(metadata, ensure_ascii=True, sort_keys=True).encode())
    os.replace(temporary, final)
    return {"url": str(final.relative_to(output)), "nsamples": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--holdout_size", type=int, default=HOLDOUT_SIZE)
    parser.add_argument("--seed", type=int, default=HOLDOUT_SEED)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    output = args.output_dir.resolve()
    if output == input_dir or output.suffix.lower() == ".parquet":
        raise ValueError("Output must be separate from the parquet input directory")
    if args.holdout_size <= 0:
        raise ValueError("holdout_size must be positive")
    files = sorted(input_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {input_dir}")
    output.mkdir(parents=True, exist_ok=True)
    source_fingerprints = {path.name: fingerprint(path) for path in files}
    records: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    global_index = 0
    shard_rows: list[tuple[str, bytes, dict[str, Any]]] = []
    shard_number = 0
    for path in files:
        parquet = pq.ParquetFile(path)
        required = {"image_name", "image", "caption", "post_aligned_caption", "ocr_result"}
        missing = required - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        row_index = 0
        for batch in parquet.iter_batches(batch_size=args.batch_size, columns=list(required)):
            for row in batch.to_pylist():
                source = f"{path.name} row {row_index}"
                payload = image_bytes(row["image"], source)
                with Image.open(io.BytesIO(payload)) as image:
                    image.load()
                    width, height, fmt = image.width, image.height, image.format
                texts, ocr = parse_ocr(row.get("ocr_result"))
                prompt, clean_caption = make_prompt(row, texts)
                key = f"lex10k-{global_index:06d}"
                metadata = {
                    "prompt": prompt,
                    "original_caption": clean_caption,
                    "post_aligned_caption": row.get("post_aligned_caption") or "",
                    "texts": texts,
                    "ocr_result": ocr,
                    "height": height,
                    "width": width,
                    "source_img_name": row.get("image_name"),
                    "source_image_format": fmt,
                    "source_parquet": path.name,
                    "source_row": row_index,
                    "source_index": global_index,
                }
                records.append(metadata | {"key": key})
                shard_rows.append((key, payload, metadata))
                global_index += 1
                row_index += 1
                if len(shard_rows) == SHARD_SIZE:
                    shards.append(write_shard(output, f"lex10k-{shard_number:05d}.tar", shard_rows))
                    shard_rows = []
                    shard_number += 1
        if row_index != parquet.metadata.num_rows:
            raise RuntimeError(f"{path}: converted {row_index}, expected {parquet.metadata.num_rows}")
    if shard_rows:
        shards.append(write_shard(output, f"lex10k-{shard_number:05d}.tar", shard_rows))
    if global_index < args.holdout_size:
        raise ValueError("holdout_size exceeds dataset size")
    holdout_indices = sorted(random.Random(args.seed).sample(range(global_index), args.holdout_size))
    holdout = {
        "dataset": "LeX-10K",
        "seed": args.seed,
        "total_samples": global_index,
        "holdout_size": args.holdout_size,
        "exclude_indices": holdout_indices,
        "samples": [records[index] for index in holdout_indices],
        "source_fingerprints": source_fingerprints,
        "created_at": now(),
    }
    atomic_json(output / "lex10k_holdout_30.json", holdout)
    atomic_json(output / "wids-meta.json", {"wids_version": 1, "name": "lex10k", "base_path": str(output), "shardlist": shards})
    atomic_json(output / "conversion-manifest.json", {"dataset": "LeX-10K", "total_samples": global_index, "shard_size": SHARD_SIZE, "inputs": source_fingerprints, "shards": shards, "created_at": now()})
    current = {path.name: fingerprint(path) for path in files}
    if current != source_fingerprints:
        raise RuntimeError("A source parquet changed during conversion")
    print(f"Converted {global_index} samples into {len(shards)} shards")
    print(f"Holdout: {output / 'lex10k_holdout_30.json'}")


if __name__ == "__main__":
    main()
