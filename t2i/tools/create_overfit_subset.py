#!/usr/bin/env python3
"""Create a tiny, disjoint AnyWord WIDS subset for an overfit smoke test."""

from __future__ import annotations

import argparse
import json
import random
import tarfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def collect_candidates(source: Path) -> dict[str, list[dict]]:
    candidates = {"3-4": [], "5-8": []}
    for shard_path in sorted((source / "shards").glob("*.tar")):
        with tarfile.open(shard_path, "r") as archive:
            members = {member.name: member for member in archive if member.isfile()}
            json_members = sorted(name for name in members if name.endswith(".json"))
            for json_name in json_members:
                key = json_name[:-5]
                image_name = f"{key}.jpg"
                if image_name not in members:
                    continue
                item = json.load(archive.extractfile(members[json_name]))
                annotations = [
                    annotation
                    for annotation in item.get("annotations", [])
                    if annotation.get("valid", True) and isinstance(annotation.get("text"), str)
                ]
                if len(annotations) != 1:
                    continue
                text = annotations[0]["text"]
                length = len(text)
                bucket = "3-4" if 3 <= length <= 4 else "5-8" if 5 <= length <= 8 else None
                if bucket is None:
                    continue
                candidates[bucket].append(
                    {
                        "key": key,
                        "shard": str(shard_path),
                        "json_name": json_name,
                        "image_name": image_name,
                        "text": text,
                        "length": length,
                        "prompt": item["prompt"],
                        "source_parquet": item.get("source_parquet"),
                        "source_row": item.get("source_row"),
                    }
                )
    return candidates


def read_selected(selected: list[dict]) -> None:
    archives: dict[str, tarfile.TarFile] = {}
    try:
        for entry in selected:
            archive = archives.setdefault(entry["shard"], tarfile.open(entry["shard"], "r"))
            entry["image_bytes"] = archive.extractfile(entry["image_name"]).read()
            entry["json_bytes"] = archive.extractfile(entry["json_name"]).read()
    finally:
        for archive in archives.values():
            archive.close()


def write_split(output: Path, split: str, entries: list[dict]) -> None:
    split_dir = output / split
    shard_dir = split_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_name = f"overfit64_{split}-00000.tar"
    shard_path = shard_dir / shard_name
    with tarfile.open(shard_path, "w") as archive:
        for entry in entries:
            for suffix, payload in ((".jpg", entry["image_bytes"]), (".json", entry["json_bytes"])):
                info = tarfile.TarInfo(f"{entry['key']}{suffix}")
                info.size = len(payload)
                archive.addfile(info, __import__("io").BytesIO(payload))
    meta = {
        "base_path": str(split_dir),
        "name": f"anyword-overfit64-{split}",
        "shardlist": [{"nsamples": len(entries), "url": f"shards/{shard_name}"}],
        "wids_version": 1,
    }
    (split_dir / "wids-meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.output.resolve() == args.source.resolve():
        raise ValueError("output must be different from source")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    candidates = collect_candidates(args.source)
    selected: dict[str, list[dict]] = {}
    for bucket in ("3-4", "5-8"):
        rng.shuffle(candidates[bucket])
        if len(candidates[bucket]) < 32:
            raise RuntimeError(f"not enough {bucket} candidates: {len(candidates[bucket])}")
        selected[bucket] = candidates[bucket][:32]

    train = selected["3-4"][:16] + selected["5-8"][:16]
    validation = selected["3-4"][16:] + selected["5-8"][16:]
    read_selected(train + validation)
    write_split(args.output, "train", train)
    write_split(args.output, "validation", validation)

    manifest = {
        "seed": args.seed,
        "source": str(args.source),
        "counts": {"total": 64, "train": 32, "validation": 32},
        "length_buckets": {"3-4": 32, "5-8": 32},
        "train": [{k: v for k, v in entry.items() if k not in {"image_bytes", "json_bytes"}} for entry in train],
        "validation": [{k: v for k, v in entry.items() if k not in {"image_bytes", "json_bytes"}} for entry in validation],
    }
    (args.output / "selection-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "train": len(train), "validation": len(validation)}, indent=2))


if __name__ == "__main__":
    main()
