#!/usr/bin/env python3
"""Append paired T2I and white-background text prompts sampled from WIDS data."""

from __future__ import annotations

import argparse
import json
import random
import re
import tarfile
import tempfile
from pathlib import Path
from typing import Any


TEMPLATE = (
    "Generate the text '{text}' centered on a pure white background. Use clean, sharp, highly legible "
    "typography with accurate letterforms, balanced spacing, crisp edges, strong contrast, and a minimal "
    "composition. Include no additional text, symbols, objects, decorations, borders, shadows, logos, "
    "or watermarks."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--prompts_file", type=Path, required=True)
    parser.add_argument("--exclude_indices", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def normalize_line(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def load_excluded_indices(path: Path) -> set[int]:
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    indices = manifest.get("exclude_indices")
    if not isinstance(indices, list) or not all(isinstance(index, int) for index in indices):
        raise ValueError(f"Invalid exclude_indices list: {path}")
    return set(indices)


def load_shardlist(dataset_root: Path) -> list[dict[str, Any]]:
    with (dataset_root / "wids-meta.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    shardlist = metadata.get("shardlist")
    if not isinstance(shardlist, list) or not shardlist:
        raise ValueError(f"wids-meta.json has no shardlist: {dataset_root}")
    return shardlist


def locate_sample(shardlist: list[dict[str, Any]], source_index: int) -> tuple[str, str, int]:
    offset = 0
    active_prefix = ""
    source_group_start = 0
    for shard in shardlist:
        sample_count = shard.get("nsamples")
        url = shard.get("url")
        if not isinstance(sample_count, int) or not isinstance(url, str):
            raise ValueError("Invalid shard entry in wids-meta.json")
        source_prefix = re.sub(r"-\d+$", "", Path(url).stem)
        if source_prefix != active_prefix:
            active_prefix = source_prefix
            source_group_start = offset
        if source_index < offset + sample_count:
            return url, source_prefix, source_index - source_group_start
        offset += sample_count
    raise IndexError(f"Source index outside dataset: {source_index}")


def load_sample(dataset_root: Path, shardlist: list[dict[str, Any]], source_index: int) -> tuple[str, dict[str, Any]]:
    shard_url, source_prefix, source_row = locate_sample(shardlist, source_index)
    shard_path = dataset_root / shard_url
    if not shard_path.is_file():
        raise FileNotFoundError(f"Shard does not exist: {shard_path}")

    member_name = f"{source_prefix}-{source_row:09d}.json"
    with tarfile.open(shard_path, "r") as archive:
        member = archive.extractfile(member_name)
        if member is None:
            raise FileNotFoundError(f"Sample not found in shard: {member_name}")
        return member_name.removesuffix(".json"), json.load(member)


def choose_samples(
    dataset_root: Path,
    shardlist: list[dict[str, Any]],
    excluded: set[int],
    count: int,
    seed: int,
) -> list[tuple[int, str, str, str]]:
    total_samples = sum(int(shard["nsamples"]) for shard in shardlist)
    if count <= 0:
        raise ValueError("--count must be positive")

    rng = random.Random(seed)
    selected: list[tuple[int, str, str, str]] = []
    attempted: set[int] = set()
    while len(selected) < count:
        if len(attempted) + len(excluded) >= total_samples:
            raise RuntimeError("Not enough eligible samples with both prompt and visual text")
        source_index = rng.randrange(total_samples)
        if source_index in excluded or source_index in attempted:
            continue
        attempted.add(source_index)
        key, sample = load_sample(dataset_root, shardlist, source_index)
        prompt = normalize_line(str(sample.get("prompt", "")))
        texts = sample.get("texts", [])
        if not isinstance(texts, list):
            continue
        visible_text = normalize_line(" ".join(normalize_line(str(text)) for text in texts if str(text).strip()))
        if not prompt or not visible_text:
            continue
        selected.append((source_index, key, prompt, visible_text))
    return selected


def append_lines(path: Path, lines: list[str]) -> None:
    original = path.read_text(encoding="utf-8") if path.exists() else ""
    prefix = "" if not original or original.endswith("\n") else "\n"
    updated = original + prefix + "\n".join(lines) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        stream.write(updated)
        temporary_path = Path(stream.name)
    temporary_path.replace(path)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    prompts_file = args.prompts_file.resolve()
    excluded_path = args.exclude_indices.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if not excluded_path.is_file():
        raise FileNotFoundError(f"Held-out manifest does not exist: {excluded_path}")
    if prompts_file.parent and not prompts_file.parent.is_dir():
        raise FileNotFoundError(f"Prompt file parent does not exist: {prompts_file.parent}")

    samples = choose_samples(
        dataset_root,
        load_shardlist(dataset_root),
        load_excluded_indices(excluded_path),
        args.count,
        args.seed,
    )
    lines: list[str] = []
    for _, _, prompt, visible_text in samples:
        lines.extend((prompt, TEMPLATE.format(text=visible_text)))

    for source_index, key, _, visible_text in samples:
        print(f"source_index={source_index} key={key} text={visible_text!r}")
    if args.dry_run:
        print(f"Dry run: would append {len(lines)} prompts to {prompts_file}")
        return

    append_lines(prompts_file, lines)
    print(f"Appended {len(lines)} prompts to {prompts_file}")


if __name__ == "__main__":
    main()
