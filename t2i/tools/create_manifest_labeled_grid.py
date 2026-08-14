#!/usr/bin/env python3
"""Build a labelled validation grid from a holdout manifest."""

import argparse
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def transcript(sample: dict[str, Any]) -> str:
    """Return the annotated visible strings in annotation reading order."""
    lines: list[str] = []
    for entry in sample.get("ocr_result", []):
        try:
            value = str(entry[1][0]).strip()
        except (IndexError, TypeError):
            continue
        if value:
            lines.append(value)
    if not lines:
        lines = [str(value).strip() for value in sample.get("texts", []) if str(value).strip()]
    if not lines:
        raise ValueError(f"Sample {sample.get('key', '<unknown>')} has no OCR transcript.")
    return "\n".join(lines)


def load_samples(manifest_path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"Manifest has no samples: {manifest_path}")
    if any(not isinstance(sample, dict) for sample in samples):
        raise ValueError(f"Manifest contains a non-object sample: {manifest_path}")
    return samples


def write_indexed_prompts(samples: list[dict[str, Any]], output_path: Path) -> None:
    prompts: dict[str, dict[str, str]] = {}
    for index, sample in enumerate(samples):
        prompt = str(sample.get("prompt", "")).strip()
        if not prompt:
            raise ValueError(f"Sample {index} has an empty prompt.")
        prompts[f"{index:03d}"] = {"prompt": prompt}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(prompts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_font(size: int, font_path: str | None) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if font_path:
        return ImageFont.truetype(font_path, size=size)
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def fit_font(label: str, tile_size: int, label_height: int, font_path: str | None) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    lines = label.splitlines()
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    for size in range(min(34, label_height // max(1, len(lines))), 11, -1):
        font = load_font(size, font_path)
        boxes = [probe.textbbox((0, 0), line, font=font) for line in lines]
        widest = max((box[2] - box[0] for box in boxes), default=0)
        line_height = max((box[3] - box[1] for box in boxes), default=0)
        if widest <= tile_size - 28 and len(lines) * line_height + (len(lines) - 1) * 4 <= label_height - 20:
            return font
    return load_font(12, font_path)


def load_top_grid_tiles(source_grid: Path, count: int, columns: int) -> list[Image.Image]:
    with Image.open(source_grid) as image:
        image = image.convert("RGB")
        if image.width % columns:
            raise ValueError(f"Grid width {image.width} is not divisible by {columns}.")
        tile_size = image.width // columns
        if image.height % tile_size:
            raise ValueError(f"Grid height {image.height} is not a whole number of tile rows.")
        if image.height // tile_size < (count + columns - 1) // columns:
            raise ValueError(f"Grid has too few tiles for {count} samples.")
        return [
            image.crop(
                (
                    (index % columns) * tile_size,
                    (index // columns) * tile_size,
                    (index % columns + 1) * tile_size,
                    (index // columns + 1) * tile_size,
                )
            ).copy()
            for index in range(count)
        ]


def load_indexed_tiles(image_dir: Path, count: int) -> list[Image.Image]:
    tiles: list[Image.Image] = []
    missing: list[str] = []
    for index in range(count):
        path = image_dir / f"{index:03d}.jpg"
        if not path.is_file():
            missing.append(path.name)
            continue
        with Image.open(path) as image:
            tiles.append(image.convert("RGB").copy())
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} inference images in {image_dir}: {', '.join(missing)}")
    return tiles


def save_grid(
    tiles: list[Image.Image], labels: list[str], output_path: Path, columns: int, tile_size: int,
    label_height: int, font_path: str | None,
) -> None:
    if len(tiles) != len(labels):
        raise ValueError("The tile count and label count differ.")
    rows = (len(tiles) + columns - 1) // columns
    cell_height = tile_size + label_height
    canvas = Image.new("RGB", (columns * tile_size, rows * cell_height), "white")
    draw = ImageDraw.Draw(canvas)

    for index, (tile, label) in enumerate(zip(tiles, labels)):
        x = (index % columns) * tile_size
        y = (index // columns) * cell_height
        canvas.paste(tile.resize((tile_size, tile_size), Image.Resampling.LANCZOS), (x, y))
        font = fit_font(label, tile_size, label_height, font_path)
        line_boxes = [draw.textbbox((0, 0), line, font=font) for line in label.splitlines()]
        line_heights = [box[3] - box[1] for box in line_boxes]
        content_height = sum(line_heights) + 4 * (len(line_boxes) - 1)
        line_y = y + tile_size + (label_height - content_height) / 2
        for line, box, height in zip(label.splitlines(), line_boxes, line_heights):
            width = box[2] - box[0]
            draw.text((x + (tile_size - width) / 2, line_y - box[1]), line, font=font, fill="black")
            line_y += height + 4

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    if output_path.suffix.lower() == ".webp":
        # Preserve source tiles exactly while adding only the label bands.
        canvas.save(temporary, format="WEBP", lossless=True, method=6)
    else:
        canvas.save(temporary)
    os.replace(temporary, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source_grid", type=Path)
    parser.add_argument("--image_dir", type=Path)
    parser.add_argument("--write_indexed_prompts", type=Path)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--tile_size", type=int, default=1024)
    parser.add_argument("--label_height", type=int, default=192)
    parser.add_argument("--font_path")
    args = parser.parse_args()
    if bool(args.source_grid) == bool(args.image_dir):
        parser.error("Specify exactly one of --source_grid and --image_dir.")

    samples = load_samples(args.manifest)
    if args.write_indexed_prompts:
        write_indexed_prompts(samples, args.write_indexed_prompts)
    labels = [transcript(sample) for sample in samples]
    tiles = (
        load_top_grid_tiles(args.source_grid, len(samples), args.columns)
        if args.source_grid
        else load_indexed_tiles(args.image_dir, len(samples))
    )
    save_grid(tiles, labels, args.output, args.columns, args.tile_size, args.label_height, args.font_path)
    print(f"Saved {len(tiles)} labelled tiles to {args.output}")


if __name__ == "__main__":
    main()
