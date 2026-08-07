#!/usr/bin/env python3
"""Convert the AnyWord LAION parquet files to PixelDiT's local WIDS format.

The converter intentionally has a fixed input location.  It never alters an
input parquet: all manifests, tar shards, and metadata are written below the
separate output directory. The PixelDiT loader expects a ``.jpg`` member name;
the raw image bytes are never transcoded and their actual format is recorded.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
import os
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "dataset" / "parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dataset" / "pixeldit_wids_text"
SHARD_SIZE = 10_000
MANIFEST_NAME = "conversion-manifest.json"
WIDS_META_NAME = "wids-meta.json"
REQUIRED_COLUMNS = {
    "caption",
    "image",
    "annotations",
    "image_width",
    "image_height",
    "img_name",
}


class ConversionError(RuntimeError):
    """Raised when an input or existing conversion cannot be trusted."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing the input parquet files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Separate directory for WIDS shards and conversion metadata.",
    )
    parser.add_argument(
        "--verify_only",
        action="store_true",
        help="Verify an existing output directory without converting parquet rows.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Parquet rows read per batch; does not affect output shard size.",
    )
    parser.add_argument(
        "--verify_workers",
        type=int,
        default=10,
        help="Processes used to verify independent tar shards (default: 10).",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        json.dump(value, stream, ensure_ascii=True, indent=2, sort_keys=True)
        stream.write("\n")
        temp_path = Path(stream.name)
    os.replace(temp_path, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(path: Path) -> dict[str, Any]:
    stat_before = path.stat()
    digest = sha256_file(path)
    stat_after = path.stat()
    if (stat_before.st_size, stat_before.st_mtime_ns) != (stat_after.st_size, stat_after.st_mtime_ns):
        raise ConversionError(f"Input changed while computing SHA-256: {path}")
    return {
        "name": path.name,
        "size_bytes": stat_after.st_size,
        "mtime_ns": stat_after.st_mtime_ns,
        "sha256": digest,
    }


def fingerprint_inputs(paths: list[Path], workers: int) -> dict[str, dict[str, Any]]:
    if workers == 1 or len(paths) == 1:
        fingerprints = [fingerprint(path) for path in paths]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(paths))) as executor:
            fingerprints = list(executor.map(fingerprint, paths))
    return {item["name"]: item for item in fingerprints}


def input_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("*.parquet"))
    if not files:
        raise ConversionError(f"No parquet files found in input directory: {input_dir}")
    return files


def validate_paths(output_dir: Path, parquet_files: Iterable[Path], input_dir: Path) -> Path:
    input_dir = input_dir.resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir == input_dir:
        raise ConversionError("Output directory must not be the input parquet directory.")
    if output_dir.suffix.lower() == ".parquet":
        raise ConversionError("Output directory must not be a parquet file path.")
    if output_dir in {path.resolve() for path in parquet_files}:
        raise ConversionError("Output directory must not point to an input parquet file.")
    return output_dir


def load_manifest(path: Path, input_dir: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "format_version": 1,
            "created_at": utc_now(),
            "input_dir": str(input_dir),
            "output_format": "PixelDiT WIDS: key.jpg + key.json",
            "shard_size": SHARD_SIZE,
            "inputs": {},
        }
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ConversionError(f"Invalid conversion manifest: {path}") from error
    if manifest.get("format_version") != 1 or manifest.get("input_dir") != str(input_dir):
        raise ConversionError(f"Manifest is not compatible with this converter: {path}")
    return manifest


def completed_input(record: dict[str, Any] | None, source_fingerprint: dict[str, Any], output_dir: Path) -> bool:
    if not record or record.get("status") != "complete":
        return False
    if record.get("source_before") != source_fingerprint or record.get("source_after") != source_fingerprint:
        raise ConversionError(
            f"Completed input {source_fingerprint['name']} no longer matches its recorded fingerprint. "
            "Use a new output directory; do not overwrite the existing conversion."
        )
    shards = record.get("shards", [])
    if not shards:
        raise ConversionError(f"Completed input {source_fingerprint['name']} has no recorded shards.")
    for shard in shards:
        shard_path = output_dir / shard["url"]
        if not shard_path.is_file():
            raise ConversionError(f"Recorded completed shard is missing: {shard_path}")
    return True


def read_schema(path: Path) -> tuple[pq.ParquetFile, int]:
    parquet = pq.ParquetFile(path)
    columns = set(parquet.schema_arrow.names)
    missing = REQUIRED_COLUMNS - columns
    if missing:
        raise ConversionError(f"{path} is missing required columns: {sorted(missing)}")
    return parquet, parquet.metadata.num_rows


def image_dimensions(image_bytes: bytes, source: str) -> tuple[int, int, str]:
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load()
            if not image.format:
                raise ConversionError(f"Cannot determine image format for {source}.")
            return image.width, image.height, image.format
    except Exception as error:
        raise ConversionError(f"Cannot decode JPEG for {source}: {error}") from error


def as_bytes(value: Any, source: str) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, bytearray):
        return bytes(value)
    raise ConversionError(f"{source} has non-bytes image payload: {type(value).__name__}")


def image_payload(row: dict[str, Any], source: str) -> tuple[bytes, Any]:
    image = row["image"]
    if not isinstance(image, dict) or "bytes" not in image:
        raise ConversionError(f"{source} has an invalid image struct.")
    return as_bytes(image["bytes"], source), image.get("path")


def normalize_annotations(value: Any, source: str) -> list[dict[str, Any]]:
    """Keep OCR annotations JSON-safe while preserving their original fields."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise ConversionError(f"{source} has a non-list annotations value.")
    annotations: list[dict[str, Any]] = []
    for index, annotation in enumerate(value):
        if not isinstance(annotation, dict):
            raise ConversionError(f"{source} annotation {index} is not a struct.")
        item = dict(annotation)
        polygon = item.get("polygon")
        if polygon is None:
            item["polygon"] = []
        elif not isinstance(polygon, list):
            raise ConversionError(f"{source} annotation {index} has a non-list polygon.")
        text = item.get("text")
        if text is not None and not isinstance(text, str):
            item["text"] = str(text)
        annotations.append(item)
    return annotations


def build_training_prompt(caption: Any, annotations: list[dict[str, Any]]) -> tuple[str, str, list[str]]:
    """Make OCR text explicit for a caption-only T2I text encoder.

    AnyText supplies these strings through a separate ``texts`` condition and
    uses ``*`` placeholders in its caption. PixelDiT has only a prompt string,
    so the exact visible words must be materialized in that prompt.
    """
    original_caption = "" if caption is None else str(caption).strip()
    clean_caption = " ".join(original_caption.replace("*", " ").split())
    texts: list[str] = []
    for annotation in annotations:
        if annotation.get("valid", True) is False:
            continue
        text = annotation.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(" ".join(text.split()))
    if not texts:
        return clean_caption, clean_caption, texts
    quoted = [f'"{text}"' for text in texts]
    if len(quoted) == 1:
        visible_text = f"the visible word {quoted[0]}"
    elif len(quoted) == 2:
        visible_text = f"the visible words {quoted[0]} and {quoted[1]}"
    else:
        visible_text = f"the visible words {', '.join(quoted[:-1])}, and {quoted[-1]}"
    base = clean_caption.rstrip(" .")
    prompt = f"{base}, with {visible_text} clearly rendered in the image."
    return prompt, clean_caption, texts


def add_tar_member(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mtime = 0
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(payload))


def write_shard(output_dir: Path, shard_name: str, samples: list[tuple[str, bytes, dict[str, Any]]]) -> dict[str, Any]:
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    final_path = shard_dir / shard_name
    temporary_path = final_path.with_suffix(".tar.tmp")
    with tarfile.open(temporary_path, mode="w") as archive:
        for key, image_bytes, metadata in samples:
            add_tar_member(archive, f"{key}.jpg", image_bytes)
            add_tar_member(
                archive,
                f"{key}.json",
                json.dumps(metadata, ensure_ascii=True, sort_keys=True).encode("utf-8"),
            )
    os.replace(temporary_path, final_path)
    return {"url": str(final_path.relative_to(output_dir)), "nsamples": len(samples)}


def row_metadata(
    row: dict[str, Any],
    path: Path,
    row_index: int,
    image_width: int,
    image_height: int,
    source_format: str,
    source_path: Any,
    annotations: list[dict[str, Any]],
    wm_score: Any,
) -> dict[str, Any]:
    expected_width = row["image_width"]
    expected_height = row["image_height"]
    if expected_width is not None and int(expected_width) != image_width:
        raise ConversionError(f"{path.name} row {row_index}: parquet width disagrees with JPEG dimensions.")
    if expected_height is not None and int(expected_height) != image_height:
        raise ConversionError(f"{path.name} row {row_index}: parquet height disagrees with JPEG dimensions.")
    prompt, original_caption, texts = build_training_prompt(row["caption"], annotations)
    return {
        "prompt": prompt,
        "original_caption": original_caption,
        "texts": texts,
        "annotations": annotations,
        "wm_score": wm_score,
        "height": image_height,
        "width": image_width,
        "source_img_name": row["img_name"],
        "source_image_format": source_format,
        "source_parquet": path.name,
        "source_path": source_path,
        "source_row": row_index,
    }


def convert_input(path: Path, output_dir: Path, batch_size: int) -> tuple[int, list[dict[str, Any]]]:
    parquet, expected_rows = read_schema(path)
    columns = ["image", "caption", "annotations", "image_width", "image_height", "img_name"]
    if "wm_score" in parquet.schema_arrow.names:
        columns.append("wm_score")
    current_shard: list[tuple[str, bytes, dict[str, Any]]] = []
    shards: list[dict[str, Any]] = []
    row_index = 0
    shard_index = 0

    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        for row in batch.to_pylist():
            source = f"{path.name} row {row_index}"
            image_bytes, source_path = image_payload(row, source)
            width, height, source_format = image_dimensions(image_bytes, source)
            annotations = normalize_annotations(row.get("annotations"), source)
            key = f"{path.stem}-{row_index:09d}"
            current_shard.append(
                (
                    key,
                    image_bytes,
                    row_metadata(
                        row,
                        path,
                        row_index,
                        width,
                        height,
                        source_format,
                        source_path,
                        annotations,
                        row.get("wm_score"),
                    ),
                )
            )
            row_index += 1
            if len(current_shard) == SHARD_SIZE:
                shards.append(write_shard(output_dir, f"{path.stem}-{shard_index:05d}.tar", current_shard))
                current_shard = []
                shard_index += 1

    if current_shard:
        shards.append(write_shard(output_dir, f"{path.stem}-{shard_index:05d}.tar", current_shard))
    if row_index != expected_rows:
        raise ConversionError(f"{path}: converted {row_index} rows, parquet metadata reported {expected_rows}.")
    return row_index, shards


def all_shards(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    shards: list[dict[str, Any]] = []
    for record in manifest["inputs"].values():
        if record.get("status") == "complete":
            shards.extend(record["shards"])
    return shards


def write_wids_meta(output_dir: Path, manifest: dict[str, Any]) -> None:
    shardlist = all_shards(manifest)
    if not shardlist:
        return
    atomic_json_write(
        output_dir / WIDS_META_NAME,
        {
            "wids_version": 1,
            "name": "anyword-laion-rgb",
            "base_path": str(output_dir),
            "shardlist": shardlist,
        },
    )


def verify_shard(output_dir: Path, shard: dict[str, Any]) -> int:
    shard_path = output_dir / shard["url"]
    paired: dict[str, set[str]] = {}
    with tarfile.open(shard_path, mode="r") as archive:
        for member in archive:
            if not member.isfile():
                continue
            key, extension = member.name.rsplit(".", 1)
            if extension not in {"jpg", "json"}:
                raise ConversionError(f"Unexpected tar member {member.name} in {shard_path}")
            paired.setdefault(key, set()).add(extension)
            payload = archive.extractfile(member)
            if payload is None:
                raise ConversionError(f"Cannot read tar member {member.name} in {shard_path}")
            content = payload.read()
            if extension == "jpg":
                width, height, _ = image_dimensions(content, f"{shard_path}:{member.name}")
                paired[key].add(f"size:{width}x{height}")
            else:
                item = json.loads(content.decode("utf-8"))
                for field in (
                    "prompt",
                    "original_caption",
                    "texts",
                    "annotations",
                    "height",
                    "width",
                    "source_parquet",
                    "source_row",
                ):
                    if field not in item:
                        raise ConversionError(f"Missing {field!r} in {shard_path}:{member.name}")
                if any(text not in item["prompt"] for text in item["texts"]):
                    raise ConversionError(f"Prompt does not contain all annotated text in {shard_path}:{member.name}")
                paired[key].add(f"metadata:{item['width']}x{item['height']}")
    if len(paired) != shard["nsamples"]:
        raise ConversionError(f"{shard_path}: metadata says {shard['nsamples']} samples, found {len(paired)}.")
    for key, fields in paired.items():
        dimensions = {field.split(":", 1)[1] for field in fields if field.startswith(("size:", "metadata:"))}
        if not {"jpg", "json"}.issubset(fields) or len(dimensions) != 1:
            raise ConversionError(f"{shard_path}: invalid image/JSON pair for key {key}.")
    return len(paired)


def verify_output(
    output_dir: Path,
    expected_fingerprints: dict[str, dict[str, Any]],
    verify_workers: int,
    input_dir: Path,
) -> None:
    manifest_path = output_dir / MANIFEST_NAME
    meta_path = output_dir / WIDS_META_NAME
    manifest = load_manifest(manifest_path, input_dir)
    if not meta_path.is_file():
        raise ConversionError(f"WIDS metadata is missing: {meta_path}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    shards = metadata.get("shardlist", [])
    manifest_shards = all_shards(manifest)
    if shards != manifest_shards:
        raise ConversionError("wids-meta.json shard list differs from conversion manifest.")
    if verify_workers < 1:
        raise ConversionError("--verify_workers must be positive.")
    if verify_workers == 1 or len(shards) == 1:
        total = sum(verify_shard(output_dir, shard) for shard in shards)
    else:
        workers = min(verify_workers, len(shards))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            total = sum(executor.map(verify_shard, [output_dir] * len(shards), shards))
    expected_total = sum(record.get("rows", 0) for record in manifest["inputs"].values() if record.get("status") == "complete")
    if total != expected_total:
        raise ConversionError(f"WIDS total {total} does not match manifest total {expected_total}.")
    for name, current in expected_fingerprints.items():
        record = manifest["inputs"].get(name)
        if not completed_input(record, current, output_dir):
            raise ConversionError(f"Input {name} is not completed in the conversion manifest.")
    print(f"Verified {total} samples across {len(shards)} WIDS shards in {output_dir}")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ConversionError("--batch_size must be positive.")
    if args.verify_workers <= 0:
        raise ConversionError("--verify_workers must be positive.")
    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise ConversionError(f"Input directory does not exist: {input_dir}")
    parquet_files = input_files(input_dir)
    output_dir = validate_paths(args.output_dir, parquet_files, input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # This expensive snapshot is intentional: it proves the read-only inputs
    # did not change between conversion start and final verification.
    before = fingerprint_inputs(parquet_files, args.verify_workers)
    if args.verify_only:
        verify_output(output_dir, before, args.verify_workers, input_dir)
        return

    manifest_path = output_dir / MANIFEST_NAME
    manifest = load_manifest(manifest_path, input_dir)
    manifest["last_started_at"] = utc_now()
    for path in parquet_files:
        source_before = before[path.name]
        record = manifest["inputs"].get(path.name)
        if completed_input(record, source_before, output_dir):
            print(f"Skipping completed input: {path.name}")
            continue

        print(f"Converting {path.name}")
        rows, shards = convert_input(path, output_dir, args.batch_size)
        source_after = fingerprint(path)
        if source_after != source_before:
            raise ConversionError(
                f"Input parquet changed while converting {path.name}; output is retained for inspection, "
                "but this input was not marked complete."
            )
        manifest["inputs"][path.name] = {
            "status": "complete",
            "rows": rows,
            "shards": shards,
            "source_before": source_before,
            "source_after": source_after,
            "completed_at": utc_now(),
        }
        atomic_json_write(manifest_path, manifest)
        write_wids_meta(output_dir, manifest)

    after = fingerprint_inputs(parquet_files, args.verify_workers)
    if after != before:
        raise ConversionError("At least one source parquet changed during conversion; refusing final verification.")
    manifest["last_verified_at"] = utc_now()
    atomic_json_write(manifest_path, manifest)
    write_wids_meta(output_dir, manifest)
    verify_output(output_dir, after, args.verify_workers, input_dir)


if __name__ == "__main__":
    main()
