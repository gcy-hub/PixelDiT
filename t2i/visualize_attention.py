#!/usr/bin/env python3
"""Generate PixelDiT T2I images with phrase-to-image joint-attention maps.

The script samples one image per JSON example, records the conditional branch
of PixelDiT's joint attention, and overlays a selected word or phrase map on
the generated image. It is designed for analysis, not high-throughput serving.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import yaml


T2I_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = T2I_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diffusion import DPMS
from diffusion.data.datasets import utils as dataset_utils
from diffusion.model.builder import build_model, get_tokenizer_and_text_encoder
from diffusion.model.utils import get_weight_dtype, prepare_prompt_ar
from diffusion.utils.config import PixDiTConfig, model_init_config


def load_pixeldit_config(path: Path) -> PixDiTConfig:
    """Apply a sparse PixelDiT YAML file over the project's dataclass defaults."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"PixelDiT config must be a mapping: {path}")
    config = PixDiTConfig()

    def update(target: Any, values: dict[str, Any]) -> None:
        for key, value in values.items():
            if not hasattr(target, key):
                raise ValueError(f"Unknown PixelDiT config field: {key}")
            current = getattr(target, key)
            if is_dataclass(current) and isinstance(value, dict):
                update(current, value)
            else:
                setattr(target, key, value)

    update(config, raw)
    return config


def parse_csv_ints(value: str, flag: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{flag} must be a comma-separated list of integers.") from exc
    if not values:
        raise ValueError(f"{flag} cannot be empty.")
    if any(item <= 0 for item in values):
        raise ValueError(f"{flag} values must be positive.")
    return values


def parse_layers(value: str, layer_count: int) -> list[int]:
    if value.strip().lower() == "all":
        return list(range(layer_count))
    try:
        layers = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("--layers must be a comma-separated list of zero-based layer indices.") from exc
    if not layers:
        raise ValueError("--layers cannot be empty.")
    if any(layer < 0 or layer >= layer_count for layer in layers):
        raise ValueError(f"--layers must be in [0, {layer_count - 1}] or use 'all'.")
    return sorted(set(layers))


def safe_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return normalized or "phrase"


class JointAttentionRecorder:
    """Keeps phrase attention maps while avoiding full joint-attention storage."""

    def __init__(
        self,
        target_token_indices: list[int],
        layer_indices: list[int],
        capture_calls: list[int],
        query_chunk_size: int,
    ) -> None:
        self.target_token_indices = target_token_indices
        self.layer_indices = set(layer_indices)
        self.capture_calls = set(capture_calls)
        self.query_chunk_size = query_chunk_size
        self.call_index = 0
        self.current_record: dict[str, Any] | None = None
        self.records: list[dict[str, Any]] = []

    def begin_model_call(self, timestep: torch.Tensor) -> None:
        self.call_index += 1
        self.current_record = None
        if self.call_index not in self.capture_calls:
            return
        self.current_record = {
            "call": self.call_index,
            "timestep": float(timestep[-1].detach().float().cpu()),
            "layers": {},
        }
        self.records.append(self.current_record)

    @torch.no_grad()
    def record(
        self,
        *,
        layer_index: int,
        image_queries: torch.Tensor,
        joint_keys: torch.Tensor,
        text_token_count: int,
        attn_mask: torch.Tensor | None,
        head_dim: int,
    ) -> None:
        if self.current_record is None or layer_index not in self.layer_indices:
            return
        if max(self.target_token_indices) >= text_token_count:
            raise RuntimeError("Selected phrase token does not exist in this attention layer.")

        # CFG concatenates [unconditional, conditional], so retain the latter.
        batch_index = image_queries.shape[0] - 1
        queries = image_queries[batch_index : batch_index + 1]
        keys = joint_keys[batch_index : batch_index + 1]
        target_indices = torch.tensor(self.target_token_indices, device=queries.device)
        key_transposed = keys.transpose(-2, -1)
        map_chunks: list[torch.Tensor] = []

        for start in range(0, queries.shape[-2], self.query_chunk_size):
            end = min(start + self.query_chunk_size, queries.shape[-2])
            scores = torch.matmul(queries[..., start:end, :], key_transposed) * (head_dim ** -0.5)
            if attn_mask is not None:
                mask = attn_mask[batch_index : batch_index + 1]
                if mask.shape[-2] != 1:
                    mask = mask[..., start:end, :]
                if mask.dtype == torch.bool:
                    scores = scores.masked_fill(~mask, float("-inf"))
                else:
                    scores = scores + mask
            probabilities = scores.float().softmax(dim=-1)
            phrase_attention = probabilities.index_select(-1, target_indices).sum(dim=-1)
            map_chunks.append(phrase_attention.mean(dim=1).squeeze(0).cpu())

        self.current_record["layers"][layer_index] = torch.cat(map_chunks)

    def aggregate(self) -> torch.Tensor:
        maps = [
            attention_map
            for record in self.records
            for attention_map in record["layers"].values()
        ]
        if not maps:
            captured = ",".join(str(item) for item in sorted(self.capture_calls))
            raise RuntimeError(f"No attention maps were captured. Requested model calls: {captured}.")
        return torch.stack(maps).mean(dim=0)

    def per_call_maps(self) -> list[tuple[int, float, torch.Tensor]]:
        result = []
        for record in self.records:
            maps = list(record["layers"].values())
            if maps:
                result.append((record["call"], record["timestep"], torch.stack(maps).mean(dim=0)))
        return result

    def summary(self) -> list[dict[str, Any]]:
        return [
            {
                "model_call": record["call"],
                "timestep": record["timestep"],
                "layers": sorted(record["layers"]),
            }
            for record in self.records
        ]


def load_examples(path: Path) -> list[dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("The examples JSON must be a non-empty list.")
    examples = []
    for index, item in enumerate(data):
        if not isinstance(item, dict) or not all(isinstance(item.get(key), str) for key in ("prompt", "phrase")):
            raise ValueError(f"Example {index} must contain string fields 'prompt' and 'phrase'.")
        examples.append(
            {
                "name": str(item.get("name") or f"example_{index + 1}"),
                "prompt": item["prompt"],
                "phrase": item["phrase"],
            }
        )
    return examples


def tokenize_with_offsets(tokenizer, prompt: str, max_length: int):
    common_kwargs = {
        "max_length": max_length,
        "padding": "max_length",
        "truncation": True,
        "return_tensors": "pt",
    }
    try:
        batch = tokenizer(prompt, return_offsets_mapping=True, **common_kwargs)
        offsets = batch.pop("offset_mapping")[0].tolist()
    except (NotImplementedError, TypeError, ValueError):
        batch = tokenizer(prompt, **common_kwargs)
        offsets = None
    return batch, offsets


def find_phrase_token_indices(
    tokenizer,
    retained_ids: list[int],
    retained_offsets: list[list[int]] | None,
    prompt: str,
    phrase: str,
) -> list[int]:
    phrase_start = prompt.casefold().find(phrase.casefold())
    if phrase_start < 0:
        raise ValueError(f"Phrase {phrase!r} is not present in the model input prompt.")
    phrase_end = phrase_start + len(phrase)

    if retained_offsets is not None:
        selected = [
            index
            for index, offset in enumerate(retained_offsets)
            if offset[0] < phrase_end and offset[1] > phrase_start
        ]
        if selected:
            return selected

    # Slow tokenizers can omit character offsets. Try common SentencePiece word
    # boundary variants and require a contiguous match in the retained sequence.
    for candidate in (phrase, f" {phrase}"):
        candidate_ids = tokenizer(candidate, add_special_tokens=False)["input_ids"]
        if not candidate_ids:
            continue
        for start in range(len(retained_ids) - len(candidate_ids) + 1):
            if retained_ids[start : start + len(candidate_ids)] == candidate_ids:
                return list(range(start, start + len(candidate_ids)))
    raise RuntimeError(
        f"Could not map phrase {phrase!r} to Gemma tokens. Inspect tokens.json and pass a more specific phrase."
    )


def prepare_conditioning(config, tokenizer, text_encoder, prompt: str, phrase: str, device: torch.device):
    image_size = int(config.model.image_size)
    base_ratios = getattr(dataset_utils, f"ASPECT_RATIO_{image_size}_TEST")
    prompt_clean, _, _, _, image_hw = prepare_prompt_ar(prompt, base_ratios, device=device, show=False)
    prompt_clean = prompt_clean.strip()
    chi_prompt = "\n".join(config.text_encoder.chi_prompt or [])
    model_input_prompt = f"{chi_prompt}{prompt_clean}"
    max_length = int(config.text_encoder.model_max_length)
    max_length_all = max_length
    if chi_prompt:
        max_length_all = len(tokenizer.encode(chi_prompt)) + max_length - 2

    token_batch, offsets = tokenize_with_offsets(tokenizer, model_input_prompt, max_length_all)
    full_ids = token_batch.input_ids[0].tolist()
    full_mask = token_batch.attention_mask[0].tolist()
    full_length = len(full_ids)
    selected_positions = [0] + list(range(full_length - max_length + 1, full_length))
    retained_ids = [full_ids[index] for index in selected_positions]
    retained_offsets = [offsets[index] for index in selected_positions] if offsets is not None else None
    target_indices = find_phrase_token_indices(tokenizer, retained_ids, retained_offsets, model_input_prompt, phrase)

    token_batch = token_batch.to(device)
    selected_tensor = torch.tensor(selected_positions, device=device)
    with torch.inference_mode():
        hidden_states = text_encoder(token_batch.input_ids, token_batch.attention_mask)[0]
    conditional = hidden_states.index_select(1, selected_tensor)[:, None]
    selected_mask = torch.tensor([full_mask[index] for index in selected_positions], device=device).unsqueeze(0)
    token_strings = tokenizer.convert_ids_to_tokens(retained_ids)
    token_rows = [
        {
            "index": index,
            "id": token_id,
            "token": token_strings[index],
            "offset": retained_offsets[index] if retained_offsets is not None else None,
            "selected_for_phrase": index in target_indices,
        }
        for index, token_id in enumerate(retained_ids)
    ]
    image_hw = image_hw.float()
    image_ar = (image_hw[:, 0] / image_hw[:, 1]).unsqueeze(1)
    return {
        "conditional": conditional,
        "mask": selected_mask,
        "image_hw": image_hw,
        "image_ar": image_ar,
        "height": int(image_hw[0, 0].item()),
        "width": int(image_hw[0, 1].item()),
        "prompt_clean": prompt_clean,
        "model_input_prompt": model_input_prompt,
        "target_indices": target_indices,
        "tokens": token_rows,
    }


def make_unconditional(config, tokenizer, text_encoder, negative_prompt: str, device: torch.device) -> torch.Tensor:
    max_length = int(config.text_encoder.model_max_length)
    tokens = tokenizer(
        negative_prompt,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        return text_encoder(tokens.input_ids, tokens.attention_mask)[0][:, None]


def load_model(config, model_path: Path, device: torch.device):
    model_kwargs = model_init_config(config, latent_size=int(config.model.image_size))
    model = build_model(
        config.model.model,
        use_fp32_attention=config.model.get("fp32_attention", False),
        **model_kwargs,
    ).to(device)
    # The official PixelDiT checkpoint is a trusted PyTorch checkpoint.
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict.pop("pos_embed", None)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.eval().to(get_weight_dtype(config.model.mixed_precision))
    return model, list(missing), list(unexpected)


def image_from_sample(sample: torch.Tensor) -> Image.Image:
    pixels = ((sample.detach().float().cpu().clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
    array = pixels.squeeze(0).permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB")


def normalized_heatmap(attention_map: torch.Tensor, grid_height: int, grid_width: int, height: int, width: int):
    if attention_map.numel() != grid_height * grid_width:
        raise ValueError("Attention map shape does not match the image patch grid.")
    heatmap = attention_map.view(1, 1, grid_height, grid_width).float()
    heatmap = F.interpolate(heatmap, size=(height, width), mode="bilinear", align_corners=False)[0, 0]
    low, high = torch.quantile(heatmap, torch.tensor([0.01, 0.99]))
    if float(high - low) < 1e-8:
        return torch.zeros_like(heatmap)
    return ((heatmap - low) / (high - low)).clamp(0, 1)


def colorize_heatmap(heatmap: torch.Tensor) -> np.ndarray:
    value = heatmap.cpu().numpy()
    red = np.clip(2.0 * value - 0.15, 0.0, 1.0)
    green = np.clip(1.8 - np.abs(2.0 * value - 1.0) * 1.8, 0.0, 1.0)
    blue = np.clip(1.1 - 1.8 * value, 0.0, 1.0)
    return (np.stack([red, green, blue], axis=-1) * 255).astype(np.uint8)


def save_attention_images(base_image: Image.Image, attention_map: torch.Tensor, output_dir: Path, stem: str) -> None:
    grid_height = base_image.height // 16
    grid_width = base_image.width // 16
    heatmap = normalized_heatmap(attention_map, grid_height, grid_width, base_image.height, base_image.width)
    heatmap_rgb = colorize_heatmap(heatmap)
    base_rgb = np.asarray(base_image).astype(np.float32)
    alpha = (heatmap.cpu().numpy()[..., None] * 0.58).astype(np.float32)
    overlay = (base_rgb * (1.0 - alpha) + heatmap_rgb.astype(np.float32) * alpha).round().astype(np.uint8)
    Image.fromarray(heatmap_rgb, mode="RGB").save(output_dir / f"{stem}_map.png")
    Image.fromarray(overlay, mode="RGB").save(output_dir / f"{stem}_overlay.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=T2I_ROOT / "configs/PixelDiT_1024px_pixel_diffusion_stage3.yaml")
    parser.add_argument(
        "--model_path",
        type=Path,
        default=PROJECT_ROOT / "ckpts/PixelDiT-1300M-1024px/pixeldit_t2i_v1.pth",
        help="PixelDiT base or fine-tuned checkpoint.",
    )
    parser.add_argument("--examples_json", type=Path, default=T2I_ROOT / "attention_examples.json")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--gpu", type=int, default=0, help="Physical CUDA device index.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=3.5)
    parser.add_argument("--layers", default="10,11,12,13", help="Zero-based layers, or 'all'.")
    parser.add_argument("--capture_calls", default="10,20,30,40,50", help="Model forward calls to record.")
    parser.add_argument("--query_chunk_size", type=int, default=128)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--save_per_call", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gpu < 0 or args.steps <= 0 or args.cfg_scale <= 0 or args.query_chunk_size <= 0:
        raise ValueError("--gpu, --steps, --cfg_scale, and --query_chunk_size must be positive.")
    config_path = args.config.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    examples_path = args.examples_json.expanduser().resolve()
    if not config_path.is_file() or not model_path.is_file() or not examples_path.is_file():
        raise FileNotFoundError("--config, --model_path, and --examples_json must point to existing files.")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    default_gemma = PROJECT_ROOT / "ckpts/gemma-2-2b-it"
    os.environ.setdefault("PIXDIT_GEMMA_PATH", str(default_gemma))
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for PixelDiT attention visualization.")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    config = load_pixeldit_config(config_path)
    examples = load_examples(examples_path)
    layers = parse_layers(args.layers, int(config.model.extra["patch_depth"]))
    capture_calls = parse_csv_ints(args.capture_calls, "--capture_calls")
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else (
        T2I_ROOT / "output" / f"attention_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    tokenizer, text_encoder = get_tokenizer_and_text_encoder(config.text_encoder.text_encoder_name, device=device)
    text_encoder.eval()
    model = None
    model, missing, unexpected = load_model(config, model_path, device)
    unconditional = make_unconditional(config, tokenizer, text_encoder, args.negative_prompt, device)
    metadata: dict[str, Any] = {
        "config": str(config_path),
        "model_path": str(model_path),
        "gemma_path": os.environ["PIXDIT_GEMMA_PATH"],
        "seed": args.seed,
        "steps": args.steps,
        "cfg_scale": args.cfg_scale,
        "layers": layers,
        "capture_calls": capture_calls,
        "missing_checkpoint_keys": missing,
        "unexpected_checkpoint_keys": unexpected,
        "examples": [],
    }

    try:
        for example in examples:
            conditioning = prepare_conditioning(
                config, tokenizer, text_encoder, example["prompt"], example["phrase"], device
            )
            recorder = JointAttentionRecorder(
                conditioning["target_indices"], layers, capture_calls, args.query_chunk_size
            )
            model.core.set_attention_recorder(recorder)
            generator = torch.Generator(device=device).manual_seed(args.seed)
            noise = torch.randn(
                1,
                3,
                conditioning["height"],
                conditioning["width"],
                device=device,
                generator=generator,
            )
            model_kwargs = {
                "data_info": {"img_hw": conditioning["image_hw"], "aspect_ratio": conditioning["image_ar"]},
                "mask": conditioning["mask"],
            }
            with torch.inference_mode():
                solver = DPMS(
                    model.forward_with_dpmsolver,
                    condition=conditioning["conditional"],
                    uncondition=unconditional,
                    cfg_scale=args.cfg_scale,
                    model_type="flow",
                    guidance_type="classifier-free",
                    model_kwargs=model_kwargs,
                    schedule="FLOW",
                    interval_guidance=[0, 1],
                )
                sample = solver.sample(
                    noise,
                    steps=args.steps,
                    order=2,
                    skip_type="time_uniform_flow",
                    method="multistep",
                    flow_shift=float(config.scheduler.flow_shift),
                )
            model.core.set_attention_recorder(None)

            stem = safe_name(example["name"])
            image = image_from_sample(sample)
            image.save(output_dir / f"{stem}_generated.png")
            aggregate_map = recorder.aggregate()
            save_attention_images(image, aggregate_map, output_dir, stem)
            if args.save_per_call:
                per_call_dir = output_dir / f"{stem}_per_call"
                per_call_dir.mkdir()
                for call, timestep, attention_map in recorder.per_call_maps():
                    save_attention_images(image, attention_map, per_call_dir, f"call_{call:03d}_t{timestep:.4f}")

            token_payload = {
                "prompt": example["prompt"],
                "model_input_prompt": conditioning["model_input_prompt"],
                "phrase": example["phrase"],
                "phrase_token_indices": conditioning["target_indices"],
                "tokens": conditioning["tokens"],
            }
            (output_dir / f"{stem}_tokens.json").write_text(json.dumps(token_payload, indent=2), encoding="utf-8")
            metadata["examples"].append(
                {
                    "name": example["name"],
                    "prompt": example["prompt"],
                    "phrase": example["phrase"],
                    "captured": recorder.summary(),
                }
            )
            print(f"Saved {stem} attention images under {output_dir}")
            torch.cuda.empty_cache()
    finally:
        if model is not None:
            model.core.set_attention_recorder(None)

    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Finished attention visualization: {output_dir}")


if __name__ == "__main__":
    main()
