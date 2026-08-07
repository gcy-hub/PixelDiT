#!/usr/bin/env python3
"""Append configured validation prompts and authored training-style T2I prompts."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import yaml


AUTHORED_TRAINING_STYLE_PROMPTS = [
    'a red and cream bakery storefront sign, with the visible words "MORNING" and "BAKERY" clearly rendered in the image.',
    'a green canvas farmers market tote bag, with the visible words "FRESH" and "HARVEST" clearly rendered in the image.',
    'a colorful elementary school science fair poster, with the visible words "SCIENCE" and "DISCOVERY" clearly rendered in the image.',
    'a matte black coffee bag label, with the visible words "NORTH" and "ROAST" clearly rendered in the image.',
    'a bright travel magazine cover showing a tropical island, with the visible words "ISLAND" and "ESCAPE" clearly rendered in the image.',
    'a vintage roadside motel neon sign at night, with the visible words "SUNSET" and "MOTEL" clearly rendered in the image.',
    'a cardboard jigsaw puzzle box with an illustrated forest, with the visible words "WILD" and "TRAILS" clearly rendered in the image.',
    'a blue and gold hardcover astronomy book, with the visible words "ATLAS" and "STARS" clearly rendered in the image.',
    'a cheerful children\'s storybook cover with a small fox, with the visible words "FOX" and "FRIENDS" clearly rendered in the image.',
    'a weathered garage door sign, with the visible words "RIVER" and "REPAIRS" clearly rendered in the image.',
    'an orange basketball jersey hanging in a locker, with the visible word "TIGERS" clearly rendered in the image.',
    'a music festival admission wristband, with the visible words "ECHO" and "FEST" clearly rendered in the image.',
    'a letterpress business card on textured paper, with the visible words "STUDIO" and "NORTH" clearly rendered in the image.',
    'a dramatic adventure movie poster featuring a mountain climber, with the visible words "SUMMIT" and "BEYOND" clearly rendered in the image.',
    'an embroidered camping patch on a khaki backpack, with the visible words "TRAIL" and "CLUB" clearly rendered in the image.',
    'a pink cosmetic tube label on a clean studio background, with the visible words "BLOOM" and "CARE" clearly rendered in the image.',
    'a retro vinyl record album cover with abstract waves, with the visible words "MIDNIGHT" and "SIGNALS" clearly rendered in the image.',
    'a striped ice cream cart in a sunny park, with the visible words "SCOOP" and "SMILE" clearly rendered in the image.',
    'a navy baseball cap with white embroidery, with the visible words "COAST" and "CREW" clearly rendered in the image.',
    'a watercolor travel postcard of a mountain lake, with the visible words "ALPINE" and "LAKE" clearly rendered in the image.',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompts_file", type=Path, required=True)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def load_validation_prompts(config_path: Path) -> list[str]:
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    prompts = config.get("train", {}).get("validation_prompts")
    if not isinstance(prompts, list) or not prompts or not all(isinstance(prompt, str) and prompt.strip() for prompt in prompts):
        raise ValueError(f"No valid train.validation_prompts in {config_path}")
    return [" ".join(prompt.split()) for prompt in prompts]


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
    config_path = args.config.resolve()
    prompts_path = args.prompts_file.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config does not exist: {config_path}")
    if not prompts_path.parent.is_dir():
        raise FileNotFoundError(f"Prompt file parent does not exist: {prompts_path.parent}")

    validation_prompts = load_validation_prompts(config_path)
    additions = validation_prompts + AUTHORED_TRAINING_STYLE_PROMPTS
    existing = set(prompts_path.read_text(encoding="utf-8").splitlines()) if prompts_path.exists() else set()
    duplicates = [prompt for prompt in additions if prompt in existing]
    if duplicates:
        raise ValueError(f"Refusing to append {len(duplicates)} duplicate prompt(s).")

    if args.dry_run:
        print(f"Dry run: would append {len(validation_prompts)} validation prompts and "
              f"{len(AUTHORED_TRAINING_STYLE_PROMPTS)} authored prompts to {prompts_path}")
        return

    append_lines(prompts_path, additions)
    print(f"Appended {len(validation_prompts)} validation prompts and "
          f"{len(AUTHORED_TRAINING_STYLE_PROMPTS)} authored prompts to {prompts_path}")


if __name__ == "__main__":
    main()
