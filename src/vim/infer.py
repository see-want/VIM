"""Infer ongoing action and mission from images using the selected configuration."""

import argparse
from pathlib import Path

from PIL import Image

from .io import load_config
from .media import close_images, resize_image
from .models import generate, load_model
from .prompts import context_text, messages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vim27b.yaml")
    parser.add_argument("--model", default="checkpoints/vim27b_lora_v2", help="LoRA adapter directory")
    parser.add_argument("--base-model", help="Local Qwen base-model directory")
    parser.add_argument(
        "--front", type=Path, nargs="+", required=True,
        help="Four oldest-to-newest images, or one action-start image for single_frame",
    )
    parser.add_argument(
        "--wrist", type=Path, nargs="+",
        help="Images aligned with --front; front_only retains the paired prompt but ignores pixels",
    )
    parser.add_argument("--history", nargs="*", default=[], help="Completed actions in the current mission")
    args = parser.parse_args()
    config = load_config(args.config)
    data = config["data"]
    single = data.get("single_frame", False)
    front_only = data.get("front_only", False)
    views = [("--front", args.front)]
    if args.wrist and not front_only:
        views.append(("--wrist", args.wrist))
    expected = 1 if single else 4
    for flag, paths in views:
        if len(paths) != expected:
            parser.error(f"{args.config} requires {expected} image(s) for {flag}; got {len(paths)}")
    frames = []
    for _, paths in views:
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        frames.append(images)
    if single:
        frames = [[resize_image(f[0])] for f in frames]
    context = "" if data.get("no_context", False) else context_text(args.history)
    # Match the front-only training protocol for episodes with paired cameras.
    paired_prompt = front_only and bool(args.wrist)
    example = {"messages": messages(frames, context, single_frame=single, paired_prompt=paired_prompt)}
    if single:
        example["images"] = [f[0] for f in frames]
    model, processor = load_model(args.model, config, base_model=args.base_model)
    prediction = generate(model, processor, [example], config["generation"]["max_new_tokens"])[0]
    close_images(example)
    print(prediction)


if __name__ == "__main__":
    main()
