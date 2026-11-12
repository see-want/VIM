"""Read synchronized RGB observations from video or image sequences."""

import math
from pathlib import Path

import av
from PIL import Image


def resize_image(image):
    width, height = image.size
    rh, rw = max(32, round(height / 32) * 32), max(32, round(width / 32) * 32)
    if rh * rw > 16384 * 28 * 28:
        scale = math.sqrt(height * width / (16384 * 28 * 28))
        rh, rw = math.floor(height / scale / 32) * 32, math.floor(width / scale / 32) * 32
    elif rh * rw < 4 * 28 * 28:
        scale = math.sqrt(4 * 28 * 28 / (height * width))
        rh, rw = math.ceil(height * scale / 32) * 32, math.ceil(width * scale / 32) * 32
    result = image.resize((rw, rh))
    image.close()
    return result


def frame_indices(view, primary_fps, indices):
    mapped = [
        view.get("offset", 0) + max(0, math.floor(i / primary_fps * view["fps"] + 1e-9))
        for i in indices
    ]
    length = view.get("length")
    return [min(i, length - 1) for i in mapped] if length else mapped


def read_frames(view, indices, root):
    if "frames" in view:
        result = []
        for i in indices:
            with Image.open(Path(root) / view["frames"][i]) as image:
                result.append(image.convert("RGB"))
        return result
    wanted = set(indices)
    decoded = {}
    with av.open(str(Path(root) / view["path"])) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i in wanted:
                decoded[i] = frame.to_image().convert("RGB")
                if len(decoded) == len(wanted):
                    break
    if len(decoded) != len(wanted):
        raise ValueError(f"Missing requested frames in {view['path']}")
    return [decoded[i].copy() for i in indices]


def close_images(example):
    for turn in example["messages"]:
        for block in turn["content"]:
            images = block.get("video", []) if block["type"] == "video" else []
            if block["type"] == "image":
                images = [block["image"]]
            for image in images:
                image.close()
