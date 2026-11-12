import sys
from pathlib import Path

import pytest
from PIL import Image

from vim import infer
from vim.data import IntentDataset
from vim.io import load_config, write_jsonl
from vim.media import close_images
from vim.prompts import context_text

ROOT = Path(__file__).resolve().parents[1]


def snapshot(value):
    if isinstance(value, Image.Image):
        return value.mode, value.size, value.tobytes()
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [snapshot(item) for item in value]
    return value


@pytest.mark.parametrize("name", ["vim27b", "front_only", "no_context", "single_frame"])
@pytest.mark.parametrize("paired", [False, True])
def test_image_cli_matches_dataset_inputs(tmp_path, monkeypatch, capsys, name, paired):
    views = []
    for view_index, view in enumerate(["front", "wrist"] if paired else ["front"]):
        paths = []
        for index in range(12):
            path = tmp_path / f"{view}_{index}.png"
            Image.new("RGB", (35, 51), (index * 20, view_index * 100, 30)).save(path)
            paths.append(path.name)
        views.append({"id": view, "name": view, "frames": paths, "fps": 10})
    history = "pick up the block"
    manifest = tmp_path / "test.jsonl"
    write_jsonl(manifest, [{
        "episode_id": "demo", "dataset": "demo", "fps": 10, "views": views,
        "actions": [{"id": "a", "start": 4, "end": 10, "mission_start": 0,
                     "action": "place the block", "mission": "put the block in the tray",
                     "context": context_text([history])}],
    }])
    config_path = ROOT / "configs" / f"{name}.yaml"
    config = load_config(config_path)
    dataset = IntentDataset(manifest, "test", config["data"], cutoffs=[70])
    row = dataset.row(0)
    expected = dataset.example(0)
    expected_snapshot = snapshot(expected)
    close_images(expected)
    argv = ["vim-infer", "--config", str(config_path), "--front"]
    argv += [str(tmp_path / views[0]["frames"][i]) for i in row["views"][0]["indices"]]
    if paired:
        argv += ["--wrist"]
        if name == "front_only":
            # Wrist paths signal paired data but their pixels must not be read.
            argv += [str(tmp_path / f"absent_wrist_{i}.png") for i in range(4)]
        else:
            argv += [str(tmp_path / views[1]["frames"][i]) for i in row["views"][1]["indices"]]
    argv += ["--history", history]
    captured = []

    def generate(model, processor, examples, max_new_tokens):
        captured.append(snapshot(examples[0]))
        return ["Ongoing action: place the block\nOngoing mission: put the block in the tray"]

    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(infer, "load_model", lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(infer, "generate", generate)
    infer.main()
    assert captured == [expected_snapshot]
    assert "Ongoing action: place the block" in capsys.readouterr().out


@pytest.mark.parametrize("name,front_count,wrist_count", [
    ("vim27b", 3, 0), ("vim27b", 4, 3),
    ("single_frame", 4, 0), ("single_frame", 1, 4),
])
def test_wrong_image_counts_fail_before_reading_images_or_loading_model(
    tmp_path, monkeypatch, capsys, name, front_count, wrist_count,
):
    argv = ["vim-infer", "--config", str(ROOT / "configs" / f"{name}.yaml"), "--front"]
    argv += [str(tmp_path / f"front_{i}.png") for i in range(front_count)]
    if wrist_count:
        argv += ["--wrist"] + [str(tmp_path / f"wrist_{i}.png") for i in range(wrist_count)]

    def unexpected(*args, **kwargs):
        pytest.fail("Input validation must happen before image/model loading")

    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(infer.Image, "open", unexpected)
    monkeypatch.setattr(infer, "load_model", unexpected)
    with pytest.raises(SystemExit) as error:
        infer.main()
    assert error.value.code == 2
    assert "requires" in capsys.readouterr().err
