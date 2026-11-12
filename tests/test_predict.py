import sys
from pathlib import Path

import pytest
from PIL import Image

from vim import predict
from vim.data import IntentDataset
from vim.io import read_jsonl, write_jsonl
from vim.sampling import CUTOFFS, prefix_frames

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def manifest(tmp_path):
    views = []
    for view_index, name in enumerate(["front", "side", "overhead"]):
        paths = []
        for index in range(12):
            path = tmp_path / f"{name}_{index}.png"
            Image.new("RGB", (32, 32), (index * 20, view_index * 80, 0)).save(path)
            paths.append(path.name)
        views.append({"id": name, "name": name, "fps": 10, "frames": paths})
    path = tmp_path / "test.jsonl"
    write_jsonl(path, [{
        "episode_id": "demo", "dataset": "demo", "fps": 10, "views": views,
        "actions": [{"id": "a", "start": 4, "end": 10, "mission_start": 0,
                     "action": "place the block", "mission": "put the block in the tray",
                     "context": "Robot has no previous actions."}],
    }])
    return path


def test_custom_cutoffs_use_causal_frames_and_stable_camera_selection(manifest):
    reference = IntentDataset(manifest, "test")
    reference = {reference.row(i)["cutoff"]: reference.row(i) for i in range(len(reference))}
    dataset = IntentDataset(manifest, "test", cutoffs=range(101))
    rows = {dataset.row(i)["cutoff"]: dataset.row(i) for i in range(len(dataset))}
    assert len(rows) == 101
    for cutoff, row in rows.items():
        frames = row["primary_frames"]
        assert frames == sorted(frames)
        assert frames[0] == 4
        assert frames[-1] == 4 + round(6 * cutoff / 100)
        assert all(4 <= frame <= 10 for frame in frames)
    for cutoff in CUTOFFS:
        assert rows[cutoff]["primary_frames"] == reference[cutoff]["primary_frames"]
        assert rows[cutoff]["views"] == reference[cutoff]["views"]
    assert rows[50]["views"][0]["id"] == reference[40]["views"][0]["id"]
    assert rows[100]["views"][0]["id"] == reference[95]["views"][0]["id"]
    subset = IntentDataset(manifest, "test", cutoffs=[100, 50, 0])
    for index, cutoff in enumerate([100, 50, 0]):
        assert subset.row(index)["views"] == rows[cutoff]["views"]


@pytest.mark.parametrize("cutoff", [-5, 101])
def test_invalid_cutoffs_are_rejected_in_sampling_and_dataset(manifest, cutoff):
    with pytest.raises(ValueError, match="0.*100"):
        prefix_frames({"start": 4, "end": 10}, cutoff)
    with pytest.raises(ValueError, match="0.*100"):
        IntentDataset(manifest, "test", cutoffs=[cutoff])


@pytest.mark.parametrize("cutoff", ["-5", "101", "50.5"])
def test_invalid_cutoff_cli_does_not_load_model(tmp_path, monkeypatch, capsys, cutoff):
    argv = ["vim-predict", "--data", str(tmp_path / "absent.jsonl"),
            "--model", "unused", "--output", str(tmp_path / "output.jsonl"),
            "--cutoffs", cutoff]

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid cutoffs must be rejected before loading the model")

    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(predict, "load_model", unexpected)
    with pytest.raises(SystemExit) as error:
        predict.main()
    assert error.value.code == 2
    assert "--cutoffs" in capsys.readouterr().err


@pytest.mark.parametrize("name,cutoffs,expected", [
    ("vim27b", [0, 50, 100], [0, 50, 100]),
    ("single_frame", None, [0]),
    ("single_frame", [0], [0]),
])
def test_prediction_cli_writes_requested_cutoffs(
    manifest, tmp_path, monkeypatch, name, cutoffs, expected,
):
    output = tmp_path / "predictions.jsonl"
    argv = ["vim-predict", "--config", str(ROOT / "configs" / f"{name}.yaml"),
            "--data", str(manifest), "--model", "unused", "--output", str(output)]
    if cutoffs is not None:
        argv += ["--cutoffs", *map(str, cutoffs)]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(predict, "load_model", lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(predict, "generate", lambda model, processor, examples, tokens: [
        "Ongoing action: place the block\nOngoing mission: put the block in the tray"
        for _ in examples
    ])
    predict.main()
    assert [row["cutoff"] for row in read_jsonl(output)] == expected


def test_single_frame_rejects_nonzero_cutoffs(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [
        "vim-predict", "--config", str(ROOT / "configs/single_frame.yaml"),
        "--data", str(tmp_path / "absent.jsonl"), "--model", "unused",
        "--output", str(tmp_path / "output.jsonl"), "--cutoffs", "50",
    ])

    def unexpected(*args, **kwargs):
        pytest.fail("Single-frame cutoff validation must happen before model loading")

    monkeypatch.setattr(predict, "load_model", unexpected)
    with pytest.raises(SystemExit) as error:
        predict.main()
    assert error.value.code == 2
    assert "single_frame" in capsys.readouterr().err
