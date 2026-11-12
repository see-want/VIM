import shutil
import sys

import av
import numpy as np
import pytest
from PIL import Image

from vim.data import IntentDataset
from vim.io import read_jsonl, write_jsonl
from vim.media import close_images, frame_indices, read_frames
from vim.prepare import convert, copy_media, main


@pytest.fixture
def annotation():
    return {
        "episode_id": "robot/1",
        "dataset": "robot",
        "fps": 10,
        "num_frames": 8,
        "actions": [
            {"id": 0, "label": "pick up the lemon", "start_frame": 0, "end_frame": 2},
            {"id": 1, "label": "place the lemon", "start_frame": 3, "end_frame": 5},
            {"id": 2, "label": "close the drawer", "start_frame": 6, "end_frame": 7},
        ],
        "missions": [
            {
                "id": 0,
                "label": "put the lemon on the table",
                "action_ids": [0, 1],
                "start_frame": 0,
                "end_frame": 5,
            },
            {
                "id": 1,
                "label": "close the drawer",
                "action_ids": [2],
                "start_frame": 6,
                "end_frame": 7,
            },
        ],
    }


def test_annotation_targets_and_mission_history(annotation, tmp_path):
    source = {**annotation, "views": {"front": "front.mp4"}}
    record = convert(source, tmp_path)
    assert [a["context"] for a in record["actions"]] == [
        "Robot has no previous actions.",
        'Robot already finished actions: "pick up the lemon".',
        "Robot has no previous actions.",
    ]
    assert record["actions"][1] == {
        "id": 1,
        "start": 3,
        "end": 5,
        "action": "place the lemon",
        "mission": "put the lemon on the table",
        "mission_id": 0,
        "mission_start": 0,
        "context": 'Robot already finished actions: "pick up the lemon".',
    }
    assert convert(record, tmp_path) == record


@pytest.mark.parametrize("previous_end", [2, 3, 4])
def test_overlapping_actions_are_not_completed_history(annotation, tmp_path, previous_end):
    annotation["actions"][0]["end_frame"] = previous_end
    record = convert({**annotation, "views": {"front": "front.mp4"}}, tmp_path)
    expected = (
        'Robot already finished actions: "pick up the lemon".'
        if previous_end == 2
        else "Robot has no previous actions."
    )
    assert record["actions"][1]["context"] == expected


def test_original_annotations_preserve_camera_timing(annotation, tmp_path):
    source = {
        "parent_episode_key": annotation["episode_id"],
        "dataset": "robot",
        "fps": 10,
        "atomic_tasks": [
            {
                "atomic_task_id": a["id"],
                "atomic_task": a["label"],
                "start_frame": a["start_frame"],
                "end_frame": a["end_frame"],
            }
            for a in annotation["actions"]
        ],
        "short_term_missions": [
            {
                "mission_id": m["id"],
                "mission": m["label"],
                "atomic_task_ids": m["action_ids"],
                "start_frame": m["start_frame"],
            }
            for m in annotation["missions"]
        ],
        "views": [
            {
                "view_id": "camera",
                "view_type": "front",
                "fps": 20,
                "video_path": "front.mp4",
                "source_frame_offset": 3,
                "source_frame_count": 19,
                "is_primary": True,
            }
        ],
    }
    record = convert(source, tmp_path)
    expected = convert({**annotation, "views": {"front": "front.mp4"}}, tmp_path)
    assert record["actions"] == expected["actions"]
    view = record["views"][0]
    assert (view["id"], view["name"], view["primary"]) == ("camera", "front", True)
    assert frame_indices(view, record["fps"], [0, 3, 5, 7]) == [3, 9, 13, 17]


@pytest.mark.parametrize("copy", [False, True])
def test_prepared_dataset_can_move_with_media(annotation, tmp_path, monkeypatch, copy):
    package = tmp_path / "release"
    source = package / "source"
    source.mkdir(parents=True)
    video = source / "camera.mp4"
    with av.open(str(video), "w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 32, 32, "yuv420p"
        for i in range(9):
            frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), i * 25, np.uint8), "rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    metadata = []
    for split in ("train", "val", "test"):
        record = {**annotation, "episode_id": f"robot/{split}"}
        write_jsonl(source / "raw" / f"{split}.jsonl", [record])
        metadata.append(
            {
                "id": record["episode_id"],
                "views": {"front": "camera.mp4", "wrist": {"path": "camera.mp4", "start_frame": 1}},
            }
        )
    write_jsonl(source / "episodes.jsonl", metadata)
    output = package / "prepared"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vim-prepare",
            "--input",
            str(source / "raw"),
            "--output",
            str(output),
            "--episodes",
            str(source / "episodes.jsonl"),
        ] + (["--copy-media"] if copy else []),
    )
    main()
    if copy:
        assert len(list((output / "media").iterdir())) == 1
    else:
        assert not (output / "media").exists()
    moved = tmp_path / "download" / "vi_amo"
    moved.parent.mkdir()
    shutil.move(package, moved)
    if copy:
        shutil.rmtree(moved / "source")
    for split, expected in (("train", 18), ("val", 3), ("test", 27)):
        manifest = moved / "prepared" / f"{split}.jsonl"
        assert str(tmp_path) not in manifest.read_text()
        record = read_jsonl(manifest)[0]
        assert record["episode_id"] == f"robot/{split}"
        dataset = IntentDataset(manifest, split)
        assert len(dataset) == expected
        for i in range(len(dataset)):
            example = dataset.example(i, training=split != "test")
            videos = [
                item["video"]
                for item in example["messages"][1]["content"]
                if item["type"] == "video"
            ]
            assert len(videos) == 2
            assert all(len(frames) == 4 for frames in videos)
            assert np.asarray(videos[1][0]).mean() > np.asarray(videos[0][0]).mean() + 10
            assert (example["messages"][-1]["role"] == "assistant") == (split != "test")
            close_images(example)


def test_packaged_image_sequences_keep_order(annotation, tmp_path):
    paths = []
    for i in range(8):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (32, 32), (i * 20, 0, 0)).save(path)
        paths.append(path.name)
    source = {**annotation, "views": [{"id": "front", "frames": paths}]}
    record = convert(source, tmp_path)
    output = tmp_path / "release"
    copy_media(record, output, {})
    frames = read_frames(record["views"][0], [7, 0, 7, 3], output)
    assert [image.getpixel((0, 0)) for image in frames] == [
        (140, 0, 0),
        (0, 0, 0),
        (140, 0, 0),
        (60, 0, 0),
    ]
    assert source["views"][0]["frames"] == paths
    for image in frames:
        image.close()


def test_missing_cameras_requests_metadata(annotation, tmp_path):
    with pytest.raises(ValueError, match="--episodes"):
        convert(annotation, tmp_path)
