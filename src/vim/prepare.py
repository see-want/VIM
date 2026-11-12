"""Convert annotations and package portable VI AMO episode manifests."""

import argparse
import hashlib
import os
import shutil
from pathlib import Path

from .io import read_jsonl, write_jsonl
from .prompts import context_text


def convert_views(record, root):
    sources = record["views"]
    if isinstance(sources, dict):
        sources = [
            {**({"path": view} if isinstance(view, str) else view), "id": name}
            for name, view in sources.items()
        ]
    views = []
    for index, source in enumerate(sources):
        identity = source.get("id", source.get("view_id", str(index)))
        view = {
            "id": identity,
            "name": source.get("name", source.get("view_type", identity)),
            "fps": source.get("fps", record["fps"]),
            "offset": source.get(
                "offset", source.get("start_frame", source.get("source_frame_offset", 0))
            ),
            "length": source.get(
                "length", source.get("source_frame_count", source.get("frame_count"))
            ),
            "primary": source.get("primary", source.get("is_primary", index == 0)),
        }
        if "frames" in source:
            view["frames"] = [str((root / path).resolve()) for path in source["frames"]]
        else:
            path = source["path"] if "path" in source else source["video_path"]
            view["path"] = str((root / path).resolve())
        views.append(view)
    return views


def convert(record, root, episodes=None):
    identity = record.get("parent_episode_key", record.get("episode_id"))
    if "views" not in record:
        metadata = (episodes or {}).get(str(identity))
        if metadata is None:
            raise ValueError(f"{identity}: missing views; supply camera metadata with --episodes.")
        metadata, root = metadata
        record = {**metadata, **record}
    if "atomic_tasks" in record:
        atomics = [
            {
                "id": a["atomic_task_id"],
                "label": a["atomic_task"],
                "start_frame": a["start_frame"],
                "end_frame": a["end_frame"],
            }
            for a in record["atomic_tasks"]
        ]
        missions = [
            {
                "id": m["mission_id"],
                "label": m["mission"],
                "action_ids": m["atomic_task_ids"],
                "start_frame": m["start_frame"],
            }
            for m in record["short_term_missions"]
        ]
    elif "missions" in record:
        atomics, missions = record["actions"], record["missions"]
    else:
        return {**record, "views": convert_views(record, Path(root))}
    atomics = sorted(atomics, key=lambda a: a["start_frame"])
    actions = []
    for atomic in atomics:
        mission = next(m for m in missions if atomic["id"] in m["action_ids"])
        previous = [
            a["label"]
            for a in atomics
            if a["id"] in mission["action_ids"] and a["end_frame"] < atomic["start_frame"]
        ]
        actions.append(
            {
                "id": atomic["id"],
                "start": atomic["start_frame"],
                "end": atomic["end_frame"],
                "action": atomic["label"],
                "mission": mission["label"],
                "mission_id": mission["id"],
                "mission_start": mission["start_frame"],
                "context": context_text(previous),
            }
        )
    return {
        "episode_id": identity,
        "dataset": record["dataset"],
        "fps": record["fps"],
        "views": convert_views(record, Path(root)),
        "actions": actions,
    }


def copy_media(record, output, copied):
    def copy(source):
        if source not in copied:
            path = Path(source)
            digest = hashlib.sha256(source.encode()).hexdigest()[:16]
            relative = Path("media") / f"{digest}{path.suffix}"
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if path != target.resolve():
                shutil.copyfile(path, target)
            copied[source] = relative.as_posix()
        return copied[source]

    for view in record["views"]:
        if "frames" in view:
            view["frames"] = [copy(path) for path in view["frames"]]
        else:
            view["path"] = copy(view["path"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="Directory with train/val/test.jsonl"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--episodes", type=Path, help="Camera metadata JSONL for annotation exports"
    )
    parser.add_argument(
        "--copy-media", action="store_true", help="Include media with relative paths"
    )
    args = parser.parse_args()
    episodes = {}
    if args.episodes:
        for row in read_jsonl(args.episodes):
            identity = row.get("id", row.get("parent_episode_key", row.get("episode_id")))
            episodes[str(identity)] = (row, args.episodes.resolve().parent)
    copied = {}
    for split in ("train", "val", "test"):
        rows = [convert(r, args.input, episodes) for r in read_jsonl(args.input / f"{split}.jsonl")]
        for row in rows:
            if args.copy_media:
                copy_media(row, args.output, copied)
            else:
                for view in row["views"]:
                    if "frames" in view:
                        view["frames"] = [os.path.relpath(p, args.output) for p in view["frames"]]
                    else:
                        view["path"] = os.path.relpath(view["path"], args.output)
        write_jsonl(args.output / f"{split}.jsonl", rows)
        print(f"{split}: {len(rows)} episodes, {sum(len(r['actions']) for r in rows)} actions")


if __name__ == "__main__":
    main()
