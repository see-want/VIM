"""Lazy image loading for training plans and causal evaluation inputs."""

import random
from bisect import bisect_right
from pathlib import Path

from torch.utils.data import Dataset

from .io import read_jsonl
from .media import frame_indices, read_frames
from .prompts import messages, target_text
from .sampling import CUTOFFS, PLANS, prefix_frames, training_frames


def select_views(views, count, seed, front_only=False, primary_only=False):
    primary = next((v for v in views if v.get("primary")), views[0])
    front = next((v for v in views if "front" in v["name"].lower()), primary)
    wrist = next((v for v in views if "wrist" in v["name"].lower()), None)
    if front_only:
        return [[front] for _ in range(count)]
    if wrist is not None and wrist is not front:
        return [[front, wrist] for _ in range(count)]
    if primary_only:
        return [[primary] for _ in range(count)]
    schedule = list(views) + [primary] * max(0, count - len(views))
    random.Random(seed).shuffle(schedule)
    return [[v] for v in schedule[:count]]


class IntentDataset(Dataset):
    def __init__(self, manifest, mode="train", config=None, cutoffs=CUTOFFS):
        config = config or {}
        self.root = Path(manifest).resolve().parent
        self.records = read_jsonl(manifest)
        self.samples = []
        self.single = config.get("single_frame", False)
        self.no_context = config.get("no_context", False)
        self.front_only = config.get("front_only", False)
        seed = config.get("seed", 3407)
        plans = config.get("plans", PLANS)
        mission_window = config.get("mission_window", False)
        for episode_index, episode in enumerate(self.records):
            sampled = training_frames(
                episode["actions"],
                seed + episode_index,
                plans if mode == "train" else ("uniform",),
                mission_window,
            )
            for action_index, action in enumerate(episode["actions"]):
                view_seed = seed + episode_index + 1009 * action_index
                views = episode["views"]
                if self.single:
                    indices = [[action["start"]]] * (len(plans) if mode == "train" else 1)
                    cameras = select_views(views, len(PLANS), view_seed)[0]
                    camera_sets = [cameras] * len(indices)
                    tags = [0] * len(indices)
                elif mode == "test":
                    indices = [prefix_frames(action, c, mission_window) for c in cutoffs]
                    # Preserve the original camera schedule. Intermediate cutoffs reuse
                    # the preceding slot; early observations reuse the first slot.
                    late_cutoffs = CUTOFFS[3:]
                    late = select_views(views, len(late_cutoffs), view_seed, self.front_only)
                    camera_sets = [
                        late[max(0, bisect_right(late_cutoffs, c) - 1)] for c in cutoffs
                    ]
                    tags = list(cutoffs)
                else:
                    indices = sampled[action_index]
                    camera_sets = select_views(
                        views,
                        len(indices),
                        view_seed,
                        config.get("front_only", False),
                        mode == "val",
                    )
                    tags = list(plans) if mode == "train" else ["uniform"]
                for variant, (frames, cameras, tag) in enumerate(zip(indices, camera_sets, tags)):
                    self.samples.append(
                        {
                            "episode": episode_index,
                            "action_index": action_index,
                            "frames": frames,
                            "views": cameras,
                            "tag": tag,
                            "variant": variant,
                        }
                    )

    def __len__(self):
        return len(self.samples)

    def row(self, index):
        sample = self.samples[index]
        episode = self.records[sample["episode"]]
        action = episode["actions"][sample["action_index"]]
        views = [
            {**view, "indices": frame_indices(view, episode["fps"], sample["frames"])}
            for view in sample["views"]
        ]
        return {
            "sample_id": f"{episode['episode_id']}/{action['id']}/{sample['tag']}/{sample['variant']}",
            "episode_id": episode["episode_id"],
            "dataset": episode["dataset"],
            "action_id": action["id"],
            "cutoff": sample["tag"],
            "context": "" if self.no_context else action["context"],
            "action": action["action"],
            "mission": action["mission"],
            "views": views,
            "primary_frames": sample["frames"],
        }

    def example(self, index, training=False):
        row = self.row(index)
        frames = [read_frames(v, v["indices"], self.root) for v in row["views"]]
        target = target_text(row["action"], row["mission"]) if training else None
        original_views = self.records[self.samples[index]["episode"]]["views"]
        paired_prompt = self.front_only and any(
            "wrist" in v["name"].lower() for v in original_views
        )
        if self.single:
            from .media import resize_image

            frames = [[resize_image(f[0])] for f in frames]
        result = {"messages": messages(frames, row["context"], target, self.single, paired_prompt)}
        if self.single:
            result["images"] = [f[0] for f in frames]
        return result

    def __getitem__(self, index):
        return self.example(index, training=True)
