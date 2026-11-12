# Data format

The VI AMO v2 Hugging Face release contains `train.jsonl`, `val.jsonl`, `test.jsonl`,
`videos/`, and `meta/`. Each JSONL line describes an episode. Run `vim-prepare`
once, then use its output manifests for training, prediction, and evaluation:

```bash
vim-prepare --input data/vi_amo --output data/vi_amo/prepared
```

## Episode annotations

```json
{
  "episode_id": "demo_001",
  "dataset": "demo",
  "instruction": "Put the red block in the tray",
  "fps": 10,
  "num_frames": 32,
  "views": [
    {"id": "front", "name": "front", "path": "videos/demo_001/front.mp4",
     "fps": 10, "offset": 0, "length": 32, "primary": true},
    {"id": "wrist", "name": "wrist", "path": "videos/demo_001/wrist.mp4",
     "fps": 10, "offset": 0, "length": 32, "primary": false}
  ],
  "actions": [
    {"id": "a1", "label": "pick up the red block", "start_frame": 0, "end_frame": 15},
    {"id": "a2", "label": "place the red block in the tray", "start_frame": 16, "end_frame": 31}
  ],
  "missions": [
    {"id": "m1", "label": "put the red block in the tray", "start_frame": 0,
     "end_frame": 31, "action_ids": ["a1", "a2"]}
  ]
}
```

Frame indices are zero-based and both interval ends are inclusive. Each view maps
the episode timeline to its own FPS and frame offset. Release videos start at
episode frame zero. Paths are relative to the containing manifest.

`source_family` groups source subsets. `temporal_schema` identifies serial or
overlapping action intervals; bimanual actions can include `hand` values of
`left`, `right`, or `both`. The full-episode `instruction` is metadata; VIM's
language input is the completed-action history.

## Prepared manifests

Preparation preserves episode order, splits, labels, and boundaries. It maps each
action to its mission and computes history from actions in that mission whose
end frame precedes the current action's start frame. This also handles overlapping
bimanual actions.

Each prepared action contains:

| Field | Meaning |
| --- | --- |
| `id`, `start`, `end` | Action identity and inclusive frame interval |
| `action`, `mission` | OA and OM target labels |
| `mission_id`, `mission_start` | Mission identity and first frame |
| `context` | Completed actions within that mission |

The model target is:

```text
Ongoing action: pick up the red block
Ongoing mission: put the red block in the tray
```

Prepared media paths are relative to the output manifest. Videos are read on
demand as RGB, with four observations per view or one for the single-frame
ablation. Moving the dataset together with its `prepared/` directory preserves
these paths. A view can also use `frames: ["frames/000.png", "frames/001.png"]`
instead of a video `path`.

Evaluation defaults to the nine action-progress cutoffs used in the paper.
`vim-predict --cutoffs` also accepts integer percentages from 0 to 100; the final
frame is sampled at the requested progress. For episodes without a paired wrist
camera, intermediate cutoffs use the camera assigned to the preceding slot in
the original 25, 40, 55, 70, 85, 95% schedule; cutoffs below 25% use its first slot.
This preserves camera selection at the default cutoffs and when selecting a
subset of cutoffs.

## Custom annotations

`vim-prepare` also accepts already prepared manifests, annotations with
`parent_episode_key`, `atomic_tasks`, and `short_term_missions`, and
[`vi_amo_data_annotation`](https://github.com/vision-intention-action/vi_amo_data_annotation)
exports. For exports without cameras, provide the episode-input JSONL through
`--episodes`; records are matched by episode identity:

```json
{"id":"demo_001","fps":10,"views":{"front":"videos/front.mp4","wrist":"videos/wrist.mp4"}}
```

```bash
vim-prepare --input annotations --episodes episodes.jsonl \
  --output data/custom --copy-media
```

`--copy-media` copies each referenced video or image once into the output's
`media/` directory. Omit it to reference the existing files, and omit `--episodes`
when camera metadata is already present in the annotations.
