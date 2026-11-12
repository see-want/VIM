# Ablations

Use the dataset, base model, and metric models downloaded in the
[README](../README.md). Each ablation trains a separate adapter and uses the same
configuration for training, prediction, and scoring.

## Front-only example

```bash
vim-train --config configs/front_only.yaml \
  --model checkpoints/qwen3.5-27b \
  --data data/vi_amo/prepared --output outputs/front_only

vim-predict --config configs/front_only.yaml \
  --base-model checkpoints/qwen3.5-27b \
  --model outputs/front_only/final_adapter \
  --data data/vi_amo/prepared/test.jsonl \
  --output outputs/front_only/predictions.jsonl

vim-evaluate --config configs/front_only.yaml \
  --predictions outputs/front_only/predictions.jsonl \
  --train data/vi_amo/prepared/train.jsonl \
  --sbert checkpoints/all-mpnet-base-v2 \
  --siglip checkpoints/siglip-so400m-patch14-384 \
  --device cuda --output outputs/front_only/evaluation
```

For no-context or single-frame experiments, substitute `configs/no_context.yaml`
or `configs/single_frame.yaml` and use a separate output directory.

## Image inference

Use `vim-infer --config configs/<ablation>.yaml` with the adapter trained for that
configuration and the same base model. The released `vim27b_lora_v2` adapter is
the full VIM model, so train a separate adapter for each ablation.

| Configuration | Image and history inputs |
| --- | --- |
| `front_only` | Four front images. For paired episodes, keep `--wrist` to indicate paired cameras: its files are not read, while the paired prompt is retained as in training. Omit it for single-camera episodes. |
| `no_context` | Four chronological images per camera spanning the observed prefix of the current mission. Any `--history` text is ignored. |
| `single_frame` | Exactly one image per camera, taken at action start. Use `--front front_start.png` and, when available, `--wrist wrist_start.png`. History is retained. |

Image inference uses the files you supply; it cannot select the mission prefix
or action start from timestamps. `vim-predict` selects these frames from the
dataset annotations. The single-frame evaluation uses only cutoff 0 and rejects
other `--cutoffs` values.
