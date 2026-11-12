# VIM

Vision-Intention Model for predicting an operator's **hierarchical intent** from
visual teleoperation history, jointly identifying their **ongoing action (OA)**
and **ongoing mission (OM)**.

VIM uses four chronological observations, synchronized front and wrist views when
available, and the completed actions in the current mission. We finetune Qwen3.5
27B with LoRA on a single Nvidia A100 80GB GPU.

## File Layout

```text
configs/          Model and ablation settings
src/vim/          Data preparation, sampling, training, inference, and metrics
checkpoints/      Local model downloads (ignored by Git)
examples/lemon/   Paired images and annotations for direct inference
docs/             Dataset format and ablation examples
tests/            Data loading, sampling, prompts, and metrics
```

## Install

Use Python 3.11 or 3.12 with CUDA and a compatible PyTorch installation.
The 27B configuration uses BF16 weights and an A100 80GB GPU.

```bash
git clone https://github.com/vision-intention-action/VIM.git
cd VIM
python -m venv .venv
source .venv/bin/activate
pip install -e ".[train,eval]"
```

## Download and prepare data

VI AMO v2 contains one MP4 per episode and camera, with centralized JSONL annotations
and fixed train/val/test splits of 5,500/550/550 episodes.

| Dataset | Revision | Actions | Missions |
| --- | --- | ---: | ---: |
| [VI AMO v2](https://huggingface.co/datasets/viamodelanddataset/vi_amo_v2) | `v2` | 13,653 | 9,702 |

The dataset repository is currently private. Log in with an account that has access,
then download and prepare VI AMO v2:

```bash
hf auth login
hf download viamodelanddataset/vi_amo_v2 --repo-type dataset \
  --revision v2 --local-dir data/vi_amo

vim-prepare --input data/vi_amo --output data/vi_amo/prepared
```

```text
data/vi_amo/
  train.jsonl, val.jsonl, test.jsonl   Episode annotations
  videos/                            Episode and camera MP4 files
  meta/                              Counts, provenance, and checksums
  prepared/                          Training and evaluation manifests
```

Preparation derives OA/OM targets and completed-action history while preserving
the supplied splits and frame boundaries. It references the downloaded videos
without copying or re-encoding them. Paths are relative, so the entire
`data/vi_amo/` directory can be moved. See [data format](docs/data.md) for the
schema and custom annotations.

## Download the base model

Training and inference share the same [Qwen3.5-27B](https://huggingface.co/unsloth/Qwen3.5-27B) base model:

```bash
hf download unsloth/Qwen3.5-27B \
  --revision 68443620da0c3847c1ba5042b5dcf1e78c42bfba \
  --local-dir checkpoints/qwen3.5-27b
```

## Download the released adapter

The [released 27B LoRA](https://huggingface.co/viamodelanddataset/vim27b_lora_v2)
is trained on **VI AMO v2** (13,653 actions and 9,702 missions) and is the
checkpoint used for the v2 **VIM (Ours)** results. The adapter weights and
processor are hosted on Hugging Face; local model downloads are ignored by Git.

Run this command from the cloned `VIM/` repository root:

```bash
hf download viamodelanddataset/vim27b_lora_v2 \
  --revision eab0ae83c54fce6294b91a40bd9ebe30fa60275a \
  --local-dir checkpoints/vim27b_lora_v2
(cd checkpoints/vim27b_lora_v2 && sha256sum --check SHA256SUMS)
```

The base model and adapter must be in separate directories:

```text
checkpoints/
  qwen3.5-27b/            Base model downloaded above
  vim27b_lora_v2/         Released adapter and processor
    adapter_config.json
    adapter_model.safetensors
    processor_config.json
    tokenizer.json
    tokenizer_config.json
    chat_template.jinja
    checkpoint.json
    SHA256SUMS
```

Use `--base-model checkpoints/qwen3.5-27b` with
`--model checkpoints/vim27b_lora_v2`, as in the commands below. To use this
checkpoint, skip training and proceed to [prediction and evaluation](#predict-and-evaluate)
or [prediction from images](#predict-from-images).

The adapter SHA-256 is
`4759b2b337c8003114e8d955776118fd5d9dc9ebc82a36236690335c8f0d8e0f`.
See [checkpoint metadata](https://huggingface.co/viamodelanddataset/vim27b_lora_v2/blob/eab0ae83c54fce6294b91a40bd9ebe30fa60275a/checkpoint.json)
for the training revision and evaluation provenance.

## Train your own adapter (optional)

```bash
vim-train --config configs/vim27b.yaml \
  --model checkpoints/qwen3.5-27b \
  --data data/vi_amo/prepared --output outputs/vim27b
```

The default configuration trains for one epoch with six sampling plans per
action, LoRA rank/alpha 16, learning rate `1e-4`, batch size 4, and four gradient
accumulation steps. Validation and checkpoint saving run every 1,000 steps.
Edit `configs/vim27b.yaml` to change these settings.

The final adapter and processor are saved in `outputs/vim27b/final_adapter/`;
`config.json` and `train_metrics.json` record the configuration and training results.

## Predict and evaluate

Generate test predictions with the released VI AMO v2 adapter:

```bash
vim-predict --config configs/vim27b.yaml \
  --base-model checkpoints/qwen3.5-27b \
  --model checkpoints/vim27b_lora_v2 \
  --data data/vi_amo/prepared/test.jsonl \
  --output outputs/vim27b/predictions.jsonl
```

To use an adapter you trained yourself, replace `--model checkpoints/vim27b_lora_v2`
with `--model outputs/vim27b/final_adapter` in the prediction and image-inference
commands.

Each action is evaluated at 0, 5, 15, 25, 40, 55, 70, 85, and 95% of its duration,
using only observations available by that point and greedy decoding. Prediction
records include reference labels, context, selected frames, and model output.
Use `--cutoffs 0 50 100` to select other integer percentages between 0 and 100.
For the single-frame ablation, omit `--cutoffs` or use `--cutoffs 0` to evaluate
at action start. Use the default nine cutoffs to reproduce the main paper results.

Download the [MPNet](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
and [SigLIP](https://huggingface.co/google/siglip-so400m-patch14-384) metric models, then score:

```bash
hf download sentence-transformers/all-mpnet-base-v2 \
  --include "*.json" --include "*.safetensors" --include "vocab.txt" --include "tokenizer.*" \
  --local-dir checkpoints/all-mpnet-base-v2
hf download google/siglip-so400m-patch14-384 \
  --include "*.json" --include "*.safetensors" --include "*.model" \
  --local-dir checkpoints/siglip-so400m-patch14-384

vim-evaluate --config configs/vim27b.yaml \
  --predictions outputs/vim27b/predictions.jsonl \
  --train data/vi_amo/prepared/train.jsonl \
  --sbert checkpoints/all-mpnet-base-v2 \
  --siglip checkpoints/siglip-so400m-patch14-384 \
  --device cuda --output outputs/vim27b/evaluation
```

Evaluation writes `scores.jsonl` and `metrics.json`, including token F1, SBERT
similarity, SigLIP similarity, and label-bank retrieval accuracy. It reports
overall, per-cutoff, per-dataset, and seen/unseen results using the training labels.
Scores are fractions; multiply Top-10 by 100 for a percentage. Use `--device cpu`
for CPU scoring.

## Predict from images

Supply four RGB images per camera, oldest to newest, with front and wrist images
aligned at each index. `--history` lists completed actions in the current mission.
The included [example](examples/lemon/example.json) shows a robot placing a lemon
on the table after picking it up from a bowl:

```bash
vim-infer --base-model checkpoints/qwen3.5-27b \
  --model checkpoints/vim27b_lora_v2 \
  --front examples/lemon/front_0.png examples/lemon/front_1.png examples/lemon/front_2.png examples/lemon/front_3.png \
  --wrist examples/lemon/wrist_0.png examples/lemon/wrist_1.png examples/lemon/wrist_2.png examples/lemon/wrist_3.png \
  --history "pick up the lemon from the bowl"
```

An example of VIM's output:

```text
Ongoing action: place the lemon on the table
Ongoing mission: put the lemon from the bowl on the table
```

Omit `--wrist` for one camera and `--history` when no action has finished.
For ablation adapters, pass the matching `--config`; the image count and use of
wrist images and history follow that configuration. See
[image inference with ablations](docs/ablations.md#image-inference) for the input requirements.

## Other configurations

| Configuration | Experiment |
| --- | --- |
| `configs/vim27b.yaml` | Qwen3.5-27B |
| `configs/front_only.yaml` | Front camera; retain the paired prompt for paired episodes |
| `configs/no_context.yaml` | Remove history text; observe the active mission prefix |
| `configs/single_frame.yaml` | One image per view at action start |

Each ablation trains its own adapter; use the same configuration for training,
prediction, and scoring.
See the [Front-only example](docs/ablations.md) for a complete command sequence.
For a base-Qwen baseline, pass the base-model directory as `vim-predict --model`
and omit `--base-model`.

## License

The code in this repository is released under the [MIT License](LICENSE).
Datasets, model weights, and third-party dependencies remain subject to their
respective licenses.
