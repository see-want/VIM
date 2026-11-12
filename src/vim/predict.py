"""Predict ongoing actions and missions at causal observation cutoffs."""

import argparse
import json
from pathlib import Path

from .io import load_config
from .models import generate, load_model
from .sampling import CUTOFFS


def cutoff_percent(value):
    try:
        cutoff = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer from 0 to 100") from None
    if not 0 <= cutoff <= 100:
        raise argparse.ArgumentTypeError("must be an integer from 0 to 100")
    return cutoff


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vim27b.yaml")
    parser.add_argument("--data", type=Path, required=True, help="Test JSONL manifest")
    parser.add_argument("--model", required=True, help="Base model or saved LoRA adapter")
    parser.add_argument("--base-model", help="Local base model to load with the adapter")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cutoffs", type=cutoff_percent, nargs="+",
        help="Integer percentages from 0 to 100; defaults to the paper's nine cutoffs (single_frame: 0)",
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if config["data"].get("single_frame", False):
        if args.cutoffs is not None and args.cutoffs != [0]:
            parser.error("single_frame evaluates only at action start; omit --cutoffs or use --cutoffs 0")
        cutoffs = (0,)
    else:
        cutoffs = args.cutoffs if args.cutoffs is not None else CUTOFFS
    from .data import IntentDataset
    from .media import close_images

    dataset = IntentDataset(args.data, "test", config["data"], cutoffs)
    model, processor = load_model(args.model, config, base_model=args.base_model)
    count = min(len(dataset), args.limit) if args.limit else len(dataset)
    batch_size = config["generation"]["batch_size"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        for start in range(0, count, batch_size):
            indices = list(range(start, min(start + batch_size, count)))
            examples = [dataset.example(i) for i in indices]
            predictions = generate(
                model, processor, examples, config["generation"]["max_new_tokens"]
            )
            for index, prediction, example in zip(indices, predictions, examples):
                row = dataset.row(index)
                row["prediction"] = prediction
                stream.write(json.dumps(row, ensure_ascii=True) + "\n")
                close_images(example)
            stream.flush()
            print(f"Predicted {min(start + batch_size, count)}/{count}", flush=True)


if __name__ == "__main__":
    main()
