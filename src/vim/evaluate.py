"""Score OA and OM predictions with the paper's four primary metrics."""

import argparse
from pathlib import Path

import numpy as np

from .encoders import encode
from .io import load_config, read_jsonl, write_json, write_jsonl
from .metrics import normalize, parse_prediction, retrieval, token_f1

METRICS = ("f1", "sbert", "siglip", "top1", "top5", "top10", "mrr")


def aggregate(rows):
    return {
        "count": len(rows),
        **{
            field: {
                metric: float(np.mean([row[field][metric] for row in rows])) for metric in METRICS
            }
            for field in ("action", "mission")
        },
    }


def evaluate(rows, config, device="cuda", train_rows=None):
    parsed = [parse_prediction(row["prediction"]) for row in rows]
    texts = [
        text
        for row, prediction in zip(rows, parsed)
        for field in ("action", "mission")
        for text in (row[field], prediction[field])
    ]
    sbert = encode(texts, config["metrics"]["sbert"], "sbert", device)
    siglip = encode(texts, config["metrics"]["siglip"], "siglip", device)
    scored = [
        {"sample_id": row["sample_id"], "cutoff": row["cutoff"], "dataset": row["dataset"]}
        for row in rows
    ]
    for field in ("action", "mission"):
        predictions = [p[field] for p in parsed]
        targets = [r[field] for r in rows]
        ranks = retrieval(predictions, targets, siglip)
        for i, (prediction, target) in enumerate(zip(predictions, targets)):
            scored[i][field] = {
                "prediction": prediction,
                "target": target,
                "f1": token_f1(prediction, target),
                "sbert": float(np.clip(sbert[prediction] @ sbert[target], -1, 1)),
                "siglip": float(np.clip(siglip[prediction] @ siglip[target], -1, 1)),
                **ranks[i],
            }
    summary = {"overall": aggregate(scored)}
    for key in ("cutoff", "dataset"):
        values = sorted({row[key] for row in scored}, key=str)
        summary[f"by_{key}"] = {
            str(value): aggregate([r for r in scored if r[key] == value]) for value in values
        }
    if train_rows:
        summary["novelty"] = {}
        seen_labels = {
            field: {normalize(a[field]) for r in train_rows for a in r["actions"]}
            for field in ("action", "mission")
        }
        for field in ("action", "mission"):
            seen = seen_labels[field]
            for group, present in (("seen", True), ("unseen", False)):
                subset = [r for r in scored if (normalize(r[field]["target"]) in seen) == present]
                summary["novelty"][f"{field}_{group}"] = (
                    aggregate(subset) if subset else {"count": 0}
                )
        for group, present in (("seen/seen", True), ("unseen/unseen", False)):
            subset = [
                r
                for r in scored
                if all(
                    (normalize(r[field]["target"]) in seen_labels[field]) == present
                    for field in ("action", "mission")
                )
            ]
            summary["novelty"][group] = aggregate(subset) if subset else {"count": 0}
    return scored, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vim27b.yaml")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--train", type=Path, help="Training manifest for seen/unseen groups")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sbert", help="Override MPNet model ID or local path")
    parser.add_argument("--siglip", help="Override SigLIP model ID or local path")
    args = parser.parse_args()
    config = load_config(args.config)
    for name in ("sbert", "siglip"):
        if getattr(args, name):
            config["metrics"][name] = getattr(args, name)
    rows = read_jsonl(args.predictions)
    scored, summary = evaluate(
        rows, config, args.device, read_jsonl(args.train) if args.train else None
    )
    write_jsonl(args.output / "scores.jsonl", scored)
    write_json(args.output / "metrics.json", summary)
    print(summary["overall"])


if __name__ == "__main__":
    main()
