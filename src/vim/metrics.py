"""Text overlap and label-retrieval metrics."""

import re
from collections import Counter

import numpy as np


def normalize(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def parse_prediction(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    result = {}
    for field in ("action", "mission"):
        match = re.search(
            rf"^[ \t]*Ongoing {field}[ \t]*:[ \t]*([^\r\n]*)$",
            text,
            flags=re.MULTILINE | re.IGNORECASE,
        )
        result[field] = match.group(1).strip() if match else ""
    return result


def token_f1(prediction, target):
    predicted, expected = normalize(prediction).split(), normalize(target).split()
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    return 2 * overlap / (len(predicted) + len(expected)) if overlap else 0.0


def label_bank(texts):
    aliases = {}
    for text in texts:
        aliases.setdefault(normalize(text), set()).add(text.strip())
    keys = sorted(aliases)
    return [min(aliases[key]) for key in keys], {key: i for i, key in enumerate(keys)}


def retrieval(predictions, targets, embeddings):
    labels, lookup = label_bank(targets)
    bank = np.stack([embeddings[text] for text in labels])
    results = []
    for prediction, target in zip(predictions, targets):
        valid = bool(normalize(prediction))
        scores = np.clip(bank @ embeddings[prediction], -1, 1)
        target_score = scores[lookup[normalize(target)]]
        rank = int(np.count_nonzero(scores >= target_score - 1e-7)) if valid else len(bank) + 1
        results.append(
            {
                "rank": rank,
                "top1": float(valid and rank <= 1),
                "top5": float(valid and rank <= min(5, len(bank))),
                "top10": float(valid and rank <= min(10, len(bank))),
                "mrr": 1 / rank if valid else 0.0,
            }
        )
    return results
