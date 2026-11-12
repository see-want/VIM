"""Small helpers for experiment files."""

import json
from pathlib import Path

import yaml


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    parent = config.pop("extends", None)
    if parent:
        base = load_config(path.parent / parent)
        for key, value in config.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                base[key].update(value)
            else:
                base[key] = value
        return base
    return config
