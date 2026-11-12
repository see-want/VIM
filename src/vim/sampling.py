"""Deterministic action-local training plans and evaluation cutoffs."""

import itertools

import numpy as np

PLANS = (
    "uniform",
    "prefix_random:0.3",
    "prefix_random:0.6",
    "prefix_random:0.8",
    "local",
    "irregular",
)
CUTOFFS = (0, 5, 15, 25, 40, 55, 70, 85, 95)


def uniform(start, end, count=4):
    return np.rint(np.linspace(start, end, count)).astype(int).tolist()


def monotonic(values, length, count=4):
    indices = np.sort(np.clip(np.rint(values).astype(int), 0, length - 1))
    if length >= count:
        for i in range(count):
            indices[i] = np.clip(indices[i], indices[i - 1] + 1 if i else 0, length - count + i)
    return indices.tolist()


def sample_plan(length, plan, rng, count=4):
    last = length - 1
    if plan == "uniform":
        return uniform(0, last, count)
    if plan.startswith("prefix_random:"):
        stop = int(np.clip(np.ceil(length * float(plan.split(":")[1])), 1, length))
        edges = np.linspace(0, stop, count + 1).astype(int)
        return [
            int(rng.integers(a, b)) if b > a else min(a, stop - 1)
            for a, b in itertools.pairwise(edges)
        ]
    if plan == "local":
        span = max(count - 1, round(last * rng.uniform(0.25, 0.55)))
        start_max = max(0, last - span)
        start = int(rng.integers(0, start_max + 1)) if start_max else 0
        end = min(last, start + span)
        first = start + rng.uniform(0.0, 0.10) * span
        final = start + rng.uniform(0.75, 1.0) * span
        middle = first + np.sort(rng.beta(1.2, 2.4, size=count - 2)) * max(final - first, 1.0)
        return monotonic([first, *middle, min(end, final)], length, count)
    if plan == "irregular":
        span = max(count - 1, round(0.40 * last))
        first = int(rng.integers(0, max(0, round(0.40 * last)) + 1))
        low = max(first + span, round(0.60 * last))
        if low > last:
            first, low = max(0, last - span), min(last, first + span)
        final = int(rng.integers(low, last + 1)) if low < last else last
        middle = np.sort(rng.uniform(first, final, size=count - 2))
        return monotonic([first, *middle, final], length, count)
    raise ValueError(f"Unknown sampling plan: {plan}")


def training_frames(actions, seed=3407, plans=PLANS, mission_window=False):
    action_seeds = np.random.SeedSequence(seed).spawn(len(actions))
    history_seeds = iter(
        np.random.SeedSequence([seed, 0x4D495353]).spawn(len(actions) * len(plans))
    )
    result = []
    for action, action_seed in zip(actions, action_seeds):
        start = action["start"]
        end = action["end"]
        frames = [
            [start + i for i in sample_plan(end - start + 1, plan, np.random.default_rng(child))]
            for plan, child in zip(plans, action_seed.spawn(len(plans)))
        ]
        for index, plan in enumerate(plans):
            history_seed = next(history_seeds)
            if mission_window and action["mission_start"] < start:
                cutoff = frames[index][-1]
                history = sample_plan(
                    cutoff - action["mission_start"],
                    plan,
                    np.random.default_rng(history_seed),
                    count=3,
                )
                frames[index] = [action["mission_start"] + i for i in history] + [cutoff]
        result.append(frames)
    return result


def prefix_frames(action, cutoff, mission_window=False):
    if not 0 <= cutoff <= 100:
        raise ValueError("cutoff must be between 0 and 100")
    end = action["start"] + int(np.rint((action["end"] - action["start"]) * (cutoff / 100)))
    start = action["mission_start"] if mission_window else action["start"]
    return uniform(start, end)
