import numpy as np
import pytest

from vim.media import frame_indices
from vim.metrics import parse_prediction, retrieval, token_f1
from vim.prompts import context_text, messages
from vim.sampling import CUTOFFS, PLANS, prefix_frames, training_frames


@pytest.fixture
def actions():
    return [
        {"start": 0, "end": 39, "mission_start": 0},
        {"start": 40, "end": 79, "mission_start": 0},
    ]


def test_six_plans_are_deterministic_and_action_local(actions):
    first = training_frames(actions)
    assert first == training_frames(actions)
    assert len(first[0]) == len(PLANS) == 6
    for action, plans in zip(actions, first):
        for frames in plans:
            assert len(frames) == 4
            assert frames == sorted(frames)
            assert action["start"] <= frames[0] <= frames[-1] <= action["end"]


def test_mission_history_preserves_current_action_cutoffs(actions):
    atomic = training_frames(actions)
    mission = training_frames(actions, mission_window=True)
    assert atomic[0] == mission[0]
    for before, after in zip(atomic[1], mission[1]):
        assert before[-1] == after[-1]
        assert 0 <= after[0] < after[-1]


def test_causal_cutoffs_and_single_frame_actions(actions):
    for cutoff in CUTOFFS:
        frames = prefix_frames(actions[1], cutoff)
        assert frames[0] == 40
        assert frames[-1] == 40 + round(39 * cutoff / 100)
    short = [{"start": 8, "end": 8, "mission_start": 8}]
    assert all(frames == [8] * 4 for frames in training_frames(short)[0])
    assert prefix_frames({"start": 0, "end": 45}, 70) == [0, 10, 21, 31]


def test_view_mapping_floors_and_clamps():
    view = {"fps": 15, "offset": 7, "length": 12}
    assert frame_indices(view, 30, [0, 1, 4, 20]) == [7, 7, 9, 11]


def test_context_and_prompt_variants():
    context = context_text(["pick up the block"])
    assert context == 'Robot already finished actions: "pick up the block".'
    paired = messages([[None] * 4, [None] * 4], context)
    assert sum(c["type"] == "video" for c in paired[1]["content"]) == 2
    no_context = messages([[None] * 4], "")
    assert "execution context" not in str(no_context)
    front = messages([[None] * 4], context, paired_prompt=True)
    assert "Jointly use both views." in front[0]["content"][0]["text"]


def test_missing_fields_score_zero_and_ties_are_pessimistic():
    prediction = parse_prediction("<think>reasoning</think>\nOngoing action: pick up the block")
    assert prediction == {"action": "pick up the block", "mission": ""}
    empty_action = parse_prediction("Ongoing action: \nOngoing mission: move the block")
    assert empty_action == {"action": "", "mission": "move the block"}
    assert token_f1("", "put the block away") == 0
    embeddings = {"a": np.array([1.0, 0.0]), "b": np.array([1.0, 0.0]), "": np.zeros(2)}
    scores = retrieval(["a", ""], ["a", "b"], embeddings)
    assert scores[0]["rank"] == 2
    assert scores[0]["top1"] == 0
    assert scores[1]["top10"] == 0


def test_token_f1_counts_repeated_words():
    assert token_f1("red red block", "red block") == pytest.approx(0.8)
