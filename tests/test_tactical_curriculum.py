from __future__ import annotations

import numpy as np

from game.board import BLACK, Board
from game.encoder import index_to_action
from train.auxiliary_labels import THREAT_CHANNELS


def test_tactical_curriculum_generates_balanced_tactical_labels():
    from train.tactical_curriculum import generate_tactical_curriculum_dataset

    arrays, stats = generate_tactical_curriculum_dataset(
        rule_mode="forbidden",
        repeats=1,
        smoothing=0.0,
    )

    assert arrays["states"].shape[1:] == (4, 15, 15)
    assert arrays["policies"].shape[1:] == (225,)
    assert arrays["values"].shape[1:] == (1,)
    assert arrays["threat_labels"].shape[1:] == (12, 15, 15)
    assert arrays["forbidden_labels"].shape[1:] == (1, 15, 15)
    assert arrays["tactical_scores"].shape[1:] == (225,)
    assert np.allclose(arrays["policies"].sum(axis=1), 1.0)
    assert np.count_nonzero(arrays["values"]) > 0

    channel_counts = stats["threat_positive_counts"]
    for name in (
        "own_win_point",
        "opponent_win_point",
        "own_open_four",
        "opponent_open_four",
        "own_blocked_four",
        "opponent_blocked_four",
        "own_open_three",
        "opponent_open_three",
        "own_double_four",
        "own_double_three",
    ):
        assert channel_counts[name] > 0, name
    assert stats["forbidden_positive_count"] > 0


def test_curriculum_forbidden_policy_targets_do_not_select_forbidden_points():
    from train.tactical_curriculum import generate_tactical_curriculum_samples

    samples = generate_tactical_curriculum_samples(rule_mode="forbidden", repeats=1)
    forbidden_samples = [sample for sample in samples if sample.category.startswith("forbidden_")]
    assert forbidden_samples
    forbidden_index = THREAT_CHANNELS.index("own_double_three")
    assert forbidden_index >= 0

    for sample in forbidden_samples:
        action = int(sample.action)
        x, y = index_to_action(action)
        board = Board()
        for sx, sy, color in sample.stones:
            board.grid[sx][sy] = color
        board.move_count = len(sample.stones)
        board.current_player = sample.current_player
        assert board.is_legal_move(x, y)
        fx, fy = index_to_action(sample.forbidden_action)
        assert sample.forbidden_labels[0, fy, fx] == 1.0
        assert sample.policy[sample.forbidden_action] == 0.0
        assert sample.current_player == BLACK
