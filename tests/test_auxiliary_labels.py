from __future__ import annotations

import numpy as np

from game.board import BLACK, WHITE, Board
from game.encoder import action_to_index


def _set_stones(board: Board, coords: list[tuple[int, int]], color: int) -> None:
    for x, y in coords:
        board.grid[x][y] = color


def test_threat_labels_mark_own_and_opponent_win_points_without_mutation():
    from train.auxiliary_labels import THREAT_CHANNELS, generate_threat_label

    board = Board()
    _set_stones(board, [(3, 7), (4, 7), (5, 7), (6, 7)], BLACK)
    _set_stones(board, [(8, 3), (8, 4), (8, 5), (8, 6)], WHITE)
    before = [col[:] for col in board.grid]

    labels = generate_threat_label(board, BLACK, rule_mode="basic")

    assert labels.shape == (len(THREAT_CHANNELS), 15, 15)
    assert labels.dtype == np.float32
    assert labels[THREAT_CHANNELS.index("own_win_point"), 7, 2] == 1.0
    assert labels[THREAT_CHANNELS.index("own_win_point"), 7, 7] == 1.0
    assert labels[THREAT_CHANNELS.index("opponent_win_point"), 2, 8] == 1.0
    assert labels[THREAT_CHANNELS.index("opponent_win_point"), 7, 8] == 1.0
    assert board.grid == before


def test_forbidden_label_marks_black_forbidden_and_white_is_zero():
    from train.auxiliary_labels import generate_forbidden_label

    board = Board()
    _set_stones(board, [(6, 7), (8, 7)], BLACK)
    _set_stones(board, [(7, 6), (7, 8)], BLACK)

    black_label = generate_forbidden_label(board, BLACK, rule_mode="forbidden")
    white_label = generate_forbidden_label(board, WHITE, rule_mode="forbidden")

    assert black_label.shape == (1, 15, 15)
    assert black_label.dtype == np.float32
    assert black_label[0, 7, 7] == 1.0
    assert white_label.sum() == 0.0


def test_tactical_score_label_shape_and_occupied_points():
    from train.auxiliary_labels import generate_tactical_score_label

    board = Board()
    board.grid[7][7] = BLACK
    scores = generate_tactical_score_label(board, WHITE)

    assert scores.shape == (225,)
    assert scores.dtype == np.float32
    assert scores[action_to_index(7, 7)] == 0.0


def test_auxiliary_labels_use_candidate_actions_and_shared_threat_cache(monkeypatch):
    import train.auxiliary_labels as aux

    board = Board()
    board.grid[7][7] = BLACK
    board.grid[8][7] = WHITE

    calls: list[tuple[int, int, str]] = []

    def fake_classify(board_arg, action, color, rule_mode="basic", cache=None):
        key = (int(action), int(color), str(rule_mode))
        calls.append(key)
        if cache is not None and key in cache:
            return cache[key]
        result = set()
        if cache is not None:
            cache[key] = result
        return result

    monkeypatch.setattr(aux, "classify_move_threats", fake_classify)

    labels = aux.build_auxiliary_labels(
        board,
        BLACK,
        rule_mode="basic",
        candidate_radius=1,
    )

    unique_actions = {action for action, _, _ in calls}
    expected_candidates = set(aux._candidate_actions(board, candidate_radius=1))
    assert unique_actions == expected_candidates
    assert len(unique_actions) < len(board.get_legal_moves())
    # own/opponent threats are requested by threat labels and reused by tactical scores.
    assert len(calls) == len(set(calls))
    assert labels["threat_labels"].shape == (12, 15, 15)
    assert labels["tactical_scores"].shape == (225,)


def test_basic_forbidden_label_does_not_scan_forbidden_actions(monkeypatch):
    import train.auxiliary_labels as aux

    board = Board()
    board.grid[7][7] = BLACK

    def forbidden_should_not_run(*args, **kwargs):
        raise AssertionError("basic mode must not scan forbidden actions")

    monkeypatch.setattr(aux, "is_forbidden_action", forbidden_should_not_run)

    labels = aux.generate_forbidden_label(board, BLACK, rule_mode="basic")

    assert labels.shape == (1, 15, 15)
    assert labels.sum() == 0.0
