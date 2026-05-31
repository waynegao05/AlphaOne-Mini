"""Tests for Gomoku tactical threat recognition."""

from __future__ import annotations

from game.board import BLACK, EMPTY, WHITE, Board
from game.encoder import action_to_index


def _set_stones(board: Board, stones: list[tuple[int, int, int]]) -> None:
    for x, y, color in stones:
        board.grid[x][y] = color
    board.move_count = sum(
        1
        for x in range(board.BOARD_SIZE)
        for y in range(board.BOARD_SIZE)
        if board.grid[x][y] != EMPTY
    )


def test_find_immediate_winning_moves_for_four_in_row():
    from engine.threats import find_immediate_winning_moves

    board = Board()
    _set_stones(board, [(5, 7, BLACK), (6, 7, BLACK), (7, 7, BLACK), (8, 7, BLACK)])

    wins = find_immediate_winning_moves(board, BLACK)

    assert action_to_index(4, 7) in wins
    assert action_to_index(9, 7) in wins


def test_find_immediate_blocking_moves_blocks_opponent_four():
    from engine.threats import find_immediate_blocking_moves

    board = Board()
    _set_stones(board, [(5, 7, WHITE), (6, 7, WHITE), (7, 7, WHITE), (8, 7, WHITE)])

    blocks = find_immediate_blocking_moves(board, BLACK)

    assert action_to_index(4, 7) in blocks
    assert action_to_index(9, 7) in blocks


def test_open_four_and_blocked_four_are_classified():
    from engine.threats import classify_move_threats

    open_board = Board()
    _set_stones(open_board, [(5, 7, BLACK), (6, 7, BLACK), (7, 7, BLACK)])
    open_threats = classify_move_threats(
        open_board, action_to_index(8, 7), BLACK, rule_mode="basic"
    )
    assert "open_four" in open_threats

    blocked_board = Board()
    _set_stones(
        blocked_board,
        [(5, 7, WHITE), (6, 7, BLACK), (7, 7, BLACK), (8, 7, BLACK)],
    )
    blocked_threats = classify_move_threats(
        blocked_board, action_to_index(9, 7), BLACK, rule_mode="basic"
    )
    assert "blocked_four" in blocked_threats
    assert "open_four" not in blocked_threats


def test_open_three_double_four_and_double_three_are_classified():
    from engine.threats import classify_move_threats

    open_three_board = Board()
    _set_stones(open_three_board, [(6, 7, BLACK), (7, 7, BLACK)])
    threats = classify_move_threats(
        open_three_board, action_to_index(8, 7), BLACK, rule_mode="basic"
    )
    assert "open_three" in threats

    double_four_board = Board()
    _set_stones(
        double_four_board,
        [(6, 7, BLACK), (8, 7, BLACK), (9, 7, BLACK), (7, 6, BLACK), (7, 8, BLACK), (7, 9, BLACK)],
    )
    threats = classify_move_threats(
        double_four_board, action_to_index(7, 7), BLACK, rule_mode="forbidden"
    )
    assert "double_four" in threats
    assert "forbidden" in threats

    double_three_board = Board()
    _set_stones(
        double_three_board,
        [(6, 7, BLACK), (8, 7, BLACK), (7, 6, BLACK), (7, 8, BLACK)],
    )
    threats = classify_move_threats(
        double_three_board, action_to_index(7, 7), BLACK, rule_mode="forbidden"
    )
    assert "open_three" in threats
    assert "double_three" in threats
    assert "forbidden" in threats


def test_forbidden_black_overline_is_not_basic_winning_move():
    from engine.threats import classify_move_threats, find_immediate_winning_moves

    board = Board()
    _set_stones(
        board,
        [(5, 7, BLACK), (6, 7, BLACK), (7, 7, BLACK), (8, 7, BLACK), (9, 7, BLACK)],
    )
    overline_action = action_to_index(10, 7)

    assert overline_action not in find_immediate_winning_moves(
        board, BLACK, rule_mode="forbidden"
    )
    threats = classify_move_threats(board, overline_action, BLACK, rule_mode="forbidden")
    assert "overline_forbidden" in threats
    assert "forbidden" in threats


def test_white_overline_is_treated_as_winning_move():
    from engine.threats import find_immediate_winning_moves

    board = Board()
    _set_stones(
        board,
        [(5, 7, WHITE), (6, 7, WHITE), (7, 7, WHITE), (8, 7, WHITE), (9, 7, WHITE)],
    )

    wins = find_immediate_winning_moves(board, WHITE, rule_mode="forbidden")

    assert action_to_index(10, 7) in wins


def test_fast_classification_skips_expensive_threes_without_poisoning_cache():
    from engine.threats import classify_move_threats

    board = Board()
    _set_stones(board, [(6, 7, BLACK), (7, 7, BLACK)])
    action = action_to_index(8, 7)
    cache = {}

    fast_threats = classify_move_threats(
        board,
        action,
        BLACK,
        rule_mode="basic",
        include_double_threats=False,
        include_open_three=False,
        cache=cache,
    )
    full_threats = classify_move_threats(
        board,
        action,
        BLACK,
        rule_mode="basic",
        cache=cache,
    )

    assert "open_three" not in fast_threats
    assert "open_three" in full_threats
