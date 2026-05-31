"""Tests for engineering double-three forbidden move rules."""

from __future__ import annotations

from game.board import BLACK, EMPTY, WHITE, Board
from game.rules_basic import check_winner as check_winner_basic
from game.rules_forbidden import (
    check_forbidden_double_three,
    count_open_three_directions,
    find_open_three_threats,
    get_game_result_forbidden,
    has_exact_five,
    has_overline,
    is_double_four,
    is_double_three,
    is_open_four_after_move,
)


def _set_stones(board: Board, coords: list[tuple[int, int]], color: int) -> None:
    for x, y in coords:
        board.grid[x][y] = color
    board.move_count = sum(
        1
        for x in range(board.BOARD_SIZE)
        for y in range(board.BOARD_SIZE)
        if board.grid[x][y] != EMPTY
    )


def _set_last_move(board: Board, x: int, y: int, color: int) -> tuple[int, int, int]:
    board.last_move = (x, y, color)
    return board.last_move


def test_black_cross_double_open_three_is_forbidden():
    board = Board()
    stones = [(6, 7), (7, 7), (8, 7), (7, 6), (7, 8)]
    _set_stones(board, stones, BLACK)
    last_move = _set_last_move(board, 7, 7, BLACK)

    assert is_open_four_after_move(board, 5, 7, (1, 0), BLACK) is True
    assert count_open_three_directions(board, 7, 7, BLACK) >= 2
    assert is_double_three(board, 7, 7, BLACK) is True
    assert check_forbidden_double_three(board, last_move) is True

    result = get_game_result_forbidden(board, last_move)
    assert result.is_over is True
    assert result.winner == WHITE
    assert result.reason == "black_double_three_forbidden"
    assert result.forbidden is True


def test_black_single_open_three_is_not_forbidden():
    board = Board()
    stones = [(6, 7), (7, 7), (8, 7)]
    _set_stones(board, stones, BLACK)
    last_move = _set_last_move(board, 7, 7, BLACK)

    assert find_open_three_threats(board, 7, 7, BLACK)
    assert count_open_three_directions(board, 7, 7, BLACK) == 1
    assert is_double_three(board, 7, 7, BLACK) is False

    result = get_game_result_forbidden(board, last_move)
    assert result.is_over is False
    assert result.winner is None
    assert result.reason == "ongoing"


def test_black_broken_open_three_is_recognized():
    board = Board()
    stones = [(5, 7), (7, 7), (8, 7)]  # F8, H8, I8 with G8 empty.
    _set_stones(board, stones, BLACK)
    _set_last_move(board, 7, 7, BLACK)

    threats = find_open_three_threats(board, 7, 7, BLACK)
    horizontal_threats = [threat for threat in threats if threat.direction == (1, 0)]

    assert horizontal_threats
    assert any(threat.threat_type == "broken_open_three" for threat in horizontal_threats)
    assert count_open_three_directions(board, 7, 7, BLACK) == 1
    assert is_double_three(board, 7, 7, BLACK) is False


def test_same_direction_multiple_open_three_threats_count_once():
    board = Board()
    stones = [(6, 7), (7, 7), (8, 7)]  # F8 and J8 can both extend.
    _set_stones(board, stones, BLACK)
    _set_last_move(board, 7, 7, BLACK)

    threats = find_open_three_threats(board, 7, 7, BLACK)

    assert len([threat for threat in threats if threat.direction == (1, 0)]) >= 2
    assert count_open_three_directions(board, 7, 7, BLACK) == 1
    assert is_double_three(board, 7, 7, BLACK) is False


def test_black_exact_five_has_priority_over_double_three():
    board = Board()
    exact_five = [(5, 7), (6, 7), (7, 7), (8, 7), (9, 7)]
    vertical_three = [(7, 5), (7, 6), (7, 7)]
    diagonal_three = [(5, 5), (6, 6), (7, 7)]
    _set_stones(board, list(set(exact_five + vertical_three + diagonal_three)), BLACK)
    last_move = _set_last_move(board, 7, 7, BLACK)

    assert has_exact_five(board, 7, 7, BLACK) is True
    assert is_double_three(board, 7, 7, BLACK) is True

    result = get_game_result_forbidden(board, last_move)
    assert result.is_over is True
    assert result.winner == BLACK
    assert result.reason == "black_exact_five"
    assert result.forbidden is False


def test_black_double_four_has_priority_over_double_three_without_exact_five():
    board = Board()
    double_four = [(6, 7), (7, 7), (8, 7), (9, 7), (7, 6), (7, 8), (7, 9)]
    diagonal_three = [(5, 5), (6, 6), (7, 7)]
    anti_diagonal_three = [(5, 9), (6, 8), (7, 7)]
    _set_stones(
        board, list(set(double_four + diagonal_three + anti_diagonal_three)), BLACK
    )
    last_move = _set_last_move(board, 7, 7, BLACK)

    assert has_exact_five(board, 7, 7, BLACK) is False
    assert is_double_four(board, 7, 7, BLACK) is True
    assert is_double_three(board, 7, 7, BLACK) is True

    result = get_game_result_forbidden(board, last_move)
    assert result.is_over is True
    assert result.winner == WHITE
    assert result.reason == "black_double_four_forbidden"
    assert result.forbidden is True


def test_black_overline_has_priority_over_double_three_without_exact_five():
    board = Board()
    overline = [(5, 7), (6, 7), (7, 7), (8, 7), (9, 7), (10, 7)]
    vertical_three = [(7, 5), (7, 6), (7, 7)]
    diagonal_three = [(5, 5), (6, 6), (7, 7)]
    _set_stones(board, list(set(overline + vertical_three + diagonal_three)), BLACK)
    last_move = _set_last_move(board, 7, 7, BLACK)

    assert has_exact_five(board, 7, 7, BLACK) is False
    assert has_overline(board, 7, 7, BLACK) is True
    assert is_double_three(board, 7, 7, BLACK) is True

    result = get_game_result_forbidden(board, last_move)
    assert result.is_over is True
    assert result.winner == WHITE
    assert result.reason == "black_overline_forbidden"
    assert result.forbidden is True


def test_white_double_open_three_is_not_forbidden():
    board = Board()
    stones = [(6, 7), (7, 7), (8, 7), (7, 6), (7, 8)]
    _set_stones(board, stones, WHITE)
    last_move = _set_last_move(board, 7, 7, WHITE)

    assert is_double_three(board, 7, 7, WHITE) is True
    assert check_forbidden_double_three(board, last_move) is False

    result = get_game_result_forbidden(board, last_move)
    assert result.is_over is False
    assert result.winner is None
    assert result.reason == "ongoing"


def test_closed_three_is_not_open_three():
    board = Board()
    _set_stones(board, [(6, 7), (7, 7), (8, 7)], BLACK)
    _set_stones(board, [(5, 7)], WHITE)
    last_move = _set_last_move(board, 7, 7, BLACK)

    assert count_open_three_directions(board, 7, 7, BLACK) == 0
    assert is_double_three(board, 7, 7, BLACK) is False

    result = get_game_result_forbidden(board, last_move)
    assert result.is_over is False
    assert result.reason == "ongoing"


def test_basic_mode_does_not_treat_double_three_as_forbidden():
    board = Board()
    stones = [(6, 7), (7, 7), (8, 7), (7, 6), (7, 8)]
    _set_stones(board, stones, BLACK)
    last_move = _set_last_move(board, 7, 7, BLACK)

    assert check_winner_basic(board, last_move) == 0
    assert get_game_result_forbidden(board, last_move).winner == WHITE


def test_basic_mode_still_treats_black_overline_as_black_win():
    board = Board()
    overline = [(5, 7), (6, 7), (7, 7), (8, 7), (9, 7), (10, 7)]
    _set_stones(board, overline, BLACK)
    last_move = _set_last_move(board, 10, 7, BLACK)

    assert check_winner_basic(board, last_move) == BLACK
