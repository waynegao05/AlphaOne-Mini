"""黑方四四禁手规则测试。"""

from __future__ import annotations

from game.board import BLACK, EMPTY, WHITE, Board
from game.rules_basic import check_winner as check_winner_basic
from game.rules_forbidden import (
    check_forbidden_double_four,
    check_winner_forbidden,
    count_four_threat_directions,
    find_four_threats,
    get_game_result_forbidden,
    has_exact_five,
    has_overline,
    is_double_four,
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


def test_black_cross_double_four_is_forbidden():
    board = Board()
    # Last move H8. Horizontal: G8,H8,I8,J8. Vertical: H7,H8,H9,H10.
    stones = [(6, 7), (7, 7), (8, 7), (9, 7), (7, 6), (7, 8), (7, 9)]
    _set_stones(board, stones, BLACK)
    last_move = _set_last_move(board, 7, 7, BLACK)

    assert is_double_four(board, 7, 7, BLACK) is True
    assert check_forbidden_double_four(board, last_move) is True
    assert check_winner_forbidden(board, last_move) == WHITE

    result = get_game_result_forbidden(board, last_move)
    assert result.is_over is True
    assert result.winner == WHITE
    assert result.reason == "black_double_four_forbidden"
    assert result.forbidden is True


def test_black_single_four_is_not_forbidden():
    board = Board()
    stones = [(6, 7), (7, 7), (8, 7), (9, 7)]  # G8-J8
    _set_stones(board, stones, BLACK)
    last_move = _set_last_move(board, 7, 7, BLACK)

    assert find_four_threats(board, 7, 7, BLACK)
    assert count_four_threat_directions(board, 7, 7, BLACK) == 1
    assert is_double_four(board, 7, 7, BLACK) is False

    result = get_game_result_forbidden(board, last_move)
    assert result.is_over is False
    assert result.winner is None
    assert result.reason == "ongoing"


def test_open_four_counts_as_one_direction_only():
    board = Board()
    stones = [(6, 7), (7, 7), (8, 7), (9, 7)]  # F8/K8 are both completions.
    _set_stones(board, stones, BLACK)
    last_move = _set_last_move(board, 7, 7, BLACK)

    threats = find_four_threats(board, 7, 7, BLACK)
    assert len(threats) >= 2
    assert {threat.empty_position for threat in threats} == {(5, 7), (10, 7)}
    assert count_four_threat_directions(board, 7, 7, BLACK) == 1
    assert is_double_four(board, 7, 7, BLACK) is False
    assert get_game_result_forbidden(board, last_move).reason == "ongoing"


def test_black_exact_five_has_priority_over_double_four():
    board = Board()
    exact_five = [(5, 7), (6, 7), (7, 7), (8, 7), (9, 7)]  # F8-J8
    vertical_four = [(7, 5), (7, 6), (7, 7), (7, 8)]       # H6-H9
    diagonal_four = [(4, 4), (5, 5), (6, 6), (7, 7)]       # E5-H8
    _set_stones(board, list(set(exact_five + vertical_four + diagonal_four)), BLACK)
    last_move = _set_last_move(board, 7, 7, BLACK)

    assert has_exact_five(board, 7, 7, BLACK) is True
    assert is_double_four(board, 7, 7, BLACK) is True

    result = get_game_result_forbidden(board, last_move)
    assert result.is_over is True
    assert result.winner == BLACK
    assert result.reason == "black_exact_five"
    assert result.forbidden is False


def test_black_overline_has_priority_over_double_four_when_no_exact_five():
    board = Board()
    overline = [(5, 7), (6, 7), (7, 7), (8, 7), (9, 7), (10, 7)]  # F8-K8
    vertical_four = [(7, 5), (7, 6), (7, 7), (7, 8)]
    diagonal_four = [(4, 4), (5, 5), (6, 6), (7, 7)]
    _set_stones(board, list(set(overline + vertical_four + diagonal_four)), BLACK)
    last_move = _set_last_move(board, 7, 7, BLACK)

    assert has_exact_five(board, 7, 7, BLACK) is False
    assert has_overline(board, 7, 7, BLACK) is True
    assert is_double_four(board, 7, 7, BLACK) is True

    result = get_game_result_forbidden(board, last_move)
    assert result.is_over is True
    assert result.winner == WHITE
    assert result.reason == "black_overline_forbidden"
    assert result.forbidden is True


def test_white_double_four_is_not_forbidden():
    board = Board()
    stones = [(6, 7), (7, 7), (8, 7), (9, 7), (7, 6), (7, 8), (7, 9)]
    _set_stones(board, stones, WHITE)
    last_move = _set_last_move(board, 7, 7, WHITE)

    assert is_double_four(board, 7, 7, WHITE) is True
    assert check_forbidden_double_four(board, last_move) is False

    result = get_game_result_forbidden(board, last_move)
    assert result.is_over is False
    assert result.winner is None
    assert result.reason == "ongoing"


def test_white_five_or_overline_still_wins():
    board = Board()
    stones = [(2, 3), (3, 3), (4, 3), (5, 3), (6, 3), (7, 3)]
    _set_stones(board, stones, WHITE)
    last_move = _set_last_move(board, 7, 3, WHITE)

    result = get_game_result_forbidden(board, last_move)
    assert result.is_over is True
    assert result.winner == WHITE
    assert result.reason == "white_five_or_more"


def test_basic_mode_does_not_treat_double_four_as_forbidden():
    board = Board()
    double_four = [(6, 7), (7, 7), (8, 7), (9, 7), (7, 6), (7, 8), (7, 9)]
    _set_stones(board, double_four, BLACK)
    last_move = _set_last_move(board, 7, 7, BLACK)

    assert check_winner_basic(board, last_move) == 0
    assert check_winner_forbidden(board, last_move) == WHITE


def test_basic_mode_still_treats_black_overline_as_black_win():
    board = Board()
    overline = [(5, 7), (6, 7), (7, 7), (8, 7), (9, 7), (10, 7)]
    _set_stones(board, overline, BLACK)
    last_move = _set_last_move(board, 10, 7, BLACK)

    assert check_winner_basic(board, last_move) == BLACK
