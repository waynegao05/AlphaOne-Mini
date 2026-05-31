"""黑方长连禁手规则测试。"""

from __future__ import annotations

from game.board import BLACK, WHITE, Board
from game.rules_basic import check_winner as check_winner_basic
from game.rules_forbidden import (
    check_forbidden_overline,
    check_winner_forbidden,
    get_game_result_forbidden,
    has_exact_five,
    has_overline,
    is_game_over_forbidden,
)


def _set_stones(board: Board, coords: list[tuple[int, int]], color: int) -> None:
    for x, y in coords:
        board.grid[x][y] = color
    board.move_count = sum(
        1 for x in range(board.BOARD_SIZE) for y in range(board.BOARD_SIZE) if board.grid[x][y] != 0
    )


def _last(board: Board, x: int, y: int, color: int) -> tuple[int, int, int]:
    board.last_move = (x, y, color)
    return board.last_move


def test_black_exact_five_wins_in_forbidden_mode():
    board = Board()
    stones = [(7, 7), (8, 7), (9, 7), (10, 7), (11, 7)]  # H8-L8
    _set_stones(board, stones, BLACK)
    last_move = _last(board, 11, 7, BLACK)

    assert has_exact_five(board, 11, 7, BLACK) is True
    assert has_overline(board, 11, 7, BLACK) is False
    assert check_winner_forbidden(board, last_move) == BLACK

    result = get_game_result_forbidden(board, last_move)
    assert result.is_over is True
    assert result.winner == BLACK
    assert result.reason == "black_exact_five"
    assert result.forbidden is False


def test_black_pure_overline_is_forbidden_and_white_wins():
    board = Board()
    stones = [(6, 7), (7, 7), (8, 7), (9, 7), (10, 7), (11, 7)]  # G8-L8
    _set_stones(board, stones, BLACK)
    last_move = _last(board, 11, 7, BLACK)

    assert has_exact_five(board, 11, 7, BLACK) is False
    assert has_overline(board, 11, 7, BLACK) is True
    assert check_forbidden_overline(board, last_move) is True
    assert check_winner_forbidden(board, last_move) == WHITE

    result = get_game_result_forbidden(board, last_move)
    assert result.is_over is True
    assert result.winner == WHITE
    assert result.reason == "black_overline_forbidden"
    assert result.forbidden is True


def test_black_exact_five_has_priority_over_simultaneous_overline():
    board = Board()
    horizontal_six = [(5, 7), (6, 7), (7, 7), (8, 7), (9, 7), (10, 7)]
    vertical_five = [(7, 5), (7, 6), (7, 7), (7, 8), (7, 9)]
    _set_stones(board, list(set(horizontal_six + vertical_five)), BLACK)
    last_move = _last(board, 7, 7, BLACK)

    assert has_overline(board, 7, 7, BLACK) is True
    assert has_exact_five(board, 7, 7, BLACK) is True
    assert check_forbidden_overline(board, last_move) is False

    result = get_game_result_forbidden(board, last_move)
    assert result.is_over is True
    assert result.winner == BLACK
    assert result.reason == "black_exact_five"
    assert result.forbidden is False


def test_white_exact_five_wins():
    board = Board()
    stones = [(2, 3), (3, 3), (4, 3), (5, 3), (6, 3)]
    _set_stones(board, stones, WHITE)
    last_move = _last(board, 6, 3, WHITE)

    result = get_game_result_forbidden(board, last_move)
    assert result.is_over is True
    assert result.winner == WHITE
    assert result.reason == "white_five_or_more"
    assert result.forbidden is False


def test_white_overline_still_wins_without_forbidden_penalty():
    board = Board()
    stones = [(2, 3), (3, 3), (4, 3), (5, 3), (6, 3), (7, 3)]
    _set_stones(board, stones, WHITE)
    last_move = _last(board, 7, 3, WHITE)

    assert has_overline(board, 7, 3, WHITE) is True
    result = get_game_result_forbidden(board, last_move)
    assert result.is_over is True
    assert result.winner == WHITE
    assert result.reason == "white_five_or_more"
    assert result.forbidden is False


def test_ongoing_position_is_not_over():
    board = Board()
    _set_stones(board, [(7, 7), (8, 7), (9, 7)], BLACK)
    last_move = _last(board, 9, 7, BLACK)

    result = get_game_result_forbidden(board, last_move)
    assert result.is_over is False
    assert result.winner is None
    assert result.reason == "ongoing"
    assert is_game_over_forbidden(board, last_move) is False


def test_basic_mode_still_treats_black_overline_as_black_win():
    board = Board()
    stones = [(6, 7), (7, 7), (8, 7), (9, 7), (10, 7), (11, 7)]
    _set_stones(board, stones, BLACK)
    last_move = _last(board, 11, 7, BLACK)

    assert check_winner_basic(board, last_move) == BLACK
    assert check_winner_forbidden(board, last_move) == WHITE


def test_full_board_draw_check_scans_for_existing_winner():
    board = Board()
    _set_stones(board, [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)], BLACK)
    board.grid[14][14] = WHITE
    board.move_count = board.BOARD_SIZE * board.BOARD_SIZE
    last_move = _last(board, 14, 14, WHITE)

    result = get_game_result_forbidden(board, last_move)
    assert result.is_over is True
    assert result.winner == BLACK
    assert result.reason == "black_exact_five"
