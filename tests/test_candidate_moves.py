"""Tests for tactical candidate move generation."""

from __future__ import annotations

from game.board import BLACK, EMPTY, WHITE, Board
from game.encoder import action_to_index, index_to_action


def _set_stones(board: Board, stones: list[tuple[int, int, int]]) -> None:
    for x, y, color in stones:
        board.grid[x][y] = color
    board.move_count = sum(
        1
        for x in range(board.BOARD_SIZE)
        for y in range(board.BOARD_SIZE)
        if board.grid[x][y] != EMPTY
    )


def test_empty_board_returns_center_first():
    from engine.candidate_moves import generate_candidate_moves

    board = Board()
    actions = generate_candidate_moves(board, radius=2)

    assert actions
    assert actions[0] == action_to_index(7, 7)


def test_non_empty_board_returns_only_nearby_empty_points():
    from engine.candidate_moves import generate_candidate_moves

    board = Board()
    _set_stones(board, [(7, 7, BLACK), (10, 10, WHITE)])

    actions = generate_candidate_moves(board, radius=1)
    coords = {index_to_action(action) for action in actions}

    assert (7, 7) not in coords
    assert (10, 10) not in coords
    assert coords
    assert all(
        min(abs(x - 7), abs(x - 10)) <= 1 and min(abs(y - 7), abs(y - 10)) <= 1
        for x, y in coords
    )


def test_candidate_moves_do_not_leave_board_near_edges():
    from engine.candidate_moves import generate_candidate_moves

    board = Board()
    _set_stones(board, [(0, 0, BLACK)])

    actions = generate_candidate_moves(board, radius=2)

    assert action_to_index(0, 0) not in actions
    for action in actions:
        x, y = index_to_action(action)
        assert 0 <= x < board.BOARD_SIZE
        assert 0 <= y < board.BOARD_SIZE


def test_candidate_moves_respect_max_candidates_and_order_by_center():
    from engine.candidate_moves import generate_candidate_moves

    board = Board()
    _set_stones(board, [(7, 7, BLACK)])

    actions = generate_candidate_moves(board, radius=2, max_candidates=3)

    assert len(actions) == 3
    assert len(set(actions)) == 3
    assert all(action != action_to_index(7, 7) for action in actions)
