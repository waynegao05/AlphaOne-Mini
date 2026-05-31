"""Tests for VCT-lite open-three forcing search."""

from __future__ import annotations

from engine.vct_search import (
    find_vct_attack_candidates,
    find_vct_defense_moves,
    vct_first_move,
)
from game.board import BLACK, BOARD_SIZE, WHITE, Board
from game.encoder import action_to_index


def _set_stones(board: Board, black=(), white=(), current_player=BLACK) -> None:
    for x, y in black:
        board.grid[x][y] = BLACK
    for x, y in white:
        board.grid[x][y] = WHITE
    board.move_count = len(tuple(black)) + len(tuple(white))
    board.current_player = current_player


def _snapshot(board: Board):
    return {
        "grid": [row[:] for row in board.grid],
        "move_count": board.move_count,
        "current_player": board.current_player,
        "last_move": board.last_move,
    }


def test_find_vct_attack_candidates_finds_open_three_creator():
    board = Board()
    _set_stones(board, black=[(5, 7), (6, 7)])

    expected = action_to_index(7, 7, BOARD_SIZE)

    candidates = find_vct_attack_candidates(board, BLACK, rule_mode="basic")

    assert expected in candidates


def test_vct_first_move_prefers_double_open_three_center():
    board = Board()
    _set_stones(
        board,
        black=[(6, 7), (8, 7), (7, 6), (7, 8)],
        current_player=BLACK,
    )
    center = action_to_index(7, 7, BOARD_SIZE)

    assert vct_first_move(board, BLACK, max_depth=5, rule_mode="basic") == center


def test_single_open_three_is_not_claimed_as_forced_vct():
    board = Board()
    _set_stones(board, black=[(5, 7), (6, 7)], current_player=BLACK)

    assert vct_first_move(board, BLACK, max_depth=7, rule_mode="basic") is None


def test_find_vct_defense_moves_lists_all_open_three_extension_cells():
    board = Board()
    _set_stones(board, black=[(5, 7), (6, 7), (7, 7)], current_player=WHITE)

    defenses = set(find_vct_defense_moves(board, BLACK, rule_mode="basic"))

    assert action_to_index(4, 7, BOARD_SIZE) in defenses
    assert action_to_index(8, 7, BOARD_SIZE) in defenses


def test_vct_first_move_respects_node_budget():
    board = Board()
    _set_stones(
        board,
        black=[(6, 7), (8, 7), (7, 6), (7, 8)],
        current_player=BLACK,
    )

    assert (
        vct_first_move(
            board,
            BLACK,
            max_depth=7,
            rule_mode="basic",
            node_budget=0,
        )
        is None
    )


def test_vct_first_move_does_not_ignore_opponent_immediate_win():
    board = Board()
    _set_stones(
        board,
        black=[(6, 7), (8, 7), (7, 6), (7, 8)],
        white=[(0, 0), (1, 0), (2, 0), (3, 0)],
        current_player=BLACK,
    )

    assert vct_first_move(board, BLACK, max_depth=7, rule_mode="basic") is None


def test_vct_first_move_none_on_quiet_position():
    board = Board()
    _set_stones(board, black=[(7, 7)], white=[(8, 8)], current_player=BLACK)

    assert vct_first_move(board, BLACK, max_depth=5, rule_mode="basic") is None


def test_vct_search_does_not_mutate_board():
    board = Board()
    _set_stones(
        board,
        black=[(6, 7), (8, 7), (7, 6), (7, 8)],
        white=[(5, 5)],
        current_player=BLACK,
    )
    before = _snapshot(board)

    _ = find_vct_attack_candidates(board, BLACK, rule_mode="basic")
    _ = vct_first_move(board, BLACK, max_depth=5, rule_mode="basic")

    assert _snapshot(board) == before
