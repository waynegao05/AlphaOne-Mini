from __future__ import annotations

import pytest

from game.board import BLACK, EMPTY, WHITE, Board
from game.encoder import action_to_index


def _snapshot(board: Board):
    return {
        "grid": [col[:] for col in board.grid],
        "current_player": board.current_player,
        "last_move": board.last_move,
        "move_count": board.move_count,
    }


def _assert_restored(board: Board, before) -> None:
    assert board.grid == before["grid"]
    assert board.current_player == before["current_player"]
    assert board.last_move == before["last_move"]
    assert board.move_count == before["move_count"]


def test_temporary_move_changes_state_inside_and_restores_after_exit():
    from engine.simulation import temporary_move

    board = Board()
    board.place_stone(7, 7)
    before = _snapshot(board)

    with temporary_move(board, action_to_index(8, 7), WHITE):
        assert board.grid[8][7] == WHITE
        assert board.last_move == (8, 7, WHITE)
        assert board.current_player == BLACK
        assert board.move_count == before["move_count"] + 1

    _assert_restored(board, before)


def test_temporary_move_restores_after_exception():
    from engine.simulation import temporary_move

    board = Board()
    before = _snapshot(board)

    with pytest.raises(RuntimeError, match="boom"):
        with temporary_move(board, action_to_index(7, 7), BLACK):
            assert board.grid[7][7] == BLACK
            raise RuntimeError("boom")

    _assert_restored(board, before)


def test_temporary_move_rejects_occupied_points():
    from engine.simulation import temporary_move

    board = Board()
    board.place_stone(7, 7)
    before = _snapshot(board)

    with pytest.raises(ValueError, match="occupied|illegal"):
        with temporary_move(board, action_to_index(7, 7), WHITE):
            pass

    _assert_restored(board, before)


def test_temporary_move_supports_nested_different_points():
    from engine.simulation import temporary_move

    board = Board()
    before = _snapshot(board)

    with temporary_move(board, action_to_index(7, 7), BLACK):
        assert board.grid[7][7] == BLACK
        with temporary_move(board, action_to_index(8, 7), WHITE):
            assert board.grid[8][7] == WHITE
            assert board.move_count == before["move_count"] + 2
        assert board.grid[8][7] == EMPTY
        assert board.grid[7][7] == BLACK

    _assert_restored(board, before)


def test_repeated_temporary_moves_leave_board_unchanged():
    from engine.simulation import temporary_move

    board = Board()
    board.place_stone(7, 7)
    before = _snapshot(board)

    for action in (action_to_index(8, 7), action_to_index(6, 7), action_to_index(7, 8)):
        with temporary_move(board, action, WHITE):
            assert board.last_move is not None

    _assert_restored(board, before)


def test_temporary_move_matches_copy_place_stone_rule_state():
    from engine.simulation import temporary_move
    from game.rules_basic import check_winner

    board = Board()
    for x in (3, 4, 5, 6):
        board.grid[x][7] = BLACK
    board.move_count = 4
    board.current_player = BLACK
    before = _snapshot(board)

    copied = board.copy()
    copied.current_player = BLACK
    copied.place_stone(7, 7)
    expected = check_winner(copied, copied.last_move)

    with temporary_move(board, action_to_index(7, 7), BLACK):
        assert check_winner(board, board.last_move) == expected
        assert board.last_move == copied.last_move
        assert board.current_player == copied.current_player
        assert board.move_count == copied.move_count

    _assert_restored(board, before)


def test_threat_classification_does_not_pollute_board():
    from engine.threats import classify_move_threats, find_immediate_winning_moves

    board = Board()
    for x in (3, 4, 5, 6):
        board.grid[x][7] = BLACK
    board.move_count = 4
    board.current_player = BLACK
    before = _snapshot(board)

    threats = classify_move_threats(board, action_to_index(7, 7), BLACK)
    wins = find_immediate_winning_moves(board, BLACK)

    assert "five" in threats
    assert action_to_index(2, 7) in wins
    assert action_to_index(7, 7) in wins
    _assert_restored(board, before)
