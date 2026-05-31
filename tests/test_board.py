"""棋盘 Board 测试。"""

import pytest

from game.board import BLACK, BOARD_SIZE, EMPTY, WHITE, Board


def test_initial_board_is_empty():
    board = Board()
    assert board.move_count == 0
    assert board.last_move is None
    assert board.current_player == BLACK
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            assert board.grid[x][y] == EMPTY


def test_black_moves_first():
    board = Board()
    assert board.current_player == BLACK


def test_player_alternation_after_moves():
    board = Board()
    board.place_stone(7, 7)
    assert board.grid[7][7] == BLACK
    assert board.current_player == WHITE
    board.place_stone(7, 8)
    assert board.grid[7][8] == WHITE
    assert board.current_player == BLACK


def test_last_move_and_move_count_update():
    board = Board()
    board.place_stone(0, 0)
    assert board.last_move == (0, 0, BLACK)
    assert board.move_count == 1
    board.place_stone(14, 14)
    assert board.last_move == (14, 14, WHITE)
    assert board.move_count == 2


def test_cannot_place_on_existing_stone():
    board = Board()
    board.place_stone(7, 7)
    with pytest.raises(ValueError):
        board.place_stone(7, 7)


def test_cannot_place_out_of_bounds():
    board = Board()
    with pytest.raises(ValueError):
        board.place_stone(-1, 0)
    with pytest.raises(ValueError):
        board.place_stone(0, BOARD_SIZE)


def test_is_empty_and_is_legal_move():
    board = Board()
    assert board.is_empty(0, 0)
    assert board.is_legal_move(0, 0)
    assert not board.is_legal_move(-1, 0)
    assert not board.is_legal_move(0, BOARD_SIZE)
    board.place_stone(0, 0)
    assert not board.is_empty(0, 0)
    assert not board.is_legal_move(0, 0)


def test_get_legal_moves_count_changes():
    board = Board()
    assert len(board.get_legal_moves()) == BOARD_SIZE * BOARD_SIZE
    board.place_stone(7, 7)
    moves = board.get_legal_moves()
    assert len(moves) == BOARD_SIZE * BOARD_SIZE - 1
    assert (7, 7) not in moves


def test_copy_is_independent():
    board = Board()
    board.place_stone(7, 7)
    board.place_stone(8, 8)
    snapshot = board.copy()

    # 状态一致
    assert snapshot.move_count == board.move_count
    assert snapshot.current_player == board.current_player
    assert snapshot.last_move == board.last_move
    assert snapshot.grid[7][7] == BLACK
    assert snapshot.grid[8][8] == WHITE

    # 修改副本不影响原棋盘
    snapshot.place_stone(0, 0)
    assert snapshot.grid[0][0] == BLACK  # 副本下一手是黑
    assert board.grid[0][0] == EMPTY
    assert board.move_count == 2
    assert snapshot.move_count == 3


def test_reset_restores_initial_state():
    board = Board()
    board.place_stone(7, 7)
    board.place_stone(8, 8)
    board.reset()
    assert board.move_count == 0
    assert board.last_move is None
    assert board.current_player == BLACK
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            assert board.grid[x][y] == EMPTY
