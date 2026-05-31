"""game/encoder.py 的编码与索引映射测试。"""

import numpy as np
import pytest

from game.board import BLACK, BOARD_SIZE, WHITE, Board
from game.encoder import (
    action_to_index,
    encode_board,
    index_to_action,
    legal_moves_to_mask,
)


# ---- action_to_index / index_to_action -------------------------------------
def test_action_to_index_corners_and_center():
    assert action_to_index(0, 0) == 0      # A1
    assert action_to_index(1, 0) == 1      # B1
    assert action_to_index(0, 1) == 15     # A2
    assert action_to_index(7, 7) == 112    # H8 (天元)
    assert action_to_index(14, 14) == 224  # O15


def test_index_to_action_corners_and_center():
    assert index_to_action(0) == (0, 0)
    assert index_to_action(1) == (1, 0)
    assert index_to_action(15) == (0, 1)
    assert index_to_action(112) == (7, 7)
    assert index_to_action(224) == (14, 14)


def test_action_index_round_trip_for_all_cells():
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            idx = action_to_index(x, y)
            assert index_to_action(idx) == (x, y)


def test_action_to_index_out_of_range_raises():
    with pytest.raises(ValueError):
        action_to_index(-1, 0)
    with pytest.raises(ValueError):
        action_to_index(0, BOARD_SIZE)


def test_index_to_action_out_of_range_raises():
    with pytest.raises(ValueError):
        index_to_action(-1)
    with pytest.raises(ValueError):
        index_to_action(BOARD_SIZE * BOARD_SIZE)


# ---- encode_board ----------------------------------------------------------
def test_encode_board_shape_and_dtype():
    board = Board()
    arr = encode_board(board)
    assert arr.shape == (4, BOARD_SIZE, BOARD_SIZE)
    assert arr.dtype == np.float32


def test_encode_board_empty_board_is_all_zero_except_player_plane():
    board = Board()  # current_player = BLACK
    arr = encode_board(board)
    assert arr[0].sum() == 0.0  # 当前玩家暂无棋子
    assert arr[1].sum() == 0.0  # 对手暂无棋子
    assert arr[2].sum() == 0.0  # 没有 last_move
    assert (arr[3] == 1.0).all()  # 黑方持子，全 1


def test_encode_board_black_perspective_planes():
    board = Board()
    board.place_stone(0, 0)   # 黑棋落 A1
    board.place_stone(1, 1)   # 白棋落 B2
    # 此时 current_player == BLACK
    arr = encode_board(board)
    # 通道 0: 当前玩家(黑) 在 A1
    assert arr[0, 0, 0] == 1.0
    # 通道 1: 对手(白) 在 B2
    assert arr[1, 1, 1] == 1.0
    # 黑棋只在 A1，其余为 0
    assert arr[0].sum() == 1.0
    assert arr[1].sum() == 1.0
    # 通道 3: 全 1
    assert (arr[3] == 1.0).all()


def test_encode_board_white_perspective_swaps_planes():
    board = Board()
    board.place_stone(0, 0)  # 黑棋 A1, current_player -> WHITE
    arr = encode_board(board)  # 默认走 board.current_player == WHITE
    # 当前玩家是白，黑棋应在通道 1 (对手平面)
    assert arr[1, 0, 0] == 1.0
    assert arr[0, 0, 0] == 0.0
    # 通道 3: 全 -1
    assert (arr[3] == -1.0).all()


def test_encode_board_explicit_current_player_overrides_board_state():
    board = Board()
    board.place_stone(7, 7)  # 黑棋 H8
    # board.current_player 现在是 WHITE，但显式传 BLACK -> 视角换回黑方
    arr = encode_board(board, current_player=BLACK)
    assert arr[0, 7, 7] == 1.0  # 黑棋视角下，自己的子在通道 0
    assert (arr[3] == 1.0).all()


def test_encode_board_last_move_plane():
    board = Board()
    board.place_stone(7, 7)  # H8 黑
    arr = encode_board(board)
    assert arr[2, 7, 7] == 1.0
    assert arr[2].sum() == 1.0


def test_encode_board_no_last_move_plane_is_zero():
    board = Board()
    arr = encode_board(board)
    assert arr[2].sum() == 0.0


def test_encode_board_invalid_current_player_raises():
    board = Board()
    with pytest.raises(ValueError):
        encode_board(board, current_player=2)


# ---- legal_moves_to_mask ---------------------------------------------------
def test_legal_moves_mask_shape_and_dtype():
    board = Board()
    mask = legal_moves_to_mask(board)
    assert mask.shape == (BOARD_SIZE * BOARD_SIZE,)
    assert mask.dtype == np.float32


def test_legal_moves_mask_all_legal_initially():
    board = Board()
    mask = legal_moves_to_mask(board)
    assert mask.sum() == BOARD_SIZE * BOARD_SIZE
    assert (mask == 1.0).all()


def test_legal_moves_mask_marks_occupied_cells_as_zero():
    board = Board()
    board.place_stone(7, 7)   # H8 黑
    board.place_stone(0, 0)   # A1 白
    mask = legal_moves_to_mask(board)
    assert mask[action_to_index(7, 7)] == 0.0
    assert mask[action_to_index(0, 0)] == 0.0
    assert mask.sum() == BOARD_SIZE * BOARD_SIZE - 2


def test_legal_moves_mask_consistent_with_board_get_legal_moves():
    board = Board()
    board.place_stone(7, 7)
    board.place_stone(8, 8)
    board.place_stone(7, 8)
    mask = legal_moves_to_mask(board)
    legal_indexes = {action_to_index(x, y) for (x, y) in board.get_legal_moves()}
    for idx in range(BOARD_SIZE * BOARD_SIZE):
        if idx in legal_indexes:
            assert mask[idx] == 1.0
        else:
            assert mask[idx] == 0.0
