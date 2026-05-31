"""棋盘状态 -> 神经网络输入的编码与动作索引映射。

约定：
- 棋盘 ``BOARD_SIZE = 15``。
- ``board.grid[x][y]``：``x`` 是列索引(0..14)，``y`` 是行索引(0..14)。
- ``EMPTY = 0``、``BLACK = 1``、``WHITE = -1``，与前两批保持一致。

动作索引(action_index)与外部坐标的关系::

    action_index = y * BOARD_SIZE + x
    A1  -> (x=0,  y=0)  -> 0
    B1  -> (x=1,  y=0)  -> 1
    A2  -> (x=0,  y=1)  -> 15
    H8  -> (x=7,  y=7)  -> 112
    O15 -> (x=14, y=14) -> 224

编码张量约定：``ndarray`` 形状 ``(4, BOARD_SIZE, BOARD_SIZE)``，索引顺序
``planes[c, y, x]``(行先于列，符合 CNN 默认 ``[C, H, W]``)。
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .board import BLACK, BOARD_SIZE, EMPTY, WHITE, Board


def action_to_index(x: int, y: int, board_size: int = BOARD_SIZE) -> int:
    """``(x, y)`` -> 动作索引。``x`` 必须是列、``y`` 必须是行。"""
    if not (0 <= x < board_size and 0 <= y < board_size):
        raise ValueError(
            f"action_to_index 越界: x={x}, y={y}, board_size={board_size}"
        )
    return y * board_size + x


def index_to_action(index: int, board_size: int = BOARD_SIZE) -> Tuple[int, int]:
    """动作索引 -> ``(x, y)``。"""
    total = board_size * board_size
    if not (0 <= index < total):
        raise ValueError(
            f"index_to_action 越界: index={index}, 合法范围 [0, {total})"
        )
    y, x = divmod(index, board_size)
    return (x, y)


def encode_board(
    board: Board, current_player: Optional[int] = None
) -> np.ndarray:
    """把 ``board`` 编码成 ``(4, BOARD_SIZE, BOARD_SIZE)`` 的 float32 张量。

    通道含义：
    - ``[0]`` 当前玩家的棋子位置(从当前玩家视角)
    - ``[1]`` 对手玩家的棋子位置
    - ``[2]`` ``last_move`` 落点；``board.last_move`` 为 ``None`` 时全 0
    - ``[3]`` 当前玩家标识平面：黑棋为 1、白棋为 -1

    ``current_player``：若显式传入则以此为准，否则使用 ``board.current_player``。
    """
    if current_player is None:
        current_player = board.current_player
    if current_player not in (BLACK, WHITE):
        raise ValueError(
            f"current_player 必须是 BLACK(1) 或 WHITE(-1)，实际 {current_player!r}"
        )

    opponent = -current_player
    planes = np.zeros((4, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    grid = board.grid
    for x in range(BOARD_SIZE):
        column = grid[x]
        for y in range(BOARD_SIZE):
            stone = column[y]
            if stone == current_player:
                planes[0, y, x] = 1.0
            elif stone == opponent:
                planes[1, y, x] = 1.0
            # EMPTY -> 两个通道都保持 0

    if board.last_move is not None:
        lx, ly, _ = board.last_move
        if 0 <= lx < BOARD_SIZE and 0 <= ly < BOARD_SIZE:
            planes[2, ly, lx] = 1.0

    if current_player == BLACK:
        planes[3, :, :] = 1.0
    else:  # WHITE
        planes[3, :, :] = -1.0

    return planes


def legal_moves_to_mask(board: Board, board_size: int = BOARD_SIZE) -> np.ndarray:
    """返回长度 ``board_size * board_size`` 的合法动作 mask(float32)。

    - 空点对应 ``1.0``、已被占据的点对应 ``0.0``。
    - 当前阶段只检查"是否为空"，不检查禁手。
    """
    mask = np.zeros(board_size * board_size, dtype=np.float32)
    grid = board.grid
    for x in range(board_size):
        column = grid[x]
        for y in range(board_size):
            if column[y] == EMPTY:
                mask[y * board_size + x] = 1.0
    return mask


__all__ = [
    "action_to_index",
    "index_to_action",
    "encode_board",
    "legal_moves_to_mask",
]
