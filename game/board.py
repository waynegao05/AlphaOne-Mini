"""棋盘状态。

约定：
- 棋盘大小 ``BOARD_SIZE = 15``。
- ``grid[x][y]`` 中 ``x`` 是列索引，``y`` 是行索引；与 :mod:`game.coordinates` 一致。
- 棋子值：``EMPTY = 0``、``BLACK = 1``、``WHITE = -1``。
- 黑棋先行，每次落子后 ``current_player`` 自动取反。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

BOARD_SIZE = 15
EMPTY = 0
BLACK = 1
WHITE = -1


class Board:
    """15x15 五子棋棋盘。"""

    BOARD_SIZE = BOARD_SIZE
    EMPTY = EMPTY
    BLACK = BLACK
    WHITE = WHITE

    def __init__(self) -> None:
        self.grid: List[List[int]] = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.current_player: int = BLACK
        self.move_count: int = 0
        self.last_move: Optional[Tuple[int, int, int]] = None
        self.reset()

    # ---- 基础维护 ---------------------------------------------------------
    def reset(self) -> None:
        """重置棋盘到初始状态。"""
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                self.grid[x][y] = EMPTY
        self.current_player = BLACK
        self.move_count = 0
        self.last_move = None

    def copy(self) -> "Board":
        """返回当前棋盘的深拷贝。"""
        new_board = Board.__new__(Board)
        new_board.grid = [row[:] for row in self.grid]
        new_board.current_player = self.current_player
        new_board.move_count = self.move_count
        new_board.last_move = self.last_move
        return new_board

    # ---- 查询 -------------------------------------------------------------
    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE

    def is_empty(self, x: int, y: int) -> bool:
        if not self.in_bounds(x, y):
            return False
        return self.grid[x][y] == EMPTY

    def is_legal_move(self, x: int, y: int) -> bool:
        """是否为合法落子位置(只考虑基础规则)。"""
        return self.in_bounds(x, y) and self.grid[x][y] == EMPTY

    def get_legal_moves(self) -> List[Tuple[int, int]]:
        """返回所有合法落子位置 ``(x, y)`` 列表。"""
        moves: List[Tuple[int, int]] = []
        for x in range(BOARD_SIZE):
            row = self.grid[x]
            for y in range(BOARD_SIZE):
                if row[y] == EMPTY:
                    moves.append((x, y))
        return moves

    # ---- 落子 -------------------------------------------------------------
    def place_stone(self, x: int, y: int) -> None:
        """在 ``(x, y)`` 落下当前玩家的棋子，并切换 ``current_player``。

        非法位置抛 ``ValueError``。
        """
        if not self.is_legal_move(x, y):
            raise ValueError(f"非法落子: ({x}, {y})")
        color = self.current_player
        self.grid[x][y] = color
        self.last_move = (x, y, color)
        self.move_count += 1
        self.current_player = -color

    # ---- 调试辅助 ---------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"Board(move_count={self.move_count}, "
            f"current_player={self.current_player}, last_move={self.last_move})"
        )


__all__ = ["Board", "BOARD_SIZE", "EMPTY", "BLACK", "WHITE"]
