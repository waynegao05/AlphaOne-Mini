"""基础五子棋胜负判断规则。

当前阶段只实现：
- 任意一方连成 5 子或以上即判胜（暂不区分黑方禁手 / 长连禁手）。
- 棋盘下满且无人获胜判平局。
"""

from __future__ import annotations

from typing import Optional, Tuple

from .board import BLACK, BOARD_SIZE, EMPTY, WHITE, Board


def count_continuous_stones(
    board: Board, x: int, y: int, dx: int, dy: int, color: int
) -> int:
    """从 ``(x, y)`` 出发(不含自身)，沿 ``(dx, dy)`` 方向数同色 ``color`` 棋子数量。"""
    if dx == 0 and dy == 0:
        raise ValueError("(dx, dy) 不能同时为 0")
    count = 0
    nx, ny = x + dx, y + dy
    while 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and board.grid[nx][ny] == color:
        count += 1
        nx += dx
        ny += dy
    return count


def check_five_or_more(board: Board, x: int, y: int, color: int) -> bool:
    """判断以 ``(x, y)`` 为穿过点是否存在 ``color`` 的 5 连或以上。"""
    if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
        return False
    if board.grid[x][y] != color or color == EMPTY:
        return False
    # 4 个方向：横、纵、主对角(↘)、副对角(↗)。
    directions = ((1, 0), (0, 1), (1, 1), (1, -1))
    for dx, dy in directions:
        forward = count_continuous_stones(board, x, y, dx, dy, color)
        backward = count_continuous_stones(board, x, y, -dx, -dy, color)
        if forward + backward + 1 >= 5:
            return True
    return False


def check_winner(board: Board, last_move: Optional[Tuple[int, int, int]]) -> int:
    """根据 ``last_move`` 判断是否产生赢家。

    返回值：
    - ``BLACK``(1) 黑胜
    - ``WHITE``(-1) 白胜
    - ``0`` 当前无胜者
    """
    if last_move is None:
        return 0
    x, y, color = last_move
    if color == EMPTY:
        return 0
    if check_five_or_more(board, x, y, color):
        return color
    return 0


def _find_any_winner(board: Board) -> int:
    """Return a winner found anywhere on the board, or 0 if none exists."""
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            color = board.grid[x][y]
            if color != EMPTY and check_five_or_more(board, x, y, color):
                return color
    return 0


def is_draw(board: Board) -> bool:
    """棋盘下满且无人获胜则为平局。"""
    if board.move_count < BOARD_SIZE * BOARD_SIZE:
        return False
    return _find_any_winner(board) == 0


def is_game_over(
    board: Board, last_move: Optional[Tuple[int, int, int]] = None
) -> bool:
    """对局是否结束：有胜者或平局。"""
    if last_move is None:
        last_move = board.last_move
    if check_winner(board, last_move) != 0:
        return True
    if board.move_count >= BOARD_SIZE * BOARD_SIZE and _find_any_winner(board) != 0:
        return True
    return is_draw(board)


__all__ = [
    "count_continuous_stones",
    "check_five_or_more",
    "check_winner",
    "is_draw",
    "is_game_over",
    "BLACK",
    "WHITE",
    "EMPTY",
]
