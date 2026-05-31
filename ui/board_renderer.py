"""命令行棋盘渲染。

约定：
- 黑棋显示 ``X``、白棋显示 ``O``、空点显示 ``.``。
- ``highlight_last=True`` 时把最近一手用 ``(X)`` / ``(O)`` 包起来(3 字符等宽，
  与普通格 ``" X "`` / ``" . "`` 对齐)。
- ``show_coords=True`` 时打印列字母 ``A-O`` 与行号 ``1-15``。
"""

from __future__ import annotations

from typing import Optional

from game.board import BLACK, EMPTY, WHITE, BOARD_SIZE, Board


_COLUMN_LETTERS = "ABCDEFGHIJKLMNO"


def stone_to_char(value: int) -> str:
    """把棋子值转成单字符显示。"""
    if value == BLACK:
        return "X"
    if value == WHITE:
        return "O"
    return "."


def render_board(
    board: Board,
    show_coords: bool = True,
    highlight_last: bool = True,
) -> str:
    """把 ``board`` 渲染成多行字符串。"""
    n = min(BOARD_SIZE, len(_COLUMN_LETTERS))

    last_x: Optional[int] = None
    last_y: Optional[int] = None
    if highlight_last and board.last_move is not None:
        lx, ly, _ = board.last_move
        last_x, last_y = lx, ly

    lines = []

    if show_coords:
        header = "    " + "".join(f" {_COLUMN_LETTERS[i]} " for i in range(n))
        lines.append(header)

    for y in range(n):
        cells = []
        for x in range(n):
            ch = stone_to_char(board.grid[x][y])
            if (x, y) == (last_x, last_y):
                cells.append(f"({ch})")
            else:
                cells.append(f" {ch} ")
        if show_coords:
            row = f"{y + 1:>2}  " + "".join(cells)
        else:
            row = "    " + "".join(cells)
        lines.append(row)

    return "\n".join(lines)


__all__ = ["stone_to_char", "render_board"]
