"""坐标系统。

外部坐标格式：字母(列) + 数字(行)，例如 ``A1``、``H8``、``O15``。
- 列字母 ``A``-``O``，对应内部列索引 ``x = 0..14``。
- 行数字 ``1``-``15``，对应内部行索引 ``y = 0..14``。

与项目约定保持一致：
- ``A1``  -> ``(0, 0)``
- ``H8``  -> ``(7, 7)``  天元
- ``O15`` -> ``(14, 14)``
"""

from __future__ import annotations

BOARD_SIZE = 15
_COLUMN_LETTERS = "ABCDEFGHIJKLMNO"


def is_valid_coord(coord: str) -> bool:
    """判断 ``coord`` 是否是合法的外部坐标字符串。"""
    if not isinstance(coord, str):
        return False
    if len(coord) < 2 or len(coord) > 3:
        return False
    letter = coord[0].upper()
    if letter not in _COLUMN_LETTERS:
        return False
    number_part = coord[1:]
    if not number_part.isdigit():
        return False
    number = int(number_part)
    return 1 <= number <= BOARD_SIZE


def coord_to_index(coord: str) -> tuple[int, int]:
    """外部坐标 -> 内部 ``(x, y)`` 索引。

    ``x`` 是列索引(对应字母)，``y`` 是行索引(对应数字)。
    """
    if not is_valid_coord(coord):
        raise ValueError(f"非法坐标: {coord!r}")
    letter = coord[0].upper()
    number = int(coord[1:])
    x = _COLUMN_LETTERS.index(letter)
    y = number - 1
    return (x, y)


def index_to_coord(x: int, y: int) -> str:
    """内部 ``(x, y)`` 索引 -> 外部坐标字符串。"""
    if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
        raise ValueError(f"坐标越界: ({x}, {y})")
    return f"{_COLUMN_LETTERS[x]}{y + 1}"


__all__ = [
    "BOARD_SIZE",
    "is_valid_coord",
    "coord_to_index",
    "index_to_coord",
]
