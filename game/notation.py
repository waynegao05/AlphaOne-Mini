"""棋谱单步落子 token 的解析与格式化。

支持的 token 形式(忽略大小写)::

    B(J,10)            # 黑棋落在 J10
    W(H,8)             # 白棋落在 H8
    B(A,1)             # 角点
    W(O,15)            # 角点
    B(J,10)MARK[1]     # 带标注

解析结果用 :class:`MoveRecord` 承载。仅做单步级别的解析，序列校验放在
:mod:`records.parser` 中处理。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .board import BLACK, WHITE
from .coordinates import BOARD_SIZE, coord_to_index, index_to_coord, is_valid_coord


class NotationError(ValueError):
    """棋谱单步级别的格式或字段错误。"""


@dataclass
class MoveRecord:
    """一手落子的结构化表示。

    - ``color``: ``BLACK`` (1) / ``WHITE`` (-1)
    - ``x``, ``y``: 内部坐标(``x`` 是列, ``y`` 是行)
    - ``coord``: 标准化后的外部坐标，如 ``"J10"``
    - ``raw``: 原始 token 字符串(便于调试 / 错误报告)
    - ``mark``: ``MARK[...]`` 中的标注内容；没有则为 ``None``
    """

    color: int
    x: int
    y: int
    coord: str
    raw: str
    mark: Optional[str] = None


# 单步 token 整体匹配。使用宽松的字符类(再人工校验值域)，便于报出更具体的错误。
_MOVE_TOKEN_RE = re.compile(
    r"""
    ^\s*
    ([A-Za-z])              # 1: 颜色字符
    \s*\(\s*
    ([A-Za-z])              # 2: 列字母
    \s*,\s*
    (-?\d+)                 # 3: 行号(允许负数以便给出更明确的错误)
    \s*\)
    (?:                     # 可选 MARK[...]
        \s*[Mm][Aa][Rr][Kk]\s*\[\s*
        ([^\]]*)            # 4: MARK 内容
        \s*\]
    )?
    \s*$
    """,
    re.VERBOSE,
)

# 提取整谱中的落子 token 时使用的宽松匹配模式：
# - 颜色和列字母都允许任意字母，由 :func:`parse_move_token` 做值域校验。
# - 这样 ``X(H,8)`` 这类非法颜色 token 也会被提取出来并清晰报错，而不是默默跳过。
_EXTRACT_TOKEN_RE = re.compile(
    r"[A-Za-z]\s*\(\s*[A-Za-z]\s*,\s*-?\d+\s*\)(?:\s*MARK\s*\[[^\]]*\])?",
    re.IGNORECASE,
)


def normalize_move_text(text: str) -> str:
    """归一化单步 token：去掉首尾空白并把字母统一成大写。

    保留括号内的逗号与数字原样；只对字母做大小写规范化。
    """
    if not isinstance(text, str):
        raise NotationError(f"token 必须是字符串: {text!r}")
    return text.strip().upper()


def validate_mark_value(mark: Optional[str]) -> None:
    """Validate competition MARK value when strict parsing is requested.

    The CCGC record format defines MARK values as integers in ``[-2, 2]``:
    -2, -1, 0, 1, 2.
    """
    if mark is None:
        return
    text = str(mark).strip()
    try:
        value = int(text)
    except ValueError as exc:
        raise NotationError(f"MARK 必须是 -2..2 的整数: {mark!r}") from exc
    if value < -2 or value > 2:
        raise NotationError(f"MARK 超出范围 -2..2: {mark!r}")


def parse_move_token(token: str, strict_mark: bool = False) -> MoveRecord:
    """解析一个单步落子 token 为 :class:`MoveRecord`。

    校验顺序：先做正则结构校验，再分别校验颜色字符、列字母、行号是否在合法范围。
    任何异常都抛 :class:`NotationError`，并附带原始 ``token`` 便于定位。
    """
    if not isinstance(token, str):
        raise NotationError(f"token 必须是字符串: {token!r}")

    raw = token
    stripped = token.strip()
    if not stripped:
        raise NotationError("空 token")

    match = _MOVE_TOKEN_RE.match(stripped)
    if match is None:
        raise NotationError(f"格式错误，无法解析落子 token: {raw!r}")

    color_char = match.group(1).upper()
    letter = match.group(2).upper()
    number_str = match.group(3)
    mark = match.group(4)
    if strict_mark:
        validate_mark_value(mark)

    if color_char == "B":
        color = BLACK
    elif color_char == "W":
        color = WHITE
    else:
        raise NotationError(f"非法颜色 {color_char!r}: {raw!r}")

    coord = f"{letter}{number_str}"
    if not is_valid_coord(coord):
        raise NotationError(
            f"非法坐标 {coord!r} (合法范围: A1 ~ O{BOARD_SIZE}): {raw!r}"
        )
    x, y = coord_to_index(coord)

    return MoveRecord(color=color, x=x, y=y, coord=coord, raw=raw, mark=mark)


def format_move(
    color: int, x: int, y: int, mark: Optional[str] = None
) -> str:
    """把 ``(color, x, y[, mark])`` 格式化为标准 token，例如 ``B(J,10)``。"""
    if color == BLACK:
        color_char = "B"
    elif color == WHITE:
        color_char = "W"
    else:
        raise NotationError(f"非法颜色: {color!r}")

    coord = index_to_coord(x, y)  # 例如 "J10"
    letter = coord[0]
    number = coord[1:]
    token = f"{color_char}({letter},{number})"
    if mark is not None:
        token += f"MARK[{mark}]"
    return token


__all__ = [
    "NotationError",
    "MoveRecord",
    "normalize_move_text",
    "parse_move_token",
    "validate_mark_value",
    "format_move",
    "_EXTRACT_TOKEN_RE",
]
