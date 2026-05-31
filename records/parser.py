"""棋谱解析。

从棋谱原文提取落子 token、解析成 :class:`MoveRecord` 列表，并能把列表
还原到 :class:`Board` 上。

棋谱原文示例::

    {{C5}[先手参赛队 B][后手参赛队 W][先手胜][2017.07.29 14:00 重庆][2017 CCGC];
    B(J,10);W(L,10);B(J,11);W(L,12);B(H,10);W(H,8);B(K,8)}

当前阶段只关注落子序列，元信息保留为原始字符串供上层使用。
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence

from game.board import BLACK, WHITE, Board
from game.notation import (
    MoveRecord,
    NotationError,
    _EXTRACT_TOKEN_RE,
    parse_move_token,
)


class RecordError(ValueError):
    """棋谱级别的错误：序列校验、棋盘冲突等。"""


_MOVE_CANDIDATE_RE = re.compile(
    r"(?<![A-Za-z])"
    r"([A-Za-z]\s*\([^;{}\[\]]*\)(?:\s*MARK\s*\[[^\]]*\])?)"
    r"(?!\s*[A-Za-z])",
    re.IGNORECASE,
)


def _color_name(color: int) -> str:
    if color == BLACK:
        return "黑"
    if color == WHITE:
        return "白"
    return f"未知({color})"


def extract_move_tokens(record_text: str) -> List[str]:
    """从棋谱原文中提取所有落子 token 字符串(保持原始大小写)。

    支持忽略空白、中文元信息、外层 ``{...}`` 包装、可选 ``MARK[...]`` 标注。
    """
    if not isinstance(record_text, str):
        raise RecordError(f"棋谱必须是字符串: {record_text!r}")
    return _EXTRACT_TOKEN_RE.findall(record_text)


def parse_record(record_text: str, strict_mark: bool = False) -> List[MoveRecord]:
    """从棋谱原文解析得到有序的 :class:`MoveRecord` 列表。

    若提取不到任何落子，抛 :class:`RecordError`。
    单步格式错误会原样冒泡为 :class:`game.notation.NotationError`。
    """
    if not isinstance(record_text, str):
        raise RecordError(f"妫嬭氨蹇呴』鏄瓧绗︿覆: {record_text!r}")

    tokens = _MOVE_CANDIDATE_RE.findall(record_text)
    if not tokens:
        raise RecordError(f"棋谱中未找到任何落子 token: {record_text!r}")
    moves: List[MoveRecord] = []
    for idx, token in enumerate(tokens, start=1):
        try:
            moves.append(parse_move_token(token, strict_mark=strict_mark))
        except NotationError as exc:
            raise RecordError(f"第 {idx} 手解析失败: {exc}") from exc
    return moves


def validate_move_sequence(moves: Sequence[MoveRecord]) -> None:
    """对 ``moves`` 序列做颜色顺序、重复落子等校验(不依赖 Board)。"""
    if not moves:
        raise RecordError("空棋谱：落子序列为空")

    if moves[0].color != BLACK:
        raise RecordError(
            f"第 1 手必须是黑棋，实际为 {_color_name(moves[0].color)}: {moves[0].raw!r}"
        )

    seen: set = set()
    expected_color = BLACK
    for idx, move in enumerate(moves, start=1):
        if move.color != expected_color:
            raise RecordError(
                f"第 {idx} 手颜色顺序错误，期望 {_color_name(expected_color)} "
                f"实际 {_color_name(move.color)}: {move.raw!r}"
            )
        if (move.x, move.y) in seen:
            raise RecordError(f"第 {idx} 手重复落子 {move.coord}: {move.raw!r}")
        seen.add((move.x, move.y))
        expected_color = -expected_color


def apply_moves_to_board(
    moves: Iterable[MoveRecord], board: Optional[Board] = None
) -> Board:
    """把 ``moves`` 依次落到 ``board`` 上并返回该 ``board``。

    - ``board=None`` 时新建一张空棋盘。
    - 在落子前先做整体的 :func:`validate_move_sequence` 校验。
    - 与 ``board`` 当前轮次冲突、或落子点已有棋子时抛 :class:`RecordError`。
    """
    move_list = list(moves)
    validate_move_sequence(move_list)

    if board is None:
        board = Board()

    for idx, move in enumerate(move_list, start=1):
        if board.current_player != move.color:
            raise RecordError(
                f"第 {idx} 手与 board 当前轮次不一致: "
                f"board 期望 {_color_name(board.current_player)}, "
                f"棋谱给出 {_color_name(move.color)}: {move.raw!r}"
            )
        if not board.is_legal_move(move.x, move.y):
            raise RecordError(
                f"第 {idx} 手在 {move.coord} 处非法落子(越界或已被占): {move.raw!r}"
            )
        board.place_stone(move.x, move.y)

    return board


__all__ = [
    "RecordError",
    "extract_move_tokens",
    "parse_record",
    "validate_move_sequence",
    "apply_moves_to_board",
]
