"""棋谱导出。

当前阶段提供两种格式：

简洁格式(必须支持)::

    B(J,10);W(L,10);B(J,11)

完整外层格式(可选)::

    {C5;B(J,10);W(L,10);B(J,11)}

允许在外层格式中附带元信息(以 ``[...]`` 形式)，例如::

    {C5;[2017.07.29 14:00 重庆];B(J,10);W(L,10)}
"""

from __future__ import annotations

from typing import Iterable, List, Mapping, Optional, Sequence, Union

from game.notation import MoveRecord, format_move
from records.metadata import RecordMetadata


MetadataLike = Union[Sequence[str], Mapping[str, str], None]


def _format_moves(moves: Iterable[MoveRecord]) -> List[str]:
    return [format_move(m.color, m.x, m.y, m.mark) for m in moves]


def export_moves(moves: Iterable[MoveRecord]) -> str:
    """导出为 ``"B(J,10);W(L,10);B(J,11)"`` 这样的简洁字符串。"""
    tokens = _format_moves(moves)
    return ";".join(tokens)


def _format_metadata(metadata: MetadataLike) -> List[str]:
    """把 ``metadata`` 规范成 ``[item]`` 形式的字符串列表。"""
    if metadata is None:
        return []
    parts: List[str] = []
    if isinstance(metadata, Mapping):
        for key, value in metadata.items():
            if value is None or value == "":
                parts.append(f"[{key}]")
            else:
                parts.append(f"[{key} {value}]")
    else:
        for item in metadata:
            if not isinstance(item, str):
                raise TypeError(
                    f"metadata 项必须是字符串或映射，实际 {type(item).__name__}: {item!r}"
                )
            parts.append(f"[{item}]")
    return parts


def export_record(
    moves: Iterable[MoveRecord],
    metadata: MetadataLike = None,
    rule_tag: str = "C5",
) -> str:
    """导出为 ``{C5;[meta];...;B(J,10);W(L,10);...}`` 的完整外层格式。

    - ``rule_tag`` 默认 ``"C5"`` (五子棋)。
    - ``metadata`` 可以是字符串列表或 ``key->value`` 映射；为空时省略。
    - ``moves`` 可以为空(只导出 header)。
    """
    body_parts: List[str] = [rule_tag]
    body_parts.extend(_format_metadata(metadata))
    body_parts.extend(_format_moves(moves))
    return "{" + ";".join(body_parts) + "}"


def export_standard_record(
    moves: Iterable[MoveRecord],
    metadata: RecordMetadata | None = None,
) -> str:
    """Export the CCGC-style bracketed header format.

    Example:
    ``{[C5][先手参赛队 B][后手参赛队 W][先手胜];B(J,10);W(H,8)}``
    """
    if metadata is None:
        metadata = RecordMetadata()
    header = "".join(f"[{item}]" for item in metadata.header_items())
    move_text = export_moves(moves)
    if move_text:
        return "{" + header + ";" + move_text + "}"
    return "{" + header + ";}"


__all__ = ["export_moves", "export_record", "export_standard_record"]
