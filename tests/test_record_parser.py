"""records/parser.py 与 records/exporter.py 的集成测试。"""

import pytest

from game.board import BLACK, WHITE, Board
from game.notation import MoveRecord, NotationError
from records.exporter import export_moves, export_record
from records.parser import (
    RecordError,
    apply_moves_to_board,
    extract_move_tokens,
    parse_record,
    validate_move_sequence,
)


SAMPLE_RECORD = (
    "{{C5}[先手参赛队 B][后手参赛队 W][先手胜][2017.07.29 14:00 重庆][2017 CCGC];"
    "B(J,10);W(L,10);B(J,11);W(L,12);B(H,10);W(H,8);B(K,8)}"
)


# ---- extract_move_tokens ---------------------------------------------------
def test_extract_move_tokens_from_full_record():
    tokens = extract_move_tokens(SAMPLE_RECORD)
    assert tokens == [
        "B(J,10)",
        "W(L,10)",
        "B(J,11)",
        "W(L,12)",
        "B(H,10)",
        "W(H,8)",
        "B(K,8)",
    ]


def test_extract_move_tokens_ignores_metadata_letters():
    # 元信息中出现的孤立 B / W 字符不应被误判成落子
    text = "[先手 B][后手 W][胜方 B];B(J,10)"
    tokens = extract_move_tokens(text)
    assert tokens == ["B(J,10)"]


def test_extract_move_tokens_with_mark():
    text = "B(J,10)MARK[1];W(H,8)MARK[2]"
    tokens = extract_move_tokens(text)
    assert tokens == ["B(J,10)MARK[1]", "W(H,8)MARK[2]"]


def test_extract_move_tokens_empty_text():
    assert extract_move_tokens("") == []


# ---- parse_record ----------------------------------------------------------
def test_parse_record_returns_ordered_move_records():
    moves = parse_record(SAMPLE_RECORD)
    assert len(moves) == 7
    assert all(isinstance(m, MoveRecord) for m in moves)
    assert [m.coord for m in moves] == [
        "J10", "L10", "J11", "L12", "H10", "H8", "K8"
    ]
    assert [m.color for m in moves] == [
        BLACK, WHITE, BLACK, WHITE, BLACK, WHITE, BLACK
    ]


def test_parse_record_empty_raises():
    with pytest.raises(RecordError):
        parse_record("")
    with pytest.raises(RecordError):
        parse_record("[only metadata]")


def test_parse_record_propagates_token_errors():
    with pytest.raises(RecordError):
        parse_record("B(J,10);X(H,8)")


def test_parse_record_rejects_negative_row_in_later_token():
    with pytest.raises(RecordError):
        parse_record("B(J,10);W(A,-1)")


def test_parse_record_rejects_malformed_later_token():
    with pytest.raises(RecordError):
        parse_record("B(J,10);W(H)")


# ---- validate_move_sequence ------------------------------------------------
def _make_move(color: int, coord: str, x: int, y: int) -> MoveRecord:
    raw = f"{'B' if color == BLACK else 'W'}({coord[0]},{coord[1:]})"
    return MoveRecord(color=color, x=x, y=y, coord=coord, raw=raw)


def test_validate_first_move_must_be_black():
    moves = [_make_move(WHITE, "H8", 7, 7)]
    with pytest.raises(RecordError):
        validate_move_sequence(moves)


def test_validate_color_alternation_violation():
    moves = [
        _make_move(BLACK, "J10", 9, 9),
        _make_move(BLACK, "K10", 10, 9),
    ]
    with pytest.raises(RecordError):
        validate_move_sequence(moves)


def test_validate_duplicate_position():
    moves = [
        _make_move(BLACK, "J10", 9, 9),
        _make_move(WHITE, "J10", 9, 9),
    ]
    with pytest.raises(RecordError):
        validate_move_sequence(moves)


def test_validate_empty_raises():
    with pytest.raises(RecordError):
        validate_move_sequence([])


def test_validate_legal_sequence_ok():
    moves = [
        _make_move(BLACK, "J10", 9, 9),
        _make_move(WHITE, "L10", 11, 9),
        _make_move(BLACK, "J11", 9, 10),
    ]
    validate_move_sequence(moves)  # 不抛异常即通过


# ---- apply_moves_to_board --------------------------------------------------
def test_apply_moves_to_board_replays_correctly():
    moves = parse_record(SAMPLE_RECORD)
    board = apply_moves_to_board(moves)
    assert board.move_count == 7
    # 黑方下了 4 手，最后一手是 K8 (BLACK)，因此下一手应为白方
    assert board.current_player == WHITE
    assert board.last_move == (10, 7, BLACK)  # K8 -> (10, 7)
    # 抽几个点验证棋子颜色
    assert board.grid[9][9] == BLACK   # J10
    assert board.grid[11][9] == WHITE  # L10
    assert board.grid[7][7] == WHITE   # H8


def test_apply_moves_rejects_duplicate_position():
    moves = [
        _make_move(BLACK, "J10", 9, 9),
        _make_move(WHITE, "J10", 9, 9),
    ]
    with pytest.raises(RecordError):
        apply_moves_to_board(moves)


def test_apply_moves_rejects_consecutive_same_color():
    moves = [
        _make_move(BLACK, "J10", 9, 9),
        _make_move(BLACK, "K10", 10, 9),
    ]
    with pytest.raises(RecordError):
        apply_moves_to_board(moves)


def test_apply_moves_to_existing_board_must_match_turn():
    board = Board()
    board.place_stone(7, 7)  # 黑落 H8 -> 当前轮到白
    moves = [_make_move(BLACK, "J10", 9, 9)]  # 仍以黑开始 -> validate 会先通过
    # validate_move_sequence 不感知 board，所以这里报的是 board 轮次冲突
    with pytest.raises(RecordError):
        apply_moves_to_board(moves, board=board)


# ---- exporter --------------------------------------------------------------
def test_export_moves_simple_format():
    moves = parse_record("B(J,10);W(H,8);B(K,8)")
    assert export_moves(moves) == "B(J,10);W(H,8);B(K,8)"


def test_export_moves_preserves_mark():
    moves = parse_record("B(J,10)MARK[1];W(H,8)")
    assert export_moves(moves) == "B(J,10)MARK[1];W(H,8)"


def test_export_record_outer_format():
    moves = parse_record("B(J,10);W(L,10);B(J,11)")
    assert export_record(moves) == "{C5;B(J,10);W(L,10);B(J,11)}"


def test_export_record_with_metadata_list():
    moves = parse_record("B(J,10);W(H,8)")
    out = export_record(moves, metadata=["2017.07.29", "重庆"])
    assert out == "{C5;[2017.07.29];[重庆];B(J,10);W(H,8)}"


def test_export_record_with_metadata_mapping():
    moves = parse_record("B(J,10)")
    out = export_record(moves, metadata={"date": "2017.07.29", "site": "重庆"})
    assert out == "{C5;[date 2017.07.29];[site 重庆];B(J,10)}"


def test_export_round_trip_simple():
    original_text = "B(J,10);W(L,10);B(J,11);W(L,12);B(H,10);W(H,8);B(K,8)"
    moves = parse_record(original_text)
    exported = export_moves(moves)
    assert exported == original_text
    reparsed = parse_record(exported)
    assert [(m.color, m.x, m.y) for m in reparsed] == [
        (m.color, m.x, m.y) for m in moves
    ]


def test_export_round_trip_full_record():
    moves = parse_record(SAMPLE_RECORD)
    full_text = export_record(moves)
    # 再解析出来落子序列要保持一致
    reparsed = parse_record(full_text)
    assert [(m.color, m.x, m.y) for m in reparsed] == [
        (m.color, m.x, m.y) for m in moves
    ]
