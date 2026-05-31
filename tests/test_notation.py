"""game/notation.py 的单步 token 解析与格式化测试。"""

import pytest

from game.board import BLACK, WHITE
from game.notation import (
    MoveRecord,
    NotationError,
    format_move,
    normalize_move_text,
    parse_move_token,
)


# ---- 正常解析 --------------------------------------------------------------
def test_parse_b_j10_is_black_at_9_9():
    move = parse_move_token("B(J,10)")
    assert isinstance(move, MoveRecord)
    assert move.color == BLACK
    assert (move.x, move.y) == (9, 9)
    assert move.coord == "J10"
    assert move.raw == "B(J,10)"
    assert move.mark is None


def test_parse_w_h8_is_white_at_7_7():
    move = parse_move_token("W(H,8)")
    assert move.color == WHITE
    assert (move.x, move.y) == (7, 7)
    assert move.coord == "H8"


def test_parse_b_a1_corner():
    move = parse_move_token("B(A,1)")
    assert move.color == BLACK
    assert (move.x, move.y) == (0, 0)


def test_parse_w_o15_corner():
    move = parse_move_token("W(O,15)")
    assert move.color == WHITE
    assert (move.x, move.y) == (14, 14)


def test_parse_with_mark():
    move = parse_move_token("B(J,10)MARK[1]")
    assert move.color == BLACK
    assert (move.x, move.y) == (9, 9)
    assert move.mark == "1"


def test_parse_with_strict_mark_range():
    assert parse_move_token("B(J,10)MARK[-2]", strict_mark=True).mark == "-2"
    assert parse_move_token("B(J,10)MARK[2]", strict_mark=True).mark == "2"
    with pytest.raises(NotationError):
        parse_move_token("B(J,10)MARK[3]", strict_mark=True)
    with pytest.raises(NotationError):
        parse_move_token("B(J,10)MARK[abc]", strict_mark=True)


def test_parse_with_mark_text():
    move = parse_move_token("W(H,8)MARK[标注内容]")
    assert move.mark == "标注内容"


def test_parse_lowercase_token():
    # 大小写都应该被接受
    move = parse_move_token("b(j,10)")
    assert move.color == BLACK
    assert (move.x, move.y) == (9, 9)


def test_parse_with_whitespace():
    move = parse_move_token("  B( J , 10 )  ")
    assert move.color == BLACK
    assert (move.x, move.y) == (9, 9)


# ---- 异常处理 --------------------------------------------------------------
def test_invalid_color_raises():
    with pytest.raises(NotationError):
        parse_move_token("X(J,10)")


def test_invalid_letter_raises():
    with pytest.raises(NotationError):
        parse_move_token("B(P,10)")


def test_invalid_row_low_raises():
    with pytest.raises(NotationError):
        parse_move_token("B(A,0)")


def test_invalid_row_high_raises():
    with pytest.raises(NotationError):
        parse_move_token("B(A,16)")


def test_format_error_missing_paren():
    with pytest.raises(NotationError):
        parse_move_token("B J,10")


def test_format_error_missing_comma_and_value():
    with pytest.raises(NotationError):
        parse_move_token("B(J)")


def test_format_error_empty_string():
    with pytest.raises(NotationError):
        parse_move_token("")


def test_format_error_non_string():
    with pytest.raises(NotationError):
        parse_move_token(None)  # type: ignore[arg-type]


# ---- format_move -----------------------------------------------------------
def test_format_move_basic():
    assert format_move(BLACK, 9, 9) == "B(J,10)"
    assert format_move(WHITE, 7, 7) == "W(H,8)"
    assert format_move(BLACK, 0, 0) == "B(A,1)"
    assert format_move(WHITE, 14, 14) == "W(O,15)"


def test_format_move_with_mark():
    assert format_move(BLACK, 9, 9, mark="1") == "B(J,10)MARK[1]"


def test_format_move_invalid_color():
    with pytest.raises(NotationError):
        format_move(2, 0, 0)


def test_format_move_invalid_index():
    with pytest.raises(ValueError):
        format_move(BLACK, 15, 0)


# ---- format -> parse round-trip --------------------------------------------
@pytest.mark.parametrize(
    "color,x,y,mark",
    [
        (BLACK, 0, 0, None),
        (WHITE, 14, 14, None),
        (BLACK, 7, 7, None),
        (BLACK, 9, 9, "1"),
        (WHITE, 5, 12, "abc"),
    ],
)
def test_format_parse_round_trip(color, x, y, mark):
    token = format_move(color, x, y, mark=mark)
    parsed = parse_move_token(token)
    assert parsed.color == color
    assert (parsed.x, parsed.y) == (x, y)
    assert parsed.mark == mark


# ---- normalize_move_text ---------------------------------------------------
def test_normalize_move_text_uppercases_and_trims():
    assert normalize_move_text("  b(j,10) ") == "B(J,10)"


def test_normalize_move_text_rejects_non_string():
    with pytest.raises(NotationError):
        normalize_move_text(123)  # type: ignore[arg-type]
