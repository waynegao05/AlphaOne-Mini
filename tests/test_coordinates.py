"""坐标系统测试。"""

import pytest

from game.coordinates import coord_to_index, index_to_coord, is_valid_coord


# ---- 三个关键点位的双向转换 -------------------------------------------------
def test_a1_to_index():
    assert coord_to_index("A1") == (0, 0)


def test_h8_to_index_is_tengen():
    # H8 是天元
    assert coord_to_index("H8") == (7, 7)


def test_o15_to_index():
    assert coord_to_index("O15") == (14, 14)


def test_index_to_coord_a1():
    assert index_to_coord(0, 0) == "A1"


def test_index_to_coord_h8():
    assert index_to_coord(7, 7) == "H8"


def test_index_to_coord_o15():
    assert index_to_coord(14, 14) == "O15"


def test_round_trip_all_cells():
    for x in range(15):
        for y in range(15):
            assert coord_to_index(index_to_coord(x, y)) == (x, y)


# ---- 合法性识别 ------------------------------------------------------------
@pytest.mark.parametrize("coord", ["A1", "H8", "O15", "a1", "h8", "o15", "B10", "G7"])
def test_valid_coords(coord):
    assert is_valid_coord(coord)


@pytest.mark.parametrize(
    "coord",
    [
        "",          # 空串
        "A",         # 缺数字
        "1",         # 缺字母
        "P1",        # 列超界(只到 O)
        "A0",        # 行从 1 起
        "A16",       # 行最大 15
        "AA1",       # 字母过长
        "A1A",       # 多余字符
        "ZZ",        # 全字母
        "@1",        # 非字母
        "A 1",       # 含空格
        "A-1",       # 含负号
    ],
)
def test_invalid_coords(coord):
    assert not is_valid_coord(coord)


def test_coord_to_index_raises_on_invalid():
    with pytest.raises(ValueError):
        coord_to_index("P1")
    with pytest.raises(ValueError):
        coord_to_index("A0")
    with pytest.raises(ValueError):
        coord_to_index("A16")


def test_index_to_coord_raises_on_out_of_range():
    with pytest.raises(ValueError):
        index_to_coord(-1, 0)
    with pytest.raises(ValueError):
        index_to_coord(15, 0)
    with pytest.raises(ValueError):
        index_to_coord(0, 15)


def test_is_valid_coord_handles_non_string():
    assert not is_valid_coord(None)  # type: ignore[arg-type]
    assert not is_valid_coord(123)   # type: ignore[arg-type]
