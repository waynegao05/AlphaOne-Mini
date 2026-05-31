"""ui/board_renderer.py 的单元测试。"""

from __future__ import annotations

from game.board import BLACK, EMPTY, WHITE, Board
from ui.board_renderer import render_board, stone_to_char


def test_stone_to_char_mapping():
    assert stone_to_char(EMPTY) == "."
    assert stone_to_char(BLACK) == "X"
    assert stone_to_char(WHITE) == "O"


def test_render_board_returns_string_with_coordinates():
    board = Board()

    text = render_board(board)

    assert isinstance(text, str)
    for col in "ABCDEFGHIJKLMNO":
        assert col in text
    assert " 1" in text
    assert "15" in text


def test_render_board_shows_empty_black_and_white_stones():
    board = Board()
    board.place_stone(0, 0)  # black
    board.place_stone(7, 7)  # white

    text = render_board(board, highlight_last=False)

    assert "." in text
    assert "X" in text
    assert "O" in text


def test_render_board_highlights_last_move_without_breaking_output():
    board = Board()
    board.place_stone(0, 0)
    board.place_stone(14, 14)

    text = render_board(board, highlight_last=True)

    assert "(O)" in text
    assert "A" in text
    assert "O" in text
    assert "15" in text
