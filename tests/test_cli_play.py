"""ui/cli_play.py 的单元测试。"""

from __future__ import annotations

import pytest

from game.board import BLACK, WHITE
from ui.cli_play import is_quit_command, parse_human_move, run_cli_game


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A1", 0),
        ("H8", 112),
        ("h8", 112),
        ("O15", 224),
    ],
)
def test_parse_human_move_coordinates(text, expected):
    assert parse_human_move(text) == expected


@pytest.mark.parametrize("text", ["", "P1", "A0", "A16", "AA1", "H", "8H"])
def test_parse_human_move_invalid_input_raises(text):
    with pytest.raises(ValueError):
        parse_human_move(text)


@pytest.mark.parametrize("text", ["q", "Q", "quit", " exit "])
def test_quit_commands_are_recognized(text):
    assert is_quit_command(text) is True


def test_run_cli_game_can_quit_without_blocking_input():
    class UnusedAI:
        def select_action(self, board):  # pragma: no cover - should not be called
            raise AssertionError("AI should not move when human quits immediately")

    outputs = []
    winner = run_cli_game(
        ai_player=UnusedAI(),
        human_color=BLACK,
        input_fn=lambda _prompt: "q",
        output_fn=outputs.append,
        max_moves=4,
    )

    assert winner is None
    assert any("退出" in line for line in outputs)


def test_run_cli_game_allows_ai_move_for_white_human_then_quit():
    class FirstMoveAI:
        def select_action(self, board):
            return parse_human_move("H8")

    inputs = iter(["q"])
    outputs = []

    winner = run_cli_game(
        ai_player=FirstMoveAI(),
        human_color=WHITE,
        input_fn=lambda _prompt: next(inputs),
        output_fn=outputs.append,
        max_moves=4,
    )

    assert winner is None
    assert any("AI 落子: H8" in line for line in outputs)
