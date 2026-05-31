from __future__ import annotations

from pathlib import Path

from game.board import Board
from game.encoder import action_to_index


def _write_ai(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "AI.py"
    path.write_text(source, encoding="utf-8")
    return path


def test_external_ai_adapter_accepts_select_action(tmp_path):
    from engine.external_ai_adapter import ExternalAIAdapter

    path = _write_ai(
        tmp_path,
        "def select_action(board):\n"
        "    return 112\n",
    )
    board = Board()
    adapter = ExternalAIAdapter(path, rule_mode="basic")

    assert adapter.select_action(board) == action_to_index(7, 7)
    assert adapter.decision_reason == "external_ai"


def test_external_ai_adapter_accepts_get_move_tuple(tmp_path):
    from engine.external_ai_adapter import ExternalAIAdapter

    path = _write_ai(
        tmp_path,
        "def get_move(board, color):\n"
        "    return (7, 7)\n",
    )
    adapter = ExternalAIAdapter(path, rule_mode="basic")

    assert adapter.select_action(Board()) == action_to_index(7, 7)


def test_external_ai_adapter_falls_back_on_illegal_action(tmp_path):
    from engine.external_ai_adapter import ExternalAIAdapter

    path = _write_ai(
        tmp_path,
        "def select_action(board):\n"
        "    return 112\n",
    )
    board = Board()
    board.place_stone(7, 7)
    adapter = ExternalAIAdapter(path, rule_mode="basic")

    action = adapter.select_action(board)

    assert action != action_to_index(7, 7)
    x = action % 15
    y = action // 15
    assert board.is_legal_move(x, y)
    assert adapter.decision_reason == "external_illegal_fallback"


def test_external_ai_adapter_falls_back_on_exception(tmp_path):
    from engine.external_ai_adapter import ExternalAIAdapter

    path = _write_ai(
        tmp_path,
        "def select_action(board):\n"
        "    raise RuntimeError('boom')\n",
    )
    board = Board()
    adapter = ExternalAIAdapter(path, rule_mode="basic")

    action = adapter.select_action(board)

    assert isinstance(action, int)
    assert adapter.decision_reason.startswith("external_error_fallback")
