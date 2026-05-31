"""Smoke tests for the dependency-free standalone Tkinter Gomoku file."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STANDALONE = ROOT / "main_standalone_tkinter_play.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("standalone_gomoku", STANDALONE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_standalone_file_exists_and_uses_no_project_imports():
    assert STANDALONE.exists()
    text = STANDALONE.read_text(encoding="utf-8")
    forbidden_imports = ("from game.", "from engine.", "from model.", "from evaluate.")
    assert not any(item in text for item in forbidden_imports)


def test_standalone_ai_takes_immediate_win_and_blocks():
    module = _load_module()

    board = module.StandaloneBoard()
    for x in (3, 4, 5, 6):
        board.grid[x][7] = module.BLACK
    board.current_player = module.BLACK
    ai = module.StandaloneStrongAI(rule_mode="basic")
    win_action = ai.select_action(board)
    assert module.index_to_xy(win_action) in {(2, 7), (7, 7)}

    board = module.StandaloneBoard()
    for x in (3, 4, 5, 6):
        board.grid[x][7] = module.WHITE
    board.current_player = module.BLACK
    block_action = ai.select_action(board)
    assert module.index_to_xy(block_action) in {(2, 7), (7, 7)}


def test_standalone_ai_prefers_center_on_empty_board():
    module = _load_module()
    board = module.StandaloneBoard()
    ai = module.StandaloneStrongAI(rule_mode="basic")

    assert ai.select_action(board) == module.xy_to_index(7, 7)
