from __future__ import annotations

import sys
from pathlib import Path

import pytest

from game.board import Board
from game.encoder import action_to_index


def test_external_ai_adapter_loads_same_directory_dependency(tmp_path: Path):
    from engine.external_ai_adapter import ExternalAIAdapter

    ai_dir = tmp_path / "external"
    ai_dir.mkdir()
    (ai_dir / "graphics.py").write_text("ACTION = 112\n", encoding="utf-8")
    (ai_dir / "AI.py").write_text(
        "import graphics\n"
        "def select_action(board):\n"
        "    return graphics.ACTION\n",
        encoding="utf-8",
    )
    before = list(sys.path)

    adapter = ExternalAIAdapter(ai_dir / "AI.py", rule_mode="basic")

    assert adapter.select_action(Board()) == action_to_index(7, 7)
    assert sys.path == before


def test_external_ai_adapter_reports_missing_dependency(tmp_path: Path):
    from engine.external_ai_adapter import ExternalAIAdapter

    ai_dir = tmp_path / "external"
    ai_dir.mkdir()
    (ai_dir / "AI.py").write_text(
        "import missing_graphics_helper\n"
        "def select_action(board):\n"
        "    return 112\n",
        encoding="utf-8",
    )

    with pytest.raises(ModuleNotFoundError) as excinfo:
        ExternalAIAdapter(ai_dir / "AI.py", rule_mode="basic")

    assert "missing_graphics_helper" in str(excinfo.value)


def test_external_ai_adapter_supports_legacy_ai1_graphics_program(tmp_path: Path):
    from engine.external_ai_adapter import ExternalAIAdapter

    ai_dir = tmp_path / "legacy"
    ai_dir.mkdir()
    (ai_dir / "graphics.py").write_text(
        '"""Simple object oriented graphics library by John Zelle"""\n'
        "import tkinter\n"
        "class GraphWin:\n"
        "    def __init__(self, *args, **kwargs):\n"
        "        raise RuntimeError('real graphics should have been stubbed')\n",
        encoding="utf-8",
    )
    (ai_dir / "AI.py").write_text(
        "from graphics import *\n"
        "win = GraphWin('legacy', 10, 10)\n"
        "num = [[0 for _ in range(16)] for _ in range(16)]\n"
        "ai = 1\n"
        "go_first = 1\n"
        "start = 1\n"
        "def go(x, y):\n"
        "    raise RuntimeError('drawing go should have been patched')\n"
        "def AI1():\n"
        "    return go(7, 7)\n",
        encoding="utf-8",
    )

    adapter = ExternalAIAdapter(ai_dir / "AI.py", rule_mode="basic")

    assert adapter.select_action(Board()) == action_to_index(7, 7)
    assert adapter.decision_reason == "external_ai"
