"""Tests for optional candidate-move pruning in MCTS."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from game.board import BOARD_SIZE, Board  # noqa: E402
from game.encoder import action_to_index  # noqa: E402
from mcts.mcts import MCTS  # noqa: E402


class UniformNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x):  # type: ignore[override]
        batch = x.shape[0]
        logits = torch.zeros(batch, BOARD_SIZE * BOARD_SIZE, device=x.device)
        value = torch.zeros(batch, 1, device=x.device)
        return logits, value


def test_candidate_mcts_prunes_expansion_without_changing_default_behavior():
    board = Board()
    board.place_stone(7, 7)
    board.place_stone(8, 8)

    full = MCTS(UniformNet(), num_simulations=1)
    full.run(board)

    candidate = MCTS(
        UniformNet(),
        num_simulations=1,
        use_candidate_moves=True,
        candidate_radius=1,
    )
    candidate.run(board)

    assert len(full._last_root.children) == BOARD_SIZE * BOARD_SIZE - 2
    assert 0 < len(candidate._last_root.children) < len(full._last_root.children)
    assert action_to_index(7, 7) not in candidate._last_root.children
    assert action_to_index(8, 8) not in candidate._last_root.children


def test_candidate_mcts_empty_board_keeps_center_available():
    board = Board()
    mcts = MCTS(
        UniformNet(),
        num_simulations=1,
        use_candidate_moves=True,
        candidate_radius=1,
    )

    mcts.run(board)

    assert set(mcts._last_root.children.keys()) == {action_to_index(7, 7)}
