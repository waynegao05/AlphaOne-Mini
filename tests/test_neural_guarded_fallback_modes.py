from __future__ import annotations

import torch

from game.board import BLACK, Board
from game.encoder import action_to_index


class FlatPolicyModel(torch.nn.Module):
    def forward(self, x):
        return (
            torch.zeros((x.shape[0], 225), dtype=torch.float32, device=x.device),
            torch.zeros((x.shape[0], 1), dtype=torch.float32, device=x.device),
        )


class PreferredPolicyModel(torch.nn.Module):
    def __init__(self, action: int):
        super().__init__()
        self.action = int(action)

    def forward(self, x):
        logits = torch.full((x.shape[0], 225), -5.0, dtype=torch.float32, device=x.device)
        logits[:, self.action] = 5.0
        return logits, torch.zeros((x.shape[0], 1), dtype=torch.float32, device=x.device)


class FixedFallback:
    def __init__(self, action: int):
        self.action = int(action)
        self.calls = 0
        self.name = "FixedFallback"

    def select_action(self, board):
        self.calls += 1
        return self.action


def test_fallback_mode_off_does_not_call_hybrid_on_high_entropy():
    from engine.neural_guarded_player import NeuralGuardedPlayer

    fallback = FixedFallback(action_to_index(7, 7))
    player = NeuralGuardedPlayer(
        v2_model=FlatPolicyModel(),
        hybrid_player=fallback,
        entropy_threshold=1.0,
        fallback_mode="off",
        device="cpu",
        enable_decision_log=True,
    )

    action = player.select_action(Board())

    assert fallback.calls == 0
    assert action is not None
    assert player.decision_reason == "v2_policy"
    assert player.decision_log[-1]["fallback_mode"] == "off"


def test_conservative_fallback_disables_high_entropy_fallback():
    from engine.neural_guarded_player import NeuralGuardedPlayer

    fallback = FixedFallback(action_to_index(7, 7))
    player = NeuralGuardedPlayer(
        v2_model=FlatPolicyModel(),
        hybrid_player=fallback,
        entropy_threshold=1.0,
        fallback_mode="conservative",
        device="cpu",
    )

    player.select_action(Board())

    assert fallback.calls == 0
    assert player.decision_reason == "v2_policy"


def test_conservative_fallback_disables_low_margin_fallback():
    from engine.neural_guarded_player import NeuralGuardedPlayer

    fallback = FixedFallback(action_to_index(7, 7))
    player = NeuralGuardedPlayer(
        v2_model=FlatPolicyModel(),
        hybrid_player=fallback,
        margin_threshold=0.99,
        fallback_mode="conservative",
        device="cpu",
    )

    player.select_action(Board())

    assert fallback.calls == 0
    assert player.decision_reason == "v2_policy"


def test_normal_fallback_keeps_high_entropy_behavior():
    from engine.neural_guarded_player import NeuralGuardedPlayer

    fallback = FixedFallback(action_to_index(7, 7))
    player = NeuralGuardedPlayer(
        v2_model=FlatPolicyModel(),
        hybrid_player=fallback,
        entropy_threshold=1.0,
        fallback_mode="normal",
        device="cpu",
    )

    assert player.select_action(Board()) == action_to_index(7, 7)
    assert fallback.calls == 1
    assert player.decision_reason == "high_entropy_fallback"


def test_aggressive_fallback_can_trigger_uncertainty_fallback():
    from engine.neural_guarded_player import NeuralGuardedPlayer

    fallback = FixedFallback(action_to_index(7, 7))
    player = NeuralGuardedPlayer(
        v2_model=FlatPolicyModel(),
        hybrid_player=fallback,
        fallback_mode="aggressive",
        device="cpu",
    )

    assert player.select_action(Board()) == action_to_index(7, 7)
    assert fallback.calls == 1
    assert player.decision_reason == "high_entropy_fallback"


def test_conservative_fallback_still_filters_forbidden_action():
    from engine.neural_guarded_player import NeuralGuardedPlayer

    board = Board()
    for x, y in [(6, 7), (8, 7), (7, 6), (7, 8)]:
        board.grid[x][y] = BLACK
    board.move_count = 4
    board.current_player = BLACK
    forbidden_center = action_to_index(7, 7)
    fallback = FixedFallback(action_to_index(3, 3))
    player = NeuralGuardedPlayer(
        v2_model=PreferredPolicyModel(forbidden_center),
        hybrid_player=fallback,
        fallback_mode="conservative",
        rule_mode="forbidden",
        device="cpu",
        enable_decision_log=True,
    )

    action = player.select_action(board)

    assert action != forbidden_center
    assert player.decision_log[-1]["whether_forbidden_filtered"] is True
