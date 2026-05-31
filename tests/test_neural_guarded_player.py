from __future__ import annotations

import numpy as np
import torch

from game.board import BLACK, WHITE, Board
from game.encoder import action_to_index, index_to_action


class FakePolicyModel(torch.nn.Module):
    def __init__(self, preferred_action: int | None = None, flat: bool = False):
        super().__init__()
        self.preferred_action = preferred_action
        self.flat = flat

    def forward(self, x):
        batch = x.shape[0]
        logits = torch.zeros((batch, 225), dtype=torch.float32, device=x.device)
        if self.flat:
            return logits, torch.zeros((batch, 1), dtype=torch.float32, device=x.device)
        logits -= 5.0
        if self.preferred_action is not None:
            logits[:, int(self.preferred_action)] = 5.0
        return logits, torch.zeros((batch, 1), dtype=torch.float32, device=x.device)


class FixedFallback:
    def __init__(self, action: int):
        self.action = int(action)
        self.name = "FixedFallback"
        self.calls = 0

    def select_action(self, board):
        self.calls += 1
        return self.action


def _place(board: Board, stones):
    for x, y, color in stones:
        board.grid[x][y] = color
    board.move_count = len(stones)


def test_neural_guarded_player_takes_immediate_win():
    from engine.neural_guarded_player import NeuralGuardedPlayer

    board = Board()
    _place(board, [(5, 7, BLACK), (6, 7, BLACK), (7, 7, BLACK), (8, 7, BLACK)])
    board.current_player = BLACK
    player = NeuralGuardedPlayer(v2_model=FakePolicyModel(action_to_index(0, 0)), device="cpu")

    action = player.select_action(board)

    assert action in {action_to_index(4, 7), action_to_index(9, 7)}
    assert player.decision_reason == "immediate_win"


def test_neural_guarded_player_blocks_opponent_immediate_win():
    from engine.neural_guarded_player import NeuralGuardedPlayer

    board = Board()
    _place(board, [(5, 7, WHITE), (6, 7, WHITE), (7, 7, WHITE), (8, 7, WHITE)])
    board.current_player = BLACK
    player = NeuralGuardedPlayer(v2_model=FakePolicyModel(action_to_index(0, 0)), device="cpu")

    action = player.select_action(board)

    assert action in {action_to_index(4, 7), action_to_index(9, 7)}
    assert player.decision_reason == "immediate_block"


def test_neural_guarded_player_empty_board_prefers_center():
    from engine.neural_guarded_player import NeuralGuardedPlayer

    player = NeuralGuardedPlayer(v2_model=FakePolicyModel(action_to_index(7, 7)), device="cpu")

    assert player.select_action(Board()) == action_to_index(7, 7)
    assert player.decision_reason in {"v2_policy", "hybrid_fallback"}


def test_neural_guarded_player_does_not_return_occupied_point():
    from engine.neural_guarded_player import NeuralGuardedPlayer

    board = Board()
    board.grid[7][7] = BLACK
    board.move_count = 1
    board.current_player = WHITE
    fallback_action = action_to_index(8, 8)
    player = NeuralGuardedPlayer(
        v2_model=FakePolicyModel(action_to_index(7, 7)),
        hybrid_player=FixedFallback(fallback_action),
        device="cpu",
    )

    action = player.select_action(board)

    assert action == fallback_action
    assert player.decision_reason in {"high_entropy_fallback", "illegal_neural_fallback"}
    x, y = index_to_action(action)
    assert board.is_legal_move(x, y)


def test_neural_guarded_player_filters_forbidden_action():
    from engine.neural_guarded_player import NeuralGuardedPlayer

    board = Board()
    for x, y in [(6, 7), (8, 7), (7, 6), (7, 8)]:
        board.grid[x][y] = BLACK
    board.move_count = 4
    board.current_player = BLACK
    forbidden_center = action_to_index(7, 7)
    safe = action_to_index(3, 3)
    player = NeuralGuardedPlayer(
        v2_model=FakePolicyModel(forbidden_center),
        hybrid_player=FixedFallback(safe),
        rule_mode="forbidden",
        device="cpu",
    )

    action = player.select_action(board)

    assert action == safe
    assert player.decision_reason in {"high_entropy_fallback", "forbidden_filtered", "illegal_neural_fallback"}


def test_neural_guarded_player_hybrid_fallback_can_trigger_on_high_entropy():
    from engine.neural_guarded_player import NeuralGuardedPlayer

    fallback = FixedFallback(action_to_index(7, 7))
    player = NeuralGuardedPlayer(
        v2_model=FakePolicyModel(flat=True),
        hybrid_player=fallback,
        entropy_threshold=1.0,
        device="cpu",
    )

    assert player.select_action(Board()) == action_to_index(7, 7)
    assert player.decision_reason == "high_entropy_fallback"
    assert fallback.calls == 1


def test_neural_guarded_player_records_decision_context():
    from engine.neural_guarded_player import NeuralGuardedPlayer

    player = NeuralGuardedPlayer(v2_model=FakePolicyModel(action_to_index(7, 7)), device="cpu")
    action = player.select_action(Board())

    assert player.decision_reason
    assert player.last_decision["action"] == action
    assert "reason" in player.last_decision


def test_neural_guarded_player_can_log_detailed_decisions():
    from engine.neural_guarded_player import NeuralGuardedPlayer

    player = NeuralGuardedPlayer(
        v2_model=FakePolicyModel(action_to_index(7, 7)),
        device="cpu",
        enable_decision_log=True,
    )

    action = player.select_action(Board())

    assert action == action_to_index(7, 7)
    assert len(player.decision_log) == 1
    entry = player.decision_log[0]
    assert entry["move_index"] == 0
    assert entry["current_player"] == BLACK
    assert entry["selected_action"] == action
    assert entry["decision_reason"] in {"v2_policy", "hybrid_fallback"}
    assert "guardrail_candidate_actions" in entry
    assert "tactical_specialist_top5" in entry
    assert "v2_top5" in entry
    assert "hybrid_action" in entry
    assert "final_action_source" in entry
    assert "policy_entropy" in entry
    assert "top1_top2_margin" in entry
    assert entry["whether_action_legal"] is True


def test_neural_guarded_player_records_high_entropy_fallback_reason():
    from engine.neural_guarded_player import NeuralGuardedPlayer

    fallback = FixedFallback(action_to_index(7, 7))
    player = NeuralGuardedPlayer(
        v2_model=FakePolicyModel(flat=True),
        hybrid_player=fallback,
        entropy_threshold=1.0,
        device="cpu",
        enable_decision_log=True,
    )

    assert player.select_action(Board()) == action_to_index(7, 7)
    assert player.decision_reason == "high_entropy_fallback"
    assert player.decision_log[-1]["whether_fallback_used"] is True
    assert player.decision_log[-1]["hybrid_action"] == action_to_index(7, 7)


def test_neural_guarded_player_vs_random_arena_smoke():
    from engine.neural_guarded_player import NeuralGuardedPlayer
    from evaluate.arena import Arena
    from evaluate.players import RandomPlayer

    player = NeuralGuardedPlayer(
        v2_model=FakePolicyModel(action_to_index(7, 7)),
        hybrid_player=FixedFallback(action_to_index(7, 7)),
        device="cpu",
        name="NeuralGuardedTest",
    )
    arena = Arena(player, RandomPlayer(seed=1), max_moves=12)
    result = arena.play_one_game()

    assert result.winner in {-1, 0, 1}
    assert all(0 <= action < 225 for action in result.moves)
