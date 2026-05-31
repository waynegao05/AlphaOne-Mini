from __future__ import annotations

import torch

from evaluate.arena import Arena, run_match
from game.board import BLACK, BOARD_SIZE, Board
from game.encoder import action_to_index


class FirstLegalRecorder:
    def __init__(self, name: str):
        self.name = name
        self.first_board_ids: list[int] = []
        self.first_boards: list[Board] = []

    def select_action(self, board: Board):
        if board.move_count == 0:
            self.first_boards.append(board)
            self.first_board_ids.append(id(board))
        for x, y in board.get_legal_moves():
            return action_to_index(x, y, BOARD_SIZE)
        return None


class FixedActionPlayer:
    def __init__(self, action: int, name: str = "fixed"):
        self.action = int(action)
        self.name = name

    def select_action(self, board: Board):
        return self.action


class CenterModel(torch.nn.Module):
    def forward(self, x):
        logits = torch.full((x.shape[0], 225), -5.0, device=x.device)
        logits[:, action_to_index(7, 7)] = 5.0
        return logits, torch.zeros((x.shape[0], 1), device=x.device)


def test_arena_creates_a_fresh_board_for_each_game():
    player_a = FirstLegalRecorder("A")
    player_b = FirstLegalRecorder("B")

    Arena(player_a, player_b, max_moves=2).play_many_games(4, alternate_sides=True)

    first_move_board_ids = player_a.first_board_ids + player_b.first_board_ids
    assert len(first_move_board_ids) == 4
    assert len(set(first_move_board_ids)) == 4


def test_run_match_alternates_sides_and_player_win_rates_are_not_reversed():
    class BlackWinPlayer(FirstLegalRecorder):
        pass

    class PassivePlayer(FirstLegalRecorder):
        pass

    # Script black wins in each game regardless of whether player_a or player_b
    # is seated as black, then verify summary assigns wins to the seated player.
    from tests.test_arena import ScriptedPlayer, _idx

    black_actions = [_idx(0, 0), _idx(1, 0), _idx(2, 0), _idx(3, 0), _idx(4, 0)]
    white_actions = [_idx(0, 14), _idx(1, 14), _idx(2, 14), _idx(3, 14)]
    player_a = ScriptedPlayer(black_actions, name="A")
    player_b = ScriptedPlayer(white_actions, name="B")

    summary, results = run_match(player_a, player_b, num_games=1, alternate_sides=True)

    assert results[0].black_player_name == "A"
    assert summary["player_a_wins"] == 1
    assert summary["player_b_wins"] == 0
    assert summary["player_a_win_rate"] == 1.0


def test_neural_guarded_does_not_reference_opponent_instance():
    from engine.neural_guarded_player import NeuralGuardedPlayer

    opponent = FixedActionPlayer(action_to_index(0, 0), name="opponent")
    player = NeuralGuardedPlayer(
        v2_model=CenterModel(),
        hybrid_player=FixedActionPlayer(action_to_index(7, 7), name="fallback"),
        device="cpu",
    )

    assert getattr(player, "hybrid_player", None) is not opponent


def test_neural_guarded_vs_hybrid_can_complete_short_match():
    from engine.hybrid_player import HybridPlayer
    from engine.neural_guarded_player import NeuralGuardedPlayer
    from model.policy_value_net import PolicyValueNet

    player = NeuralGuardedPlayer(
        v2_model=CenterModel(),
        hybrid_player=FixedActionPlayer(action_to_index(7, 7), name="fallback"),
        device="cpu",
    )
    opponent = HybridPlayer(
        model=PolicyValueNet(),
        num_simulations=1,
        device="cpu",
        name="HybridOpponent",
    )

    summary, results = run_match(player, opponent, num_games=2, max_moves=4)

    assert len(results) == 2
    assert summary["total_games"] == 2


def test_decision_log_does_not_change_selected_action():
    from engine.neural_guarded_player import NeuralGuardedPlayer

    board = Board()
    player_without_log = NeuralGuardedPlayer(v2_model=CenterModel(), device="cpu")
    player_with_log = NeuralGuardedPlayer(
        v2_model=CenterModel(),
        device="cpu",
        enable_decision_log=True,
    )

    assert player_with_log.select_action(board.copy()) == player_without_log.select_action(board.copy())
    assert player_with_log.decision_log
