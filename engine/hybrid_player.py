"""Hybrid tactical fast path plus existing MCTS player fallback."""

from __future__ import annotations

from typing import Optional

from game.board import BOARD_SIZE, Board
from game.encoder import index_to_action

from .candidate_moves import generate_candidate_moves
from .heuristic import score_moves
from .tactical_player import TacticalPlayer
from .threats import (
    find_immediate_blocking_moves,
    find_immediate_winning_moves,
    find_open_four_moves,
    is_forbidden_action,
)


class HybridPlayer:
    """Use tactical forcing moves first, then delegate to MCTS."""

    def __init__(
        self,
        mcts_player=None,
        model=None,
        num_simulations: int = 50,
        c_puct: float = 5.0,
        device: str = "cpu",
        rule_mode: str = "basic",
        candidate_radius: int = 2,
        max_candidates: int = 80,
        name: str = "HybridPlayer",
    ) -> None:
        self.name = name
        self.rule_mode = rule_mode
        self.candidate_radius = int(candidate_radius)
        self.max_candidates = int(max_candidates)
        self.tactical = TacticalPlayer(
            name=f"{name}_tactical",
            rule_mode=rule_mode,
            candidate_radius=candidate_radius,
            max_candidates=max_candidates,
        )

        if mcts_player is None:
            if model is None:
                raise ValueError("mcts_player or model must be provided")
            from evaluate.players import ModelMCTSPlayer

            mcts_player = ModelMCTSPlayer(
                model=model,
                num_simulations=num_simulations,
                c_puct=c_puct,
                device=device,
                board_size=BOARD_SIZE,
                name=f"{name}_mcts",
            )
        self.mcts_player = mcts_player

    def _legal_for_color(self, board: Board, action: int, color: int) -> bool:
        try:
            x, y = index_to_action(int(action), BOARD_SIZE)
        except ValueError:
            return False
        if not board.is_legal_move(x, y):
            return False
        if self.rule_mode == "forbidden" and is_forbidden_action(
            board, int(action), color, self.rule_mode
        ):
            return False
        return True

    def _legal_actions(self, board: Board, actions: list[int], color: int) -> list[int]:
        return [action for action in actions if self._legal_for_color(board, action, color)]

    def _choose_tactical(self, board: Board, color: int) -> Optional[int]:
        tactical_groups = (
            find_immediate_winning_moves(board, color, self.rule_mode),
            find_immediate_blocking_moves(board, color, self.rule_mode),
            find_open_four_moves(board, color, self.rule_mode),
            find_open_four_moves(board, -color, self.rule_mode),
        )
        for actions in tactical_groups:
            legal = self._legal_actions(board, actions, color)
            if legal:
                return self.tactical._pick_best(board, legal, color)
        candidates = generate_candidate_moves(
            board,
            radius=self.candidate_radius,
            max_candidates=self.max_candidates,
        )
        legal_candidates = self._legal_actions(board, candidates, color)
        if legal_candidates:
            scored = score_moves(board, legal_candidates, color, self.rule_mode)
            if scored and scored[0][1] >= 20_000:
                return int(scored[0][0])
        return None

    def select_action(self, board: Board) -> Optional[int]:
        if not board.get_legal_moves():
            return None
        color = int(board.current_player)

        tactical_action = self._choose_tactical(board, color)
        if tactical_action is not None:
            return tactical_action

        try:
            action = self.mcts_player.select_action(board)
        except Exception:
            return self.tactical.select_action(board)
        if action is None:
            return self.tactical.select_action(board)
        action = int(action)
        if not self._legal_for_color(board, action, color):
            return self.tactical.select_action(board)
        return action

    def select_move(self, board: Board):
        action = self.select_action(board)
        if action is None:
            return None
        return index_to_action(action, BOARD_SIZE)


__all__ = ["HybridPlayer"]
