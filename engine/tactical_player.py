"""Rule-based tactical player for Gomoku."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import random
from typing import Optional

from game.board import BOARD_SIZE, Board
from game.encoder import action_to_index, index_to_action

from .candidate_moves import generate_candidate_moves
from .heuristic import score_moves
from .threats import (
    ThreatSet,
    find_immediate_blocking_moves,
    find_immediate_winning_moves,
    find_open_four_moves,
    is_forbidden_action,
)


@dataclass
class TacticalDecision:
    action: Optional[int]
    color: int
    candidates: list[int]
    threat_cache: dict[tuple[object, ...], ThreatSet]


class TacticalPlayer:
    """A deterministic tactical player that returns action indices."""

    def __init__(
        self,
        name: str = "TacticalPlayer",
        color: Optional[int] = None,
        rule_mode: str = "basic",
        candidate_radius: int = 2,
        max_candidates: int = 80,
        random_tie_break: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.name = name
        self.color = color
        self.rule_mode = rule_mode
        self.candidate_radius = int(candidate_radius)
        self.max_candidates = int(max_candidates)
        self.random_tie_break = bool(random_tie_break)
        self._rng = random.Random(seed)

    def _current_color(self, board: Board) -> int:
        return int(self.color if self.color is not None else board.current_player)

    def _is_legal_for_self(self, board: Board, action: int, color: int) -> bool:
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

    def _filter_legal(self, board: Board, actions: list[int], color: int) -> list[int]:
        return [int(action) for action in actions if self._is_legal_for_self(board, int(action), color)]

    def _pick_best(
        self,
        board: Board,
        actions: list[int],
        color: int,
        threat_cache: dict[tuple[object, ...], ThreatSet] | None = None,
    ) -> int:
        scored = score_moves(
            board,
            actions,
            color,
            self.rule_mode,
            threat_cache=threat_cache,
        )
        if not scored:
            raise RuntimeError("no candidate actions to choose from")
        best_score = scored[0][1]
        ties = [action for action, score in scored if score == best_score]
        if self.random_tie_break and len(ties) > 1:
            return int(self._rng.choice(ties))
        return int(ties[0])

    def _fallback_first_legal(self, board: Board, color: int) -> Optional[int]:
        for x, y in board.get_legal_moves():
            action = action_to_index(x, y, BOARD_SIZE)
            if self._is_legal_for_self(board, action, color):
                return action
        return None

    def _call_threat_finder(
        self,
        finder,
        board: Board,
        color: int,
        candidates: list[int],
        threat_cache: dict[tuple[object, ...], ThreatSet],
    ) -> list[int]:
        kwargs = {"actions": candidates}
        if "cache" in inspect.signature(finder).parameters:
            kwargs["cache"] = threat_cache
        return finder(board, color, self.rule_mode, **kwargs)

    def select_action_with_context(self, board: Board) -> TacticalDecision:
        threat_cache: dict[tuple[object, ...], ThreatSet] = {}
        color = self._current_color(board)
        if not board.get_legal_moves():
            return TacticalDecision(None, color, [], threat_cache)

        if board.move_count == 0:
            center = action_to_index(BOARD_SIZE // 2, BOARD_SIZE // 2, BOARD_SIZE)
            if self._is_legal_for_self(board, center, color):
                return TacticalDecision(center, color, [center], threat_cache)

        candidates = generate_candidate_moves(
            board,
            radius=self.candidate_radius,
            max_candidates=self.max_candidates,
        )
        if not candidates:
            fallback = self._fallback_first_legal(board, color)
            return TacticalDecision(fallback, color, [], threat_cache)

        wins = self._filter_legal(
            board,
            self._call_threat_finder(
                find_immediate_winning_moves,
                board,
                color,
                candidates,
                threat_cache,
            ),
            color,
        )
        if wins:
            return TacticalDecision(
                self._pick_best(board, wins, color, threat_cache),
                color,
                candidates,
                threat_cache,
            )

        blocks = self._filter_legal(
            board,
            self._call_threat_finder(
                find_immediate_blocking_moves,
                board,
                color,
                candidates,
                threat_cache,
            ),
            color,
        )
        if blocks:
            return TacticalDecision(
                self._pick_best(board, blocks, color, threat_cache),
                color,
                candidates,
                threat_cache,
            )

        own_open_four = self._filter_legal(
            board,
            self._call_threat_finder(
                find_open_four_moves,
                board,
                color,
                candidates,
                threat_cache,
            ),
            color,
        )
        if own_open_four:
            return TacticalDecision(
                self._pick_best(board, own_open_four, color, threat_cache),
                color,
                candidates,
                threat_cache,
            )

        opponent_open_four = self._filter_legal(
            board,
            self._call_threat_finder(
                find_open_four_moves,
                board,
                -color,
                candidates,
                threat_cache,
            ),
            color,
        )
        if opponent_open_four:
            return TacticalDecision(
                self._pick_best(board, opponent_open_four, color, threat_cache),
                color,
                candidates,
                threat_cache,
            )

        candidates = self._filter_legal(board, candidates, color)
        if candidates:
            return TacticalDecision(
                self._pick_best(board, candidates, color, threat_cache),
                color,
                candidates,
                threat_cache,
            )

        return TacticalDecision(
            self._fallback_first_legal(board, color), color, candidates, threat_cache
        )

    def select_action(self, board: Board) -> Optional[int]:
        return self.select_action_with_context(board).action

    def select_move(self, board: Board):
        action = self.select_action(board)
        if action is None:
            return None
        return index_to_action(action, BOARD_SIZE)


__all__ = ["TacticalDecision", "TacticalPlayer"]
