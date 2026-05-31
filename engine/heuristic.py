"""Explainable tactical move scoring."""

from __future__ import annotations

from typing import Iterable, List, Tuple

from game.board import BLACK, BOARD_SIZE, EMPTY, Board
from game.encoder import index_to_action

from .threats import classify_move_threats, is_forbidden_action


FORBIDDEN_SCORE = -1_000_000.0
ILLEGAL_SCORE = float("-inf")

OWN_SCORES = {
    "five": 100_000,
    "open_four": 50_000,
    "double_four": 40_000,
    "blocked_four": 20_000,
    "double_three": 12_000,
    "open_three": 8_000,
}

BLOCK_SCORES = {
    "five": 90_000,
    "open_four": 45_000,
    "double_four": 35_000,
    "blocked_four": 18_000,
    "double_three": 10_000,
    "open_three": 7_000,
}


def _positional_bonus(board: Board, x: int, y: int) -> float:
    center = (BOARD_SIZE - 1) / 2.0
    center_bonus = max(0.0, BOARD_SIZE - (abs(x - center) + abs(y - center)))
    neighbor_bonus = 0.0
    for nx in range(max(0, x - 2), min(BOARD_SIZE, x + 3)):
        for ny in range(max(0, y - 2), min(BOARD_SIZE, y + 3)):
            if nx == x and ny == y:
                continue
            if board.grid[nx][ny] != EMPTY:
                distance = max(abs(nx - x), abs(ny - y))
                neighbor_bonus += 4.0 if distance == 1 else 1.0
    return center_bonus * 3.0 + neighbor_bonus


def _score_threats(threats: set[str], scores: dict[str, int]) -> float:
    if "illegal" in threats:
        return ILLEGAL_SCORE
    return float(sum(value for name, value in scores.items() if name in threats))


def evaluate_move_heuristic(
    board: Board,
    action: int,
    color: int,
    rule_mode: str = "basic",
    threat_cache: dict[tuple[object, ...], set[str]] | None = None,
) -> float:
    """Return a stable tactical score for playing ``action`` as ``color``."""
    try:
        x, y = index_to_action(int(action), BOARD_SIZE)
    except ValueError:
        return ILLEGAL_SCORE
    if color not in (BLACK, -BLACK):
        return ILLEGAL_SCORE
    if not board.is_legal_move(x, y):
        return ILLEGAL_SCORE

    if rule_mode == "forbidden" and color == BLACK and is_forbidden_action(
        board, int(action), color, rule_mode
    ):
        return FORBIDDEN_SCORE

    own_threats = classify_move_threats(
        board, int(action), color, rule_mode, cache=threat_cache
    )
    opponent_threats = classify_move_threats(
        board, int(action), -color, rule_mode, cache=threat_cache
    )
    if "forbidden" in own_threats:
        return FORBIDDEN_SCORE

    score = _score_threats(own_threats, OWN_SCORES)
    score += _score_threats(opponent_threats - {"forbidden", "overline_forbidden"}, BLOCK_SCORES)
    score += _positional_bonus(board, x, y)
    return float(score)


def score_moves(
    board: Board,
    actions: Iterable[int],
    color: int,
    rule_mode: str = "basic",
    threat_cache: dict[tuple[object, ...], set[str]] | None = None,
) -> List[Tuple[int, float]]:
    """Score and sort actions by descending score, stable by action index."""
    scored = [
        (
            int(action),
            evaluate_move_heuristic(
                board,
                int(action),
                color,
                rule_mode,
                threat_cache=threat_cache,
            ),
        )
        for action in actions
    ]
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored


__all__ = [
    "FORBIDDEN_SCORE",
    "ILLEGAL_SCORE",
    "evaluate_move_heuristic",
    "score_moves",
]
