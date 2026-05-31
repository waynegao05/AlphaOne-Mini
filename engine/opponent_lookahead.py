"""Candidate-pruned opponent lookahead for AlphaOne-Mini.

This is not a full-width game-tree search. Full Gomoku branching is too large
for a responsive Tkinter player. Instead we:

1. Generate local candidate moves.
2. Rank them with the existing tactical heuristic.
3. Convert opponent move scores into a probability-like distribution.
4. Search only the most likely / most dangerous branches for 3-4 plies.

That gives practical "see a few moves ahead" behavior without freezing the UI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from game.board import BLACK, BOARD_SIZE, Board
from game.encoder import index_to_action

from .candidate_moves import generate_candidate_moves
from .heuristic import evaluate_move_heuristic, score_moves
from .simulation import temporary_move
from .threat_safety import RISK_SCORES, threat_risk_score
from .threats import classify_move_threats, is_forbidden_action


WIN_SCORE = 1_000_000.0


@dataclass(frozen=True)
class PredictedMove:
    action: int
    score: float
    probability: float
    threats: tuple[str, ...]


@dataclass(frozen=True)
class LookaheadResult:
    action: int
    score: float
    principal_variation: tuple[int, ...]
    opponent_top_moves: tuple[PredictedMove, ...]
    reason: str


def _legal_actions(
    board: Board,
    color: int,
    *,
    rule_mode: str,
    candidate_radius: int,
    max_candidates: int,
) -> list[int]:
    actions = generate_candidate_moves(
        board,
        radius=candidate_radius,
        max_candidates=max_candidates,
    )
    legal: list[int] = []
    for action in actions:
        try:
            x, y = index_to_action(int(action), BOARD_SIZE)
        except ValueError:
            continue
        if not board.is_legal_move(x, y):
            continue
        if rule_mode == "forbidden" and color == BLACK and is_forbidden_action(
            board, int(action), color, rule_mode
        ):
            continue
        legal.append(int(action))
    return legal


def _terminal_move_score(board: Board, action: int, color: int, rule_mode: str) -> Optional[float]:
    threats = classify_move_threats(
        board,
        int(action),
        int(color),
        rule_mode,
        include_double_threats=False,
        include_open_three=False,
        include_four_threats=False,
    )
    if "five" in threats and "forbidden" not in threats:
        return WIN_SCORE
    return None


def predict_likely_moves(
    board: Board,
    color: int,
    *,
    rule_mode: str = "basic",
    candidate_radius: int = 2,
    max_candidates: int = 40,
    top_k: int = 8,
    temperature: float = 18_000.0,
) -> list[PredictedMove]:
    """Predict likely moves for ``color`` from heuristic scores.

    The returned probability is not learned from the external AI; it is a
    calibrated tactical prior: high-scoring tactical moves get most mass.
    """

    actions = _legal_actions(
        board,
        int(color),
        rule_mode=rule_mode,
        candidate_radius=candidate_radius,
        max_candidates=max_candidates,
    )
    if not actions:
        return []

    scored = score_moves(board, actions, int(color), rule_mode)[: max(1, int(top_k))]
    raw_scores: list[float] = []
    threats_by_action: dict[int, tuple[str, ...]] = {}
    for action, heuristic_score in scored:
        threats = classify_move_threats(board, int(action), int(color), rule_mode)
        risk_bonus = threat_risk_score(threats)
        score = float(heuristic_score + 0.35 * risk_bonus)
        raw_scores.append(score)
        threats_by_action[int(action)] = tuple(sorted(threats))

    scale = max(1.0, float(temperature))
    max_score = max(raw_scores)
    weights = [math.exp((score - max_score) / scale) for score in raw_scores]
    total = sum(weights) or 1.0
    return [
        PredictedMove(
            action=int(action),
            score=float(score),
            probability=float(weight / total),
            threats=threats_by_action[int(action)],
        )
        for (action, _), score, weight in zip(scored, raw_scores, weights)
    ]


def _static_eval(
    board: Board,
    root_color: int,
    *,
    rule_mode: str,
    candidate_radius: int,
    max_candidates: int,
) -> float:
    eval_limit = min(int(max_candidates), 16)
    own_actions = _legal_actions(
        board,
        root_color,
        rule_mode=rule_mode,
        candidate_radius=candidate_radius,
        max_candidates=eval_limit,
    )
    opp_actions = _legal_actions(
        board,
        -root_color,
        rule_mode=rule_mode,
        candidate_radius=candidate_radius,
        max_candidates=eval_limit,
    )
    own_best = score_moves(board, own_actions, root_color, rule_mode)[:1]
    opp_best = score_moves(board, opp_actions, -root_color, rule_mode)[:1]
    own_score = own_best[0][1] if own_best else 0.0
    opp_score = opp_best[0][1] if opp_best else 0.0
    return float(own_score - 0.92 * opp_score)


def _search(
    board: Board,
    to_move: int,
    root_color: int,
    depth: int,
    *,
    rule_mode: str,
    candidate_radius: int,
    max_candidates: int,
    branch_factor: int,
    alpha: float,
    beta: float,
) -> tuple[float, tuple[int, ...]]:
    if depth <= 0:
        return (
            _static_eval(
                board,
                root_color,
                rule_mode=rule_mode,
                candidate_radius=candidate_radius,
                max_candidates=max_candidates,
            ),
            (),
        )

    predictions = predict_likely_moves(
        board,
        to_move,
        rule_mode=rule_mode,
        candidate_radius=candidate_radius,
        max_candidates=max_candidates,
        top_k=branch_factor,
    )
    if not predictions:
        return (
            _static_eval(
                board,
                root_color,
                rule_mode=rule_mode,
                candidate_radius=candidate_radius,
                max_candidates=max_candidates,
            ),
            (),
        )

    maximizing = to_move == root_color
    best_score = float("-inf") if maximizing else float("inf")
    best_line: tuple[int, ...] = ()

    for predicted in predictions:
        terminal = _terminal_move_score(board, predicted.action, to_move, rule_mode)
        if terminal is not None:
            score = terminal if to_move == root_color else -terminal
            line = (predicted.action,)
        else:
            with temporary_move(board, predicted.action, to_move):
                child_score, child_line = _search(
                    board,
                    -to_move,
                    root_color,
                    depth - 1,
                    rule_mode=rule_mode,
                    candidate_radius=candidate_radius,
                    max_candidates=max_candidates,
                    branch_factor=branch_factor,
                    alpha=alpha,
                    beta=beta,
                )
            score = child_score
            line = (predicted.action, *child_line)

        if maximizing:
            if score > best_score:
                best_score, best_line = score, line
            alpha = max(alpha, best_score)
            if alpha >= beta:
                break
        else:
            if score < best_score:
                best_score, best_line = score, line
            beta = min(beta, best_score)
            if alpha >= beta:
                break

    return float(best_score), best_line


def select_lookahead_move(
    board: Board,
    color: int,
    *,
    rule_mode: str = "basic",
    depth: int = 4,
    branch_factor: int = 3,
    candidate_radius: int = 2,
    max_candidates: int = 24,
    trigger_risk: float = RISK_SCORES["open_three"],
) -> Optional[LookaheadResult]:
    """Return a move selected by 3-4 ply opponent modeling, or ``None``.

    Quiet positions intentionally return ``None`` so the neural/MCTS layer keeps
    handling normal play. The lookahead takes over when the opponent's likely
    replies include meaningful tactical pressure.
    """

    color = int(color)
    opponent_predictions = predict_likely_moves(
        board,
        -color,
        rule_mode=rule_mode,
        candidate_radius=candidate_radius,
        max_candidates=max_candidates,
        top_k=branch_factor,
    )
    if not opponent_predictions:
        return None
    opponent_top_risk = max(threat_risk_score(move.threats) for move in opponent_predictions)
    if opponent_top_risk < trigger_risk:
        return None

    actions = _legal_actions(
        board,
        color,
        rule_mode=rule_mode,
        candidate_radius=candidate_radius,
        max_candidates=max_candidates,
    )
    if not actions:
        return None

    root_candidates = score_moves(board, actions, color, rule_mode)[:branch_factor]
    best_score = float("-inf")
    best_action: Optional[int] = None
    best_line: tuple[int, ...] = ()

    for action, heuristic_score in root_candidates:
        terminal = _terminal_move_score(board, action, color, rule_mode)
        if terminal is not None:
            score = terminal
            line = (int(action),)
        else:
            with temporary_move(board, int(action), color):
                child_score, child_line = _search(
                    board,
                    -color,
                    color,
                    max(0, int(depth) - 1),
                    rule_mode=rule_mode,
                    candidate_radius=candidate_radius,
                    max_candidates=max_candidates,
                    branch_factor=branch_factor,
                    alpha=float("-inf"),
                    beta=float("inf"),
                )
            score = float(child_score + 0.03 * heuristic_score)
            line = (int(action), *child_line)

        if score > best_score or (score == best_score and (best_action is None or action < best_action)):
            best_score = score
            best_action = int(action)
            best_line = line

    if best_action is None:
        return None

    reason = f"lookahead:depth={int(depth)}"
    if best_line:
        reason += f":pv={len(best_line)}"
    return LookaheadResult(
        action=best_action,
        score=float(best_score),
        principal_variation=best_line,
        opponent_top_moves=tuple(opponent_predictions),
        reason=reason,
    )


__all__ = [
    "LookaheadResult",
    "PredictedMove",
    "predict_likely_moves",
    "select_lookahead_move",
]
