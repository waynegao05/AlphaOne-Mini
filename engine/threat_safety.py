"""One-ply tactical safety guard for practical mid-game defense.

This module is deliberately shallow. It does not try to prove a VCF/VCT line;
those searches live in ``vcf_search.py`` and ``vct_search.py``. Instead, it
answers a narrower question before MCTS is allowed to move:

    "If we play this candidate, how dangerous is the opponent's best reply?"

The guard is useful in dense practical games where the neural prior can prefer
a plausible-looking move that gives the opponent an immediate tactical
escalation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from game.board import BLACK, BOARD_SIZE, Board
from game.encoder import index_to_action

from .candidate_moves import generate_candidate_moves
from .heuristic import evaluate_move_heuristic, score_moves
from .simulation import temporary_move
from .threats import classify_move_threats, is_forbidden_action


RISK_SCORES = {
    "five": 1_000_000.0,
    "double_four": 220_000.0,
    "open_four": 180_000.0,
    "blocked_four": 45_000.0,
    "double_three": 35_000.0,
    "open_three": 12_000.0,
}


@dataclass(frozen=True)
class SafetyEvaluation:
    action: int
    own_score: float
    reply_risk: float
    combined_score: float
    opponent_best_reply: Optional[int]
    opponent_best_threats: tuple[str, ...]


def threat_risk_score(threats: Iterable[str]) -> float:
    names = set(threats)
    if "forbidden" in names:
        return 0.0
    return float(sum(score for name, score in RISK_SCORES.items() if name in names))


def _legal_actions(board: Board, actions: Iterable[int], color: int, rule_mode: str) -> list[int]:
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


def evaluate_opponent_reply_risk(
    board: Board,
    color: int,
    *,
    rule_mode: str = "basic",
    candidate_radius: int = 2,
    max_replies: int = 40,
) -> tuple[float, Optional[int], tuple[str, ...]]:
    """Return the opponent's best tactical reply risk on the current board."""

    opponent = -int(color)
    replies = _legal_actions(
        board,
        generate_candidate_moves(
            board,
            radius=candidate_radius,
            max_candidates=max_replies,
        ),
        opponent,
        rule_mode,
    )
    best_risk = 0.0
    best_reply: Optional[int] = None
    best_threats: tuple[str, ...] = ()
    for reply in replies:
        threats = classify_move_threats(board, reply, opponent, rule_mode)
        risk = threat_risk_score(threats)
        if risk > best_risk or (risk == best_risk and best_reply is not None and reply < best_reply):
            best_risk = risk
            best_reply = int(reply)
            best_threats = tuple(sorted(threats))
    return best_risk, best_reply, best_threats


def evaluate_move_safety(
    board: Board,
    action: int,
    color: int,
    *,
    rule_mode: str = "basic",
    candidate_radius: int = 2,
    max_replies: int = 40,
) -> SafetyEvaluation:
    """Score one candidate by own heuristic minus opponent reply risk."""

    own_score = evaluate_move_heuristic(board, int(action), int(color), rule_mode)
    try:
        with temporary_move(board, int(action), int(color)):
            reply_risk, reply, threats = evaluate_opponent_reply_risk(
                board,
                int(color),
                rule_mode=rule_mode,
                candidate_radius=candidate_radius,
                max_replies=max_replies,
            )
    except ValueError:
        reply_risk, reply, threats = float("inf"), None, ("illegal",)
    combined = float(own_score - 0.65 * reply_risk)
    return SafetyEvaluation(
        action=int(action),
        own_score=float(own_score),
        reply_risk=float(reply_risk),
        combined_score=combined,
        opponent_best_reply=reply,
        opponent_best_threats=threats,
    )


def select_threat_safe_move(
    board: Board,
    color: int,
    *,
    rule_mode: str = "basic",
    candidate_radius: int = 2,
    max_candidates: int = 80,
    evaluate_top_n: int = 36,
    max_replies: int = 40,
    trigger_risk: float = RISK_SCORES["open_three"],
) -> Optional[SafetyEvaluation]:
    """Choose a defensive minimax move when opponent pressure is non-trivial.

    Returns ``None`` in quiet positions so MCTS can still handle normal play.
    """

    color = int(color)
    candidates = _legal_actions(
        board,
        generate_candidate_moves(
            board,
            radius=candidate_radius,
            max_candidates=max_candidates,
        ),
        color,
        rule_mode,
    )
    if not candidates:
        return None

    current_risk, _reply, _threats = evaluate_opponent_reply_risk(
        board,
        color,
        rule_mode=rule_mode,
        candidate_radius=candidate_radius,
        max_replies=max_replies,
    )
    if current_risk < trigger_risk:
        return None

    # Start from the explainable heuristic ordering, but run the more expensive
    # reply-risk check only on the strongest subset.
    ordered = [action for action, _score in score_moves(board, candidates, color, rule_mode)]
    evaluations = [
        evaluate_move_safety(
            board,
            action,
            color,
            rule_mode=rule_mode,
            candidate_radius=candidate_radius,
            max_replies=max_replies,
        )
        for action in ordered[: max(1, int(evaluate_top_n))]
    ]
    evaluations.sort(
        key=lambda item: (
            item.reply_risk,
            -item.combined_score,
            -item.own_score,
            item.action,
        )
    )
    best = evaluations[0]

    # Only take over MCTS when the chosen move actually improves the opponent's
    # best reply risk or when the current board is already highly tactical.
    if best.reply_risk < current_risk or current_risk >= RISK_SCORES["blocked_four"]:
        return best
    return None


__all__ = [
    "RISK_SCORES",
    "SafetyEvaluation",
    "evaluate_move_safety",
    "evaluate_opponent_reply_risk",
    "select_threat_safe_move",
    "threat_risk_score",
]
