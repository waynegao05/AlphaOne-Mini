"""Generate tactical auxiliary labels from the current board state."""

from __future__ import annotations

import numpy as np

from engine.candidate_moves import generate_candidate_moves
from engine.heuristic import BLOCK_SCORES, OWN_SCORES, _positional_bonus
from engine.threats import ThreatSet, classify_move_threats, is_forbidden_action
from game.board import BLACK, BOARD_SIZE, EMPTY, WHITE, Board
from game.encoder import action_to_index, index_to_action


THREAT_CHANNELS = [
    "own_win_point",
    "opponent_win_point",
    "own_open_four",
    "opponent_open_four",
    "own_blocked_four",
    "opponent_blocked_four",
    "own_open_three",
    "opponent_open_three",
    "own_double_four",
    "opponent_double_four",
    "own_double_three",
    "opponent_double_three",
]


def _legal_actions(board: Board) -> list[int]:
    return [action_to_index(x, y, BOARD_SIZE) for x, y in board.get_legal_moves()]


def _candidate_actions(
    board: Board,
    candidate_radius: int = 2,
    max_candidates: int | None = None,
) -> list[int]:
    actions = generate_candidate_moves(
        board,
        radius=candidate_radius,
        max_candidates=max_candidates,
        include_center=True,
    )
    if actions:
        return actions
    return _legal_actions(board)


def _cached_threats(
    board: Board,
    action: int,
    color: int,
    rule_mode: str,
    cache: dict[tuple[object, ...], ThreatSet],
) -> ThreatSet:
    key = (int(action), int(color), str(rule_mode), True, True, True)
    legacy_key = (int(action), int(color), str(rule_mode))
    if key in cache:
        return set(cache[key])
    if legacy_key in cache:
        return set(cache[legacy_key])
    return classify_move_threats(board, action, color, rule_mode, cache=cache)


def _mark(labels: np.ndarray, channel_name: str, action: int) -> None:
    x, y = index_to_action(action, BOARD_SIZE)
    labels[THREAT_CHANNELS.index(channel_name), y, x] = 1.0


def generate_threat_label(
    board: Board,
    current_player: int,
    rule_mode: str = "basic",
    actions: list[int] | None = None,
    candidate_radius: int = 2,
    max_candidates: int | None = None,
    threat_cache: dict[tuple[object, ...], ThreatSet] | None = None,
) -> np.ndarray:
    """Return `[12, 15, 15]` tactical threat labels for legal empty points."""
    if current_player not in (BLACK, WHITE):
        raise ValueError("current_player must be BLACK(1) or WHITE(-1)")
    labels = np.zeros((len(THREAT_CHANNELS), BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    opponent = -current_player
    source = actions if actions is not None else _candidate_actions(board, candidate_radius, max_candidates)
    cache = threat_cache if threat_cache is not None else {}
    for action in source:
        own_threats = _cached_threats(board, action, current_player, rule_mode, cache)
        opp_threats = _cached_threats(board, action, opponent, rule_mode, cache)
        if "forbidden" not in own_threats:
            if "five" in own_threats:
                _mark(labels, "own_win_point", action)
            if "open_four" in own_threats:
                _mark(labels, "own_open_four", action)
            if "blocked_four" in own_threats:
                _mark(labels, "own_blocked_four", action)
            if "open_three" in own_threats:
                _mark(labels, "own_open_three", action)
            if "double_four" in own_threats:
                _mark(labels, "own_double_four", action)
            if "double_three" in own_threats:
                _mark(labels, "own_double_three", action)

        if "forbidden" not in opp_threats:
            if "five" in opp_threats:
                _mark(labels, "opponent_win_point", action)
            if "open_four" in opp_threats:
                _mark(labels, "opponent_open_four", action)
            if "blocked_four" in opp_threats:
                _mark(labels, "opponent_blocked_four", action)
            if "open_three" in opp_threats:
                _mark(labels, "opponent_open_three", action)
            if "double_four" in opp_threats:
                _mark(labels, "opponent_double_four", action)
            if "double_three" in opp_threats:
                _mark(labels, "opponent_double_three", action)
    return labels


def generate_forbidden_label(
    board: Board,
    current_player: int,
    rule_mode: str = "forbidden",
    actions: list[int] | None = None,
    candidate_radius: int = 2,
    max_candidates: int | None = None,
) -> np.ndarray:
    """Return `[1, 15, 15]` forbidden-action labels for black in forbidden mode."""
    labels = np.zeros((1, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    if rule_mode != "forbidden" or current_player != BLACK:
        return labels
    source = actions if actions is not None else _candidate_actions(board, candidate_radius, max_candidates)
    for action in source:
        if is_forbidden_action(board, action, BLACK, "forbidden"):
            x, y = index_to_action(action, BOARD_SIZE)
            labels[0, y, x] = 1.0
    return labels


def _score_from_threats(
    board: Board,
    action: int,
    own_threats: ThreatSet,
    opponent_threats: ThreatSet,
) -> float:
    if "illegal" in own_threats or "forbidden" in own_threats:
        return 0.0
    x, y = index_to_action(action, BOARD_SIZE)
    score = sum(value for name, value in OWN_SCORES.items() if name in own_threats)
    score += sum(
        value
        for name, value in BLOCK_SCORES.items()
        if name in (opponent_threats - {"forbidden", "overline_forbidden"})
    )
    score += _positional_bonus(board, x, y)
    return max(0.0, float(score))


def generate_tactical_score_label(
    board: Board,
    current_player: int,
    rule_mode: str = "basic",
    actions: list[int] | None = None,
    candidate_radius: int = 2,
    max_candidates: int | None = None,
    threat_cache: dict[tuple[object, ...], ThreatSet] | None = None,
) -> np.ndarray:
    """Return normalized non-negative heuristic scores as a flat `[225]` vector."""
    scores = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
    raw: list[tuple[int, float]] = []
    source = actions if actions is not None else _candidate_actions(board, candidate_radius, max_candidates)
    opponent = -current_player
    cache = threat_cache if threat_cache is not None else {}
    for action in source:
        x, y = index_to_action(action, BOARD_SIZE)
        if not board.is_legal_move(x, y):
            continue
        own_threats = _cached_threats(board, action, current_player, rule_mode, cache)
        opponent_threats = _cached_threats(board, action, opponent, rule_mode, cache)
        raw.append((action, _score_from_threats(board, action, own_threats, opponent_threats)))
    if not raw:
        return scores
    max_score = max(score for _, score in raw)
    if max_score <= 0:
        for action, _ in raw:
            scores[action] = 1.0
        scores /= np.float32(scores.sum())
        return scores
    for action, score in raw:
        scores[action] = np.float32(score / max_score)
    return scores


def build_auxiliary_labels(
    board: Board,
    current_player: int,
    rule_mode: str = "basic",
    candidate_radius: int = 2,
    max_candidates: int | None = None,
    actions: list[int] | None = None,
    threat_cache: dict[tuple[object, ...], ThreatSet] | None = None,
) -> dict[str, np.ndarray]:
    """Build all auxiliary labels without mutating ``board``."""
    source = actions if actions is not None else _candidate_actions(
        board, candidate_radius, max_candidates
    )
    cache = threat_cache if threat_cache is not None else {}
    return {
        "threat_labels": generate_threat_label(
            board,
            current_player,
            rule_mode,
            actions=source,
            threat_cache=cache,
        ),
        "forbidden_labels": generate_forbidden_label(
            board,
            current_player,
            rule_mode,
            actions=source,
        ),
        "tactical_scores": generate_tactical_score_label(
            board,
            current_player,
            rule_mode,
            actions=source,
            threat_cache=cache,
        ),
    }


__all__ = [
    "THREAT_CHANNELS",
    "generate_threat_label",
    "generate_forbidden_label",
    "generate_tactical_score_label",
    "build_auxiliary_labels",
]
