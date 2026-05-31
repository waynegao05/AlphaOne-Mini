"""Engineering VCT (Victory by Continuous Threats) search.

This is a bounded, testable VCT implementation for practical play.  It treats
direct wins and VCF as terminal winning threats, then searches open-three
threat chains by requiring that **every enumerated defensive reply** still
leaves the attacker with a continuation.

It is still not a formal proof of tournament-level strength by itself; the
search is depth- and node-bounded so it can be used from the GUI.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Set, Tuple

from game.board import BLACK, BOARD_SIZE, EMPTY, WHITE, Board
from game.encoder import action_to_index, index_to_action
from game.rules_forbidden import find_open_three_threats

from .candidate_moves import generate_candidate_moves
from .heuristic import evaluate_move_heuristic
from .simulation import temporary_stone
from .threats import (
    classify_move_threats,
    find_immediate_winning_moves,
    is_forbidden_action,
)
from .vcf_search import (
    find_vcf_attack_candidates,
    vcf_first_move,
)


class _BudgetExhausted(Exception):
    """Raised when the VCT node budget is exhausted."""


def _take_node(state: dict) -> None:
    state["nodes"] += 1
    if state["nodes"] > state["budget"]:
        raise _BudgetExhausted
    # Yield the GIL every 512 nodes so the GUI stays responsive even during a
    # deep VCT search running on a background thread.
    if state["nodes"] & 127 == 0:
        import time

        time.sleep(0)


def _board_key(board: Board) -> Tuple[Tuple[int, ...], ...]:
    return tuple(tuple(row) for row in board.grid)


def _is_legal_for_color(board: Board, action: int, color: int, rule_mode: str) -> bool:
    try:
        x, y = index_to_action(int(action), BOARD_SIZE)
    except ValueError:
        return False
    if not board.is_legal_move(x, y):
        return False
    if rule_mode == "forbidden" and color == BLACK and is_forbidden_action(
        board, int(action), color, rule_mode
    ):
        return False
    return True


def _legal_actions_from_points(
    board: Board,
    points: Iterable[tuple[int, int]],
    defender_color: int,
    rule_mode: str,
) -> List[int]:
    out: list[int] = []
    seen: set[int] = set()
    for x, y in points:
        if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
            continue
        action = action_to_index(x, y, BOARD_SIZE)
        if action in seen:
            continue
        if _is_legal_for_color(board, action, defender_color, rule_mode):
            seen.add(action)
            out.append(action)
    return sorted(out)


def _open_three_extension_cells(board: Board, attacker_color: int) -> Set[int]:
    cells: set[int] = set()
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            if board.grid[x][y] != attacker_color:
                continue
            for threat in find_open_three_threats(board, x, y, attacker_color):
                ex, ey = threat.extension_position
                if board.grid[ex][ey] == EMPTY:
                    cells.add(action_to_index(ex, ey, BOARD_SIZE))
    return cells


def find_vct_defense_moves(
    board: Board,
    attacker_color: int,
    rule_mode: str = "basic",
) -> List[int]:
    """Enumerate defender replies to the attacker's current forcing threats.

    Priority:
    - If attacker has immediate winning cells, those cells must be blocked.
    - Otherwise, block the extension cells of current open-three threats.

    The returned cells are legal for the defender and sorted for stable search.
    """
    if attacker_color not in (BLACK, WHITE):
        return []
    defender = -attacker_color
    wins = find_immediate_winning_moves(board, attacker_color, rule_mode)
    if wins:
        return [
            int(action)
            for action in wins
            if _is_legal_for_color(board, int(action), defender, rule_mode)
        ]
    return [
        int(action)
        for action in sorted(_open_three_extension_cells(board, attacker_color))
        if _is_legal_for_color(board, int(action), defender, rule_mode)
    ]


def _order_vct_candidates(
    board: Board,
    actions: Iterable[int],
    color: int,
    rule_mode: str,
    cache: dict[tuple[object, ...], set[str]],
) -> List[int]:
    ranked: list[tuple[int, float, int]] = []
    for action in actions:
        threats = classify_move_threats(
            board,
            int(action),
            color,
            rule_mode,
            cache=cache,
        )
        if "forbidden" in threats or "illegal" in threats:
            continue
        if "five" in threats:
            tier = 5
        elif "double_four" in threats or "open_four" in threats:
            tier = 4
        elif "blocked_four" in threats:
            tier = 3
        elif "double_three" in threats:
            tier = 2
        elif "open_three" in threats:
            tier = 1
        else:
            continue
        score = evaluate_move_heuristic(
            board,
            int(action),
            color,
            rule_mode,
            threat_cache=cache,
        )
        ranked.append((tier, score, int(action)))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [action for _, _, action in ranked]


def find_vct_attack_candidates(
    board: Board,
    color: int,
    rule_mode: str = "basic",
    candidate_radius: int = 2,
    max_candidates: int = 80,
    actions: Optional[Iterable[int]] = None,
    cache: Optional[dict[tuple[object, ...], set[str]]] = None,
) -> List[int]:
    """Return legal moves that create VCT-relevant threats."""
    if color not in (BLACK, WHITE):
        return []
    if rule_mode not in ("basic", "forbidden"):
        raise ValueError(f"unknown rule_mode: {rule_mode!r}")

    source = (
        list(actions)
        if actions is not None
        else generate_candidate_moves(
            board,
            radius=candidate_radius,
            max_candidates=max_candidates,
        )
    )
    threat_cache = cache if cache is not None else {}
    legal = [
        int(action)
        for action in source
        if _is_legal_for_color(board, int(action), color, rule_mode)
    ]
    return _order_vct_candidates(board, legal, color, rule_mode, threat_cache)


def _attacker_has_unavoidable_next_win(
    board: Board,
    color: int,
    rule_mode: str,
) -> bool:
    wins = find_immediate_winning_moves(board, color, rule_mode)
    return len(wins) >= 2


def _candidate_creates_double_three(
    board: Board,
    action: int,
    color: int,
    rule_mode: str,
    cache: dict[tuple[object, ...], set[str]],
) -> bool:
    threats = classify_move_threats(board, action, color, rule_mode, cache=cache)
    return "double_three" in threats and "forbidden" not in threats


def _vct_recurse(
    board: Board,
    color: int,
    depth: int,
    rule_mode: str,
    state: dict,
    candidate_radius: int,
    max_candidates: int,
) -> bool:
    _take_node(state)

    if depth <= 0:
        return False

    key = (color, depth, _board_key(board))
    if key in state["cache"]:
        return bool(state["cache"][key])

    def done(value: bool) -> bool:
        state["cache"][key] = bool(value)
        return bool(value)

    # If the attacker can win immediately, the threat chain succeeded.
    if find_immediate_winning_moves(board, color, rule_mode):
        return done(True)

    # VCF is a subset of VCT and is cheaper/more exact here.
    if vcf_first_move(
        board,
        color,
        max_depth=min(max(1, depth), 9),
        rule_mode=rule_mode,
        node_budget=max(1, state["budget"] - state["nodes"]),
    ) is not None:
        return done(True)

    if depth < 3:
        return done(False)

    # If the defender already has an immediate win, our open-three chain cannot
    # ignore it. The outer move selector handles direct blocks before VCT.
    if find_immediate_winning_moves(board, -color, rule_mode):
        return done(False)

    threat_cache: dict[tuple[object, ...], set[str]] = {}
    candidates = find_vct_attack_candidates(
        board,
        color,
        rule_mode=rule_mode,
        candidate_radius=candidate_radius,
        max_candidates=max_candidates,
        cache=threat_cache,
    )
    if not candidates:
        return done(False)

    for attack in candidates:
        if not _is_legal_for_color(board, attack, color, rule_mode):
            continue
        x, y = index_to_action(int(attack), BOARD_SIZE)
        with temporary_stone(board, x, y, color):
            if _attacker_has_unavoidable_next_win(board, color, rule_mode):
                return done(True)
            if find_immediate_winning_moves(board, -color, rule_mode):
                continue

            defenses = find_vct_defense_moves(board, color, rule_mode)
            if not defenses:
                continue
            all_defenses_fail = True
            for defense in defenses[: state["max_defenses"]]:
                if not _is_legal_for_color(board, defense, -color, rule_mode):
                    continue
                dx, dy = index_to_action(int(defense), BOARD_SIZE)
                with temporary_stone(board, dx, dy, -color):
                    if not _vct_recurse(
                        board,
                        color,
                        depth - 2,
                        rule_mode,
                        state,
                        candidate_radius,
                        max_candidates,
                    ):
                        all_defenses_fail = False
                        break
            if all_defenses_fail:
                return done(True)
    return done(False)


def vct_first_move(
    board: Board,
    color: int,
    max_depth: int = 7,
    rule_mode: str = "basic",
    node_budget: int = 5000,
    candidate_radius: int = 2,
    max_candidates: int = 80,
    max_defenses: int = 12,
) -> Optional[int]:
    """Return the first move of a bounded engineering VCT, or ``None``."""
    if color not in (BLACK, WHITE) or max_depth <= 0 or node_budget <= 0:
        return None
    if rule_mode not in ("basic", "forbidden"):
        raise ValueError(f"unknown rule_mode: {rule_mode!r}")

    direct = find_immediate_winning_moves(board, color, rule_mode)
    if direct:
        return int(direct[0])

    # The caller's immediate-block tier must handle this before VCT. Returning
    # None here prevents a visually attractive open-three from ignoring mate.
    if find_immediate_winning_moves(board, -color, rule_mode):
        return None

    mate = vcf_first_move(
        board,
        color,
        max_depth=min(max(1, int(max_depth)), 9),
        rule_mode=rule_mode,
        node_budget=int(node_budget),
    )
    if mate is not None:
        return int(mate)

    if max_depth < 3:
        return None

    state = {
        "nodes": 0,
        "budget": int(node_budget),
        "cache": {},
        "max_defenses": max(1, int(max_defenses)),
    }
    threat_cache: dict[tuple[object, ...], set[str]] = {}
    try:
        for attack in find_vct_attack_candidates(
            board,
            color,
            rule_mode=rule_mode,
            candidate_radius=candidate_radius,
            max_candidates=max_candidates,
            cache=threat_cache,
        ):
            _take_node(state)
            if not _is_legal_for_color(board, attack, color, rule_mode):
                continue
            x, y = index_to_action(int(attack), BOARD_SIZE)
            with temporary_stone(board, x, y, color):
                if _attacker_has_unavoidable_next_win(board, color, rule_mode):
                    return int(attack)
                defenses = find_vct_defense_moves(board, color, rule_mode)
                if not defenses:
                    continue
                proven = True
                for defense in defenses[: state["max_defenses"]]:
                    if not _is_legal_for_color(board, defense, -color, rule_mode):
                        continue
                    dx, dy = index_to_action(int(defense), BOARD_SIZE)
                    with temporary_stone(board, dx, dy, -color):
                        if not _vct_recurse(
                            board,
                            color,
                            int(max_depth) - 2,
                            rule_mode,
                            state,
                            candidate_radius,
                            max_candidates,
                        ):
                            proven = False
                            break
                if proven:
                    return int(attack)
    except _BudgetExhausted:
        return None
    return None


def vct_defends(
    board: Board,
    color: int,
    candidate: int,
    max_depth: int = 7,
    rule_mode: str = "basic",
    node_budget: int = 5000,
    candidate_radius: int = 2,
    max_candidates: int = 80,
) -> bool:
    """Whether ``candidate`` removes opponent's bounded VCT threat."""
    if not _is_legal_for_color(board, candidate, color, rule_mode):
        return False
    x, y = index_to_action(int(candidate), BOARD_SIZE)
    with temporary_stone(board, x, y, color):
        mate = vct_first_move(
            board,
            -color,
            max_depth=max_depth,
            rule_mode=rule_mode,
            node_budget=node_budget,
            candidate_radius=candidate_radius,
            max_candidates=max_candidates,
        )
    return mate is None


__all__ = [
    "find_vct_attack_candidates",
    "find_vct_defense_moves",
    "vct_defends",
    "vct_first_move",
]
