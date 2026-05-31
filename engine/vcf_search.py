"""VCF (Victory by Continuous Fours) forced-win search.

A VCF mate is a sequence in which the attacker always makes a "four threat"
(a move that creates at least one immediate-five threat next ply), forcing the
defender into a unique reply, until the attacker either:

  * has two or more simultaneous immediate-five threats — defender can block
    at most one and the attacker wins on the very next ply, or
  * lands an immediate five somewhere on the way.

Public API
----------
- :func:`vcf_first_move` : returns the **first action_index** of a VCF mate
  sequence from the side-to-move, or ``None``. Pure-read against the board.
- :func:`find_vcf_attack_candidates` : enumerates moves that create at least one
  immediate-five threat next ply (the "four-creators").

Design notes
------------
- We DO NOT rebuild ``classify_move_threats`` results: speed matters; instead
  we use a small local 5-window scan that is O(4 dir × 5 offsets × 5 cells) per
  candidate.
- Forbidden rule support is honoured: black overline (>5) is never treated as
  a win; double-three / double-four forbidden are respected via the standard
  threats module before returning a VCF answer.
- The search is depth-bounded (``max_depth``) AND node-bounded
  (``node_budget``) so a runaway position cannot freeze the GUI.
- The board is mutated only via :func:`engine.simulation.temporary_move`, so
  any exception will leave the input board untouched.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Set, Tuple

from game.board import BLACK, BOARD_SIZE, EMPTY, WHITE, Board
from game.encoder import action_to_index, index_to_action

from .candidate_moves import generate_candidate_moves
from .simulation import temporary_stone
from .threats import find_immediate_winning_moves, is_forbidden_action


_DIRECTIONS: Tuple[Tuple[int, int], ...] = ((1, 0), (0, 1), (1, 1), (1, -1))


# ---------------------------------------------------------------------------
# fast local five-window helpers (work on a board where stones are already in)
# ---------------------------------------------------------------------------
def _has_win_window_through(board: Board, x: int, y: int, color: int) -> bool:
    """Is there a 5-window through ``(x, y)`` filled entirely with ``color``?

    ``(x, y)`` must already be ``color`` on the board.
    """
    n = BOARD_SIZE
    grid = board.grid
    for dx, dy in _DIRECTIONS:
        for k in range(-4, 1):
            ok = True
            for j in range(5):
                cx = x + (k + j) * dx
                cy = y + (k + j) * dy
                if not (0 <= cx < n and 0 <= cy < n):
                    ok = False
                    break
                if grid[cx][cy] != color:
                    ok = False
                    break
            if ok:
                return True
    return False


def _has_overline_through(board: Board, x: int, y: int, color: int) -> bool:
    """Is there a >5 consecutive run of ``color`` through ``(x, y)``?

    ``(x, y)`` must already be ``color`` on the board.
    """
    n = BOARD_SIZE
    grid = board.grid
    for dx, dy in _DIRECTIONS:
        count = 1
        cx, cy = x + dx, y + dy
        while 0 <= cx < n and 0 <= cy < n and grid[cx][cy] == color:
            count += 1
            cx += dx
            cy += dy
        cx, cy = x - dx, y - dy
        while 0 <= cx < n and 0 <= cy < n and grid[cx][cy] == color:
            count += 1
            cx -= dx
            cy -= dy
        if count > 5:
            return True
    return False


def _is_legal_for_color(
    board: Board, action: int, color: int, rule_mode: str
) -> bool:
    """Cell empty + in bounds + not forbidden for ``color``."""
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


def _is_winning_placement(
    board: Board, x: int, y: int, color: int, rule_mode: str
) -> bool:
    """Assuming ``(x, y)`` is just-placed as ``color``, is this a legal win?

    In ``forbidden`` mode for black, a >5 overline is NOT a win.
    """
    if not _has_win_window_through(board, x, y, color):
        return False
    if rule_mode == "forbidden" and color == BLACK and _has_overline_through(
        board, x, y, color
    ):
        return False
    return True


def _enumerate_win_cells_after(
    board: Board, x: int, y: int, color: int
) -> Set[int]:
    """After ``(x, y)`` is placed as ``color``, return cells where ``color``
    could place next ply to make a 5-in-a-row (including broken patterns).

    Uses 5-window scan through ``(x, y)``.
    """
    n = BOARD_SIZE
    grid = board.grid
    out: Set[int] = set()
    for dx, dy in _DIRECTIONS:
        for k in range(-4, 1):
            empty_cell = None
            empties = 0
            ok = True
            for j in range(5):
                cx = x + (k + j) * dx
                cy = y + (k + j) * dy
                if not (0 <= cx < n and 0 <= cy < n):
                    ok = False
                    break
                v = grid[cx][cy]
                if v == EMPTY:
                    empties += 1
                    empty_cell = (cx, cy)
                elif v != color:
                    ok = False
                    break
                if empties > 1:
                    ok = False
                    break
            if ok and empties == 1 and empty_cell is not None:
                out.add(action_to_index(empty_cell[0], empty_cell[1], n))
    return out


def _filter_legal_winning_cells(
    board: Board, cells: Iterable[int], color: int, rule_mode: str
) -> List[int]:
    """Among ``cells``, keep only those that are a legal AND winning placement
    for ``color`` under ``rule_mode``."""
    out: List[int] = []
    for action in cells:
        if not _is_legal_for_color(board, action, color, rule_mode):
            continue
        x, y = index_to_action(int(action), BOARD_SIZE)
        with temporary_stone(board, x, y, color):
            if _is_winning_placement(board, x, y, color, rule_mode):
                out.append(int(action))
    return sorted(out)


# ---------------------------------------------------------------------------
# attack-candidate enumeration
# ---------------------------------------------------------------------------
def find_vcf_attack_candidates(
    board: Board,
    color: int,
    rule_mode: str = "basic",
    candidate_radius: int = 2,
    max_candidates: int = 80,
) -> List[int]:
    """Moves that, once played, create at least one immediate-five threat."""
    raw = generate_candidate_moves(
        board, radius=candidate_radius, max_candidates=max_candidates
    )
    attackers: List[int] = []
    for action in raw:
        if not _is_legal_for_color(board, action, color, rule_mode):
            continue
        x, y = index_to_action(int(action), BOARD_SIZE)
        with temporary_stone(board, x, y, color):
            # Quick win? (creates length-5)
            if _is_winning_placement(board, x, y, color, rule_mode):
                attackers.append(int(action))
                continue
            # Or creates at least one next-ply win cell?
            win_cells = _enumerate_win_cells_after(board, x, y, color)
            if not win_cells:
                continue
            legal_win_cells = _filter_legal_winning_cells(
                board, win_cells, color, rule_mode
            )
            if legal_win_cells:
                attackers.append(int(action))
    return attackers


# ---------------------------------------------------------------------------
# core search
# ---------------------------------------------------------------------------
class _BudgetExhausted(Exception):
    """Raised when the VCF node budget runs out."""


def _take_node(state: dict) -> None:
    state["nodes"] += 1
    if state["nodes"] > state["budget"]:
        raise _BudgetExhausted
    # Yield the GIL every 512 nodes so the GUI stays responsive even during a
    # deep VCF search running on a background thread.
    if state["nodes"] & 127 == 0:
        import time

        time.sleep(0)


def _legal_opp_block_cells(
    board: Board, opp_threat_cells: Iterable[int], defender_color: int, rule_mode: str
) -> List[int]:
    """Defender (``defender_color``) cells that block one of opponent's win cells."""
    # In standard VCF, defender plays one of the opponent's win cells to neutralise.
    out: List[int] = []
    for action in opp_threat_cells:
        if _is_legal_for_color(board, action, defender_color, rule_mode):
            out.append(int(action))
    return sorted(out)


def _vcf_recurse(
    board: Board,
    color: int,
    depth: int,
    rule_mode: str,
    state: dict,
) -> Optional[int]:
    """Return the first attacker move of a VCF mate at this position, or None.

    ``depth`` is half-moves remaining for the attacker (so depth 1 only allows
    a one-shot immediate five; depth 3 allows one four + opp reply + final win;
    etc.).
    """
    _take_node(state)

    if depth <= 0:
        return None

    # ---- attacker forced to defend? Check if opponent has an immediate-five
    # threat *right now* (i.e., it would win on opponent's next turn unless we
    # block).
    opp_immediate = find_immediate_winning_moves(
        board, -color, rule_mode
    )
    if opp_immediate:
        if len(opp_immediate) >= 2:
            # Cannot defend simultaneously.
            return None
        defense = opp_immediate[0]
        if not _is_legal_for_color(board, defense, color, rule_mode):
            return None
        # Only valid attacker move is to play 'defense'. It must itself be a
        # four-threat for us; otherwise no VCF.
        x, y = index_to_action(int(defense), BOARD_SIZE)
        with temporary_stone(board, x, y, color):
            # Did we accidentally win outright?
            if _is_winning_placement(board, x, y, color, rule_mode):
                return int(defense)
            our_wins = _enumerate_win_cells_after(board, x, y, color)
            legal_wins = _filter_legal_winning_cells(
                board, our_wins, color, rule_mode
            )
            if not legal_wins:
                # Defense doesn't extend our attack; VCF dies.
                return None
            if len(legal_wins) >= 2:
                # We have multi-threat; opponent can block only one.
                return int(defense)
            # Single four: opponent must block our legal_wins[0]
            opp_reply = legal_wins[0]
            if not _is_legal_for_color(board, opp_reply, -color, rule_mode):
                # Opp can't play there — but if they don't, we win.
                return int(defense)
            ox, oy = index_to_action(int(opp_reply), BOARD_SIZE)
            with temporary_stone(board, ox, oy, -color):
                sub = _vcf_recurse(board, color, depth - 2, rule_mode, state)
                if sub is not None:
                    return int(defense)
            return None

    # ---- Direct immediate win for attacker?
    direct = find_immediate_winning_moves(board, color, rule_mode)
    if direct:
        return int(direct[0])

    if depth < 3:
        return None

    # ---- Enumerate attacker's four-creating candidates.
    candidates = find_vcf_attack_candidates(board, color, rule_mode)
    if not candidates:
        return None

    for attack in candidates:
        if not _is_legal_for_color(board, attack, color, rule_mode):
            continue
        x, y = index_to_action(int(attack), BOARD_SIZE)
        with temporary_stone(board, x, y, color):
            # Did this just create a 5 outright?
            if _is_winning_placement(board, x, y, color, rule_mode):
                return int(attack)

            our_wins = _enumerate_win_cells_after(board, x, y, color)
            legal_wins = _filter_legal_winning_cells(
                board, our_wins, color, rule_mode
            )
            if not legal_wins:
                # The candidate didn't really create a five-threat.
                continue

            if len(legal_wins) >= 2:
                # Multi-threat: opp can block at most one. Verify forbidden does
                # not nullify all our follow-up wins under any opp reply.
                # Conservative check: at least one of our legal_wins survives
                # ANY opp move. Since opp can only block one cell, len>=2 with
                # all cells distinct guarantees one survives. We sanity-check
                # with a tiny opp-reply scan when forbidden is involved.
                if rule_mode == "forbidden" and color == BLACK:
                    # Walk a few likely opp replies and confirm we always win.
                    safe = True
                    opp_candidates = list(set(legal_wins))[:4]
                    for opp_try in opp_candidates:
                        if not _is_legal_for_color(
                            board, opp_try, -color, rule_mode
                        ):
                            continue
                        ox, oy = index_to_action(int(opp_try), BOARD_SIZE)
                        with temporary_stone(board, ox, oy, -color):
                            remaining = _filter_legal_winning_cells(
                                board, legal_wins, color, rule_mode
                            )
                            remaining = [a for a in remaining if a != opp_try]
                            if not remaining:
                                safe = False
                                break
                    if safe:
                        return int(attack)
                else:
                    return int(attack)

            # Single four-threat: opp's forced reply is legal_wins[0].
            opp_reply = legal_wins[0]
            if not _is_legal_for_color(board, opp_reply, -color, rule_mode):
                # Opp cannot legally block; we win next ply.
                return int(attack)
            ox, oy = index_to_action(int(opp_reply), BOARD_SIZE)
            with temporary_stone(board, ox, oy, -color):
                sub = _vcf_recurse(board, color, depth - 2, rule_mode, state)
                if sub is not None:
                    return int(attack)
    return None


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def vcf_first_move(
    board: Board,
    color: int,
    max_depth: int = 9,
    rule_mode: str = "basic",
    node_budget: int = 20000,
) -> Optional[int]:
    """Return the first attacker move of a forced VCF mate, or ``None``.

    Parameters
    ----------
    board : Board
        Read-only from the caller's point of view. The board is mutated
        temporarily via :func:`temporary_stone` and always restored.
    color : int
        Side to move (BLACK / WHITE).
    max_depth : int
        Half-move depth budget. 9 ≈ up to 5 attacker moves + 4 forced replies.
    rule_mode : str
        ``"basic"`` or ``"forbidden"``.
    node_budget : int
        Hard cap on visited search nodes. Returns ``None`` if exhausted.

    Returns
    -------
    Optional[int]
        action_index of the first attacker move of a mate, or ``None``.
    """
    if color not in (BLACK, WHITE):
        return None
    if max_depth <= 0:
        return None
    state = {"nodes": 0, "budget": int(node_budget)}
    try:
        return _vcf_recurse(board, int(color), int(max_depth), rule_mode, state)
    except _BudgetExhausted:
        return None


def vcf_defends(
    board: Board,
    color: int,
    candidate: int,
    max_depth: int = 9,
    rule_mode: str = "basic",
    node_budget: int = 20000,
) -> bool:
    """After ``color`` plays ``candidate``, does opponent still have a VCF mate?

    Returns True if ``candidate`` defends against opponent's VCF (i.e.,
    opponent has no mate at depth ``max_depth`` after ``candidate``).
    Returns False if opponent's mate persists.
    """
    if not _is_legal_for_color(board, candidate, color, rule_mode):
        return False
    x, y = index_to_action(int(candidate), BOARD_SIZE)
    with temporary_stone(board, x, y, color):
        mate = vcf_first_move(
            board, -color, max_depth=max_depth,
            rule_mode=rule_mode, node_budget=node_budget,
        )
    return mate is None


__all__ = [
    "find_vcf_attack_candidates",
    "vcf_first_move",
    "vcf_defends",
]
