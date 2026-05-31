"""Candidate move generation for tactical search."""

from __future__ import annotations

from typing import Iterable, Optional

from game.board import BOARD_SIZE, EMPTY, Board
from game.encoder import action_to_index


def _occupied_points(board: Board) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            if board.grid[x][y] != EMPTY:
                points.append((x, y))
    return points


def _candidate_score(
    x: int, y: int, occupied: Iterable[tuple[int, int]], center: float
) -> tuple[float, int]:
    occupied_list = list(occupied)
    center_distance = abs(x - center) + abs(y - center)
    neighbor_pressure = 0.0
    nearest = 100.0
    for ox, oy in occupied_list:
        chebyshev = max(abs(x - ox), abs(y - oy))
        manhattan = abs(x - ox) + abs(y - oy)
        nearest = min(nearest, float(manhattan))
        if chebyshev <= 1:
            neighbor_pressure += 4.0
        elif chebyshev <= 2:
            neighbor_pressure += 1.5
        else:
            neighbor_pressure += 0.25 / chebyshev
    # Larger first; action index as stable tiebreaker in ascending order.
    score = neighbor_pressure - 0.15 * center_distance - 0.05 * nearest
    return score, -action_to_index(x, y)


def generate_candidate_moves(
    board: Board,
    radius: int = 2,
    max_candidates: Optional[int] = None,
    include_center: bool = True,
) -> list[int]:
    """Return legal action indices near existing stones.

    Empty boards return the center point first.  Non-empty boards only consider
    empty points inside a Chebyshev ``radius`` around existing stones, sorted by
    proximity to stones and board center.
    """
    if radius < 0:
        raise ValueError("radius must be non-negative")
    occupied = _occupied_points(board)
    center = (BOARD_SIZE - 1) / 2.0

    if not occupied:
        actions = [action_to_index(BOARD_SIZE // 2, BOARD_SIZE // 2)] if include_center else []
        return actions[:max_candidates] if max_candidates is not None else actions

    candidates: set[tuple[int, int]] = set()
    for ox, oy in occupied:
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                x, y = ox + dx, oy + dy
                if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
                    continue
                if board.grid[x][y] != EMPTY:
                    continue
                candidates.add((x, y))

    ordered = sorted(
        candidates,
        key=lambda pos: _candidate_score(pos[0], pos[1], occupied, center),
        reverse=True,
    )
    actions = [action_to_index(x, y) for x, y in ordered]
    if max_candidates is not None:
        return actions[: max(0, int(max_candidates))]
    return actions


__all__ = ["generate_candidate_moves"]
