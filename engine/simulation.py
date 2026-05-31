"""Fast, reversible board mutation helpers for tactical simulation."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Iterator

from game.board import BLACK, EMPTY, WHITE, Board
from game.encoder import index_to_action


_OPTIONAL_STATE_FIELDS = (
    "move_history",
    "winner",
    "_legal_moves_cache",
    "legal_moves_cache",
    "hash",
    "zobrist_hash",
)


def _snapshot_optional_state(board: Board) -> dict[str, object]:
    return {
        name: copy.deepcopy(getattr(board, name))
        for name in _OPTIONAL_STATE_FIELDS
        if hasattr(board, name)
    }


def _restore_optional_state(board: Board, snapshot: dict[str, object]) -> None:
    for name, value in snapshot.items():
        setattr(board, name, value)


@contextmanager
def temporary_stone(board: Board, x: int, y: int, color: int) -> Iterator[Board]:
    """Temporarily play ``color`` at ``(x, y)`` and restore on exit.

    The temporary state matches ``board.copy(); current_player=color;
    place_stone(x, y)`` but avoids copying the whole board.  Nested calls on
    different empty points are supported because every call saves and restores
    its own affected cell and public board state.
    """
    if color not in (BLACK, WHITE):
        raise ValueError(f"invalid stone color: {color}")
    if not board.in_bounds(x, y):
        raise ValueError(f"temporary move out of bounds: ({x}, {y})")
    if board.grid[x][y] != EMPTY:
        raise ValueError(f"temporary move occupied or illegal: ({x}, {y})")

    old_cell = board.grid[x][y]
    old_current_player = board.current_player
    old_last_move = board.last_move
    old_move_count = board.move_count
    optional_state = _snapshot_optional_state(board)

    try:
        board.grid[x][y] = color
        board.last_move = (x, y, color)
        board.move_count = old_move_count + 1
        board.current_player = -color
        yield board
    finally:
        board.grid[x][y] = old_cell
        board.current_player = old_current_player
        board.last_move = old_last_move
        board.move_count = old_move_count
        _restore_optional_state(board, optional_state)


@contextmanager
def temporary_move(board: Board, action: int, color: int) -> Iterator[Board]:
    """Temporarily play an action index and restore board state on exit."""
    x, y = index_to_action(int(action))
    with temporary_stone(board, x, y, color):
        yield board


__all__ = ["temporary_move", "temporary_stone"]
