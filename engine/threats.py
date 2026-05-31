"""Tactical threat recognition for Gomoku."""

from __future__ import annotations

from typing import Iterable, Optional, Set

from game.board import BLACK, BOARD_SIZE, EMPTY, WHITE, Board
from game.encoder import action_to_index, index_to_action
from game.rules_basic import check_winner
from game.rules_forbidden import (
    DIRECTIONS,
    get_game_result_forbidden,
    is_double_four,
    is_double_three,
)
from .simulation import temporary_move


ThreatSet = Set[str]


def _in_bounds(x: int, y: int) -> bool:
    return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE


def _all_legal_actions(board: Board) -> list[int]:
    return [action_to_index(x, y) for x, y in board.get_legal_moves()]


def _line_positions(
    x: int, y: int, dx: int, dy: int, start_offset: int, length: int
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (x + (start_offset + i) * dx, y + (start_offset + i) * dy)
        for i in range(length)
    )


def _count_contiguous(
    board: Board, x: int, y: int, dx: int, dy: int, color: int
) -> int:
    count = 0
    nx, ny = x, y
    while _in_bounds(nx, ny) and board.grid[nx][ny] == color:
        count += 1
        nx += dx
        ny += dy
    return count


def _contiguous_line_length(
    board: Board, x: int, y: int, dx: int, dy: int, color: int
) -> int:
    return (
        _count_contiguous(board, x, y, dx, dy, color)
        + _count_contiguous(board, x - dx, y - dy, -dx, -dy, color)
    )


def _open_ends_for_run(
    board: Board,
    start: tuple[int, int],
    end: tuple[int, int],
    dx: int,
    dy: int,
) -> int:
    before = (start[0] - dx, start[1] - dy)
    after = (end[0] + dx, end[1] + dy)
    opens = 0
    if _in_bounds(*before) and board.grid[before[0]][before[1]] == EMPTY:
        opens += 1
    if _in_bounds(*after) and board.grid[after[0]][after[1]] == EMPTY:
        opens += 1
    return opens


def _has_contiguous_threat(
    board: Board,
    x: int,
    y: int,
    color: int,
    length: int,
    required_open_ends: int,
) -> bool:
    for dx, dy in DIRECTIONS:
        for start_offset in range(-(length - 1), 1):
            positions = _line_positions(x, y, dx, dy, start_offset, length)
            if (x, y) not in positions:
                continue
            if any(not _in_bounds(px, py) for px, py in positions):
                continue
            if any(board.grid[px][py] != color for px, py in positions):
                continue
            start, end = positions[0], positions[-1]
            if _open_ends_for_run(board, start, end, dx, dy) >= required_open_ends:
                return True
    return False


def _has_open_four(board: Board, x: int, y: int, color: int) -> bool:
    return _has_contiguous_threat(board, x, y, color, 4, 2)


def _has_blocked_four(board: Board, x: int, y: int, color: int) -> bool:
    for dx, dy in DIRECTIONS:
        for start_offset in range(-3, 1):
            positions = _line_positions(x, y, dx, dy, start_offset, 4)
            if (x, y) not in positions:
                continue
            if any(not _in_bounds(px, py) for px, py in positions):
                continue
            if any(board.grid[px][py] != color for px, py in positions):
                continue
            start, end = positions[0], positions[-1]
            if _open_ends_for_run(board, start, end, dx, dy) == 1:
                return True
    return False


def _has_open_three(board: Board, x: int, y: int, color: int) -> bool:
    if _has_contiguous_threat(board, x, y, color, 3, 2):
        return True
    return is_double_three(board, x, y, color) or bool(
        _count_open_three_directions(board, x, y, color)
    )


def _count_open_three_directions(board: Board, x: int, y: int, color: int) -> int:
    from game.rules_forbidden import count_open_three_directions

    return count_open_three_directions(board, x, y, color)


def is_forbidden_action(board: Board, action: int, color: int, rule_mode: str) -> bool:
    """Whether ``action`` is forbidden for ``color`` under ``rule_mode``."""
    if rule_mode != "forbidden" or color != BLACK:
        return False
    try:
        with temporary_move(board, action, color):
            result = get_game_result_forbidden(board, board.last_move)
            return bool(result.forbidden)
    except ValueError:
        return True


def classify_move_threats(
    board: Board,
    action: int,
    color: int,
    rule_mode: str = "basic",
    cache: Optional[dict[tuple[object, ...], ThreatSet]] = None,
    include_double_threats: bool = True,
    include_open_three: bool = True,
    include_four_threats: bool = True,
) -> ThreatSet:
    """Classify tactical effects if ``color`` plays ``action``."""
    if rule_mode not in ("basic", "forbidden"):
        raise ValueError(f"unknown rule_mode: {rule_mode!r}")
    key = (
        int(action),
        int(color),
        str(rule_mode),
        bool(include_double_threats),
        bool(include_open_three),
        bool(include_four_threats),
    )
    if cache is not None and key in cache:
        return set(cache[key])

    def done(result: ThreatSet) -> ThreatSet:
        if cache is not None:
            cache[key] = set(result)
        return result

    try:
        x, y = index_to_action(action)
    except ValueError:
        return done({"illegal"})
    if color not in (BLACK, WHITE) or not board.is_legal_move(x, y):
        return done({"illegal"})

    with temporary_move(board, action, color):
        threats: ThreatSet = set()
        last_move = board.last_move

        if rule_mode == "forbidden":
            result = get_game_result_forbidden(board, last_move)
            if result.forbidden:
                threats.add("forbidden")
                if result.reason == "black_overline_forbidden":
                    threats.add("overline_forbidden")
                elif result.reason == "black_double_four_forbidden" and include_double_threats:
                    threats.update({"double_four", "open_four"})
                elif result.reason == "black_double_three_forbidden" and include_double_threats:
                    threats.update({"double_three", "open_three"})
            if result.is_over and result.winner == color and not result.forbidden:
                threats.add("five")
            if "five" in threats:
                return done(threats)
        elif check_winner(board, last_move) == color:
            threats.add("five")

        if not (color == BLACK and rule_mode == "forbidden") and include_double_threats:
            if is_double_four(board, x, y, color):
                threats.add("double_four")
            if is_double_three(board, x, y, color):
                threats.add("double_three")

        if "forbidden" in threats:
            return done(threats)

        if include_four_threats:
            if _has_open_four(board, x, y, color):
                threats.add("open_four")
            if _has_blocked_four(board, x, y, color):
                threats.add("blocked_four")
        if include_open_three and _has_open_three(board, x, y, color):
            threats.add("open_three")

        return done(threats)


def classify_move_threat(
    board: Board,
    action: int,
    color: int,
    rule_mode: str = "basic",
    cache: Optional[dict[tuple[object, ...], ThreatSet]] = None,
    include_double_threats: bool = True,
    include_open_three: bool = True,
    include_four_threats: bool = True,
) -> ThreatSet:
    """Backward-compatible singular alias for ``classify_move_threats``."""
    return classify_move_threats(
        board,
        action,
        color,
        rule_mode,
        cache=cache,
        include_double_threats=include_double_threats,
        include_open_three=include_open_three,
        include_four_threats=include_four_threats,
    )


def find_open_four_moves(
    board: Board,
    color: int,
    rule_mode: str = "basic",
    actions: Optional[Iterable[int]] = None,
    cache: Optional[dict[tuple[object, ...], ThreatSet]] = None,
) -> list[int]:
    source = list(actions) if actions is not None else _all_legal_actions(board)
    moves = []
    for action in source:
        threats = classify_move_threats(
            board,
            action,
            color,
            rule_mode,
            include_double_threats=False,
            include_open_three=False,
            cache=cache,
        )
        if "open_four" in threats and "forbidden" not in threats:
            moves.append(action)
    return sorted(moves)


def find_blocked_four_moves(
    board: Board,
    color: int,
    rule_mode: str = "basic",
    actions: Optional[Iterable[int]] = None,
    cache: Optional[dict[tuple[object, ...], ThreatSet]] = None,
) -> list[int]:
    source = list(actions) if actions is not None else _all_legal_actions(board)
    moves = []
    for action in source:
        threats = classify_move_threats(
            board,
            action,
            color,
            rule_mode,
            include_double_threats=False,
            include_open_three=False,
            cache=cache,
        )
        if "blocked_four" in threats and "forbidden" not in threats:
            moves.append(action)
    return sorted(moves)


def find_open_three_moves(
    board: Board,
    color: int,
    rule_mode: str = "basic",
    actions: Optional[Iterable[int]] = None,
    cache: Optional[dict[tuple[object, ...], ThreatSet]] = None,
) -> list[int]:
    source = list(actions) if actions is not None else _all_legal_actions(board)
    moves = []
    for action in source:
        threats = classify_move_threats(
            board,
            action,
            color,
            rule_mode,
            include_double_threats=False,
            cache=cache,
        )
        if "open_three" in threats and "forbidden" not in threats:
            moves.append(action)
    return sorted(moves)


def find_immediate_winning_moves(
    board: Board,
    color: int,
    rule_mode: str = "basic",
    actions: Optional[Iterable[int]] = None,
    cache: Optional[dict[tuple[object, ...], ThreatSet]] = None,
) -> list[int]:
    """Return moves where ``color`` wins immediately."""
    source = list(actions) if actions is not None else _all_legal_actions(board)
    wins: list[int] = []
    for action in source:
        threats = classify_move_threats(
            board,
            action,
            color,
            rule_mode,
            include_double_threats=False,
            include_open_three=False,
            include_four_threats=False,
            cache=cache,
        )
        if "five" in threats and "forbidden" not in threats:
            wins.append(action)
    return sorted(wins)


def find_immediate_blocking_moves(
    board: Board,
    color: int,
    rule_mode: str = "basic",
    actions: Optional[Iterable[int]] = None,
    cache: Optional[dict[tuple[object, ...], ThreatSet]] = None,
) -> list[int]:
    """Return moves where ``color`` blocks opponent's immediate win."""
    opponent = -color
    source = list(actions) if actions is not None else _all_legal_actions(board)
    opponent_wins = set(
        find_immediate_winning_moves(
            board, opponent, rule_mode, actions=source, cache=cache
        )
    )
    return sorted(action for action in source if action in opponent_wins)


__all__ = [
    "classify_move_threats",
    "classify_move_threat",
    "find_immediate_winning_moves",
    "find_immediate_blocking_moves",
    "find_open_four_moves",
    "find_blocked_four_moves",
    "find_open_three_moves",
    "is_forbidden_action",
]
