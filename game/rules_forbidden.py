"""禁手规则框架：当前实现黑方长连、四四与三三禁手。

本模块与 :mod:`game.rules_basic` 分离：

- basic 模式仍然是五连或以上都判胜。
- forbidden 模式当前加入黑方长连禁手、四四禁手与三三禁手。
- 本阶段不实现指定开局、三手交换、五手 N 打。

禁手模式约定：

- 黑方某方向精确五连，黑胜。
- 黑方长连(同一方向连续 6 子或以上)且没有任何方向精确五连，判禁手，白胜。
- 黑方同一步同时形成精确五连和长连时，精确五连优先，黑胜。
- 黑方没有精确五连时，形成两个或以上独立方向的“四”判四四禁手，白胜。
- 黑方没有精确五连、长连或四四时，形成两个或以上独立方向的活三判三三禁手，白胜。
- 白方没有长连禁手，白方五连或以上都判白胜。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .board import BLACK, BOARD_SIZE, EMPTY, WHITE, Board


DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))


@dataclass(frozen=True)
class GameRuleResult:
    """禁手规则判断结果。

    ``winner`` 取值：
    - ``BLACK`` / ``WHITE``：对应一方胜。
    - ``0``：平局。
    - ``None``：对局尚未结束。
    """

    is_over: bool
    winner: Optional[int]
    reason: str
    forbidden: bool = False


@dataclass(frozen=True)
class FourThreat:
    """一个可补成精确五连的“四”窗口。"""

    direction: Tuple[int, int]
    empty_position: Tuple[int, int]
    window_positions: Tuple[Tuple[int, int], ...]


@dataclass(frozen=True)
class OpenThreeThreat:
    """A same-direction open-three threat created by the last move."""

    direction: Tuple[int, int]
    extension_position: Tuple[int, int]
    pattern_positions: Tuple[Tuple[int, int], ...]
    threat_type: str


def count_continuous_stones(
    board: Board, x: int, y: int, dx: int, dy: int, color: int
) -> int:
    """从 ``(x, y)`` 出发，不含自身，沿 ``(dx, dy)`` 统计同色连续棋子数。"""
    if dx == 0 and dy == 0:
        raise ValueError("(dx, dy) 不能同时为 0")

    count = 0
    nx, ny = x + dx, y + dy
    while (
        0 <= nx < BOARD_SIZE
        and 0 <= ny < BOARD_SIZE
        and board.grid[nx][ny] == color
    ):
        count += 1
        nx += dx
        ny += dy
    return count


def count_line(board: Board, x: int, y: int, dx: int, dy: int, color: int) -> int:
    """统计以 ``(x, y)`` 为穿过点，某一方向上的连续同色总长度。"""
    if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
        return 0
    if color == EMPTY or board.grid[x][y] != color:
        return 0
    return (
        1
        + count_continuous_stones(board, x, y, dx, dy, color)
        + count_continuous_stones(board, x, y, -dx, -dy, color)
    )


def has_exact_five(board: Board, x: int, y: int, color: int) -> bool:
    """是否存在某一方向恰好 5 连。"""
    return any(count_line(board, x, y, dx, dy, color) == 5 for dx, dy in DIRECTIONS)


def has_overline(board: Board, x: int, y: int, color: int) -> bool:
    """是否存在某一方向 6 连或以上。"""
    return any(count_line(board, x, y, dx, dy, color) >= 6 for dx, dy in DIRECTIONS)


def _has_five_or_more(board: Board, x: int, y: int, color: int) -> bool:
    return any(count_line(board, x, y, dx, dy, color) >= 5 for dx, dy in DIRECTIONS)


def _in_bounds(x: int, y: int) -> bool:
    return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE


def _window_positions(
    x: int, y: int, dx: int, dy: int, start_offset: int
) -> Tuple[Tuple[int, int], ...]:
    return tuple(
        (x + (start_offset + i) * dx, y + (start_offset + i) * dy)
        for i in range(5)
    )


def _makes_exact_five_after_fill(
    board: Board,
    empty_position: Tuple[int, int],
    dx: int,
    dy: int,
    color: int,
) -> bool:
    ex, ey = empty_position
    original = board.grid[ex][ey]
    if original != EMPTY:
        return False
    board.grid[ex][ey] = color
    try:
        return count_line(board, ex, ey, dx, dy, color) == 5
    finally:
        board.grid[ex][ey] = original


def find_four_threats(board: Board, x: int, y: int, color: int) -> List[FourThreat]:
    """查找以最后一手为参与点的“四”。

    工程化定义：
    - 枚举四个方向中所有包含 ``(x, y)`` 的长度 5 连续窗口。
    - 窗口内必须有 4 个 ``color`` 棋子和 1 个空点。
    - 临时把该空点补成 ``color`` 后，在同一方向必须形成精确五连。
    """
    if color == EMPTY or not _in_bounds(x, y) or board.grid[x][y] != color:
        return []

    threats: List[FourThreat] = []
    seen: set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()
    for dx, dy in DIRECTIONS:
        for start_offset in range(-4, 1):
            positions = _window_positions(x, y, dx, dy, start_offset)
            if any(not _in_bounds(px, py) for px, py in positions):
                continue

            values = [board.grid[px][py] for px, py in positions]
            if values.count(color) != 4 or values.count(EMPTY) != 1:
                continue

            empty_index = values.index(EMPTY)
            empty_position = positions[empty_index]
            if not _makes_exact_five_after_fill(
                board, empty_position, dx, dy, color
            ):
                continue

            key = ((dx, dy), empty_position)
            if key in seen:
                continue
            seen.add(key)
            threats.append(
                FourThreat(
                    direction=(dx, dy),
                    empty_position=empty_position,
                    window_positions=positions,
                )
            )
    return threats


def count_four_threat_directions(board: Board, x: int, y: int, color: int) -> int:
    """按方向统计“四”的数量；同一方向多个补点只算 1。"""
    return len({threat.direction for threat in find_four_threats(board, x, y, color)})


def is_double_four(board: Board, x: int, y: int, color: int = BLACK) -> bool:
    """是否形成两个或以上独立方向的“四”。"""
    return count_four_threat_directions(board, x, y, color) >= 2


def _four_segment_positions(
    x: int, y: int, dx: int, dy: int, start_offset: int
) -> Tuple[Tuple[int, int], ...]:
    return tuple(
        (x + (start_offset + i) * dx, y + (start_offset + i) * dy)
        for i in range(4)
    )


def _open_four_segments_after_fill(
    board: Board,
    fill_position: Tuple[int, int],
    direction: Tuple[int, int],
    color: int,
    required_position: Optional[Tuple[int, int]] = None,
) -> List[Tuple[Tuple[int, int], ...]]:
    """Return open-four segments created by filling one empty point."""
    fx, fy = fill_position
    dx, dy = direction
    if color == EMPTY or (dx, dy) == (0, 0) or not _in_bounds(fx, fy):
        return []
    if board.grid[fx][fy] != EMPTY:
        return []

    original = board.grid[fx][fy]
    board.grid[fx][fy] = color
    try:
        segments: List[Tuple[Tuple[int, int], ...]] = []
        for start_offset in range(-3, 1):
            positions = _four_segment_positions(fx, fy, dx, dy, start_offset)
            if any(not _in_bounds(px, py) for px, py in positions):
                continue
            if fill_position not in positions:
                continue
            if required_position is not None and required_position not in positions:
                continue
            if any(board.grid[px][py] != color for px, py in positions):
                continue

            before = (positions[0][0] - dx, positions[0][1] - dy)
            after = (positions[-1][0] + dx, positions[-1][1] + dy)
            if not (_in_bounds(*before) and _in_bounds(*after)):
                continue
            if board.grid[before[0]][before[1]] != EMPTY:
                continue
            if board.grid[after[0]][after[1]] != EMPTY:
                continue
            if count_line(board, fx, fy, dx, dy, color) != 4:
                continue
            segments.append(positions)
        return segments
    finally:
        board.grid[fx][fy] = original


def is_open_four_after_move(
    board: Board,
    x: int,
    y: int,
    direction: Tuple[int, int],
    color: int = BLACK,
) -> bool:
    """Whether filling empty ``(x, y)`` creates an open four in ``direction``."""
    return bool(_open_four_segments_after_fill(board, (x, y), direction, color))


def _classify_open_three(
    board: Board, positions: Tuple[Tuple[int, int], ...], color: int
) -> str:
    color_indexes = [
        index
        for index, (px, py) in enumerate(positions)
        if board.grid[px][py] == color
    ]
    if color_indexes in ([0, 1, 2], [1, 2, 3]):
        return "straight_open_three"
    return "broken_open_three"


def find_open_three_threats(
    board: Board, x: int, y: int, color: int = BLACK
) -> List[OpenThreeThreat]:
    """Find open-three threats involving the last move."""
    if color == EMPTY or not _in_bounds(x, y) or board.grid[x][y] != color:
        return []

    threats: List[OpenThreeThreat] = []
    seen: set[
        Tuple[Tuple[int, int], Tuple[int, int], Tuple[Tuple[int, int], ...]]
    ] = set()
    last_position = (x, y)
    for dx, dy in DIRECTIONS:
        direction = (dx, dy)
        for offset in range(-4, 5):
            if offset == 0:
                continue
            extension_position = (x + offset * dx, y + offset * dy)
            ex, ey = extension_position
            if not _in_bounds(ex, ey) or board.grid[ex][ey] != EMPTY:
                continue

            segments = _open_four_segments_after_fill(
                board, extension_position, direction, color, last_position
            )
            for positions in segments:
                values = [board.grid[px][py] for px, py in positions]
                if values.count(color) != 3 or values.count(EMPTY) != 1:
                    continue
                if extension_position not in positions:
                    continue

                key = (direction, extension_position, positions)
                if key in seen:
                    continue
                seen.add(key)
                threats.append(
                    OpenThreeThreat(
                        direction=direction,
                        extension_position=extension_position,
                        pattern_positions=positions,
                        threat_type=_classify_open_three(board, positions, color),
                    )
                )
    return threats


def count_open_three_directions(
    board: Board, x: int, y: int, color: int = BLACK
) -> int:
    """Count open threes by independent direction, not extension point."""
    return len(
        {threat.direction for threat in find_open_three_threats(board, x, y, color)}
    )


def is_double_three(board: Board, x: int, y: int, color: int = BLACK) -> bool:
    """Whether two or more independent directions form open-three threats."""
    return count_open_three_directions(board, x, y, color) >= 2


def _find_any_terminal_result(board: Board) -> Optional[GameRuleResult]:
    """满盘兜底扫描：用于避免已有胜线时误判平局。"""
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            if board.grid[x][y] == BLACK and has_exact_five(board, x, y, BLACK):
                return GameRuleResult(True, BLACK, "black_exact_five", False)

    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            if board.grid[x][y] == WHITE and _has_five_or_more(board, x, y, WHITE):
                return GameRuleResult(True, WHITE, "white_five_or_more", False)

    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            if (
                board.grid[x][y] == BLACK
                and has_overline(board, x, y, BLACK)
                and not has_exact_five(board, x, y, BLACK)
            ):
                return GameRuleResult(True, WHITE, "black_overline_forbidden", True)

    return None


def check_forbidden_overline(
    board: Board, last_move: Optional[Tuple[int, int, int]]
) -> bool:
    """黑方是否因最后一手形成单纯长连而禁手。"""
    if last_move is None:
        return False
    x, y, color = last_move
    if color != BLACK or board.grid[x][y] != BLACK:
        return False
    return has_overline(board, x, y, BLACK) and not has_exact_five(
        board, x, y, BLACK
    )


def check_forbidden_double_four(
    board: Board, last_move: Optional[Tuple[int, int, int]]
) -> bool:
    """黑方是否因最后一手形成四四而禁手。"""
    if last_move is None:
        return False
    x, y, color = last_move
    if color != BLACK or not _in_bounds(x, y) or board.grid[x][y] != BLACK:
        return False
    return is_double_four(board, x, y, BLACK)


def check_forbidden_double_three(
    board: Board, last_move: Optional[Tuple[int, int, int]]
) -> bool:
    """Whether black's last move forms a double-three forbidden move."""
    if last_move is None:
        return False
    x, y, color = last_move
    if color != BLACK or not _in_bounds(x, y) or board.grid[x][y] != BLACK:
        return False
    return is_double_three(board, x, y, BLACK)


def get_game_result_forbidden(
    board: Board, last_move: Optional[Tuple[int, int, int]] = None
) -> GameRuleResult:
    """按当前禁手规则返回完整对局状态。"""
    if last_move is None:
        last_move = board.last_move

    if last_move is not None:
        x, y, color = last_move
        if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE and board.grid[x][y] == color:
            if color == BLACK:
                if has_exact_five(board, x, y, BLACK):
                    return GameRuleResult(
                        is_over=True,
                        winner=BLACK,
                        reason="black_exact_five",
                        forbidden=False,
                    )
                if has_overline(board, x, y, BLACK):
                    return GameRuleResult(
                        is_over=True,
                        winner=WHITE,
                        reason="black_overline_forbidden",
                        forbidden=True,
                    )
                if is_double_four(board, x, y, BLACK):
                    return GameRuleResult(
                        is_over=True,
                        winner=WHITE,
                        reason="black_double_four_forbidden",
                        forbidden=True,
                    )
                if is_double_three(board, x, y, BLACK):
                    return GameRuleResult(
                        is_over=True,
                        winner=WHITE,
                        reason="black_double_three_forbidden",
                        forbidden=True,
                    )
            elif color == WHITE and _has_five_or_more(board, x, y, WHITE):
                return GameRuleResult(
                    is_over=True,
                    winner=WHITE,
                    reason="white_five_or_more",
                    forbidden=False,
                )

    if board.move_count >= BOARD_SIZE * BOARD_SIZE:
        existing_result = _find_any_terminal_result(board)
        if existing_result is not None:
            return existing_result
        return GameRuleResult(is_over=True, winner=0, reason="draw", forbidden=False)

    return GameRuleResult(is_over=False, winner=None, reason="ongoing", forbidden=False)


def check_winner_forbidden(
    board: Board, last_move: Optional[Tuple[int, int, int]]
) -> int:
    """兼容旧接口：返回胜者颜色，未结束或平局返回 0。"""
    result = get_game_result_forbidden(board, last_move)
    if result.winner is None:
        return 0
    return int(result.winner)


def is_game_over_forbidden(
    board: Board, last_move: Optional[Tuple[int, int, int]] = None
) -> bool:
    """禁手模式下对局是否结束。"""
    return get_game_result_forbidden(board, last_move).is_over


__all__ = [
    "GameRuleResult",
    "FourThreat",
    "OpenThreeThreat",
    "DIRECTIONS",
    "count_continuous_stones",
    "count_line",
    "has_exact_five",
    "has_overline",
    "find_four_threats",
    "count_four_threat_directions",
    "is_double_four",
    "is_open_four_after_move",
    "find_open_three_threats",
    "count_open_three_directions",
    "is_double_three",
    "check_forbidden_overline",
    "check_forbidden_double_four",
    "check_forbidden_double_three",
    "check_winner_forbidden",
    "is_game_over_forbidden",
    "get_game_result_forbidden",
]
