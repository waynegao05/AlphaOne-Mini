"""Dependency-free standalone Tkinter Gomoku.

Run:
    python main_standalone_tkinter_play.py

This file intentionally does not import project modules.  It bundles a 15x15
board, basic / engineering-forbidden rules, tactical scoring, bounded VCF/VCT
search, and a local Tkinter UI into one standard-library-only script.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import threading
import time
import tkinter as tk
from contextlib import contextmanager
from tkinter import filedialog, messagebox, ttk


BOARD_SIZE = 15
EMPTY = 0
BLACK = 1
WHITE = -1
DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))
LETTERS = "ABCDEFGHIJKLMNO"


def xy_to_index(x: int, y: int) -> int:
    return int(y) * BOARD_SIZE + int(x)


def index_to_xy(action: int) -> tuple[int, int]:
    action = int(action)
    if not 0 <= action < BOARD_SIZE * BOARD_SIZE:
        raise ValueError(f"action out of range: {action}")
    return action % BOARD_SIZE, action // BOARD_SIZE


def xy_to_coord(x: int, y: int) -> str:
    return f"{LETTERS[x]}{y + 1}"


class StandaloneBoard:
    def __init__(self) -> None:
        self.grid = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.current_player = BLACK
        self.last_move: tuple[int, int, int] | None = None
        self.move_count = 0

    def copy(self) -> "StandaloneBoard":
        new = StandaloneBoard()
        new.grid = [row[:] for row in self.grid]
        new.current_player = self.current_player
        new.last_move = self.last_move
        new.move_count = self.move_count
        return new

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE

    def is_legal_move(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and self.grid[x][y] == EMPTY

    def get_legal_moves(self) -> list[tuple[int, int]]:
        return [
            (x, y)
            for x in range(BOARD_SIZE)
            for y in range(BOARD_SIZE)
            if self.grid[x][y] == EMPTY
        ]

    def place_stone(self, x: int, y: int, color: int | None = None) -> None:
        color = self.current_player if color is None else int(color)
        if color not in (BLACK, WHITE):
            raise ValueError(f"invalid color: {color}")
        if not self.is_legal_move(x, y):
            raise ValueError(f"illegal move: {x}, {y}")
        self.grid[x][y] = color
        self.last_move = (x, y, color)
        self.move_count += 1
        self.current_player = -color


@contextmanager
def temporary_stone(board: StandaloneBoard, x: int, y: int, color: int):
    if not board.is_legal_move(x, y):
        raise ValueError(f"temporary move is illegal: {x}, {y}")
    old_current = board.current_player
    old_last = board.last_move
    old_count = board.move_count
    try:
        board.grid[x][y] = color
        board.last_move = (x, y, color)
        board.move_count += 1
        board.current_player = -color
        yield board
    finally:
        board.grid[x][y] = EMPTY
        board.current_player = old_current
        board.last_move = old_last
        board.move_count = old_count


def count_line(board: StandaloneBoard, x: int, y: int, dx: int, dy: int, color: int) -> int:
    if not board.in_bounds(x, y) or board.grid[x][y] != color:
        return 0
    total = 1
    nx, ny = x + dx, y + dy
    while board.in_bounds(nx, ny) and board.grid[nx][ny] == color:
        total += 1
        nx += dx
        ny += dy
    nx, ny = x - dx, y - dy
    while board.in_bounds(nx, ny) and board.grid[nx][ny] == color:
        total += 1
        nx -= dx
        ny -= dy
    return total


def has_exact_five(board: StandaloneBoard, x: int, y: int, color: int) -> bool:
    return any(count_line(board, x, y, dx, dy, color) == 5 for dx, dy in DIRECTIONS)


def has_five_or_more(board: StandaloneBoard, x: int, y: int, color: int) -> bool:
    return any(count_line(board, x, y, dx, dy, color) >= 5 for dx, dy in DIRECTIONS)


def has_overline(board: StandaloneBoard, x: int, y: int, color: int) -> bool:
    return any(count_line(board, x, y, dx, dy, color) > 5 for dx, dy in DIRECTIONS)


def check_winner_basic(board: StandaloneBoard) -> int:
    if board.last_move:
        x, y, color = board.last_move
        if has_five_or_more(board, x, y, color):
            return color
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            color = board.grid[x][y]
            if color != EMPTY and has_five_or_more(board, x, y, color):
                return color
    return 0


def _open_ends_for_run(
    board: StandaloneBoard, start: tuple[int, int], end: tuple[int, int], dx: int, dy: int
) -> int:
    before = (start[0] - dx, start[1] - dy)
    after = (end[0] + dx, end[1] + dy)
    opens = 0
    if board.in_bounds(*before) and board.grid[before[0]][before[1]] == EMPTY:
        opens += 1
    if board.in_bounds(*after) and board.grid[after[0]][after[1]] == EMPTY:
        opens += 1
    return opens


def _line_segment(
    x: int, y: int, dx: int, dy: int, start_offset: int, length: int
) -> tuple[tuple[int, int], ...]:
    return tuple((x + (start_offset + i) * dx, y + (start_offset + i) * dy) for i in range(length))


def has_open_four_at(board: StandaloneBoard, x: int, y: int, color: int) -> bool:
    for dx, dy in DIRECTIONS:
        for start in range(-3, 1):
            cells = _line_segment(x, y, dx, dy, start, 4)
            if (x, y) not in cells or any(not board.in_bounds(a, b) for a, b in cells):
                continue
            if any(board.grid[a][b] != color for a, b in cells):
                continue
            if _open_ends_for_run(board, cells[0], cells[-1], dx, dy) == 2:
                return True
    return False


def has_blocked_four_at(board: StandaloneBoard, x: int, y: int, color: int) -> bool:
    for dx, dy in DIRECTIONS:
        for start in range(-3, 1):
            cells = _line_segment(x, y, dx, dy, start, 4)
            if (x, y) not in cells or any(not board.in_bounds(a, b) for a, b in cells):
                continue
            if any(board.grid[a][b] != color for a, b in cells):
                continue
            if _open_ends_for_run(board, cells[0], cells[-1], dx, dy) == 1:
                return True
    return False


def find_open_three_extensions(
    board: StandaloneBoard, x: int, y: int, color: int
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if board.grid[x][y] != color:
        return []
    out: list[tuple[tuple[int, int], tuple[int, int]]] = []
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for dx, dy in DIRECTIONS:
        direction = (dx, dy)
        for offset in range(-4, 5):
            if offset == 0:
                continue
            ex, ey = x + offset * dx, y + offset * dy
            if not board.in_bounds(ex, ey) or board.grid[ex][ey] != EMPTY:
                continue
            with temporary_stone(board, ex, ey, color):
                if has_open_four_at(board, ex, ey, color):
                    key = (direction, (ex, ey))
                    if key not in seen:
                        seen.add(key)
                        out.append(key)
    return out


def count_open_three_directions(board: StandaloneBoard, x: int, y: int, color: int) -> int:
    return len({direction for direction, _ in find_open_three_extensions(board, x, y, color)})


def is_double_three(board: StandaloneBoard, x: int, y: int, color: int) -> bool:
    return count_open_three_directions(board, x, y, color) >= 2


def four_threat_directions(board: StandaloneBoard, x: int, y: int, color: int) -> int:
    dirs: set[tuple[int, int]] = set()
    for dx, dy in DIRECTIONS:
        for start in range(-4, 1):
            cells = _line_segment(x, y, dx, dy, start, 5)
            if (x, y) not in cells or any(not board.in_bounds(a, b) for a, b in cells):
                continue
            values = [board.grid[a][b] for a, b in cells]
            if values.count(color) == 4 and values.count(EMPTY) == 1:
                dirs.add((dx, dy))
    return len(dirs)


def is_double_four(board: StandaloneBoard, x: int, y: int, color: int) -> bool:
    return four_threat_directions(board, x, y, color) >= 2


def forbidden_reason_after_move(board: StandaloneBoard, action: int, color: int) -> str | None:
    if color != BLACK:
        return None
    x, y = index_to_xy(action)
    if not board.is_legal_move(x, y):
        return "illegal"
    with temporary_stone(board, x, y, color):
        if has_exact_five(board, x, y, BLACK):
            return None
        if has_overline(board, x, y, BLACK):
            return "overline"
        if is_double_four(board, x, y, BLACK):
            return "double_four"
        if is_double_three(board, x, y, BLACK):
            return "double_three"
    return None


def game_result(board: StandaloneBoard, rule_mode: str = "basic") -> tuple[bool, int, str]:
    if rule_mode == "basic":
        winner = check_winner_basic(board)
        if winner:
            return True, winner, "five"
    else:
        if board.last_move:
            x, y, color = board.last_move
            if color == BLACK:
                if has_exact_five(board, x, y, BLACK):
                    return True, BLACK, "black_exact_five"
                if has_overline(board, x, y, BLACK):
                    return True, WHITE, "black_overline_forbidden"
                if is_double_four(board, x, y, BLACK):
                    return True, WHITE, "black_double_four_forbidden"
                if is_double_three(board, x, y, BLACK):
                    return True, WHITE, "black_double_three_forbidden"
            elif color == WHITE and has_five_or_more(board, x, y, WHITE):
                return True, WHITE, "white_five_or_more"
    if board.move_count >= BOARD_SIZE * BOARD_SIZE:
        return True, 0, "draw"
    return False, 0, "ongoing"


def generate_candidate_moves(
    board: StandaloneBoard, radius: int = 2, max_candidates: int | None = 80
) -> list[int]:
    occupied = [
        (x, y)
        for x in range(BOARD_SIZE)
        for y in range(BOARD_SIZE)
        if board.grid[x][y] != EMPTY
    ]
    if not occupied:
        return [xy_to_index(7, 7)]
    candidates: set[tuple[int, int]] = set()
    for ox, oy in occupied:
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                x, y = ox + dx, oy + dy
                if board.is_legal_move(x, y):
                    candidates.add((x, y))
    center = 7

    def score(pos: tuple[int, int]) -> tuple[float, int]:
        x, y = pos
        near = sum(
            1.0 / max(1, max(abs(x - ox), abs(y - oy)))
            for ox, oy in occupied
            if max(abs(x - ox), abs(y - oy)) <= radius
        )
        center_bonus = -0.1 * (abs(x - center) + abs(y - center))
        return near + center_bonus, -xy_to_index(x, y)

    ordered = sorted(candidates, key=score, reverse=True)
    actions = [xy_to_index(x, y) for x, y in ordered]
    return actions[:max_candidates] if max_candidates is not None else actions


def is_legal_for_color(board: StandaloneBoard, action: int, color: int, rule_mode: str) -> bool:
    try:
        x, y = index_to_xy(action)
    except ValueError:
        return False
    if not board.is_legal_move(x, y):
        return False
    if rule_mode == "forbidden" and color == BLACK:
        return forbidden_reason_after_move(board, action, color) is None
    return True


def immediate_winning_moves(
    board: StandaloneBoard, color: int, rule_mode: str = "basic", actions: list[int] | None = None
) -> list[int]:
    source = actions if actions is not None else [xy_to_index(x, y) for x, y in board.get_legal_moves()]
    out: list[int] = []
    for action in source:
        if not is_legal_for_color(board, action, color, rule_mode):
            continue
        x, y = index_to_xy(action)
        with temporary_stone(board, x, y, color):
            over, winner, _ = game_result(board, rule_mode)
            if over and winner == color:
                out.append(action)
    return sorted(out)


def blocking_moves(board: StandaloneBoard, color: int, rule_mode: str = "basic") -> list[int]:
    source = [xy_to_index(x, y) for x, y in board.get_legal_moves()]
    opp_wins = set(immediate_winning_moves(board, -color, rule_mode, source))
    return sorted(a for a in source if a in opp_wins and is_legal_for_color(board, a, color, rule_mode))


def move_threats(board: StandaloneBoard, action: int, color: int, rule_mode: str = "basic") -> set[str]:
    if not is_legal_for_color(board, action, color, rule_mode):
        return {"forbidden"} if rule_mode == "forbidden" and color == BLACK else {"illegal"}
    x, y = index_to_xy(action)
    with temporary_stone(board, x, y, color):
        threats: set[str] = set()
        over, winner, _ = game_result(board, rule_mode)
        if over and winner == color:
            threats.add("five")
            return threats
        if has_open_four_at(board, x, y, color):
            threats.add("open_four")
        if has_blocked_four_at(board, x, y, color):
            threats.add("blocked_four")
        if is_double_four(board, x, y, color):
            threats.add("double_four")
        if count_open_three_directions(board, x, y, color) >= 1:
            threats.add("open_three")
        if is_double_three(board, x, y, color):
            threats.add("double_three")
        return threats


def open_four_moves(board: StandaloneBoard, color: int, rule_mode: str = "basic") -> list[int]:
    out: list[int] = []
    for action in generate_candidate_moves(board, radius=2, max_candidates=None):
        threats = move_threats(board, action, color, rule_mode)
        if "open_four" in threats and "forbidden" not in threats:
            out.append(action)
    return sorted(out)


def evaluate_move(board: StandaloneBoard, action: int, color: int, rule_mode: str = "basic") -> float:
    if not is_legal_for_color(board, action, color, rule_mode):
        return -1_000_000.0
    own = move_threats(board, action, color, rule_mode)
    opp = move_threats(board, action, -color, rule_mode)
    weights = {
        "five": 100_000,
        "open_four": 50_000,
        "double_four": 40_000,
        "blocked_four": 20_000,
        "double_three": 12_000,
        "open_three": 8_000,
    }
    block_weights = {
        "five": 90_000,
        "open_four": 45_000,
        "double_four": 35_000,
        "blocked_four": 18_000,
        "double_three": 10_000,
        "open_three": 7_000,
    }
    x, y = index_to_xy(action)
    center = 7
    score = sum(weights.get(t, 0) for t in own)
    score += sum(block_weights.get(t, 0) for t in opp)
    score += max(0, 20 - abs(x - center) - abs(y - center)) * 3
    for nx in range(max(0, x - 2), min(BOARD_SIZE, x + 3)):
        for ny in range(max(0, y - 2), min(BOARD_SIZE, y + 3)):
            if board.grid[nx][ny] != EMPTY:
                score += 2 if max(abs(nx - x), abs(ny - y)) == 1 else 0.5
    return float(score)


def winning_cells_after_current_position(board: StandaloneBoard, color: int, rule_mode: str) -> list[int]:
    return immediate_winning_moves(board, color, rule_mode)


def vcf_attack_candidates(board: StandaloneBoard, color: int, rule_mode: str) -> list[int]:
    out: list[int] = []
    for action in generate_candidate_moves(board, radius=2, max_candidates=80):
        if not is_legal_for_color(board, action, color, rule_mode):
            continue
        x, y = index_to_xy(action)
        with temporary_stone(board, x, y, color):
            if immediate_winning_moves(board, color, rule_mode):
                out.append(action)
    return sorted(out, key=lambda a: (-evaluate_move(board, a, color, rule_mode), a))


class SearchBudgetExceeded(Exception):
    pass


def _take_node(state: dict) -> None:
    state["nodes"] += 1
    if state["nodes"] > state["budget"]:
        raise SearchBudgetExceeded


def _vcf_recurse(board: StandaloneBoard, color: int, depth: int, rule_mode: str, state: dict) -> bool:
    _take_node(state)
    if depth <= 0:
        return False
    if immediate_winning_moves(board, color, rule_mode):
        return True
    if immediate_winning_moves(board, -color, rule_mode):
        return False
    if depth < 3:
        return False
    for attack in vcf_attack_candidates(board, color, rule_mode):
        x, y = index_to_xy(attack)
        with temporary_stone(board, x, y, color):
            wins = immediate_winning_moves(board, color, rule_mode)
            if len(wins) >= 2:
                return True
            if not wins:
                continue
            all_forced = True
            for defense in wins:
                if not is_legal_for_color(board, defense, -color, rule_mode):
                    continue
                dx, dy = index_to_xy(defense)
                with temporary_stone(board, dx, dy, -color):
                    if not _vcf_recurse(board, color, depth - 2, rule_mode, state):
                        all_forced = False
                        break
            if all_forced:
                return True
    return False


def vcf_first_move(
    board: StandaloneBoard,
    color: int,
    rule_mode: str = "basic",
    max_depth: int = 7,
    node_budget: int = 10000,
) -> int | None:
    if node_budget <= 0:
        return None
    wins = immediate_winning_moves(board, color, rule_mode)
    if wins:
        return wins[0]
    if immediate_winning_moves(board, -color, rule_mode):
        return None
    state = {"nodes": 0, "budget": int(node_budget)}
    try:
        for attack in vcf_attack_candidates(board, color, rule_mode):
            x, y = index_to_xy(attack)
            with temporary_stone(board, x, y, color):
                wins = immediate_winning_moves(board, color, rule_mode)
                if len(wins) >= 2:
                    return attack
                if not wins:
                    continue
                ok = True
                for defense in wins:
                    dx, dy = index_to_xy(defense)
                    with temporary_stone(board, dx, dy, -color):
                        if not _vcf_recurse(board, color, max_depth - 2, rule_mode, state):
                            ok = False
                            break
                if ok:
                    return attack
    except SearchBudgetExceeded:
        return None
    return None


def vcf_defends(board: StandaloneBoard, color: int, candidate: int, rule_mode: str, depth: int = 5) -> bool:
    if not is_legal_for_color(board, candidate, color, rule_mode):
        return False
    x, y = index_to_xy(candidate)
    with temporary_stone(board, x, y, color):
        return vcf_first_move(board, -color, rule_mode, max_depth=depth) is None


def vct_attack_candidates(board: StandaloneBoard, color: int, rule_mode: str) -> list[int]:
    actions = generate_candidate_moves(board, radius=2, max_candidates=80)
    return sorted(
        [
            action
            for action in actions
            if is_legal_for_color(board, action, color, rule_mode)
            and move_threats(board, action, color, rule_mode)
            & {"open_three", "double_three", "blocked_four", "open_four", "double_four"}
        ],
        key=lambda a: (-evaluate_move(board, a, color, rule_mode), a),
    )


def vct_defense_moves(board: StandaloneBoard, attacker_color: int, rule_mode: str) -> list[int]:
    defender = -attacker_color
    wins = immediate_winning_moves(board, attacker_color, rule_mode)
    if wins:
        return [a for a in wins if is_legal_for_color(board, a, defender, rule_mode)]
    cells: set[int] = set()
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            if board.grid[x][y] != attacker_color:
                continue
            for _, point in find_open_three_extensions(board, x, y, attacker_color):
                action = xy_to_index(*point)
                if is_legal_for_color(board, action, defender, rule_mode):
                    cells.add(action)
    return sorted(cells)


def _vct_recurse(board: StandaloneBoard, color: int, depth: int, rule_mode: str, state: dict) -> bool:
    _take_node(state)
    if depth <= 0:
        return False
    if immediate_winning_moves(board, color, rule_mode):
        return True
    if vcf_first_move(board, color, rule_mode, max_depth=min(depth, 7), node_budget=2000):
        return True
    if immediate_winning_moves(board, -color, rule_mode):
        return False
    if depth < 3:
        return False
    for attack in vct_attack_candidates(board, color, rule_mode):
        x, y = index_to_xy(attack)
        with temporary_stone(board, x, y, color):
            if len(immediate_winning_moves(board, color, rule_mode)) >= 2:
                return True
            defenses = vct_defense_moves(board, color, rule_mode)
            if not defenses:
                continue
            ok = True
            for defense in defenses[:12]:
                dx, dy = index_to_xy(defense)
                with temporary_stone(board, dx, dy, -color):
                    if not _vct_recurse(board, color, depth - 2, rule_mode, state):
                        ok = False
                        break
            if ok:
                return True
    return False


def vct_first_move(
    board: StandaloneBoard,
    color: int,
    rule_mode: str = "basic",
    max_depth: int = 5,
    node_budget: int = 5000,
) -> int | None:
    if node_budget <= 0:
        return None
    if immediate_winning_moves(board, -color, rule_mode):
        return None
    state = {"nodes": 0, "budget": int(node_budget)}
    try:
        for attack in vct_attack_candidates(board, color, rule_mode):
            x, y = index_to_xy(attack)
            with temporary_stone(board, x, y, color):
                if len(immediate_winning_moves(board, color, rule_mode)) >= 2:
                    return attack
                defenses = vct_defense_moves(board, color, rule_mode)
                if not defenses:
                    continue
                ok = True
                for defense in defenses[:12]:
                    dx, dy = index_to_xy(defense)
                    with temporary_stone(board, dx, dy, -color):
                        if not _vct_recurse(board, color, max_depth - 2, rule_mode, state):
                            ok = False
                            break
                if ok:
                    return attack
    except SearchBudgetExceeded:
        return None
    return None


def vct_defends(board: StandaloneBoard, color: int, candidate: int, rule_mode: str, depth: int = 5) -> bool:
    if not is_legal_for_color(board, candidate, color, rule_mode):
        return False
    x, y = index_to_xy(candidate)
    with temporary_stone(board, x, y, color):
        return vct_first_move(board, -color, rule_mode, max_depth=depth) is None


class StandaloneTacticalAI:
    def __init__(self, rule_mode: str = "basic", name: str = "Tactical AI") -> None:
        self.rule_mode = rule_mode
        self.name = name
        self.last_decision_reason = ""

    def select_action(self, board: StandaloneBoard) -> int | None:
        color = board.current_player
        wins = immediate_winning_moves(board, color, self.rule_mode)
        if wins:
            self.last_decision_reason = "immediate_win"
            return wins[0]
        blocks = blocking_moves(board, color, self.rule_mode)
        if blocks:
            self.last_decision_reason = "immediate_block"
            return max(blocks, key=lambda a: evaluate_move(board, a, color, self.rule_mode))
        own_open = open_four_moves(board, color, self.rule_mode)
        if own_open:
            self.last_decision_reason = "open_four_attack"
            return max(own_open, key=lambda a: evaluate_move(board, a, color, self.rule_mode))
        opp_open = open_four_moves(board, -color, self.rule_mode)
        legal_blocks = [a for a in opp_open if is_legal_for_color(board, a, color, self.rule_mode)]
        if legal_blocks:
            self.last_decision_reason = "open_four_defense"
            return max(legal_blocks, key=lambda a: evaluate_move(board, a, color, self.rule_mode))
        candidates = [
            a
            for a in generate_candidate_moves(board, radius=2, max_candidates=80)
            if is_legal_for_color(board, a, color, self.rule_mode)
        ]
        if not candidates:
            return None
        self.last_decision_reason = "heuristic"
        return max(candidates, key=lambda a: evaluate_move(board, a, color, self.rule_mode))


class StandaloneStrongAI(StandaloneTacticalAI):
    def __init__(self, rule_mode: str = "basic", name: str = "Standalone Strong AI") -> None:
        super().__init__(rule_mode=rule_mode, name=name)
        self.vcf_depth = 7
        self.vcf_defense_depth = 5
        self.vcf_node_budget = 10000
        self.vct_depth = 5
        self.vct_node_budget = 5000

    def select_action(self, board: StandaloneBoard) -> int | None:
        color = board.current_player
        wins = immediate_winning_moves(board, color, self.rule_mode)
        if wins:
            self.last_decision_reason = "immediate_win"
            return wins[0]
        blocks = blocking_moves(board, color, self.rule_mode)
        if blocks:
            self.last_decision_reason = "immediate_block"
            return max(blocks, key=lambda a: evaluate_move(board, a, color, self.rule_mode))
        move = vcf_first_move(board, color, self.rule_mode, self.vcf_depth, self.vcf_node_budget)
        if move is not None and is_legal_for_color(board, move, color, self.rule_mode):
            self.last_decision_reason = "vcf_attack"
            return move
        opp_vcf = vcf_first_move(board, -color, self.rule_mode, self.vcf_defense_depth, self.vcf_node_budget)
        if opp_vcf is not None and vcf_defends(board, color, opp_vcf, self.rule_mode, self.vcf_defense_depth):
            self.last_decision_reason = "vcf_defense"
            return opp_vcf
        opp_vct = vct_first_move(board, -color, self.rule_mode, self.vct_depth, self.vct_node_budget)
        if opp_vct is not None and vct_defends(board, color, opp_vct, self.rule_mode, self.vct_depth):
            self.last_decision_reason = "vct_defense"
            return opp_vct
        move = vct_first_move(board, color, self.rule_mode, self.vct_depth, self.vct_node_budget)
        if move is not None and is_legal_for_color(board, move, color, self.rule_mode):
            self.last_decision_reason = "vct_attack"
            return move
        return super().select_action(board)


class StandaloneRandomAI:
    def __init__(self, rule_mode: str = "basic", name: str = "Random AI") -> None:
        self.rule_mode = rule_mode
        self.name = name
        self.last_decision_reason = ""

    def select_action(self, board: StandaloneBoard) -> int | None:
        color = board.current_player
        legal = [
            xy_to_index(x, y)
            for x, y in board.get_legal_moves()
            if is_legal_for_color(board, xy_to_index(x, y), color, self.rule_mode)
        ]
        self.last_decision_reason = "random"
        return random.choice(legal) if legal else None


PLAYER_LABELS = {
    "Human": None,
    "Standalone Strong AI": StandaloneStrongAI,
    "Tactical AI": StandaloneTacticalAI,
    "Random AI": StandaloneRandomAI,
}


class StandaloneGomokuApp:
    def __init__(self, root: tk.Tk, rule_mode: str = "basic") -> None:
        self.root = root
        self.root.title("Standalone Gomoku Strong AI")
        self.board = StandaloneBoard()
        self.rule_mode_var = tk.StringVar(value=rule_mode)
        self.black_player_var = tk.StringVar(value="Human")
        self.white_player_var = tk.StringVar(value="Standalone Strong AI")
        self.delay_var = tk.IntVar(value=250)
        self.running = False
        self.paused = False
        self.ai_busy = False
        self.players: dict[int, object | None] = {BLACK: None, WHITE: None}
        self.moves: list[dict[str, object]] = []
        self.cell = 36
        self.margin = 38
        self.last_ai_reason = ""
        self._build_ui()
        self.start_game()

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=6)
        top.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(top, text="Black").pack(side=tk.LEFT)
        ttk.Combobox(top, textvariable=self.black_player_var, values=list(PLAYER_LABELS), width=22, state="readonly").pack(side=tk.LEFT, padx=4)
        ttk.Label(top, text="White").pack(side=tk.LEFT)
        ttk.Combobox(top, textvariable=self.white_player_var, values=list(PLAYER_LABELS), width=22, state="readonly").pack(side=tk.LEFT, padx=4)
        ttk.Label(top, text="Rule").pack(side=tk.LEFT)
        ttk.Combobox(top, textvariable=self.rule_mode_var, values=["basic", "forbidden"], width=10, state="readonly").pack(side=tk.LEFT, padx=4)
        ttk.Label(top, text="Delay ms").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.delay_var, width=6).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Start", command=self.start_game).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Pause", command=self.toggle_pause).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Step", command=self.step_once).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Save JSON", command=self.save_json).pack(side=tk.LEFT, padx=4)

        body = ttk.Frame(self.root, padding=6)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        size = self.margin * 2 + self.cell * (BOARD_SIZE - 1)
        self.canvas = tk.Canvas(body, width=size + 40, height=size + 60, bg="#d9a85f")
        self.canvas.pack(side=tk.LEFT)
        self.canvas.bind("<Button-1>", self.on_click)
        right = ttk.Frame(body)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        ttk.Label(right, text="Move Log").pack(anchor="w")
        self.log = tk.Text(right, height=28, width=58)
        self.log.pack(fill=tk.BOTH, expand=True)
        self.status = tk.StringVar(value="")
        ttk.Label(self.root, textvariable=self.status, anchor="w", padding=6).pack(side=tk.BOTTOM, fill=tk.X)
        self.draw_board()

    def make_player(self, label: str):
        cls = PLAYER_LABELS[label]
        if cls is None:
            return None
        return cls(rule_mode=self.rule_mode_var.get())

    def start_game(self) -> None:
        self.board = StandaloneBoard()
        self.moves = []
        self.players = {
            BLACK: self.make_player(self.black_player_var.get()),
            WHITE: self.make_player(self.white_player_var.get()),
        }
        self.running = True
        self.paused = False
        self.ai_busy = False
        self.last_ai_reason = ""
        self.log.delete("1.0", tk.END)
        self.draw_board()
        self.update_status()
        self.schedule_ai_if_needed()

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        self.update_status()
        if not self.paused:
            self.schedule_ai_if_needed()

    def step_once(self) -> None:
        if self.is_ai_turn():
            self.run_ai_turn_once()

    def is_ai_turn(self) -> bool:
        return self.players.get(self.board.current_player) is not None

    def schedule_ai_if_needed(self) -> None:
        if not self.running or self.paused or self.ai_busy:
            return
        over, _, _ = game_result(self.board, self.rule_mode_var.get())
        if over:
            self.update_status()
            return
        if self.is_ai_turn():
            self.root.after(max(0, int(self.delay_var.get())), self.run_ai_turn_once)

    def run_ai_turn_once(self) -> None:
        if not self.running or self.paused or self.ai_busy or not self.is_ai_turn():
            return
        player = self.players[self.board.current_player]
        color = self.board.current_player
        snapshot = self.board.copy()
        self.ai_busy = True

        def worker():
            start = time.perf_counter()
            try:
                action = player.select_action(snapshot)  # type: ignore[union-attr]
                reason = getattr(player, "last_decision_reason", "ai")
                error = None
            except Exception as exc:  # noqa: BLE001 - GUI must not crash
                action, reason, error = None, "error", str(exc)
            elapsed = time.perf_counter() - start
            self.root.after(0, lambda: self.finish_ai_move(color, action, reason, elapsed, error))

        threading.Thread(target=worker, daemon=True).start()

    def finish_ai_move(self, color: int, action, reason: str, elapsed: float, error: str | None) -> None:
        self.ai_busy = False
        if error:
            messagebox.showerror("AI error", error)
            self.running = False
            self.update_status()
            return
        if action is None or not is_legal_for_color(self.board, int(action), color, self.rule_mode_var.get()):
            legal = [
                xy_to_index(x, y)
                for x, y in self.board.get_legal_moves()
                if is_legal_for_color(self.board, xy_to_index(x, y), color, self.rule_mode_var.get())
            ]
            if not legal:
                self.running = False
                self.update_status()
                return
            action = max(legal, key=lambda a: evaluate_move(self.board, a, color, self.rule_mode_var.get()))
            reason = f"{reason}|fallback_legal"
        self.apply_action(int(action), reason, elapsed)

    def on_click(self, event) -> None:
        if not self.running or self.paused or self.ai_busy or self.is_ai_turn():
            return
        x = round((event.x - self.margin) / self.cell)
        y = round((event.y - self.margin) / self.cell)
        if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
            return
        action = xy_to_index(x, y)
        if not is_legal_for_color(self.board, action, self.board.current_player, self.rule_mode_var.get()):
            messagebox.showwarning("Illegal move", "That point is occupied or forbidden.")
            return
        self.apply_action(action, "human", 0.0)

    def apply_action(self, action: int, reason: str, elapsed: float) -> None:
        color = self.board.current_player
        x, y = index_to_xy(action)
        self.board.place_stone(x, y, color)
        entry = {
            "move": self.board.move_count,
            "color": "B" if color == BLACK else "W",
            "coord": xy_to_coord(x, y),
            "action": action,
            "reason": reason,
            "elapsed": round(elapsed, 4),
        }
        self.moves.append(entry)
        self.last_ai_reason = reason
        self.log.insert(tk.END, f"{entry['move']:03d} {entry['color']} {entry['coord']} action={action} reason={reason} time={elapsed:.3f}s\n")
        self.log.see(tk.END)
        over, winner, result_reason = game_result(self.board, self.rule_mode_var.get())
        if over:
            self.running = False
            if winner == BLACK:
                self.last_ai_reason = f"Black wins ({result_reason})"
            elif winner == WHITE:
                self.last_ai_reason = f"White wins ({result_reason})"
            else:
                self.last_ai_reason = "Draw"
        self.draw_board()
        self.update_status()
        self.schedule_ai_if_needed()

    def draw_board(self) -> None:
        self.canvas.delete("all")
        m, c = self.margin, self.cell
        for i in range(BOARD_SIZE):
            self.canvas.create_line(m, m + i * c, m + (BOARD_SIZE - 1) * c, m + i * c, fill="#4b3217")
            self.canvas.create_line(m + i * c, m, m + i * c, m + (BOARD_SIZE - 1) * c, fill="#4b3217")
            self.canvas.create_text(m + i * c, m + BOARD_SIZE * c - 2, text=LETTERS[i], font=("Arial", 10, "bold"))
            self.canvas.create_text(m - 22, m + (BOARD_SIZE - 1 - i) * c, text=str(BOARD_SIZE - i), font=("Arial", 9))
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                color = self.board.grid[x][y]
                if color == EMPTY:
                    continue
                px, py = m + x * c, m + y * c
                fill = "black" if color == BLACK else "white"
                outline = "black"
                self.canvas.create_oval(px - 14, py - 14, px + 14, py + 14, fill=fill, outline=outline, width=2)
        if self.board.last_move:
            x, y, _ = self.board.last_move
            px, py = m + x * c, m + y * c
            self.canvas.create_rectangle(px - 6, py - 6, px + 6, py + 6, outline="red", width=2)

    def update_status(self) -> None:
        color = "Black" if self.board.current_player == BLACK else "White"
        player = self.black_player_var.get() if self.board.current_player == BLACK else self.white_player_var.get()
        over, winner, reason = game_result(self.board, self.rule_mode_var.get())
        if over:
            result = "Draw" if winner == 0 else ("Black wins" if winner == BLACK else "White wins")
            text = f"{result} | reason={reason} | moves={self.board.move_count} | last={self.last_ai_reason}"
        else:
            mode = "paused" if self.paused else "running"
            text = f"{mode} | rule={self.rule_mode_var.get()} | turn={color} | player={player} | moves={self.board.move_count} | last={self.last_ai_reason}"
        self.status.set(text)

    def save_json(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save game JSON",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        payload = {
            "board_size": BOARD_SIZE,
            "rule_mode": self.rule_mode_var.get(),
            "black_player": self.black_player_var.get(),
            "white_player": self.white_player_var.get(),
            "moves": self.moves,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Standalone dependency-free Gomoku Tkinter UI")
    parser.add_argument("--rule-mode", choices=["basic", "forbidden"], default="basic")
    parser.add_argument("--black-player", choices=list(PLAYER_LABELS), default="Human")
    parser.add_argument("--white-player", choices=list(PLAYER_LABELS), default="Standalone Strong AI")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    root = tk.Tk()
    app = StandaloneGomokuApp(root, rule_mode=args.rule_mode)
    app.black_player_var.set(args.black_player)
    app.white_player_var.set(args.white_player)
    app.start_game()
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
