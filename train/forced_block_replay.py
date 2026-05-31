"""Construct forced-block replay samples for Hybrid-survival training.

The generated positions are synthetic tactical defense cases. They do not
change rules or player logic; they only create supervised labels for positions
that natural Hybrid-only mining failed to collect often enough.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Iterable

import numpy as np

from game.board import BLACK, BOARD_SIZE, EMPTY, WHITE, Board
from game.encoder import action_to_index
from train.hybrid_survival import AnnotatedMistakeSample, save_annotated_samples
from train.mistake_mining import MistakeSample, make_mistake_sample


DIRECTIONS: tuple[tuple[int, int], ...] = ((1, 0), (0, 1), (1, 1), (1, -1))


def _in_bounds(x: int, y: int) -> bool:
    return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE


def _line_points(
    target: tuple[int, int],
    direction: tuple[int, int],
    offsets: Iterable[int],
) -> list[tuple[int, int]]:
    tx, ty = target
    dx, dy = direction
    return [(tx + dx * int(offset), ty + dy * int(offset)) for offset in offsets]


def _valid_points(points: Iterable[tuple[int, int]]) -> bool:
    return all(_in_bounds(x, y) for x, y in points)


def _board_from(
    stones: Iterable[tuple[int, int, int]],
    current_player: int,
) -> Board:
    board = Board()
    seen: set[tuple[int, int]] = set()
    count = 0
    for x, y, color in stones:
        if not _in_bounds(x, y):
            raise ValueError(f"stone out of bounds: {(x, y)}")
        if (x, y) in seen:
            raise ValueError(f"duplicate stone in forced replay pattern: {(x, y)}")
        seen.add((x, y))
        board.grid[x][y] = int(color)
        count += 1
    board.current_player = int(current_player)
    board.move_count = count
    board.last_move = None
    return board


def _make_sample(
    *,
    stones: list[tuple[int, int, int]],
    current_player: int,
    target: tuple[int, int],
    reason: str,
    board_pattern_type: str,
    direction: tuple[int, int] | str,
    rule_mode: str,
    value: float = 0.5,
) -> AnnotatedMistakeSample:
    board = _board_from(stones, current_player)
    tx, ty = target
    if not board.is_legal_move(tx, ty):
        raise ValueError(f"forced-block target is not legal: {(tx, ty)}")
    action = action_to_index(tx, ty, BOARD_SIZE)
    sample = make_mistake_sample(
        board,
        teacher_action=action,
        final_winner=0,
        rule_mode=rule_mode,
        reasons=(reason,),
    )
    sample = MistakeSample(
        state=sample.state,
        policy=sample.policy,
        value=float(value),
        threat_labels=sample.threat_labels,
        forbidden_labels=sample.forbidden_labels,
        tactical_scores=sample.tactical_scores,
        teacher_action=sample.teacher_action,
        reasons=sample.reasons,
    )
    metadata = {
        "student_action": None,
        "teacher_action": int(action),
        "student_heuristic_score": None,
        "teacher_heuristic_score": None,
        "heuristic_delta": None,
        "remaining_moves": 20,
        "reason": [reason],
        "teacher_type": "forced_block_replay",
        "game_id": None,
        "ply_index": None,
        "final_result": 0,
        "teacher_action_extends_game": True,
        "student_to_move": int(current_player),
        "player_to_move": int(current_player),
        "opponent_color": int(-current_player),
        "rule_mode": rule_mode,
        "target_source": "forced_block_replay",
        "target_source_detail": board_pattern_type,
        "expected_action": int(action),
        "board_pattern_type": board_pattern_type,
        "direction": direction if isinstance(direction, str) else [int(direction[0]), int(direction[1])],
        "value_source": "forced_defense_positive",
    }
    return AnnotatedMistakeSample(sample=sample, metadata=metadata)


def _append_line_case(
    samples: list[AnnotatedMistakeSample],
    *,
    target: tuple[int, int],
    direction: tuple[int, int],
    current_player: int,
    opponent_offsets: tuple[int, ...],
    blocker_offsets: tuple[int, ...] = (),
    reason: str,
    board_pattern_type: str,
    rule_mode: str,
    value: float = 0.5,
) -> None:
    opponent = -int(current_player)
    required = _line_points(target, direction, opponent_offsets + blocker_offsets)
    if not _valid_points([target, *required]):
        return
    stones: list[tuple[int, int, int]] = []
    occupied: set[tuple[int, int]] = {target}
    for x, y in _line_points(target, direction, opponent_offsets):
        if (x, y) in occupied:
            return
        occupied.add((x, y))
        stones.append((x, y, opponent))
    for x, y in _line_points(target, direction, blocker_offsets):
        if (x, y) in occupied:
            return
        occupied.add((x, y))
        stones.append((x, y, int(current_player)))
    samples.append(
        _make_sample(
            stones=stones,
            current_player=current_player,
            target=target,
            reason=reason,
            board_pattern_type=board_pattern_type,
            direction=direction,
            rule_mode=rule_mode,
            value=value,
        )
    )


def generate_immediate_block_positions(rule_mode: str = "basic") -> list[AnnotatedMistakeSample]:
    """Return positions where the side to move must block an opponent four."""
    samples: list[AnnotatedMistakeSample] = []
    center_targets = [(7, 7)]
    edge_targets = {
        (1, 0): (4, 1),
        (0, 1): (1, 4),
        (1, 1): (4, 4),
        (1, -1): (4, 10),
    }
    for current_player in (BLACK, WHITE):
        for direction in DIRECTIONS:
            for target in center_targets:
                _append_line_case(
                    samples,
                    target=target,
                    direction=direction,
                    current_player=current_player,
                    opponent_offsets=(-4, -3, -2, -1),
                    blocker_offsets=(-5,),
                    reason="missed_immediate_block",
                    board_pattern_type="center_immediate_block",
                    rule_mode=rule_mode,
                    value=1.0,
                )
            _append_line_case(
                samples,
                target=edge_targets[direction],
                direction=direction,
                current_player=current_player,
                opponent_offsets=(-4, -3, -2, -1),
                blocker_offsets=(),
                reason="missed_immediate_block",
                board_pattern_type="edge_immediate_block",
                rule_mode=rule_mode,
                value=1.0,
            )
    return samples


def generate_open_four_defense_positions(rule_mode: str = "basic") -> list[AnnotatedMistakeSample]:
    """Return positions where the side to move blocks an opponent open-four threat."""
    samples: list[AnnotatedMistakeSample] = []
    for current_player in (BLACK, WHITE):
        for direction in DIRECTIONS:
            _append_line_case(
                samples,
                target=(7, 7),
                direction=direction,
                current_player=current_player,
                opponent_offsets=(-3, -2, -1),
                blocker_offsets=(),
                reason="missed_open_four_defense",
                board_pattern_type="opponent_open_four_defense",
                rule_mode=rule_mode,
                value=0.5,
            )
    return samples


def generate_blocked_four_defense_positions(rule_mode: str = "basic") -> list[AnnotatedMistakeSample]:
    """Return positions where the side to move blocks an opponent blocked four."""
    samples: list[AnnotatedMistakeSample] = []
    for current_player in (BLACK, WHITE):
        for direction in DIRECTIONS:
            _append_line_case(
                samples,
                target=(7, 7),
                direction=direction,
                current_player=current_player,
                opponent_offsets=(-3, -2, -1),
                blocker_offsets=(-4,),
                reason="missed_blocked_four_defense",
                board_pattern_type="opponent_blocked_four_defense",
                rule_mode=rule_mode,
                value=0.5,
            )
    return samples


def generate_double_threat_defense_positions(rule_mode: str = "basic") -> list[AnnotatedMistakeSample]:
    """Return fork-blocking positions for opponent double threats."""
    samples: list[AnnotatedMistakeSample] = []
    target = (7, 7)
    templates = [
        (
            "opponent_double_three_defense",
            [
                (5, 7),
                (6, 7),
                (7, 5),
                (7, 6),
            ],
        ),
        (
            "opponent_cross_four_defense",
            [
                (4, 7),
                (5, 7),
                (6, 7),
                (7, 4),
                (7, 5),
                (7, 6),
            ],
        ),
    ]
    for current_player in (BLACK, WHITE):
        opponent = -int(current_player)
        for board_pattern_type, opponent_points in templates:
            stones = [(x, y, opponent) for x, y in opponent_points]
            samples.append(
                _make_sample(
                    stones=stones,
                    current_player=current_player,
                    target=target,
                    reason="missed_double_threat_defense",
                    board_pattern_type=board_pattern_type,
                    direction="fork",
                    rule_mode=rule_mode,
                    value=0.5,
                )
            )
    return samples


def generate_forced_block_replay_samples(rule_mode: str = "basic") -> list[AnnotatedMistakeSample]:
    """Build all forced-block replay samples."""
    return (
        generate_immediate_block_positions(rule_mode)
        + generate_open_four_defense_positions(rule_mode)
        + generate_blocked_four_defense_positions(rule_mode)
        + generate_double_threat_defense_positions(rule_mode)
    )


def _reason_distribution(samples: list[AnnotatedMistakeSample]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for item in samples:
        counter.update(item.metadata.get("reason", []))
    return dict(sorted(counter.items()))


def save_forced_block_replay_dataset(
    *,
    output_path: str,
    metadata_path: str,
    rule_mode: str = "basic",
) -> dict:
    """Save standalone forced-block replay dataset and metadata."""
    samples = generate_forced_block_replay_samples(rule_mode)
    shapes = save_annotated_samples(samples, output_path, metadata_path)
    summary = {
        "version": "forced_block_replay",
        "rule_mode": rule_mode,
        "output_path": output_path,
        "metadata_path": metadata_path,
        "total_samples": int(shapes["states"][0]),
        "shapes": shapes,
        "reason_distribution": _reason_distribution(samples),
    }
    summary_path = os.path.splitext(output_path)[0] + "_summary.json"
    parent = os.path.dirname(os.path.abspath(summary_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


__all__ = [
    "generate_immediate_block_positions",
    "generate_open_four_defense_positions",
    "generate_blocked_four_defense_positions",
    "generate_double_threat_defense_positions",
    "generate_forced_block_replay_samples",
    "save_forced_block_replay_dataset",
]
