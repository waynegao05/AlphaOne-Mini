"""Deterministic tactical curriculum samples for policy pretraining."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from engine.threats import is_forbidden_action
from game.board import BLACK, BOARD_SIZE, WHITE, Board
from game.encoder import action_to_index, encode_board, index_to_action
from selfplay.data_augmentation import augment_training_batch
from train.auxiliary_labels import THREAT_CHANNELS, build_auxiliary_labels
from train.tactical_distillation import make_policy_target, save_tactical_dataset


DEFAULT_CURRICULUM_DATA_PATH = os.path.join(
    "outputs", "supervised", "tactical_curriculum_latest.npz"
)

DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))


@dataclass(frozen=True)
class CurriculumSample:
    """A single deterministic tactical supervised sample."""

    category: str
    stones: tuple[tuple[int, int, int], ...]
    current_player: int
    action: int
    policy: np.ndarray
    value: float
    threat_labels: np.ndarray
    forbidden_labels: np.ndarray
    tactical_scores: np.ndarray
    forbidden_action: int = -1


def _board_from_stones(
    stones: Iterable[tuple[int, int, int]],
    current_player: int,
) -> Board:
    board = Board()
    count = 0
    for x, y, color in stones:
        if not board.in_bounds(x, y):
            raise ValueError(f"stone out of bounds: {(x, y, color)}")
        if board.grid[x][y] != 0:
            raise ValueError(f"duplicate stone: {(x, y, color)}")
        board.grid[x][y] = int(color)
        count += 1
    board.move_count = count
    board.current_player = int(current_player)
    board.last_move = None
    return board


def _line_stones(
    target: tuple[int, int],
    direction: tuple[int, int],
    offsets: Iterable[int],
    color: int,
) -> list[tuple[int, int, int]]:
    tx, ty = target
    dx, dy = direction
    return [(tx + offset * dx, ty + offset * dy, int(color)) for offset in offsets]


def _make_sample(
    category: str,
    stones: Iterable[tuple[int, int, int]],
    current_player: int,
    target: tuple[int, int],
    *,
    rule_mode: str,
    value: float,
    smoothing: float,
    extra_label_actions: Iterable[int] = (),
    forbidden_action: int = -1,
) -> CurriculumSample:
    stones_tuple = tuple(stones)
    board = _board_from_stones(stones_tuple, current_player)
    x, y = target
    if not board.is_legal_move(x, y):
        raise ValueError(f"target is not legal for {category}: {target}")
    action = action_to_index(x, y, BOARD_SIZE)
    actions = sorted({action, *[int(a) for a in extra_label_actions if int(a) >= 0]})
    aux = build_auxiliary_labels(
        board,
        current_player,
        rule_mode,
        actions=actions,
    )
    return CurriculumSample(
        category=category,
        stones=stones_tuple,
        current_player=int(current_player),
        action=action,
        policy=make_policy_target(action, smoothing=smoothing),
        value=float(value),
        threat_labels=aux["threat_labels"],
        forbidden_labels=aux["forbidden_labels"],
        tactical_scores=aux["tactical_scores"],
        forbidden_action=int(forbidden_action),
    )


def _safe_black_action(
    stones: Iterable[tuple[int, int, int]],
    forbidden_action: int,
    rule_mode: str,
) -> tuple[int, int]:
    board = _board_from_stones(stones, BLACK)
    preferred = [
        (7, 9),
        (9, 7),
        (5, 5),
        (8, 8),
        (6, 6),
        (0, 0),
        (14, 14),
    ]
    for x, y in preferred:
        if not board.is_legal_move(x, y):
            continue
        action = action_to_index(x, y)
        if action == forbidden_action:
            continue
        if not is_forbidden_action(board, action, BLACK, rule_mode):
            return x, y
    for x, y in board.get_legal_moves():
        action = action_to_index(x, y)
        if action != forbidden_action and not is_forbidden_action(board, action, BLACK, rule_mode):
            return x, y
    raise ValueError("no safe black action available")


def _basic_tactical_samples(rule_mode: str, smoothing: float) -> list[CurriculumSample]:
    samples: list[CurriculumSample] = []
    target = (7, 7)
    for direction in DIRECTIONS:
        for color in (BLACK, WHITE):
            opponent = -color
            samples.append(
                _make_sample(
                    "immediate_win",
                    _line_stones(target, direction, (-4, -3, -2, -1), color),
                    color,
                    target,
                    rule_mode=rule_mode,
                    value=1.0,
                    smoothing=smoothing,
                )
            )
            samples.append(
                _make_sample(
                    "must_block_win",
                    _line_stones(target, direction, (-4, -3, -2, -1), opponent),
                    color,
                    target,
                    rule_mode=rule_mode,
                    value=0.25,
                    smoothing=smoothing,
                )
            )
            samples.append(
                _make_sample(
                    "own_open_four",
                    _line_stones(target, direction, (-3, -2, -1), color),
                    color,
                    target,
                    rule_mode=rule_mode,
                    value=0.75,
                    smoothing=smoothing,
                )
            )
            samples.append(
                _make_sample(
                    "defend_open_four",
                    _line_stones(target, direction, (-3, -2, -1), opponent),
                    color,
                    target,
                    rule_mode=rule_mode,
                    value=0.25,
                    smoothing=smoothing,
                )
            )
            blocked_stones = _line_stones(target, direction, (-3, -2, -1), color)
            blocked_stones += _line_stones(target, direction, (-4,), opponent)
            samples.append(
                _make_sample(
                    "own_blocked_four",
                    blocked_stones,
                    color,
                    target,
                    rule_mode=rule_mode,
                    value=0.55,
                    smoothing=smoothing,
                )
            )
            defend_blocked = _line_stones(target, direction, (-3, -2, -1), opponent)
            defend_blocked += _line_stones(target, direction, (-4,), color)
            samples.append(
                _make_sample(
                    "defend_blocked_four",
                    defend_blocked,
                    color,
                    target,
                    rule_mode=rule_mode,
                    value=0.15,
                    smoothing=smoothing,
                )
            )
            samples.append(
                _make_sample(
                    "own_open_three",
                    _line_stones(target, direction, (-2, -1), color),
                    color,
                    target,
                    rule_mode=rule_mode,
                    value=0.35,
                    smoothing=smoothing,
                )
            )
            samples.append(
                _make_sample(
                    "defend_open_three",
                    _line_stones(target, direction, (-2, -1), opponent),
                    color,
                    target,
                    rule_mode=rule_mode,
                    value=0.05,
                    smoothing=smoothing,
                )
            )

            extension_target = (
                target[0] + direction[0],
                target[1] + direction[1],
            )
            win_extension_target = (
                target[0] + 2 * direction[0],
                target[1] + 2 * direction[1],
            )
            if 0 <= extension_target[0] < BOARD_SIZE and 0 <= extension_target[1] < BOARD_SIZE:
                samples.append(
                    _make_sample(
                        "extension_open_four",
                        _line_stones(extension_target, direction, (-3, -2, -1), color),
                        color,
                        extension_target,
                        rule_mode=rule_mode,
                        value=0.75,
                        smoothing=smoothing,
                    )
                )
                samples.append(
                    _make_sample(
                        "defend_extension_open_four",
                        _line_stones(extension_target, direction, (-3, -2, -1), opponent),
                        color,
                        extension_target,
                        rule_mode=rule_mode,
                        value=0.25,
                        smoothing=smoothing,
                    )
                )
            if 0 <= win_extension_target[0] < BOARD_SIZE and 0 <= win_extension_target[1] < BOARD_SIZE:
                samples.append(
                    _make_sample(
                        "extension_immediate_win",
                        _line_stones(win_extension_target, direction, (-4, -3, -2, -1), color),
                        color,
                        win_extension_target,
                        rule_mode=rule_mode,
                        value=1.0,
                        smoothing=smoothing,
                    )
                )
                samples.append(
                    _make_sample(
                        "defend_extension_immediate_win",
                        _line_stones(win_extension_target, direction, (-4, -3, -2, -1), opponent),
                        color,
                        win_extension_target,
                        rule_mode=rule_mode,
                        value=0.25,
                        smoothing=smoothing,
                    )
                )

    double_four_stones = (
        (5, 7, BLACK),
        (6, 7, BLACK),
        (8, 7, BLACK),
        (7, 5, BLACK),
        (7, 6, BLACK),
        (7, 8, BLACK),
    )
    double_three_stones = (
        (6, 7, BLACK),
        (8, 7, BLACK),
        (7, 6, BLACK),
        (7, 8, BLACK),
    )
    for color, stones in (
        (BLACK, double_four_stones),
        (WHITE, tuple((x, y, WHITE) for x, y, _ in double_four_stones)),
    ):
        opponent_stones = tuple((x, y, -color) for x, y, _ in stones)
        samples.append(
            _make_sample(
                "own_double_four",
                stones,
                color,
                target,
                rule_mode="basic" if color == BLACK and rule_mode == "forbidden" else rule_mode,
                value=0.85,
                smoothing=smoothing,
            )
        )
        samples.append(
            _make_sample(
                "defend_double_four",
                opponent_stones,
                color,
                target,
                rule_mode=rule_mode,
                value=0.25,
                smoothing=smoothing,
            )
        )
    for color, stones in (
        (BLACK, double_three_stones),
        (WHITE, tuple((x, y, WHITE) for x, y, _ in double_three_stones)),
    ):
        opponent_stones = tuple((x, y, -color) for x, y, _ in stones)
        samples.append(
            _make_sample(
                "own_double_three",
                stones,
                color,
                target,
                rule_mode="basic" if color == BLACK and rule_mode == "forbidden" else rule_mode,
                value=0.55,
                smoothing=smoothing,
            )
        )
        samples.append(
            _make_sample(
                "defend_double_three",
                opponent_stones,
                color,
                target,
                rule_mode=rule_mode,
                value=0.15,
                smoothing=smoothing,
            )
        )
    return samples


def _forbidden_samples(smoothing: float) -> list[CurriculumSample]:
    samples: list[CurriculumSample] = []
    target = (7, 7)
    forbidden_specs = [
        (
            "forbidden_overline",
            tuple((x, 7, BLACK) for x in (2, 3, 4, 5, 6)),
            (7, 7),
        ),
        (
            "forbidden_double_four",
            (
                (5, 7, BLACK),
                (6, 7, BLACK),
                (8, 7, BLACK),
                (7, 5, BLACK),
                (7, 6, BLACK),
                (7, 8, BLACK),
            ),
            target,
        ),
        (
            "forbidden_double_three",
            (
                (6, 7, BLACK),
                (8, 7, BLACK),
                (7, 6, BLACK),
                (7, 8, BLACK),
            ),
            target,
        ),
    ]
    for category, stones, forbidden_xy in forbidden_specs:
        forbidden_action = action_to_index(*forbidden_xy)
        safe_xy = _safe_black_action(stones, forbidden_action, "forbidden")
        samples.append(
            _make_sample(
                category,
                stones,
                BLACK,
                safe_xy,
                rule_mode="forbidden",
                value=-0.5,
                smoothing=smoothing,
                extra_label_actions=(forbidden_action,),
                forbidden_action=forbidden_action,
            )
        )
    return samples


def generate_tactical_curriculum_samples(
    rule_mode: str = "basic",
    repeats: int = 16,
    smoothing: float = 0.0,
    include_forbidden: bool = True,
) -> list[CurriculumSample]:
    """Generate deterministic tactical curriculum samples."""
    if rule_mode not in ("basic", "forbidden"):
        raise ValueError(f"unknown rule_mode: {rule_mode!r}")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    base = _basic_tactical_samples(rule_mode, smoothing)
    if include_forbidden:
        base.extend(_forbidden_samples(smoothing))
    samples: list[CurriculumSample] = []
    for _ in range(int(repeats)):
        samples.extend(base)
    return samples


def summarize_curriculum_arrays(arrays: dict[str, np.ndarray], samples: list[CurriculumSample]) -> dict:
    """Summarize sample counts and auxiliary-label density."""
    category_counts: dict[str, int] = {}
    for sample in samples:
        category_counts[sample.category] = category_counts.get(sample.category, 0) + 1
    threat_counts = {}
    for idx, name in enumerate(THREAT_CHANNELS):
        threat_counts[name] = int(np.count_nonzero(arrays["threat_labels"][:, idx] > 0.5))
    return {
        "num_samples": int(arrays["states"].shape[0]),
        "category_counts": category_counts,
        "threat_positive_counts": threat_counts,
        "forbidden_positive_count": int(np.count_nonzero(arrays["forbidden_labels"] > 0.5)),
        "value_distribution": {
            str(float(value)): int(count)
            for value, count in zip(*np.unique(arrays["values"], return_counts=True))
        },
    }


def samples_to_arrays(samples: list[CurriculumSample]) -> dict[str, np.ndarray]:
    """Convert curriculum samples to npz-compatible arrays."""
    if not samples:
        raise ValueError("no curriculum samples")
    states = []
    policies = []
    values = []
    threat_labels = []
    forbidden_labels = []
    tactical_scores = []
    for sample in samples:
        board = _board_from_stones(sample.stones, sample.current_player)
        states.append(encode_board(board, current_player=sample.current_player))
        policies.append(sample.policy)
        values.append([sample.value])
        threat_labels.append(sample.threat_labels)
        forbidden_labels.append(sample.forbidden_labels)
        tactical_scores.append(sample.tactical_scores)
    return {
        "states": np.stack(states).astype(np.float32),
        "policies": np.stack(policies).astype(np.float32),
        "values": np.asarray(values, dtype=np.float32),
        "threat_labels": np.stack(threat_labels).astype(np.float32),
        "forbidden_labels": np.stack(forbidden_labels).astype(np.float32),
        "tactical_scores": np.stack(tactical_scores).astype(np.float32),
    }


def generate_tactical_curriculum_dataset(
    output_path: str | None = None,
    rule_mode: str = "basic",
    repeats: int = 16,
    smoothing: float = 0.0,
    include_forbidden: bool = True,
    use_augmentation: bool = False,
) -> tuple[dict[str, np.ndarray], dict]:
    """Generate curriculum arrays, optionally save them, and return stats."""
    samples = generate_tactical_curriculum_samples(
        rule_mode=rule_mode,
        repeats=repeats,
        smoothing=smoothing,
        include_forbidden=include_forbidden,
    )
    arrays = samples_to_arrays(samples)
    if use_augmentation:
        arrays = augment_training_batch(
            arrays["states"],
            arrays["policies"],
            arrays["values"],
            threat_labels=arrays["threat_labels"],
            forbidden_labels=arrays["forbidden_labels"],
            tactical_scores=arrays["tactical_scores"],
        )
    if output_path:
        save_tactical_dataset(
            arrays["states"],
            arrays["policies"],
            arrays["values"],
            output_path,
            threat_labels=arrays["threat_labels"],
            forbidden_labels=arrays["forbidden_labels"],
            tactical_scores=arrays["tactical_scores"],
        )
    stats = summarize_curriculum_arrays(arrays, samples)
    return arrays, stats


__all__ = [
    "DEFAULT_CURRICULUM_DATA_PATH",
    "CurriculumSample",
    "generate_tactical_curriculum_samples",
    "generate_tactical_curriculum_dataset",
    "samples_to_arrays",
    "summarize_curriculum_arrays",
]
