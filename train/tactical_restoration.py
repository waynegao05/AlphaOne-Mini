"""Tactical-restoration branch data construction."""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from typing import Iterable, Mapping

from game.board import BLACK, BOARD_SIZE, WHITE, Board
from game.encoder import action_to_index
from train.mistake_mining import (
    MistakeSample,
    _create_teacher,
    _load_student_player,
    center_replay_samples,
    collect_mistake_samples_from_players,
    curriculum_replay_samples,
    fixed_tactical_validation_samples,
    make_mistake_sample,
    reason_distribution,
    save_mistake_samples,
)
from train.mistake_replay_balancer import apply_reason_weights, cap_reason_ratio
from train.progress import format_seconds, progress_print


DEFAULT_TACTICAL_RESTORATION_DATA_PATH = os.path.join(
    "outputs", "supervised", "tactical_restoration_latest.npz"
)
DEFAULT_TACTICAL_VALIDATION_PATH = os.path.join(
    "outputs", "supervised", "tactical_validation_set.npz"
)

DEFAULT_REASON_WEIGHTS = {
    "missed_immediate_block": 5.0,
    "missed_open_four_defense": 4.0,
    "missed_blocked_four_defense": 3.0,
    "tactical_draw_replay": 3.0,
    "curriculum_replay": 2.0,
    "center_replay": 1.0,
    "low_heuristic_move": 1.0,
}


def _tag(samples: Iterable[MistakeSample], reason: str) -> list[MistakeSample]:
    tagged: list[MistakeSample] = []
    for sample in samples:
        reasons = tuple(dict.fromkeys((*sample.reasons, reason)))
        tagged.append(replace(sample, reasons=reasons))
    return tagged


def _board(stones: Iterable[tuple[int, int, int]], current_player: int) -> Board:
    board = Board()
    count = 0
    for x, y, color in stones:
        board.grid[x][y] = int(color)
        count += 1
    board.move_count = count
    board.current_player = int(current_player)
    board.last_move = None
    return board


def defense_replay_samples(repeats: int = 16, rule_mode: str = "basic") -> list[MistakeSample]:
    """Create deterministic tactical-defense replay samples."""
    if repeats <= 0:
        return []
    base_cases = [
        (
            _board([(5, 7, WHITE), (6, 7, WHITE), (7, 7, WHITE), (8, 7, WHITE)], BLACK),
            action_to_index(9, 7, BOARD_SIZE),
            ("missed_immediate_block",),
        ),
        (
            _board([(5, 7, BLACK), (6, 7, BLACK), (7, 7, BLACK), (8, 7, BLACK)], BLACK),
            action_to_index(9, 7, BOARD_SIZE),
            ("missed_immediate_win",),
        ),
        (
            _board([(5, 7, WHITE), (6, 7, WHITE), (7, 7, WHITE)], BLACK),
            action_to_index(8, 7, BOARD_SIZE),
            ("missed_open_four_defense",),
        ),
        (
            _board([(5, 7, WHITE), (6, 7, WHITE), (7, 7, WHITE), (4, 7, BLACK)], BLACK),
            action_to_index(8, 7, BOARD_SIZE),
            ("missed_blocked_four_defense",),
        ),
    ]
    samples: list[MistakeSample] = []
    for _ in range(int(repeats)):
        for board, action, reasons in base_cases:
            samples.append(
                make_mistake_sample(
                    board,
                    teacher_action=action,
                    final_winner=0,
                    rule_mode=rule_mode,
                    reasons=reasons,
                )
            )
    return samples


def build_tactical_restoration_samples(
    tactical_samples: list[MistakeSample],
    *,
    replay_samples: Iterable[MistakeSample] = (),
    reason_weights: Mapping[str, float] | None = None,
    max_low_heuristic_ratio: float = 0.15,
) -> tuple[list[MistakeSample], dict]:
    """Apply restoration-specific weighting and low-heuristic capping."""
    weights = dict(DEFAULT_REASON_WEIGHTS)
    if reason_weights:
        weights.update({key: float(value) for key, value in reason_weights.items()})
    weighted, weight_summary = apply_reason_weights(tactical_samples, weights)
    capped, cap_summary = cap_reason_ratio(
        weighted,
        reason="low_heuristic_move",
        max_ratio=float(max_low_heuristic_ratio),
    )
    final = capped + list(replay_samples)
    return final, {
        "samples_before": len(tactical_samples),
        "reason_distribution_before": reason_distribution(tactical_samples),
        "reason_weighting": weight_summary,
        "low_heuristic_cap": cap_summary,
        "reason_distribution_after": reason_distribution(final),
        "final_samples": len(final),
    }


def build_tactical_restoration_dataset(
    v1_checkpoint: str,
    output_path: str = DEFAULT_TACTICAL_RESTORATION_DATA_PATH,
    validation_output: str = DEFAULT_TACTICAL_VALIDATION_PATH,
    curriculum_data: str = os.path.join("outputs", "supervised", "tactical_curriculum_latest.npz"),
    games: int = 20,
    rule_mode: str = "basic",
    num_simulations: int = 50,
    device: str = "cuda",
    max_moves: int = 80,
    min_score_gap: float = 5000.0,
    reason_weights: Mapping[str, float] | None = None,
    max_low_heuristic_ratio: float = 0.15,
    defense_repeats: int = 24,
    curriculum_replay_count: int = 512,
    center_replay_repeats: int = 128,
    stats_path: str | None = None,
) -> dict:
    """Build a Tactical-only restoration dataset from v1 draw/loss positions."""
    start = time.perf_counter()
    progress_print(
        f"START tactical_restoration_data games={games} checkpoint={v1_checkpoint}",
        "restore",
    )
    teacher = _create_teacher("tactical", rule_mode, device, num_simulations)

    draw_student = _load_student_player(v1_checkpoint, num_simulations, device)
    draw_samples, draw_summary = collect_mistake_samples_from_players(
        draw_student,
        teacher,
        games=int(games),
        rule_mode=rule_mode,
        max_moves=int(max_moves),
        min_score_gap=float(min_score_gap),
        collect_losses=False,
        include_draws=True,
    )
    draw_samples = _tag(draw_samples, "tactical_draw_replay")

    loss_student = _load_student_player(v1_checkpoint, num_simulations, device)
    loss_samples, loss_summary = collect_mistake_samples_from_players(
        loss_student,
        teacher,
        games=int(games),
        rule_mode=rule_mode,
        max_moves=int(max_moves),
        min_score_gap=float(min_score_gap),
        collect_losses=True,
        include_draws=False,
    )
    loss_samples = _tag(loss_samples, "tactical_loss_near_end")

    replay = []
    defense = defense_replay_samples(defense_repeats, rule_mode=rule_mode)
    replay.extend(defense)
    curriculum = curriculum_replay_samples(curriculum_data, max_samples=int(curriculum_replay_count))
    replay.extend(curriculum)
    center = center_replay_samples(int(center_replay_repeats))
    replay.extend(center)

    final_samples, balance_summary = build_tactical_restoration_samples(
        draw_samples + loss_samples,
        replay_samples=replay,
        reason_weights=reason_weights,
        max_low_heuristic_ratio=max_low_heuristic_ratio,
    )
    shapes = save_mistake_samples(final_samples, output_path)

    validation = fixed_tactical_validation_samples(rule_mode)
    validation.extend(draw_samples[: min(96, len(draw_samples))])
    validation_shapes = save_mistake_samples(validation, validation_output)

    summary = {
        "version": "tactical_restoration",
        "v1_checkpoint": v1_checkpoint,
        "output_path": output_path,
        "validation_output": validation_output,
        "games": int(games),
        "draw_samples": len(draw_samples),
        "loss_samples": len(loss_samples),
        "defense_replay_samples": len(defense),
        "curriculum_replay_samples": len(curriculum),
        "center_replay_samples": len(center),
        "final_samples": int(shapes["states"][0]),
        "validation_samples": int(validation_shapes["states"][0]),
        "balance_summary": balance_summary,
        "reason_distribution": balance_summary["reason_distribution_after"],
        "draw_collection": draw_summary,
        "loss_collection": loss_summary,
        "elapsed_sec": time.perf_counter() - start,
    }
    if stats_path:
        os.makedirs(os.path.dirname(os.path.abspath(stats_path)), exist_ok=True)
        with open(stats_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
    progress_print(
        f"DONE tactical_restoration_data final={summary['final_samples']} "
        f"draw={len(draw_samples)} loss={len(loss_samples)} "
        f"elapsed={format_seconds(summary['elapsed_sec'])}",
        "restore",
    )
    return summary


__all__ = [
    "DEFAULT_REASON_WEIGHTS",
    "DEFAULT_TACTICAL_RESTORATION_DATA_PATH",
    "DEFAULT_TACTICAL_VALIDATION_PATH",
    "build_tactical_restoration_dataset",
    "build_tactical_restoration_samples",
    "defense_replay_samples",
]
