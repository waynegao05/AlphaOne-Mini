"""Mine losing positions against tactical teachers and build correction data."""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np
import torch

from engine.candidate_moves import generate_candidate_moves
from engine.heuristic import evaluate_move_heuristic
from engine.hybrid_player import HybridPlayer
from engine.tactical_player import TacticalPlayer
from engine.threats import (
    find_blocked_four_moves,
    find_immediate_blocking_moves,
    find_immediate_winning_moves,
    find_open_four_moves,
    find_open_three_moves,
)
from evaluate.players import ModelMCTSPlayer
from game.board import BLACK, BOARD_SIZE, WHITE, Board
from game.encoder import action_to_index, encode_board, index_to_action
from game.rules_basic import _find_any_winner, check_winner, is_game_over
from game.rules_forbidden import get_game_result_forbidden
from model.checkpoint import load_checkpoint
from model.model_factory import create_model_from_metadata
from model.policy_value_net import PolicyValueNet
from train.auxiliary_labels import build_auxiliary_labels
from train.progress import format_seconds, progress_print
from train.tactical_distillation import make_policy_target, save_tactical_dataset


DEFAULT_MISTAKE_DATA_PATH = os.path.join(
    "outputs", "supervised", "mistake_mining_latest.npz"
)
DEFAULT_MISTAKE_STATS_PATH = os.path.join(
    "outputs", "supervised", "mistake_mining_latest_stats.json"
)

CRITICAL_REASONS = (
    "missed_immediate_block",
    "missed_immediate_win",
    "missed_open_four_defense",
    "missed_open_four_attack",
    "missed_blocked_four_defense",
)


@dataclass
class MistakeSample:
    state: np.ndarray
    policy: np.ndarray
    value: float
    threat_labels: np.ndarray
    forbidden_labels: np.ndarray
    tactical_scores: np.ndarray
    teacher_action: int
    reasons: tuple[str, ...]


@dataclass
class _CandidateRecord:
    board: Board
    student_action: int
    teacher_action: int
    current_player: int
    ply: int
    reasons: tuple[str, ...]
    score_gap: float


def _game_status(board: Board, rule_mode: str) -> tuple[bool, int]:
    if rule_mode == "forbidden":
        result = get_game_result_forbidden(board, board.last_move)
        if result.is_over:
            return True, 0 if result.winner is None else int(result.winner)
        return False, 0
    if rule_mode != "basic":
        raise ValueError(f"unknown rule_mode: {rule_mode!r}")
    if is_game_over(board, board.last_move):
        winner = check_winner(board, board.last_move)
        if winner == 0:
            winner = _find_any_winner(board)
        return True, int(winner)
    return False, 0


def _sample_value(final_winner: int, current_player: int) -> float:
    if final_winner == 0:
        return 0.0
    return 1.0 if int(final_winner) == int(current_player) else -1.0


def _safe_select(player, board: Board) -> int | None:
    action = player.select_action(board)
    if action is None:
        return None
    action = int(action)
    x, y = index_to_action(action, BOARD_SIZE)
    if not board.is_legal_move(x, y):
        return None
    return action


def _tactical_reasons(
    board: Board,
    current_player: int,
    student_action: int,
    teacher_action: int,
    rule_mode: str,
    min_score_gap: float,
) -> tuple[tuple[str, ...], float]:
    reasons: list[str] = []
    if teacher_action != student_action:
        wins = set(find_immediate_winning_moves(board, current_player, rule_mode))
        blocks = set(find_immediate_blocking_moves(board, current_player, rule_mode))
        own_open_fours = set(find_open_four_moves(board, current_player, rule_mode))
        opp_open_fours = set(find_open_four_moves(board, -current_player, rule_mode))
        own_blocked_fours = set(find_blocked_four_moves(board, current_player, rule_mode))
        opp_blocked_fours = set(find_blocked_four_moves(board, -current_player, rule_mode))
        own_open_threes = set(find_open_three_moves(board, current_player, rule_mode))
        opp_open_threes = set(find_open_three_moves(board, -current_player, rule_mode))
        if wins and student_action not in wins:
            reasons.append("missed_immediate_win")
        if blocks and student_action not in blocks:
            reasons.append("missed_immediate_block")
        if own_open_fours and student_action not in own_open_fours:
            reasons.append("missed_open_four_attack")
        if opp_open_fours and student_action not in opp_open_fours:
            reasons.append("missed_open_four_defense")
        if own_blocked_fours and student_action not in own_blocked_fours:
            reasons.append("missed_blocked_four_attack")
        if opp_blocked_fours and student_action not in opp_blocked_fours:
            reasons.append("missed_blocked_four_defense")
        if own_open_threes and student_action not in own_open_threes:
            reasons.append("missed_open_three_attack")
        if opp_open_threes and student_action not in opp_open_threes:
            reasons.append("missed_open_three_defense")

    teacher_score = evaluate_move_heuristic(board, teacher_action, current_player, rule_mode)
    student_score = evaluate_move_heuristic(board, student_action, current_player, rule_mode)
    score_gap = float(teacher_score - student_score)
    if teacher_action != student_action and score_gap >= float(min_score_gap):
        reasons.append("low_heuristic_move")
    if teacher_action != student_action and not reasons:
        reasons.append("teacher_student_disagree")
    return tuple(dict.fromkeys(reasons)), score_gap


def make_mistake_sample(
    board: Board,
    teacher_action: int,
    final_winner: int,
    rule_mode: str = "basic",
    reasons: Iterable[str] = (),
) -> MistakeSample:
    """Create a supervised correction sample from a board snapshot."""
    current_player = int(board.current_player)
    teacher_action = int(teacher_action)
    x, y = index_to_action(teacher_action, BOARD_SIZE)
    if not board.is_legal_move(x, y):
        raise ValueError(f"teacher_action is not legal: {teacher_action}")
    candidates = set(
        generate_candidate_moves(board, radius=2, max_candidates=96, include_center=True)
    )
    candidates.add(teacher_action)
    aux = build_auxiliary_labels(
        board,
        current_player,
        rule_mode,
        actions=sorted(candidates),
    )
    return MistakeSample(
        state=encode_board(board, current_player=current_player).astype(np.float32),
        policy=make_policy_target(teacher_action, smoothing=0.0),
        value=_sample_value(int(final_winner), current_player),
        threat_labels=aux["threat_labels"],
        forbidden_labels=aux["forbidden_labels"],
        tactical_scores=aux["tactical_scores"],
        teacher_action=teacher_action,
        reasons=tuple(reasons),
    )


def _samples_to_arrays(samples: list[MistakeSample]) -> dict[str, np.ndarray]:
    if not samples:
        return {
            "states": np.zeros((0, 4, BOARD_SIZE, BOARD_SIZE), dtype=np.float32),
            "policies": np.zeros((0, BOARD_SIZE * BOARD_SIZE), dtype=np.float32),
            "values": np.zeros((0, 1), dtype=np.float32),
            "threat_labels": np.zeros((0, 12, BOARD_SIZE, BOARD_SIZE), dtype=np.float32),
            "forbidden_labels": np.zeros((0, 1, BOARD_SIZE, BOARD_SIZE), dtype=np.float32),
            "tactical_scores": np.zeros((0, BOARD_SIZE * BOARD_SIZE), dtype=np.float32),
        }
    return {
        "states": np.stack([sample.state for sample in samples]).astype(np.float32),
        "policies": np.stack([sample.policy for sample in samples]).astype(np.float32),
        "values": np.asarray([[sample.value] for sample in samples], dtype=np.float32),
        "threat_labels": np.stack([sample.threat_labels for sample in samples]).astype(np.float32),
        "forbidden_labels": np.stack([sample.forbidden_labels for sample in samples]).astype(np.float32),
        "tactical_scores": np.stack([sample.tactical_scores for sample in samples]).astype(np.float32),
    }


def _save_arrays(arrays: dict[str, np.ndarray], path: str) -> None:
    save_tactical_dataset(
        arrays["states"],
        arrays["policies"],
        arrays["values"],
        path,
        threat_labels=arrays["threat_labels"],
        forbidden_labels=arrays["forbidden_labels"],
        tactical_scores=arrays["tactical_scores"],
    )


def _load_array_dict(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key], dtype=np.float32) for key in data.files}


def reason_distribution(samples: Iterable[MistakeSample]) -> dict[str, int]:
    """Count sample reasons for reporting and oversampling diagnostics."""
    counts: dict[str, int] = {}
    for sample in samples:
        for reason in sample.reasons:
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def oversample_critical_samples(
    samples: list[MistakeSample],
    critical_repeat: int = 3,
    critical_reasons: Iterable[str] = CRITICAL_REASONS,
) -> list[MistakeSample]:
    """Repeat high-value tactical errors while leaving non-critical samples unchanged.

    `critical_repeat=3` means a critical sample appears three total times.
    """
    repeat = max(1, int(critical_repeat))
    critical = set(critical_reasons)
    expanded: list[MistakeSample] = []
    for sample in samples:
        copies = repeat if critical.intersection(sample.reasons) else 1
        expanded.extend([sample] * copies)
    return expanded


def center_replay_samples(repeats: int = 64) -> list[MistakeSample]:
    """Build empty-board H8 replay samples to preserve center opening preference."""
    if repeats <= 0:
        return []
    center_action = action_to_index(7, 7, BOARD_SIZE)
    return [
        make_mistake_sample(
            Board(),
            teacher_action=center_action,
            final_winner=0,
            rule_mode="basic",
            reasons=("center_replay",),
        )
        for _ in range(int(repeats))
    ]


def curriculum_replay_samples(
    curriculum_path: str,
    max_samples: int = 512,
) -> list[MistakeSample]:
    """Load a subset of curriculum npz samples as replay data."""
    if max_samples <= 0:
        return []
    arrays = _load_array_dict(curriculum_path)
    required = (
        "states",
        "policies",
        "values",
        "threat_labels",
        "forbidden_labels",
        "tactical_scores",
    )
    missing = [key for key in required if key not in arrays]
    if missing:
        raise ValueError(f"curriculum dataset missing keys: {missing}")
    n = min(int(max_samples), int(arrays["states"].shape[0]))
    samples: list[MistakeSample] = []
    for idx in range(n):
        policy = np.asarray(arrays["policies"][idx], dtype=np.float32)
        value = float(np.asarray(arrays["values"][idx]).reshape(-1)[0])
        samples.append(
            MistakeSample(
                state=np.asarray(arrays["states"][idx], dtype=np.float32),
                policy=policy,
                value=value,
                threat_labels=np.asarray(arrays["threat_labels"][idx], dtype=np.float32),
                forbidden_labels=np.asarray(arrays["forbidden_labels"][idx], dtype=np.float32),
                tactical_scores=np.asarray(arrays["tactical_scores"][idx], dtype=np.float32),
                teacher_action=int(np.argmax(policy)),
                reasons=("curriculum_replay",),
            )
        )
    return samples


def _tag_samples(samples: Iterable[MistakeSample], reason: str) -> list[MistakeSample]:
    tagged: list[MistakeSample] = []
    for sample in samples:
        reasons = tuple(dict.fromkeys((*sample.reasons, reason)))
        tagged.append(replace(sample, reasons=reasons))
    return tagged


def save_mistake_samples(samples: list[MistakeSample], path: str) -> dict:
    """Save MistakeSample objects as the project npz training format."""
    arrays = _samples_to_arrays(samples)
    _save_arrays(arrays, path)
    return {key: list(value.shape) for key, value in arrays.items()}


def fixed_tactical_validation_samples(rule_mode: str = "basic") -> list[MistakeSample]:
    """Small fixed validation set for must-win, must-block and center cases."""
    def board_from(stones: list[tuple[int, int, int]], current_player: int) -> Board:
        board = Board()
        for x, y, color in stones:
            board.grid[x][y] = int(color)
        board.move_count = len(stones)
        board.current_player = int(current_player)
        board.last_move = None
        return board

    cases = [
        (board_from([(5, 7, BLACK), (6, 7, BLACK), (7, 7, BLACK), (8, 7, BLACK)], BLACK), action_to_index(9, 7), "fixed_own_four_win"),
        (board_from([(5, 7, WHITE), (6, 7, WHITE), (7, 7, WHITE), (8, 7, WHITE)], BLACK), action_to_index(9, 7), "fixed_block_opponent_four"),
        (board_from([(5, 7, BLACK), (6, 7, BLACK), (7, 7, BLACK)], BLACK), action_to_index(8, 7), "fixed_own_open_four"),
        (board_from([(5, 7, WHITE), (6, 7, WHITE), (7, 7, WHITE)], BLACK), action_to_index(8, 7), "fixed_defend_open_four"),
        (Board(), action_to_index(7, 7), "fixed_center_opening"),
    ]
    return [
        make_mistake_sample(
            board,
            teacher_action=action,
            final_winner=0,
            rule_mode=rule_mode,
            reasons=(reason,),
        )
        for board, action, reason in cases
    ]


def append_center_replay_samples(
    data_path: str,
    output_path: str | None = None,
    repeats: int = 64,
) -> int:
    """Append empty-board center replay samples to avoid opening drift."""
    if repeats <= 0:
        return 0
    output_path = output_path or data_path
    arrays = _load_array_dict(data_path)
    replay = center_replay_samples(int(repeats))
    replay_arrays = _samples_to_arrays(replay)
    merged = {
        key: np.concatenate([arrays[key], replay_arrays[key]], axis=0)
        for key in (
            "states",
            "policies",
            "values",
            "threat_labels",
            "forbidden_labels",
            "tactical_scores",
        )
    }
    _save_arrays(merged, output_path)
    return int(repeats)


def _write_stats(summary: dict, stats_path: str | None) -> None:
    if not stats_path:
        return
    parent = os.path.dirname(os.path.abspath(stats_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def _apply_random_opening(board: Board, rng: random.Random, moves: int) -> int:
    """Apply a small legal random opening to diversify deterministic teachers."""
    applied = 0
    for _ in range(max(0, int(moves))):
        candidates = generate_candidate_moves(
            board,
            radius=2,
            max_candidates=32,
            include_center=True,
        )
        legal = []
        for action in candidates:
            x, y = index_to_action(int(action), BOARD_SIZE)
            if board.is_legal_move(x, y):
                legal.append(int(action))
        if not legal:
            break
        action = rng.choice(legal)
        x, y = index_to_action(action, BOARD_SIZE)
        board.place_stone(x, y)
        applied += 1
        over, _ = _game_status(board, "basic")
        if over:
            break
    return applied


def collect_mistake_samples_from_players(
    student_player,
    teacher_player,
    games: int = 20,
    rule_mode: str = "basic",
    max_moves: int = 80,
    min_score_gap: float = 5_000.0,
    endgame_window: tuple[int, int] = (5, 15),
    collect_losses: bool = True,
    include_draws: bool = False,
    opening_seed: int | None = None,
    opening_moves: int = 0,
) -> tuple[list[MistakeSample], dict]:
    """Collect teacher-labelled samples from games the student loses or draws."""
    if games <= 0:
        raise ValueError("games must be positive")
    start = time.perf_counter()
    progress_print(
        f"START mistake_collection games={games} rule_mode={rule_mode} max_moves={max_moves}",
        "mistake",
    )
    samples: list[MistakeSample] = []
    reason_counts: dict[str, int] = {}
    game_summaries = []

    for game_idx in range(int(games)):
        board = Board()
        applied_opening_moves = 0
        if opening_seed is not None and opening_moves > 0:
            rng = random.Random(int(opening_seed) + game_idx)
            applied_opening_moves = _apply_random_opening(board, rng, opening_moves)
        student_color = BLACK if game_idx % 2 == 0 else WHITE
        teacher_color = -student_color
        candidates: list[_CandidateRecord] = []
        winner = 0
        moves = 0
        for ply in range(int(max_moves)):
            over, winner = _game_status(board, rule_mode)
            if over:
                break
            current_player = int(board.current_player)
            if current_player == student_color:
                snapshot = board.copy()
                student_action = _safe_select(student_player, board)
                teacher_action = _safe_select(teacher_player, snapshot)
                if student_action is None:
                    winner = teacher_color
                    break
                if teacher_action is None:
                    teacher_action = student_action
                reasons, score_gap = _tactical_reasons(
                    snapshot,
                    current_player,
                    student_action,
                    teacher_action,
                    rule_mode,
                    min_score_gap,
                )
                candidates.append(
                    _CandidateRecord(
                        board=snapshot,
                        student_action=student_action,
                        teacher_action=teacher_action,
                        current_player=current_player,
                        ply=ply,
                        reasons=reasons,
                        score_gap=score_gap,
                    )
                )
                action = student_action
            else:
                action = _safe_select(teacher_player, board)
                if action is None:
                    winner = student_color
                    break
            x, y = index_to_action(int(action), BOARD_SIZE)
            board.place_stone(x, y)
            moves += 1
        else:
            winner = 0

        over, final_winner = _game_status(board, rule_mode)
        if over:
            winner = final_winner
        student_lost = winner != 0 and winner != student_color
        student_draw = winner == 0
        collect_game = (bool(collect_losses) and student_lost) or (
            bool(include_draws) and student_draw
        )
        added = 0
        if collect_game:
            final_ply = moves
            min_window, max_window = endgame_window
            selected: list[_CandidateRecord] = []
            for record in candidates:
                distance_to_end = final_ply - record.ply
                in_endgame_window = min_window <= distance_to_end <= max_window
                if record.reasons or in_endgame_window:
                    selected.append(record)
            if not selected:
                selected = candidates[-max_window:]
            seen: set[tuple[int, int]] = set()
            for record in selected:
                key = (record.ply, record.teacher_action)
                if key in seen:
                    continue
                seen.add(key)
                sample_reasons = list(record.reasons)
                distance_to_end = final_ply - record.ply
                if (
                    student_lost
                    and min_window <= distance_to_end <= max_window
                    and "near_loss_position" not in sample_reasons
                ):
                    sample_reasons.append("near_loss_position")
                sample = make_mistake_sample(
                    record.board,
                    teacher_action=record.teacher_action,
                    final_winner=winner,
                    rule_mode=rule_mode,
                    reasons=sample_reasons,
                )
                samples.append(sample)
                added += 1
                for reason in sample.reasons:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
        game_summaries.append(
            {
                "game": game_idx + 1,
                "student_color": student_color,
                "winner": winner,
                "moves": moves,
                "opening_moves": applied_opening_moves,
                "student_lost": student_lost,
                "student_draw": student_draw,
                "samples_added": added,
            }
        )
        progress_print(
            f"game {game_idx + 1}/{games} complete winner={winner} "
            f"student_color={student_color} moves={moves} added={added}",
            "mistake",
        )

    summary = {
        "num_samples": int(len(samples)),
        "games": int(games),
        "rule_mode": rule_mode,
        "reason_counts": reason_counts,
        "teacher": getattr(teacher_player, "name", type(teacher_player).__name__),
        "student": getattr(student_player, "name", type(student_player).__name__),
        "game_summaries": game_summaries,
        "elapsed_sec": time.perf_counter() - start,
    }
    progress_print(
        f"DONE mistake_collection samples={summary['num_samples']} "
        f"elapsed={format_seconds(summary['elapsed_sec'])}",
        "mistake",
    )
    return samples, summary


def collect_mistake_positions_from_players(
    student_player,
    teacher_player,
    games: int = 20,
    rule_mode: str = "basic",
    output_path: str = DEFAULT_MISTAKE_DATA_PATH,
    max_moves: int = 80,
    min_score_gap: float = 5_000.0,
    endgame_window: tuple[int, int] = (5, 15),
    stats_path: str | None = DEFAULT_MISTAKE_STATS_PATH,
) -> dict:
    """Collect teacher-labelled samples from games the student loses and save npz."""
    samples, summary = collect_mistake_samples_from_players(
        student_player,
        teacher_player,
        games=games,
        rule_mode=rule_mode,
        max_moves=max_moves,
        min_score_gap=min_score_gap,
        endgame_window=endgame_window,
    )
    arrays = _samples_to_arrays(samples)
    _save_arrays(arrays, output_path)
    summary["output_path"] = output_path
    summary["num_samples"] = int(arrays["states"].shape[0])
    _write_stats(summary, stats_path)
    return summary


def _load_student_player(
    checkpoint_path: str,
    num_simulations: int,
    device: str,
):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"student checkpoint not found: {checkpoint_path}")
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=device)
    model = create_model_from_metadata(state.get("metadata", {}), fallback_model_type="advanced")
    load_checkpoint(model, checkpoint_path, device=device)
    model.eval()
    return ModelMCTSPlayer(
        model=model,
        num_simulations=int(num_simulations),
        device=device,
        name="latest_advanced_ModelMCTSPlayer",
    )


def _create_teacher(teacher_type: str, rule_mode: str, device: str, num_simulations: int):
    normalized = teacher_type.lower()
    if normalized == "tactical":
        return TacticalPlayer(rule_mode=rule_mode, name="TacticalPlayer")
    if normalized == "hybrid":
        return HybridPlayer(
            model=PolicyValueNet(),
            num_simulations=max(1, min(int(num_simulations), 10)),
            device=device,
            rule_mode=rule_mode,
            name="HybridPlayer",
        )
    raise ValueError("teacher_type must be 'tactical' or 'hybrid'")


def collect_mistake_positions(
    student_checkpoint: str,
    teacher_type: str = "tactical",
    games: int = 20,
    rule_mode: str = "basic",
    num_simulations: int = 50,
    device: str = "cuda",
    output_path: str = DEFAULT_MISTAKE_DATA_PATH,
    max_moves: int = 80,
    min_score_gap: float = 5_000.0,
    stats_path: str | None = DEFAULT_MISTAKE_STATS_PATH,
) -> dict:
    """Load a student checkpoint, play against a teacher, and save mistake data."""
    student = _load_student_player(student_checkpoint, num_simulations, device)
    teacher = _create_teacher(teacher_type, rule_mode, device, num_simulations)
    summary = collect_mistake_positions_from_players(
        student,
        teacher,
        games=games,
        rule_mode=rule_mode,
        output_path=output_path,
        max_moves=max_moves,
        min_score_gap=min_score_gap,
        stats_path=stats_path,
    )
    summary["student_checkpoint"] = student_checkpoint
    summary["teacher_type"] = teacher_type
    _write_stats(summary, stats_path)
    return summary


def build_mistake_training_dataset_v2(
    student_checkpoint: str,
    teachers: Iterable[str] = ("tactical", "hybrid"),
    games_per_teacher: int = 20,
    seeds: Iterable[int] = (2026,),
    rule_mode: str = "basic",
    num_simulations: int = 50,
    device: str = "cuda",
    output_path: str = DEFAULT_MISTAKE_DATA_PATH,
    max_moves: int = 80,
    min_score_gap: float = 5_000.0,
    include_center_replay: bool = True,
    center_replay_repeats: int = 128,
    include_curriculum_replay: bool = False,
    curriculum_data: str | None = None,
    curriculum_replay_count: int = 512,
    oversample_critical: bool = False,
    critical_repeat: int = 3,
    stats_path: str | None = DEFAULT_MISTAKE_STATS_PATH,
) -> dict:
    """Build a v2 mistake-mining dataset with multiple teachers and replay data."""
    teachers = tuple(str(item).lower() for item in teachers)
    seeds = tuple(int(seed) for seed in seeds)
    if not teachers:
        raise ValueError("at least one teacher is required")
    if not seeds:
        raise ValueError("at least one seed is required")

    all_samples: list[MistakeSample] = []
    collection_summaries: list[dict] = []
    start = time.perf_counter()
    progress_print(
        f"START mistake_v2 teachers={teachers} seeds={seeds} games_per_teacher={games_per_teacher}",
        "mistake",
    )

    for teacher_type in teachers:
        for seed in seeds:
            student = _load_student_player(student_checkpoint, num_simulations, device)
            teacher = _create_teacher(teacher_type, rule_mode, device, num_simulations)
            samples, summary = collect_mistake_samples_from_players(
                student,
                teacher,
                games=int(games_per_teacher),
                rule_mode=rule_mode,
                max_moves=int(max_moves),
                min_score_gap=float(min_score_gap),
                include_draws=True,
                opening_seed=int(seed),
                opening_moves=2,
            )
            summary["teacher_type"] = teacher_type
            summary["seed"] = int(seed)
            collection_summaries.append(summary)
            all_samples.extend(samples)

    raw_samples = list(all_samples)
    reason_before = reason_distribution(raw_samples)
    critical_count = sum(
        bool(set(sample.reasons).intersection(CRITICAL_REASONS))
        for sample in raw_samples
    )

    if oversample_critical:
        all_samples = oversample_critical_samples(
            all_samples,
            critical_repeat=int(critical_repeat),
        )
    oversampled_count = len(all_samples) - len(raw_samples)

    curriculum_added = 0
    if include_curriculum_replay:
        if not curriculum_data:
            raise ValueError("curriculum_data is required when include_curriculum_replay=True")
        replay = curriculum_replay_samples(
            curriculum_data,
            max_samples=int(curriculum_replay_count),
        )
        curriculum_added = len(replay)
        all_samples.extend(replay)

    center_added = 0
    if include_center_replay:
        replay = center_replay_samples(int(center_replay_repeats))
        center_added = len(replay)
        all_samples.extend(replay)

    if not all_samples:
        raise RuntimeError("mistake v2 produced no samples")

    arrays = _samples_to_arrays(all_samples)
    _save_arrays(arrays, output_path)
    summary = {
        "student_checkpoint": student_checkpoint,
        "output_path": output_path,
        "teachers": list(teachers),
        "seeds": list(seeds),
        "games_per_teacher": int(games_per_teacher),
        "raw_mistake_samples": len(raw_samples),
        "critical_samples": int(critical_count),
        "critical_repeat": int(critical_repeat),
        "oversample_critical": bool(oversample_critical),
        "oversampled_samples_added": int(oversampled_count),
        "curriculum_replay_samples": int(curriculum_added),
        "center_replay_samples": int(center_added),
        "final_samples": int(arrays["states"].shape[0]),
        "reason_distribution_before": reason_before,
        "reason_distribution_after": reason_distribution(all_samples),
        "collections": collection_summaries,
        "rule_mode": rule_mode,
        "num_simulations": int(num_simulations),
        "max_moves": int(max_moves),
        "elapsed_sec": time.perf_counter() - start,
    }
    _write_stats(summary, stats_path)
    progress_print(
        f"DONE mistake_v2 raw={summary['raw_mistake_samples']} final={summary['final_samples']} "
        f"critical={summary['critical_samples']} curriculum={curriculum_added} "
        f"center={center_added} elapsed={format_seconds(summary['elapsed_sec'])} output={output_path}",
        "mistake",
    )
    return summary


def build_mistake_training_dataset_v3(
    student_checkpoint: str,
    teachers: Iterable[str] = ("tactical", "hybrid"),
    games_per_teacher: int = 30,
    seeds: Iterable[int] = (2026, 7, 21),
    rule_mode: str = "basic",
    num_simulations: int = 50,
    device: str = "cuda",
    output_path: str = DEFAULT_MISTAKE_DATA_PATH,
    max_moves: int = 80,
    min_score_gap: float = 5_000.0,
    teacher_balance: dict[str, float] | None = None,
    reason_weights: dict[str, float] | None = None,
    max_low_heuristic_ratio: float = 0.25,
    include_center_replay: bool = True,
    center_replay_repeats: int = 128,
    include_curriculum_replay: bool = True,
    curriculum_data: str | None = None,
    curriculum_replay_count: int = 512,
    include_v1_tactical_draw_replay: bool = False,
    v1_checkpoint: str | None = None,
    v1_draw_replay_games: int = 20,
    validation_teacher: str = "tactical",
    validation_output: str = os.path.join("outputs", "supervised", "tactical_validation_set.npz"),
    validation_holdout: int = 96,
    stats_path: str | None = DEFAULT_MISTAKE_STATS_PATH,
) -> dict:
    """Build a teacher-balanced v3 mistake-mining dataset and validation set."""
    from train.mistake_replay_balancer import build_teacher_balanced_replay

    teachers = tuple(str(item).lower() for item in teachers)
    seeds = tuple(int(seed) for seed in seeds)
    if not teachers:
        raise ValueError("at least one teacher is required")
    start = time.perf_counter()
    progress_print(
        f"START mistake_v3 teachers={teachers} seeds={seeds} games_per_teacher={games_per_teacher}",
        "mistake",
    )

    teacher_groups: dict[str, list[MistakeSample]] = {teacher: [] for teacher in teachers}
    collection_summaries: list[dict] = []
    for teacher_type in teachers:
        for seed in seeds:
            student = _load_student_player(student_checkpoint, num_simulations, device)
            teacher = _create_teacher(teacher_type, rule_mode, device, num_simulations)
            samples, summary = collect_mistake_samples_from_players(
                student,
                teacher,
                games=int(games_per_teacher),
                rule_mode=rule_mode,
                max_moves=int(max_moves),
                min_score_gap=float(min_score_gap),
                collect_losses=True,
                include_draws=True,
                opening_seed=int(seed),
                opening_moves=2,
            )
            tagged = _tag_samples(samples, f"teacher_{teacher_type}")
            teacher_groups[teacher_type].extend(tagged)
            summary["teacher_type"] = teacher_type
            summary["seed"] = int(seed)
            collection_summaries.append(summary)

    validation_samples = fixed_tactical_validation_samples(rule_mode)
    validation_key = str(validation_teacher or "tactical").lower()
    heldout = []
    if validation_key in teacher_groups and validation_holdout > 0:
        count = min(int(validation_holdout), len(teacher_groups[validation_key]))
        heldout = teacher_groups[validation_key][:count]
        teacher_groups[validation_key] = teacher_groups[validation_key][count:]
        validation_samples.extend(_tag_samples(heldout, "tactical_validation_holdout"))

    replay_samples: list[MistakeSample] = []
    v1_draw_count = 0
    if include_v1_tactical_draw_replay:
        if not v1_checkpoint:
            raise ValueError("v1_checkpoint is required for v1 tactical draw replay")
        v1_student = _load_student_player(v1_checkpoint, num_simulations, device)
        tactical_teacher = _create_teacher("tactical", rule_mode, device, num_simulations)
        v1_samples, v1_summary = collect_mistake_samples_from_players(
            v1_student,
            tactical_teacher,
            games=int(v1_draw_replay_games),
            rule_mode=rule_mode,
            max_moves=int(max_moves),
            min_score_gap=float(min_score_gap),
            collect_losses=False,
            include_draws=True,
            opening_seed=None,
            opening_moves=0,
        )
        replay = _tag_samples(v1_samples, "v1_tactical_draw_replay")
        v1_draw_count = len(replay)
        replay_samples.extend(replay)
        validation_samples.extend(_tag_samples(v1_samples[: min(64, len(v1_samples))], "v1_tactical_draw_validation"))
        collection_summaries.append({**v1_summary, "teacher_type": "tactical", "source": "v1_tactical_draw_replay"})

    curriculum_added = 0
    if include_curriculum_replay:
        if not curriculum_data:
            raise ValueError("curriculum_data is required when include_curriculum_replay=True")
        curriculum = curriculum_replay_samples(curriculum_data, max_samples=int(curriculum_replay_count))
        curriculum_added = len(curriculum)
        replay_samples.extend(curriculum)

    center_added = 0
    if include_center_replay:
        center = center_replay_samples(int(center_replay_repeats))
        center_added = len(center)
        replay_samples.extend(center)

    raw_teacher_counts = {key: len(value) for key, value in teacher_groups.items()}
    balanced_samples, balance_summary = build_teacher_balanced_replay(
        teacher_groups,
        teacher_balance=teacher_balance,
        reason_weights=reason_weights,
        max_low_heuristic_ratio=float(max_low_heuristic_ratio),
        replay_samples=replay_samples,
    )
    if not balanced_samples:
        raise RuntimeError("mistake v3 produced no samples")

    shapes = save_mistake_samples(balanced_samples, output_path)
    validation_shapes = save_mistake_samples(validation_samples, validation_output)
    summary = {
        "version": "v3_teacher_balanced",
        "student_checkpoint": student_checkpoint,
        "output_path": output_path,
        "validation_output": validation_output,
        "teachers": list(teachers),
        "seeds": list(seeds),
        "games_per_teacher": int(games_per_teacher),
        "raw_teacher_counts": raw_teacher_counts,
        "teacher_balance": dict(teacher_balance or {}),
        "reason_weights": dict(reason_weights or {}),
        "max_low_heuristic_ratio": float(max_low_heuristic_ratio),
        "v1_tactical_draw_replay_samples": int(v1_draw_count),
        "curriculum_replay_samples": int(curriculum_added),
        "center_replay_samples": int(center_added),
        "validation_samples": int(len(validation_samples)),
        "validation_holdout_samples": int(len(heldout)),
        "final_samples": int(shapes["states"][0]),
        "validation_shapes": validation_shapes,
        "balance_summary": balance_summary,
        "reason_distribution_before": balance_summary.get("reason_distribution_before", {}),
        "reason_distribution_after": balance_summary.get("reason_distribution_after", {}),
        "low_heuristic_ratio_after": balance_summary.get("low_heuristic_cap", {}).get("ratio_after", 0.0),
        "collections": collection_summaries,
        "rule_mode": rule_mode,
        "num_simulations": int(num_simulations),
        "max_moves": int(max_moves),
        "elapsed_sec": time.perf_counter() - start,
    }
    _write_stats(summary, stats_path)
    progress_print(
        f"DONE mistake_v3 final={summary['final_samples']} "
        f"v1_draw={v1_draw_count} curriculum={curriculum_added} center={center_added} "
        f"low_ratio={summary['low_heuristic_ratio_after']:.3f} "
        f"elapsed={format_seconds(summary['elapsed_sec'])} output={output_path}",
        "mistake",
    )
    return summary


__all__ = [
    "DEFAULT_MISTAKE_DATA_PATH",
    "DEFAULT_MISTAKE_STATS_PATH",
    "MistakeSample",
    "CRITICAL_REASONS",
    "build_mistake_training_dataset_v2",
    "build_mistake_training_dataset_v3",
    "collect_mistake_positions",
    "collect_mistake_samples_from_players",
    "collect_mistake_positions_from_players",
    "append_center_replay_samples",
    "center_replay_samples",
    "curriculum_replay_samples",
    "fixed_tactical_validation_samples",
    "make_mistake_sample",
    "oversample_critical_samples",
    "reason_distribution",
    "save_mistake_samples",
]
