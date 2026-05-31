"""Hybrid-only mistake replay for the Hybrid-survival branch."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, replace
from typing import Iterable, Mapping

import numpy as np

from engine.heuristic import evaluate_move_heuristic
from game.board import BLACK, BOARD_SIZE, Board
from game.encoder import index_to_action
from train.mistake_mining import (
    MistakeSample,
    _apply_random_opening,
    _create_teacher,
    _game_status,
    _load_student_player,
    _safe_select,
    _tactical_reasons,
    center_replay_samples,
    collect_mistake_samples_from_players,
    curriculum_replay_samples,
    make_mistake_sample,
    reason_distribution,
    save_mistake_samples,
)
from train.mistake_replay_balancer import apply_reason_weights, cap_reason_ratio
from train.progress import format_seconds, progress_print


DEFAULT_HYBRID_SURVIVAL_DATA_PATH = os.path.join(
    "outputs", "supervised", "hybrid_survival_latest.npz"
)
DEFAULT_HYBRID_SURVIVAL_STATS_PATH = os.path.join(
    "outputs", "supervised", "hybrid_survival_latest_stats.json"
)
DEFAULT_HYBRID_SURVIVAL_V2_DATA_PATH = os.path.join(
    "outputs", "supervised", "hybrid_survival_v2.npz"
)
DEFAULT_HYBRID_SURVIVAL_V2_METADATA_PATH = os.path.join(
    "outputs", "supervised", "hybrid_survival_v2_metadata.jsonl"
)
DEFAULT_HYBRID_SURVIVAL_V2_STATS_PATH = os.path.join(
    "outputs", "supervised", "hybrid_survival_v2_stats.json"
)
DEFAULT_HYBRID_SURVIVAL_V3_DATA_PATH = os.path.join(
    "outputs", "supervised", "hybrid_survival_v3_forced_block.npz"
)
DEFAULT_HYBRID_SURVIVAL_V3_METADATA_PATH = os.path.join(
    "outputs", "supervised", "hybrid_survival_v3_forced_block_metadata.jsonl"
)
DEFAULT_HYBRID_SURVIVAL_V3_STATS_PATH = os.path.join(
    "outputs", "supervised", "hybrid_survival_v3_forced_block_stats.json"
)

HYBRID_SURVIVAL_REASON_WEIGHTS = {
    "missed_immediate_block": 5,
    "missed_open_four_defense": 4,
    "missed_blocked_four_defense": 3,
    "missed_open_three_defense": 2,
    "near_loss_position": 2,
    "teacher_student_disagree": 1,
    "low_heuristic_move": 1,
}

FORCED_DEFENSE_REASONS = {
    "missed_immediate_block",
    "missed_immediate_win",
    "missed_open_four_defense",
    "missed_blocked_four_defense",
    "missed_double_threat_defense",
}


@dataclass
class AnnotatedMistakeSample:
    """A training sample plus per-sample metadata for auditability."""

    sample: MistakeSample
    metadata: dict


def _write_json(payload: dict, path: str | None) -> None:
    if not path:
        return
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _write_jsonl(records: list[dict], path: str | None) -> None:
    if not path:
        return
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            payload = dict(record)
            payload["sample_index"] = index
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _load_required_arrays(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key], dtype=np.float32) for key in data.files}
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
        raise ValueError(f"replay dataset missing keys: {missing}")
    return arrays


def npz_replay_samples(
    path: str,
    max_samples: int = 128,
    reason: str = "replay",
) -> list[MistakeSample]:
    """Load replay samples from the project npz training format.

    This is intentionally label-only replay: it does not regenerate teacher moves
    and does not call TacticalPlayer or HybridPlayer.
    """
    if max_samples <= 0:
        return []
    if not os.path.exists(path):
        progress_print(f"SKIP replay missing path={path}", "hybrid-survival")
        return []
    arrays = _load_required_arrays(path)
    n = min(int(max_samples), int(arrays["states"].shape[0]))
    samples: list[MistakeSample] = []
    for idx in range(n):
        policy = np.asarray(arrays["policies"][idx], dtype=np.float32)
        samples.append(
            MistakeSample(
                state=np.asarray(arrays["states"][idx], dtype=np.float32),
                policy=policy,
                value=float(np.asarray(arrays["values"][idx]).reshape(-1)[0]),
                threat_labels=np.asarray(arrays["threat_labels"][idx], dtype=np.float32),
                forbidden_labels=np.asarray(arrays["forbidden_labels"][idx], dtype=np.float32),
                tactical_scores=np.asarray(arrays["tactical_scores"][idx], dtype=np.float32),
                teacher_action=int(np.argmax(policy)),
                reasons=(reason,),
            )
        )
    return samples


def _annotate_replay(
    samples: Iterable[MistakeSample],
    *,
    target_source: str,
    reason: str,
    value_source: str = "replay_dataset",
) -> list[AnnotatedMistakeSample]:
    annotated = []
    for idx, sample in enumerate(samples):
        metadata = {
            "student_action": None,
            "teacher_action": int(sample.teacher_action),
            "student_heuristic_score": None,
            "teacher_heuristic_score": None,
            "heuristic_delta": None,
            "remaining_moves": None,
            "reason": list(sample.reasons or (reason,)),
            "teacher_type": "replay",
            "game_id": None,
            "ply_index": None,
            "final_result": None,
            "teacher_action_extends_game": None,
            "student_to_move": None,
            "rule_mode": None,
            "target_source": target_source,
            "value_source": value_source,
            "replay_index": int(idx),
        }
        annotated.append(AnnotatedMistakeSample(sample=sample, metadata=metadata))
    return annotated


def _tag(samples: Iterable[MistakeSample], reason: str) -> list[MistakeSample]:
    tagged = []
    for sample in samples:
        tagged.append(replace(sample, reasons=tuple(dict.fromkeys((*sample.reasons, reason)))))
    return tagged


def _count_replay(samples: Iterable[MistakeSample], reason: str) -> int:
    return sum(reason in sample.reasons for sample in samples)


def build_hybrid_survival_samples(
    hybrid_samples: list[MistakeSample],
    *,
    replay_samples: Iterable[MistakeSample] = (),
    reason_weights: Mapping[str, float] | None = None,
    max_low_heuristic_ratio: float = 0.20,
) -> tuple[list[MistakeSample], dict]:
    """Build final Hybrid-survival samples from Hybrid-teacher mistakes and replay."""
    reason_weights = dict(reason_weights or HYBRID_SURVIVAL_REASON_WEIGHTS)
    tagged_hybrid = _tag(hybrid_samples, "teacher_hybrid")
    weighted, weight_summary = apply_reason_weights(tagged_hybrid, reason_weights)
    capped, cap_summary = cap_reason_ratio(
        weighted,
        reason="low_heuristic_move",
        max_ratio=float(max_low_heuristic_ratio),
    )
    replay = list(replay_samples)
    final = capped + replay
    summary = {
        "branch": "hybrid_survival",
        "teacher": "hybrid",
        "raw_hybrid_samples": len(hybrid_samples),
        "weighted_hybrid_samples": len(weighted),
        "capped_hybrid_samples": len(capped),
        "replay_samples": len(replay),
        "curriculum_replay_samples": _count_replay(replay, "curriculum_replay"),
        "center_replay_samples": _count_replay(replay, "center_replay"),
        "tactical_restoration_curriculum_replay_samples": _count_replay(
            replay,
            "tactical_restoration_curriculum_replay",
        ),
        "reason_weighting": weight_summary,
        "low_heuristic_cap": cap_summary,
        "reason_distribution_before": reason_distribution(tagged_hybrid),
        "reason_distribution_after": reason_distribution(final),
        "final_samples": len(final),
    }
    return final, summary


def _teacher_extends_game(
    board: Board,
    student_action: int,
    teacher_action: int,
    current_player: int,
    rule_mode: str,
) -> bool:
    """One-ply approximation: teacher avoids an immediate student-action loss."""
    try:
        student_board = board.copy()
        sx, sy = index_to_action(int(student_action), BOARD_SIZE)
        student_board.place_stone(sx, sy)
        student_over, student_winner = _game_status(student_board, rule_mode)

        teacher_board = board.copy()
        tx, ty = index_to_action(int(teacher_action), BOARD_SIZE)
        teacher_board.place_stone(tx, ty)
        teacher_over, teacher_winner = _game_status(teacher_board, rule_mode)
    except Exception:
        return False
    opponent = -int(current_player)
    return bool(
        student_over
        and int(student_winner) == opponent
        and not (teacher_over and int(teacher_winner) == opponent)
    )


def _value_for_reasons(
    reasons: Iterable[str],
    current_player: int,
    target_source: str,
) -> tuple[int, str]:
    reason_set = set(reasons)
    if "missed_immediate_win" in reason_set:
        return int(current_player), "tactical_win"
    if target_source == "v2_self_survival":
        return 0, "neutral_v2_survival"
    if reason_set.intersection(FORCED_DEFENSE_REASONS):
        return 0, "neutral_forced_defense"
    return 0, "neutral_survival"


def _make_annotated_sample(
    board: Board,
    *,
    target_action: int,
    student_action: int | None,
    teacher_action: int | None,
    student_score: float | None,
    teacher_score: float | None,
    final_winner: int,
    rule_mode: str,
    reasons: Iterable[str],
    teacher_type: str,
    game_id: int | None,
    ply_index: int | None,
    remaining_moves: int | None,
    target_source: str,
    teacher_action_extends_game: bool | None,
) -> AnnotatedMistakeSample:
    reasons = tuple(dict.fromkeys(str(reason) for reason in reasons))
    value_winner, value_source = _value_for_reasons(
        reasons,
        int(board.current_player),
        target_source,
    )
    sample = make_mistake_sample(
        board,
        teacher_action=int(target_action),
        final_winner=value_winner,
        rule_mode=rule_mode,
        reasons=reasons,
    )
    metadata = {
        "student_action": None if student_action is None else int(student_action),
        "teacher_action": None if teacher_action is None else int(teacher_action),
        "student_heuristic_score": None if student_score is None else float(student_score),
        "teacher_heuristic_score": None if teacher_score is None else float(teacher_score),
        "heuristic_delta": None
        if student_score is None or teacher_score is None
        else float(teacher_score - student_score),
        "remaining_moves": None if remaining_moves is None else int(remaining_moves),
        "reason": list(reasons),
        "teacher_type": teacher_type,
        "game_id": None if game_id is None else int(game_id),
        "ply_index": None if ply_index is None else int(ply_index),
        "final_result": int(final_winner),
        "teacher_action_extends_game": teacher_action_extends_game,
        "student_to_move": int(board.current_player),
        "rule_mode": rule_mode,
        "target_source": target_source,
        "value_source": value_source,
    }
    return AnnotatedMistakeSample(sample=sample, metadata=metadata)


def _repeat_annotated(samples: list[AnnotatedMistakeSample], count: int) -> list[AnnotatedMistakeSample]:
    if count <= 0 or not samples:
        return []
    if len(samples) >= count:
        return list(samples[:count])
    repeated = []
    idx = 0
    while len(repeated) < count:
        source = samples[idx % len(samples)]
        metadata = dict(source.metadata)
        metadata["duplicate_copy"] = len(repeated) // len(samples)
        repeated.append(AnnotatedMistakeSample(sample=source.sample, metadata=metadata))
        idx += 1
    return repeated


def _sort_by_delta(samples: list[AnnotatedMistakeSample]) -> list[AnnotatedMistakeSample]:
    return sorted(
        samples,
        key=lambda item: (
            float(item.metadata.get("heuristic_delta") or 0.0),
            int(item.metadata.get("remaining_moves") or 0),
        ),
        reverse=True,
    )


def collect_hybrid_survival_v2_candidates(
    student_checkpoint: str,
    *,
    games: int = 30,
    seeds: Iterable[int] = (2026, 7, 21),
    rule_mode: str = "basic",
    num_simulations: int = 50,
    device: str = "cuda",
    max_moves: int = 80,
    min_score_gap: float = 5000.0,
    long_game_min_moves: int = 50,
    opening_moves: int = 2,
) -> tuple[list[AnnotatedMistakeSample], list[AnnotatedMistakeSample], list[dict]]:
    """Collect high-value Hybrid corrections and v2 self-survival replay samples."""
    corrections: list[AnnotatedMistakeSample] = []
    self_survival: list[AnnotatedMistakeSample] = []
    summaries: list[dict] = []
    game_id = 0
    for seed in tuple(int(seed) for seed in seeds):
        student = _load_student_player(student_checkpoint, num_simulations, device)
        teacher = _create_teacher("hybrid", rule_mode, device, num_simulations)
        for local_game in range(int(games)):
            game_id += 1
            board = Board()
            if opening_moves > 0:
                import random

                _apply_random_opening(board, random.Random(int(seed) + local_game), opening_moves)
            student_color = BLACK if local_game % 2 == 0 else -BLACK
            teacher_color = -student_color
            records: list[dict] = []
            winner = 0
            moves = 0
            for ply in range(int(max_moves)):
                over, winner = _game_status(board, rule_mode)
                if over:
                    break
                current_player = int(board.current_player)
                if current_player == student_color:
                    snapshot = board.copy()
                    student_action = _safe_select(student, board)
                    teacher_action = _safe_select(teacher, snapshot)
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
                    student_score = evaluate_move_heuristic(
                        snapshot,
                        int(student_action),
                        current_player,
                        rule_mode,
                    )
                    teacher_score = evaluate_move_heuristic(
                        snapshot,
                        int(teacher_action),
                        current_player,
                        rule_mode,
                    )
                    records.append(
                        {
                            "board": snapshot,
                            "student_action": int(student_action),
                            "teacher_action": int(teacher_action),
                            "current_player": current_player,
                            "ply": int(ply),
                            "reasons": tuple(reasons),
                            "score_gap": float(score_gap),
                            "student_score": float(student_score),
                            "teacher_score": float(teacher_score),
                            "teacher_extends": _teacher_extends_game(
                                snapshot,
                                int(student_action),
                                int(teacher_action),
                                current_player,
                                rule_mode,
                            ),
                        }
                    )
                    action = student_action
                else:
                    action = _safe_select(teacher, board)
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
            long_game = moves >= int(long_game_min_moves)
            added_correction = 0
            added_survival = 0
            for record in records:
                remaining = int(moves - record["ply"])
                reason_list = list(record["reasons"])
                base_reason_set = set(reason_list)
                has_forced_reason = bool(base_reason_set.intersection(FORCED_DEFENSE_REASONS))
                if has_forced_reason:
                    # Keep forced-defense samples clean so generic near-loss or
                    # low-heuristic tags do not dominate/cap the most important
                    # training bucket.
                    reason_list = [
                        reason
                        for reason in reason_list
                        if reason in FORCED_DEFENSE_REASONS
                    ]
                elif (
                    student_lost
                    and 8 <= remaining <= 30
                    and "near_loss_position" not in reason_list
                ):
                    reason_list.append("near_loss_position")
                reason_set = set(reason_list)
                high_value = bool(reason_set.intersection(FORCED_DEFENSE_REASONS))
                open_three = "missed_open_three_defense" in reason_set
                clear_delta = (
                    int(record["teacher_action"]) != int(record["student_action"])
                    and float(record["teacher_score"] - record["student_score"]) >= float(min_score_gap)
                )
                valid_window = 8 <= remaining <= 30
                if (
                    student_lost
                    and int(record["teacher_action"]) != int(record["student_action"])
                    and remaining > 3
                    and valid_window
                    and (high_value or open_three or clear_delta)
                ):
                    corrections.append(
                        _make_annotated_sample(
                            record["board"],
                            target_action=record["teacher_action"],
                            student_action=record["student_action"],
                            teacher_action=record["teacher_action"],
                            student_score=record["student_score"],
                            teacher_score=record["teacher_score"],
                            final_winner=winner,
                            rule_mode=rule_mode,
                            reasons=reason_list,
                            teacher_type="hybrid",
                            game_id=game_id,
                            ply_index=record["ply"],
                            remaining_moves=remaining,
                            target_source="hybrid_teacher",
                            teacher_action_extends_game=record["teacher_extends"],
                        )
                    )
                    added_correction += 1
                if (
                    long_game
                    and remaining >= 12
                    and int(record["student_action"]) is not None
                    and record["board"].is_legal_move(*index_to_action(record["student_action"], BOARD_SIZE))
                ):
                    self_survival.append(
                        _make_annotated_sample(
                            record["board"],
                            target_action=record["student_action"],
                            student_action=record["student_action"],
                            teacher_action=record["teacher_action"],
                            student_score=record["student_score"],
                            teacher_score=record["teacher_score"],
                            final_winner=winner,
                            rule_mode=rule_mode,
                            reasons=("v2_self_survival_replay",),
                            teacher_type="hybrid",
                            game_id=game_id,
                            ply_index=record["ply"],
                            remaining_moves=remaining,
                            target_source="v2_self_survival",
                            teacher_action_extends_game=record["teacher_extends"],
                        )
                    )
                    added_survival += 1
            summaries.append(
                {
                    "game_id": game_id,
                    "seed": int(seed),
                    "local_game": local_game + 1,
                    "student_color": int(student_color),
                    "winner": int(winner),
                    "moves": int(moves),
                    "student_lost": bool(student_lost),
                    "long_game": bool(long_game),
                    "correction_samples": int(added_correction),
                    "self_survival_samples": int(added_survival),
                }
            )
            progress_print(
                f"v2 game {game_id} complete winner={winner} moves={moves} "
                f"corrections={added_correction} survival={added_survival}",
                "hybrid-survival",
            )
    return corrections, self_survival, summaries


def build_hybrid_survival_v2_samples(
    corrections: list[AnnotatedMistakeSample],
    self_survival: list[AnnotatedMistakeSample],
    replay_samples: Iterable[AnnotatedMistakeSample] = (),
    *,
    target_samples: int = 2400,
    max_open_three_ratio: float = 0.25,
    max_near_loss_ratio: float = 0.15,
    max_low_heuristic_ratio: float = 0.10,
    max_negative_value_ratio: float = 0.55,
) -> tuple[list[AnnotatedMistakeSample], dict]:
    """Apply v2 sample recipe and hard caps without training."""
    target_samples = max(100, int(target_samples))
    forced = [
        sample
        for sample in corrections
        if set(sample.metadata.get("reason", [])).intersection(FORCED_DEFENSE_REASONS)
    ]
    open_three = [
        sample
        for sample in corrections
        if "missed_open_three_defense" in sample.metadata.get("reason", [])
        and not set(sample.metadata.get("reason", [])).intersection(FORCED_DEFENSE_REASONS)
    ]
    low_diverse = [
        sample
        for sample in corrections
        if "low_heuristic_move" in sample.metadata.get("reason", [])
        and not set(sample.metadata.get("reason", [])).intersection(FORCED_DEFENSE_REASONS)
    ]

    forced_target = int(round(target_samples * 0.45))
    survival_target = int(round(target_samples * 0.25))
    replay_target = int(round(target_samples * 0.20))
    center_target = int(round(target_samples * 0.05))
    low_target = target_samples - forced_target - survival_target - replay_target - center_target

    replay = list(replay_samples)
    curriculum_like = [
        sample
        for sample in replay
        if "center_replay" not in sample.metadata.get("reason", [])
    ]
    center = [
        sample
        for sample in replay
        if "center_replay" in sample.metadata.get("reason", [])
    ]

    selected = []
    selected.extend(_repeat_annotated(_sort_by_delta(forced), forced_target))
    selected.extend(_repeat_annotated(_sort_by_delta(self_survival), survival_target))
    selected.extend(_repeat_annotated(curriculum_like, replay_target))
    selected.extend(_repeat_annotated(center, center_target))

    # Low-diverse bucket is intentionally small and can include capped open-three defense.
    low_pool = _sort_by_delta(low_diverse + open_three)
    selected.extend(_repeat_annotated(low_pool, max(0, low_target)))

    # Hard caps trim dominant labels after ratio assembly.
    def has_reason(sample: AnnotatedMistakeSample, reason: str) -> bool:
        return reason in sample.metadata.get("reason", [])

    def cap_reason(samples: list[AnnotatedMistakeSample], reason: str, max_ratio: float) -> list[AnnotatedMistakeSample]:
        other = [sample for sample in samples if not has_reason(sample, reason)]
        with_reason = [sample for sample in samples if has_reason(sample, reason)]
        max_count = int((float(max_ratio) * len(other)) / max(1e-9, 1.0 - float(max_ratio)))
        return other + with_reason[: max(0, max_count)]

    selected = cap_reason(selected, "low_heuristic_move", max_low_heuristic_ratio)
    selected = cap_reason(selected, "near_loss_position", max_near_loss_ratio)
    selected = cap_reason(selected, "missed_open_three_defense", max_open_three_ratio)

    # If negative labels still dominate, prefer neutral/value-positive replay and forced samples.
    negative = [sample for sample in selected if float(sample.sample.value) < -1e-6]
    non_negative = [sample for sample in selected if float(sample.sample.value) >= -1e-6]
    max_negative = int((float(max_negative_value_ratio) * len(non_negative)) / max(1e-9, 1.0 - float(max_negative_value_ratio)))
    selected = non_negative + negative[: max(0, max_negative)]

    reason_dist = reason_distribution([sample.sample for sample in selected])
    forced_count = sum(
        bool(set(sample.metadata.get("reason", [])).intersection(FORCED_DEFENSE_REASONS))
        for sample in selected
    )
    negative_count = sum(float(sample.sample.value) < -1e-6 for sample in selected)
    summary = {
        "target_samples": int(target_samples),
        "raw_corrections": int(len(corrections)),
        "raw_self_survival": int(len(self_survival)),
        "raw_forced_defense": int(len(forced)),
        "raw_open_three_defense": int(len(open_three)),
        "raw_low_diverse": int(len(low_diverse)),
        "final_samples": int(len(selected)),
        "forced_defense_combined_ratio": forced_count / max(1, len(selected)),
        "negative_value_ratio": negative_count / max(1, len(selected)),
        "reason_distribution_after": reason_dist,
        "target_mix": {
            "forced_defense": forced_target,
            "v2_self_survival": survival_target,
            "curriculum_fixed": replay_target,
            "center": center_target,
            "low_diverse": max(0, low_target),
        },
    }
    return selected, summary


def build_hybrid_survival_v3_samples(
    forced_block_replay: list[AnnotatedMistakeSample],
    self_survival: list[AnnotatedMistakeSample],
    replay_samples: Iterable[AnnotatedMistakeSample] = (),
    hybrid_teacher_samples: Iterable[AnnotatedMistakeSample] = (),
    *,
    target_samples: int = 2400,
) -> tuple[list[AnnotatedMistakeSample], dict]:
    """Apply the forced-block v3 sample recipe without training.

    Target mix:
    - 20% missed_immediate_block
    - 10% missed_open_four_defense
    - 10% missed_blocked_four_defense
    - 5% missed_double_threat_defense
    - 25% v2 self-survival replay
    - 20% curriculum/fixed tactical replay
    - 5% center replay
    - 5% high-delta Hybrid teacher samples
    """
    target_samples = max(100, int(target_samples))

    def by_reason(samples: Iterable[AnnotatedMistakeSample], reason: str) -> list[AnnotatedMistakeSample]:
        return [
            sample
            for sample in samples
            if reason in (sample.metadata.get("reason", []) or [])
        ]

    forced_block_replay = list(forced_block_replay)
    self_survival = list(self_survival)
    replay = list(replay_samples)
    hybrid_teacher = _sort_by_delta(list(hybrid_teacher_samples))

    immediate_target = int(round(target_samples * 0.20))
    open_four_target = int(round(target_samples * 0.10))
    blocked_four_target = int(round(target_samples * 0.10))
    double_threat_target = int(round(target_samples * 0.05))
    survival_target = int(round(target_samples * 0.25))
    curriculum_target = int(round(target_samples * 0.20))
    center_target = int(round(target_samples * 0.05))
    hybrid_target = target_samples - (
        immediate_target
        + open_four_target
        + blocked_four_target
        + double_threat_target
        + survival_target
        + curriculum_target
        + center_target
    )

    curriculum_like = [
        sample
        for sample in replay
        if "center_replay" not in (sample.metadata.get("reason", []) or [])
    ]
    center = [
        sample
        for sample in replay
        if "center_replay" in (sample.metadata.get("reason", []) or [])
    ]

    selected: list[AnnotatedMistakeSample] = []
    selected.extend(_repeat_annotated(by_reason(forced_block_replay, "missed_immediate_block"), immediate_target))
    selected.extend(_repeat_annotated(by_reason(forced_block_replay, "missed_open_four_defense"), open_four_target))
    selected.extend(_repeat_annotated(by_reason(forced_block_replay, "missed_blocked_four_defense"), blocked_four_target))
    selected.extend(_repeat_annotated(by_reason(forced_block_replay, "missed_double_threat_defense"), double_threat_target))
    selected.extend(_repeat_annotated(_sort_by_delta(self_survival), survival_target))
    selected.extend(_repeat_annotated(curriculum_like, curriculum_target))
    selected.extend(_repeat_annotated(center, center_target))
    selected.extend(_repeat_annotated(hybrid_teacher, max(0, hybrid_target)))

    if len(selected) < target_samples:
        fill_pool = (
            by_reason(forced_block_replay, "missed_immediate_block")
            or forced_block_replay
            or self_survival
            or curriculum_like
            or center
        )
        selected.extend(_repeat_annotated(fill_pool, target_samples - len(selected)))
    selected = selected[:target_samples]

    reason_dist = reason_distribution([sample.sample for sample in selected])
    total = max(1, len(selected))
    immediate_count = reason_dist.get("missed_immediate_block", 0)
    forced_count = sum(
        bool(set(item.metadata.get("reason", []) or []).intersection(FORCED_DEFENSE_REASONS))
        for item in selected
    )
    near_loss_count = reason_dist.get("near_loss_position", 0)
    low_heuristic_count = reason_dist.get("low_heuristic_move", 0)
    negative_count = sum(float(item.sample.value) < -1e-6 for item in selected)
    summary = {
        "target_samples": int(target_samples),
        "raw_forced_block_replay": int(len(forced_block_replay)),
        "raw_self_survival": int(len(self_survival)),
        "raw_replay": int(len(replay)),
        "raw_hybrid_teacher": int(len(hybrid_teacher)),
        "final_samples": int(len(selected)),
        "reason_distribution_after": reason_dist,
        "missed_immediate_block_ratio": immediate_count / total,
        "forced_defense_combined_ratio": forced_count / total,
        "near_loss_ratio": near_loss_count / total,
        "low_heuristic_ratio": low_heuristic_count / total,
        "negative_value_ratio": negative_count / total,
        "target_mix": {
            "missed_immediate_block": immediate_target,
            "missed_open_four_defense": open_four_target,
            "missed_blocked_four_defense": blocked_four_target,
            "missed_double_threat_defense": double_threat_target,
            "v2_self_survival": survival_target,
            "curriculum_fixed": curriculum_target,
            "center": center_target,
            "high_delta_hybrid": max(0, hybrid_target),
        },
    }
    return selected, summary


def save_annotated_samples(
    annotated: list[AnnotatedMistakeSample],
    output_path: str,
    metadata_path: str,
) -> dict:
    samples = [item.sample for item in annotated]
    shapes = save_mistake_samples(samples, output_path)
    _write_jsonl([item.metadata for item in annotated], metadata_path)
    return shapes


def build_hybrid_survival_v2_dataset(
    student_checkpoint: str,
    *,
    games: int = 30,
    seeds: Iterable[int] = (2026, 7, 21),
    rule_mode: str = "basic",
    num_simulations: int = 50,
    device: str = "cuda",
    output_path: str = DEFAULT_HYBRID_SURVIVAL_V2_DATA_PATH,
    metadata_path: str = DEFAULT_HYBRID_SURVIVAL_V2_METADATA_PATH,
    stats_path: str | None = DEFAULT_HYBRID_SURVIVAL_V2_STATS_PATH,
    max_moves: int = 80,
    min_score_gap: float = 5000.0,
    target_samples: int = 2400,
    curriculum_data: str | None = os.path.join("outputs", "supervised", "tactical_curriculum_latest.npz"),
    curriculum_replay_count: int = 512,
    center_replay_repeats: int = 128,
) -> dict:
    """Build the v2 Hybrid-survival dataset with per-sample JSONL metadata."""
    start = time.perf_counter()
    progress_print(
        f"START hybrid_survival_v2 student={student_checkpoint} games={games} "
        f"seeds={tuple(seeds)} teacher=hybrid",
        "hybrid-survival",
    )
    corrections, self_survival, collection_summaries = collect_hybrid_survival_v2_candidates(
        student_checkpoint,
        games=games,
        seeds=seeds,
        rule_mode=rule_mode,
        num_simulations=num_simulations,
        device=device,
        max_moves=max_moves,
        min_score_gap=min_score_gap,
    )

    replay: list[AnnotatedMistakeSample] = []
    if curriculum_data:
        replay.extend(
            _annotate_replay(
                curriculum_replay_samples(curriculum_data, max_samples=curriculum_replay_count),
                target_source="curriculum_replay",
                reason="curriculum_replay",
            )
        )
    replay.extend(
        _annotate_replay(
            center_replay_samples(center_replay_repeats),
            target_source="center_replay",
            reason="center_replay",
            value_source="neutral_center",
        )
    )
    selected, build_summary = build_hybrid_survival_v2_samples(
        corrections,
        self_survival,
        replay,
        target_samples=target_samples,
    )
    shapes = save_annotated_samples(selected, output_path, metadata_path)
    reason_dist = reason_distribution([item.sample for item in selected])
    summary = {
        "version": "hybrid_survival_v2",
        "student_checkpoint": student_checkpoint,
        "output_path": output_path,
        "metadata_path": metadata_path,
        "seeds": list(tuple(int(seed) for seed in seeds)),
        "games_per_seed": int(games),
        "rule_mode": rule_mode,
        "num_simulations": int(num_simulations),
        "max_moves": int(max_moves),
        "min_score_gap": float(min_score_gap),
        "shapes": shapes,
        "reason_distribution_after": reason_dist,
        "collections": collection_summaries,
        "build_summary": build_summary,
        "final_samples": int(shapes["states"][0]),
        "elapsed_sec": time.perf_counter() - start,
    }
    _write_json(summary, stats_path)
    progress_print(
        f"DONE hybrid_survival_v2 final={summary['final_samples']} "
        f"forced_ratio={build_summary['forced_defense_combined_ratio']:.3f} "
        f"neg_ratio={build_summary['negative_value_ratio']:.3f} "
        f"elapsed={format_seconds(summary['elapsed_sec'])} output={output_path}",
        "hybrid-survival",
    )
    return summary


def build_hybrid_survival_v3_forced_block_dataset(
    student_checkpoint: str,
    *,
    games: int = 30,
    seeds: Iterable[int] = (2026, 7, 21),
    rule_mode: str = "basic",
    num_simulations: int = 50,
    device: str = "cuda",
    output_path: str = DEFAULT_HYBRID_SURVIVAL_V3_DATA_PATH,
    metadata_path: str = DEFAULT_HYBRID_SURVIVAL_V3_METADATA_PATH,
    stats_path: str | None = DEFAULT_HYBRID_SURVIVAL_V3_STATS_PATH,
    max_moves: int = 80,
    min_score_gap: float = 5000.0,
    target_samples: int = 2400,
    curriculum_data: str | None = os.path.join("outputs", "supervised", "tactical_curriculum_latest.npz"),
    curriculum_replay_count: int = 512,
    center_replay_repeats: int = 128,
) -> dict:
    """Build Hybrid-survival v3 with constructed forced-block replay."""
    from train.forced_block_replay import generate_forced_block_replay_samples

    start = time.perf_counter()
    progress_print(
        f"START hybrid_survival_v3_forced_block student={student_checkpoint} "
        f"games={games} seeds={tuple(seeds)} teacher=hybrid",
        "hybrid-survival",
    )
    corrections, self_survival, collection_summaries = collect_hybrid_survival_v2_candidates(
        student_checkpoint,
        games=games,
        seeds=seeds,
        rule_mode=rule_mode,
        num_simulations=num_simulations,
        device=device,
        max_moves=max_moves,
        min_score_gap=min_score_gap,
    )

    forced_block_replay = generate_forced_block_replay_samples(rule_mode)
    replay: list[AnnotatedMistakeSample] = []
    if curriculum_data:
        replay.extend(
            _annotate_replay(
                curriculum_replay_samples(curriculum_data, max_samples=curriculum_replay_count),
                target_source="curriculum_replay",
                reason="curriculum_replay",
            )
        )
    replay.extend(
        _annotate_replay(
            center_replay_samples(center_replay_repeats),
            target_source="center_replay",
            reason="center_replay",
            value_source="neutral_center",
        )
    )

    selected, build_summary = build_hybrid_survival_v3_samples(
        forced_block_replay,
        self_survival,
        replay,
        corrections,
        target_samples=target_samples,
    )
    shapes = save_annotated_samples(selected, output_path, metadata_path)
    reason_dist = reason_distribution([item.sample for item in selected])
    summary = {
        "version": "hybrid_survival_v3_forced_block",
        "student_checkpoint": student_checkpoint,
        "output_path": output_path,
        "metadata_path": metadata_path,
        "seeds": list(tuple(int(seed) for seed in seeds)),
        "games_per_seed": int(games),
        "rule_mode": rule_mode,
        "num_simulations": int(num_simulations),
        "max_moves": int(max_moves),
        "min_score_gap": float(min_score_gap),
        "shapes": shapes,
        "reason_distribution_after": reason_dist,
        "collections": collection_summaries,
        "build_summary": build_summary,
        "final_samples": int(shapes["states"][0]),
        "elapsed_sec": time.perf_counter() - start,
    }
    _write_json(summary, stats_path)
    progress_print(
        f"DONE hybrid_survival_v3_forced_block final={summary['final_samples']} "
        f"immediate_ratio={build_summary['missed_immediate_block_ratio']:.3f} "
        f"forced_ratio={build_summary['forced_defense_combined_ratio']:.3f} "
        f"elapsed={format_seconds(summary['elapsed_sec'])} output={output_path}",
        "hybrid-survival",
    )
    return summary


def build_hybrid_survival_dataset(
    student_checkpoint: str,
    *,
    games: int = 30,
    seeds: Iterable[int] = (2026, 7, 21),
    rule_mode: str = "basic",
    num_simulations: int = 50,
    device: str = "cuda",
    output_path: str = DEFAULT_HYBRID_SURVIVAL_DATA_PATH,
    max_moves: int = 80,
    min_score_gap: float = 5000.0,
    include_center_replay: bool = True,
    center_replay_repeats: int = 128,
    include_curriculum_replay: bool = True,
    curriculum_data: str | None = os.path.join("outputs", "supervised", "tactical_curriculum_latest.npz"),
    curriculum_replay_count: int = 512,
    tactical_restoration_data: str | None = os.path.join("outputs", "supervised", "tactical_restoration_dataset.npz"),
    tactical_restoration_replay_count: int = 128,
    reason_weights: Mapping[str, float] | None = None,
    max_low_heuristic_ratio: float = 0.20,
    stats_path: str | None = DEFAULT_HYBRID_SURVIVAL_STATS_PATH,
) -> dict:
    """Collect Hybrid-only mistake data and save an npz dataset."""
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds:
        raise ValueError("at least one seed is required")
    start = time.perf_counter()
    progress_print(
        f"START hybrid_survival student={student_checkpoint} games={games} "
        f"seeds={seeds} teacher=hybrid",
        "hybrid-survival",
    )
    hybrid_samples: list[MistakeSample] = []
    collection_summaries: list[dict] = []
    for seed in seeds:
        student = _load_student_player(student_checkpoint, num_simulations, device)
        teacher = _create_teacher("hybrid", rule_mode, device, num_simulations)
        samples, summary = collect_mistake_samples_from_players(
            student,
            teacher,
            games=int(games),
            rule_mode=rule_mode,
            max_moves=int(max_moves),
            min_score_gap=float(min_score_gap),
            endgame_window=(10, 30),
            collect_losses=True,
            include_draws=False,
            opening_seed=int(seed),
            opening_moves=2,
        )
        summary["seed"] = int(seed)
        summary["teacher_type"] = "hybrid"
        collection_summaries.append(summary)
        hybrid_samples.extend(samples)

    replay_samples: list[MistakeSample] = []
    if include_curriculum_replay:
        if not curriculum_data:
            raise ValueError("curriculum_data is required when include_curriculum_replay=True")
        replay_samples.extend(
            curriculum_replay_samples(curriculum_data, max_samples=int(curriculum_replay_count))
        )
    if include_center_replay:
        replay_samples.extend(center_replay_samples(int(center_replay_repeats)))
    replay_samples.extend(
        npz_replay_samples(
            tactical_restoration_data or "",
            max_samples=int(tactical_restoration_replay_count),
            reason="tactical_restoration_curriculum_replay",
        )
    )
    final_samples, build_summary = build_hybrid_survival_samples(
        hybrid_samples,
        replay_samples=replay_samples,
        reason_weights=reason_weights,
        max_low_heuristic_ratio=max_low_heuristic_ratio,
    )
    if not final_samples:
        raise RuntimeError("hybrid survival produced no samples")
    shapes = save_mistake_samples(final_samples, output_path)
    summary = {
        **build_summary,
        "student_checkpoint": student_checkpoint,
        "output_path": output_path,
        "seeds": list(seeds),
        "games_per_seed": int(games),
        "rule_mode": rule_mode,
        "num_simulations": int(num_simulations),
        "max_moves": int(max_moves),
        "min_score_gap": float(min_score_gap),
        "shapes": shapes,
        "collections": collection_summaries,
        "elapsed_sec": time.perf_counter() - start,
    }
    _write_json(summary, stats_path)
    progress_print(
        f"DONE hybrid_survival raw={summary['raw_hybrid_samples']} "
        f"final={summary['final_samples']} low_ratio="
        f"{summary['low_heuristic_cap'].get('ratio_after', 0.0):.3f} "
        f"elapsed={format_seconds(summary['elapsed_sec'])} output={output_path}",
        "hybrid-survival",
    )
    return summary


__all__ = [
    "DEFAULT_HYBRID_SURVIVAL_DATA_PATH",
    "DEFAULT_HYBRID_SURVIVAL_STATS_PATH",
    "DEFAULT_HYBRID_SURVIVAL_V2_DATA_PATH",
    "DEFAULT_HYBRID_SURVIVAL_V2_METADATA_PATH",
    "DEFAULT_HYBRID_SURVIVAL_V2_STATS_PATH",
    "DEFAULT_HYBRID_SURVIVAL_V3_DATA_PATH",
    "DEFAULT_HYBRID_SURVIVAL_V3_METADATA_PATH",
    "DEFAULT_HYBRID_SURVIVAL_V3_STATS_PATH",
    "AnnotatedMistakeSample",
    "HYBRID_SURVIVAL_REASON_WEIGHTS",
    "build_hybrid_survival_dataset",
    "build_hybrid_survival_samples",
    "build_hybrid_survival_v2_dataset",
    "build_hybrid_survival_v2_samples",
    "build_hybrid_survival_v3_forced_block_dataset",
    "build_hybrid_survival_v3_samples",
    "collect_hybrid_survival_v2_candidates",
    "npz_replay_samples",
    "save_annotated_samples",
]
