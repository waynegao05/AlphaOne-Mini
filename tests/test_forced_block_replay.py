from __future__ import annotations

import json

import numpy as np

from game.encoder import index_to_action


def _occupied_from_state(state: np.ndarray, action: int) -> bool:
    x, y = index_to_action(int(action))
    return bool(state[0, y, x] != 0 or state[1, y, x] != 0)


def test_immediate_block_replay_has_expected_metadata_and_labels():
    from train.auxiliary_labels import THREAT_CHANNELS
    from train.forced_block_replay import generate_immediate_block_positions

    samples = generate_immediate_block_positions(rule_mode="basic")

    assert len(samples) >= 16
    assert {item.metadata["player_to_move"] for item in samples} == {1, -1}
    opponent_win_idx = THREAT_CHANNELS.index("opponent_win_point")
    for item in samples:
        action = int(np.argmax(item.sample.policy))
        x, y = index_to_action(action)
        assert item.metadata["reason"] == ["missed_immediate_block"]
        assert item.metadata["expected_action"] == action
        assert item.metadata["target_source"] == "forced_block_replay"
        assert not _occupied_from_state(item.sample.state, action)
        assert item.sample.threat_labels[opponent_win_idx, y, x] == 1.0
        assert np.isclose(item.sample.policy.sum(), 1.0)


def test_forced_block_dataset_contains_required_reasons_and_valid_targets():
    from train.forced_block_replay import generate_forced_block_replay_samples

    samples = generate_forced_block_replay_samples(rule_mode="basic")
    reasons = {reason for item in samples for reason in item.metadata["reason"]}

    assert {
        "missed_immediate_block",
        "missed_open_four_defense",
        "missed_blocked_four_defense",
        "missed_double_threat_defense",
    }.issubset(reasons)
    for item in samples:
        action = int(np.argmax(item.sample.policy))
        assert item.metadata["teacher_action"] == action
        assert item.metadata["target_source"] == "forced_block_replay"
        assert not _occupied_from_state(item.sample.state, action)


def test_save_forced_block_dataset_writes_npz_and_metadata(tmp_path):
    from train.forced_block_replay import save_forced_block_replay_dataset

    data_path = tmp_path / "forced_block.npz"
    metadata_path = tmp_path / "forced_block.jsonl"

    summary = save_forced_block_replay_dataset(
        output_path=str(data_path),
        metadata_path=str(metadata_path),
        rule_mode="basic",
    )

    assert data_path.exists()
    assert metadata_path.exists()
    assert summary["total_samples"] > 0
    with np.load(data_path, allow_pickle=False) as data:
        assert data["states"].shape[0] == summary["total_samples"]
        assert data["policies"].shape[1] == 225
        assert np.allclose(data["policies"].sum(axis=1), 1.0)
    first = json.loads(metadata_path.read_text(encoding="utf-8").splitlines()[0])
    assert first["target_source"] == "forced_block_replay"


def test_hybrid_survival_v3_builder_meets_forced_block_ratios(tmp_path):
    from train.forced_block_replay import generate_forced_block_replay_samples
    from train.hybrid_survival import (
        AnnotatedMistakeSample,
        build_hybrid_survival_v3_samples,
    )
    from train.mistake_mining import make_mistake_sample
    from game.board import Board
    from game.encoder import action_to_index

    def replay(reason: str, count: int):
        result = []
        for idx in range(count):
            sample = make_mistake_sample(
                Board(),
                teacher_action=action_to_index(7, 7),
                final_winner=0,
                reasons=(reason,),
            )
            result.append(
                AnnotatedMistakeSample(
                    sample=sample,
                    metadata={
                        "reason": [reason],
                        "target_source": reason,
                        "teacher_action": int(sample.teacher_action),
                        "remaining_moves": 20,
                    },
                )
            )
        return result

    forced = generate_forced_block_replay_samples(rule_mode="basic")
    self_survival = replay("v2_self_survival_replay", 8)
    curriculum = replay("curriculum_replay", 8)
    center = replay("center_replay", 4)
    hybrid = replay("teacher_student_disagree", 8)

    final, summary = build_hybrid_survival_v3_samples(
        forced,
        self_survival,
        curriculum + center,
        hybrid,
        target_samples=100,
    )

    assert len(final) == summary["final_samples"]
    assert summary["missed_immediate_block_ratio"] >= 0.15
    assert summary["forced_defense_combined_ratio"] >= 0.45
    assert summary["near_loss_ratio"] <= 0.10
    assert summary["low_heuristic_ratio"] <= 0.10
