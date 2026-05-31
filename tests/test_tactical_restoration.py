from __future__ import annotations

import numpy as np

from game.encoder import action_to_index
from train.mistake_mining import MistakeSample


def _sample(*reasons: str) -> MistakeSample:
    policy = np.zeros(225, dtype=np.float32)
    policy[action_to_index(7, 7)] = 1.0
    return MistakeSample(
        state=np.zeros((4, 15, 15), dtype=np.float32),
        policy=policy,
        value=0.0,
        threat_labels=np.zeros((12, 15, 15), dtype=np.float32),
        forbidden_labels=np.zeros((1, 15, 15), dtype=np.float32),
        tactical_scores=np.zeros(225, dtype=np.float32),
        teacher_action=action_to_index(7, 7),
        reasons=tuple(reasons),
    )


def test_tactical_restoration_balancer_caps_low_heuristic_and_keeps_replay():
    from train.tactical_restoration import build_tactical_restoration_samples

    base = [_sample("low_heuristic_move") for _ in range(20)]
    base += [_sample("missed_immediate_block") for _ in range(4)]
    replay = [_sample("curriculum_replay"), _sample("center_replay")]

    samples, summary = build_tactical_restoration_samples(
        base,
        replay_samples=replay,
        reason_weights={"missed_immediate_block": 5, "low_heuristic_move": 1},
        max_low_heuristic_ratio=0.15,
    )

    low = sum("low_heuristic_move" in sample.reasons for sample in samples)
    assert low / len(samples) <= 0.15
    assert summary["reason_distribution_after"]["curriculum_replay"] == 1
    assert summary["reason_distribution_after"]["center_replay"] == 1


def test_defense_replay_samples_include_required_reasons():
    from train.tactical_restoration import defense_replay_samples

    samples = defense_replay_samples(repeats=1)
    reasons = {reason for sample in samples for reason in sample.reasons}

    assert "missed_immediate_block" in reasons
    assert "missed_open_four_defense" in reasons
    assert "missed_blocked_four_defense" in reasons
    assert "missed_immediate_win" in reasons


def test_main_tactical_restoration_train_parses_two_candidates():
    from main_tactical_restoration_train import parse_args

    args = parse_args(
        [
            "--v1-checkpoint",
            "outputs/checkpoints/latest_advanced_mistake_tuned.pt",
            "--curriculum-checkpoint",
            "outputs/checkpoints/curriculum_advanced.pt",
            "--epochs",
            "4",
            "--device",
            "cuda",
        ]
    )

    assert args.v1_checkpoint.endswith("latest_advanced_mistake_tuned.pt")
    assert args.curriculum_checkpoint.endswith("curriculum_advanced.pt")
    assert args.epochs == 4
