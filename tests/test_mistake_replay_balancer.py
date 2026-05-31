from __future__ import annotations

import numpy as np

from train.mistake_mining import MistakeSample


def _sample(*reasons: str) -> MistakeSample:
    policy = np.zeros(225, dtype=np.float32)
    policy[112] = 1.0
    return MistakeSample(
        state=np.zeros((4, 15, 15), dtype=np.float32),
        policy=policy,
        value=0.0,
        threat_labels=np.zeros((12, 15, 15), dtype=np.float32),
        forbidden_labels=np.zeros((1, 15, 15), dtype=np.float32),
        tactical_scores=np.zeros(225, dtype=np.float32),
        teacher_action=112,
        reasons=tuple(reasons),
    )


def test_parse_ratio_and_weight_specs():
    from train.mistake_replay_balancer import parse_ratio_spec, parse_weight_spec

    assert parse_ratio_spec("tactical:0.5,hybrid:0.5") == {
        "tactical": 0.5,
        "hybrid": 0.5,
    }
    assert parse_weight_spec("missed_immediate_block:5,low_heuristic_move:1") == {
        "missed_immediate_block": 5.0,
        "low_heuristic_move": 1.0,
    }


def test_balance_teacher_groups_oversamples_underrepresented_teacher():
    from train.mistake_replay_balancer import balance_teacher_groups

    balanced, summary = balance_teacher_groups(
        {
            "tactical": [_sample("missed_immediate_block")],
            "hybrid": [_sample("low_heuristic_move") for _ in range(3)],
        },
        {"tactical": 0.5, "hybrid": 0.5},
    )

    assert summary["teacher_counts_after"]["tactical"] == summary["teacher_counts_after"]["hybrid"]
    assert len(balanced) == 4


def test_cap_reason_ratio_limits_low_heuristic_dominance():
    from train.mistake_replay_balancer import cap_reason_ratio

    samples = [_sample("low_heuristic_move") for _ in range(10)]
    samples += [_sample("missed_immediate_block") for _ in range(6)]

    capped, summary = cap_reason_ratio(samples, reason="low_heuristic_move", max_ratio=0.25)

    low_count = sum("low_heuristic_move" in sample.reasons for sample in capped)
    assert low_count / len(capped) <= 0.25
    assert summary["removed"] > 0


def test_apply_reason_weights_repeats_critical_more_than_low_heuristic():
    from train.mistake_replay_balancer import apply_reason_weights

    expanded, summary = apply_reason_weights(
        [_sample("missed_immediate_block"), _sample("low_heuristic_move")],
        {"missed_immediate_block": 5, "low_heuristic_move": 1},
    )

    assert len(expanded) == 6
    assert summary["weighted_samples_added"] == 4
