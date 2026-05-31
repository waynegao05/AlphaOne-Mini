from __future__ import annotations

import json
import os


def test_deep_benchmark_seed_helper_resets_random_generators():
    import random

    import numpy as np
    import torch

    from evaluate.deep_benchmark import _set_global_seed

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    _set_global_seed(123)
    first = (random.random(), float(np.random.rand()), float(torch.rand(1).item()))

    random.seed(555)
    np.random.seed(555)
    torch.manual_seed(555)
    _set_global_seed(123)
    second = (random.random(), float(np.random.rand()), float(torch.rand(1).item()))

    assert first == second


def test_deep_benchmark_seed_for_key_is_stable_and_key_specific():
    from evaluate.deep_benchmark import _seed_for_key

    assert _seed_for_key(2026, "hybrid") == _seed_for_key(2026, "hybrid")
    assert _seed_for_key(2026, "hybrid") != _seed_for_key(2026, "neural_guarded")


def test_run_deep_benchmark_skips_missing_checkpoints(tmp_path):
    from evaluate.deep_benchmark import run_deep_benchmark

    output = tmp_path / "benchmark.json"
    summary = run_deep_benchmark(
        games=1,
        device="cuda",
        allow_cpu_fallback=True,
        num_simulations=2,
        rule_mode="basic",
        output=str(output),
        checkpoints={
            "cnn": str(tmp_path / "missing_cnn.pt"),
            "advanced": str(tmp_path / "missing_advanced.pt"),
        },
    )

    assert output.exists()
    assert "TacticalPlayer_vs_RandomPlayer" in summary["matches"]
    assert "HybridPlayer_vs_RandomPlayer" in summary["matches"]
    assert summary["is_smoke_test"] is True
    assert summary["timestamp"]
    match = summary["matches"]["TacticalPlayer_vs_RandomPlayer"]
    for key in (
        "total_games",
        "player_a_wins",
        "player_b_wins",
        "draws",
        "player_a_win_rate",
        "player_b_win_rate",
        "draw_rate",
        "black_win_rate",
        "white_win_rate",
        "avg_moves",
        "rule_mode",
        "num_simulations",
        "device",
        "is_smoke_test",
        "timestamp",
    ):
        assert key in match
    assert summary["skipped_checkpoints"]
    with open(output, "r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    assert loaded["games"] == 1


def test_run_deep_benchmark_accepts_explicit_matchups(tmp_path):
    from evaluate.deep_benchmark import run_deep_benchmark

    output = tmp_path / "benchmark.json"
    summary = run_deep_benchmark(
        games=1,
        device="cuda",
        allow_cpu_fallback=True,
        num_simulations=1,
        rule_mode="basic",
        output=str(output),
        matchups=["tactical_vs_hybrid"],
        max_moves=8,
    )

    assert list(summary["matches"].keys()) == ["tactical_vs_hybrid"]
    assert output.exists()


def test_main_deep_benchmark_parses_matchups_and_curriculum_checkpoint():
    from main_deep_benchmark import parse_args

    args = parse_args(
        [
            "--matchups",
            "latest_advanced_vs_tactical",
            "latest_advanced_vs_curriculum",
            "--curriculum-advanced-checkpoint",
            "outputs/checkpoints/curriculum_advanced.pt",
        ]
    )

    assert args.matchups == [
        "latest_advanced_vs_tactical",
        "latest_advanced_vs_curriculum",
    ]
    assert args.curriculum_advanced_checkpoint == "outputs/checkpoints/curriculum_advanced.pt"


def test_main_deep_benchmark_parses_mistake_tuned_checkpoint():
    from main_deep_benchmark import parse_args

    args = parse_args(
        [
            "--matchups",
            "mistake_tuned_vs_latest",
            "--mistake-tuned-checkpoint",
            "outputs/checkpoints/latest_advanced_mistake_tuned.pt",
        ]
    )

    assert args.matchups == ["mistake_tuned_vs_latest"]
    assert args.mistake_tuned_checkpoint == "outputs/checkpoints/latest_advanced_mistake_tuned.pt"


def test_main_deep_benchmark_parses_mistake_v2_checkpoint():
    from main_deep_benchmark import parse_args

    args = parse_args(
        [
            "--matchups",
            "mistake_v2_from_latest_vs_tactical",
            "--mistake-v2-checkpoint",
            "outputs/checkpoints/latest_advanced_mistake_v2_from_latest.pt",
        ]
    )

    assert args.matchups == ["mistake_v2_from_latest_vs_tactical"]
    assert args.mistake_v2_checkpoint == "outputs/checkpoints/latest_advanced_mistake_v2_from_latest.pt"


def test_main_deep_benchmark_parses_mistake_v3_checkpoint():
    from main_deep_benchmark import parse_args

    args = parse_args(
        [
            "--matchups",
            "mistake_v3_vs_mistake_v2",
            "--mistake-v3-checkpoint",
            "outputs/checkpoints/latest_advanced_mistake_v3_teacher_balanced.pt",
        ]
    )

    assert args.matchups == ["mistake_v3_vs_mistake_v2"]
    assert args.mistake_v3_checkpoint == "outputs/checkpoints/latest_advanced_mistake_v3_teacher_balanced.pt"


def test_main_deep_benchmark_parses_tactical_restoration_checkpoints():
    from main_deep_benchmark import parse_args

    args = parse_args(
        [
            "--matchups",
            "tactical_restoration_v1_vs_tactical",
            "tactical_restoration_curriculum_vs_tactical",
            "--tactical-restoration-v1-checkpoint",
            "outputs/checkpoints/latest_advanced_tactical_restoration_from_v1.pt",
            "--tactical-restoration-curriculum-checkpoint",
            "outputs/checkpoints/latest_advanced_tactical_restoration_from_curriculum.pt",
        ]
    )

    assert args.tactical_restoration_v1_checkpoint.endswith("from_v1.pt")
    assert args.tactical_restoration_curriculum_checkpoint.endswith("from_curriculum.pt")


def test_main_deep_benchmark_parses_hybrid_survival_checkpoint():
    from main_deep_benchmark import parse_args

    args = parse_args(
        [
            "--matchups",
            "hybrid_survival_vs_hybrid",
            "hybrid_survival_vs_v2",
            "tactical_restoration_curriculum_vs_v3",
            "--hybrid-survival-checkpoint",
            "outputs/checkpoints/latest_advanced_hybrid_survival_from_v2.pt",
            "--mistake-v3-checkpoint",
            "outputs/checkpoints/latest_advanced_mistake_v3_teacher_balanced.pt",
        ]
    )

    assert args.hybrid_survival_checkpoint.endswith("hybrid_survival_from_v2.pt")
    assert args.mistake_v3_checkpoint.endswith("mistake_v3_teacher_balanced.pt")


def test_main_deep_benchmark_parses_hybrid_survival_v2_checkpoint():
    from main_deep_benchmark import parse_args

    args = parse_args(
        [
            "--matchups",
            "hybrid_survival_v2_vs_hybrid",
            "hybrid_survival_v2_vs_v2",
            "--hybrid-survival-v2-checkpoint",
            "outputs/checkpoints/latest_advanced_hybrid_survival_v2_from_v2.pt",
        ]
    )

    assert args.hybrid_survival_v2_checkpoint.endswith("hybrid_survival_v2_from_v2.pt")


def test_main_deep_benchmark_parses_hybrid_survival_v3_checkpoint():
    from main_deep_benchmark import parse_args

    args = parse_args(
        [
            "--matchups",
            "hybrid_survival_v3_vs_hybrid",
            "hybrid_survival_v3_vs_v2",
            "--hybrid-survival-v3-checkpoint",
            "outputs/checkpoints/latest_advanced_hybrid_survival_v3_forced_block.pt",
        ]
    )

    assert args.hybrid_survival_v3_checkpoint.endswith("hybrid_survival_v3_forced_block.pt")


def test_main_deep_benchmark_parses_neural_guarded_flag():
    from main_deep_benchmark import parse_args

    args = parse_args(
        [
            "--neural-guarded",
            "--fallback-mode",
            "conservative",
            "--matchups",
            "neural_guarded_vs_tactical",
            "--tactical-restoration-curriculum-checkpoint",
            "outputs/checkpoints/latest_advanced_tactical_restoration_from_curriculum.pt",
            "--mistake-v2-checkpoint",
            "outputs/checkpoints/latest_advanced_mistake_v2_from_latest.pt",
        ]
    )

    assert args.neural_guarded is True
    assert args.fallback_mode == "conservative"
    assert args.matchups == ["neural_guarded_vs_tactical"]


def test_parse_matchup_aliases_for_v2_v3_and_hybrid_survival():
    from evaluate.deep_benchmark import _parse_matchups

    parsed = _parse_matchups(
        [
            "hybrid_survival_vs_v2",
            "tactical_restoration_curriculum_vs_v3",
            "hybrid_survival_from_v2_vs_tactical_restoration_curriculum",
            "hybrid_survival_v2_from_v2_vs_hybrid",
            "hybrid_survival_v3_forced_block_vs_v2",
            "neural_guarded_vs_tactical_restoration_curriculum",
            "neural_guarded_full_vs_hybrid",
            "neural_guarded_no_guardrail_vs_hybrid",
            "neural_guarded_no_tactical_specialist_vs_hybrid",
            "neural_guarded_no_v2_vs_hybrid",
            "neural_guarded_no_hybrid_fallback_vs_hybrid",
            "neural_guarded_guardrail_only_vs_hybrid",
            "neural_guarded_hybrid_only_vs_hybrid",
            "neural_guarded_conservative_fallback_vs_hybrid",
            "neural_guarded_fallback_off_vs_hybrid",
            "neural_guarded_aggressive_fallback_vs_hybrid",
        ]
    )

    assert parsed == [
        ("hybrid_survival_vs_v2", "hybrid_survival", "mistake_v2"),
        ("tactical_restoration_curriculum_vs_v3", "tactical_restoration_curriculum", "mistake_v3"),
        (
            "hybrid_survival_from_v2_vs_tactical_restoration_curriculum",
            "hybrid_survival",
            "tactical_restoration_curriculum",
        ),
        ("hybrid_survival_v2_from_v2_vs_hybrid", "hybrid_survival_v2", "hybrid"),
        ("hybrid_survival_v3_forced_block_vs_v2", "hybrid_survival_v3", "mistake_v2"),
        ("neural_guarded_vs_tactical_restoration_curriculum", "neural_guarded", "tactical_restoration_curriculum"),
        ("neural_guarded_full_vs_hybrid", "neural_guarded_full", "hybrid"),
        ("neural_guarded_no_guardrail_vs_hybrid", "neural_guarded_no_guardrail", "hybrid"),
        ("neural_guarded_no_tactical_specialist_vs_hybrid", "neural_guarded_no_tactical_specialist", "hybrid"),
        ("neural_guarded_no_v2_vs_hybrid", "neural_guarded_no_v2", "hybrid"),
        ("neural_guarded_no_hybrid_fallback_vs_hybrid", "neural_guarded_no_hybrid_fallback", "hybrid"),
        ("neural_guarded_guardrail_only_vs_hybrid", "neural_guarded_guardrail_only", "hybrid"),
        ("neural_guarded_hybrid_only_vs_hybrid", "neural_guarded_hybrid_only", "hybrid"),
        ("neural_guarded_conservative_fallback_vs_hybrid", "neural_guarded_conservative_fallback", "hybrid"),
        ("neural_guarded_fallback_off_vs_hybrid", "neural_guarded_fallback_off", "hybrid"),
        ("neural_guarded_aggressive_fallback_vs_hybrid", "neural_guarded_aggressive_fallback", "hybrid"),
    ]
