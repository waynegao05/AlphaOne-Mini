from __future__ import annotations

import os
import torch


def _checkpoint(path, model_type="advanced"):
    torch.save(
        {
            "model_state_dict": {"dummy": torch.tensor([1.0])},
            "metadata": {"model_type": model_type},
        },
        path,
    )


def _benchmark(total_games: int, win_rate: float):
    return {
        "rule_mode": "basic",
        "matches": {
            "latest_advanced_vs_tactical": {
                "total_games": total_games,
                "player_a_win_rate": win_rate,
                "player_a_name": "latest_advanced",
                "player_b_name": "tactical",
            }
        },
    }


def test_smoke_benchmark_only_marks_provisional_best(tmp_path):
    from train.promotion import promote_checkpoint_if_eligible

    candidate = tmp_path / "latest_advanced.pt"
    best = tmp_path / "best_advanced.pt"
    _checkpoint(candidate)

    decision = promote_checkpoint_if_eligible(
        candidate_path=str(candidate),
        best_path=str(best),
        benchmark_summary=_benchmark(total_games=4, win_rate=1.0),
        match_key="latest_advanced_vs_tactical",
        threshold=0.55,
        min_games=20,
    )

    assert decision["promoted"] is False
    assert decision["provisional_best"] is True
    assert not best.exists()


def test_promotes_best_advanced_when_threshold_met_with_enough_games(tmp_path):
    from train.promotion import promote_checkpoint_if_eligible

    candidate = tmp_path / "latest_advanced.pt"
    best = tmp_path / "best_advanced.pt"
    _checkpoint(candidate)

    decision = promote_checkpoint_if_eligible(
        candidate_path=str(candidate),
        best_path=str(best),
        benchmark_summary=_benchmark(total_games=30, win_rate=0.7),
        match_key="latest_advanced_vs_tactical",
        threshold=0.55,
        min_games=20,
    )

    assert decision["promoted"] is True
    assert decision["provisional_best"] is False
    assert best.exists()
    state = torch.load(best, map_location="cpu", weights_only=False)
    metadata = state["metadata"]
    assert metadata["promoted_from"] == os.path.abspath(candidate)
    assert metadata["benchmark_games"] == 30
    assert metadata["win_rate"] == 0.7
    assert metadata["opponent"] == "tactical"
    assert metadata["rule_mode"] == "basic"
    assert metadata["timestamp"]


def test_does_not_promote_when_threshold_not_met(tmp_path):
    from train.promotion import promote_checkpoint_if_eligible

    candidate = tmp_path / "latest_advanced.pt"
    best = tmp_path / "best_advanced.pt"
    _checkpoint(candidate)

    decision = promote_checkpoint_if_eligible(
        candidate_path=str(candidate),
        best_path=str(best),
        benchmark_summary=_benchmark(total_games=30, win_rate=0.4),
        match_key="latest_advanced_vs_tactical",
        threshold=0.55,
        min_games=20,
    )

    assert decision["promoted"] is False
    assert decision["provisional_best"] is False
    assert not best.exists()
