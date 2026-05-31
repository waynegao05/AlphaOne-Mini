from __future__ import annotations

import json
import os


def test_run_deep_training_pipeline_smoke(tmp_path):
    from train.deep_training_pipeline import run_deep_training_pipeline

    summary = run_deep_training_pipeline(
        {
            "device": "cuda",
            "allow_cpu_fallback": True,
            "model_type": "advanced",
            "rule_mode": "basic",
            "tactical_games": 1,
            "pretrain_epochs": 1,
            "selfplay_games": 0,
            "finetune_epochs": 0,
            "batch_size": 4,
            "num_simulations": 2,
            "benchmark_games": 1,
            "use_augmentation": True,
            "use_auxiliary_loss": True,
            "output_dir": str(tmp_path / "deep"),
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "supervised_dir": str(tmp_path / "supervised"),
            "evaluation_dir": str(tmp_path / "evaluation"),
            "max_moves": 6,
        }
    )

    assert summary["model_type"] == "advanced"
    assert summary["pretrain"]["status"] == "completed"
    assert os.path.exists(summary["paths"]["pretrained_checkpoint"])
    assert os.path.exists(summary["paths"]["summary"])
    with open(summary["paths"]["summary"], "r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    assert loaded["model_type"] == "advanced"
