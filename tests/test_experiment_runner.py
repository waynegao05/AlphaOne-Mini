from __future__ import annotations

import json
import os

import numpy as np


def test_run_experiment_smoke_and_resume(tmp_path, capsys):
    from train.experiment_runner import run_experiment

    base = {
        "experiment_name": "pytest_smoke",
        "seed": 1,
        "device": "cuda",
        "allow_cpu_fallback": True,
        "model_type": "advanced",
        "board_size": 15,
        "rule_mode": "basic",
        "tactical_games": 1,
        "use_augmentation": False,
        "use_auxiliary_loss": True,
        "pretrain_epochs": 1,
        "finetune_epochs": 0,
        "batch_size": 4,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "num_simulations": 2,
        "selfplay_games": 0,
        "benchmark_games": 1,
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "log_dir": str(tmp_path / "logs"),
        "output_dir": str(tmp_path / "experiments"),
        "resume_from": None,
        "promote": False,
        "loss_weights": {"policy": 1.0, "value": 1.0, "threat": 0.3, "forbidden": 0.2},
        "scheduler": "constant",
        "warmup_epochs": 0,
        "grad_clip": 5.0,
        "mixed_precision": False,
        "max_moves": 5,
    }
    summary = run_experiment(base)
    output = capsys.readouterr().out
    assert summary["status"] == "completed"
    assert "[experiment] START stage=tactical_distillation" in output
    assert "[experiment] DONE stage=tactical_distillation" in output
    assert "[experiment] START stage=pretrain" in output
    assert "[experiment] DONE stage=benchmark" in output
    assert "[experiment] DONE experiment=pytest_smoke" in output
    ckpt = summary["paths"]["pretrained_checkpoint"]
    assert os.path.exists(ckpt)
    assert os.path.exists(summary["paths"]["summary"])
    assert os.path.exists(summary["paths"]["report"])

    resumed = dict(base)
    resumed["resume_from"] = ckpt
    resumed_summary = run_experiment(resumed)
    assert resumed_summary["resume"]["resumed"] is True
    with open(resumed_summary["paths"]["summary"], "r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    assert loaded["experiment_name"] == "pytest_smoke"


def test_run_experiment_selfplay_and_finetune_smoke(tmp_path, capsys):
    from train.experiment_runner import run_experiment

    config = {
        "experiment_name": "pytest_selfplay",
        "seed": 2,
        "device": "cpu",
        "allow_cpu_fallback": True,
        "model_type": "cnn",
        "board_size": 15,
        "rule_mode": "basic",
        "tactical_games": 1,
        "use_augmentation": False,
        "use_auxiliary_loss": False,
        "pretrain_epochs": 1,
        "finetune_epochs": 1,
        "batch_size": 4,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "num_simulations": 1,
        "selfplay_games": 1,
        "benchmark_games": 0,
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "log_dir": str(tmp_path / "logs"),
        "output_dir": str(tmp_path / "experiments"),
        "resume_from": None,
        "promote": False,
        "loss_weights": {"policy": 1.0, "value": 1.0},
        "scheduler": "constant",
        "warmup_epochs": 0,
        "grad_clip": 5.0,
        "mixed_precision": False,
        "max_moves": 1,
    }

    summary = run_experiment(config)
    output = capsys.readouterr().out

    assert summary["status"] == "completed"
    assert "[experiment] START stage=selfplay" in output
    assert "[selfplay] game 1/1 complete" in output
    assert "[experiment] DONE stage=finetune" in output
    assert summary["selfplay"]["status"] == "completed"
    assert summary["selfplay"]["num_samples"] > 0
    assert summary["finetune"]["status"] == "completed"
    assert os.path.exists(summary["paths"]["selfplay_data"])
    assert os.path.exists(summary["paths"]["latest_checkpoint"])


def test_run_experiment_uses_online_augmentation_for_pretrain(monkeypatch, tmp_path):
    import train.experiment_runner as runner

    captured = {}

    def fake_generate_tactical_dataset(**kwargs):
        captured["generate_use_augmentation"] = kwargs["use_augmentation"]
        path = kwargs["output_path"]
        states = np.zeros((1, 4, 15, 15), dtype=np.float32)
        policies = np.zeros((1, 225), dtype=np.float32)
        values = np.zeros((1, 1), dtype=np.float32)
        np.savez_compressed(path, states=states, policies=policies, values=values)
        return states, policies, values

    def fake_train_policy_pretrain(*args, **kwargs):
        captured["augment_dataset"] = kwargs["augment_dataset"]
        return [{"epoch": 1, "total_loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}]

    monkeypatch.setattr(runner, "generate_tactical_dataset", fake_generate_tactical_dataset)
    monkeypatch.setattr(runner, "train_policy_pretrain", fake_train_policy_pretrain)
    monkeypatch.setattr(
        runner,
        "run_model_comparison",
        lambda **kwargs: {"matches": {}, "summary": {"status": "ok"}},
    )

    summary = runner.run_experiment(
        {
            "experiment_name": "online_aug_test",
            "seed": 3,
            "device": "cpu",
            "allow_cpu_fallback": True,
            "model_type": "cnn",
            "board_size": 15,
            "rule_mode": "basic",
            "tactical_games": 1,
            "use_augmentation": True,
            "use_auxiliary_loss": False,
            "pretrain_epochs": 1,
            "finetune_epochs": 0,
            "batch_size": 4,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "num_simulations": 1,
            "selfplay_games": 0,
            "benchmark_games": 0,
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "log_dir": str(tmp_path / "logs"),
            "output_dir": str(tmp_path / "experiments"),
            "resume_from": None,
            "promote": False,
            "loss_weights": {"policy": 1.0, "value": 1.0},
            "scheduler": "constant",
            "warmup_epochs": 0,
            "grad_clip": 5.0,
            "mixed_precision": False,
            "max_moves": 1,
        }
    )

    assert summary["status"] == "completed"
    assert captured == {
        "generate_use_augmentation": False,
        "augment_dataset": True,
    }
