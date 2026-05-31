"""CUDA-oriented deep model training smoke pipeline."""

from __future__ import annotations

import json
import os
import time
from typing import Any

import numpy as np

from model.model_factory import create_model
from selfplay.replay_buffer import ReplayBuffer
from selfplay.self_play import SelfPlayGame
from train.supervised_pretrain import train_policy_pretrain
from train.tactical_distillation import generate_tactical_dataset
from train.progress import format_seconds, progress_print
from utils.device import describe_device, get_device


def _path(*parts: str) -> str:
    return os.path.join(*parts)


def _save_summary(summary: dict, path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def run_deep_training_pipeline(config: dict[str, Any] | None = None) -> dict:
    """Run tactical distillation, pretraining, optional self-play fine-tune, benchmark."""
    pipeline_start = time.perf_counter()
    cfg = {
        "device": "cuda",
        "allow_cpu_fallback": False,
        "model_type": "advanced",
        "rule_mode": "basic",
        "tactical_games": 10,
        "pretrain_epochs": 3,
        "selfplay_games": 2,
        "finetune_epochs": 1,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "num_simulations": 50,
        "benchmark_games": 10,
        "use_augmentation": True,
        "use_auxiliary_loss": True,
        "seed": 2026,
        "output_dir": _path("outputs", "deep_training"),
        "checkpoint_dir": _path("outputs", "checkpoints"),
        "supervised_dir": _path("outputs", "supervised"),
        "selfplay_dir": _path("outputs", "selfplay_data"),
        "evaluation_dir": _path("outputs", "evaluation"),
        "max_moves": 30,
        "resume_from": None,
        "scheduler": "constant",
        "warmup_epochs": 0,
        "grad_clip": 5.0,
        "mixed_precision": False,
    }
    if config:
        cfg.update(config)

    progress_print(
        f"START deep_training_pipeline model_type={cfg['model_type']} device={cfg['device']}",
        "deep_train",
    )
    device = get_device(cfg["device"], allow_cpu_fallback=bool(cfg["allow_cpu_fallback"]))
    progress_print(f"DONE stage=device resolved={describe_device(device)}", "deep_train")
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["supervised_dir"], exist_ok=True)
    os.makedirs(cfg["selfplay_dir"], exist_ok=True)
    os.makedirs(cfg["evaluation_dir"], exist_ok=True)
    os.makedirs(cfg["output_dir"], exist_ok=True)

    tactical_data_path = _path(cfg["supervised_dir"], "tactical_distill_latest.npz")
    pretrained_path = _path(cfg["checkpoint_dir"], "pretrained_advanced.pt")
    selfplay_path = _path(cfg["selfplay_dir"], "selfplay_latest.npz")
    latest_path = _path(cfg["checkpoint_dir"], "latest_advanced.pt")
    benchmark_path = _path(cfg["evaluation_dir"], "deep_benchmark_latest.json")
    summary_path = _path(cfg["output_dir"], "deep_training_summary.json")

    stage_start = time.perf_counter()
    progress_print("START stage=tactical_distillation", "deep_train")
    states, policies, values = generate_tactical_dataset(
        num_games=int(cfg["tactical_games"]),
        output_path=tactical_data_path,
        rule_mode=cfg["rule_mode"],
        max_moves=int(cfg["max_moves"]),
        seed=int(cfg["seed"]),
        include_auxiliary_labels=bool(cfg["use_auxiliary_loss"]),
        use_augmentation=bool(cfg["use_augmentation"]),
        progress_interval=int(cfg.get("progress_interval", 10)),
    )
    progress_print(
        f"DONE stage=tactical_distillation samples={int(states.shape[0])} "
        f"path={tactical_data_path} elapsed={format_seconds(time.perf_counter() - stage_start)}",
        "deep_train",
    )

    stage_start = time.perf_counter()
    progress_print("START stage=pretrain", "deep_train")
    model = create_model(cfg["model_type"])
    pretrain_history = train_policy_pretrain(
        model,
        data_path=tactical_data_path,
        checkpoint_dir=cfg["checkpoint_dir"],
        epochs=int(cfg["pretrain_epochs"]),
        batch_size=int(cfg["batch_size"]),
        lr=float(cfg["learning_rate"]),
        device=str(device),
        use_auxiliary_loss=bool(cfg["use_auxiliary_loss"]),
        checkpoint_name="pretrained_advanced.pt",
        model_type=cfg["model_type"],
        resume_from=cfg.get("resume_from"),
        scheduler_type=str(cfg.get("scheduler", "constant")),
        warmup_epochs=int(cfg.get("warmup_epochs", 0)),
        grad_clip=float(cfg.get("grad_clip", 5.0)),
        mixed_precision=bool(cfg.get("mixed_precision", False)),
    )
    progress_print(
        f"DONE stage=pretrain epochs={len(pretrain_history)} checkpoint={pretrained_path} "
        f"elapsed={format_seconds(time.perf_counter() - stage_start)}",
        "deep_train",
    )

    selfplay_status = {"status": "skipped", "num_samples": 0}
    finetune_status = {"status": "skipped"}
    if int(cfg["selfplay_games"]) > 0:
        stage_start = time.perf_counter()
        progress_print("START stage=selfplay", "deep_train")
        game = SelfPlayGame(
            model=model,
            num_simulations=int(cfg["num_simulations"]),
            device=str(device),
            max_moves=int(cfg["max_moves"]),
            rng=np.random.default_rng(int(cfg["seed"])),
        )
        samples = []
        total_games = int(cfg["selfplay_games"])
        for game_idx in range(1, total_games + 1):
            game_samples = game.play_game()
            samples.extend(game_samples)
            progress_print(
                f"game {game_idx}/{total_games} complete samples={len(game_samples)} "
                f"total_samples={len(samples)} winner={game.last_winner} moves={game.last_move_count}",
                "selfplay",
            )
        buffer = ReplayBuffer(capacity=max(1, len(samples) or 1), seed=int(cfg["seed"]))
        buffer.extend(samples)
        buffer.save(selfplay_path)
        selfplay_status = {"status": "completed", "num_samples": len(samples)}
        progress_print(
            f"DONE stage=selfplay games={total_games} samples={len(samples)} "
            f"path={selfplay_path} elapsed={format_seconds(time.perf_counter() - stage_start)}",
            "deep_train",
        )
        if int(cfg["finetune_epochs"]) > 0 and len(samples) > 0:
            stage_start = time.perf_counter()
            progress_print("START stage=finetune", "deep_train")
            finetune_history = train_policy_pretrain(
                model,
                data_path=selfplay_path,
                checkpoint_dir=cfg["checkpoint_dir"],
                epochs=int(cfg["finetune_epochs"]),
                batch_size=int(cfg["batch_size"]),
                lr=float(cfg["learning_rate"]),
                device=str(device),
                use_auxiliary_loss=False,
                checkpoint_name="latest_advanced.pt",
                model_type=cfg["model_type"],
                scheduler_type=str(cfg.get("scheduler", "constant")),
                warmup_epochs=int(cfg.get("warmup_epochs", 0)),
                grad_clip=float(cfg.get("grad_clip", 5.0)),
                mixed_precision=bool(cfg.get("mixed_precision", False)),
            )
            finetune_status = {"status": "completed", "history": finetune_history}
            progress_print(
                f"DONE stage=finetune epochs={len(finetune_history)} checkpoint={latest_path} "
                f"elapsed={format_seconds(time.perf_counter() - stage_start)}",
                "deep_train",
            )
    else:
        progress_print("SKIP stage=selfplay selfplay_games=0", "deep_train")

    from evaluate.deep_benchmark import run_deep_benchmark

    stage_start = time.perf_counter()
    progress_print("START stage=benchmark", "deep_train")
    benchmark = run_deep_benchmark(
        games=int(cfg["benchmark_games"]),
        device=str(device),
        allow_cpu_fallback=True,
        num_simulations=int(cfg["num_simulations"]),
        rule_mode=cfg["rule_mode"],
        output=benchmark_path,
        checkpoints={"advanced": latest_path if os.path.exists(latest_path) else pretrained_path},
        max_moves=int(cfg["max_moves"]),
    )
    progress_print(
        f"DONE stage=benchmark matches={len(benchmark.get('matches', {}))} "
        f"path={benchmark_path} elapsed={format_seconds(time.perf_counter() - stage_start)}",
        "deep_train",
    )

    summary = {
        "model_type": cfg["model_type"],
        "rule_mode": cfg["rule_mode"],
        "device": describe_device(device),
        "cuda_available": __import__("torch").cuda.is_available(),
        "resume": {"resumed": bool(cfg.get("resume_from")), "from": cfg.get("resume_from")},
        "scheduler": cfg.get("scheduler", "constant"),
        "mixed_precision": bool(cfg.get("mixed_precision", False)),
        "pretrain": {"status": "completed", "history": pretrain_history},
        "selfplay": selfplay_status,
        "finetune": finetune_status,
        "benchmark": benchmark,
        "paths": {
            "tactical_data": tactical_data_path,
            "pretrained_checkpoint": pretrained_path,
            "selfplay_data": selfplay_path,
            "latest_checkpoint": latest_path,
            "benchmark": benchmark_path,
            "summary": summary_path,
        },
        "data": {"num_samples": int(states.shape[0])},
        "note": "Smoke-scale deep training loop; not competition-strength.",
    }
    _save_summary(summary, summary_path)
    progress_print(
        f"DONE deep_training_pipeline summary={summary_path} "
        f"elapsed={format_seconds(time.perf_counter() - pipeline_start)}",
        "deep_train",
    )
    return summary


__all__ = ["run_deep_training_pipeline"]
