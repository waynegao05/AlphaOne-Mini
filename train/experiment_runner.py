"""End-to-end experiment runner with config, logging, resume, and reports."""

from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Mapping

import numpy as np
import torch

from evaluate.model_comparison import run_model_comparison
from model.checkpoint import load_checkpoint_checked, load_checkpoint_metadata
from model.model_factory import create_model
from selfplay.replay_buffer import ReplayBuffer
from selfplay.self_play import SelfPlayGame
from train.config import load_config, merge_overrides, save_resolved_config, validate_config
from train.logger import TrainingLogger
from train.promotion import promote_checkpoint_if_eligible
from train.progress import format_seconds, progress_print
from train.supervised_pretrain import train_policy_pretrain
from train.tactical_distillation import generate_tactical_dataset
from utils.device import describe_device, get_device


def _parameter_count(model: torch.nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_config(config: str | Mapping[str, Any] | None, overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    if isinstance(config, str):
        resolved = load_config(config)
    elif config is None:
        resolved = load_config(None)
    else:
        resolved = merge_overrides(load_config(None), dict(config))
    resolved = merge_overrides(resolved, overrides or {})
    validate_config(resolved)
    return resolved


def _write_summary(summary: dict, path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def _stage_start(name: str) -> float:
    progress_print(f"START stage={name}", "experiment")
    return time.perf_counter()


def _stage_done(name: str, start: float, **fields: Any) -> None:
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    suffix = f" {details}" if details else ""
    progress_print(
        f"DONE stage={name} elapsed={format_seconds(time.perf_counter() - start)}{suffix}",
        "experiment",
    )


def run_experiment(
    config: str | Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict:
    """Run a reproducible deep-training experiment."""
    cfg = _resolve_config(config, overrides)
    experiment_name = str(cfg["experiment_name"])
    experiment_start = time.perf_counter()
    progress_print(
        f"START experiment={experiment_name} model_type={cfg['model_type']} "
        f"device={cfg['device']} rule_mode={cfg['rule_mode']}",
        "experiment",
    )
    exp_dir = os.path.join(str(cfg["output_dir"]), experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(str(cfg["checkpoint_dir"]), exist_ok=True)

    summary_path = os.path.join(exp_dir, "summary.json")
    report_path = os.path.join(exp_dir, "experiment_report.md")
    compare_json = os.path.join(exp_dir, "model_comparison.json")
    compare_md = os.path.join(exp_dir, "model_comparison.md")
    tactical_data_path = os.path.join(exp_dir, "tactical_distill.npz")
    selfplay_data_path = os.path.join(exp_dir, "selfplay_latest.npz")
    checkpoint_name = (
        f"pretrained_{cfg['model_type']}.pt"
        if cfg["model_type"] != "cnn"
        else "pretrained_cnn.pt"
    )
    latest_checkpoint_name = (
        f"latest_{cfg['model_type']}.pt"
        if cfg["model_type"] != "cnn"
        else "latest_cnn.pt"
    )
    checkpoint_path = os.path.join(str(cfg["checkpoint_dir"]), checkpoint_name)
    latest_checkpoint_path = os.path.join(str(cfg["checkpoint_dir"]), latest_checkpoint_name)

    logger = TrainingLogger(experiment_name, root_dir=str(cfg["log_dir"]))
    logger.save_config(cfg)
    save_resolved_config(cfg, os.path.join(exp_dir, "config_resolved.yaml"))

    summary: dict[str, Any] = {
        "experiment_name": experiment_name,
        "status": "running",
        "config": cfg,
        "model_type": cfg["model_type"],
        "paths": {
            "experiment_dir": exp_dir,
            "summary": summary_path,
            "report": report_path,
            "tactical_data": tactical_data_path,
            "selfplay_data": selfplay_data_path,
            "pretrained_checkpoint": checkpoint_path,
            "latest_checkpoint": latest_checkpoint_path,
            "model_comparison_json": compare_json,
            "model_comparison_md": compare_md,
        },
        "resume": {"resumed": False, "from": cfg.get("resume_from")},
    }

    try:
        _set_seed(int(cfg["seed"]))
        device_stage = _stage_start("device")
        device = get_device(
            str(cfg["device"]),
            allow_cpu_fallback=bool(cfg["allow_cpu_fallback"]),
        )
        summary["device"] = describe_device(device)
        summary["cuda_available"] = bool(torch.cuda.is_available())
        _stage_done("device", device_stage, resolved=summary["device"])

        model_stage = _stage_start("model_init")
        model = create_model(str(cfg["model_type"]), board_size=int(cfg["board_size"]))
        summary["parameter_count"] = _parameter_count(model)
        _stage_done("model_init", model_stage, parameters=summary["parameter_count"])

        if cfg.get("resume_from"):
            resume_stage = _stage_start("resume")
            metadata = load_checkpoint_metadata(str(cfg["resume_from"]), device=str(device))
            checkpoint_type = metadata.get("model_type")
            if checkpoint_type and checkpoint_type != cfg["model_type"]:
                raise ValueError(
                    f"resume checkpoint model_type {checkpoint_type!r} does not match "
                    f"config model_type {cfg['model_type']!r}"
                )
            summary["resume"] = {
                "resumed": True,
                "from": cfg["resume_from"],
                "metadata": metadata,
            }
            # Load now so self-play/fine-tune can resume even when pretraining is skipped.
            load_checkpoint_checked(
                model,
                str(cfg["resume_from"]),
                device=str(device),
                expected_model_type=str(cfg["model_type"]),
            )
            _stage_done("resume", resume_stage, checkpoint=cfg["resume_from"])

        start = time.perf_counter()
        states = None
        if int(cfg["tactical_games"]) > 0:
            tactical_stage = _stage_start("tactical_distillation")
            states, _, _ = generate_tactical_dataset(
                num_games=int(cfg["tactical_games"]),
                output_path=tactical_data_path,
                rule_mode=str(cfg["rule_mode"]),
                max_moves=int(cfg.get("max_moves", 30)),
                seed=int(cfg["seed"]),
                include_auxiliary_labels=bool(cfg["use_auxiliary_loss"]),
                use_augmentation=False,
                progress_interval=int(cfg.get("progress_interval", 10)),
            )
            _stage_done(
                "tactical_distillation",
                tactical_stage,
                samples=int(states.shape[0]),
                path=tactical_data_path,
            )
        else:
            progress_print("SKIP stage=tactical_distillation tactical_games=0", "experiment")
        summary["data"] = {
            "num_samples": int(states.shape[0]) if states is not None else 0,
            "use_augmentation": bool(cfg["use_augmentation"]),
            "use_auxiliary_loss": bool(cfg["use_auxiliary_loss"]),
        }

        history = []
        if int(cfg["pretrain_epochs"]) > 0 and states is not None:
            pretrain_stage = _stage_start("pretrain")
            history = train_policy_pretrain(
                model,
                data_path=tactical_data_path,
                checkpoint_dir=str(cfg["checkpoint_dir"]),
                epochs=int(cfg["pretrain_epochs"]),
                batch_size=int(cfg["batch_size"]),
                lr=float(cfg["learning_rate"]),
                device=str(device),
                weight_decay=float(cfg["weight_decay"]),
                grad_clip=float(cfg.get("grad_clip", 5.0)),
                use_auxiliary_loss=bool(cfg["use_auxiliary_loss"]),
                loss_weights=dict(cfg.get("loss_weights", {})),
                checkpoint_name=checkpoint_name,
                model_type=str(cfg["model_type"]),
                resume_from=cfg.get("resume_from"),
                scheduler_type=str(cfg.get("scheduler", "constant")),
                warmup_epochs=int(cfg.get("warmup_epochs", 0)),
                mixed_precision=bool(cfg.get("mixed_precision", False)),
                augment_dataset=bool(cfg["use_augmentation"]),
            )
            _stage_done(
                "pretrain",
                pretrain_stage,
                epochs=len(history),
                checkpoint=checkpoint_path,
            )
        else:
            progress_print("SKIP stage=pretrain", "experiment")
        for record in history:
            logger.log_epoch(
                {
                    **record,
                    "learning_rate": record.get("learning_rate", cfg["learning_rate"]),
                    "duration_sec": 0.0,
                    "device": summary["device"],
                    "model_type": cfg["model_type"],
                    "checkpoint_path": checkpoint_path,
                }
            )
        summary["pretrain"] = {
            "status": "completed" if history else "skipped",
            "history": history,
            "duration_sec": time.perf_counter() - start,
        }

        summary["selfplay"] = {
            "status": "skipped",
            "requested_games": int(cfg["selfplay_games"]),
            "num_samples": 0,
        }
        summary["finetune"] = {
            "status": "skipped",
            "requested_epochs": int(cfg["finetune_epochs"]),
        }
        if int(cfg["selfplay_games"]) > 0:
            selfplay_stage = _stage_start("selfplay")
            game = SelfPlayGame(
                model=model,
                num_simulations=int(cfg["num_simulations"]),
                device=str(device),
                max_moves=int(cfg.get("max_moves", 30)),
                rng=np.random.default_rng(int(cfg["seed"])),
            )
            samples = []
            total_games = int(cfg["selfplay_games"])
            for game_idx in range(1, total_games + 1):
                game_samples = game.play_game()
                samples.extend(game_samples)
                progress_print(
                    f"game {game_idx}/{total_games} complete "
                    f"samples={len(game_samples)} total_samples={len(samples)} "
                    f"winner={game.last_winner} moves={game.last_move_count}",
                    "selfplay",
                )
            buffer = ReplayBuffer(capacity=max(1, len(samples) or 1), seed=int(cfg["seed"]))
            buffer.extend(samples)
            buffer.save(selfplay_data_path)
            _stage_done(
                "selfplay",
                selfplay_stage,
                games=total_games,
                samples=len(samples),
                path=selfplay_data_path,
            )
            summary["selfplay"] = {
                "status": "completed",
                "requested_games": int(cfg["selfplay_games"]),
                "num_samples": len(samples),
                "path": selfplay_data_path,
            }
            if int(cfg["finetune_epochs"]) > 0 and samples:
                finetune_stage = _stage_start("finetune")
                finetune_history = train_policy_pretrain(
                    model,
                    data_path=selfplay_data_path,
                    checkpoint_dir=str(cfg["checkpoint_dir"]),
                    epochs=int(cfg["finetune_epochs"]),
                    batch_size=int(cfg["batch_size"]),
                    lr=float(cfg["learning_rate"]),
                    device=str(device),
                    weight_decay=float(cfg["weight_decay"]),
                    grad_clip=float(cfg.get("grad_clip", 5.0)),
                    use_auxiliary_loss=False,
                    checkpoint_name=latest_checkpoint_name,
                    model_type=str(cfg["model_type"]),
                    scheduler_type=str(cfg.get("scheduler", "constant")),
                    warmup_epochs=int(cfg.get("warmup_epochs", 0)),
                    mixed_precision=bool(cfg.get("mixed_precision", False)),
                )
                _stage_done(
                    "finetune",
                    finetune_stage,
                    epochs=len(finetune_history),
                    checkpoint=latest_checkpoint_path,
                )
                for record in finetune_history:
                    logger.log_epoch(
                        {
                            **record,
                            "phase": "finetune",
                            "learning_rate": record.get("learning_rate", cfg["learning_rate"]),
                            "duration_sec": 0.0,
                            "device": summary["device"],
                            "model_type": cfg["model_type"],
                            "checkpoint_path": latest_checkpoint_path,
                        }
                    )
                summary["finetune"] = {
                    "status": "completed",
                    "requested_epochs": int(cfg["finetune_epochs"]),
                    "history": finetune_history,
                    "checkpoint": latest_checkpoint_path,
                }
            elif int(cfg["finetune_epochs"]) > 0:
                summary["finetune"] = {
                    "status": "skipped_no_samples",
                    "requested_epochs": int(cfg["finetune_epochs"]),
                }
                progress_print("SKIP stage=finetune reason=no_samples", "experiment")
        else:
            progress_print("SKIP stage=selfplay selfplay_games=0", "experiment")
            if int(cfg["finetune_epochs"]) > 0:
                progress_print("SKIP stage=finetune reason=no_selfplay", "experiment")

        benchmark_stage = _stage_start("benchmark")
        benchmark = run_model_comparison(
            games=int(cfg["benchmark_games"]),
            rule_mode=str(cfg["rule_mode"]),
            device=str(device),
            allow_cpu_fallback=True,
            output_json=compare_json,
            output_md=compare_md,
            checkpoints={
                str(cfg["model_type"]): (
                    latest_checkpoint_path if os.path.exists(latest_checkpoint_path) else checkpoint_path
                )
            },
            max_moves=int(cfg.get("max_moves", 30)),
            num_simulations=int(cfg["num_simulations"]),
        )
        _stage_done(
            "benchmark",
            benchmark_stage,
            matches=len(benchmark.get("matches", {})),
            output=compare_json,
        )
        logger.log_benchmark(benchmark)
        summary["benchmark"] = benchmark
        summary["promotion"] = {"status": "skipped", "promote": bool(cfg.get("promote", False))}
        if bool(cfg.get("promote", False)):
            promotion_stage = _stage_start("promotion")
            match_key = f"{cfg['model_type']}_vs_random"
            candidate = latest_checkpoint_path if os.path.exists(latest_checkpoint_path) else checkpoint_path
            if match_key in benchmark.get("matches", {}) and os.path.exists(candidate):
                summary["promotion"] = promote_checkpoint_if_eligible(
                    candidate_path=candidate,
                    best_path=os.path.join(str(cfg["checkpoint_dir"]), "best_advanced.pt"),
                    benchmark_summary=benchmark,
                    match_key=match_key,
                    threshold=float(cfg.get("promotion_threshold", 0.55)),
                    min_games=int(cfg.get("promotion_min_games", 20)),
                )
            else:
                summary["promotion"] = {
                    "status": "skipped_no_matching_benchmark",
                    "promote": True,
                    "match_key": match_key,
                    "candidate_path": candidate,
                }
            _stage_done(
                "promotion",
                promotion_stage,
                promoted=summary["promotion"].get("promoted", False),
                provisional=summary["promotion"].get("provisional_best", False),
            )
        else:
            progress_print("SKIP stage=promotion promote=False", "experiment")
        summary["status"] = "completed"
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = str(exc)
        _write_summary(summary, summary_path)
        progress_print(f"FAILED experiment={experiment_name} error={exc}", "experiment")
        raise

    _write_summary(summary, summary_path)
    progress_print(f"DONE stage=summary path={summary_path}", "experiment")
    from tools.generate_experiment_report import generate_experiment_report

    report_stage = _stage_start("report")
    generate_experiment_report(summary_path, report_path)
    generate_experiment_report(summary_path, os.path.join("docs", "experiment_report_latest.md"))
    _stage_done("report", report_stage, path=report_path)
    progress_print(
        f"DONE experiment={experiment_name} elapsed={format_seconds(time.perf_counter() - experiment_start)} "
        f"summary={summary_path}",
        "experiment",
    )
    return summary


__all__ = ["run_experiment"]
