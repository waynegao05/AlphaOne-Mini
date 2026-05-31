"""Run a configured deep training experiment."""

from __future__ import annotations

import argparse

from train.experiment_runner import run_experiment


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AlphaZero-mini experiment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment-name")
    parser.add_argument("--device")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--model-type", choices=["cnn", "resnet", "advanced"])
    parser.add_argument("--resume-from")
    parser.add_argument("--rule-mode", choices=["basic", "forbidden"])
    parser.add_argument("--tactical-games", type=int)
    parser.add_argument("--pretrain-epochs", type=int)
    parser.add_argument("--selfplay-games", type=int)
    parser.add_argument("--finetune-epochs", type=int)
    parser.add_argument("--benchmark-games", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--num-simulations", type=int)
    parser.add_argument("--max-moves", type=int)
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--log-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--use-augmentation", action="store_true")
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--use-auxiliary-loss", action="store_true")
    parser.add_argument("--no-auxiliary-loss", action="store_true")
    parser.add_argument("--scheduler", choices=["constant", "cosine", "step", "warmup_cosine"])
    parser.add_argument("--warmup-epochs", type=int)
    parser.add_argument("--grad-clip", type=float)
    parser.add_argument("--mixed-precision", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    overrides = {
        "experiment_name": args.experiment_name,
        "device": args.device,
        "model_type": args.model_type,
        "resume_from": args.resume_from,
        "rule_mode": args.rule_mode,
        "tactical_games": args.tactical_games,
        "pretrain_epochs": args.pretrain_epochs,
        "selfplay_games": args.selfplay_games,
        "finetune_epochs": args.finetune_epochs,
        "benchmark_games": args.benchmark_games,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_simulations": args.num_simulations,
        "max_moves": args.max_moves,
        "checkpoint_dir": args.checkpoint_dir,
        "log_dir": args.log_dir,
        "output_dir": args.output_dir,
        "scheduler": args.scheduler,
        "warmup_epochs": args.warmup_epochs,
        "grad_clip": args.grad_clip,
        "mixed_precision": True if args.mixed_precision else None,
    }
    if args.allow_cpu_fallback:
        overrides["allow_cpu_fallback"] = True
    if args.use_augmentation:
        overrides["use_augmentation"] = True
    if args.no_augmentation:
        overrides["use_augmentation"] = False
    if args.use_auxiliary_loss:
        overrides["use_auxiliary_loss"] = True
    if args.no_auxiliary_loss:
        overrides["use_auxiliary_loss"] = False
    summary = run_experiment(args.config, overrides)
    print(f"experiment status: {summary['status']}")
    print(f"summary: {summary['paths']['summary']}")
    print(f"report: {summary['paths']['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
