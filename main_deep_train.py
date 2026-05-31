"""Command-line deep learning training smoke pipeline."""

from __future__ import annotations

import argparse

from train.deep_training_pipeline import run_deep_training_pipeline


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep AlphaZero-mini training pipeline")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--model-type", choices=["cnn", "resnet", "advanced"], default="advanced")
    parser.add_argument("--rule-mode", choices=["basic", "forbidden"], default="basic")
    parser.add_argument("--tactical-games", type=int, default=10)
    parser.add_argument("--pretrain-epochs", type=int, default=3)
    parser.add_argument("--selfplay-games", type=int, default=2)
    parser.add_argument("--finetune-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-simulations", type=int, default=50)
    parser.add_argument("--benchmark-games", type=int, default=10)
    parser.add_argument("--use-augmentation", action="store_true")
    parser.add_argument("--use-auxiliary-loss", action="store_true")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--scheduler", choices=["constant", "cosine", "step", "warmup_cosine"], default="constant")
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", default="outputs/deep_training")
    parser.add_argument("--max-moves", type=int, default=30)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    summary = run_deep_training_pipeline(
        {
            "device": args.device,
            "allow_cpu_fallback": args.allow_cpu_fallback,
            "model_type": args.model_type,
            "rule_mode": args.rule_mode,
            "tactical_games": args.tactical_games,
            "pretrain_epochs": args.pretrain_epochs,
            "selfplay_games": args.selfplay_games,
            "finetune_epochs": args.finetune_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "num_simulations": args.num_simulations,
            "benchmark_games": args.benchmark_games,
            "use_augmentation": args.use_augmentation,
            "use_auxiliary_loss": args.use_auxiliary_loss,
            "resume_from": args.resume_from,
            "scheduler": args.scheduler,
            "warmup_epochs": args.warmup_epochs,
            "grad_clip": args.grad_clip,
            "mixed_precision": args.mixed_precision,
            "seed": args.seed,
            "output_dir": args.output_dir,
            "max_moves": args.max_moves,
        }
    )
    print(f"deep training summary: {summary['paths']['summary']}")
    print("note: smoke-scale run, not competition-strength.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
