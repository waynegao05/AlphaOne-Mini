"""Command-line deep benchmark entry point."""

from __future__ import annotations

import argparse

from evaluate.deep_benchmark import DEFAULT_DEEP_BENCHMARK_OUTPUT, run_deep_benchmark
from train.config import load_config, merge_overrides


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep model benchmark smoke test")
    parser.add_argument("--config")
    parser.add_argument("--games", type=int)
    parser.add_argument("--device")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--num-simulations", type=int)
    parser.add_argument("--rule-mode", choices=["basic", "forbidden"])
    parser.add_argument("--cnn-checkpoint")
    parser.add_argument("--resnet-checkpoint")
    parser.add_argument("--advanced-checkpoint")
    parser.add_argument("--curriculum-advanced-checkpoint")
    parser.add_argument("--mistake-tuned-checkpoint")
    parser.add_argument("--mistake-v2-checkpoint")
    parser.add_argument("--mistake-v3-checkpoint")
    parser.add_argument("--hybrid-survival-checkpoint")
    parser.add_argument("--hybrid-survival-v2-checkpoint")
    parser.add_argument("--hybrid-survival-v3-checkpoint")
    parser.add_argument("--neural-guarded", action="store_true")
    parser.add_argument(
        "--fallback-mode",
        choices=["off", "conservative", "normal", "aggressive"],
        default=None,
    )
    parser.add_argument("--tactical-restoration-v1-checkpoint")
    parser.add_argument("--tactical-restoration-curriculum-checkpoint")
    parser.add_argument(
        "--pretrained-advanced-checkpoint",
    )
    parser.add_argument("--matchups", nargs="+")
    parser.add_argument("--output")
    parser.add_argument("--output-md")
    parser.add_argument("--max-moves", type=int)
    parser.add_argument("--seed", type=int)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_config(args.config) if args.config else {}
    overrides = {
        "benchmark_games": args.games,
        "device": args.device,
        "allow_cpu_fallback": True if args.allow_cpu_fallback else None,
        "num_simulations": args.num_simulations,
        "rule_mode": args.rule_mode,
        "cnn_checkpoint": args.cnn_checkpoint,
        "resnet_checkpoint": args.resnet_checkpoint,
        "advanced_checkpoint": args.advanced_checkpoint,
        "curriculum_advanced_checkpoint": args.curriculum_advanced_checkpoint,
        "mistake_tuned_checkpoint": args.mistake_tuned_checkpoint,
        "mistake_v2_checkpoint": args.mistake_v2_checkpoint,
        "mistake_v3_checkpoint": args.mistake_v3_checkpoint,
        "hybrid_survival_checkpoint": args.hybrid_survival_checkpoint,
        "hybrid_survival_v2_checkpoint": args.hybrid_survival_v2_checkpoint,
        "hybrid_survival_v3_checkpoint": args.hybrid_survival_v3_checkpoint,
        "neural_guarded": True if args.neural_guarded else None,
        "fallback_mode": args.fallback_mode,
        "tactical_restoration_v1_checkpoint": args.tactical_restoration_v1_checkpoint,
        "tactical_restoration_curriculum_checkpoint": args.tactical_restoration_curriculum_checkpoint,
        "pretrained_advanced_checkpoint": args.pretrained_advanced_checkpoint,
        "matchups": args.matchups,
        "output": args.output,
        "output_md": args.output_md,
        "max_moves": args.max_moves,
        "seed": args.seed,
    }
    config = merge_overrides(config, overrides)
    games = int(config.get("benchmark_games", 20))
    device = str(config.get("device", "cuda"))
    output = str(config.get("output", DEFAULT_DEEP_BENCHMARK_OUTPUT))
    summary = run_deep_benchmark(
        games=games,
        device=device,
        allow_cpu_fallback=bool(config.get("allow_cpu_fallback", False)),
        num_simulations=int(config.get("num_simulations", 50)),
        rule_mode=str(config.get("rule_mode", "basic")),
        output=output,
        output_md=config.get("output_md"),
        checkpoints={
            "cnn": str(config.get("cnn_checkpoint", "outputs/checkpoints/latest.pt")),
            "resnet": str(config.get("resnet_checkpoint", "outputs/checkpoints/latest_resnet.pt")),
            "advanced": str(config.get("advanced_checkpoint", "outputs/checkpoints/latest_advanced.pt")),
            "latest_advanced": str(config.get("advanced_checkpoint", "outputs/checkpoints/latest_advanced.pt")),
            "curriculum": str(config.get("curriculum_advanced_checkpoint", "outputs/checkpoints/curriculum_advanced.pt")),
            "mistake_tuned": str(config.get("mistake_tuned_checkpoint", "outputs/checkpoints/latest_advanced_mistake_tuned.pt")),
            "mistake_v2": str(config.get("mistake_v2_checkpoint", "outputs/checkpoints/latest_advanced_mistake_v2_from_latest.pt")),
            "mistake_v3": str(config.get("mistake_v3_checkpoint", "outputs/checkpoints/latest_advanced_mistake_v3_teacher_balanced.pt")),
            "hybrid_survival": str(config.get("hybrid_survival_checkpoint", "outputs/checkpoints/latest_advanced_hybrid_survival_from_v2.pt")),
            "hybrid_survival_v2": str(config.get("hybrid_survival_v2_checkpoint", "outputs/checkpoints/latest_advanced_hybrid_survival_v2_from_v2.pt")),
            "hybrid_survival_v3": str(config.get("hybrid_survival_v3_checkpoint", "outputs/checkpoints/latest_advanced_hybrid_survival_v3_forced_block.pt")),
            "tactical_restoration_v1": str(config.get("tactical_restoration_v1_checkpoint", "outputs/checkpoints/latest_advanced_tactical_restoration_from_v1.pt")),
            "tactical_restoration_curriculum": str(config.get("tactical_restoration_curriculum_checkpoint", "outputs/checkpoints/latest_advanced_tactical_restoration_from_curriculum.pt")),
            "pretrained_advanced": str(config.get("pretrained_advanced_checkpoint", "outputs/checkpoints/pretrained_advanced.pt")),
        },
        matchups=config.get("matchups"),
        max_moves=int(config.get("max_moves", 80)),
        seed=int(config.get("seed", 2026)),
        neural_guarded_fallback_mode=str(config.get("fallback_mode", "normal")),
    )
    print(f"deep benchmark saved: {output}")
    print(f"matches: {', '.join(summary['matches'].keys())}")
    if summary["skipped_checkpoints"]:
        print(f"skipped checkpoints: {len(summary['skipped_checkpoints'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
