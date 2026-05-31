"""端到端流水线入口。

跑一遍 ``self-play -> train -> evaluate -> (可选 promote)``，并把
``pipeline_summary.json`` 写到 ``--base-output-dir`` 下。

CLI::

    python main_pipeline.py --selfplay-games 1 --num-simulations 5 \\
                           --train-epochs 1 --evaluate-games 2 --device cpu

注意：本脚本只跑一次链路，不做循环训练 / Elo / 模型库管理。
"""

from __future__ import annotations

import argparse
import sys

from pipeline.run_pipeline import run_pipeline


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AlphaZero mini 完整流水线 (self-play -> train -> evaluate)"
    )
    parser.add_argument("--base-output-dir", type=str, default="outputs")
    parser.add_argument("--selfplay-games", type=int, default=1)
    parser.add_argument("--num-simulations", type=int, default=10)
    parser.add_argument("--train-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--evaluate-games", type=int, default=2)
    parser.add_argument("--evaluate-max-moves", type=int, default=225)
    parser.add_argument(
        "--opponent",
        type=str,
        choices=["random", "best", "self"],
        default="random",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--skip-selfplay", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    parser.add_argument("--max-moves", type=int, default=225)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = {
        "base_output_dir": args.base_output_dir,
        "selfplay_games": args.selfplay_games,
        "num_simulations": args.num_simulations,
        "max_moves": args.max_moves,
        "train_epochs": args.train_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "evaluate_games": args.evaluate_games,
        "evaluate_max_moves": args.evaluate_max_moves,
        "opponent": args.opponent,
        "device": args.device,
        "threshold": args.threshold,
        "promote": args.promote,
        "skip_selfplay": args.skip_selfplay,
        "skip_train": args.skip_train,
        "skip_evaluate": args.skip_evaluate,
    }

    summary = run_pipeline(config)

    print()
    print("=" * 60)
    print("流水线完成。各阶段状态:")
    for stage in ("selfplay", "train", "evaluate", "promote"):
        info = summary.get(stage, {})
        status = info.get("status", "?")
        extra = ""
        if stage == "selfplay" and status == "ok":
            extra = f" ({info.get('num_samples', 0)} samples)"
        elif stage == "train" and status == "ok":
            extra = f" (final_loss={info.get('final_loss')})"
        elif stage == "evaluate" and status == "ok":
            extra = f" (candidate 胜率 {info.get('candidate_win_rate', 0.0):.3f})"
        elif stage == "promote" and status not in ("ok", "promoted"):
            extra = f" ({info.get('reason', '')})"
        print(f"  {stage:<10s}: {status}{extra}")
    print(f"\n摘要文件: {summary.get('summary_path')}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
