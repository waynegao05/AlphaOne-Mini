"""模型评估入口。

支持三种 opponent：

- ``random`` : candidate vs :class:`evaluate.players.RandomPlayer`(基线 smoke)。
- ``best``   : candidate vs ``best.pt``。仅在该模式下，配合 ``--promote`` 时才允许把
               candidate 晋级为新的 ``best.pt``。
- ``self``   : candidate vs candidate(共享同一份模型，用于流水线 smoke test)。

CLI::

    python main_evaluate.py --opponent random --games 10 --num-simulations 20
    python main_evaluate.py --opponent best --games 20 --threshold 0.55 --promote

注意：本脚本只评估、不训练、不重新生成自博弈数据。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Callable, Dict, Optional

from evaluate.arena import run_match
from evaluate.metrics import format_match_summary, should_promote
from evaluate.players import RandomPlayer


DEFAULT_CANDIDATE = os.path.join("outputs", "checkpoints", "latest.pt")
DEFAULT_BEST = os.path.join("outputs", "checkpoints", "best.pt")
DEFAULT_OUTPUT = os.path.join("outputs", "evaluation", "eval_latest.json")


def _build_candidate(path: str, device: str) -> "PolicyValueNet":
    from model.checkpoint import load_checkpoint
    from model.policy_value_net import PolicyValueNet

    model = PolicyValueNet()
    if os.path.exists(path):
        load_checkpoint(model, path, device=device)
        print(f"已加载 candidate: {path}")
    else:
        print(
            f"提示: candidate checkpoint 不存在 ({path})，"
            "改用随机初始化模型 smoke test。"
        )
    model.eval()
    return model


def _build_opponent(
    opponent: str,
    candidate: "PolicyValueNet",
    best_path: str,
    num_simulations: int,
    c_puct: float,
    device: str,
):
    from evaluate.players import ModelMCTSPlayer

    if opponent == "random":
        return RandomPlayer(name="random")

    if opponent == "best":
        from model.checkpoint import load_checkpoint
        from model.policy_value_net import PolicyValueNet

        if not os.path.exists(best_path):
            print(
                f"错误: best checkpoint 不存在 ({best_path})。\n"
                "可以改用 --opponent random，或先把 candidate 复制为 best:\n"
                f"  cp {DEFAULT_CANDIDATE} {best_path}"
            )
            sys.exit(1)
        best_model = PolicyValueNet()
        load_checkpoint(best_model, best_path, device=device)
        best_model.eval()
        return ModelMCTSPlayer(
            model=best_model,
            num_simulations=num_simulations,
            c_puct=c_puct,
            device=device,
            name="best",
        )

    if opponent == "self":
        return ModelMCTSPlayer(
            model=candidate,
            num_simulations=num_simulations,
            c_puct=c_puct,
            device=device,
            name="candidate_self",
        )

    raise ValueError(f"未知的 --opponent: {opponent!r}")


def _save_summary(summary: Dict[str, Any], path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)


def _maybe_promote(
    *,
    opponent: str,
    candidate_win_rate: float,
    threshold: float,
    promote_flag: bool,
    candidate_path: str,
    best_path: str,
    candidate_model: "PolicyValueNet",
    num_games: int,
    save_fn: Optional[Callable[..., Any]] = None,
) -> bool:
    """仅在 ``opponent == best`` 且达阈值且传 --promote 时把 candidate 写成 best。"""
    if opponent != "best":
        return False
    if not should_promote(candidate_win_rate, threshold):
        print(
            f"未达晋级阈值: candidate 胜率 {candidate_win_rate:.3f} < {threshold:.3f}"
        )
        return False
    if not promote_flag:
        print(
            f"已达晋级阈值 (胜率 {candidate_win_rate:.3f} >= {threshold:.3f})，"
            "但未传 --promote，未自动替换 best.pt"
        )
        return False
    if not os.path.exists(candidate_path):
        print("WARNING: candidate checkpoint 不存在，无法晋级")
        return False

    metadata = {
        "promoted_from": os.path.abspath(candidate_path),
        "candidate_win_rate": float(candidate_win_rate),
        "threshold": float(threshold),
        "num_games": int(num_games),
    }
    if save_fn is None:
        from model.checkpoint import save_checkpoint

        save_fn = save_checkpoint
    save_fn(candidate_model, best_path, metadata=metadata)
    print(
        f"已晋级: candidate -> best ({best_path})，胜率 "
        f"{candidate_win_rate:.3f} >= {threshold:.3f}"
    )
    return True


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AlphaZero mini 评估 (candidate vs random / best / self)"
    )
    parser.add_argument("--candidate", type=str, default=DEFAULT_CANDIDATE)
    parser.add_argument("--best", type=str, default=DEFAULT_BEST)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--num-simulations", type=int, default=50)
    parser.add_argument("--c-puct", type=float, default=5.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument(
        "--opponent",
        type=str,
        choices=["random", "best", "self"],
        default="random",
    )
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--no-alternate-sides",
        action="store_true",
        help="禁用双方交换执黑(默认开启)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    from evaluate.players import ModelMCTSPlayer

    args = parse_args(argv)

    candidate = _build_candidate(args.candidate, args.device)
    candidate_player = ModelMCTSPlayer(
        model=candidate,
        num_simulations=args.num_simulations,
        c_puct=args.c_puct,
        device=args.device,
        name="candidate",
    )
    opponent_player = _build_opponent(
        opponent=args.opponent,
        candidate=candidate,
        best_path=args.best,
        num_simulations=args.num_simulations,
        c_puct=args.c_puct,
        device=args.device,
    )

    print(
        f"开始评估: candidate vs {opponent_player.name}, "
        f"games={args.games}, sims={args.num_simulations}, "
        f"c_puct={args.c_puct}, device={args.device}"
    )

    summary, _results = run_match(
        candidate_player,
        opponent_player,
        num_games=args.games,
        alternate_sides=not args.no_alternate_sides,
    )

    print()
    print(format_match_summary(summary))

    candidate_win_rate = float(summary.get("player_a_win_rate", 0.0))
    promoted = _maybe_promote(
        opponent=args.opponent,
        candidate_win_rate=candidate_win_rate,
        threshold=args.threshold,
        promote_flag=args.promote,
        candidate_path=args.candidate,
        best_path=args.best,
        candidate_model=candidate,
        num_games=args.games,
    )

    summary["opponent"] = args.opponent
    summary["candidate_path"] = os.path.abspath(args.candidate)
    summary["best_path"] = os.path.abspath(args.best)
    summary["threshold"] = float(args.threshold)
    summary["promoted"] = bool(promoted)

    _save_summary(summary, args.output)
    print(f"\n评估结果已保存: {os.path.abspath(args.output)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
