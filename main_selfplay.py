"""自博弈最小入口脚本。

跑 ``num_games`` 盘自博弈，把样本写入 ``outputs/selfplay_data/selfplay_latest.npz``，
并打印盘数 / 样本数 / 黑胜白胜平局统计。

CLI::

    python main_selfplay.py --num_games 2 --num_simulations 20

注意：
- 当前模型完全未训练，棋力很弱、对局会比较随意，目的是验证整条数据流通。
- 不在这里写训练循环、不计算 loss、不更新参数。
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

from game.board import BLACK, WHITE
from model.policy_value_net import PolicyValueNet
from selfplay.replay_buffer import ReplayBuffer
from selfplay.self_play import SelfPlayGame


def run_selfplay(
    num_games: int = 2,
    num_simulations: int = 20,
    temperature: float = 1.0,
    temperature_drop_step: int = 20,
    max_moves: int = 225,
    output_path: str = os.path.join(
        "outputs", "selfplay_data", "selfplay_latest.npz"
    ),
    seed: Optional[int] = None,
) -> dict:
    """跑自博弈、写盘并返回统计结果。"""
    print(
        f"启动自博弈: {num_games} 盘, MCTS {num_simulations} 次/步, "
        f"temperature={temperature}, drop_step={temperature_drop_step}"
    )

    model = PolicyValueNet()
    model.eval()

    game = SelfPlayGame(
        model=model,
        num_simulations=num_simulations,
        temperature=temperature,
        temperature_drop_step=temperature_drop_step,
        max_moves=max_moves,
    )

    buffer = ReplayBuffer(capacity=max(50_000, num_games * max_moves + 1), seed=seed)

    black_wins = 0
    white_wins = 0
    draws = 0

    for i in range(num_games):
        samples = game.play_game()
        winner = game.last_winner
        if winner == BLACK:
            black_wins += 1
            winner_label = "黑胜"
        elif winner == WHITE:
            white_wins += 1
            winner_label = "白胜"
        else:
            draws += 1
            winner_label = "平局"
        buffer.extend(samples)
        print(
            f"  对局 {i + 1}/{num_games}: {len(samples)} 步, "
            f"winner={winner_label}"
        )

    buffer.save(output_path)

    stats = {
        "num_games": num_games,
        "num_samples": len(buffer),
        "black_wins": black_wins,
        "white_wins": white_wins,
        "draws": draws,
        "output_path": os.path.abspath(output_path),
    }

    print()
    print(f"完成: 总盘数 {stats['num_games']}")
    print(f"  黑胜 {stats['black_wins']} | 白胜 {stats['white_wins']} | 平局 {stats['draws']}")
    print(f"  样本总数: {stats['num_samples']}")
    print(f"  保存路径: {stats['output_path']}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="AlphaZero mini self-play 数据生成")
    parser.add_argument("--num_games", type=int, default=2)
    parser.add_argument("--num_simulations", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperature_drop_step", type=int, default=20)
    parser.add_argument("--max_moves", type=int, default=225)
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("outputs", "selfplay_data", "selfplay_latest.npz"),
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    run_selfplay(
        num_games=args.num_games,
        num_simulations=args.num_simulations,
        temperature=args.temperature,
        temperature_drop_step=args.temperature_drop_step,
        max_moves=args.max_moves,
        output_path=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
