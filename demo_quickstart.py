"""项目最小演示脚本。

跑一轮极小参数的 ``self-play -> train -> evaluate`` 流水线，验证整条链路通畅。
跑完后提示用户可以继续运行 ``main_play.py`` 进行人机对弈。

CLI::

    python demo_quickstart.py
"""

from __future__ import annotations

import sys

from pipeline.run_pipeline import run_pipeline


BANNER = (
    "============================================================\n"
    "  AlphaZero mini - Quickstart Demo\n"
    "============================================================\n"
    "\n"
    "本脚本会用极小参数跑一遍完整流水线，主要目的是验证项目能跑通：\n"
    "  1) 1 盘自博弈 (5 次 MCTS 模拟 / 步)\n"
    "  2) 1 轮训练 (batch_size=8)\n"
    "  3) 2 盘评估 (vs RandomPlayer)\n"
    "棋力不会很强，主要看流水线是否走通。\n"
)


NEXT_STEPS = (
    "\n============================================================\n"
    "  下一步可以继续:\n"
    "    1) 跑更多自博弈:\n"
    "         python main_selfplay.py --num_games 4 --num_simulations 20\n"
    "    2) 训练更多轮:\n"
    "         python main_train.py --epochs 3 --batch-size 32\n"
    "    3) 评估对随机基线:\n"
    "         python main_evaluate.py --opponent random --games 10\n"
    "    4) 跟 AI 下棋:\n"
    "         python main_play.py --human-color black --num-simulations 20\n"
    "============================================================\n"
)


def main() -> int:
    print(BANNER)

    config = {
        "selfplay_games": 1,
        "num_simulations": 5,
        "max_moves": 225,
        "train_epochs": 1,
        "batch_size": 8,
        "evaluate_games": 2,
        "evaluate_max_moves": 60,
        "opponent": "random",
        "device": "cpu",
        "promote": False,
    }

    summary = run_pipeline(config)

    print()
    print("阶段状态:")
    for stage in ("selfplay", "train", "evaluate", "promote"):
        info = summary.get(stage, {})
        print(f"  {stage:<10s}: {info.get('status', '?')}")

    paths = []
    if summary.get("selfplay", {}).get("status") == "ok":
        paths.append(("自博弈数据", summary["selfplay"].get("output_path")))
    if summary.get("train", {}).get("status") == "ok":
        paths.append(("训练 checkpoint", summary["train"].get("checkpoint_path")))
    if summary.get("evaluate", {}).get("status") == "ok":
        paths.append(("评估结果", summary["evaluate"].get("output_path")))
    paths.append(("流水线摘要", summary.get("summary_path")))

    print("\n关键产物:")
    for name, path in paths:
        if path:
            print(f"  {name}: {path}")

    print(NEXT_STEPS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
