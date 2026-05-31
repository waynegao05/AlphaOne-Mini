"""端到端流水线：``self-play -> train -> evaluate -> (可选 promote)``。

设计目标：
- 把第五~七批的脚本黏成一条最小可运行 MVP，便于 ``demo_quickstart`` 与
  ``main_pipeline`` 直接调用。
- 每个阶段都返回一个状态字典(``status / 关键统计 / 落盘路径``)，便于上层
  打印与做集成测试。
- 默认参数极小，能在普通笔记本上几秒内跑完，专门用于流程 smoke。
- 不实现自动 self-play→train→evaluate 大循环；仍然是单次链路。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from evaluate.metrics import should_promote
from game.board import BLACK, WHITE


# ---------------------------------------------------------------------------
# 默认配置：所有参数都尽量小，让流水线 smoke 能 < 几秒跑完
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "base_output_dir": "outputs",
    # ---- self-play ----
    "selfplay_games": 1,
    "num_simulations": 10,
    "temperature": 1.0,
    "temperature_drop_step": 20,
    "max_moves": 225,
    # ---- train ----
    "train_epochs": 1,
    "batch_size": 8,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "grad_clip": 5.0,
    # ---- evaluate ----
    "evaluate_games": 2,
    "evaluate_max_moves": 225,
    "c_puct": 5.0,
    "opponent": "random",
    "threshold": 0.55,
    "promote": False,
    # ---- general ----
    "device": "cpu",
    "skip_selfplay": False,
    "skip_train": False,
    "skip_evaluate": False,
    "verbose": True,
}


def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg)


# ---------------------------------------------------------------------------
# 阶段函数
# ---------------------------------------------------------------------------
def run_selfplay_step(
    *,
    output_path: str,
    num_games: int = 1,
    num_simulations: int = 10,
    temperature: float = 1.0,
    temperature_drop_step: int = 20,
    max_moves: int = 225,
    device: str = "cpu",
    model: Optional[Any] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """跑 ``num_games`` 盘自博弈并把样本写到 ``output_path``。"""
    from model.policy_value_net import PolicyValueNet
    from selfplay.replay_buffer import ReplayBuffer
    from selfplay.self_play import SelfPlayGame

    if model is None:
        model = PolicyValueNet()
    model.eval()

    game = SelfPlayGame(
        model=model,
        num_simulations=num_simulations,
        temperature=temperature,
        temperature_drop_step=temperature_drop_step,
        max_moves=max_moves,
        device=device,
    )

    buffer = ReplayBuffer(capacity=max(50_000, num_games * max_moves + 1))

    black_wins = white_wins = draws = 0
    for i in range(num_games):
        samples = game.play_game()
        winner = game.last_winner
        if winner == BLACK:
            black_wins += 1
        elif winner == WHITE:
            white_wins += 1
        else:
            draws += 1
        buffer.extend(samples)
        _log(
            verbose,
            f"  [self-play] game {i + 1}/{num_games}: "
            f"{len(samples)} samples, winner={winner}",
        )

    buffer.save(output_path)
    return {
        "status": "ok",
        "num_games": num_games,
        "num_samples": len(buffer),
        "black_wins": black_wins,
        "white_wins": white_wins,
        "draws": draws,
        "output_path": os.path.abspath(output_path),
    }


def run_train_step(
    *,
    data_path: str,
    checkpoint_dir: str,
    epochs: int = 1,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    grad_clip: float = 5.0,
    device: str = "cpu",
    model: Optional[Any] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """读自博弈数据训练 PolicyValueNet，写到 ``checkpoint_dir/latest.pt``。"""
    from model.policy_value_net import PolicyValueNet
    from train.train import train_model

    if not os.path.exists(data_path):
        return {
            "status": "skipped",
            "reason": f"自博弈数据文件不存在: {data_path}",
        }
    if model is None:
        model = PolicyValueNet()

    history = train_model(
        model=model,
        data_path=data_path,
        checkpoint_dir=checkpoint_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
        grad_clip=grad_clip,
    )
    final = history[-1] if history else {}
    return {
        "status": "ok",
        "epochs": epochs,
        "history": history,
        "checkpoint_path": os.path.abspath(os.path.join(checkpoint_dir, "latest.pt")),
        "final_loss": final.get("loss"),
        "final_policy_loss": final.get("policy_loss"),
        "final_value_loss": final.get("value_loss"),
    }


def run_evaluate_step(
    *,
    candidate_path: str,
    best_path: str,
    output_path: str,
    opponent: str = "random",
    games: int = 2,
    num_simulations: int = 10,
    c_puct: float = 5.0,
    device: str = "cpu",
    board_size: int = 15,
    max_moves: int = 225,
    verbose: bool = True,
) -> Dict[str, Any]:
    """评估 candidate 模型(对 random / best / self)，结果写到 ``output_path``。"""
    from evaluate.arena import run_match
    from evaluate.players import ModelMCTSPlayer, RandomPlayer
    from model.checkpoint import load_checkpoint
    from model.policy_value_net import PolicyValueNet

    candidate = PolicyValueNet(board_size=board_size)
    if os.path.exists(candidate_path):
        load_checkpoint(candidate, candidate_path, device=device)
        _log(verbose, f"  [evaluate] 已加载 candidate: {candidate_path}")
    else:
        _log(
            verbose,
            f"  [evaluate] candidate 不存在 ({candidate_path})，使用随机初始化模型",
        )
    candidate.eval()

    candidate_player = ModelMCTSPlayer(
        model=candidate,
        num_simulations=num_simulations,
        c_puct=c_puct,
        device=device,
        board_size=board_size,
        name="candidate",
    )

    if opponent == "random":
        opp = RandomPlayer(name="random", board_size=board_size)
    elif opponent == "best":
        if not os.path.exists(best_path):
            return {
                "status": "skipped",
                "reason": f"best checkpoint 不存在: {best_path}",
                "opponent": opponent,
            }
        best_model = PolicyValueNet(board_size=board_size)
        load_checkpoint(best_model, best_path, device=device)
        best_model.eval()
        opp = ModelMCTSPlayer(
            model=best_model,
            num_simulations=num_simulations,
            c_puct=c_puct,
            device=device,
            board_size=board_size,
            name="best",
        )
    elif opponent == "self":
        opp = ModelMCTSPlayer(
            model=candidate,
            num_simulations=num_simulations,
            c_puct=c_puct,
            device=device,
            board_size=board_size,
            name="candidate_self",
        )
    else:
        raise ValueError(f"未知 opponent: {opponent!r}")

    summary, _results = run_match(
        candidate_player,
        opp,
        num_games=games,
        alternate_sides=True,
        board_size=board_size,
        max_moves=max_moves,
    )

    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    return {
        "status": "ok",
        "opponent": opponent,
        "games": games,
        "candidate_win_rate": float(summary.get("player_a_win_rate", 0.0)),
        "draw_rate": float(summary.get("draw_rate", 0.0)),
        "summary": summary,
        "output_path": os.path.abspath(output_path),
    }


def run_promote_step(
    *,
    candidate_path: str,
    best_path: str,
    candidate_win_rate: float,
    threshold: float = 0.55,
    enable: bool = False,
    opponent: str = "random",
    num_games: int = 0,
) -> Dict[str, Any]:
    """晋级判断：必须 ``opponent == "best"`` + ``enable=True`` + 达阈值 + 文件存在。"""
    if opponent != "best":
        return {"status": "skipped", "reason": "opponent != 'best'，不晋级"}
    if not enable:
        return {"status": "skipped", "reason": "promote 标志为 False"}
    if not should_promote(candidate_win_rate, threshold):
        return {
            "status": "rejected",
            "reason": f"候选胜率 {candidate_win_rate:.3f} < 阈值 {threshold:.3f}",
            "candidate_win_rate": float(candidate_win_rate),
            "threshold": float(threshold),
        }
    if not os.path.exists(candidate_path):
        return {
            "status": "rejected",
            "reason": f"candidate ckpt 不存在: {candidate_path}",
        }

    from model.checkpoint import load_checkpoint, save_checkpoint
    from model.policy_value_net import PolicyValueNet

    model = PolicyValueNet()
    load_checkpoint(model, candidate_path)
    metadata = {
        "promoted_from": os.path.abspath(candidate_path),
        "candidate_win_rate": float(candidate_win_rate),
        "threshold": float(threshold),
        "num_games": int(num_games),
    }
    save_checkpoint(model, best_path, metadata=metadata)
    return {
        "status": "promoted",
        "best_path": os.path.abspath(best_path),
        "candidate_win_rate": float(candidate_win_rate),
        "threshold": float(threshold),
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# 摘要落盘
# ---------------------------------------------------------------------------
def save_pipeline_summary(summary: Dict[str, Any], path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def run_pipeline(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """跑完整流水线，返回所有阶段的统计字典(并在 base_output_dir 下落盘)。"""
    cfg: Dict[str, Any] = {**DEFAULT_CONFIG, **(config or {})}
    verbose = bool(cfg["verbose"])

    base_dir = str(cfg["base_output_dir"])
    os.makedirs(base_dir, exist_ok=True)

    selfplay_path = os.path.join(base_dir, "selfplay_data", "selfplay_latest.npz")
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    candidate_path = os.path.join(checkpoint_dir, "latest.pt")
    best_path = os.path.join(checkpoint_dir, "best.pt")
    eval_path = os.path.join(base_dir, "evaluation", "eval_latest.json")
    summary_path = os.path.join(base_dir, "pipeline_summary.json")

    summary: Dict[str, Any] = {
        "config": dict(cfg),
        "selfplay": {"status": "skipped", "reason": "skip_selfplay=True"},
        "train": {"status": "skipped", "reason": "skip_train=True"},
        "evaluate": {"status": "skipped", "reason": "skip_evaluate=True"},
        "promote": {"status": "skipped", "reason": "未触发"},
    }

    # 1) self-play
    if not cfg["skip_selfplay"]:
        _log(verbose, "[1/4] self-play 开始")
        summary["selfplay"] = run_selfplay_step(
            output_path=selfplay_path,
            num_games=cfg["selfplay_games"],
            num_simulations=cfg["num_simulations"],
            temperature=cfg["temperature"],
            temperature_drop_step=cfg["temperature_drop_step"],
            max_moves=cfg["max_moves"],
            device=cfg["device"],
            verbose=verbose,
        )
        _log(verbose, f"      -> {summary['selfplay'].get('num_samples', 0)} 个样本")

    # 2) train
    if not cfg["skip_train"]:
        _log(verbose, "[2/4] train 开始")
        summary["train"] = run_train_step(
            data_path=selfplay_path,
            checkpoint_dir=checkpoint_dir,
            epochs=cfg["train_epochs"],
            batch_size=cfg["batch_size"],
            learning_rate=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
            grad_clip=cfg["grad_clip"],
            device=cfg["device"],
            verbose=verbose,
        )
        _log(
            verbose,
            f"      -> final_loss={summary['train'].get('final_loss')}",
        )

    # 3) evaluate
    if not cfg["skip_evaluate"]:
        _log(verbose, "[3/4] evaluate 开始")
        summary["evaluate"] = run_evaluate_step(
            candidate_path=candidate_path,
            best_path=best_path,
            output_path=eval_path,
            opponent=cfg["opponent"],
            games=cfg["evaluate_games"],
            num_simulations=cfg["num_simulations"],
            c_puct=cfg["c_puct"],
            device=cfg["device"],
            max_moves=cfg["evaluate_max_moves"],
            verbose=verbose,
        )
        _log(
            verbose,
            f"      -> candidate 胜率 = "
            f"{summary['evaluate'].get('candidate_win_rate', 0.0):.3f}",
        )

    # 4) promote (仅 opponent == best 且达阈值且 promote=True)
    candidate_win_rate = (
        summary["evaluate"].get("candidate_win_rate", 0.0)
        if summary["evaluate"].get("status") == "ok"
        else 0.0
    )
    summary["promote"] = run_promote_step(
        candidate_path=candidate_path,
        best_path=best_path,
        candidate_win_rate=candidate_win_rate,
        threshold=cfg["threshold"],
        enable=bool(cfg["promote"]),
        opponent=cfg["opponent"],
        num_games=cfg["evaluate_games"],
    )
    if summary["promote"]["status"] == "promoted":
        _log(verbose, f"[4/4] 已晋级 best.pt: {summary['promote']['best_path']}")
    else:
        _log(verbose, f"[4/4] 未晋级: {summary['promote'].get('reason', '')}")

    summary["summary_path"] = os.path.abspath(summary_path)
    # 落盘 pipeline summary
    save_pipeline_summary(summary, summary_path)
    _log(verbose, f"\npipeline_summary -> {summary['summary_path']}")
    return summary


__all__ = [
    "DEFAULT_CONFIG",
    "run_pipeline",
    "run_selfplay_step",
    "run_train_step",
    "run_evaluate_step",
    "run_promote_step",
    "save_pipeline_summary",
]
