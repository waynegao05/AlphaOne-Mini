"""AlphaZero mini 的自博弈数据生成。

每盘对局产出若干个 :class:`SelfPlaySample`，用于后续 PolicyValueNet 训练：

- ``state`` : 形如 ``(4, 15, 15)`` 的 float32 编码张量(由
  :func:`game.encoder.encode_board` 给出，已经从该步当前玩家视角编码)。
- ``pi``    : 形如 ``(225,)`` 的 float32 概率分布(MCTS 在该步输出的搜索概率)。
- ``z``     : 标量胜负值 ``+1.0`` / ``-1.0`` / ``0.0``，**严格按该步的
  ``current_player`` 视角** 计算(详见 :meth:`SelfPlayGame.assign_values`)。

对外接口：
- :class:`SelfPlayGame.play_game`  : 跑一盘，返回样本列表。
- :class:`SelfPlayGame.play_games` : 跑多盘，返回拼接后的样本列表。

注：
- 本模块只生成数据，不写训练循环、不做 loss / 反向传播。
- 不在 MCTS 内部加 Dirichlet noise；当前阶段保持最小化。
- 终局判断完全依赖 :mod:`game.rules_basic`(基础五连)，不实现禁手 / 三手交换 / 五手 N 打。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from game.board import BLACK, BOARD_SIZE, WHITE, Board
from game.encoder import (
    action_to_index,
    encode_board,
    index_to_action,
    legal_moves_to_mask,
)
from game.rules_basic import _find_any_winner, check_winner, is_game_over


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class TrajectoryStep:
    """对局过程中保存的临时一步信息(尚未填入 z)。"""

    state: np.ndarray            # (4, 15, 15) float32
    pi: np.ndarray               # (225,) float32
    current_player: int          # 该步即将落子的颜色：BLACK(1) / WHITE(-1)
    action: int                  # 实际选择的 action_index


@dataclass
class SelfPlaySample:
    """最终训练样本。"""

    state: np.ndarray            # (4, 15, 15) float32
    pi: np.ndarray               # (225,) float32
    z: float                     # +1.0 / -1.0 / 0.0


# ---------------------------------------------------------------------------
# 自博弈主类
# ---------------------------------------------------------------------------
class SelfPlayGame:
    """用 MCTS 驱动一方 vs 一方自博弈，产生训练样本。

    Parameters
    ----------
    model : torch.nn.Module | None
        策略价值网络。如果同时提供 ``mcts``，则可以为空。
    mcts : object | None
        已经实例化好的 MCTS 对象(必须支持 ``run(board)->np.ndarray``)。
        为空时会用 ``model`` + 其他参数自行构造一个真实 :class:`mcts.mcts.MCTS`。
    board_size : int
        棋盘边长，默认 15。
    num_simulations : int
        构造内部 MCTS 时使用的模拟次数。
    c_puct : float
        构造内部 MCTS 时的 PUCT 系数。
    temperature : float
        初期(``move_number < temperature_drop_step``)的采样温度，默认 1.0。
    temperature_drop_step : int
        到达该手数后温度退化为 0(确定性 argmax)。
    device : str
        构造内部 MCTS 时使用的设备。
    max_moves : int
        单盘最大手数；达到上限仍未分胜负，按平局处理(防死循环)。
    rng : np.random.Generator | None
        采样 RNG，便于复现。
    """

    def __init__(
        self,
        model=None,
        mcts=None,
        board_size: int = BOARD_SIZE,
        num_simulations: int = 100,
        c_puct: float = 5.0,
        temperature: float = 1.0,
        temperature_drop_step: int = 20,
        device: str = "cpu",
        max_moves: int = BOARD_SIZE * BOARD_SIZE,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if mcts is None:
            if model is None:
                raise ValueError("必须至少提供 model 或 mcts 之一")
            from mcts.mcts import MCTS  # 延迟导入(避免静态测试时强依赖 torch)

            mcts = MCTS(
                model=model,
                board_size=board_size,
                num_simulations=num_simulations,
                c_puct=c_puct,
                device=device,
                temperature=temperature,
            )

        self.model = model
        self.mcts = mcts
        self.board_size = int(board_size)
        self.action_size = self.board_size * self.board_size
        self.temperature = float(temperature)
        self.temperature_drop_step = int(temperature_drop_step)
        self.max_moves = int(max_moves)
        self.rng = rng if rng is not None else np.random.default_rng()

        # 上一盘的状态，便于外部读取 / 测试。
        self.last_winner: Optional[int] = None
        self.last_move_count: int = 0
        self.last_trajectory: List[TrajectoryStep] = []

    # ---- 温度调度 ---------------------------------------------------------
    def get_temperature(self, move_number: int) -> float:
        """前 ``temperature_drop_step`` 手用初始温度，之后退化为 0。"""
        if move_number < self.temperature_drop_step:
            return self.temperature
        return 0.0

    # ---- 动作选取 ---------------------------------------------------------
    def _sanitize_policy(self, pi: np.ndarray, board: Board) -> np.ndarray:
        """Return a legal, non-negative, normalized policy vector."""
        pi_arr = np.asarray(pi, dtype=np.float64)
        if pi_arr.shape != (self.action_size,):
            raise ValueError(
                f"pi shape must be ({self.action_size},), got {pi_arr.shape}"
            )

        pi_arr = np.where(np.isfinite(pi_arr) & (pi_arr > 0.0), pi_arr, 0.0)
        mask = legal_moves_to_mask(board, self.board_size).astype(np.float64)
        pi_arr *= mask

        total = float(pi_arr.sum())
        if total > 0.0:
            pi_arr /= total
        elif mask.sum() > 0.0:
            pi_arr = mask / float(mask.sum())
        else:
            pi_arr = np.zeros(self.action_size, dtype=np.float64)

        return pi_arr.astype(np.float32)

    def _select_action(
        self, pi: np.ndarray, board: Board, temperature: float
    ) -> int:
        """根据概率 ``pi`` 与 ``temperature`` 选 action_index。

        - ``temperature <= 1e-3`` 时取 ``argmax(pi)``。
        - 否则按 ``pi`` 归一化后采样。
        - 安全兜底：选出的 action 必须合法；若不合法则在合法点中按 ``pi`` 取最大。
        """
        if temperature <= 1e-3:
            if pi.sum() <= 0:
                return self._fallback_legal_action(board)
            action = int(np.argmax(pi))
        else:
            total = float(pi.sum())
            if total <= 0:
                return self._fallback_legal_action(board)
            probs = (pi.astype(np.float64) / total)
            # 数值稳健：再做一次归一(防止浮点误差让 sum 略偏)
            probs = probs / probs.sum()
            action = int(self.rng.choice(self.action_size, p=probs))

        x, y = index_to_action(action, self.board_size)
        if board.is_legal_move(x, y):
            return action

        # pi 与合法 mask 不一致时(理论上 MCTS 已经保证)，退化到合法点 argmax(pi)。
        mask = legal_moves_to_mask(board, self.board_size)
        if mask.sum() == 0:
            raise RuntimeError("没有合法动作可下")
        scores = np.where(mask > 0, np.maximum(pi.astype(np.float64), 1e-12), -1.0)
        return int(np.argmax(scores))

    def _fallback_legal_action(self, board: Board) -> int:
        legal = board.get_legal_moves()
        if not legal:
            raise RuntimeError("没有合法动作可下")
        x, y = legal[0]
        return action_to_index(x, y, self.board_size)

    # ---- 主流程 -----------------------------------------------------------
    def play_game(self) -> List[SelfPlaySample]:
        """跑完一盘自博弈，返回所有样本。

        每一步的具体过程：
        1. 检查终局 / 是否到 ``max_moves``；
        2. 记录 ``current_player``，用 :func:`encode_board` 编码当前棋盘；
        3. 设置 MCTS 当步温度，调 ``mcts.run(board)`` 拿到 ``pi``；
        4. 用本类的温度策略选一个合法 action；
        5. 写入 ``trajectory``，在真实棋盘上落子。

        盘终后用 :meth:`assign_values` 把 winner 反传到每一步的 ``z``。
        """
        board = Board()
        trajectory: List[TrajectoryStep] = []
        move_number = 0

        while move_number < self.max_moves:
            if is_game_over(board, board.last_move):
                break

            current_player = board.current_player
            state = encode_board(board, current_player)

            # 设当步温度，让 MCTS 内部的概率聚合也跟随。
            temperature = self.get_temperature(move_number)
            if hasattr(self.mcts, "temperature"):
                self.mcts.temperature = temperature

            pi = self.mcts.run(board)
            pi = self._sanitize_policy(pi, board)

            # 兜底：极端情况下 pi 全 0(根节点是终局之类) -> 跳出。
            if pi.sum() <= 0 and is_game_over(board, board.last_move):
                break

            action = self._select_action(pi, board, temperature)

            trajectory.append(
                TrajectoryStep(
                    state=state,
                    pi=pi.astype(np.float32, copy=True),
                    current_player=current_player,
                    action=int(action),
                )
            )

            x, y = index_to_action(action, self.board_size)
            board.place_stone(x, y)
            move_number += 1

        # 终局结算：先看 last_move，是否形成 5 连；否则全盘扫一次兜底。
        winner = check_winner(board, board.last_move)
        if winner == 0:
            winner = _find_any_winner(board)

        self.last_winner = int(winner)
        self.last_move_count = move_number
        self.last_trajectory = trajectory

        return self.assign_values(trajectory, winner)

    # ---- z 标签 -----------------------------------------------------------
    def assign_values(
        self, trajectory: List[TrajectoryStep], winner: int
    ) -> List[SelfPlaySample]:
        """根据 ``winner`` 把每一步的 ``z`` 填好。

        - winner == 0          : 所有 z = 0.0
        - winner == cur_player : z = +1.0
        - winner == -cur_player: z = -1.0

        ``cur_player`` 是 **该步保存样本时即将落子的玩家**，与该步偶数 / 奇数无关。
        """
        samples: List[SelfPlaySample] = []
        for step in trajectory:
            if winner == 0:
                z = 0.0
            elif winner == step.current_player:
                z = 1.0
            else:
                z = -1.0
            samples.append(SelfPlaySample(state=step.state, pi=step.pi, z=float(z)))
        return samples

    # ---- 多盘 -------------------------------------------------------------
    def play_games(self, num_games: int) -> List[SelfPlaySample]:
        """连续跑 ``num_games`` 盘并把样本拼成一个列表返回。"""
        if num_games <= 0:
            return []
        all_samples: List[SelfPlaySample] = []
        for _ in range(num_games):
            all_samples.extend(self.play_game())
        return all_samples


__all__ = [
    "SelfPlaySample",
    "TrajectoryStep",
    "SelfPlayGame",
]
