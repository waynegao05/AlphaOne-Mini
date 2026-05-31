"""MCTS 主循环(AlphaZero 风格)。

每次模拟的标准流程：

1. **Selection**     : 从 root 沿 PUCT 最高的子节点向下，直到叶节点或终局。
2. **Expansion**     : 叶节点不是终局时，调 ``model`` 得到 logits + value，
                       softmax 后用 ``legal_moves_to_mask`` 过滤、再归一化得到
                       ``priors``，并在叶节点上扩展子节点。
3. **Terminal**      : 叶节点是终局时，跳过模型，根据胜负 / 平局直接计算 value。
4. **Backup**        : 从叶节点沿父链回传 value，每向上一层取一次相反数。

完成 ``num_simulations`` 次后，按根节点子节点的访问次数(配合 ``temperature``)
得到 shape=``[225]`` 的动作概率分布(非法动作严格为 0)。

注意：
- 本模块只做"搜索"。``board`` 由调用方传入，不会被修改(内部统一用 ``board.copy()``)。
- 不在 MCTS 内部做 Dirichlet noise / 训练样本生成 / 自博弈，那些留给后续阶段。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

import torch
import torch.nn.functional as F

from game.board import BLACK, BOARD_SIZE, WHITE, Board
from game.encoder import (
    action_to_index,
    encode_board,
    index_to_action,
    legal_moves_to_mask,
)
from game.rules_basic import _find_any_winner, check_winner, is_game_over

from .node import MCTSNode


class MCTS:
    """AlphaZero 风格 MCTS。

    Parameters
    ----------
    model : torch.nn.Module
        策略价值网络，接受 ``[B, C, H, W]``，返回 ``(policy_logits, value)``。
    board_size : int
        棋盘边长，默认 15。
    num_simulations : int
        每次 ``run`` 执行多少次模拟，默认 100。
    c_puct : float
        PUCT 探索系数，默认 5.0。
    device : str
        模型运行设备，默认 ``"cpu"``。
    temperature : float
        默认 ``run`` 输出概率的温度，``select_action`` 默认也用它。
    """

    def __init__(
        self,
        model,
        board_size: int = BOARD_SIZE,
        num_simulations: int = 100,
        c_puct: float = 5.0,
        device: str = "cpu",
        temperature: float = 1.0,
        use_candidate_moves: bool = False,
        candidate_radius: int = 2,
        candidate_max_candidates: Optional[int] = None,
    ) -> None:
        self.model = model
        self.board_size = board_size
        self.num_simulations = num_simulations
        self.c_puct = float(c_puct)
        self.device = device
        self.temperature = float(temperature)
        self.action_size = board_size * board_size
        self.use_candidate_moves = bool(use_candidate_moves)
        self.candidate_radius = int(candidate_radius)
        self.candidate_max_candidates = candidate_max_candidates
        self.model.to(self.device)
        # 上一轮 ``run`` 用过的根节点(便于测试 / 调试，不参与搜索复用)。
        self._last_root: Optional[MCTSNode] = None

    # ---- Board helpers -----------------------------------------------------
    def clone_board(self, board: Board) -> Board:
        """统一通过 ``Board.copy()`` 创建搜索副本。"""
        return board.copy()

    def apply_action(self, board: Board, action: int) -> None:
        """把 action_index 还原成 ``(x, y)`` 后调用 ``board.place_stone``。

        ``Board.place_stone`` 内部会自动切换 ``current_player``，
        MCTS 不需要再重复切换。
        """
        x, y = index_to_action(action, self.board_size)
        board.place_stone(x, y)

    # ---- Leaf evaluation ---------------------------------------------------
    def evaluate_leaf(self, board: Board) -> Tuple[np.ndarray, float]:
        """编码 + 推理，得到合法动作 mask 后归一化的 ``priors`` 与标量 ``value``。

        - ``policy_logits`` 在这里做 softmax；softmax 不放在网络内部以便训练。
        - 非法动作位置的概率被强制置 0；剩余概率重新归一化。
        - ``value`` 是从 ``board.current_player`` 视角的胜率估计。
        """
        planes = encode_board(board)  # (4, 15, 15) float32
        x = torch.from_numpy(planes).unsqueeze(0).to(self.device)

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                logits, value = self.model(x)
                probs = F.softmax(logits, dim=1)
                probs_np = probs.detach().cpu().numpy()[0].astype(np.float64)
                value_scalar = float(value.detach().cpu().numpy()[0, 0])
        finally:
            if was_training:
                self.model.train()

        mask = legal_moves_to_mask(board, self.board_size).astype(np.float64)
        mask = self._candidate_action_mask(board, mask)
        masked = probs_np * mask
        total = masked.sum()
        if total > 0:
            priors = masked / total
        elif mask.sum() > 0:
            # 模型给合法点全 0 概率的极端情况：退化为合法动作上的均匀分布。
            priors = mask / mask.sum()
        else:
            priors = np.zeros_like(mask)

        return priors.astype(np.float32), value_scalar

    def _candidate_action_mask(self, board: Board, legal_mask: np.ndarray) -> np.ndarray:
        """Return a candidate-action mask, falling back to legal actions if empty."""
        if not self.use_candidate_moves:
            return legal_mask
        from engine.candidate_moves import generate_candidate_moves
        from engine.threats import (
            find_immediate_blocking_moves,
            find_immediate_winning_moves,
        )

        legal_actions = {int(idx) for idx in np.flatnonzero(legal_mask > 0)}
        candidate_actions = set(
            generate_candidate_moves(
                board,
                radius=self.candidate_radius,
                max_candidates=self.candidate_max_candidates,
            )
        )
        current = int(board.current_player)
        candidate_actions.update(
            find_immediate_winning_moves(board, current, rule_mode="basic")
        )
        candidate_actions.update(
            find_immediate_blocking_moves(board, current, rule_mode="basic")
        )
        candidate_actions &= legal_actions
        if not candidate_actions:
            return legal_mask

        mask = np.zeros_like(legal_mask, dtype=np.float64)
        for action in candidate_actions:
            mask[int(action)] = 1.0
        return legal_mask * mask

    # ---- Terminal value ----------------------------------------------------
    def get_terminal_value(self, board: Board, current_player: int) -> float:
        """从 ``current_player`` 视角看终局价值。

        - 黑胜 + 当前玩家是黑 -> +1
        - 黑胜 + 当前玩家是白 -> -1
        - 白胜 + 当前玩家是白 -> +1
        - 白胜 + 当前玩家是黑 -> -1
        - 平局 / 未分胜负 -> 0

        ``check_winner`` 只通过 ``last_move`` 判断；为兜底真实胜负，
        当其返回 0 时再做一次全盘扫描。
        """
        winner = check_winner(board, board.last_move)
        if winner == 0:
            winner = _find_any_winner(board)
        if winner == 0:
            return 0.0
        return 1.0 if winner == current_player else -1.0

    # ---- Expansion helper --------------------------------------------------
    def expand_node(
        self, node: MCTSNode, board: Board, priors: np.ndarray
    ) -> None:
        """用 ``priors`` (size=action_size) 在 ``node`` 上扩展合法动作。

        - 只为 ``priors > 0`` 的合法动作建立子节点(非法动作的 prior 已被
          ``evaluate_leaf`` 置 0)。
        - 子节点的 ``current_player`` 与本节点相反。
        """
        action_priors: Dict[int, float] = {}
        for action_idx in range(self.action_size):
            p = float(priors[action_idx])
            if p > 0.0:
                action_priors[action_idx] = p
        child_player = -node.current_player if node.current_player is not None else None
        node.expand(action_priors, child_player)

    # ---- 单次模拟 ---------------------------------------------------------
    def _simulate(self, root: MCTSNode, original_board: Board) -> None:
        sim_board = self.clone_board(original_board)
        node = root

        # Selection: 沿 PUCT 最高路径走，直到叶节点或终局。
        while not node.is_leaf():
            if is_game_over(sim_board, sim_board.last_move):
                break
            action, child = node.select_child(self.c_puct)
            self.apply_action(sim_board, action)
            node = child

        # 此刻 node 是 leaf 或终局。
        if is_game_over(sim_board, sim_board.last_move):
            value = self.get_terminal_value(sim_board, node.current_player)
        else:
            priors, value = self.evaluate_leaf(sim_board)
            self.expand_node(node, sim_board, priors)

        node.backup(value)

    # ---- 主入口 -----------------------------------------------------------
    def run(self, board: Board) -> np.ndarray:
        """从 ``board`` 当前局面开始搜索，返回 shape=``[action_size]`` 的概率。

        - 返回的概率严格满足：非法动作=0，且(对非终局局面)概率和=1。
        - 终局局面或没有可扩展子节点时，返回全零概率。
        """
        root = MCTSNode(current_player=board.current_player)
        self._last_root = root

        for _ in range(self.num_simulations):
            self._simulate(root, board)

        return self.get_action_probs(root, temperature=self.temperature)

    # ---- 概率聚合 ---------------------------------------------------------
    def get_action_probs(
        self, root: MCTSNode, temperature: float = 1.0
    ) -> np.ndarray:
        """把根节点子节点的访问次数转成动作概率向量。"""
        probs = np.zeros(self.action_size, dtype=np.float32)
        if not root.children:
            # 终局或未扩展：没有可选动作。
            return probs

        visit_counts = np.zeros(self.action_size, dtype=np.float64)
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count

        total_visits = visit_counts.sum()
        if total_visits <= 0:
            # 全部子节点未被访问(极少发生)：在已扩展的合法动作上做均匀分布。
            for action in root.children:
                probs[action] = 1.0
            s = probs.sum()
            if s > 0:
                probs /= s
            return probs

        # 温度接近 0 -> 一手 deterministic
        if temperature is None or temperature <= 1e-3:
            best_action = int(np.argmax(visit_counts))
            probs[best_action] = 1.0
            return probs

        if abs(temperature - 1.0) < 1e-6:
            weights = visit_counts.copy()
        else:
            weights = np.power(visit_counts, 1.0 / float(temperature))

        weights_sum = weights.sum()
        if weights_sum <= 0:
            best_action = int(np.argmax(visit_counts))
            probs[best_action] = 1.0
            return probs

        probs = (weights / weights_sum).astype(np.float32)
        return probs

    # ---- 选动作 -----------------------------------------------------------
    def select_action(
        self,
        board: Board,
        temperature: Optional[float] = None,
        deterministic: bool = False,
    ) -> int:
        """跑一次 ``run``，根据概率选一个 action_index 返回。

        - ``deterministic=True`` 或 ``temperature`` 接近 0 时取概率最大的合法动作。
        - 其余情况按概率采样。
        """
        if temperature is None:
            temperature = self.temperature

        probs = self.run(board)

        if probs.sum() <= 0:
            # 终局或没扩展：退化为第一个合法落子，避免崩溃。
            legal = board.get_legal_moves()
            if not legal:
                raise RuntimeError("没有可用合法动作，无法 select_action")
            x, y = legal[0]
            return action_to_index(x, y, self.board_size)

        if deterministic or temperature <= 1e-3:
            return int(np.argmax(probs))

        # 数值稳健：归一化一次再采样
        probs_norm = probs.astype(np.float64)
        probs_norm = probs_norm / probs_norm.sum()
        return int(np.random.choice(self.action_size, p=probs_norm))


__all__ = ["MCTS"]
