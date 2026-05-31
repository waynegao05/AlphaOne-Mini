"""Player 接口与基础实现。

第七批扩展(在保持第一批 :class:`RandomPlayer.select_move` 接口不变的前提下)：

- 所有玩家都暴露统一的 ``select_action(board) -> int | None`` 与 ``name`` 属性，
  便于 :class:`evaluate.arena.Arena` 一致处理。
- 新增 :class:`MCTSPlayer` 与 :class:`ModelMCTSPlayer`(后者是基于 PolicyValueNet
  的便利封装，并支持 ``from_checkpoint`` 直接加载)。

注意：评估场景(``deterministic=True``)不会改变模型参数；MCTS 内部已经使用
``model.eval()`` + ``torch.no_grad()``。
"""

from __future__ import annotations

import random
from typing import Optional, Tuple

from game.board import Board
from game.encoder import action_to_index, index_to_action


# ---------------------------------------------------------------------------
# 随机玩家
# ---------------------------------------------------------------------------
class RandomPlayer:
    """从所有合法落子中随机均匀选择一手。

    第一批已有的 ``select_move(board) -> (x, y) | None`` 接口保持不变；
    第七批新增 ``select_action(board) -> int | None`` 与 ``name`` 字段。
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        name: Optional[str] = None,
        board_size: int = 15,
    ) -> None:
        self._rng = random.Random(seed)
        self.name = name if name is not None else "RandomPlayer"
        self.board_size = int(board_size)

    def select_move(self, board: Board) -> Optional[Tuple[int, int]]:
        """选择一手落子；棋盘已无空点时返回 ``None``。"""
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            return None
        return self._rng.choice(legal_moves)

    def select_action(self, board: Board) -> Optional[int]:
        """返回 action_index；无合法落子时返回 ``None``。"""
        move = self.select_move(board)
        if move is None:
            return None
        x, y = move
        return action_to_index(x, y, self.board_size)


# ---------------------------------------------------------------------------
# MCTS 玩家
# ---------------------------------------------------------------------------
class MCTSPlayer:
    """基于 :class:`mcts.mcts.MCTS` 的玩家。

    可以传入已经构造好的 ``mcts``，或仅传入 ``model`` + MCTS 参数让本类内部构造。
    评估时一般用 ``deterministic=True``，让选动作直接走访问次数 argmax，结果稳定。
    """

    def __init__(
        self,
        mcts=None,
        model=None,
        num_simulations: int = 50,
        c_puct: float = 5.0,
        device: str = "cpu",
        deterministic: bool = True,
        board_size: int = 15,
        name: Optional[str] = None,
    ) -> None:
        if mcts is None:
            if model is None:
                raise ValueError("必须至少提供 model 或 mcts 之一")
            from mcts.mcts import MCTS  # 延迟导入

            mcts = MCTS(
                model=model,
                board_size=board_size,
                num_simulations=num_simulations,
                c_puct=c_puct,
                device=device,
            )
        self.mcts = mcts
        self.model = model
        self.board_size = int(board_size)
        self.deterministic = bool(deterministic)
        self.name = name if name is not None else "MCTSPlayer"

    def select_action(self, board: Board) -> Optional[int]:
        """返回 action_index。"""
        if not board.get_legal_moves():
            return None
        return int(
            self.mcts.select_action(board, deterministic=self.deterministic)
        )

    def select_move(self, board: Board) -> Optional[Tuple[int, int]]:
        """与第一批 RandomPlayer 风格保持兼容：返回 ``(x, y)`` 元组。"""
        action = self.select_action(board)
        if action is None:
            return None
        return index_to_action(action, self.board_size)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        num_simulations: int = 50,
        c_puct: float = 5.0,
        device: str = "cpu",
        board_size: int = 15,
        deterministic: bool = True,
        name: Optional[str] = None,
    ) -> "MCTSPlayer":
        """从 ``checkpoint_path`` 加载 PolicyValueNet 后构造一个 MCTSPlayer。"""
        from model.checkpoint import load_checkpoint
        from model.policy_value_net import PolicyValueNet

        model = PolicyValueNet(board_size=board_size)
        load_checkpoint(model, checkpoint_path, device=device)
        model.eval()
        return cls(
            model=model,
            num_simulations=num_simulations,
            c_puct=c_puct,
            device=device,
            deterministic=deterministic,
            board_size=board_size,
            name=name,
        )


class ModelMCTSPlayer(MCTSPlayer):
    """显式表达"基于 PolicyValueNet"语义的便利子类。"""

    def __init__(
        self,
        model,
        num_simulations: int = 50,
        c_puct: float = 5.0,
        device: str = "cpu",
        deterministic: bool = True,
        board_size: int = 15,
        name: Optional[str] = None,
    ) -> None:
        # 评估默认不训练，保险起见切到 eval。
        try:
            model.eval()
        except AttributeError:
            pass
        super().__init__(
            mcts=None,
            model=model,
            num_simulations=num_simulations,
            c_puct=c_puct,
            device=device,
            deterministic=deterministic,
            board_size=board_size,
            name=name if name is not None else "ModelMCTSPlayer",
        )


__all__ = ["RandomPlayer", "MCTSPlayer", "ModelMCTSPlayer"]
