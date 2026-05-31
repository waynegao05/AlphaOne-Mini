"""MCTS 节点结构。

每个 :class:`MCTSNode` 维护以下统计量：

- ``visit_count`` (N) : 访问次数
- ``total_value`` (W) : 累计价值，从节点 ``current_player`` 视角
- ``mean_value``  (Q) : ``W / N``，N=0 时返回 0
- ``prior_prob``  (P) : 从父节点扩展时由策略网络给出的先验概率
- ``current_player``  : 在该节点轮到谁走

PUCT::

    score = Q + c_puct * P * sqrt(parent_N) / (1 + N)

backup 时每向上一层把 value 取一次反，因为父子节点的 ``current_player``
正好相反。这一点是 AlphaZero 风格 MCTS 的关键。
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Optional, Tuple


class MCTSNode:
    """搜索树节点。

    构造参数：
    - ``parent``         : 父节点，根节点为 ``None``
    - ``prior_prob``     : 从父节点扩展时分给本节点的先验
    - ``action``         : 父节点 -> 本节点采取的动作索引(根节点为 ``None``)
    - ``current_player`` : 在该节点轮到谁走(``BLACK=1`` / ``WHITE=-1``)
    """

    __slots__ = (
        "parent",
        "prior_prob",
        "action",
        "children",
        "visit_count",
        "total_value",
        "current_player",
        "is_expanded",
    )

    def __init__(
        self,
        parent: Optional["MCTSNode"] = None,
        prior_prob: float = 0.0,
        action: Optional[int] = None,
        current_player: Optional[int] = None,
    ) -> None:
        self.parent = parent
        self.prior_prob = float(prior_prob)
        self.action = action
        self.children: Dict[int, "MCTSNode"] = {}
        self.visit_count: int = 0
        self.total_value: float = 0.0
        self.current_player = current_player
        self.is_expanded: bool = False

    # ---- 状态查询 ----------------------------------------------------------
    @property
    def mean_value(self) -> float:
        """``Q = W / N``，未被访问时返回 0。"""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def is_leaf(self) -> bool:
        """未扩展过即视为叶节点。"""
        return not self.is_expanded

    def has_child(self, action: int) -> bool:
        return action in self.children

    def get_child(self, action: int) -> Optional["MCTSNode"]:
        return self.children.get(action)

    # ---- 扩展 --------------------------------------------------------------
    def expand(
        self,
        action_priors: Mapping[int, float],
        child_player: int,
    ) -> None:
        """根据 ``action_priors`` 创建子节点。

        - ``action_priors`` 仅包含合法动作 -> 先验概率(已经做过 mask + 归一化)。
        - ``child_player`` 是子节点的当前玩家(通常是 ``-self.current_player``)。

        重复调用会跳过已存在的 action(不会覆盖统计量)。
        """
        for action, prior in action_priors.items():
            if action in self.children:
                continue
            self.children[action] = MCTSNode(
                parent=self,
                prior_prob=float(prior),
                action=int(action),
                current_player=child_player,
            )
        self.is_expanded = True

    # ---- 选子(PUCT) --------------------------------------------------------
    def select_child(self, c_puct: float) -> Tuple[int, "MCTSNode"]:
        """返回 ``(action, child)``，PUCT 分数最高的子节点。

        若节点没有任何子节点会抛 ``RuntimeError``，调用方应先确保已 ``expand``。
        """
        if not self.children:
            raise RuntimeError("select_child 在未扩展节点上被调用")

        # parent_N 用 self.visit_count；首次选择时 self.visit_count >= 1
        # (因为本节点在它被扩展后必然被一次 backup 计入)。
        # 即便取 0，这里也加 1 以避免所有 U=0 时全平。
        sqrt_parent_n = math.sqrt(max(1, self.visit_count))

        best_score = -float("inf")
        best_action: Optional[int] = None
        best_child: Optional[MCTSNode] = None
        for action, child in self.children.items():
            # child.mean_value is stored from the child's current-player view.
            # The parent is the opponent, so selection needs the negated Q.
            q_from_parent_view = -child.mean_value
            u = c_puct * child.prior_prob * sqrt_parent_n / (1.0 + child.visit_count)
            score = q_from_parent_view + u
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        assert best_action is not None and best_child is not None
        return best_action, best_child

    # ---- 价值更新 ----------------------------------------------------------
    def update(self, value: float) -> None:
        """直接累计本节点的 visit_count 与 total_value(不沿父链)。"""
        self.visit_count += 1
        self.total_value += float(value)

    def backup(self, value: float) -> None:
        """从本节点开始沿父链反传 ``value``，每层取相反数。

        约定：``value`` 是从 **本节点** ``current_player`` 视角看的胜率
        (1=赢、-1=输、0=持平)。父节点的 ``current_player`` 与本节点相反，
        所以反传到父节点时取负，再向上又取负，依此类推。
        """
        node: Optional["MCTSNode"] = self
        v = float(value)
        while node is not None:
            node.update(v)
            v = -v
            node = node.parent


__all__ = ["MCTSNode"]
