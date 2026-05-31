"""轻量策略价值网络(PolicyValueNet)。

输入  : ``[batch_size, input_channels=4, board_size=15, board_size=15]``
输出  :
    - ``policy_logits`` : ``[batch_size, board_size * board_size]`` (未做 softmax)
    - ``value``         : ``[batch_size, 1]`` (经过 ``tanh``，范围 ``[-1, 1]``)

设计注记：
- ``forward`` 返回的是 logits，softmax 留给 MCTS / 训练损失函数处理。
- 不在模型内部做合法动作 mask、不调用 :class:`Board` 或规则模块；
  这些放在外部协同模块中处理。
- 网络容量很小(三层 64 通道 conv)，适合作为 mini AlphaZero 的早期 baseline。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyValueNet(nn.Module):
    """三层卷积主干 + policy/value 双头。"""

    def __init__(
        self,
        board_size: int = 15,
        input_channels: int = 4,
        hidden_channels: int = 64,
    ) -> None:
        super().__init__()
        self.board_size = board_size
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        cells = board_size * board_size

        # ---- 共享主干 ----------------------------------------------------
        self.conv1 = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_channels)

        # ---- policy head -------------------------------------------------
        self.policy_conv = nn.Conv2d(hidden_channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * cells, cells)

        # ---- value head --------------------------------------------------
        self.value_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(cells, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor):
        """``x`` 形状必须为 ``[B, input_channels, board_size, board_size]``。"""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # policy
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = torch.flatten(p, start_dim=1)
        policy_logits = self.policy_fc(p)  # 不做 softmax

        # value
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = torch.flatten(v, start_dim=1)
        v = F.relu(self.value_fc1(v))
        v = self.value_fc2(v)
        value = torch.tanh(v)

        return policy_logits, value


__all__ = ["PolicyValueNet"]
