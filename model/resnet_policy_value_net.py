"""Small ResNet policy-value network baseline."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Two-convolution residual block for 15x15 Gomoku features."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class SmallResNetPolicyValueNet(nn.Module):
    """ResNet baseline with policy and value heads."""

    model_type = "resnet"

    def __init__(
        self,
        board_size: int = 15,
        input_channels: int = 4,
        channels: int = 64,
        blocks: int = 4,
    ) -> None:
        super().__init__()
        self.board_size = int(board_size)
        self.input_channels = int(input_channels)
        self.channels = int(channels)
        self.blocks = int(blocks)
        cells = self.board_size * self.board_size

        self.stem_conv = nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(channels)
        self.backbone = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])

        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * cells, cells)

        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(cells, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.stem_bn(self.stem_conv(x)))
        x = self.backbone(x)

        p = F.relu(self.policy_bn(self.policy_conv(x)))
        policy_logits = self.policy_fc(torch.flatten(p, start_dim=1))

        v = F.relu(self.value_bn(self.value_conv(x)))
        v = F.relu(self.value_fc1(torch.flatten(v, start_dim=1)))
        value = torch.tanh(self.value_fc2(v))
        return policy_logits, value


__all__ = ["ResidualBlock", "SmallResNetPolicyValueNet"]
