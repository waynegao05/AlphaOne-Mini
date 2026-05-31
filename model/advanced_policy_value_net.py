"""Advanced multi-task policy-value network for Gomoku."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet_policy_value_net import ResidualBlock


class SEBlock(nn.Module):
    """Squeeze-and-excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.shape
        pooled = F.adaptive_avg_pool2d(x, 1).view(batch, channels)
        weights = torch.sigmoid(self.fc2(F.relu(self.fc1(pooled)))).view(
            batch, channels, 1, 1
        )
        return x * weights


class AdvancedPolicyValueNet(nn.Module):
    """ResNet + attention + policy/value/threat/forbidden heads."""

    model_type = "advanced"

    def __init__(
        self,
        board_size: int = 15,
        input_channels: int = 4,
        channels: int = 96,
        blocks: int = 6,
        attention_every: int = 2,
        num_threat_channels: int = 12,
        use_tactical_score: bool = True,
    ) -> None:
        super().__init__()
        self.board_size = int(board_size)
        self.input_channels = int(input_channels)
        self.channels = int(channels)
        self.blocks = int(blocks)
        self.attention_every = int(attention_every)
        self.num_threat_channels = int(num_threat_channels)
        self.use_tactical_score = bool(use_tactical_score)
        cells = self.board_size * self.board_size

        self.stem_conv = nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(channels)

        layers: list[nn.Module] = []
        for index in range(blocks):
            layers.append(ResidualBlock(channels))
            if attention_every > 0 and (index + 1) % attention_every == 0:
                layers.append(SEBlock(channels))
        self.backbone = nn.Sequential(*layers)

        self.policy_conv = nn.Conv2d(channels, 4, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(4)
        self.policy_fc = nn.Linear(4 * cells, cells)

        self.value_conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(2)
        self.value_fc1 = nn.Linear(2 * cells, 128)
        self.value_fc2 = nn.Linear(128, 1)

        self.threat_head = nn.Conv2d(channels, num_threat_channels, kernel_size=1)
        self.forbidden_head = nn.Conv2d(channels, 1, kernel_size=1)
        self.tactical_conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.tactical_bn = nn.BatchNorm2d(2)
        self.tactical_fc = nn.Linear(2 * cells, cells)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.stem_bn(self.stem_conv(x)))
        return self.backbone(x)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        features = self._features(x)

        p = F.relu(self.policy_bn(self.policy_conv(features)))
        policy_logits = self.policy_fc(torch.flatten(p, start_dim=1))

        v = F.relu(self.value_bn(self.value_conv(features)))
        v = F.relu(self.value_fc1(torch.flatten(v, start_dim=1)))
        value = torch.tanh(self.value_fc2(v))

        if not return_aux:
            return policy_logits, value

        tactical = F.relu(self.tactical_bn(self.tactical_conv(features)))
        tactical_score = self.tactical_fc(torch.flatten(tactical, start_dim=1))
        return {
            "policy_logits": policy_logits,
            "value": value,
            "threat_logits": self.threat_head(features),
            "forbidden_logits": self.forbidden_head(features),
            "tactical_score": tactical_score,
        }


__all__ = ["AdvancedPolicyValueNet", "SEBlock"]
