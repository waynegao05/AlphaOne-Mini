"""AlphaZero 风格 loss 函数。

总 loss::

    total_loss = policy_loss + value_loss

- ``policy_loss`` : soft target 的交叉熵
  ``- mean( sum( target_policy * log_softmax(policy_logits), dim=1 ) )``。
  注意：``target_policy`` 是 MCTS 的搜索概率分布，不是 one-hot；
  所以这里不能用 ``CrossEntropyLoss`` 的"类别索引"形式。
- ``value_loss`` : MSE
  ``mean( (pred_value - target_value) ** 2 )``。

L2 正则建议交给 optimizer 的 ``weight_decay``，不在这里手动加。
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def policy_loss_fn(
    policy_logits: torch.Tensor, target_policy: torch.Tensor
) -> torch.Tensor:
    """Soft target 交叉熵。

    Parameters
    ----------
    policy_logits : ``[batch, action_size]``  未做 softmax 的网络原始输出。
    target_policy : ``[batch, action_size]``  概率分布(MCTS 搜索概率)。
    """
    if policy_logits.shape != target_policy.shape:
        raise ValueError(
            f"policy_logits shape {tuple(policy_logits.shape)} 与 "
            f"target_policy shape {tuple(target_policy.shape)} 不一致"
        )
    log_probs = F.log_softmax(policy_logits, dim=1)
    per_sample = -(target_policy * log_probs).sum(dim=1)
    return per_sample.mean()


def value_loss_fn(
    pred_value: torch.Tensor, target_value: torch.Tensor
) -> torch.Tensor:
    """MSE，自动把 ``[batch]`` 形状的 target 升成 ``[batch, 1]``。"""
    if pred_value.dim() == 1:
        pred_value = pred_value.unsqueeze(-1)
    if target_value.dim() == 1:
        target_value = target_value.unsqueeze(-1)
    if pred_value.shape != target_value.shape:
        raise ValueError(
            f"pred_value shape {tuple(pred_value.shape)} 与 "
            f"target_value shape {tuple(target_value.shape)} 不一致"
        )
    return F.mse_loss(pred_value, target_value)


def alphazero_loss(
    policy_logits: torch.Tensor,
    pred_value: torch.Tensor,
    target_policy: torch.Tensor,
    target_value: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 ``(total_loss, policy_loss, value_loss)``。"""
    p_loss = policy_loss_fn(policy_logits, target_policy)
    v_loss = value_loss_fn(pred_value, target_value)
    total = p_loss + v_loss
    return total, p_loss, v_loss


__all__ = ["policy_loss_fn", "value_loss_fn", "alphazero_loss"]
