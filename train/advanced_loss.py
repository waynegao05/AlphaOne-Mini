"""Advanced multi-task loss for the deep Gomoku model."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F

from .loss import policy_loss_fn, value_loss_fn


DEFAULT_ADVANCED_LOSS_WEIGHTS = {
    "policy": 1.0,
    "value": 1.0,
    "threat": 0.3,
    "forbidden": 0.2,
    "tactical_score": 0.1,
}


def _zero_like_loss(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def advanced_policy_value_loss(
    model_outputs,
    target_policy: torch.Tensor,
    target_value: torch.Tensor,
    threat_labels: torch.Tensor | None = None,
    forbidden_labels: torch.Tensor | None = None,
    tactical_score_labels: torch.Tensor | None = None,
    loss_weights: Mapping[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute policy, value, and optional auxiliary losses."""
    weights = dict(DEFAULT_ADVANCED_LOSS_WEIGHTS)
    if loss_weights:
        weights.update({key: float(value) for key, value in loss_weights.items()})

    if isinstance(model_outputs, dict):
        policy_logits = model_outputs["policy_logits"]
        pred_value = model_outputs["value"]
    else:
        policy_logits, pred_value = model_outputs

    p_loss = policy_loss_fn(policy_logits, target_policy)
    v_loss = value_loss_fn(pred_value, target_value)
    threat_loss = _zero_like_loss(policy_logits)
    forbidden_loss = _zero_like_loss(policy_logits)
    tactical_score_loss = _zero_like_loss(policy_logits)

    if isinstance(model_outputs, dict) and threat_labels is not None and "threat_logits" in model_outputs:
        threat_loss = F.binary_cross_entropy_with_logits(
            model_outputs["threat_logits"], threat_labels
        )
    if (
        isinstance(model_outputs, dict)
        and forbidden_labels is not None
        and "forbidden_logits" in model_outputs
    ):
        forbidden_loss = F.binary_cross_entropy_with_logits(
            model_outputs["forbidden_logits"], forbidden_labels
        )
    if (
        isinstance(model_outputs, dict)
        and tactical_score_labels is not None
        and "tactical_score" in model_outputs
    ):
        tactical_score_loss = F.mse_loss(model_outputs["tactical_score"], tactical_score_labels)

    total = (
        weights["policy"] * p_loss
        + weights["value"] * v_loss
        + weights["threat"] * threat_loss
        + weights["forbidden"] * forbidden_loss
        + weights["tactical_score"] * tactical_score_loss
    )
    return {
        "total_loss": total,
        "policy_loss": p_loss,
        "value_loss": v_loss,
        "threat_loss": threat_loss,
        "forbidden_loss": forbidden_loss,
        "tactical_score_loss": tactical_score_loss,
    }


__all__ = ["DEFAULT_ADVANCED_LOSS_WEIGHTS", "advanced_policy_value_loss"]
