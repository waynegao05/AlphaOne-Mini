"""Factory helpers for supported policy-value model families."""

from __future__ import annotations

from typing import Any, Mapping

from .advanced_policy_value_net import AdvancedPolicyValueNet
from .policy_value_net import PolicyValueNet
from .resnet_policy_value_net import SmallResNetPolicyValueNet


def create_model(
    model_type: str = "advanced",
    board_size: int = 15,
    input_channels: int = 4,
    **kwargs: Any,
):
    """Create a model by type: ``cnn``, ``resnet``, or ``advanced``."""
    normalized = (model_type or "advanced").lower()
    if normalized == "cnn":
        allowed = {"hidden_channels"}
        filtered = {key: value for key, value in kwargs.items() if key in allowed}
        return PolicyValueNet(
            board_size=board_size,
            input_channels=input_channels,
            **filtered,
        )
    if normalized == "resnet":
        allowed = {"channels", "blocks"}
        filtered = {key: value for key, value in kwargs.items() if key in allowed}
        return SmallResNetPolicyValueNet(
            board_size=board_size,
            input_channels=input_channels,
            **filtered,
        )
    if normalized == "advanced":
        allowed = {
            "channels",
            "blocks",
            "attention_every",
            "num_threat_channels",
            "use_tactical_score",
        }
        filtered = {key: value for key, value in kwargs.items() if key in allowed}
        return AdvancedPolicyValueNet(
            board_size=board_size,
            input_channels=input_channels,
            **filtered,
        )
    raise ValueError(f"unknown model_type: {model_type!r}")


def create_model_from_metadata(
    metadata: Mapping[str, Any] | None,
    fallback_model_type: str = "advanced",
    **kwargs: Any,
):
    """Create a model from checkpoint metadata plus optional overrides."""
    metadata = metadata or {}
    model_type = str(metadata.get("model_type", fallback_model_type))
    model_kwargs = dict(metadata.get("model_kwargs", {}) or {})
    model_kwargs.update(kwargs)
    return create_model(model_type=model_type, **model_kwargs)


__all__ = ["create_model", "create_model_from_metadata"]
