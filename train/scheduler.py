"""Learning-rate scheduler factory."""

from __future__ import annotations

import math

import torch


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "constant",
    total_epochs: int = 1,
    warmup_epochs: int = 0,
    step_size: int = 10,
    gamma: float = 0.1,
):
    """Create a torch LR scheduler by name."""
    name = (scheduler_type or "constant").lower()
    total_epochs = max(1, int(total_epochs))
    warmup_epochs = max(0, int(warmup_epochs))
    if name == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, int(step_size)),
            gamma=float(gamma),
        )
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_epochs),
        )
    if name == "warmup_cosine":
        def lr_lambda(epoch: int) -> float:
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return float(epoch + 1) / float(warmup_epochs)
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    raise ValueError(f"unknown scheduler: {scheduler_type!r}")


__all__ = ["create_scheduler"]
