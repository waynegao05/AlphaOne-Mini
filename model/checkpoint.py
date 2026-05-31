"""模型 checkpoint 的保存与加载。

当前阶段只持久化：
- ``model_state_dict``
- 可选的 ``optimizer_state_dict``
- 可选的 ``metadata`` (任意可被 :func:`torch.save` 序列化的对象)

不包含训练曲线、replay buffer 等，留给后续训练循环阶段处理。
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch


def _ensure_parent_dir(path: str) -> None:
    """如果 ``path`` 的父目录不存在，则创建之。"""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def save_checkpoint(
    model: torch.nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """把 ``model`` (及可选的 ``optimizer`` / ``metadata``)写到 ``path``。

    返回实际写入的 ``state`` 字典(便于上层做 round-trip 校验)。
    """
    _ensure_parent_dir(path)

    state: Dict[str, Any] = {"model_state_dict": model.state_dict()}
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if metadata is not None:
        state["metadata"] = metadata

    torch.save(state, path)
    return state


def load_checkpoint(
    model: torch.nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """从 ``path`` 恢复模型(以及可选的 optimizer)。

    - 把模型参数 load 进 ``model``。
    - 若 ``optimizer`` 不为空且 checkpoint 中存在对应键，则一并恢复。
    - 返回完整的 checkpoint 字典(其中 ``metadata`` 字段可由上层使用)。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint 不存在: {path}")

    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=device)

    if "model_state_dict" not in state:
        raise KeyError(f"checkpoint 缺少 'model_state_dict' 字段: {path}")
    model.load_state_dict(state["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    return state


def load_checkpoint_metadata(path: str, device: str = "cpu") -> Dict[str, Any]:
    """Read checkpoint metadata without requiring a prebuilt model."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint 涓嶅瓨鍦? {path}")
    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=device)
    metadata = state.get("metadata", {})
    if metadata is None:
        metadata = {}
    return dict(metadata)


def load_checkpoint_checked(
    model: torch.nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
    expected_model_type: Optional[str] = None,
    allow_model_type_override: bool = False,
) -> Dict[str, Any]:
    """Load checkpoint and validate model_type metadata when requested."""
    metadata = load_checkpoint_metadata(path, device=device)
    checkpoint_model_type = metadata.get("model_type")
    if (
        expected_model_type is not None
        and checkpoint_model_type is not None
        and checkpoint_model_type != expected_model_type
        and not allow_model_type_override
    ):
        raise ValueError(
            f"checkpoint model_type {checkpoint_model_type!r} does not match "
            f"expected {expected_model_type!r}"
        )
    return load_checkpoint(model, path, optimizer=optimizer, device=device)


__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "load_checkpoint_metadata",
    "load_checkpoint_checked",
]
