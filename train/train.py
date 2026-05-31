"""AlphaZero mini 训练循环。

提供：

- :class:`SelfPlayDataset` : ``(states, policies, values)`` -> torch Dataset
- :func:`load_selfplay_npz` : 读取第五批 :class:`selfplay.replay_buffer.ReplayBuffer`
  保存的 ``.npz``，自动把 ``values`` 兼容到 ``[N, 1]``，并对 shape 做完整校验。
- :func:`create_dataloader` : 用合理默认值包一个 ``torch.utils.data.DataLoader``。
- :func:`train_one_epoch`   : 跑一个 epoch，返回平均 ``total/policy/value`` loss。
- :func:`train_model`       : 端到端训练若干 epoch，每个 epoch 后写一份
  ``latest`` checkpoint(走 :mod:`model.checkpoint`，与第三批保持一致)。

本模块只做训练；不生成自博弈数据、不做模型评估、不做 best 模型替换。
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from model.checkpoint import save_checkpoint

from .advanced_loss import advanced_policy_value_loss
from .loss import alphazero_loss
from .progress import format_seconds, progress_print


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SelfPlayDataset(Dataset):
    """把 ``(states, policies, values)`` 包成 torch Dataset。

    - ``states``   形状 ``(N, 4, 15, 15)``，dtype 转 ``torch.float32``。
    - ``policies`` 形状 ``(N, 225)``，dtype 转 ``torch.float32``。
    - ``values``   ``(N,)`` 或 ``(N, 1)``，统一升成 ``(N, 1) float32``。
    """

    def __init__(
        self,
        states: np.ndarray,
        policies: np.ndarray,
        values: np.ndarray,
    ) -> None:
        states_np = np.asarray(states, dtype=np.float32)
        policies_np = np.asarray(policies, dtype=np.float32)
        values_np = np.asarray(values, dtype=np.float32)

        if states_np.ndim != 4 or states_np.shape[1:] != (4, 15, 15):
            raise ValueError(
                f"states shape 必须为 (N, 4, 15, 15)，实际 {states_np.shape}"
            )
        if policies_np.ndim != 2 or policies_np.shape[1] != 225:
            raise ValueError(
                f"policies shape 必须为 (N, 225)，实际 {policies_np.shape}"
            )
        if values_np.ndim == 1:
            values_np = values_np.reshape(-1, 1)
        if values_np.ndim != 2 or values_np.shape[1] != 1:
            raise ValueError(
                f"values shape 必须为 (N,) 或 (N, 1)，实际 {values_np.shape}"
            )
        n = states_np.shape[0]
        if not (policies_np.shape[0] == n and values_np.shape[0] == n):
            raise ValueError(
                "三种数组的样本数不一致: "
                f"states={n}, policies={policies_np.shape[0]}, values={values_np.shape[0]}"
            )

        self.states = torch.from_numpy(states_np).float()
        self.policies = torch.from_numpy(policies_np).float()
        self.values = torch.from_numpy(values_np).float()

    def __len__(self) -> int:
        return self.states.size(0)

    def __getitem__(self, index: int):
        return self.states[index], self.policies[index], self.values[index]


class AuxiliarySelfPlayDataset(Dataset):
    """Dataset variant that preserves optional advanced auxiliary labels."""

    def __init__(self, arrays: Dict[str, np.ndarray]) -> None:
        self.tensors = {
            key: torch.from_numpy(np.asarray(value, dtype=np.float32)).float()
            for key, value in arrays.items()
        }

    def __len__(self) -> int:
        return int(self.tensors["states"].shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.tensors.items()}


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def load_selfplay_npz(
    path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取 ReplayBuffer 写出的 ``.npz`` 文件。

    返回 ``(states, policies, values)``，``values`` 总是 ``[N, 1] float32``。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"自博弈数据文件不存在: {path}\n"
            "请先运行: python main_selfplay.py"
        )

    with np.load(path, allow_pickle=False) as data:
        for key in ("states", "policies", "values"):
            if key not in data.files:
                raise KeyError(f"npz 文件缺少 '{key}' 字段: {path}")
        states = np.array(data["states"], dtype=np.float32, copy=True)
        policies = np.array(data["policies"], dtype=np.float32, copy=True)
        values = np.array(data["values"], dtype=np.float32, copy=True)

    if states.ndim != 4 or states.shape[1:] != (4, 15, 15):
        raise ValueError(
            f"states shape 必须为 (N, 4, 15, 15)，实际 {states.shape}"
        )
    if policies.ndim != 2 or policies.shape[1] != 225:
        raise ValueError(
            f"policies shape 必须为 (N, 225)，实际 {policies.shape}"
        )
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2 or values.shape[1] != 1:
        raise ValueError(
            f"values shape 必须为 (N,) 或 (N, 1)，实际 {values.shape}"
        )
    n = states.shape[0]
    if not (policies.shape[0] == n and values.shape[0] == n):
        raise ValueError(
            "样本数不一致: "
            f"states={n}, policies={policies.shape[0]}, values={values.shape[0]}"
        )

    return states, policies, values


def load_training_npz_with_optional_aux(path: str) -> Dict[str, np.ndarray]:
    """Load policy-value arrays plus optional advanced auxiliary labels."""
    states, policies, values = load_selfplay_npz(path)
    arrays: Dict[str, np.ndarray] = {
        "states": states,
        "policies": policies,
        "values": values,
    }
    with np.load(path, allow_pickle=False) as data:
        for key in ("threat_labels", "forbidden_labels", "tactical_scores"):
            if key in data.files:
                arrays[key] = np.asarray(data[key], dtype=np.float32)
    n = states.shape[0]
    if "threat_labels" in arrays and arrays["threat_labels"].shape != (n, 12, 15, 15):
        raise ValueError(f"invalid threat_labels shape: {arrays['threat_labels'].shape}")
    if "forbidden_labels" in arrays and arrays["forbidden_labels"].shape != (n, 1, 15, 15):
        raise ValueError(f"invalid forbidden_labels shape: {arrays['forbidden_labels'].shape}")
    if "tactical_scores" in arrays and arrays["tactical_scores"].shape != (n, 225):
        raise ValueError(f"invalid tactical_scores shape: {arrays['tactical_scores'].shape}")
    return arrays


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------
def create_dataloader(
    states: np.ndarray,
    policies: np.ndarray,
    values: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    """把 numpy 数组 -> :class:`SelfPlayDataset` -> :class:`DataLoader`。"""
    dataset = SelfPlayDataset(states, policies, values)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )


# ---------------------------------------------------------------------------
# 训练循环
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    grad_clip: Optional[float] = None,
) -> Dict[str, float]:
    """跑一个 epoch，返回平均 ``total / policy / value`` loss。

    - ``model.train()``、``zero_grad`` -> ``forward`` -> ``alphazero_loss`` ->
      ``backward`` -> 可选 ``clip_grad_norm_`` -> ``step``。
    - 平均 loss 用样本数加权(``loss.item() * batch_size``)。
    """
    model.train()
    total_loss_sum = 0.0
    policy_loss_sum = 0.0
    value_loss_sum = 0.0
    n_samples = 0
    n_batches = 0

    for batch in dataloader:
        batch_states, batch_policies, batch_values = batch
        batch_states = batch_states.to(device)
        batch_policies = batch_policies.to(device)
        batch_values = batch_values.to(device)

        policy_logits, pred_value = model(batch_states)
        total_loss, p_loss, v_loss = alphazero_loss(
            policy_logits, pred_value, batch_policies, batch_values
        )

        optimizer.zero_grad()
        total_loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        bs = batch_states.size(0)
        total_loss_sum += float(total_loss.item()) * bs
        policy_loss_sum += float(p_loss.item()) * bs
        value_loss_sum += float(v_loss.item()) * bs
        n_samples += bs
        n_batches += 1

    if n_samples == 0:
        return {
            "total_loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "num_batches": 0,
            "num_samples": 0,
        }
    return {
        "total_loss": total_loss_sum / n_samples,
        "policy_loss": policy_loss_sum / n_samples,
        "value_loss": value_loss_sum / n_samples,
        "num_batches": n_batches,
        "num_samples": n_samples,
    }


def train_one_epoch_advanced(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    grad_clip: Optional[float] = None,
    loss_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """One epoch using optional auxiliary labels and advanced loss."""
    model.train()
    sums = {
        "total_loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "threat_loss": 0.0,
        "forbidden_loss": 0.0,
        "tactical_score_loss": 0.0,
    }
    n_samples = 0
    n_batches = 0
    for batch in dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}
        try:
            outputs = model(batch["states"], return_aux=True)
        except TypeError:
            policy_logits, value = model(batch["states"])
            outputs = {"policy_logits": policy_logits, "value": value}
        losses = advanced_policy_value_loss(
            outputs,
            batch["policies"],
            batch["values"],
            threat_labels=batch.get("threat_labels"),
            forbidden_labels=batch.get("forbidden_labels"),
            tactical_score_labels=batch.get("tactical_scores"),
            loss_weights=loss_weights,
        )
        optimizer.zero_grad()
        losses["total_loss"].backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        bs = int(batch["states"].size(0))
        for key in sums:
            sums[key] += float(losses[key].item()) * bs
        n_samples += bs
        n_batches += 1
    if n_samples == 0:
        return {**{key: 0.0 for key in sums}, "num_batches": 0, "num_samples": 0}
    stats = {key: value / n_samples for key, value in sums.items()}
    stats["num_batches"] = n_batches
    stats["num_samples"] = n_samples
    return stats


def train_model(
    model: torch.nn.Module,
    data_path: str,
    checkpoint_dir: str = os.path.join("outputs", "checkpoints"),
    epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    grad_clip: Optional[float] = 5.0,
    shuffle: bool = True,
    num_workers: int = 0,
    model_type: Optional[str] = None,
    metadata_extra: Optional[Dict[str, Any]] = None,
    use_auxiliary_loss: bool = False,
    loss_weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """从 ``data_path`` 加载自博弈数据并训练 ``epochs`` 轮。

    每个 epoch 完成后把模型 + optimizer + metadata 写到
    ``{checkpoint_dir}/latest.pt``。返回每个 epoch 的统计组成的 ``history`` 列表。
    """
    total_start = time.perf_counter()
    progress_print(
        f"START train_model data={data_path} epochs={int(epochs)} "
        f"batch_size={int(batch_size)} device={device} "
        f"model_type={model_type or getattr(model, 'model_type', 'cnn')}",
        "train",
    )
    arrays = load_training_npz_with_optional_aux(data_path)
    use_aux_data = bool(use_auxiliary_loss) and any(
        key in arrays for key in ("threat_labels", "forbidden_labels", "tactical_scores")
    )
    if use_aux_data:
        dataloader = DataLoader(
            AuxiliarySelfPlayDataset(arrays),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
    else:
        dataloader = create_dataloader(
            arrays["states"],
            arrays["policies"],
            arrays["values"],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_path = os.path.join(checkpoint_dir, "latest.pt")

    history: List[Dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        progress_print(f"epoch {epoch}/{epochs} start", "train")
        if use_aux_data:
            stats = train_one_epoch_advanced(
                model,
                dataloader,
                optimizer,
                device=device,
                grad_clip=grad_clip,
                loss_weights=loss_weights,
            )
        else:
            stats = train_one_epoch(
                model,
                dataloader,
                optimizer,
                device=device,
                grad_clip=grad_clip,
            )
        record = {
            "epoch": epoch,
            "loss": stats["total_loss"],
            "total_loss": stats["total_loss"],
            "policy_loss": stats["policy_loss"],
            "value_loss": stats["value_loss"],
            "threat_loss": stats.get("threat_loss", 0.0),
            "forbidden_loss": stats.get("forbidden_loss", 0.0),
            "tactical_score_loss": stats.get("tactical_score_loss", 0.0),
            "num_samples": stats["num_samples"],
            "num_batches": stats["num_batches"],
        }
        history.append(record)

        print(
            f"[epoch {epoch}/{epochs}] "
            f"total={stats['total_loss']:.4f} "
            f"policy={stats['policy_loss']:.4f} "
            f"value={stats['value_loss']:.4f} "
            f"(samples={stats['num_samples']}, batches={stats['num_batches']})"
        )

        metadata = {
            "epoch": epoch,
            "loss": stats["total_loss"],
            "policy_loss": stats["policy_loss"],
            "value_loss": stats["value_loss"],
            "data_path": os.path.abspath(data_path),
            "num_samples": stats["num_samples"],
            "num_batches": stats["num_batches"],
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "device": str(device),
            "model_type": model_type or getattr(model, "model_type", "cnn"),
            "use_auxiliary_loss": bool(use_auxiliary_loss),
            "used_auxiliary_data": bool(use_aux_data),
            "loss_weights": loss_weights or {},
        }
        if metadata_extra:
            metadata.update(metadata_extra)
        save_checkpoint(model, latest_path, optimizer=optimizer, metadata=metadata)
        progress_print(
            f"epoch {epoch}/{epochs} complete total={stats['total_loss']:.4f} "
            f"policy={stats['policy_loss']:.4f} value={stats['value_loss']:.4f} "
            f"threat={stats.get('threat_loss', 0.0):.4f} "
            f"forbidden={stats.get('forbidden_loss', 0.0):.4f} "
            f"samples={stats['num_samples']} batches={stats['num_batches']} "
            f"checkpoint={latest_path} elapsed={format_seconds(time.perf_counter() - epoch_start)}",
            "train",
        )

    progress_print(
        f"DONE train_model checkpoint={latest_path} "
        f"elapsed={format_seconds(time.perf_counter() - total_start)}",
        "train",
    )
    return history


__all__ = [
    "SelfPlayDataset",
    "AuxiliarySelfPlayDataset",
    "load_selfplay_npz",
    "load_training_npz_with_optional_aux",
    "create_dataloader",
    "train_one_epoch",
    "train_one_epoch_advanced",
    "train_model",
]
