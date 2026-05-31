"""Record-based supervised policy pretraining."""

from __future__ import annotations

import os
import time
from typing import Iterable, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from game.board import BOARD_SIZE, BLACK, Board
from game.encoder import action_to_index, encode_board
from game.rules_forbidden import get_game_result_forbidden
from model.checkpoint import load_checkpoint_checked, save_checkpoint
from records.parser import RecordError, parse_record, validate_move_sequence
from selfplay.data_augmentation import (
    symmetry_specs,
    transform_flat_board,
    transform_plane_stack,
    transform_policy,
    transform_state,
)
from train.advanced_loss import advanced_policy_value_loss
from train.progress import format_seconds, progress_print
from train.scheduler import create_scheduler
from train.tactical_distillation import make_policy_target
from train.train import create_dataloader, train_one_epoch


DEFAULT_SUPERVISED_DATA_PATH = os.path.join(
    "outputs", "supervised", "supervised_latest.npz"
)


class SupervisedPretrainError(ValueError):
    """Raised for invalid supervised-pretraining records or datasets."""


def _validate_arrays(
    states: np.ndarray,
    policies: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = np.asarray(states, dtype=np.float32)
    policies = np.asarray(policies, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if states.ndim != 4 or states.shape[1:] != (4, BOARD_SIZE, BOARD_SIZE):
        raise SupervisedPretrainError(f"invalid states shape: {states.shape}")
    if policies.ndim != 2 or policies.shape[1] != BOARD_SIZE * BOARD_SIZE:
        raise SupervisedPretrainError(f"invalid policies shape: {policies.shape}")
    if values.ndim != 2 or values.shape[1] != 1:
        raise SupervisedPretrainError(f"invalid values shape: {values.shape}")
    if not (states.shape[0] == policies.shape[0] == values.shape[0]):
        raise SupervisedPretrainError(
            "states, policies and values must have matching sample counts"
        )
    return states, policies, values


def build_supervised_samples_from_record(
    record_text: str,
    rule_mode: str = "basic",
    smoothing: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build `(states, policies, values)` from a single record text."""
    if rule_mode not in ("basic", "forbidden"):
        raise ValueError(f"unknown rule_mode: {rule_mode!r}")
    try:
        moves = parse_record(record_text)
        validate_move_sequence(moves)
    except (RecordError, ValueError) as exc:
        raise SupervisedPretrainError(f"invalid record: {exc}") from exc

    board = Board()
    states: list[np.ndarray] = []
    policies: list[np.ndarray] = []
    values: list[list[float]] = []

    for idx, move in enumerate(moves, start=1):
        if board.current_player != move.color:
            raise SupervisedPretrainError(
                f"invalid record at move {idx}: expected color {board.current_player}, "
                f"got {move.color}"
            )
        if not board.is_legal_move(move.x, move.y):
            raise SupervisedPretrainError(
                f"invalid record at move {idx}: occupied or out-of-board {move.coord}"
            )

        states.append(encode_board(board, current_player=move.color))
        policies.append(
            make_policy_target(
                action_to_index(move.x, move.y, BOARD_SIZE),
                smoothing=smoothing,
            )
        )
        values.append([0.0])

        board.place_stone(move.x, move.y)
        if rule_mode == "forbidden" and move.color == BLACK:
            result = get_game_result_forbidden(board, board.last_move)
            if result.forbidden:
                raise SupervisedPretrainError(
                    f"invalid record at move {idx}: forbidden black move {move.coord}"
                )

    return _validate_arrays(
        np.stack(states) if states else np.zeros((0, 4, BOARD_SIZE, BOARD_SIZE)),
        np.stack(policies) if policies else np.zeros((0, BOARD_SIZE * BOARD_SIZE)),
        np.asarray(values, dtype=np.float32),
    )


def save_supervised_dataset(
    states: np.ndarray,
    policies: np.ndarray,
    values: np.ndarray,
    path: str,
) -> None:
    states, policies, values = _validate_arrays(states, policies, values)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    np.savez_compressed(path, states=states, policies=policies, values=values)


def load_supervised_npz(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"supervised dataset not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        for key in ("states", "policies", "values"):
            if key not in data.files:
                raise SupervisedPretrainError(f"dataset missing key {key!r}: {path}")
        return _validate_arrays(data["states"], data["policies"], data["values"])


def load_pretrain_npz_with_aux(path: str) -> dict[str, np.ndarray]:
    """Load supervised/tactical npz data and preserve optional auxiliary arrays."""
    states, policies, values = load_supervised_npz(path)
    result = {"states": states, "policies": policies, "values": values}
    with np.load(path, allow_pickle=False) as data:
        for key in ("threat_labels", "forbidden_labels", "tactical_scores"):
            if key in data.files:
                result[key] = np.asarray(data[key], dtype=np.float32)
    n = states.shape[0]
    if "threat_labels" in result and result["threat_labels"].shape != (
        n,
        12,
        BOARD_SIZE,
        BOARD_SIZE,
    ):
        raise SupervisedPretrainError(
            f"invalid threat_labels shape: {result['threat_labels'].shape}"
        )
    if "forbidden_labels" in result and result["forbidden_labels"].shape != (
        n,
        1,
        BOARD_SIZE,
        BOARD_SIZE,
    ):
        raise SupervisedPretrainError(
            f"invalid forbidden_labels shape: {result['forbidden_labels'].shape}"
        )
    if "tactical_scores" in result and result["tactical_scores"].shape != (
        n,
        BOARD_SIZE * BOARD_SIZE,
    ):
        raise SupervisedPretrainError(
            f"invalid tactical_scores shape: {result['tactical_scores'].shape}"
        )
    return result


class PretrainDataset(Dataset):
    """Tensor dataset for optional advanced auxiliary labels."""

    def __init__(self, arrays: dict[str, np.ndarray], augment: bool = False) -> None:
        self.arrays = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in arrays.items()
        }
        self.augment = bool(augment)
        self._symmetry_specs = symmetry_specs()

    def __len__(self) -> int:
        base_len = int(self.arrays["states"].shape[0])
        return base_len * len(self._symmetry_specs) if self.augment else base_len

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self.augment:
            base_index = int(index) // len(self._symmetry_specs)
            rotation, flip = self._symmetry_specs[int(index) % len(self._symmetry_specs)]
        else:
            base_index = int(index)
            rotation, flip = 0, False

        sample: dict[str, np.ndarray] = {
            "states": self.arrays["states"][base_index],
            "policies": self.arrays["policies"][base_index],
            "values": self.arrays["values"][base_index],
        }
        if self.augment:
            sample["states"] = transform_state(sample["states"], rotation, flip)
            sample["policies"] = transform_policy(sample["policies"], rotation, flip)

        if "threat_labels" in self.arrays:
            threat = self.arrays["threat_labels"][base_index]
            sample["threat_labels"] = (
                transform_plane_stack(threat, rotation, flip, name="threat_labels")
                if self.augment
                else threat
            )
        if "forbidden_labels" in self.arrays:
            forbidden = self.arrays["forbidden_labels"][base_index]
            sample["forbidden_labels"] = (
                transform_plane_stack(forbidden, rotation, flip, name="forbidden_labels")
                if self.augment
                else forbidden
            )
        if "tactical_scores" in self.arrays:
            scores = self.arrays["tactical_scores"][base_index]
            sample["tactical_scores"] = (
                transform_flat_board(
                    scores,
                    rotation,
                    flip,
                    normalize=False,
                    name="tactical_scores",
                )
                if self.augment
                else scores
            )
        return {
            key: torch.from_numpy(np.asarray(value, dtype=np.float32)).float()
            for key, value in sample.items()
        }


def _forward_model(model: torch.nn.Module, states: torch.Tensor, return_aux: bool):
    if return_aux:
        try:
            return model(states, return_aux=True)
        except TypeError:
            pass
    policy_logits, value = model(states)
    return {"policy_logits": policy_logits, "value": value}


def evaluate_policy_topk(
    model: torch.nn.Module,
    data_path: str,
    device: str = "cpu",
    topks: tuple[int, ...] = (1, 3, 5),
    batch_size: int = 128,
) -> dict:
    """Evaluate policy top-k accuracy against one-hot or soft policy targets."""
    arrays = load_pretrain_npz_with_aux(data_path)
    dataset = PretrainDataset(arrays, augment=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    hits = {int(k): 0 for k in topks}
    total = 0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            states = batch["states"].to(device)
            target = batch["policies"].to(device)
            outputs = _forward_model(model, states, return_aux=False)
            logits = outputs["policy_logits"]
            expected = torch.argmax(target, dim=1)
            max_k = max(hits) if hits else 1
            top = torch.topk(logits, k=min(max_k, logits.shape[1]), dim=1).indices
            for k in hits:
                hits[k] += (top[:, :k] == expected.unsqueeze(1)).any(dim=1).sum().item()
            total += int(states.shape[0])
    if was_training:
        model.train()
    return {
        f"top{k}": (hits[k] / total if total else 0.0)
        for k in sorted(hits)
    } | {"total": total}


def build_supervised_dataset_from_records(
    record_texts: Iterable[str],
    output_path: str = DEFAULT_SUPERVISED_DATA_PATH,
    rule_mode: str = "basic",
    smoothing: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_states: list[np.ndarray] = []
    all_policies: list[np.ndarray] = []
    all_values: list[np.ndarray] = []
    for record_text in record_texts:
        states, policies, values = build_supervised_samples_from_record(
            record_text,
            rule_mode=rule_mode,
            smoothing=smoothing,
        )
        all_states.append(states)
        all_policies.append(policies)
        all_values.append(values)

    if not all_states:
        raise SupervisedPretrainError("no records provided")
    states = np.concatenate(all_states, axis=0)
    policies = np.concatenate(all_policies, axis=0)
    values = np.concatenate(all_values, axis=0)
    save_supervised_dataset(states, policies, values, output_path)
    return states, policies, values


def train_policy_pretrain(
    model: torch.nn.Module,
    data_path: str,
    checkpoint_dir: str = os.path.join("outputs", "checkpoints"),
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
    weight_decay: float = 1e-4,
    grad_clip: float | None = 5.0,
    use_auxiliary_loss: bool = False,
    loss_weights: dict | None = None,
    checkpoint_name: str = "pretrained.pt",
    model_type: str | None = None,
    resume_from: str | None = None,
    allow_model_type_override: bool = False,
    scheduler_type: str = "constant",
    warmup_epochs: int = 0,
    mixed_precision: bool = False,
    augment_dataset: bool = False,
    validation_data_path: str | None = None,
) -> List[dict]:
    """Train PolicyValueNet on supervised soft policy targets."""
    arrays = load_pretrain_npz_with_aux(data_path)
    dataloader = DataLoader(
        PretrainDataset(arrays, augment=augment_dataset),
        batch_size=batch_size,
        shuffle=True,
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    start_epoch = 0
    resume_metadata: dict = {}
    if resume_from:
        state = load_checkpoint_checked(
            model,
            resume_from,
            optimizer=optimizer,
            device=device,
            expected_model_type=model_type or getattr(model, "model_type", "cnn"),
            allow_model_type_override=allow_model_type_override,
        )
        resume_metadata = dict(state.get("metadata", {}) or {})
        start_epoch = int(resume_metadata.get("epoch", 0) or 0)
    previous_best = resume_metadata.get("best_metric")
    best_metric = None if previous_best is None else float(previous_best)
    scheduler = create_scheduler(
        optimizer,
        scheduler_type=scheduler_type,
        total_epochs=max(1, int(epochs)),
        warmup_epochs=warmup_epochs,
    )
    use_amp = bool(mixed_precision) and torch.device(device).type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    total_start = time.perf_counter()
    progress_print(
        "START supervised_pretrain "
        f"data={data_path} samples={len(dataloader.dataset)} "
        f"base_samples={arrays['states'].shape[0]} augment_dataset={bool(augment_dataset)} "
        f"epochs={int(epochs)} "
        f"batch_size={int(batch_size)} device={device} amp={use_amp} "
        f"checkpoint={checkpoint_path}",
        "train",
    )

    history: List[dict] = []
    for local_epoch in range(1, int(epochs) + 1):
        epoch_start = time.perf_counter()
        epoch = start_epoch + local_epoch
        progress_print(
            f"epoch {local_epoch}/{int(epochs)} start global_epoch={epoch}",
            "train",
        )
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
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = _forward_model(model, batch["states"], return_aux=use_auxiliary_loss)
                losses = advanced_policy_value_loss(
                    outputs,
                    batch["policies"],
                    batch["values"],
                    threat_labels=batch.get("threat_labels") if use_auxiliary_loss else None,
                    forbidden_labels=batch.get("forbidden_labels") if use_auxiliary_loss else None,
                    tactical_score_labels=batch.get("tactical_scores") if use_auxiliary_loss else None,
                    loss_weights=loss_weights,
                )
            optimizer.zero_grad()
            scaler.scale(losses["total_loss"]).backward()
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            bs = int(batch["states"].shape[0])
            for key in sums:
                sums[key] += float(losses[key].item()) * bs
            n_samples += bs
            n_batches += 1
        if n_samples:
            stats = {key: value / n_samples for key, value in sums.items()}
        else:
            stats = {key: 0.0 for key in sums}
        stats["num_samples"] = n_samples
        stats["num_batches"] = n_batches
        current_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()
        current_metric = -float(stats["total_loss"])
        best_metric = current_metric if best_metric is None else max(best_metric, current_metric)
        record = {
            "epoch": epoch,
            "loss": stats["total_loss"],
            "total_loss": stats["total_loss"],
            "policy_loss": stats["policy_loss"],
            "value_loss": stats["value_loss"],
            "threat_loss": stats["threat_loss"],
            "forbidden_loss": stats["forbidden_loss"],
            "tactical_score_loss": stats["tactical_score_loss"],
            "num_samples": stats["num_samples"],
            "num_batches": stats["num_batches"],
            "learning_rate": current_lr,
        }
        validation_stats = None
        if validation_data_path:
            validation_stats = evaluate_policy_topk(
                model,
                validation_data_path,
                device=device,
                topks=(1, 3, 5),
                batch_size=batch_size,
            )
            record["tactical_validation"] = validation_stats
        history.append(record)
        metadata = {
            "pretrain_type": "supervised_policy",
            "epoch": epoch,
            "epochs": int(epochs),
            "data_path": os.path.abspath(data_path),
            "loss": stats["total_loss"],
            "policy_loss": stats["policy_loss"],
            "value_loss": stats["value_loss"],
            "num_samples": stats["num_samples"],
            "batch_size": int(batch_size),
            "learning_rate": float(lr),
            "current_learning_rate": current_lr,
            "weight_decay": float(weight_decay),
            "device": str(device),
            "model_type": model_type or getattr(model, "model_type", "cnn"),
            "use_auxiliary_loss": bool(use_auxiliary_loss),
            "loss_weights": loss_weights or {},
            "mixed_precision": bool(mixed_precision),
            "amp_enabled": bool(use_amp),
            "augment_dataset": bool(augment_dataset),
            "validation_data_path": validation_data_path,
            "tactical_validation": validation_stats,
            "resume_from": resume_from,
            "resumed_from_epoch": start_epoch if resume_from else None,
            "best_metric": best_metric,
        }
        save_checkpoint(model, checkpoint_path, optimizer=optimizer, metadata=metadata)
        progress_print(
            f"epoch {local_epoch}/{int(epochs)} complete "
            f"global_epoch={epoch} total={stats['total_loss']:.4f} "
            f"policy={stats['policy_loss']:.4f} value={stats['value_loss']:.4f} "
            f"threat={stats['threat_loss']:.4f} forbidden={stats['forbidden_loss']:.4f} "
            f"lr={current_lr:.6g} samples={n_samples} batches={n_batches} "
            f"val_top1={(validation_stats or {}).get('top1', 0.0):.3f} "
            f"elapsed={format_seconds(time.perf_counter() - epoch_start)}",
            "train",
        )
    progress_print(
        f"DONE supervised_pretrain checkpoint={checkpoint_path} "
        f"elapsed={format_seconds(time.perf_counter() - total_start)}",
        "train",
    )
    return history


__all__ = [
    "DEFAULT_SUPERVISED_DATA_PATH",
    "SupervisedPretrainError",
    "build_supervised_samples_from_record",
    "build_supervised_dataset_from_records",
    "save_supervised_dataset",
    "load_supervised_npz",
    "load_pretrain_npz_with_aux",
    "evaluate_policy_topk",
    "train_policy_pretrain",
]
