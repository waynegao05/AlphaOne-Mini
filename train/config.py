"""YAML configuration helpers for deep training experiments."""

from __future__ import annotations

import copy
import os
from typing import Any, Mapping

import yaml


REQUIRED_CONFIG_KEYS = [
    "experiment_name",
    "seed",
    "device",
    "allow_cpu_fallback",
    "model_type",
    "board_size",
    "rule_mode",
    "tactical_games",
    "use_augmentation",
    "use_auxiliary_loss",
    "pretrain_epochs",
    "finetune_epochs",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "num_simulations",
    "selfplay_games",
    "benchmark_games",
    "checkpoint_dir",
    "log_dir",
    "output_dir",
    "resume_from",
    "promote",
    "loss_weights",
]


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment_name": "deep_advanced_cuda",
    "seed": 2026,
    "device": "cuda",
    "allow_cpu_fallback": False,
    "model_type": "advanced",
    "board_size": 15,
    "rule_mode": "basic",
    "tactical_games": 10,
    "use_augmentation": True,
    "use_auxiliary_loss": True,
    "pretrain_epochs": 3,
    "finetune_epochs": 1,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "num_simulations": 50,
    "selfplay_games": 2,
    "benchmark_games": 10,
    "checkpoint_dir": os.path.join("outputs", "checkpoints"),
    "log_dir": os.path.join("outputs", "logs"),
    "output_dir": os.path.join("outputs", "experiments"),
    "resume_from": None,
    "promote": False,
    "loss_weights": {
        "policy": 1.0,
        "value": 1.0,
        "threat": 0.3,
        "forbidden": 0.2,
        "tactical_score": 0.1,
    },
    "scheduler": "constant",
    "warmup_epochs": 0,
    "grad_clip": 5.0,
    "mixed_precision": False,
    "max_moves": 30,
}


def _deep_merge(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if value is None:
            continue
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def validate_config(config: Mapping[str, Any]) -> None:
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError(f"config missing required keys: {missing}")


def load_config(path: str | None = None) -> dict[str, Any]:
    """Load YAML config and merge it over defaults."""
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path is not None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"config file not found: {path}")
        with open(path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"config root must be a mapping: {path}")
        config = _deep_merge(config, loaded)
    validate_config(config)
    return config


def merge_overrides(config: Mapping[str, Any], overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    """Merge non-None CLI overrides into a config."""
    return _deep_merge(dict(config), overrides or {})


def save_resolved_config(config: Mapping[str, Any], path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=False, allow_unicode=True)


__all__ = [
    "DEFAULT_CONFIG",
    "REQUIRED_CONFIG_KEYS",
    "load_config",
    "merge_overrides",
    "save_resolved_config",
    "validate_config",
]
