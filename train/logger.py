"""JSONL and CSV training logger for experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping

from .config import save_resolved_config


class TrainingLogger:
    """Write experiment logs under `root_dir / experiment_name`."""

    CSV_FIELDS = [
        "event",
        "epoch",
        "total_loss",
        "policy_loss",
        "value_loss",
        "threat_loss",
        "forbidden_loss",
        "learning_rate",
        "duration_sec",
        "device",
        "model_type",
        "checkpoint_path",
    ]

    def __init__(self, experiment_name: str, root_dir: str = "outputs/logs") -> None:
        self.experiment_name = experiment_name
        self.log_dir = Path(root_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.log_dir / "train_log.jsonl"
        self.csv_path = self.log_dir / "train_log.csv"
        self.benchmark_path = self.log_dir / "benchmark_latest.json"
        self.config_path = self.log_dir / "config_resolved.yaml"
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.CSV_FIELDS)
                writer.writeheader()

    def save_config(self, config: Mapping) -> None:
        save_resolved_config(config, str(self.config_path))

    def _write_event(self, event: str, payload: Mapping) -> None:
        record = {"event": event, **dict(payload)}
        with open(self.jsonl_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        row = {field: record.get(field, "") for field in self.CSV_FIELDS}
        with open(self.csv_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.CSV_FIELDS)
            writer.writerow(row)

    def log_epoch(self, payload: Mapping) -> None:
        self._write_event("epoch", payload)

    def log_benchmark(self, payload: Mapping) -> None:
        self._write_event("benchmark", payload)
        with open(self.benchmark_path, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, ensure_ascii=False)


__all__ = ["TrainingLogger"]
