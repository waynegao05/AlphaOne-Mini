from __future__ import annotations

import csv
import json


def test_training_logger_writes_jsonl_csv_and_config(tmp_path):
    from train.logger import TrainingLogger

    logger = TrainingLogger("demo", root_dir=str(tmp_path))
    logger.save_config({"experiment_name": "demo", "batch_size": 4})
    logger.log_epoch(
        {
            "epoch": 1,
            "total_loss": 1.2,
            "policy_loss": 1.0,
            "value_loss": 0.2,
            "threat_loss": 0.1,
            "forbidden_loss": 0.05,
            "learning_rate": 0.001,
            "duration_sec": 0.5,
            "device": "cpu",
            "model_type": "advanced",
            "checkpoint_path": "ckpt.pt",
        }
    )
    logger.log_benchmark({"player_a_win_rate": 1.0})

    assert logger.jsonl_path.exists()
    assert logger.csv_path.exists()
    assert logger.config_path.exists()
    lines = logger.jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(lines[0])["event"] == "epoch"
    assert json.loads(lines[1])["event"] == "benchmark"
    with open(logger.csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["model_type"] == "advanced"
