"""pipeline/run_pipeline.py 的轻量编排测试。

这里用 monkeypatch 替换重型 self-play/train/evaluate 阶段，验证 pipeline 的
路径、skip 参数、summary 落盘和阶段衔接。真实 MCTS/训练逻辑由前面批次测试覆盖。
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture()
def pipeline_module():
    return importlib.import_module("pipeline.run_pipeline")


@pytest.fixture()
def fast_pipeline_steps(monkeypatch, pipeline_module):
    def fake_selfplay_step(*, output_path, num_games=1, **_kwargs):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake selfplay data")
        return {
            "status": "ok",
            "num_games": num_games,
            "num_samples": 8,
            "output_path": str(path.resolve()),
        }

    def fake_train_step(*, checkpoint_dir, epochs=1, **_kwargs):
        ckpt = Path(checkpoint_dir) / "latest.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_bytes(b"fake checkpoint")
        return {
            "status": "ok",
            "epochs": epochs,
            "checkpoint_path": str(ckpt.resolve()),
            "final_loss": 1.0,
        }

    def fake_evaluate_step(*, output_path, games=1, opponent="random", **_kwargs):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "total_games": games,
            "player_a_win_rate": 0.0,
            "draw_rate": 1.0,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return {
            "status": "ok",
            "opponent": opponent,
            "games": games,
            "candidate_win_rate": 0.0,
            "draw_rate": 1.0,
            "summary": payload,
            "output_path": str(path.resolve()),
        }

    monkeypatch.setattr(pipeline_module, "run_selfplay_step", fake_selfplay_step)
    monkeypatch.setattr(pipeline_module, "run_train_step", fake_train_step)
    monkeypatch.setattr(pipeline_module, "run_evaluate_step", fake_evaluate_step)
    return pipeline_module


def test_run_pipeline_tiny_config_completes_and_saves_summary(tmp_path, fast_pipeline_steps):
    base_dir = tmp_path / "outputs"

    summary = fast_pipeline_steps.run_pipeline(
        {
            "base_output_dir": str(base_dir),
            "selfplay_games": 1,
            "num_simulations": 1,
            "train_epochs": 1,
            "batch_size": 8,
            "evaluate_games": 1,
            "verbose": False,
        }
    )

    assert summary["selfplay"]["status"] == "ok"
    assert summary["train"]["status"] == "ok"
    assert summary["evaluate"]["status"] == "ok"
    assert summary["promote"]["status"] == "skipped"

    summary_path = Path(summary["summary_path"])
    assert summary_path.exists()
    saved = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved["selfplay"]["status"] == "ok"
    assert saved["train"]["status"] == "ok"
    assert saved["evaluate"]["status"] == "ok"


@pytest.mark.parametrize(
    ("flag", "stage"),
    [
        ("skip_selfplay", "selfplay"),
        ("skip_train", "train"),
        ("skip_evaluate", "evaluate"),
    ],
)
def test_run_pipeline_skip_flags_are_respected(
    tmp_path,
    monkeypatch,
    pipeline_module,
    flag,
    stage,
):
    def fail_step(**_kwargs):
        raise AssertionError(f"{stage} step should be skipped")

    def ok_selfplay_step(**_kwargs):
        return {"status": "ok", "num_samples": 1}

    def ok_train_step(**_kwargs):
        return {"status": "ok", "final_loss": 1.0}

    def ok_evaluate_step(**_kwargs):
        return {"status": "ok", "candidate_win_rate": 0.0}

    monkeypatch.setattr(pipeline_module, "run_selfplay_step", ok_selfplay_step)
    monkeypatch.setattr(pipeline_module, "run_train_step", ok_train_step)
    monkeypatch.setattr(pipeline_module, "run_evaluate_step", ok_evaluate_step)
    monkeypatch.setattr(pipeline_module, f"run_{stage}_step", fail_step)

    summary = pipeline_module.run_pipeline(
        {
            "base_output_dir": str(tmp_path / flag),
            flag: True,
            "skip_selfplay": flag == "skip_selfplay",
            "skip_train": flag == "skip_train",
            "skip_evaluate": flag == "skip_evaluate",
            "verbose": False,
        }
    )

    assert summary[stage]["status"] == "skipped"
    assert Path(summary["summary_path"]).exists()


def test_save_pipeline_summary_creates_parent_directory(tmp_path, pipeline_module):
    path = tmp_path / "missing" / "pipeline_summary.json"
    payload = {
        "selfplay": {"status": "ok"},
        "train": {"status": "ok"},
        "evaluate": {"status": "ok"},
    }

    pipeline_module.save_pipeline_summary(payload, str(path))

    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == payload
