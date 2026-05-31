from __future__ import annotations

import json


def _base_summary() -> dict:
    return {
        "experiment_name": "demo",
        "config": {
            "model_type": "advanced",
            "batch_size": 16,
            "use_augmentation": True,
            "use_auxiliary_loss": True,
        },
        "device": "cpu",
        "model_type": "advanced",
        "parameter_count": 999,
        "data": {"num_samples": 0},
        "paths": {
            "pretrained_checkpoint": "outputs/checkpoints/pretrained_advanced.pt",
            "latest_checkpoint": "outputs/checkpoints/latest_advanced.pt",
        },
        "selfplay": {"status": "completed", "num_samples": 128},
        "pretrain": {"status": "skipped", "history": []},
        "finetune": {"status": "skipped", "history": []},
        "benchmark": {"matches": {}},
    }


def _render(tmp_path, summary):
    from tools.generate_experiment_report import generate_experiment_report

    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    output = tmp_path / "report.md"
    text = generate_experiment_report(str(summary_path), str(output))
    return text, output


def test_generate_experiment_report_from_summary(tmp_path):
    from tools.generate_experiment_report import generate_experiment_report

    summary = {
        "experiment_name": "demo",
        "config": {"model_type": "advanced", "batch_size": 4, "use_augmentation": True},
        "device": "cpu",
        "model_type": "advanced",
        "parameter_count": 123,
        "data": {"num_samples": 8},
        "pretrain": {"history": [{"epoch": 1, "total_loss": 1.0, "policy_loss": 0.8}]},
        "benchmark": {"matches": {"A_vs_B": {"player_a_win_rate": 1.0, "draw_rate": 0.0, "avg_moves": 10}}},
        "paths": {},
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    output = tmp_path / "report.md"

    text = generate_experiment_report(str(summary_path), str(output))

    assert output.exists()
    assert "demo" in text
    assert "not prove competition-level strength" in text


def test_generate_experiment_report_renders_finetune_history_and_paths(tmp_path):
    from tools.generate_experiment_report import generate_experiment_report

    summary = {
        "experiment_name": "stage2",
        "config": {
            "model_type": "advanced",
            "batch_size": 32,
            "use_augmentation": True,
            "use_auxiliary_loss": True,
            "resume_from": "outputs/checkpoints/curriculum_advanced.pt",
        },
        "device": "cuda:0",
        "model_type": "advanced",
        "parameter_count": 123,
        "data": {"num_samples": 16},
        "paths": {
            "pretrained_checkpoint": "outputs/checkpoints/pretrained_advanced.pt",
            "latest_checkpoint": "outputs/checkpoints/latest_advanced.pt",
        },
        "selfplay": {"status": "completed", "num_samples": 64},
        "pretrain": {"status": "skipped", "history": []},
        "finetune": {
            "status": "completed",
            "history": [
                {
                    "epoch": 1,
                    "total_loss": 5.5028,
                    "policy_loss": 4.1819,
                    "value_loss": 1.3210,
                    "threat_loss": 0.0,
                    "forbidden_loss": 0.0,
                },
                {
                    "epoch": 2,
                    "total_loss": 3.6850,
                    "policy_loss": 2.7413,
                    "value_loss": 0.9437,
                    "threat_loss": 0.0,
                    "forbidden_loss": 0.0,
                },
            ],
        },
        "benchmark": {"matches": {}},
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    output = tmp_path / "report.md"

    text = generate_experiment_report(str(summary_path), str(output))

    assert "Finetune History" in text
    assert "5.5028" in text
    assert "3.6850" in text
    assert "4.1819" in text
    assert "0.9437" in text
    assert "outputs/checkpoints/latest_advanced.pt" in text
    assert "outputs/checkpoints/curriculum_advanced.pt" in text
    assert "Self-play samples: `64`" in text


def test_generate_experiment_report_renders_pretrain_only_history(tmp_path):
    """Only pretrain history exists; report must show it and not the placeholder."""
    summary = _base_summary()
    summary["experiment_name"] = "stage1_only"
    summary["pretrain"] = {
        "status": "completed",
        "history": [
            {
                "epoch": 1,
                "total_loss": 1.234,
                "policy_loss": 0.901,
                "value_loss": 0.333,
                "threat_loss": 0.0,
                "forbidden_loss": 0.0,
            }
        ],
    }
    # finetune stays empty -> only pretrain section should have a table

    text, output = _render(tmp_path, summary)

    assert output.exists()
    assert "Pretrain History" in text
    assert "1.2340" in text
    assert "0.9010" in text
    # The "No training history was recorded" placeholder must NOT appear when at least one history is present.
    assert "No training history was recorded" not in text
    # Pretrained checkpoint path should appear in artifacts section
    assert "outputs/checkpoints/pretrained_advanced.pt" in text
    # Self-play samples is taken from the summary, not hard-coded
    assert "Self-play samples: `128`" in text


def test_generate_experiment_report_renders_both_histories(tmp_path):
    """Both pretrain and finetune histories must render side by side."""
    summary = _base_summary()
    summary["experiment_name"] = "stage1_plus_stage2"
    summary["pretrain"] = {
        "status": "completed",
        "history": [
            {
                "epoch": 1,
                "total_loss": 0.5500,
                "policy_loss": 0.3000,
                "value_loss": 0.2500,
                "threat_loss": 0.0,
                "forbidden_loss": 0.0,
            }
        ],
    }
    summary["finetune"] = {
        "status": "completed",
        "history": [
            {
                "epoch": 1,
                "total_loss": 5.5028,
                "policy_loss": 4.1819,
                "value_loss": 1.3210,
                "threat_loss": 0.0,
                "forbidden_loss": 0.0,
            }
        ],
    }

    text, _ = _render(tmp_path, summary)

    assert "Pretrain History" in text
    assert "Finetune History" in text
    assert "0.5500" in text
    assert "5.5028" in text
    assert "No training history was recorded" not in text
    assert "No pretrain history was recorded" not in text
    assert "No finetune history was recorded" not in text


def test_generate_experiment_report_shows_placeholder_when_no_history(tmp_path):
    """When NEITHER pretrain nor finetune ran, the placeholder is the right message."""
    summary = _base_summary()
    # both pretrain and finetune are skipped with empty history -> placeholder
    text, _ = _render(tmp_path, summary)
    assert "No training history was recorded" in text


def test_generate_experiment_report_handles_missing_or_none_sections(tmp_path):
    """``pretrain`` / ``finetune`` may be missing or explicitly ``None``."""
    summary = _base_summary()
    summary["pretrain"] = None  # type: ignore[assignment]
    summary.pop("finetune")
    summary["finetune"] = {
        "status": "completed",
        "history": [
            {
                "epoch": 1,
                "total_loss": 2.0,
                "policy_loss": 1.5,
                "value_loss": 0.5,
                "threat_loss": 0.0,
                "forbidden_loss": 0.0,
            }
        ],
    }
    text, _ = _render(tmp_path, summary)
    # Must not crash and must still render finetune history
    assert "Finetune History" in text
    assert "2.0000" in text


def test_generate_experiment_report_reports_selfplay_samples_and_checkpoints(tmp_path):
    """Artifacts section must surface both selfplay count and latest checkpoint."""
    summary = _base_summary()
    summary["selfplay"] = {"status": "completed", "num_samples": 378}
    summary["paths"]["latest_checkpoint"] = "outputs/checkpoints/latest_advanced.pt"
    summary["finetune"] = {
        "status": "completed",
        "history": [
            {
                "epoch": 1,
                "total_loss": 1.0,
                "policy_loss": 0.6,
                "value_loss": 0.4,
                "threat_loss": 0.0,
                "forbidden_loss": 0.0,
            }
        ],
    }

    text, _ = _render(tmp_path, summary)
    assert "Self-play samples: `378`" in text
    assert "outputs/checkpoints/latest_advanced.pt" in text
