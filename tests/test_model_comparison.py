from __future__ import annotations


def test_run_model_comparison_outputs_json_and_markdown(tmp_path):
    from evaluate.model_comparison import run_model_comparison

    summary = run_model_comparison(
        games=1,
        rule_mode="basic",
        device="cuda",
        allow_cpu_fallback=True,
        output_json=str(tmp_path / "compare.json"),
        output_md=str(tmp_path / "compare.md"),
        checkpoints={"cnn": str(tmp_path / "missing.pt")},
        max_moves=12,
        num_simulations=2,
    )

    assert (tmp_path / "compare.json").exists()
    assert (tmp_path / "compare.md").exists()
    assert "random_vs_tactical" in summary["matches"]
    assert summary["is_smoke_test"] is True
    match = summary["matches"]["random_vs_tactical"]
    assert match["black_win_rate"] >= 0.0
    assert match["white_win_rate"] >= 0.0
    assert match["is_smoke_test"] is True
    assert match["timestamp"]
    assert summary["skipped_checkpoints"]
    assert "| Match |" in (tmp_path / "compare.md").read_text(encoding="utf-8")
