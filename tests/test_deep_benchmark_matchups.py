"""Focused tests for the ``--matchups`` extensibility of the deep benchmark.

These tests deliberately avoid loading any heavy checkpoint — they rely on the
explicit ``matchups=[...]`` argument plus rule-based players that are always
constructed by :func:`run_deep_benchmark` (``random`` / ``tactical`` /
``hybrid``). When a strong matchup names a checkpoint that does not exist on
disk, the benchmark must:

1. Add the missing model_type to ``skipped_checkpoints``.
2. Add the matchup itself to ``skipped_matches``.
3. Still write the JSON + Markdown output.
4. Never raise.

Also covers ``main_deep_benchmark.parse_args`` for the matchup names listed in
the task spec (``latest_advanced_vs_tactical`` etc.) without actually running
the benchmark.
"""

from __future__ import annotations

import json
import os


# ---------------------------------------------------------------------------
# parse_matchups: pure split-and-alias logic, no heavy deps
# ---------------------------------------------------------------------------
def test_parse_matchups_splits_each_name_into_left_and_right():
    from evaluate.deep_benchmark import _parse_matchups

    names = [
        "latest_advanced_vs_tactical",
        "latest_advanced_vs_hybrid",
        "latest_advanced_vs_curriculum",
        "curriculum_vs_tactical",
        "curriculum_vs_hybrid",
        "pretrained_advanced_vs_curriculum",
        "pretrained_advanced_vs_tactical",
        "pretrained_advanced_vs_hybrid",
        "advanced_vs_pretrained_advanced",
        "advanced_vs_curriculum",
    ]
    parsed = _parse_matchups(names)
    assert [item[0] for item in parsed] == names
    # The left key is taken before the first ``_vs_`` token, and the right key
    # after it. ``advanced_vs_pretrained_advanced`` is the tricky case because it
    # contains an underscore on the right side — we expect a single split.
    assert ("advanced_vs_pretrained_advanced", "advanced", "pretrained_advanced") in parsed
    assert ("latest_advanced_vs_curriculum", "latest_advanced", "curriculum") in parsed
    # All strong-side keys should appear at least once
    left_keys = {item[1] for item in parsed}
    right_keys = {item[2] for item in parsed}
    assert {"latest_advanced", "curriculum", "pretrained_advanced", "advanced"} <= (left_keys | right_keys)
    assert {"tactical", "hybrid"} <= right_keys


def test_parse_matchups_rejects_string_without_vs():
    import pytest

    from evaluate.deep_benchmark import _parse_matchups

    with pytest.raises(ValueError):
        _parse_matchups(["latest_advanced-tactical"])


# ---------------------------------------------------------------------------
# argparse: main_deep_benchmark.py accepts the spec'd flags
# ---------------------------------------------------------------------------
def test_main_deep_benchmark_accepts_strong_matchup_args():
    from main_deep_benchmark import parse_args

    args = parse_args(
        [
            "--matchups",
            "latest_advanced_vs_tactical",
            "latest_advanced_vs_hybrid",
            "latest_advanced_vs_curriculum",
            "--advanced-checkpoint",
            "outputs/checkpoints/latest_advanced.pt",
            "--curriculum-advanced-checkpoint",
            "outputs/checkpoints/curriculum_advanced.pt",
            "--games",
            "20",
            "--num-simulations",
            "50",
            "--rule-mode",
            "basic",
            "--output",
            "outputs/evaluation/stage2_strong_matchups.json",
            "--output-md",
            "outputs/evaluation/stage2_strong_matchups.md",
        ]
    )
    assert args.matchups == [
        "latest_advanced_vs_tactical",
        "latest_advanced_vs_hybrid",
        "latest_advanced_vs_curriculum",
    ]
    assert args.advanced_checkpoint == "outputs/checkpoints/latest_advanced.pt"
    assert args.curriculum_advanced_checkpoint == "outputs/checkpoints/curriculum_advanced.pt"
    assert args.games == 20
    assert args.num_simulations == 50
    assert args.rule_mode == "basic"
    assert args.output == "outputs/evaluation/stage2_strong_matchups.json"
    assert args.output_md == "outputs/evaluation/stage2_strong_matchups.md"


# ---------------------------------------------------------------------------
# run_deep_benchmark with --matchups + missing checkpoints
# ---------------------------------------------------------------------------
def test_run_deep_benchmark_with_explicit_matchups_skips_missing_checkpoints(tmp_path):
    """Strong matchups whose checkpoints are absent must end up in skipped_matches."""
    from evaluate.deep_benchmark import run_deep_benchmark

    output = tmp_path / "stage2_strong_matchups.json"
    output_md = tmp_path / "stage2_strong_matchups.md"

    summary = run_deep_benchmark(
        games=1,
        device="cuda",
        allow_cpu_fallback=True,
        num_simulations=1,
        rule_mode="basic",
        output=str(output),
        output_md=str(output_md),
        checkpoints={
            "latest_advanced": str(tmp_path / "missing_latest_advanced.pt"),
            "curriculum": str(tmp_path / "missing_curriculum.pt"),
            "pretrained_advanced": str(tmp_path / "missing_pretrained.pt"),
        },
        matchups=[
            "tactical_vs_random",  # always runs (no checkpoint)
            "latest_advanced_vs_tactical",  # latest_advanced missing -> skipped
            "latest_advanced_vs_curriculum",  # both missing -> skipped
        ],
        max_moves=6,
    )

    # The always-runnable matchup must be present in matches
    assert "tactical_vs_random" in summary["matches"]
    # The two strong matchups must end up in skipped_matches (introduced by the
    # ``matchups not in required_player_keys`` guard).
    skipped = summary.get("skipped_matches", [])
    skipped_names = {item["match"] for item in skipped}
    assert {"latest_advanced_vs_tactical", "latest_advanced_vs_curriculum"} <= skipped_names

    # Their missing player keys are recorded explicitly
    for item in skipped:
        if item["match"] == "latest_advanced_vs_tactical":
            assert "latest_advanced" in item["missing"]
        elif item["match"] == "latest_advanced_vs_curriculum":
            assert "latest_advanced" in item["missing"] or "curriculum" in item["missing"]

    # Skipped checkpoints metadata
    sk_paths = {item["model_type"] for item in summary["skipped_checkpoints"]}
    assert "latest_advanced" in sk_paths
    assert "curriculum" in sk_paths

    # JSON / MD must be written
    assert output.exists()
    assert output_md.exists()

    with open(output, "r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    assert "matches" in loaded
    assert "tactical_vs_random" in loaded["matches"]

    md_text = output_md.read_text(encoding="utf-8")
    assert "tactical_vs_random" in md_text
    # The skipped checkpoints section is rendered in markdown
    assert "Skipped Checkpoints" in md_text


def test_run_deep_benchmark_only_runs_requested_matchups(tmp_path):
    """When ``matchups=[...]`` is given, default matchups must NOT also be run."""
    from evaluate.deep_benchmark import DEFAULT_MATCHUPS, run_deep_benchmark

    output = tmp_path / "benchmark.json"
    summary = run_deep_benchmark(
        games=1,
        device="cuda",
        allow_cpu_fallback=True,
        num_simulations=1,
        rule_mode="basic",
        output=str(output),
        matchups=["tactical_vs_random"],
        max_moves=4,
    )
    # Only the requested matchup runs
    assert set(summary["matches"].keys()) == {"tactical_vs_random"}
    # Sanity: the defaults include other entries we should NOT have inadvertently run
    default_names = {entry[0] for entry in DEFAULT_MATCHUPS}
    assert default_names - set(summary["matches"].keys())


def test_run_deep_benchmark_alternates_sides_and_reports_player_win_rate(tmp_path):
    """Win rate must be reported per player, not just per color."""
    from evaluate.deep_benchmark import run_deep_benchmark

    output = tmp_path / "benchmark.json"
    summary = run_deep_benchmark(
        games=2,
        device="cuda",
        allow_cpu_fallback=True,
        num_simulations=1,
        rule_mode="basic",
        output=str(output),
        matchups=["tactical_vs_random"],
        max_moves=12,
    )
    match = summary["matches"]["tactical_vs_random"]
    # Both player-side aggregates and color aggregates exist independently.
    assert "player_a_win_rate" in match
    assert "player_b_win_rate" in match
    assert "black_win_rate" in match
    assert "white_win_rate" in match
    # alternate_sides=True means each player took black exactly half the games.
    assert match["total_games"] == 2
    # avg_moves is present and a finite number
    assert isinstance(match["avg_moves"], (int, float))
