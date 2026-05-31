"""Checkpoint promotion helpers for Advanced model experiments."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Mapping

import torch


DEFAULT_BEST_ADVANCED_PATH = os.path.join("outputs", "checkpoints", "best_advanced.pt")


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_match(benchmark_summary: Mapping[str, Any], match_key: str) -> Mapping[str, Any]:
    matches = benchmark_summary.get("matches", {})
    if match_key not in matches:
        raise KeyError(f"benchmark match not found: {match_key}")
    return matches[match_key]


def _infer_opponent(match_key: str, match: Mapping[str, Any]) -> str:
    if match.get("player_b_name"):
        return str(match["player_b_name"])
    if "_vs_" in match_key:
        return match_key.split("_vs_", 1)[1]
    return "unknown"


def evaluate_promotion(
    benchmark_summary: Mapping[str, Any],
    match_key: str,
    threshold: float = 0.55,
    min_games: int = 20,
) -> dict[str, Any]:
    """Evaluate promotion eligibility from one benchmark match."""
    match = _get_match(benchmark_summary, match_key)
    total_games = int(match.get("total_games", benchmark_summary.get("games", 0)) or 0)
    win_rate = float(match.get("player_a_win_rate", 0.0) or 0.0)
    threshold = float(threshold)
    min_games = int(min_games)
    meets_threshold = win_rate >= threshold
    enough_games = total_games >= min_games
    return {
        "match_key": match_key,
        "benchmark_games": total_games,
        "win_rate": win_rate,
        "threshold": threshold,
        "min_games": min_games,
        "meets_threshold": meets_threshold,
        "enough_games": enough_games,
        "promoted": bool(meets_threshold and enough_games),
        "provisional_best": bool(meets_threshold and not enough_games),
        "opponent": _infer_opponent(match_key, match),
        "rule_mode": str(match.get("rule_mode", benchmark_summary.get("rule_mode", "basic"))),
    }


def promote_checkpoint_if_eligible(
    candidate_path: str,
    best_path: str = DEFAULT_BEST_ADVANCED_PATH,
    benchmark_summary: Mapping[str, Any] | None = None,
    match_key: str = "latest_advanced_vs_tactical",
    threshold: float = 0.55,
    min_games: int = 20,
) -> dict[str, Any]:
    """Promote a candidate checkpoint to ``best_advanced.pt`` when benchmarked.

    Benchmarks below ``min_games`` can only mark ``provisional_best`` and never
    write the best checkpoint.
    """
    if benchmark_summary is None:
        raise ValueError("benchmark_summary is required for promotion")
    if not os.path.exists(candidate_path):
        raise FileNotFoundError(f"candidate checkpoint not found: {candidate_path}")

    decision = evaluate_promotion(
        benchmark_summary,
        match_key=match_key,
        threshold=threshold,
        min_games=min_games,
    )
    decision["candidate_path"] = os.path.abspath(candidate_path)
    decision["best_path"] = os.path.abspath(best_path)

    if not decision["promoted"]:
        return decision

    try:
        state = torch.load(candidate_path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(candidate_path, map_location="cpu")
    metadata = dict(state.get("metadata", {}) or {})
    metadata.update(
        {
            "promoted_from": os.path.abspath(candidate_path),
            "benchmark_games": int(decision["benchmark_games"]),
            "win_rate": float(decision["win_rate"]),
            "opponent": decision["opponent"],
            "rule_mode": decision["rule_mode"],
            "promotion_threshold": float(threshold),
            "timestamp": _utc_timestamp(),
        }
    )
    state["metadata"] = metadata
    parent = os.path.dirname(os.path.abspath(best_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save(state, best_path)
    return decision


__all__ = [
    "DEFAULT_BEST_ADVANCED_PATH",
    "evaluate_promotion",
    "promote_checkpoint_if_eligible",
]
