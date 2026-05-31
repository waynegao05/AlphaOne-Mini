"""Model/player comparison utilities with JSON and Markdown outputs."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Mapping

from engine.hybrid_player import HybridPlayer
from engine.tactical_player import TacticalPlayer
from evaluate.arena import run_match
from evaluate.deep_benchmark import _load_model_player
from evaluate.players import RandomPlayer
from model.policy_value_net import PolicyValueNet
from utils.device import describe_device, get_device


DEFAULT_COMPARE_JSON = os.path.join("outputs", "evaluation", "model_compare_latest.json")
DEFAULT_COMPARE_MD = os.path.join("outputs", "evaluation", "model_compare_latest.md")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _enrich_match(
    match: dict,
    *,
    games: int,
    rule_mode: str,
    num_simulations: int,
    device: str,
    model_type: str,
    checkpoint_path: str | None,
    timestamp: str,
) -> dict:
    total = int(match.get("total_games", games) or 0)
    black_wins = int(match.get("black_wins", 0) or 0)
    white_wins = int(match.get("white_wins", 0) or 0)
    enriched = dict(match)
    enriched.update(
        {
            "black_win_rate": black_wins / total if total else 0.0,
            "white_win_rate": white_wins / total if total else 0.0,
            "rule_mode": rule_mode,
            "num_simulations": int(num_simulations),
            "device": device,
            "model_type": model_type,
            "checkpoint_path": checkpoint_path,
            "is_smoke_test": int(games) < 20,
            "timestamp": timestamp,
        }
    )
    return enriched


def _write_json(summary: dict, path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def _markdown_table(summary: dict) -> str:
    lines = [
        "# Model Comparison",
        "",
        "Small-game smoke comparisons do not prove competition-level strength.",
        "",
        "| Match | Games | Player A Win Rate | Draw Rate | Avg Moves |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, result in summary["matches"].items():
        lines.append(
            f"| {name} | {result.get('total_games', 0)} | "
            f"{result.get('player_a_win_rate', 0.0):.3f} | "
            f"{result.get('draw_rate', 0.0):.3f} | "
            f"{result.get('avg_moves', 0.0):.1f} |"
        )
    if summary["skipped_checkpoints"]:
        lines.extend(["", "## Skipped Checkpoints"])
        for item in summary["skipped_checkpoints"]:
            lines.append(f"- `{item['model_type']}`: `{item['path']}`")
    return "\n".join(lines) + "\n"


def run_model_comparison(
    games: int = 10,
    rule_mode: str = "basic",
    device: str = "cuda",
    allow_cpu_fallback: bool = False,
    output_json: str = DEFAULT_COMPARE_JSON,
    output_md: str = DEFAULT_COMPARE_MD,
    checkpoints: Mapping[str, str] | None = None,
    max_moves: int = 80,
    num_simulations: int = 50,
) -> dict:
    """Compare baseline players and available model checkpoints."""
    resolved = get_device(device, allow_cpu_fallback=allow_cpu_fallback)
    timestamp = _timestamp()
    device_desc = describe_device(resolved)
    summary = {
        "games": int(games),
        "rule_mode": rule_mode,
        "device": device_desc,
        "num_simulations": int(num_simulations),
        "is_smoke_test": int(games) < 20,
        "timestamp": timestamp,
        "matches": {},
        "skipped_checkpoints": [],
        "note": "Smoke-scale comparison; not a rating.",
    }

    random_a = RandomPlayer(seed=0, name="RandomPlayer")
    tactical = TacticalPlayer(name="TacticalPlayer", rule_mode=rule_mode)
    match, _ = run_match(
        random_a,
        tactical,
        num_games=int(games),
        alternate_sides=True,
        max_moves=max_moves,
    )
    summary["matches"]["random_vs_tactical"] = _enrich_match(
        match,
        games=int(games),
        rule_mode=rule_mode,
        num_simulations=int(num_simulations),
        device=device_desc,
        model_type="tactical",
        checkpoint_path=None,
        timestamp=timestamp,
    )

    hybrid = HybridPlayer(
        model=PolicyValueNet(),
        num_simulations=max(1, min(int(num_simulations), 10)),
        device=str(resolved),
        rule_mode=rule_mode,
        name="HybridPlayer",
    )
    match, _ = run_match(
        hybrid,
        RandomPlayer(seed=1, name="RandomPlayer"),
        num_games=int(games),
        alternate_sides=True,
        max_moves=max_moves,
    )
    summary["matches"]["hybrid_vs_random"] = _enrich_match(
        match,
        games=int(games),
        rule_mode=rule_mode,
        num_simulations=int(num_simulations),
        device=device_desc,
        model_type="hybrid",
        checkpoint_path=None,
        timestamp=timestamp,
    )

    default_checkpoints = {
        "cnn": os.path.join("outputs", "checkpoints", "latest.pt"),
        "resnet": os.path.join("outputs", "checkpoints", "latest_resnet.pt"),
        "advanced": os.path.join("outputs", "checkpoints", "latest_advanced.pt"),
        "pretrained_advanced": os.path.join("outputs", "checkpoints", "pretrained_advanced.pt"),
    }
    if checkpoints:
        default_checkpoints.update(dict(checkpoints))

    for model_type, path in default_checkpoints.items():
        if not path or not os.path.exists(path):
            summary["skipped_checkpoints"].append({"model_type": model_type, "path": path})
            continue
        player = _load_model_player(model_type, path, str(resolved), int(num_simulations))
        match, _ = run_match(
            player,
            RandomPlayer(seed=2, name="RandomPlayer"),
            num_games=int(games),
            alternate_sides=True,
            max_moves=max_moves,
        )
        summary["matches"][f"{model_type}_vs_random"] = _enrich_match(
            match,
            games=int(games),
            rule_mode=rule_mode,
            num_simulations=int(num_simulations),
            device=device_desc,
            model_type=model_type,
            checkpoint_path=path,
            timestamp=timestamp,
        )

    _write_json(summary, output_json)
    md_text = _markdown_table(summary)
    parent = os.path.dirname(os.path.abspath(output_md))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_md, "w", encoding="utf-8") as handle:
        handle.write(md_text)
    return summary


__all__ = [
    "DEFAULT_COMPARE_JSON",
    "DEFAULT_COMPARE_MD",
    "run_model_comparison",
]
