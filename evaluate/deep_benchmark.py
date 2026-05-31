"""Lightweight benchmark orchestration for deep-model checkpoints."""

from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime, timezone
from typing import Mapping

import numpy as np
import torch

from engine.tactical_player import TacticalPlayer
from engine.hybrid_player import HybridPlayer
from engine.neural_guarded_player import NeuralGuardedPlayer
from evaluate.arena import run_match
from evaluate.players import ModelMCTSPlayer, RandomPlayer
from model.checkpoint import load_checkpoint
from model.policy_value_net import PolicyValueNet
from model.model_factory import create_model_from_metadata
from utils.device import describe_device, get_device


DEFAULT_DEEP_BENCHMARK_OUTPUT = os.path.join(
    "outputs", "evaluation", "deep_benchmark_latest.json"
)

DEFAULT_MATCHUPS = [
    ("TacticalPlayer_vs_RandomPlayer", "tactical", "random"),
    ("HybridPlayer_vs_RandomPlayer", "hybrid", "random"),
    ("pretrained_advanced_vs_random", "pretrained_advanced", "random"),
    ("pretrained_advanced_vs_tactical", "pretrained_advanced", "tactical"),
    ("pretrained_advanced_vs_hybrid", "pretrained_advanced", "hybrid"),
    ("latest_advanced_vs_pretrained_advanced", "latest_advanced", "pretrained_advanced"),
    ("tactical_vs_hybrid", "tactical", "hybrid"),
    ("advanced_vs_pretrained_advanced", "advanced", "pretrained_advanced"),
    ("cnn_vs_random", "cnn", "random"),
    ("resnet_vs_random", "resnet", "random"),
    ("advanced_vs_random", "advanced", "random"),
]


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _set_global_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_for_key(seed: int, key: str) -> int:
    return int(seed) + sum((index + 1) * ord(char) for index, char in enumerate(str(key)))


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


def _load_model_player(
    name: str,
    checkpoint_path: str,
    device: str,
    num_simulations: int,
):
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=device)
    metadata = state.get("metadata", {})
    fallback = name if name in {"cnn", "resnet", "advanced"} else "advanced"
    model = create_model_from_metadata(metadata, fallback_model_type=fallback)
    load_checkpoint(model, checkpoint_path, device=device)
    model.eval()
    return ModelMCTSPlayer(
        model=model,
        num_simulations=num_simulations,
        device=device,
        name=f"{name}_ModelMCTSPlayer",
    )


def _write_markdown(summary: dict, path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    lines = [
        "# Stage Benchmark",
        "",
        "This benchmark alternates sides and reports player win rates, not just",
        "black/white win rates. Twenty games is still a small comparison, not a",
        "stable rating.",
        "",
        f"- Games per match: `{summary['games']}`",
        f"- Rule mode: `{summary['rule_mode']}`",
        f"- Simulations: `{summary['num_simulations']}`",
        f"- Device: `{summary['device']}`",
        "",
        "| Match | Games | Player A Win Rate | Player B Win Rate | Draw Rate | Avg Moves |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in summary["matches"].items():
        lines.append(
            f"| {name} | {result.get('total_games', 0)} | "
            f"{result.get('player_a_win_rate', 0.0):.3f} | "
            f"{result.get('player_b_win_rate', 0.0):.3f} | "
            f"{result.get('draw_rate', 0.0):.3f} | "
            f"{result.get('avg_moves', 0.0):.1f} |"
        )
    if summary.get("skipped_checkpoints"):
        lines.extend(["", "## Skipped Checkpoints"])
        for item in summary["skipped_checkpoints"]:
            lines.append(f"- `{item.get('model_type')}`: `{item.get('path')}`")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _parse_matchups(matchups: list[str] | tuple[str, ...] | None) -> list[tuple[str, str, str]]:
    if not matchups:
        return list(DEFAULT_MATCHUPS)
    parsed = []
    aliases = {
        "tacticalplayer": "tactical",
        "randomplayer": "random",
        "modelmctsplayer": "advanced",
        "curriculum_advanced": "curriculum",
        "latest": "latest_advanced",
        "v2": "mistake_v2",
        "v3": "mistake_v3",
        "mistake_tuned_advanced": "mistake_tuned",
        "mistake_v2_from_latest": "mistake_v2",
        "mistake_v2_from_mistake_tuned": "mistake_v2",
        "mistake_v2_advanced": "mistake_v2",
        "mistake_v3": "mistake_v3",
        "mistake_v3_teacher_balanced": "mistake_v3",
        "mistake_v3_advanced": "mistake_v3",
        "hybrid_survival_from_v2": "hybrid_survival",
        "hybrid_survival_advanced": "hybrid_survival",
        "hybrid_survival_v2_from_v2": "hybrid_survival_v2",
        "hybrid_survival_v2_advanced": "hybrid_survival_v2",
        "hybrid_survival_v3": "hybrid_survival_v3",
        "hybrid_survival_v3_forced_block": "hybrid_survival_v3",
        "hybrid_survival_v3_from_v2": "hybrid_survival_v3",
        "hybrid_survival_v3_advanced": "hybrid_survival_v3",
        "tactical_restoration_from_v1": "tactical_restoration_v1",
        "tactical_restoration_from_curriculum": "tactical_restoration_curriculum",
        "neuralguarded": "neural_guarded",
        "neural_guarded_player": "neural_guarded",
        "neural_guarded_full": "neural_guarded_full",
        "neural_guarded_no_guardrail": "neural_guarded_no_guardrail",
        "neural_guarded_no_tactical_specialist": "neural_guarded_no_tactical_specialist",
        "neural_guarded_no_v2": "neural_guarded_no_v2",
        "neural_guarded_no_hybrid_fallback": "neural_guarded_no_hybrid_fallback",
        "neural_guarded_guardrail_only": "neural_guarded_guardrail_only",
        "neural_guarded_hybrid_only": "neural_guarded_hybrid_only",
        "neural_guarded_conservative_fallback": "neural_guarded_conservative_fallback",
        "neural_guarded_fallback_off": "neural_guarded_fallback_off",
        "neural_guarded_aggressive_fallback": "neural_guarded_aggressive_fallback",
    }
    for matchup in matchups:
        raw = str(matchup)
        if "_vs_" not in raw:
            raise ValueError(f"invalid matchup {raw!r}; expected '<player_a>_vs_<player_b>'")
        left, right = raw.split("_vs_", 1)
        left_key = aliases.get(left.lower(), left.lower())
        right_key = aliases.get(right.lower(), right.lower())
        parsed.append((raw, left_key, right_key))
    return parsed


def run_deep_benchmark(
    games: int = 20,
    device: str = "cuda",
    allow_cpu_fallback: bool = False,
    num_simulations: int = 50,
    rule_mode: str = "basic",
    output: str = DEFAULT_DEEP_BENCHMARK_OUTPUT,
    output_md: str | None = None,
    checkpoints: Mapping[str, str] | None = None,
    matchups: list[str] | tuple[str, ...] | None = None,
    max_moves: int = 80,
    seed: int = 2026,
    neural_guarded_fallback_mode: str = "normal",
) -> dict:
    """Run a smoke-scale benchmark and save a JSON summary."""
    _set_global_seed(int(seed))
    resolved = get_device(device, allow_cpu_fallback=allow_cpu_fallback)
    checkpoints = dict(checkpoints or {})
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
        "note": "Smoke-scale benchmark; not a rating.",
    }

    requested_matches = _parse_matchups(matchups)
    required_player_keys = {
        key for _, player_a, player_b in requested_matches for key in (player_a, player_b)
    }

    players: dict[str, object] = {
        "random": RandomPlayer(seed=int(seed), name="RandomPlayer"),
        "tactical": TacticalPlayer(rule_mode=rule_mode, name="TacticalPlayer"),
    }
    _set_global_seed(_seed_for_key(seed, "hybrid"))
    players["hybrid"] = HybridPlayer(
        model=PolicyValueNet(),
        num_simulations=max(1, min(int(num_simulations), 10)),
        device=str(resolved),
        rule_mode=rule_mode,
        name="HybridPlayer",
    )
    player_meta: dict[str, dict] = {
        "random": {"model_type": "random", "checkpoint_path": None},
        "tactical": {"model_type": "tactical", "checkpoint_path": None},
        "hybrid": {"model_type": "hybrid", "checkpoint_path": None},
    }

    neural_guarded_variants = {
        "neural_guarded": {},
        "neural_guarded_full": {},
        "neural_guarded_conservative_fallback": {"fallback_mode": "conservative"},
        "neural_guarded_fallback_off": {"fallback_mode": "off"},
        "neural_guarded_aggressive_fallback": {"fallback_mode": "aggressive"},
        "neural_guarded_no_guardrail": {"use_tactical_guardrail": False},
        "neural_guarded_no_tactical_specialist": {"enable_tactical_specialist": False},
        "neural_guarded_no_v2": {"enable_v2_policy": False},
        "neural_guarded_no_hybrid_fallback": {"use_hybrid_fallback": False},
        "neural_guarded_guardrail_only": {
            "enable_tactical_specialist": False,
            "enable_v2_policy": False,
            "use_hybrid_fallback": False,
            "use_tactical_fallback": False,
        },
        "neural_guarded_hybrid_only": {
            "use_tactical_guardrail": False,
            "enable_tactical_specialist": False,
            "enable_v2_policy": False,
            "use_hybrid_fallback": True,
        },
    }
    for key, variant_kwargs in neural_guarded_variants.items():
        if key not in required_player_keys:
            continue
        _set_global_seed(_seed_for_key(seed, key))
        if key in {"neural_guarded", "neural_guarded_full"}:
            variant_kwargs = dict(variant_kwargs)
            variant_kwargs.setdefault("fallback_mode", neural_guarded_fallback_mode)
        tactical_checkpoint = None if not variant_kwargs.get("enable_tactical_specialist", True) else checkpoints.get("tactical_restoration_curriculum")
        v2_checkpoint = None if not variant_kwargs.get("enable_v2_policy", True) else checkpoints.get("mistake_v2")
        players[key] = NeuralGuardedPlayer(
            tactical_checkpoint=tactical_checkpoint,
            v2_checkpoint=v2_checkpoint,
            device=str(resolved),
            rule_mode=rule_mode,
            num_simulations=int(num_simulations),
            name="NeuralGuardedPlayer" if key == "neural_guarded" else key,
            **variant_kwargs,
        )
        player_meta[key] = {
            "model_type": key,
            "checkpoint_path": {
                "tactical_checkpoint": tactical_checkpoint,
                "v2_checkpoint": v2_checkpoint,
            },
            "fallback_mode": variant_kwargs.get("fallback_mode", "normal"),
        }

    for model_type, path in dict(checkpoints).items():
        if matchups and model_type not in required_player_keys:
            continue
        if not path or not os.path.exists(path):
            summary["skipped_checkpoints"].append({"model_type": model_type, "path": path})
            continue
        players[model_type] = _load_model_player(
            model_type, path, str(resolved), int(num_simulations)
        )
        player_meta[model_type] = {"model_type": model_type, "checkpoint_path": path}

    for match_name, player_a_key, player_b_key in requested_matches:
        if player_a_key not in players or player_b_key not in players:
            summary.setdefault("skipped_matches", []).append(
                {
                    "match": match_name,
                    "missing": [
                        key
                        for key in (player_a_key, player_b_key)
                        if key not in players
                    ],
                }
            )
            continue
        print(
            f"[benchmark] START match={match_name} games={games} "
            f"num_simulations={num_simulations} time={datetime.now().strftime('%H:%M:%S')}",
            flush=True,
        )
        started = time.time()
        match_summary, _ = run_match(
            players[player_a_key],
            players[player_b_key],
            num_games=int(games),
            alternate_sides=True,
            max_moves=max_moves,
        )
        print(
            f"[benchmark] DONE match={match_name} "
            f"elapsed={time.time() - started:.1f}s time={datetime.now().strftime('%H:%M:%S')}",
            flush=True,
        )
        meta = player_meta.get(player_a_key, {})
        enriched = _enrich_match(
            match_summary,
            games=int(games),
            rule_mode=rule_mode,
            num_simulations=int(num_simulations),
            device=device_desc,
            model_type=str(meta.get("model_type", player_a_key)),
            checkpoint_path=meta.get("checkpoint_path"),
            timestamp=timestamp,
        )
        if "fallback_mode" in meta:
            enriched["fallback_mode"] = meta["fallback_mode"]
        summary["matches"][match_name] = enriched

    parent = os.path.dirname(os.path.abspath(output))
    if parent:
        os.makedirs(parent, exist_ok=True)
    if output_md is None:
        output_md = os.path.splitext(output)[0] + ".md"
    summary["markdown_output"] = output_md
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    _write_markdown(summary, output_md)
    return summary


__all__ = ["DEFAULT_DEEP_BENCHMARK_OUTPUT", "run_deep_benchmark"]
