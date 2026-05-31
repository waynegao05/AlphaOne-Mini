"""Hybrid-survival branch training entry point."""

from __future__ import annotations

import argparse
import json
import os

import torch

from model.checkpoint import load_checkpoint
from model.model_factory import create_model
from train.hybrid_survival import (
    DEFAULT_HYBRID_SURVIVAL_DATA_PATH,
    DEFAULT_HYBRID_SURVIVAL_STATS_PATH,
    DEFAULT_HYBRID_SURVIVAL_V2_DATA_PATH,
    DEFAULT_HYBRID_SURVIVAL_V2_METADATA_PATH,
    DEFAULT_HYBRID_SURVIVAL_V2_STATS_PATH,
    DEFAULT_HYBRID_SURVIVAL_V3_DATA_PATH,
    DEFAULT_HYBRID_SURVIVAL_V3_METADATA_PATH,
    DEFAULT_HYBRID_SURVIVAL_V3_STATS_PATH,
    HYBRID_SURVIVAL_REASON_WEIGHTS,
    build_hybrid_survival_dataset,
    build_hybrid_survival_v2_dataset,
    build_hybrid_survival_v3_forced_block_dataset,
)
from train.mistake_replay_balancer import parse_weight_spec
from train.supervised_pretrain import train_policy_pretrain
from utils.device import describe_device, get_device


DEFAULT_OUTPUT_CHECKPOINT = os.path.join(
    "outputs", "checkpoints", "latest_advanced_hybrid_survival_v2_from_v2.pt"
)
DEFAULT_V3_OUTPUT_CHECKPOINT = os.path.join(
    "outputs", "checkpoints", "latest_advanced_hybrid_survival_v3_forced_block.pt"
)
DEFAULT_SUMMARY = os.path.join("outputs", "supervised", "hybrid_survival_v2_train_summary.json")
DEFAULT_V3_SUMMARY = os.path.join("outputs", "supervised", "hybrid_survival_v3_forced_block_train_summary.json")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Hybrid-survival branch from v2")
    parser.add_argument("--version", choices=["v1", "v2", "v3"], default="v2")
    parser.add_argument(
        "--student-checkpoint",
        default=os.path.join("outputs", "checkpoints", "latest_advanced_mistake_v2_from_latest.pt"),
    )
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 7, 21])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", "--lr", dest="lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--rule-mode", choices=["basic", "forbidden"], default="basic")
    parser.add_argument("--num-simulations", type=int, default=50)
    parser.add_argument("--max-moves", type=int, default=80)
    parser.add_argument("--min-score-gap", type=float, default=5000.0)
    parser.add_argument("--include-center-replay", action="store_true", default=True)
    parser.add_argument("--no-center-replay", dest="include_center_replay", action="store_false")
    parser.add_argument("--center-replay-repeats", type=int, default=128)
    parser.add_argument("--include-curriculum-replay", action="store_true", default=True)
    parser.add_argument("--no-curriculum-replay", dest="include_curriculum_replay", action="store_false")
    parser.add_argument(
        "--curriculum-data",
        default=os.path.join("outputs", "supervised", "tactical_curriculum_latest.npz"),
    )
    parser.add_argument("--curriculum-replay-count", type=int, default=512)
    parser.add_argument(
        "--tactical-restoration-data",
        default=os.path.join("outputs", "supervised", "tactical_restoration_dataset.npz"),
    )
    parser.add_argument("--tactical-restoration-replay-count", type=int, default=128)
    parser.add_argument("--target-samples", type=int, default=2400)
    parser.add_argument("--reason-weights")
    parser.add_argument("--max-low-heuristic-ratio", type=float, default=0.20)
    parser.add_argument("--output-dataset")
    parser.add_argument("--metadata-output")
    parser.add_argument("--stats-output")
    parser.add_argument("--output-checkpoint")
    parser.add_argument("--summary-output")
    parser.add_argument("--mixed-precision", action="store_true")
    return parser.parse_args(argv)


def _write_json(payload: dict, path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _update_checkpoint_metadata(path: str, extra_metadata: dict) -> None:
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    metadata = dict(state.get("metadata", {}) or {})
    metadata.update(extra_metadata)
    state["metadata"] = metadata
    torch.save(state, path)


def main(argv=None) -> int:
    args = parse_args(argv)
    device = get_device(args.device, allow_cpu_fallback=args.allow_cpu_fallback)
    reason_weights = parse_weight_spec(args.reason_weights) if args.reason_weights else dict(HYBRID_SURVIVAL_REASON_WEIGHTS)
    print(
        f"[hybrid-survival-train] START device={describe_device(device)} "
        f"version={args.version} student={args.student_checkpoint} "
        f"teacher=hybrid games={args.games} seeds={args.seeds}",
        flush=True,
    )
    output_dataset = args.output_dataset or (
        DEFAULT_HYBRID_SURVIVAL_V3_DATA_PATH
        if args.version == "v3"
        else DEFAULT_HYBRID_SURVIVAL_V2_DATA_PATH
        if args.version == "v2"
        else DEFAULT_HYBRID_SURVIVAL_DATA_PATH
    )
    stats_output = args.stats_output or (
        DEFAULT_HYBRID_SURVIVAL_V3_STATS_PATH
        if args.version == "v3"
        else DEFAULT_HYBRID_SURVIVAL_V2_STATS_PATH
        if args.version == "v2"
        else DEFAULT_HYBRID_SURVIVAL_STATS_PATH
    )
    metadata_output = args.metadata_output or (
        DEFAULT_HYBRID_SURVIVAL_V3_METADATA_PATH
        if args.version == "v3"
        else DEFAULT_HYBRID_SURVIVAL_V2_METADATA_PATH
    )
    output_checkpoint = args.output_checkpoint or (
        DEFAULT_V3_OUTPUT_CHECKPOINT if args.version == "v3" else DEFAULT_OUTPUT_CHECKPOINT
    )
    summary_output = args.summary_output or (
        DEFAULT_V3_SUMMARY if args.version == "v3" else DEFAULT_SUMMARY
    )
    if args.version == "v2":
        collection = build_hybrid_survival_v2_dataset(
            student_checkpoint=args.student_checkpoint,
            games=int(args.games),
            seeds=args.seeds,
            rule_mode=args.rule_mode,
            num_simulations=int(args.num_simulations),
            device=str(device),
            output_path=output_dataset,
            metadata_path=metadata_output,
            stats_path=stats_output,
            max_moves=int(args.max_moves),
            min_score_gap=float(args.min_score_gap),
            target_samples=int(args.target_samples),
            curriculum_data=args.curriculum_data if args.include_curriculum_replay else None,
            curriculum_replay_count=int(args.curriculum_replay_count),
            center_replay_repeats=int(args.center_replay_repeats) if args.include_center_replay else 0,
        )
    elif args.version == "v3":
        collection = build_hybrid_survival_v3_forced_block_dataset(
            student_checkpoint=args.student_checkpoint,
            games=int(args.games),
            seeds=args.seeds,
            rule_mode=args.rule_mode,
            num_simulations=int(args.num_simulations),
            device=str(device),
            output_path=output_dataset,
            metadata_path=metadata_output,
            stats_path=stats_output,
            max_moves=int(args.max_moves),
            min_score_gap=float(args.min_score_gap),
            target_samples=int(args.target_samples),
            curriculum_data=args.curriculum_data if args.include_curriculum_replay else None,
            curriculum_replay_count=int(args.curriculum_replay_count),
            center_replay_repeats=int(args.center_replay_repeats) if args.include_center_replay else 0,
        )
    else:
        collection = build_hybrid_survival_dataset(
            student_checkpoint=args.student_checkpoint,
            games=int(args.games),
            seeds=args.seeds,
            rule_mode=args.rule_mode,
            num_simulations=int(args.num_simulations),
            device=str(device),
            output_path=output_dataset,
            max_moves=int(args.max_moves),
            min_score_gap=float(args.min_score_gap),
            include_center_replay=bool(args.include_center_replay),
            center_replay_repeats=int(args.center_replay_repeats),
            include_curriculum_replay=bool(args.include_curriculum_replay),
            curriculum_data=args.curriculum_data,
            curriculum_replay_count=int(args.curriculum_replay_count),
            tactical_restoration_data=args.tactical_restoration_data,
            tactical_restoration_replay_count=int(args.tactical_restoration_replay_count),
            reason_weights=reason_weights,
            max_low_heuristic_ratio=float(args.max_low_heuristic_ratio),
            stats_path=stats_output,
        )
    final_samples = int(collection.get("final_samples", 0))
    print(
        f"[hybrid-survival-train] START finetune samples={final_samples} "
        f"epochs={args.epochs} output={output_checkpoint}",
        flush=True,
    )
    model = create_model("advanced")
    load_checkpoint(model, args.student_checkpoint, device=str(device))
    checkpoint_dir = os.path.dirname(os.path.abspath(output_checkpoint))
    history = train_policy_pretrain(
        model,
        output_dataset,
        checkpoint_dir=checkpoint_dir,
        checkpoint_name=os.path.basename(output_checkpoint),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        device=str(device),
        grad_clip=5.0,
        use_auxiliary_loss=True,
        loss_weights={
            "policy": 2.0,
            "value": 1.0,
            "threat": 0.4,
            "forbidden": 0.2,
            "tactical_score": 0.1,
        },
        model_type="advanced",
        scheduler_type="cosine",
        mixed_precision=bool(args.mixed_precision),
        augment_dataset=True,
    )
    metadata = {
        "branch": "hybrid_survival_v3_forced_block"
        if args.version == "v3"
        else "hybrid_survival_v2"
        if args.version == "v2"
        else "hybrid_survival",
        "teacher": "hybrid",
        "student_checkpoint": os.path.abspath(args.student_checkpoint),
        "mistake_collection": collection,
        "history": history,
        "allow_promote": False,
    }
    _update_checkpoint_metadata(output_checkpoint, metadata)
    summary = {
        "branch": "hybrid_survival_v3_forced_block"
        if args.version == "v3"
        else "hybrid_survival_v2"
        if args.version == "v2"
        else "hybrid_survival",
        "student_checkpoint": args.student_checkpoint,
        "output_dataset": output_dataset,
        "metadata_output": metadata_output if args.version in {"v2", "v3"} else None,
        "output_checkpoint": output_checkpoint,
        "collection": collection,
        "history": history,
        "promote": False,
    }
    _write_json(summary, summary_output)
    print(
        f"[hybrid-survival-train] DONE checkpoint={output_checkpoint} "
        f"samples={final_samples} final_loss={history[-1]['total_loss'] if history else 0.0:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
