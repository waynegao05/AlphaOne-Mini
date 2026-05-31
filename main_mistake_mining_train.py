"""Mistake-mining fine-tune entry point."""

from __future__ import annotations

import argparse
import json
import os

import torch

from model.checkpoint import load_checkpoint
from model.model_factory import create_model
from train.mistake_mining import (
    DEFAULT_MISTAKE_DATA_PATH,
    DEFAULT_MISTAKE_STATS_PATH,
    build_mistake_training_dataset_v2,
    build_mistake_training_dataset_v3,
)
from train.mistake_replay_balancer import parse_ratio_spec, parse_weight_spec
from train.supervised_pretrain import train_policy_pretrain
from utils.device import describe_device, get_device


DEFAULT_OUTPUT_CHECKPOINT = os.path.join(
    "outputs", "checkpoints", "latest_advanced_mistake_tuned.pt"
)
DEFAULT_SUMMARY = os.path.join("outputs", "supervised", "mistake_mining_train_summary.json")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine mistakes against tactical teachers and fine-tune")
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--base-checkpoint")
    parser.add_argument("--teacher", choices=["tactical", "hybrid"], default="hybrid")
    parser.add_argument("--teachers", nargs="+", choices=["tactical", "hybrid"])
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--games-per-teacher", type=int)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--rule-mode", choices=["basic", "forbidden"], default="basic")
    parser.add_argument("--num-simulations", type=int, default=50)
    parser.add_argument("--max-moves", type=int, default=80)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--learning-rate", dest="lr", type=float)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--min-score-gap", type=float, default=5000.0)
    parser.add_argument("--center-replay-repeats", type=int, default=128)
    parser.add_argument("--include-center-replay", action="store_true")
    parser.add_argument("--include-curriculum-replay", action="store_true")
    parser.add_argument("--curriculum-data", default=os.path.join("outputs", "supervised", "tactical_curriculum_latest.npz"))
    parser.add_argument("--curriculum-replay-count", type=int, default=512)
    parser.add_argument("--oversample-critical", action="store_true")
    parser.add_argument("--critical-repeat", type=int, default=3)
    parser.add_argument("--teacher-balance")
    parser.add_argument("--reason-weights")
    parser.add_argument("--max-low-heuristic-ratio", type=float, default=0.25)
    parser.add_argument("--include-v1-tactical-draw-replay", action="store_true")
    parser.add_argument("--v1-checkpoint")
    parser.add_argument("--v2-checkpoint")
    parser.add_argument("--v1-draw-replay-games", type=int, default=20)
    parser.add_argument("--validation-teacher", choices=["tactical", "hybrid"])
    parser.add_argument("--validation-output", default=os.path.join("outputs", "supervised", "tactical_validation_set.npz"))
    parser.add_argument("--validation-holdout", type=int, default=96)
    parser.add_argument("--dataset-output", default=DEFAULT_MISTAKE_DATA_PATH)
    parser.add_argument("--output-dataset", dest="output_dataset")
    parser.add_argument("--stats-output", default=DEFAULT_MISTAKE_STATS_PATH)
    parser.add_argument("--output-checkpoint", default=DEFAULT_OUTPUT_CHECKPOINT)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY)
    parser.add_argument("--compare-from-checkpoints", nargs="*")
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
    student_checkpoint = args.base_checkpoint or args.student_checkpoint
    dataset_output = args.output_dataset or args.dataset_output
    teachers = args.teachers or [args.teacher]
    games_per_teacher = args.games_per_teacher or args.games
    use_v3 = bool(
        args.teacher_balance
        or args.reason_weights
        or args.include_v1_tactical_draw_replay
        or args.validation_teacher
    )
    print(
        f"[mistake-train] START device={describe_device(device)} teachers={teachers} "
        f"games_per_teacher={games_per_teacher} seeds={args.seeds} student={student_checkpoint}",
        flush=True,
    )
    if use_v3:
        collection = build_mistake_training_dataset_v3(
            student_checkpoint=student_checkpoint,
            teachers=teachers,
            games_per_teacher=games_per_teacher,
            seeds=args.seeds,
            rule_mode=args.rule_mode,
            num_simulations=args.num_simulations,
            device=str(device),
            output_path=dataset_output,
            max_moves=args.max_moves,
            min_score_gap=args.min_score_gap,
            teacher_balance=parse_ratio_spec(args.teacher_balance),
            reason_weights=parse_weight_spec(args.reason_weights),
            max_low_heuristic_ratio=float(args.max_low_heuristic_ratio),
            include_center_replay=bool(args.include_center_replay),
            center_replay_repeats=int(args.center_replay_repeats),
            include_curriculum_replay=bool(args.include_curriculum_replay),
            curriculum_data=args.curriculum_data,
            curriculum_replay_count=int(args.curriculum_replay_count),
            include_v1_tactical_draw_replay=bool(args.include_v1_tactical_draw_replay),
            v1_checkpoint=args.v1_checkpoint,
            v1_draw_replay_games=int(args.v1_draw_replay_games),
            validation_teacher=args.validation_teacher or "tactical",
            validation_output=args.validation_output,
            validation_holdout=int(args.validation_holdout),
            stats_path=args.stats_output,
        )
    else:
        collection = build_mistake_training_dataset_v2(
            student_checkpoint=student_checkpoint,
            teachers=teachers,
            games_per_teacher=games_per_teacher,
            seeds=args.seeds,
            rule_mode=args.rule_mode,
            num_simulations=args.num_simulations,
            device=str(device),
            output_path=dataset_output,
            max_moves=args.max_moves,
            min_score_gap=args.min_score_gap,
            include_center_replay=bool(args.include_center_replay),
            center_replay_repeats=int(args.center_replay_repeats),
            include_curriculum_replay=bool(args.include_curriculum_replay),
            curriculum_data=args.curriculum_data,
            curriculum_replay_count=int(args.curriculum_replay_count),
            oversample_critical=bool(args.oversample_critical),
            critical_repeat=int(args.critical_repeat),
            stats_path=args.stats_output,
        )
    final_samples = int(collection.get("final_samples", 0))
    if final_samples <= 0:
        raise RuntimeError("mistake mining produced no samples; training aborted")

    print(
        f"[mistake-train] START finetune samples={final_samples} "
        f"epochs={args.epochs} output={args.output_checkpoint}",
        flush=True,
    )
    model = create_model("advanced")
    load_checkpoint(model, student_checkpoint, device=str(device))
    checkpoint_dir = os.path.dirname(os.path.abspath(args.output_checkpoint))
    checkpoint_name = os.path.basename(args.output_checkpoint)
    history = train_policy_pretrain(
        model,
        dataset_output,
        checkpoint_dir=checkpoint_dir,
        checkpoint_name=checkpoint_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
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
        resume_from=None,
        scheduler_type="cosine",
        mixed_precision=bool(args.mixed_precision),
        augment_dataset=True,
        validation_data_path=collection.get("validation_output") if use_v3 else None,
    )
    extra_metadata = {
        "mistake_mining_version": "v3_teacher_balanced" if use_v3 else "v2",
        "student_checkpoint": os.path.abspath(student_checkpoint),
        "teachers": teachers,
        "seeds": args.seeds,
        "mistake_collection": collection,
        "history": history,
        "validation_results": history[-1].get("tactical_validation") if history else None,
        "compare_from_checkpoints": args.compare_from_checkpoints or [],
        "v1_checkpoint": args.v1_checkpoint,
        "v2_checkpoint": args.v2_checkpoint,
    }
    _update_checkpoint_metadata(args.output_checkpoint, extra_metadata)
    summary = {
        "student_checkpoint": student_checkpoint,
        "teachers": teachers,
        "seeds": args.seeds,
        "dataset_output": dataset_output,
        "output_checkpoint": args.output_checkpoint,
        "collection": collection,
        "history": history,
        "compare_from_checkpoints": args.compare_from_checkpoints or [],
        "version": "v3_teacher_balanced" if use_v3 else "v2",
    }
    _write_json(summary, args.summary_output)
    print(
        f"[mistake-train] DONE checkpoint={args.output_checkpoint} "
        f"samples={final_samples} final_loss={history[-1]['total_loss'] if history else 0.0:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
