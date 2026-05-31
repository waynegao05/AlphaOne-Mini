"""Train Tactical-restoration candidate branches."""

from __future__ import annotations

import argparse
import json
import os

import torch

from model.checkpoint import load_checkpoint
from model.model_factory import create_model
from train.supervised_pretrain import train_policy_pretrain
from train.tactical_restoration import (
    DEFAULT_REASON_WEIGHTS,
    DEFAULT_TACTICAL_RESTORATION_DATA_PATH,
    DEFAULT_TACTICAL_VALIDATION_PATH,
    build_tactical_restoration_dataset,
)
from utils.device import describe_device, get_device


DEFAULT_V1_CHECKPOINT = os.path.join("outputs", "checkpoints", "latest_advanced_mistake_tuned.pt")
DEFAULT_CURRICULUM_CHECKPOINT = os.path.join("outputs", "checkpoints", "curriculum_advanced.pt")
DEFAULT_OUTPUT_V1 = os.path.join(
    "outputs", "checkpoints", "latest_advanced_tactical_restoration_from_v1.pt"
)
DEFAULT_OUTPUT_CURRICULUM = os.path.join(
    "outputs", "checkpoints", "latest_advanced_tactical_restoration_from_curriculum.pt"
)
DEFAULT_SUMMARY = os.path.join("outputs", "supervised", "tactical_restoration_train_summary.json")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Tactical-restoration branches")
    parser.add_argument("--v1-checkpoint", default=DEFAULT_V1_CHECKPOINT)
    parser.add_argument("--curriculum-checkpoint", default=DEFAULT_CURRICULUM_CHECKPOINT)
    parser.add_argument("--output-v1-checkpoint", default=DEFAULT_OUTPUT_V1)
    parser.add_argument("--output-curriculum-checkpoint", default=DEFAULT_OUTPUT_CURRICULUM)
    parser.add_argument("--output-dataset", default=DEFAULT_TACTICAL_RESTORATION_DATA_PATH)
    parser.add_argument("--validation-output", default=DEFAULT_TACTICAL_VALIDATION_PATH)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY)
    parser.add_argument("--curriculum-data", default=os.path.join("outputs", "supervised", "tactical_curriculum_latest.npz"))
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--rule-mode", choices=["basic", "forbidden"], default="basic")
    parser.add_argument("--num-simulations", type=int, default=50)
    parser.add_argument("--max-moves", type=int, default=80)
    parser.add_argument("--defense-repeats", type=int, default=24)
    parser.add_argument("--curriculum-replay-count", type=int, default=512)
    parser.add_argument("--center-replay-repeats", type=int, default=128)
    parser.add_argument("--max-low-heuristic-ratio", type=float, default=0.15)
    parser.add_argument("--mixed-precision", action="store_true")
    return parser.parse_args(argv)


def _write_json(payload: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _update_checkpoint(path: str, extra: dict) -> None:
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    metadata = dict(state.get("metadata", {}) or {})
    metadata.update(extra)
    state["metadata"] = metadata
    torch.save(state, path)


def _train_branch(
    *,
    base_checkpoint: str,
    output_checkpoint: str,
    data_path: str,
    validation_path: str,
    device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    mixed_precision: bool,
    dataset_summary: dict,
    branch_name: str,
) -> list[dict]:
    model = create_model("advanced")
    load_checkpoint(model, base_checkpoint, device=device)
    checkpoint_dir = os.path.dirname(os.path.abspath(output_checkpoint))
    checkpoint_name = os.path.basename(output_checkpoint)
    print(
        f"[restore-train] START branch={branch_name} base={base_checkpoint} "
        f"output={output_checkpoint}",
        flush=True,
    )
    history = train_policy_pretrain(
        model,
        data_path,
        checkpoint_dir=checkpoint_dir,
        checkpoint_name=checkpoint_name,
        epochs=epochs,
        batch_size=batch_size,
        lr=learning_rate,
        weight_decay=weight_decay,
        device=device,
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
        mixed_precision=mixed_precision,
        augment_dataset=True,
        validation_data_path=validation_path,
    )
    _update_checkpoint(
        output_checkpoint,
        {
            "branch": branch_name,
            "base_checkpoint": os.path.abspath(base_checkpoint),
            "teacher": "tactical",
            "restoration_dataset": dataset_summary,
            "history": history,
            "validation_results": history[-1].get("tactical_validation") if history else None,
            "promote_allowed": False,
        },
    )
    print(
        f"[restore-train] DONE branch={branch_name} final_loss={history[-1]['total_loss'] if history else 0.0:.4f}",
        flush=True,
    )
    return history


def main(argv=None) -> int:
    args = parse_args(argv)
    device = get_device(args.device, allow_cpu_fallback=args.allow_cpu_fallback)
    print(f"[restore-train] device={describe_device(device)}", flush=True)
    dataset_summary = build_tactical_restoration_dataset(
        v1_checkpoint=args.v1_checkpoint,
        output_path=args.output_dataset,
        validation_output=args.validation_output,
        curriculum_data=args.curriculum_data,
        games=args.games,
        rule_mode=args.rule_mode,
        num_simulations=args.num_simulations,
        device=str(device),
        max_moves=args.max_moves,
        reason_weights=DEFAULT_REASON_WEIGHTS,
        max_low_heuristic_ratio=args.max_low_heuristic_ratio,
        defense_repeats=args.defense_repeats,
        curriculum_replay_count=args.curriculum_replay_count,
        center_replay_repeats=args.center_replay_repeats,
        stats_path=os.path.splitext(args.output_dataset)[0] + "_stats.json",
    )
    history_v1 = _train_branch(
        base_checkpoint=args.v1_checkpoint,
        output_checkpoint=args.output_v1_checkpoint,
        data_path=args.output_dataset,
        validation_path=args.validation_output,
        device=str(device),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        mixed_precision=bool(args.mixed_precision),
        dataset_summary=dataset_summary,
        branch_name="tactical_restoration_from_v1",
    )
    history_curriculum = _train_branch(
        base_checkpoint=args.curriculum_checkpoint,
        output_checkpoint=args.output_curriculum_checkpoint,
        data_path=args.output_dataset,
        validation_path=args.validation_output,
        device=str(device),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        mixed_precision=bool(args.mixed_precision),
        dataset_summary=dataset_summary,
        branch_name="tactical_restoration_from_curriculum",
    )
    summary = {
        "dataset": dataset_summary,
        "v1_checkpoint": args.v1_checkpoint,
        "curriculum_checkpoint": args.curriculum_checkpoint,
        "output_v1_checkpoint": args.output_v1_checkpoint,
        "output_curriculum_checkpoint": args.output_curriculum_checkpoint,
        "history_v1": history_v1,
        "history_curriculum": history_curriculum,
    }
    _write_json(summary, args.summary_output)
    print(f"[restore-train] summary saved: {args.summary_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
