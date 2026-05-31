"""Train a model on deterministic tactical curriculum positions."""

from __future__ import annotations

import argparse
import json
import os

from model.model_factory import create_model
from model.checkpoint import load_checkpoint
from train.supervised_pretrain import train_policy_pretrain
from train.tactical_curriculum import (
    DEFAULT_CURRICULUM_DATA_PATH,
    generate_tactical_curriculum_dataset,
)
from utils.device import describe_device, get_device


DEFAULT_OUTPUT = os.path.join("outputs", "checkpoints", "curriculum_advanced.pt")
DEFAULT_STATS_OUTPUT = os.path.join(
    "outputs", "supervised", "tactical_curriculum_stats.json"
)
DEFAULT_RESUME = os.path.join("outputs", "checkpoints", "pretrained_advanced.pt")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train on deterministic tactical curriculum data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--model-type", default="advanced", choices=["cnn", "resnet", "advanced"])
    parser.add_argument("--rule-mode", default="basic", choices=["basic", "forbidden"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--repeats", type=int, default=32)
    parser.add_argument("--smoothing", type=float, default=0.0)
    parser.add_argument("--use-augmentation", action="store_true")
    parser.add_argument("--no-forbidden-samples", action="store_true")
    parser.add_argument("--data-output", default=DEFAULT_CURRICULUM_DATA_PATH)
    parser.add_argument("--stats-output", default=DEFAULT_STATS_OUTPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--resume-from", default=DEFAULT_RESUME)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--resume-optimizer",
        action="store_true",
        help="Resume optimizer/epoch state instead of only initializing model weights.",
    )
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--mixed-precision", action="store_true")
    return parser.parse_args(argv)


def _write_stats(stats: dict, path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)


def main(argv=None) -> int:
    args = parse_args(argv)
    device = get_device(args.device, allow_cpu_fallback=args.allow_cpu_fallback)
    print(
        f"[curriculum] START data rule_mode={args.rule_mode} repeats={args.repeats} "
        f"augmentation={bool(args.use_augmentation)} device={describe_device(device)}",
        flush=True,
    )
    arrays, stats = generate_tactical_curriculum_dataset(
        output_path=args.data_output,
        rule_mode=args.rule_mode,
        repeats=args.repeats,
        smoothing=args.smoothing,
        include_forbidden=not args.no_forbidden_samples,
        use_augmentation=args.use_augmentation,
    )
    _write_stats(stats, args.stats_output)
    print(
        f"[curriculum] DONE data samples={arrays['states'].shape[0]} "
        f"forbidden_positive={stats['forbidden_positive_count']} "
        f"data={args.data_output}",
        flush=True,
    )

    model = create_model(args.model_type)
    checkpoint_dir = os.path.dirname(os.path.abspath(args.output))
    checkpoint_name = os.path.basename(args.output)
    resume_from = None if args.no_resume else args.resume_from
    optimizer_resume = None
    if resume_from and not os.path.exists(resume_from):
        print(f"[curriculum] init checkpoint not found, training from scratch: {resume_from}", flush=True)
        resume_from = None
    elif resume_from and args.resume_optimizer:
        optimizer_resume = resume_from
    elif resume_from:
        print(f"[curriculum] loading model weights only from {resume_from}", flush=True)
        load_checkpoint(model, resume_from, device=str(device))
    print(
        f"[curriculum] START train model_type={args.model_type} epochs={args.epochs} "
        f"batch_size={args.batch_size} init_from={resume_from} "
        f"resume_optimizer={bool(optimizer_resume)}",
        flush=True,
    )
    history = train_policy_pretrain(
        model,
        args.data_output,
        checkpoint_dir=checkpoint_dir,
        checkpoint_name=checkpoint_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=str(device),
        grad_clip=args.grad_clip,
        use_auxiliary_loss=True,
        loss_weights={
            "policy": 2.0,
            "value": 0.5,
            "threat": 0.5,
            "forbidden": 0.5,
            "tactical_score": 0.1,
        },
        model_type=args.model_type,
        resume_from=optimizer_resume,
        scheduler_type="cosine",
        mixed_precision=args.mixed_precision,
        augment_dataset=False,
    )
    print(
        f"[curriculum] DONE train checkpoint={args.output} "
        f"final_loss={history[-1]['total_loss'] if history else 0.0:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
