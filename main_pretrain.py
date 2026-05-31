"""Command-line entry point for tactical or record-based policy pretraining."""

from __future__ import annotations

import argparse
import os

from model.model_factory import create_model
from train.supervised_pretrain import (
    DEFAULT_SUPERVISED_DATA_PATH,
    build_supervised_dataset_from_records,
    train_policy_pretrain,
)
from train.tactical_distillation import (
    DEFAULT_TACTICAL_DATA_PATH,
    generate_tactical_dataset,
)
from train.progress import progress_print
from utils.device import describe_device, get_device


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Policy pretraining for AlphaZero-mini")
    parser.add_argument("--mode", choices=["tactical", "records"], default="tactical")
    parser.add_argument("--data", default=None, help="Existing supervised npz to train from")
    parser.add_argument("--record-file", default=None, help="Text file containing record text")
    parser.add_argument("--output-data", default=None)
    parser.add_argument("--checkpoint-dir", default=os.path.join("outputs", "checkpoints"))
    parser.add_argument("--checkpoint-name", default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--model-type", choices=["cnn", "resnet", "advanced"], default="advanced")
    parser.add_argument("--use-augmentation", action="store_true")
    parser.add_argument("--use-auxiliary-loss", action="store_true")
    parser.add_argument("--aux-threat-weight", type=float, default=0.3)
    parser.add_argument("--aux-forbidden-weight", type=float, default=0.2)
    parser.add_argument("--rule-mode", choices=["basic", "forbidden"], default="basic")
    parser.add_argument("--tactical-games", type=int, default=1)
    parser.add_argument("--max-moves", type=int, default=20)
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument("--smoothing", type=float, default=0.0)
    return parser.parse_args(argv)


def _resolve_output_data(args: argparse.Namespace) -> str:
    if args.output_data:
        return args.output_data
    if args.mode == "tactical":
        return DEFAULT_TACTICAL_DATA_PATH
    return DEFAULT_SUPERVISED_DATA_PATH


def prepare_data(args: argparse.Namespace) -> str:
    if args.data:
        progress_print(f"USE existing_data path={args.data}", "pretrain")
        return args.data

    output_data = _resolve_output_data(args)
    if args.mode == "tactical":
        progress_print(
            f"START stage=generate_tactical_data games={args.tactical_games} "
            f"rule_mode={args.rule_mode} output={output_data}",
            "pretrain",
        )
        states, _, _ = generate_tactical_dataset(
            num_games=args.tactical_games,
            output_path=output_data,
            rule_mode=args.rule_mode,
            max_moves=args.max_moves,
            smoothing=args.smoothing,
            seed=0,
            include_auxiliary_labels=args.use_auxiliary_loss,
            use_augmentation=args.use_augmentation,
            progress_interval=args.progress_interval,
        )
        progress_print(
            f"DONE stage=generate_tactical_data path={output_data} samples={len(states)}",
            "pretrain",
        )
        return output_data

    if not args.record_file:
        raise SystemExit("--record-file is required when --mode records and --data is omitted")
    if not os.path.exists(args.record_file):
        raise SystemExit(f"record file not found: {args.record_file}")
    with open(args.record_file, "r", encoding="utf-8") as handle:
        record_text = handle.read()
    progress_print(
        f"START stage=generate_record_data record_file={args.record_file} output={output_data}",
        "pretrain",
    )
    states, _, _ = build_supervised_dataset_from_records(
        [record_text],
        output_path=output_data,
        rule_mode=args.rule_mode,
        smoothing=args.smoothing,
    )
    progress_print(
        f"DONE stage=generate_record_data path={output_data} samples={len(states)}",
        "pretrain",
    )
    return output_data


def main(argv=None) -> int:
    args = parse_args(argv)
    device = get_device(args.device, allow_cpu_fallback=args.allow_cpu_fallback)
    progress_print(
        f"START main_pretrain mode={args.mode} model_type={args.model_type} "
        f"device={describe_device(device)}",
        "pretrain",
    )
    data_path = prepare_data(args)

    model = create_model(args.model_type)
    checkpoint_name = args.checkpoint_name or (
        "pretrained_advanced.pt" if args.model_type == "advanced" else "pretrained.pt"
    )
    history = train_policy_pretrain(
        model,
        data_path=data_path,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=str(device),
        use_auxiliary_loss=args.use_auxiliary_loss,
        loss_weights={
            "threat": args.aux_threat_weight,
            "forbidden": args.aux_forbidden_weight,
        },
        checkpoint_name=checkpoint_name,
        model_type=args.model_type,
        resume_from=args.resume_from,
    )
    checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_name)
    if history:
        final = history[-1]
        print(
            f"pretrain complete: loss={final['total_loss']:.4f} "
            f"policy={final['policy_loss']:.4f} value={final['value_loss']:.4f}"
        )
    print(f"saved checkpoint: {checkpoint_path}")
    print(f"device: {describe_device(device)}")
    print("note: this is a smoke-scale pretrain, not a competition-strength model")
    progress_print(f"DONE main_pretrain checkpoint={checkpoint_path}", "pretrain")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
