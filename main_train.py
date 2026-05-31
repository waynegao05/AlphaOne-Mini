"""训练入口脚本。

从第五批 ``main_selfplay.py`` 写出的 npz 加载数据，对 PolicyValueNet 跑若干 epoch，
并在 ``--checkpoint-dir`` 下持续刷新 ``latest.pt``。

注意：本脚本只做训练闭环，不做评估、不替换 best 模型、不做新一轮自博弈。

CLI::

    python main_train.py --data outputs/selfplay_data/selfplay_latest.npz \\
                         --checkpoint-dir outputs/checkpoints \\
                         --epochs 5 --batch-size 64 --lr 0.001
"""

from __future__ import annotations

import argparse
import os
import sys

from model.model_factory import create_model
from train.progress import progress_print
from train.train import train_model
from utils.device import describe_device, get_device


DEFAULT_DATA_PATH = os.path.join("outputs", "selfplay_data", "selfplay_latest.npz")
DEFAULT_CHECKPOINT_DIR = os.path.join("outputs", "checkpoints")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AlphaZero mini training (单轮训练，不含评估 / 自博弈)"
    )
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--checkpoint-dir", type=str, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--model-type", choices=["cnn", "resnet", "advanced"], default="advanced")
    parser.add_argument("--use-auxiliary-loss", action="store_true")
    parser.add_argument("--aux-threat-weight", type=float, default=0.3)
    parser.add_argument("--aux-forbidden-weight", type=float, default=0.2)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-shuffle", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    device = get_device(args.device, allow_cpu_fallback=args.allow_cpu_fallback)
    progress_print(
        f"START main_train model_type={args.model_type} data={args.data} device={describe_device(device)}",
        "train",
    )

    if not os.path.exists(args.data):
        print(f"错误: 自博弈数据文件不存在: {args.data}")
        print("请先运行: python main_selfplay.py")
        return 1

    print(
        f"加载数据: {args.data}\n"
        f"checkpoint dir: {args.checkpoint_dir}\n"
        f"epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}, "
        f"weight_decay={args.weight_decay}, device={describe_device(device)}, "
        f"grad_clip={args.grad_clip}"
    )

    model = create_model(args.model_type)
    history = train_model(
        model=model,
        data_path=args.data,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        device=str(device),
        grad_clip=args.grad_clip if args.grad_clip > 0 else None,
        shuffle=not args.no_shuffle,
        num_workers=args.num_workers,
        model_type=args.model_type,
        metadata_extra={
            "use_auxiliary_loss": bool(args.use_auxiliary_loss),
            "aux_threat_weight": float(args.aux_threat_weight),
            "aux_forbidden_weight": float(args.aux_forbidden_weight),
            "cuda_available": __import__("torch").cuda.is_available(),
        },
        use_auxiliary_loss=args.use_auxiliary_loss,
        loss_weights={
            "threat": args.aux_threat_weight,
            "forbidden": args.aux_forbidden_weight,
        },
    )

    print()
    print("训练完成。每个 epoch 的 loss:")
    for record in history:
        print(
            f"  epoch {record['epoch']}: total={record['total_loss']:.4f}, "
            f"policy={record['policy_loss']:.4f}, value={record['value_loss']:.4f}"
        )
    print(
        f"\nlatest checkpoint: "
        f"{os.path.abspath(os.path.join(args.checkpoint_dir, 'latest.pt'))}"
    )
    progress_print(
        f"DONE main_train checkpoint={os.path.abspath(os.path.join(args.checkpoint_dir, 'latest.pt'))}",
        "train",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
