"""Command-line human-vs-AI entry point."""

from __future__ import annotations

import argparse
import os
import sys

from game.board import BLACK, WHITE


DEFAULT_BEST = os.path.join("outputs", "checkpoints", "best.pt")
DEFAULT_LATEST = os.path.join("outputs", "checkpoints", "latest.pt")
DEFAULT_TACTICAL_SPECIALIST = os.path.join(
    "outputs", "checkpoints", "latest_advanced_tactical_restoration_from_curriculum.pt"
)
DEFAULT_V2 = os.path.join(
    "outputs", "checkpoints", "latest_advanced_mistake_v2_from_latest.pt"
)
DEFAULT_RENJUNET_EXTERNAL_ADAPTED = os.path.join(
    "outputs", "checkpoints", "pretrained_advanced_renjunet_all_external_adapted.pt"
)
DEFAULT_RENJUNET_ALL = os.path.join(
    "outputs", "checkpoints", "pretrained_advanced_renjunet_all.pt"
)
DEFAULT_RENJUNET_PHASE3 = os.path.join(
    "outputs", "checkpoints", "pretrained_advanced_renjunet_phase3.pt"
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AlphaOne-Mini command-line play")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_BEST)
    parser.add_argument(
        "--fallback-latest",
        dest="fallback_latest",
        action="store_true",
        default=True,
        help="Fallback to outputs/checkpoints/latest.pt if best.pt is missing.",
    )
    parser.add_argument("--no-fallback-latest", dest="fallback_latest", action="store_false")
    parser.add_argument("--num-simulations", type=int, default=50)
    parser.add_argument("--c-puct", type=float, default=5.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--human-color", choices=["black", "white"], default="black")
    parser.add_argument("--board-size", type=int, default=15)
    parser.add_argument("--rule-mode", choices=["basic", "forbidden"], default="basic")
    parser.add_argument(
        "--ai-player",
        choices=["model", "neural_guarded", "strong"],
        default="model",
        help=(
            "Use the original checkpoint model, the NeuralGuarded legacy AI, "
            "or AlphaOne-Mini (tactical + VCF/VCT + best available RenjuNet MCTS)."
        ),
    )
    parser.add_argument(
        "--vcf-depth",
        type=int,
        default=9,
        help="StrongPlayer VCF search depth (half-moves).",
    )
    parser.add_argument(
        "--vcf-defense-depth",
        type=int,
        default=7,
        help="StrongPlayer defensive VCF search depth (half-moves).",
    )
    parser.add_argument(
        "--vcf-node-budget",
        type=int,
        default=20000,
        help="StrongPlayer hard cap on VCF nodes per move.",
    )
    parser.add_argument(
        "--vct-depth",
        type=int,
        default=7,
        help="StrongPlayer bounded VCT search depth (half-moves).",
    )
    parser.add_argument(
        "--vct-node-budget",
        type=int,
        default=20000,
        help="StrongPlayer hard cap on VCT search nodes per move.",
    )
    parser.add_argument(
        "--lookahead-depth",
        type=int,
        default=4,
        help="AlphaOne-Mini candidate-pruned opponent lookahead depth.",
    )
    parser.add_argument(
        "--lookahead-branch-factor",
        type=int,
        default=3,
        help="AlphaOne-Mini likely-move branch factor for opponent lookahead.",
    )
    parser.add_argument(
        "--variant",
        choices=[
            "default",
            "no_hybrid_fallback",
            "conservative_fallback",
            "fallback_off",
            "aggressive_fallback",
        ],
        default="default",
        help="NeuralGuarded variant; no_hybrid_fallback is the basic demo default.",
    )
    parser.add_argument(
        "--fallback-mode",
        choices=["off", "conservative", "normal", "aggressive"],
        default=None,
        help="Optional NeuralGuarded fallback mode override.",
    )
    parser.add_argument(
        "--tactical-checkpoint",
        type=str,
        default=DEFAULT_TACTICAL_SPECIALIST,
    )
    parser.add_argument("--v2-checkpoint", type=str, default=DEFAULT_V2)
    return parser.parse_args(argv)


def _load_checkpoint_model(checkpoint_path: str, args, *, fallback_model_type: str = "cnn"):
    """Create a model matching checkpoint metadata, then load its weights."""
    import torch

    from model.checkpoint import load_checkpoint
    from model.model_factory import create_model, create_model_from_metadata

    if not os.path.exists(checkpoint_path):
        return create_model(fallback_model_type)
    try:
        state = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=args.device)
    metadata = state.get("metadata", {}) if isinstance(state, dict) else {}
    model = create_model_from_metadata(metadata, fallback_model_type=fallback_model_type)
    load_checkpoint(model, checkpoint_path, device=args.device)
    return model


def _build_neural_guarded_player(args):
    from engine.neural_guarded_player import NeuralGuardedPlayer

    fallback_mode = args.fallback_mode
    use_hybrid_fallback = True
    if args.variant == "no_hybrid_fallback":
        use_hybrid_fallback = False
        fallback_mode = fallback_mode or "normal"
    elif args.variant == "conservative_fallback":
        fallback_mode = fallback_mode or "conservative"
    elif args.variant == "fallback_off":
        fallback_mode = fallback_mode or "off"
    elif args.variant == "aggressive_fallback":
        fallback_mode = fallback_mode or "aggressive"
    else:
        fallback_mode = fallback_mode or "normal"

    player = NeuralGuardedPlayer(
        tactical_checkpoint=args.tactical_checkpoint,
        v2_checkpoint=args.v2_checkpoint,
        device=args.device,
        rule_mode=args.rule_mode,
        num_simulations=args.num_simulations,
        fallback_mode=fallback_mode,
        use_hybrid_fallback=use_hybrid_fallback,
        name=f"NeuralGuarded_{args.variant}",
    )
    print(
        "Loaded NeuralGuarded AI: "
        f"variant={args.variant}, rule_mode={args.rule_mode}, "
        f"fallback_mode={fallback_mode}, use_hybrid_fallback={use_hybrid_fallback}"
    )
    return player


def _build_model_player(args):
    from evaluate.players import ModelMCTSPlayer

    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path) and args.fallback_latest:
        if os.path.exists(DEFAULT_LATEST):
            print(
                f"best checkpoint missing ({checkpoint_path}); falling back to {DEFAULT_LATEST}"
            )
            checkpoint_path = DEFAULT_LATEST

    if os.path.exists(checkpoint_path):
        model = _load_checkpoint_model(checkpoint_path, args)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        model = _load_checkpoint_model(checkpoint_path, args)
        print(
            "WARNING: no checkpoint found; using a randomly initialized model. "
            "This is weak and intended only for smoke testing."
        )
    model.eval()

    return ModelMCTSPlayer(
        model=model,
        num_simulations=args.num_simulations,
        c_puct=args.c_puct,
        device=args.device,
        board_size=args.board_size,
        name="AI",
    )


def _build_strong_player(args):
    """AlphaOne-Mini = tactical + VCF/VCT + MCTS-on-RenjuNet checkpoint."""
    from engine.strong_player import StrongPlayer
    from evaluate.players import ModelMCTSPlayer

    checkpoint_path = args.checkpoint
    if checkpoint_path == DEFAULT_BEST:
        if os.path.exists(DEFAULT_RENJUNET_EXTERNAL_ADAPTED):
            checkpoint_path = DEFAULT_RENJUNET_EXTERNAL_ADAPTED
            print(f"AlphaOne-Mini defaulting to external-adapted RenjuNet checkpoint: {checkpoint_path}")
        elif os.path.exists(DEFAULT_RENJUNET_ALL):
            checkpoint_path = DEFAULT_RENJUNET_ALL
            print(f"AlphaOne-Mini defaulting to all-record RenjuNet checkpoint: {checkpoint_path}")
        elif os.path.exists(DEFAULT_RENJUNET_PHASE3):
            checkpoint_path = DEFAULT_RENJUNET_PHASE3
            print(f"AlphaOne-Mini defaulting to phase3 checkpoint: {checkpoint_path}")
    if not os.path.exists(checkpoint_path) and args.fallback_latest:
        if os.path.exists(DEFAULT_LATEST):
            print(
                f"best checkpoint missing ({checkpoint_path}); falling back to {DEFAULT_LATEST}"
            )
            checkpoint_path = DEFAULT_LATEST

    if os.path.exists(checkpoint_path):
        model = _load_checkpoint_model(checkpoint_path, args)
        print(f"AlphaOne-Mini loaded checkpoint: {checkpoint_path}")
    else:
        model = _load_checkpoint_model(checkpoint_path, args)
        print(
            "WARNING: no checkpoint found; AlphaOne-Mini's MCTS tier will use a "
            "randomly initialized model. Tactical + VCF/VCT tiers still work fine."
        )
    model.eval()

    mcts_player = ModelMCTSPlayer(
        model=model,
        num_simulations=args.num_simulations,
        c_puct=args.c_puct,
        device=args.device,
        board_size=args.board_size,
        name="Strong_MCTS",
    )
    return StrongPlayer(
        mcts_player=mcts_player,
        rule_mode=args.rule_mode,
        vcf_depth=args.vcf_depth,
        vcf_defense_depth=args.vcf_defense_depth,
        vcf_node_budget=args.vcf_node_budget,
        vct_depth=args.vct_depth,
        vct_node_budget=args.vct_node_budget,
        lookahead_depth=args.lookahead_depth,
        lookahead_branch_factor=args.lookahead_branch_factor,
        name="AlphaOne-Mini",
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    from ui.cli_play import run_cli_game

    if args.ai_player == "neural_guarded":
        ai_player = _build_neural_guarded_player(args)
    elif args.ai_player == "strong":
        ai_player = _build_strong_player(args)
    else:
        ai_player = _build_model_player(args)

    human_color = BLACK if args.human_color == "black" else WHITE
    run_cli_game(
        ai_player=ai_player,
        human_color=human_color,
        board_size=args.board_size,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
