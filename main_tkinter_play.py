"""Launch the local Tkinter Gomoku test UI."""

from __future__ import annotations

import argparse
import tkinter as tk

from ui.tkinter_board import GomokuTkApp
from ui.player_factory import DEFAULT_EXTERNAL_AI_PATH, PLAYER_TYPE_LABELS


PLAYER_CHOICES = list(PLAYER_TYPE_LABELS.keys())
AI_CHOICES = [
    "alphaone_mini",
    "external_ai",
]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AlphaOne-Mini Tkinter Gomoku local test UI")
    parser.add_argument("--rule-mode", choices=["basic", "forbidden"], default="forbidden")
    parser.add_argument("--black-player", choices=PLAYER_CHOICES, default="human")
    parser.add_argument("--white-player", choices=PLAYER_CHOICES, default=None)
    parser.add_argument(
        "--ai-player",
        choices=AI_CHOICES,
        default=None,
        help="Optional shortcut for selecting the white-side AI.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-simulations", type=int, default=25)
    parser.add_argument("--move-delay-ms", type=int, default=250)
    parser.add_argument("--external-ai", default=DEFAULT_EXTERNAL_AI_PATH)
    parser.add_argument(
        "--competition-protocol",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable designated opening + swap + five-move-N protocol (default: on).",
    )
    parser.add_argument(
        "--fifth-n",
        type=int,
        default=2,
        choices=range(2, 6),
        help="Number of black-5 candidates (default: 2).",
    )
    return parser.parse_args(argv)


def _normalize_ai_player(ai_player: str) -> str:
    return ai_player


def main(argv=None) -> int:
    print("[startup] parsing arguments ...")
    args = parse_args(argv)
    white_player = args.white_player
    if white_player is None:
        white_player = _normalize_ai_player(
            args.ai_player or "alphaone_mini",
        )
    print(f"[startup] device={args.device}, rule={args.rule_mode}, "
          f"protocol={args.competition_protocol}, fifth_n={args.fifth_n}, "
          f"black={args.black_player}, white={white_player}")
    print("[startup] creating Tk root window ...")
    root = tk.Tk()
    print("[startup] building UI ...")
    GomokuTkApp(
        root,
        rule_mode=args.rule_mode,
        black_player=args.black_player,
        white_player=white_player,
        external_ai_path=args.external_ai,
        device=args.device,
        num_simulations=args.num_simulations,
        move_delay_ms=args.move_delay_ms,
        competition_protocol=args.competition_protocol,
        fifth_n=args.fifth_n,
    )
    print("[startup] entering main loop ...")
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
