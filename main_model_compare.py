"""Run model/player comparison and write JSON + Markdown summaries."""

from __future__ import annotations

import argparse

from evaluate.model_comparison import (
    DEFAULT_COMPARE_JSON,
    DEFAULT_COMPARE_MD,
    run_model_comparison,
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare AlphaZero-mini players/models")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--rule-mode", choices=["basic", "forbidden"], default="basic")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--num-simulations", type=int, default=50)
    parser.add_argument("--output-json", default=DEFAULT_COMPARE_JSON)
    parser.add_argument("--output-md", default=DEFAULT_COMPARE_MD)
    parser.add_argument("--max-moves", type=int, default=80)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    summary = run_model_comparison(
        games=args.games,
        rule_mode=args.rule_mode,
        device=args.device,
        allow_cpu_fallback=args.allow_cpu_fallback,
        output_json=args.output_json,
        output_md=args.output_md,
        num_simulations=args.num_simulations,
        max_moves=args.max_moves,
    )
    print(f"model comparison saved: {args.output_json}")
    print(f"markdown table: {args.output_md}")
    print(f"matches: {', '.join(summary['matches'].keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
