"""Generate Markdown reports from experiment summaries."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any


DEFAULT_SUMMARY = os.path.join(
    "outputs", "experiments", "smoke_test_cpu", "summary.json"
)
DEFAULT_OUTPUT = os.path.join("docs", "experiment_report_latest.md")


def _format_float(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def _history_table(title: str, history: list[dict[str, Any]]) -> str:
    if not history:
        return f"### {title}\n\nNo {title.lower()} was recorded.\n"
    lines = [
        f"### {title}",
        "",
        "| Epoch | Total Loss | Policy Loss | Value Loss | Threat Loss | Forbidden Loss |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in history:
        lines.append(
            f"| {row.get('epoch', 'n/a')} | "
            f"{_format_float(row.get('total_loss', row.get('loss')))} | "
            f"{_format_float(row.get('policy_loss'))} | "
            f"{_format_float(row.get('value_loss'))} | "
            f"{_format_float(row.get('threat_loss'))} | "
            f"{_format_float(row.get('forbidden_loss'))} |"
        )
    return "\n".join(lines) + "\n"


def _loss_summary(summary: dict[str, Any]) -> str:
    pretrain_section = summary.get("pretrain") or {}
    finetune_section = summary.get("finetune") or {}
    pretrain_history = pretrain_section.get("history") or []
    finetune_history = finetune_section.get("history") or []
    if not pretrain_history and not finetune_history:
        return "No training history was recorded.\n"
    return "\n".join(
        [
            _history_table("Pretrain History", pretrain_history),
            _history_table("Finetune History", finetune_history),
        ]
    )


def _paths_summary(summary: dict[str, Any]) -> str:
    paths = summary.get("paths") or {}
    config = summary.get("config") or {}
    resume_section = summary.get("resume") or {}
    selfplay_section = summary.get("selfplay") or {}
    lines = [
        f"- Resume from: `{config.get('resume_from') or resume_section.get('from') or 'none'}`",
        f"- Self-play samples: `{selfplay_section.get('num_samples', 0)}`",
        f"- Pretrained checkpoint: `{paths.get('pretrained_checkpoint', 'n/a')}`",
        f"- Latest checkpoint: `{paths.get('latest_checkpoint', 'n/a')}`",
    ]
    return "\n".join(lines) + "\n"


def _benchmark_table(summary: dict[str, Any]) -> str:
    matches = summary.get("benchmark", {}).get("matches", {})
    if not matches:
        return "No benchmark results were recorded.\n"
    lines = [
        "| Match | Games | Player A Win Rate | Draw Rate | Avg Moves |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, result in matches.items():
        lines.append(
            f"| {name} | {result.get('total_games', 0)} | "
            f"{result.get('player_a_win_rate', 0.0):.3f} | "
            f"{result.get('draw_rate', 0.0):.3f} | "
            f"{result.get('avg_moves', 0.0):.1f} |"
        )
    return "\n".join(lines) + "\n"


def render_experiment_report(summary: dict[str, Any]) -> str:
    config = summary.get("config", {})
    experiment_name = summary.get("experiment_name", config.get("experiment_name", "unknown"))
    return (
        f"# Experiment Report: {experiment_name}\n\n"
        "This report summarizes a reproducible AlphaZero-mini training run. "
        "Small-scale smoke tests do not prove competition-level strength.\n\n"
        "## Configuration\n\n"
        f"- Model type: `{summary.get('model_type', config.get('model_type', 'unknown'))}`\n"
        f"- Parameter count: `{summary.get('parameter_count', 'unknown')}`\n"
        f"- Device: `{summary.get('device', config.get('device', 'unknown'))}`\n"
        f"- Batch size: `{config.get('batch_size', 'unknown')}`\n"
        f"- Data augmentation: `{config.get('use_augmentation', False)}`\n"
        f"- Auxiliary loss: `{config.get('use_auxiliary_loss', False)}`\n"
        f"- Tactical samples: `{summary.get('data', {}).get('num_samples', 'unknown')}`\n\n"
        "## Artifacts\n\n"
        f"{_paths_summary(summary)}\n"
        "## Loss Summary\n\n"
        f"{_loss_summary(summary)}\n\n"
        "## Benchmark Results\n\n"
        f"{_benchmark_table(summary)}\n"
        "## Current Best Model\n\n"
        "Use the checkpoint with the best repeated benchmark result. This smoke "
        "report does not promote `best.pt` automatically.\n\n"
        "## Limitations\n\n"
        "- CUDA smoke tests only prove the flow can run.\n"
        "- Real strength needs larger training, more self-play, and more games.\n"
        "- Current forbidden-rule recognizers still need formal edge-case validation.\n\n"
        "## Next Step\n\n"
        "Run fixed tactical benchmark positions and compare CNN, ResNet, and "
        "Advanced checkpoints over repeatable seeds.\n"
    )


def generate_experiment_report(
    summary_path: str = DEFAULT_SUMMARY,
    output_path: str = DEFAULT_OUTPUT,
) -> str:
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"experiment summary not found: {summary_path}")
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    text = render_experiment_report(summary)
    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(text)
    exp_report = summary.get("paths", {}).get("report")
    if exp_report and os.path.abspath(exp_report) != os.path.abspath(output_path):
        parent = os.path.dirname(os.path.abspath(exp_report))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(exp_report, "w", encoding="utf-8") as handle:
            handle.write(text)
    return text


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate experiment Markdown report")
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    generate_experiment_report(args.summary, args.output)
    print(f"experiment report saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_SUMMARY",
    "DEFAULT_OUTPUT",
    "render_experiment_report",
    "generate_experiment_report",
]
