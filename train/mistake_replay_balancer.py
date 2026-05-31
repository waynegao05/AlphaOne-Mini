"""Teacher-balanced replay utilities for mistake-mining datasets."""

from __future__ import annotations

from collections import Counter
from itertools import cycle, islice
from typing import Iterable, Mapping

from train.mistake_mining import MistakeSample, reason_distribution


def _parse_spec(spec: str | None, *, value_type=float) -> dict[str, float]:
    if not spec:
        return {}
    parsed: dict[str, float] = {}
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"invalid spec item {item!r}; expected name:value")
        key, value = item.split(":", 1)
        key = key.strip().lower()
        if not key:
            raise ValueError(f"invalid empty key in spec item {item!r}")
        parsed[key] = value_type(value.strip())
    return parsed


def parse_ratio_spec(spec: str | None) -> dict[str, float]:
    """Parse `tactical:0.5,hybrid:0.5` style teacher balance specs."""
    ratios = _parse_spec(spec, value_type=float)
    if any(value < 0 for value in ratios.values()):
        raise ValueError("teacher balance ratios must be non-negative")
    total = sum(ratios.values())
    if ratios and total <= 0:
        raise ValueError("teacher balance ratios must sum to a positive value")
    return ratios


def parse_weight_spec(spec: str | None) -> dict[str, float]:
    """Parse `reason:weight` specs used for replay oversampling."""
    weights = _parse_spec(spec, value_type=float)
    if any(value <= 0 for value in weights.values()):
        raise ValueError("reason weights must be positive")
    return weights


def _repeat_to_count(samples: list[MistakeSample], count: int) -> list[MistakeSample]:
    if count <= 0 or not samples:
        return []
    if len(samples) >= count:
        return list(samples[:count])
    return list(islice(cycle(samples), count))


def balance_teacher_groups(
    teacher_samples: Mapping[str, list[MistakeSample]],
    teacher_balance: Mapping[str, float] | None = None,
) -> tuple[list[MistakeSample], dict]:
    """Return samples balanced by teacher ratio, oversampling small groups if needed."""
    groups = {key.lower(): list(value) for key, value in teacher_samples.items()}
    groups = {key: value for key, value in groups.items() if value}
    if not groups:
        return [], {"teacher_counts_before": {}, "teacher_counts_after": {}}
    balance = {key.lower(): float(value) for key, value in (teacher_balance or {}).items()}
    if not balance:
        balance = {key: 1.0 for key in groups}
    total_weight = sum(balance.get(key, 0.0) for key in groups)
    if total_weight <= 0:
        balance = {key: 1.0 for key in groups}
        total_weight = float(len(groups))

    total_samples = sum(len(value) for value in groups.values())
    desired: dict[str, int] = {}
    remaining = total_samples
    keys = list(groups)
    for index, key in enumerate(keys):
        if index == len(keys) - 1:
            desired[key] = max(0, remaining)
        else:
            count = round(total_samples * (balance.get(key, 0.0) / total_weight))
            desired[key] = max(0, int(count))
            remaining -= desired[key]
    balanced: list[MistakeSample] = []
    teacher_counts_after: dict[str, int] = {}
    for key in keys:
        selected = _repeat_to_count(groups[key], desired.get(key, 0))
        balanced.extend(selected)
        teacher_counts_after[key] = len(selected)
    return balanced, {
        "teacher_counts_before": {key: len(value) for key, value in groups.items()},
        "teacher_counts_after": teacher_counts_after,
        "teacher_balance": dict(balance),
    }


def _sample_weight(sample: MistakeSample, reason_weights: Mapping[str, float]) -> int:
    if not reason_weights:
        return 1
    weight = 1.0
    for reason in sample.reasons:
        weight = max(weight, float(reason_weights.get(reason, 1.0)))
    return max(1, int(round(weight)))


def apply_reason_weights(
    samples: list[MistakeSample],
    reason_weights: Mapping[str, float] | None = None,
) -> tuple[list[MistakeSample], dict]:
    """Repeat samples according to the strongest configured reason weight."""
    weights = {key: float(value) for key, value in (reason_weights or {}).items()}
    expanded: list[MistakeSample] = []
    before = len(samples)
    for sample in samples:
        expanded.extend([sample] * _sample_weight(sample, weights))
    return expanded, {
        "samples_before": before,
        "samples_after": len(expanded),
        "weighted_samples_added": len(expanded) - before,
        "reason_weights": dict(weights),
    }


def cap_reason_ratio(
    samples: list[MistakeSample],
    reason: str = "low_heuristic_move",
    max_ratio: float = 0.25,
) -> tuple[list[MistakeSample], dict]:
    """Limit a reason's dominance while keeping non-reason samples untouched."""
    if not samples:
        return [], {"removed": 0, "ratio_after": 0.0}
    max_ratio = max(0.0, min(float(max_ratio), 1.0))
    if max_ratio >= 1.0:
        return list(samples), {"removed": 0, "ratio_after": 1.0}
    low = [sample for sample in samples if reason in sample.reasons]
    other = [sample for sample in samples if reason not in sample.reasons]
    if not low:
        return list(samples), {"removed": 0, "ratio_after": 0.0}
    max_low = int((max_ratio * len(other)) / max(1e-9, 1.0 - max_ratio))
    max_low = max(0, max_low)
    critical_low = [
        sample
        for sample in low
        if any(r != reason for r in sample.reasons)
    ]
    plain_low = [
        sample
        for sample in low
        if not any(r != reason for r in sample.reasons)
    ]
    kept_low = (critical_low + plain_low)[:max_low]
    capped = other + kept_low
    ratio_after = (
        sum(reason in sample.reasons for sample in capped) / len(capped)
        if capped
        else 0.0
    )
    return capped, {
        "reason": reason,
        "max_ratio": max_ratio,
        "before": len(samples),
        "after": len(capped),
        "removed": len(samples) - len(capped),
        "ratio_before": len(low) / len(samples),
        "ratio_after": ratio_after,
    }


def build_teacher_balanced_replay(
    teacher_samples: Mapping[str, list[MistakeSample]],
    *,
    teacher_balance: Mapping[str, float] | None = None,
    reason_weights: Mapping[str, float] | None = None,
    max_low_heuristic_ratio: float = 0.25,
    replay_samples: Iterable[MistakeSample] = (),
) -> tuple[list[MistakeSample], dict]:
    """Build final v3 training samples from teacher groups and replay samples."""
    balanced, balance_summary = balance_teacher_groups(teacher_samples, teacher_balance)
    weighted, weight_summary = apply_reason_weights(balanced, reason_weights)
    capped, cap_summary = cap_reason_ratio(
        weighted,
        reason="low_heuristic_move",
        max_ratio=max_low_heuristic_ratio,
    )
    final = capped + list(replay_samples)
    summary = {
        **balance_summary,
        "reason_weighting": weight_summary,
        "low_heuristic_cap": cap_summary,
        "reason_distribution_before": reason_distribution(balanced),
        "reason_distribution_after": reason_distribution(final),
        "final_samples": len(final),
    }
    summary["teacher_sample_counts"] = dict(Counter({
        key: len(value) for key, value in teacher_samples.items()
    }))
    return final, summary


__all__ = [
    "apply_reason_weights",
    "balance_teacher_groups",
    "build_teacher_balanced_replay",
    "cap_reason_ratio",
    "parse_ratio_spec",
    "parse_weight_spec",
]
