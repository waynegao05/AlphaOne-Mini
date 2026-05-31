from __future__ import annotations

import numpy as np

from game.encoder import action_to_index


def _sample(reason: str):
    from game.board import Board
    from train.mistake_mining import make_mistake_sample

    return make_mistake_sample(
        Board(),
        teacher_action=action_to_index(7, 7),
        final_winner=0,
        reasons=(reason,),
    )


def test_build_hybrid_survival_samples_caps_low_heuristic_and_keeps_replay():
    from train.hybrid_survival import build_hybrid_survival_samples

    hybrid_samples = (
        [_sample("low_heuristic_move") for _ in range(20)]
        + [_sample("missed_immediate_block") for _ in range(4)]
        + [_sample("missed_open_four_defense") for _ in range(4)]
    )
    replay = [_sample("curriculum_replay"), _sample("center_replay")]

    final, summary = build_hybrid_survival_samples(
        hybrid_samples,
        replay_samples=replay,
        max_low_heuristic_ratio=0.20,
        reason_weights={
            "missed_immediate_block": 5,
            "missed_open_four_defense": 4,
            "low_heuristic_move": 1,
        },
    )

    low_count = sum("low_heuristic_move" in sample.reasons for sample in final)
    assert low_count / len(final) <= 0.20
    assert summary["teacher"] == "hybrid"
    assert summary["curriculum_replay_samples"] == 1
    assert summary["center_replay_samples"] == 1
    assert summary["reason_distribution_after"]["missed_immediate_block"] >= 4


def test_npz_replay_samples_can_retag_reason(tmp_path):
    from game.board import Board
    from train.hybrid_survival import npz_replay_samples
    from train.mistake_mining import make_mistake_sample
    from train.tactical_distillation import save_tactical_dataset

    sample = make_mistake_sample(
        Board(),
        teacher_action=action_to_index(7, 7),
        final_winner=0,
        reasons=("source",),
    )
    path = tmp_path / "replay.npz"
    save_tactical_dataset(
        np.stack([sample.state]),
        np.stack([sample.policy]),
        np.asarray([[sample.value]], dtype=np.float32),
        str(path),
        threat_labels=np.stack([sample.threat_labels]),
        forbidden_labels=np.stack([sample.forbidden_labels]),
        tactical_scores=np.stack([sample.tactical_scores]),
    )

    replay = npz_replay_samples(
        str(path),
        max_samples=1,
        reason="tactical_restoration_curriculum_replay",
    )

    assert len(replay) == 1
    assert replay[0].reasons == ("tactical_restoration_curriculum_replay",)
    assert int(np.argmax(replay[0].policy)) == action_to_index(7, 7)


def test_main_hybrid_survival_train_parses_arguments():
    from main_hybrid_survival_train import parse_args

    args = parse_args(
        [
            "--student-checkpoint",
            "outputs/checkpoints/latest_advanced_mistake_v2_from_latest.pt",
            "--games",
            "30",
            "--seeds",
            "2026",
            "7",
            "21",
            "--learning-rate",
            "0.0001",
            "--mixed-precision",
        ]
    )

    assert args.student_checkpoint.endswith("mistake_v2_from_latest.pt")
    assert args.games == 30
    assert args.seeds == [2026, 7, 21]
    assert args.lr == 0.0001
    assert args.mixed_precision is True


def test_main_hybrid_survival_train_parses_v3_arguments():
    from main_hybrid_survival_train import parse_args

    args = parse_args(
        [
            "--version",
            "v3",
            "--student-checkpoint",
            "outputs/checkpoints/latest_advanced_mistake_v2_from_latest.pt",
            "--output-checkpoint",
            "outputs/checkpoints/latest_advanced_hybrid_survival_v3_forced_block.pt",
        ]
    )

    assert args.version == "v3"
    assert args.output_checkpoint.endswith("hybrid_survival_v3_forced_block.pt")


def test_build_hybrid_survival_v2_samples_meets_caps():
    from train.hybrid_survival import build_hybrid_survival_v2_samples, AnnotatedMistakeSample

    def annotated(reason, value=0.0, delta=10000.0, target_source="hybrid_teacher"):
        sample = _sample(reason)
        sample.value = value
        metadata = {
            "reason": [reason],
            "heuristic_delta": delta,
            "remaining_moves": 12,
            "target_source": target_source,
        }
        return AnnotatedMistakeSample(sample=sample, metadata=metadata)

    corrections = (
        [annotated("missed_immediate_block") for _ in range(3)]
        + [annotated("missed_open_four_defense") for _ in range(3)]
        + [annotated("missed_blocked_four_defense") for _ in range(3)]
        + [annotated("missed_open_three_defense") for _ in range(12)]
        + [annotated("low_heuristic_move") for _ in range(12)]
        + [annotated("near_loss_position", value=-1.0) for _ in range(12)]
    )
    survival = [annotated("v2_self_survival_replay", target_source="v2_self_survival") for _ in range(6)]
    replay = [annotated("curriculum_replay", target_source="curriculum_replay") for _ in range(6)]
    replay += [annotated("center_replay", target_source="center_replay") for _ in range(2)]

    final, summary = build_hybrid_survival_v2_samples(
        corrections,
        survival,
        replay,
        target_samples=100,
    )

    assert final
    assert summary["forced_defense_combined_ratio"] >= 0.35
    assert summary["negative_value_ratio"] <= 0.55
    dist = summary["reason_distribution_after"]
    assert dist.get("low_heuristic_move", 0) / summary["final_samples"] <= 0.10
    assert dist.get("near_loss_position", 0) / summary["final_samples"] <= 0.15


def test_save_annotated_samples_writes_metadata_jsonl(tmp_path):
    from train.hybrid_survival import AnnotatedMistakeSample, save_annotated_samples

    sample = _sample("missed_immediate_block")
    annotated = AnnotatedMistakeSample(
        sample=sample,
        metadata={
            "student_action": 1,
            "teacher_action": int(sample.teacher_action),
            "heuristic_delta": 10000.0,
            "remaining_moves": 12,
            "reason": ["missed_immediate_block"],
        },
    )
    data_path = tmp_path / "data.npz"
    metadata_path = tmp_path / "metadata.jsonl"

    shapes = save_annotated_samples([annotated], str(data_path), str(metadata_path))

    assert shapes["states"] == [1, 4, 15, 15]
    assert data_path.exists()
    assert metadata_path.exists()
    assert "missed_immediate_block" in metadata_path.read_text(encoding="utf-8")
