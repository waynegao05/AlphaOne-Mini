from __future__ import annotations

import numpy as np

from game.encoder import action_to_index


class FirstLegalPlayer:
    name = "FirstLegal"

    def select_action(self, board):
        moves = board.get_legal_moves()
        if not moves:
            return None
        x, y = moves[0]
        return action_to_index(x, y)


class TacticalWinTeacher:
    name = "Teacher"

    def select_action(self, board):
        from engine.tactical_player import TacticalPlayer

        return TacticalPlayer().select_action(board)


def test_collect_mistake_positions_from_players_writes_npz(tmp_path):
    from train.mistake_mining import collect_mistake_positions_from_players

    output = tmp_path / "mistakes.npz"
    summary = collect_mistake_positions_from_players(
        student_player=FirstLegalPlayer(),
        teacher_player=TacticalWinTeacher(),
        games=2,
        rule_mode="basic",
        output_path=str(output),
        max_moves=20,
        min_score_gap=0.0,
    )

    assert output.exists()
    assert summary["num_samples"] > 0
    with np.load(output, allow_pickle=False) as data:
        assert data["states"].shape[1:] == (4, 15, 15)
        assert data["policies"].shape[1:] == (225,)
        assert data["values"].shape[1:] == (1,)
        assert data["threat_labels"].shape[1:] == (12, 15, 15)
        assert data["forbidden_labels"].shape[1:] == (1, 15, 15)
        assert np.allclose(data["policies"].sum(axis=1), 1.0)


def test_make_teacher_sample_uses_teacher_action_and_value():
    from train.mistake_mining import make_mistake_sample
    from game.board import BLACK, Board

    board = Board()
    board.grid[5][7] = BLACK
    board.grid[6][7] = BLACK
    board.grid[7][7] = BLACK
    board.grid[8][7] = BLACK
    board.move_count = 4
    board.current_player = BLACK
    teacher_action = action_to_index(9, 7)

    sample = make_mistake_sample(
        board,
        teacher_action=teacher_action,
        final_winner=BLACK,
        rule_mode="basic",
    )

    assert sample.policy[teacher_action] == 1.0
    assert sample.value == 1.0
    assert sample.threat_labels.shape == (12, 15, 15)


def test_append_center_replay_samples_preserves_dataset_shape(tmp_path):
    from train.mistake_mining import append_center_replay_samples, make_mistake_sample
    from train.tactical_distillation import save_tactical_dataset
    from game.board import Board

    sample = make_mistake_sample(Board(), teacher_action=action_to_index(7, 7), final_winner=0)
    data_path = tmp_path / "data.npz"
    save_tactical_dataset(
        np.stack([sample.state]),
        np.stack([sample.policy]),
        np.asarray([[sample.value]], dtype=np.float32),
        str(data_path),
        threat_labels=np.stack([sample.threat_labels]),
        forbidden_labels=np.stack([sample.forbidden_labels]),
        tactical_scores=np.stack([sample.tactical_scores]),
    )

    added = append_center_replay_samples(str(data_path), repeats=3)

    assert added == 3
    with np.load(data_path, allow_pickle=False) as data:
        assert data["states"].shape[0] == 4
        assert np.argmax(data["policies"][-1]) == action_to_index(7, 7)


def test_oversample_critical_samples_repeats_only_critical_reasons():
    from train.mistake_mining import make_mistake_sample, oversample_critical_samples
    from game.board import Board

    critical = make_mistake_sample(
        Board(),
        teacher_action=action_to_index(7, 7),
        final_winner=0,
        reasons=("missed_immediate_block",),
    )
    quiet = make_mistake_sample(
        Board(),
        teacher_action=action_to_index(7, 7),
        final_winner=0,
        reasons=("teacher_student_disagree",),
    )

    samples = oversample_critical_samples([critical, quiet], critical_repeat=3)

    assert len(samples) == 4
    assert sum("missed_immediate_block" in sample.reasons for sample in samples) == 3
    assert sum("teacher_student_disagree" in sample.reasons for sample in samples) == 1


def test_build_replay_samples_from_curriculum_npz(tmp_path):
    from train.mistake_mining import curriculum_replay_samples, make_mistake_sample
    from train.tactical_distillation import save_tactical_dataset
    from game.board import Board

    sample = make_mistake_sample(
        Board(),
        teacher_action=action_to_index(7, 7),
        final_winner=0,
        reasons=("curriculum_source",),
    )
    data_path = tmp_path / "curriculum.npz"
    save_tactical_dataset(
        np.stack([sample.state, sample.state]),
        np.stack([sample.policy, sample.policy]),
        np.asarray([[0.0], [0.0]], dtype=np.float32),
        str(data_path),
        threat_labels=np.stack([sample.threat_labels, sample.threat_labels]),
        forbidden_labels=np.stack([sample.forbidden_labels, sample.forbidden_labels]),
        tactical_scores=np.stack([sample.tactical_scores, sample.tactical_scores]),
    )

    replay = curriculum_replay_samples(str(data_path), max_samples=1)

    assert len(replay) == 1
    assert replay[0].reasons == ("curriculum_replay",)
    assert replay[0].policy.shape == (225,)


def test_main_mistake_mining_train_parses_v2_arguments():
    from main_mistake_mining_train import parse_args

    args = parse_args(
        [
            "--student-checkpoint",
            "outputs/checkpoints/latest_advanced.pt",
            "--teachers",
            "tactical",
            "hybrid",
            "--games-per-teacher",
            "20",
            "--seeds",
            "2026",
            "7",
            "--learning-rate",
            "0.0002",
            "--include-center-replay",
            "--include-curriculum-replay",
            "--curriculum-data",
            "outputs/supervised/tactical_curriculum_latest.npz",
            "--oversample-critical",
            "--critical-repeat",
            "3",
            "--output-dataset",
            "outputs/supervised/mistake_mining_v2_latest.npz",
        ]
    )

    assert args.teachers == ["tactical", "hybrid"]
    assert args.games_per_teacher == 20
    assert args.seeds == [2026, 7]
    assert args.lr == 0.0002
    assert args.include_center_replay is True
    assert args.include_curriculum_replay is True
    assert args.oversample_critical is True
    assert args.output_dataset == "outputs/supervised/mistake_mining_v2_latest.npz"


def test_main_mistake_mining_train_parses_v3_balancing_arguments():
    from main_mistake_mining_train import parse_args

    args = parse_args(
        [
            "--student-checkpoint",
            "outputs/checkpoints/latest_advanced_mistake_v2_from_latest.pt",
            "--teachers",
            "tactical",
            "hybrid",
            "--teacher-balance",
            "tactical:0.5,hybrid:0.5",
            "--reason-weights",
            "missed_immediate_block:5,low_heuristic_move:1",
            "--max-low-heuristic-ratio",
            "0.25",
            "--include-v1-tactical-draw-replay",
            "--v1-checkpoint",
            "outputs/checkpoints/latest_advanced_mistake_tuned.pt",
            "--v2-checkpoint",
            "outputs/checkpoints/latest_advanced_mistake_v2_from_latest.pt",
            "--validation-teacher",
            "tactical",
        ]
    )

    assert args.teacher_balance == "tactical:0.5,hybrid:0.5"
    assert args.reason_weights == "missed_immediate_block:5,low_heuristic_move:1"
    assert args.max_low_heuristic_ratio == 0.25
    assert args.include_v1_tactical_draw_replay is True
    assert args.validation_teacher == "tactical"
