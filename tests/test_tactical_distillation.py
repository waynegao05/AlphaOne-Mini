"""Tests for tactical distillation dataset generation."""

from __future__ import annotations

import numpy as np

from game.board import BOARD_SIZE
from game.encoder import action_to_index, index_to_action


def test_make_policy_target_one_hot_and_smoothed():
    from train.tactical_distillation import make_policy_target

    action = action_to_index(7, 7)
    one_hot = make_policy_target(action)
    assert one_hot.shape == (BOARD_SIZE * BOARD_SIZE,)
    assert one_hot.dtype == np.float32
    assert one_hot.sum() == np.float32(1.0)
    assert one_hot[action] == np.float32(1.0)

    smooth = make_policy_target(action, smoothing=0.1)
    assert smooth.shape == (BOARD_SIZE * BOARD_SIZE,)
    assert smooth.sum() == np.float32(1.0)
    assert 0.8 < smooth[action] < 1.0


def test_generate_tactical_positions_returns_bounded_samples(capsys):
    from train.tactical_distillation import generate_tactical_positions

    samples = generate_tactical_positions(
        num_games=1,
        rule_mode="basic",
        max_moves=12,
        seed=0,
        progress_interval=1,
    )

    assert samples
    output = capsys.readouterr().out
    assert "[distill] START tactical_positions" in output
    assert "[distill] game 1/1 complete" in output
    assert "[distill] DONE tactical_positions" in output
    assert len(samples) <= 12
    first = samples[0]
    assert first.state.shape == (4, BOARD_SIZE, BOARD_SIZE)
    assert first.action == action_to_index(7, 7)
    assert first.current_player in (1, -1)


def test_generate_tactical_dataset_shapes_and_legal_policy_targets(tmp_path):
    from train.tactical_distillation import (
        generate_tactical_dataset,
        load_tactical_dataset,
    )

    path = tmp_path / "tactical.npz"
    states, policies, values = generate_tactical_dataset(
        num_games=1,
        output_path=str(path),
        rule_mode="basic",
        max_moves=12,
        smoothing=0.0,
        seed=1,
    )

    assert path.exists()
    assert states.shape[1:] == (4, BOARD_SIZE, BOARD_SIZE)
    assert policies.shape == (states.shape[0], BOARD_SIZE * BOARD_SIZE)
    assert values.shape == (states.shape[0], 1)
    assert states.dtype == np.float32
    assert policies.dtype == np.float32
    assert values.dtype == np.float32
    np.testing.assert_allclose(policies.sum(axis=1), np.ones(states.shape[0]))

    for state, policy in zip(states, policies):
        action = int(np.argmax(policy))
        x, y = index_to_action(action)
        occupied = state[0, y, x] + state[1, y, x]
        assert occupied == 0.0

    loaded = load_tactical_dataset(str(path))
    np.testing.assert_array_equal(loaded[0], states)
    np.testing.assert_array_equal(loaded[1], policies)
    np.testing.assert_array_equal(loaded[2], values)


def test_generate_tactical_dataset_forbidden_mode(tmp_path):
    from train.tactical_distillation import generate_tactical_dataset

    states, policies, values = generate_tactical_dataset(
        num_games=1,
        output_path=str(tmp_path / "forbidden.npz"),
        rule_mode="forbidden",
        max_moves=10,
        seed=2,
    )

    assert states.shape[0] > 0
    assert policies.shape == (states.shape[0], BOARD_SIZE * BOARD_SIZE)
    assert values.shape == (states.shape[0], 1)


def test_generate_tactical_dataset_with_auxiliary_labels_and_augmentation(tmp_path):
    from train.tactical_distillation import generate_tactical_dataset

    path = tmp_path / "advanced_tactical.npz"
    states, policies, values = generate_tactical_dataset(
        num_games=1,
        output_path=str(path),
        rule_mode="basic",
        max_moves=4,
        seed=3,
        include_auxiliary_labels=True,
        use_augmentation=True,
    )

    assert states.shape[0] % 8 == 0
    assert policies.shape == (states.shape[0], BOARD_SIZE * BOARD_SIZE)
    assert values.shape == (states.shape[0], 1)
    with np.load(str(path), allow_pickle=False) as data:
        assert data["threat_labels"].shape == (states.shape[0], 12, BOARD_SIZE, BOARD_SIZE)
        assert data["forbidden_labels"].shape == (states.shape[0], 1, BOARD_SIZE, BOARD_SIZE)
        assert data["tactical_scores"].shape == (states.shape[0], BOARD_SIZE * BOARD_SIZE)


def test_tactical_positions_share_player_threat_cache_with_auxiliary_labels(monkeypatch):
    import train.tactical_distillation as tactical_distillation

    captured = []

    def fake_build_auxiliary_labels(board, current_player, rule_mode="basic", **kwargs):
        captured.append(
            {
                "actions": kwargs.get("actions"),
                "threat_cache": kwargs.get("threat_cache"),
            }
        )
        return {
            "threat_labels": np.zeros((12, BOARD_SIZE, BOARD_SIZE), dtype=np.float32),
            "forbidden_labels": np.zeros((1, BOARD_SIZE, BOARD_SIZE), dtype=np.float32),
            "tactical_scores": np.zeros((BOARD_SIZE * BOARD_SIZE,), dtype=np.float32),
        }

    monkeypatch.setattr(
        tactical_distillation,
        "build_auxiliary_labels",
        fake_build_auxiliary_labels,
    )

    tactical_distillation.generate_tactical_positions(
        num_games=1,
        rule_mode="basic",
        max_moves=3,
        seed=11,
        include_auxiliary_labels=True,
        progress_interval=10,
    )

    assert captured
    assert all(entry["actions"] is not None for entry in captured)
    assert any(entry["threat_cache"] for entry in captured)
