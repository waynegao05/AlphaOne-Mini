"""Tests for AlphaZero-style board/policy symmetry augmentation."""

from __future__ import annotations

import numpy as np

from game.board import BOARD_SIZE
from game.encoder import action_to_index, index_to_action


def _state_with_last_move(x: int, y: int) -> np.ndarray:
    state = np.zeros((4, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    state[0, y, x] = 1.0
    state[2, y, x] = 1.0
    state[3, :, :] = 1.0
    return state


def _policy_at(x: int, y: int) -> np.ndarray:
    policy = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
    policy[action_to_index(x, y)] = 1.0
    return policy


def test_transform_action_keeps_action_index_mapping_consistent():
    from selfplay.data_augmentation import transform_action

    a1 = action_to_index(0, 0)
    assert transform_action(a1, rotation=0, flip=False) == action_to_index(0, 0)
    assert transform_action(a1, rotation=1, flip=False) == action_to_index(0, 14)
    assert transform_action(a1, rotation=2, flip=False) == action_to_index(14, 14)
    assert transform_action(a1, rotation=3, flip=False) == action_to_index(14, 0)
    assert transform_action(a1, rotation=0, flip=True) == action_to_index(14, 0)

    h8 = action_to_index(7, 7)
    for rotation in range(4):
        for flip in (False, True):
            assert transform_action(h8, rotation=rotation, flip=flip) == h8


def test_transform_state_policy_keeps_last_move_and_policy_aligned():
    from selfplay.data_augmentation import transform_state_policy

    state = _state_with_last_move(2, 4)
    policy = _policy_at(2, 4)

    new_state, new_policy = transform_state_policy(
        state,
        policy,
        rotation=1,
        flip=True,
    )
    new_action = int(np.argmax(new_policy))
    x, y = index_to_action(new_action)

    assert new_state.shape == state.shape
    assert new_policy.shape == policy.shape
    assert new_policy.sum() == np.float32(1.0)
    assert new_state[0, y, x] == np.float32(1.0)
    assert new_state[2, y, x] == np.float32(1.0)


def test_augment_state_policy_returns_eight_symmetries():
    from selfplay.data_augmentation import augment_state_policy

    state = _state_with_last_move(1, 3)
    policy = _policy_at(1, 3)

    augmented = augment_state_policy(state, policy)

    assert len(augmented) == 8
    actions = {int(np.argmax(policy_aug)) for _, policy_aug in augmented}
    assert len(actions) == 8
    for state_aug, policy_aug in augmented:
        assert state_aug.shape == (4, BOARD_SIZE, BOARD_SIZE)
        assert policy_aug.shape == (BOARD_SIZE * BOARD_SIZE,)
        assert state_aug.dtype == np.float32
        assert policy_aug.dtype == np.float32
        np.testing.assert_allclose(policy_aug.sum(), 1.0)


def test_augment_batch_expands_states_policies_and_values():
    from selfplay.data_augmentation import augment_batch

    states = np.stack([_state_with_last_move(1, 3), _state_with_last_move(5, 6)])
    policies = np.stack([_policy_at(1, 3), _policy_at(5, 6)])
    values = np.array([[1.0], [-1.0]], dtype=np.float32)

    aug_states, aug_policies, aug_values = augment_batch(states, policies, values)

    assert aug_states.shape == (16, 4, BOARD_SIZE, BOARD_SIZE)
    assert aug_policies.shape == (16, BOARD_SIZE * BOARD_SIZE)
    assert aug_values.shape == (16, 1)
    assert aug_states.dtype == np.float32
    assert aug_policies.dtype == np.float32
    assert aug_values.dtype == np.float32
    np.testing.assert_allclose(aug_policies.sum(axis=1), np.ones(16))
    assert set(float(v[0]) for v in aug_values[:8]) == {1.0}
    assert set(float(v[0]) for v in aug_values[8:]) == {-1.0}
