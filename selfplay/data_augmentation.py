"""Board symmetry augmentation for AlphaZero-style training samples."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from game.board import BOARD_SIZE
from game.encoder import action_to_index, index_to_action


def _validate_rotation(rotation: int) -> int:
    rotation = int(rotation)
    if rotation not in (0, 1, 2, 3):
        raise ValueError("rotation must be one of 0, 1, 2, 3")
    return rotation


def _transform_plane(
    plane: np.ndarray,
    rotation: int,
    flip: bool,
) -> np.ndarray:
    transformed = np.rot90(plane, k=rotation, axes=(-2, -1))
    if flip:
        transformed = np.flip(transformed, axis=-1)
    return transformed.copy()


def transform_action(
    action: int,
    rotation: int = 0,
    flip: bool = False,
    board_size: int = BOARD_SIZE,
) -> int:
    """Transform an action index with the same mapping used for policies."""
    rotation = _validate_rotation(rotation)
    x, y = index_to_action(int(action), board_size)
    marker = np.zeros((board_size, board_size), dtype=np.float32)
    marker[y, x] = 1.0
    transformed = _transform_plane(marker, rotation, bool(flip))
    new_y, new_x = np.argwhere(transformed > 0.5)[0]
    return action_to_index(int(new_x), int(new_y), board_size)


def transform_policy(
    policy: np.ndarray,
    rotation: int = 0,
    flip: bool = False,
    board_size: int = BOARD_SIZE,
) -> np.ndarray:
    """Transform a flat `[board_size * board_size]` policy distribution."""
    return transform_flat_board(
        policy,
        rotation=rotation,
        flip=flip,
        board_size=board_size,
        normalize=True,
        name="policy",
    )


def transform_flat_board(
    values: np.ndarray,
    rotation: int = 0,
    flip: bool = False,
    board_size: int = BOARD_SIZE,
    normalize: bool = False,
    name: str = "values",
) -> np.ndarray:
    """Transform a flat board-shaped vector while preserving action mapping."""
    rotation = _validate_rotation(rotation)
    values_np = np.asarray(values, dtype=np.float32)
    if values_np.shape != (board_size * board_size,):
        raise ValueError(f"{name} shape must be ({board_size * board_size},)")
    values_2d = values_np.reshape(board_size, board_size)
    transformed = _transform_plane(values_2d, rotation, bool(flip)).reshape(-1)
    total = float(transformed.sum())
    if normalize and total > 0:
        transformed = transformed / np.float32(total)
    return transformed.astype(np.float32, copy=False)


def transform_state(
    state: np.ndarray,
    rotation: int = 0,
    flip: bool = False,
    board_size: int = BOARD_SIZE,
) -> np.ndarray:
    """Transform a `[4, 15, 15]` encoded board state."""
    rotation = _validate_rotation(rotation)
    state_np = np.asarray(state, dtype=np.float32)
    if state_np.shape != (4, board_size, board_size):
        raise ValueError(f"state shape must be (4, {board_size}, {board_size})")
    return _transform_plane(state_np, rotation, bool(flip)).astype(np.float32, copy=False)


def transform_plane_stack(
    planes: np.ndarray,
    rotation: int = 0,
    flip: bool = False,
    board_size: int = BOARD_SIZE,
    name: str = "planes",
) -> np.ndarray:
    """Transform a `[channels, board_size, board_size]` plane stack."""
    rotation = _validate_rotation(rotation)
    planes_np = np.asarray(planes, dtype=np.float32)
    if planes_np.ndim != 3 or planes_np.shape[1:] != (board_size, board_size):
        raise ValueError(f"{name} shape must be (C, {board_size}, {board_size})")
    return _transform_plane(planes_np, rotation, bool(flip)).astype(np.float32, copy=False)


def transform_state_policy(
    state: np.ndarray,
    policy: np.ndarray,
    rotation: int = 0,
    flip: bool = False,
    board_size: int = BOARD_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform state and policy with the same board symmetry."""
    return (
        transform_state(state, rotation, flip, board_size),
        transform_policy(policy, rotation, flip, board_size),
    )


def symmetry_specs() -> list[tuple[int, bool]]:
    """Return the 8 standard square-board symmetries used for augmentation."""
    return [(rotation, flip) for rotation in range(4) for flip in (False, True)]


def augment_state_policy(
    state: np.ndarray,
    policy: np.ndarray,
    board_size: int = BOARD_SIZE,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return all 8 transformed `(state, policy)` pairs."""
    return [
        transform_state_policy(state, policy, rotation, flip, board_size)
        for rotation, flip in symmetry_specs()
    ]


def augment_sample(
    state: np.ndarray,
    policy: np.ndarray,
    value: np.ndarray | float,
    board_size: int = BOARD_SIZE,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return all 8 transformed samples while preserving the value target."""
    value_np = np.asarray(value, dtype=np.float32)
    if value_np.shape == ():
        value_np = value_np.reshape(1)
    return [
        (state_aug, policy_aug, value_np.astype(np.float32, copy=True))
        for state_aug, policy_aug in augment_state_policy(state, policy, board_size)
    ]


def augment_batch(
    states: np.ndarray,
    policies: np.ndarray,
    values: np.ndarray,
    board_size: int = BOARD_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand a batch of samples by the 8 board symmetries."""
    states_np = np.asarray(states, dtype=np.float32)
    policies_np = np.asarray(policies, dtype=np.float32)
    values_np = np.asarray(values, dtype=np.float32)
    if values_np.ndim == 1:
        values_np = values_np.reshape(-1, 1)
    if states_np.ndim != 4 or states_np.shape[1:] != (4, board_size, board_size):
        raise ValueError(f"states shape must be (N, 4, {board_size}, {board_size})")
    if policies_np.ndim != 2 or policies_np.shape[1] != board_size * board_size:
        raise ValueError(f"policies shape must be (N, {board_size * board_size})")
    if values_np.ndim != 2 or values_np.shape[1] != 1:
        raise ValueError("values shape must be (N, 1) or (N,)")
    if not (states_np.shape[0] == policies_np.shape[0] == values_np.shape[0]):
        raise ValueError("states, policies and values must have the same length")

    augmented_states: list[np.ndarray] = []
    augmented_policies: list[np.ndarray] = []
    augmented_values: list[np.ndarray] = []
    for state, policy, value in zip(states_np, policies_np, values_np):
        for state_aug, policy_aug in augment_state_policy(state, policy, board_size):
            augmented_states.append(state_aug)
            augmented_policies.append(policy_aug)
            augmented_values.append(value.astype(np.float32, copy=True))

    return (
        np.stack(augmented_states).astype(np.float32),
        np.stack(augmented_policies).astype(np.float32),
        np.stack(augmented_values).reshape(-1, 1).astype(np.float32),
    )


def augment_training_batch(
    states: np.ndarray,
    policies: np.ndarray,
    values: np.ndarray,
    threat_labels: np.ndarray | None = None,
    forbidden_labels: np.ndarray | None = None,
    tactical_scores: np.ndarray | None = None,
    board_size: int = BOARD_SIZE,
) -> dict[str, np.ndarray]:
    """Augment policy-value samples and optional auxiliary labels together."""
    states_np = np.asarray(states, dtype=np.float32)
    policies_np = np.asarray(policies, dtype=np.float32)
    values_np = np.asarray(values, dtype=np.float32)
    if values_np.ndim == 1:
        values_np = values_np.reshape(-1, 1)
    n = states_np.shape[0]
    threats_np = None if threat_labels is None else np.asarray(threat_labels, dtype=np.float32)
    forbidden_np = None if forbidden_labels is None else np.asarray(forbidden_labels, dtype=np.float32)
    scores_np = None if tactical_scores is None else np.asarray(tactical_scores, dtype=np.float32)
    if threats_np is not None and threats_np.shape[0] != n:
        raise ValueError("threat_labels must have the same sample count as states")
    if forbidden_np is not None and forbidden_np.shape[0] != n:
        raise ValueError("forbidden_labels must have the same sample count as states")
    if scores_np is not None and scores_np.shape != policies_np.shape:
        raise ValueError("tactical_scores shape must match policies shape")

    out: dict[str, list[np.ndarray]] = {
        "states": [],
        "policies": [],
        "values": [],
    }
    if threats_np is not None:
        out["threat_labels"] = []
    if forbidden_np is not None:
        out["forbidden_labels"] = []
    if scores_np is not None:
        out["tactical_scores"] = []

    for index in range(n):
        for rotation, flip in symmetry_specs():
            out["states"].append(transform_state(states_np[index], rotation, flip, board_size))
            out["policies"].append(transform_policy(policies_np[index], rotation, flip, board_size))
            out["values"].append(values_np[index].astype(np.float32, copy=True))
            if threats_np is not None:
                out["threat_labels"].append(
                    transform_plane_stack(threats_np[index], rotation, flip, board_size, "threat_labels")
                )
            if forbidden_np is not None:
                out["forbidden_labels"].append(
                    transform_plane_stack(forbidden_np[index], rotation, flip, board_size, "forbidden_labels")
                )
            if scores_np is not None:
                out["tactical_scores"].append(
                    transform_flat_board(
                        scores_np[index],
                        rotation,
                        flip,
                        board_size,
                        normalize=False,
                        name="tactical_scores",
                    )
                )

    return {key: np.stack(value).astype(np.float32) for key, value in out.items()}


__all__ = [
    "transform_action",
    "transform_policy",
    "transform_flat_board",
    "transform_state",
    "transform_plane_stack",
    "transform_state_policy",
    "symmetry_specs",
    "augment_state_policy",
    "augment_sample",
    "augment_batch",
    "augment_training_batch",
]
