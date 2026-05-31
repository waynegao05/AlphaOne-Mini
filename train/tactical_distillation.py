"""Generate supervised data from the tactical player."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from engine.tactical_player import TacticalPlayer
from game.board import BLACK, BOARD_SIZE, WHITE, Board
from game.encoder import encode_board, index_to_action
from game.rules_basic import _find_any_winner, check_winner, is_game_over
from game.rules_forbidden import get_game_result_forbidden
from selfplay.data_augmentation import augment_training_batch
from train.auxiliary_labels import build_auxiliary_labels
from train.progress import format_seconds, progress_print


DEFAULT_TACTICAL_DATA_PATH = os.path.join(
    "outputs", "supervised", "tactical_distill_latest.npz"
)


@dataclass
class TacticalSample:
    state: np.ndarray
    action: int
    current_player: int
    value: float = 0.0
    threat_labels: np.ndarray | None = None
    forbidden_labels: np.ndarray | None = None
    tactical_scores: np.ndarray | None = None


def make_policy_target(
    action: int,
    board_size: int = BOARD_SIZE,
    smoothing: float = 0.0,
) -> np.ndarray:
    """Create a one-hot or label-smoothed policy target."""
    action_size = board_size * board_size
    action = int(action)
    if not (0 <= action < action_size):
        raise ValueError(f"action out of range: {action}")
    smoothing = float(smoothing)
    if not (0.0 <= smoothing < 1.0):
        raise ValueError("smoothing must be in [0, 1)")

    policy = np.full(action_size, smoothing / max(1, action_size - 1), dtype=np.float32)
    policy[action] = np.float32(1.0 - smoothing)
    # Normalize once to avoid tiny float drift when smoothing is non-zero.
    policy /= np.float32(policy.sum())
    return policy


def _game_status(board: Board, rule_mode: str) -> tuple[bool, int]:
    if rule_mode == "forbidden":
        result = get_game_result_forbidden(board, board.last_move)
        if result.is_over:
            return True, 0 if result.winner is None else int(result.winner)
        return False, 0
    if rule_mode != "basic":
        raise ValueError(f"unknown rule_mode: {rule_mode!r}")

    if is_game_over(board, board.last_move):
        winner = check_winner(board, board.last_move)
        if winner == 0:
            winner = _find_any_winner(board)
        return True, int(winner)
    return False, 0


def _sample_value(winner: int, current_player: int) -> float:
    if winner == 0:
        return 0.0
    return 1.0 if winner == current_player else -1.0


def generate_tactical_positions(
    num_games: int,
    rule_mode: str = "basic",
    max_moves: int = BOARD_SIZE * BOARD_SIZE,
    seed: Optional[int] = None,
    include_auxiliary_labels: bool = False,
    progress_interval: int | None = None,
) -> list[TacticalSample]:
    """Play TacticalPlayer-vs-TacticalPlayer games and return samples.

    Each sample stores the encoded board before the tactical move, the tactical
    action, and the player-to-move perspective used later for value labels.
    """
    if num_games < 0:
        raise ValueError("num_games must be non-negative")
    if max_moves <= 0:
        raise ValueError("max_moves must be positive")

    samples: list[TacticalSample] = []
    total_games = int(num_games)
    if progress_interval is None:
        progress_interval = 1 if total_games <= 20 else 10
    progress_interval = max(1, int(progress_interval))
    start_time = time.perf_counter()
    progress_print(
        f"START tactical_positions games={total_games} rule_mode={rule_mode} "
        f"max_moves={int(max_moves)} auxiliary={bool(include_auxiliary_labels)}",
        "distill",
    )
    for game_idx in range(total_games):
        game_start = time.perf_counter()
        board = Board()
        player = TacticalPlayer(
            name="TacticalDistiller",
            rule_mode=rule_mode,
            random_tie_break=False,
            seed=None if seed is None else seed + game_idx,
        )
        game_samples: list[TacticalSample] = []

        for _ in range(int(max_moves)):
            over, winner = _game_status(board, rule_mode)
            if over:
                break

            current_player = int(board.current_player)
            state = encode_board(board, current_player=current_player)
            decision = player.select_action_with_context(board)
            action = decision.action
            aux = (
                build_auxiliary_labels(
                    board,
                    current_player,
                    rule_mode,
                    actions=decision.candidates,
                    threat_cache=decision.threat_cache,
                )
                if include_auxiliary_labels
                else {}
            )
            if action is None:
                winner = 0
                break
            x, y = index_to_action(int(action), BOARD_SIZE)
            if not board.is_legal_move(x, y):
                raise RuntimeError(f"TacticalPlayer returned illegal action {action}")

            game_samples.append(
                TacticalSample(
                    state=state.astype(np.float32, copy=False),
                    action=int(action),
                    current_player=current_player,
                    threat_labels=aux.get("threat_labels"),
                    forbidden_labels=aux.get("forbidden_labels"),
                    tactical_scores=aux.get("tactical_scores"),
                )
            )
            board.place_stone(x, y)
        else:
            winner = 0

        over, final_winner = _game_status(board, rule_mode)
        if over:
            winner = final_winner
        for sample in game_samples:
            sample.value = _sample_value(winner, sample.current_player)
        samples.extend(game_samples)
        game_number = game_idx + 1
        if (
            game_number == 1
            or game_number == total_games
            or game_number % progress_interval == 0
        ):
            progress_print(
                f"game {game_number}/{total_games} complete "
                f"samples={len(game_samples)} total_samples={len(samples)} "
                f"winner={winner} moves={len(game_samples)} "
                f"elapsed={format_seconds(time.perf_counter() - game_start)}",
                "distill",
            )

    progress_print(
        f"DONE tactical_positions games={total_games} total_samples={len(samples)} "
        f"elapsed={format_seconds(time.perf_counter() - start_time)}",
        "distill",
    )
    return samples


def _samples_to_arrays(
    samples: list[TacticalSample],
    smoothing: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not samples:
        return (
            np.zeros((0, 4, BOARD_SIZE, BOARD_SIZE), dtype=np.float32),
            np.zeros((0, BOARD_SIZE * BOARD_SIZE), dtype=np.float32),
            np.zeros((0, 1), dtype=np.float32),
        )
    states = np.stack([sample.state for sample in samples]).astype(np.float32)
    policies = np.stack(
        [make_policy_target(sample.action, smoothing=smoothing) for sample in samples]
    ).astype(np.float32)
    values = np.asarray([[sample.value] for sample in samples], dtype=np.float32)
    return states, policies, values


def _samples_to_array_dict(
    samples: list[TacticalSample],
    smoothing: float = 0.0,
    include_auxiliary_labels: bool = False,
) -> dict[str, np.ndarray]:
    states, policies, values = _samples_to_arrays(samples, smoothing=smoothing)
    arrays = {"states": states, "policies": policies, "values": values}
    if include_auxiliary_labels:
        if samples:
            arrays["threat_labels"] = np.stack(
                [sample.threat_labels for sample in samples]
            ).astype(np.float32)
            arrays["forbidden_labels"] = np.stack(
                [sample.forbidden_labels for sample in samples]
            ).astype(np.float32)
            arrays["tactical_scores"] = np.stack(
                [sample.tactical_scores for sample in samples]
            ).astype(np.float32)
        else:
            arrays["threat_labels"] = np.zeros((0, 12, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
            arrays["forbidden_labels"] = np.zeros((0, 1, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
            arrays["tactical_scores"] = np.zeros((0, BOARD_SIZE * BOARD_SIZE), dtype=np.float32)
    return arrays


def save_tactical_dataset(
    states: np.ndarray,
    policies: np.ndarray,
    values: np.ndarray,
    path: str,
    threat_labels: np.ndarray | None = None,
    forbidden_labels: np.ndarray | None = None,
    tactical_scores: np.ndarray | None = None,
) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = {
        "states": np.asarray(states, dtype=np.float32),
        "policies": np.asarray(policies, dtype=np.float32),
        "values": np.asarray(values, dtype=np.float32),
    }
    if threat_labels is not None:
        payload["threat_labels"] = np.asarray(threat_labels, dtype=np.float32)
    if forbidden_labels is not None:
        payload["forbidden_labels"] = np.asarray(forbidden_labels, dtype=np.float32)
    if tactical_scores is not None:
        payload["tactical_scores"] = np.asarray(tactical_scores, dtype=np.float32)
    np.savez_compressed(path, **payload)


def load_tactical_dataset(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"tactical dataset not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        for key in ("states", "policies", "values"):
            if key not in data.files:
                raise KeyError(f"dataset missing key {key!r}: {path}")
        states = np.asarray(data["states"], dtype=np.float32)
        policies = np.asarray(data["policies"], dtype=np.float32)
        values = np.asarray(data["values"], dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if states.ndim != 4 or states.shape[1:] != (4, BOARD_SIZE, BOARD_SIZE):
        raise ValueError(f"invalid states shape: {states.shape}")
    if policies.ndim != 2 or policies.shape[1] != BOARD_SIZE * BOARD_SIZE:
        raise ValueError(f"invalid policies shape: {policies.shape}")
    if values.ndim != 2 or values.shape[1] != 1:
        raise ValueError(f"invalid values shape: {values.shape}")
    if not (states.shape[0] == policies.shape[0] == values.shape[0]):
        raise ValueError("states, policies and values must have the same length")
    return states, policies, values


def generate_tactical_dataset(
    num_games: int,
    output_path: str = DEFAULT_TACTICAL_DATA_PATH,
    rule_mode: str = "basic",
    temperature: float = 0.1,
    max_moves: int = BOARD_SIZE * BOARD_SIZE,
    seed: Optional[int] = None,
    smoothing: Optional[float] = None,
    include_auxiliary_labels: bool = False,
    use_augmentation: bool = False,
    progress_interval: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate, save, and return tactical distillation arrays."""
    if smoothing is None:
        smoothing = max(0.0, min(0.25, float(temperature)))
    samples = generate_tactical_positions(
        num_games=num_games,
        rule_mode=rule_mode,
        max_moves=max_moves,
        seed=seed,
        include_auxiliary_labels=include_auxiliary_labels,
        progress_interval=progress_interval,
    )
    arrays = _samples_to_array_dict(
        samples,
        smoothing=float(smoothing),
        include_auxiliary_labels=include_auxiliary_labels,
    )
    if use_augmentation:
        arrays = augment_training_batch(
            arrays["states"],
            arrays["policies"],
            arrays["values"],
            threat_labels=arrays.get("threat_labels"),
            forbidden_labels=arrays.get("forbidden_labels"),
            tactical_scores=arrays.get("tactical_scores"),
        )
    save_tactical_dataset(
        arrays["states"],
        arrays["policies"],
        arrays["values"],
        output_path,
        threat_labels=arrays.get("threat_labels"),
        forbidden_labels=arrays.get("forbidden_labels"),
        tactical_scores=arrays.get("tactical_scores"),
    )
    return arrays["states"], arrays["policies"], arrays["values"]


__all__ = [
    "DEFAULT_TACTICAL_DATA_PATH",
    "TacticalSample",
    "generate_tactical_positions",
    "generate_tactical_dataset",
    "make_policy_target",
    "save_tactical_dataset",
    "load_tactical_dataset",
]
