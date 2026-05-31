"""Tactical Gomoku engine helpers."""

from .candidate_moves import generate_candidate_moves
from .heuristic import evaluate_move_heuristic
from .hybrid_player import HybridPlayer
from .strong_player import StrongPlayer
from .tactical_player import TacticalPlayer
from .vct_search import (
    find_vct_attack_candidates,
    find_vct_defense_moves,
    vct_defends,
    vct_first_move,
)
from .threats import (
    classify_move_threats,
    find_immediate_blocking_moves,
    find_immediate_winning_moves,
)

# NeuralGuardedPlayer imports torch (and torch.nn.functional) at module
# level.  On Windows machines where the NVIDIA driver is too new for the
# installed PyTorch CUDA libraries, ``import torch`` hangs the process.
# We defer the import so that code paths that only need StrongPlayer /
# TacticalPlayer never touch torch and remain safe.


def __getattr__(name: str):
    if name == "NeuralGuardedPlayer":
        from .neural_guarded_player import NeuralGuardedPlayer as _N

        return _N
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "generate_candidate_moves",
    "evaluate_move_heuristic",
    "HybridPlayer",
    "NeuralGuardedPlayer",
    "StrongPlayer",
    "TacticalPlayer",
    "classify_move_threats",
    "find_immediate_blocking_moves",
    "find_immediate_winning_moves",
    "find_vct_attack_candidates",
    "find_vct_defense_moves",
    "vct_defends",
    "vct_first_move",
]
