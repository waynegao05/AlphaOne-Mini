"""Player factory for local Tkinter Gomoku UI slots."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from game.board import BOARD_SIZE


DEFAULT_TACTICAL_CHECKPOINT = os.path.join(
    "outputs", "checkpoints", "latest_advanced_tactical_restoration_from_curriculum.pt"
)
DEFAULT_V2_CHECKPOINT = os.path.join(
    "outputs", "checkpoints", "latest_advanced_mistake_v2_from_latest.pt"
)
DEFAULT_LATEST_ADVANCED = os.path.join("outputs", "checkpoints", "latest_advanced.pt")
DEFAULT_RENJUNET_EXTERNAL_ADAPTED = os.path.join(
    "outputs", "checkpoints", "pretrained_advanced_renjunet_all_external_adapted.pt"
)
DEFAULT_RENJUNET_ALL = os.path.join(
    "outputs", "checkpoints", "pretrained_advanced_renjunet_all.pt"
)
DEFAULT_RENJUNET_PHASE3 = os.path.join(
    "outputs", "checkpoints", "pretrained_advanced_renjunet_phase3.pt"
)
DEFAULT_EXTERNAL_AI_PATH = os.environ.get(
    "ALPHAONE_EXTERNAL_AI_PATH",
    os.path.join("external_ai", "AI.py"),
)


# Only these choices are exposed in the GUI. Old internal names remain accepted
# by normalize_player_type so existing scripts do not break abruptly.
PLAYER_TYPE_LABELS = {
    "human": "Human",
    "alphaone_mini": "AlphaOne-Mini",
    "external_ai": "External AI.py",
}

DISPLAY_TO_PLAYER_TYPE = {label: key for key, label in PLAYER_TYPE_LABELS.items()}


@dataclass
class PlayerSlot:
    player_type: str
    color: int
    player: object | None
    label: str
    is_human: bool
    is_external: bool = False


def normalize_player_type(value: str) -> str:
    text = str(value).strip()
    if text in PLAYER_TYPE_LABELS:
        return text
    if text in DISPLAY_TO_PLAYER_TYPE:
        return DISPLAY_TO_PLAYER_TYPE[text]
    lowered = text.lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "human": "human",
        "alphaone": "alphaone_mini",
        "alphaonemini": "alphaone_mini",
        "alphaone_mini": "alphaone_mini",
        "alphaone_mini_ai": "alphaone_mini",
        # Compatibility aliases from earlier builds. They now point to the
        # single integrated player: StrongPlayer + best available RenjuNet MCTS.
        "strong": "alphaone_mini",
        "strongplayer": "alphaone_mini",
        "neural_guarded": "alphaone_mini",
        "neuralguarded_no_hybrid_fallback": "alphaone_mini",
        "neural_guarded_no_hybrid_fallback": "alphaone_mini",
        "tactical": "alphaone_mini",
        "tacticalplayer": "alphaone_mini",
        "hybrid": "alphaone_mini",
        "hybridplayer": "alphaone_mini",
        "latest_advanced": "alphaone_mini",
        "latest_advanced_model": "alphaone_mini",
        "external_ai.py": "external_ai",
        "external_ai": "external_ai",
    }
    if lowered in aliases:
        return aliases[lowered]
    raise ValueError(f"unknown player type: {value}")


# ---------------------------------------------------------------------------
# safe torch detection — ``import torch`` can hang the process on Windows
# when the NVIDIA driver is too new for the PyTorch CUDA libraries (e.g.
# driver CUDA 13.0 + PyTorch cu126).  We probe in a **subprocess** so a
# hung import never contaminates this process's import lock.
# ---------------------------------------------------------------------------
_TORCH_IMPORT_SAFE: Optional[bool] = None


def _is_torch_import_safe(timeout: float = 5.0) -> bool:
    """Return ``True`` if ``import torch`` completes without hanging."""
    global _TORCH_IMPORT_SAFE
    if _TORCH_IMPORT_SAFE is not None:
        return _TORCH_IMPORT_SAFE

    import subprocess
    import sys

    try:
        proc = subprocess.run(
            [sys.executable, "-c", "import torch"],
            capture_output=True,
            timeout=float(timeout),
        )
        safe = proc.returncode == 0
    except (subprocess.TimeoutExpired, OSError, ValueError):
        safe = False

    _TORCH_IMPORT_SAFE = safe
    if not safe:
        print(
            "[player_factory] WARNING: 'import torch' timed out after "
            f"{timeout:.0f}s in subprocess — neural MCTS is disabled. "
            "Falling back to tactical engine."
        )
    else:
        print("[player_factory] torch import verified — neural MCTS enabled.")
    return safe


def resolve_device(requested_device: str) -> str:
    """Resolve a device string, returning ``"cpu"`` when CUDA is unusable.

    Uses the subprocess-based :func:`_is_torch_import_safe` guard so that a
    broken CUDA driver never hangs the UI thread.
    """
    if requested_device == "cuda":
        if not _is_torch_import_safe():
            return "cpu"
        try:
            import torch

            cuda_available = False
            import threading

            result_holder: dict = {"done": False, "value": False}

            def _probe() -> None:
                try:
                    result_holder["value"] = torch.cuda.is_available()
                except Exception:
                    result_holder["value"] = False
                finally:
                    result_holder["done"] = True

            t = threading.Thread(target=_probe, daemon=True)
            t.start()
            t.join(timeout=3.0)
            cuda_available = result_holder.get("value", False)
            if not result_holder.get("done", False):
                return "cpu"
            return "cuda" if cuda_available else "cpu"
        except Exception:
            return "cpu"
    return requested_device


def create_player(
    player_type: str,
    color: int,
    *,
    rule_mode: str = "basic",
    device: str = "cpu",
    external_ai_path: Optional[str] = None,
    num_simulations: int = 50,
):
    normalized = normalize_player_type(player_type)
    label = PLAYER_TYPE_LABELS[normalized]
    resolved_device = resolve_device(device)

    if normalized == "human":
        return PlayerSlot(normalized, int(color), None, label, True)

    if normalized == "alphaone_mini":
        from engine.strong_player import StrongPlayer

        mcts_player = None
        if _is_torch_import_safe():
            try:
                mcts_player = _build_alphaone_renjunet_player(
                    resolved_device, int(num_simulations)
                )
            except Exception:
                # If model loading fails (missing checkpoint, etc.), fall back
                # to pure tactical play.
                mcts_player = None

        # --- interactive-GUI-friendly search limits ---
        # Tournament settings (depth 9 / budget 20k) are far too heavy for a
        # responsive GUI.  These reduced values keep the AI responsive (<500ms)
        # on an open board while staying reasonably strong.
        player = StrongPlayer(
            mcts_player=mcts_player,
            rule_mode=rule_mode,
            vcf_depth=4,
            vcf_defense_depth=2,
            vcf_node_budget=2000,
            vct_depth=3,
            vct_node_budget=2000,
            lookahead_depth=1,
            lookahead_branch_factor=2,
            name="AlphaOne-Mini",
        )
        return PlayerSlot(normalized, int(color), player, label, False)

    if normalized == "external_ai":
        from engine.external_ai_adapter import ExternalAIAdapter

        path = external_ai_path or DEFAULT_EXTERNAL_AI_PATH
        player = ExternalAIAdapter(path, rule_mode=rule_mode, board_size=BOARD_SIZE)
        return PlayerSlot(normalized, int(color), player, label, False, is_external=True)

    raise ValueError(f"unknown player type: {player_type}")


def _build_latest_advanced_player(device: str, num_simulations: int):
    return _build_checkpoint_player(
        DEFAULT_LATEST_ADVANCED,
        device,
        num_simulations,
        name="latest_advanced",
        fallback_model_type="advanced",
    )


def _build_alphaone_renjunet_player(device: str, num_simulations: int):
    checkpoint_path = DEFAULT_RENJUNET_EXTERNAL_ADAPTED
    if not os.path.exists(checkpoint_path):
        checkpoint_path = DEFAULT_RENJUNET_ALL
    if not os.path.exists(checkpoint_path):
        checkpoint_path = DEFAULT_RENJUNET_PHASE3
    if not os.path.exists(checkpoint_path):
        checkpoint_path = DEFAULT_LATEST_ADVANCED
    return _build_checkpoint_player(
        checkpoint_path,
        device,
        num_simulations,
        name="alphaone_renjunet_mcts",
        fallback_model_type="advanced",
    )


def _build_checkpoint_player(
    checkpoint_path: str,
    device: str,
    num_simulations: int,
    *,
    name: str,
    fallback_model_type: str = "advanced",
):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    from evaluate.players import ModelMCTSPlayer
    from model.checkpoint import load_checkpoint
    from model.model_factory import create_model_from_metadata
    import torch

    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=device)
    model = create_model_from_metadata(
        state.get("metadata", {}),
        fallback_model_type=fallback_model_type,
    )
    load_checkpoint(model, checkpoint_path, device=device)
    model.eval()
    return ModelMCTSPlayer(
        model=model,
        num_simulations=int(num_simulations),
        device=device,
        name=name,
    )


__all__ = [
    "DEFAULT_EXTERNAL_AI_PATH",
    "DEFAULT_LATEST_ADVANCED",
    "DEFAULT_RENJUNET_EXTERNAL_ADAPTED",
    "DEFAULT_RENJUNET_ALL",
    "DEFAULT_RENJUNET_PHASE3",
    "DEFAULT_TACTICAL_CHECKPOINT",
    "DEFAULT_V2_CHECKPOINT",
    "DISPLAY_TO_PLAYER_TYPE",
    "PLAYER_TYPE_LABELS",
    "PlayerSlot",
    "create_player",
    "normalize_player_type",
    "resolve_device",
]
