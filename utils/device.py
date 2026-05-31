"""Device selection helpers for training and smoke tests."""

from __future__ import annotations

from typing import Any

import torch


def get_device(
    preferred: str | torch.device = "cuda",
    allow_cpu_fallback: bool = False,
) -> torch.device:
    """Resolve a torch device with explicit CUDA fallback behavior."""
    device = torch.device(preferred)
    if device.type == "cuda":
        safe = _safe_cuda_available()
        if safe:
            return torch.device("cuda")
        if allow_cpu_fallback:
            print("WARNING: CUDA requested but unavailable; falling back to CPU.")
            return torch.device("cpu")
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. "
            "Use --allow-cpu-fallback or --device cpu for functional smoke tests."
        )
    return device


def _safe_cuda_available(timeout: float = 3.0) -> bool:
    """Return ``True`` if CUDA is usable without hanging the process.

    ``torch.cuda.is_available()`` can hang indefinitely on Windows when the
    NVIDIA driver is in a bad state.  This helper runs the check in a daemon
    thread with a short watchdog so callers never block.
    """
    import threading

    result: dict = {"done": False, "value": False}

    def _probe() -> None:
        try:
            result["value"] = torch.cuda.is_available()
        except Exception:
            result["value"] = False
        finally:
            result["done"] = True

    t = threading.Thread(target=_probe, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if not result.get("done", False):
        return False
    return bool(result.get("value", False))


def assert_cuda_available() -> torch.device:
    """Return the CUDA device or raise a clear error."""
    return get_device("cuda", allow_cpu_fallback=False)


def describe_device(device: str | torch.device) -> str:
    """Return a short human-readable device description."""
    device = torch.device(device)
    if device.type == "cuda" and _safe_cuda_available():
        index = 0 if device.index is None else int(device.index)
        return f"cuda:{index} ({torch.cuda.get_device_name(index)})"
    return device.type


def move_batch_to_device(batch: Any, device: str | torch.device) -> Any:
    """Recursively move tensors inside common batch containers to ``device``."""
    resolved = torch.device(device)
    if torch.is_tensor(batch):
        return batch.to(resolved)
    if isinstance(batch, dict):
        return {key: move_batch_to_device(value, resolved) for key, value in batch.items()}
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(item, resolved) for item in batch)
    if isinstance(batch, list):
        return [move_batch_to_device(item, resolved) for item in batch]
    return batch


__all__ = [
    "get_device",
    "assert_cuda_available",
    "describe_device",
    "move_batch_to_device",
]
