"""Small stdout progress helpers for long-running training commands."""

from __future__ import annotations

from datetime import datetime


def format_seconds(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{sec:04.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h{int(minutes):02d}m{sec:04.1f}s"


def progress_print(message: str, prefix: str = "progress") -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{prefix}] {message} time={timestamp}", flush=True)


__all__ = ["format_seconds", "progress_print"]
