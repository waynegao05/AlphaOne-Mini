"""Text file helpers for CCGC record files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


DEFAULT_RECORD_ENCODING = "gb2312"
DEFAULT_FALLBACK_ENCODINGS = ("gbk", "utf-8")


def read_record_file(
    path: str | Path,
    encoding: str = DEFAULT_RECORD_ENCODING,
    fallback_encodings: Iterable[str] = DEFAULT_FALLBACK_ENCODINGS,
) -> str:
    """Read a record text file, preferring GB2312 as specified by CCGC."""
    file_path = Path(path)
    encodings = [encoding]
    encodings.extend(enc for enc in fallback_encodings if enc not in encodings)
    last_error: UnicodeDecodeError | None = None
    for enc in encodings:
        try:
            return file_path.read_text(encoding=enc)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return file_path.read_text(encoding=encoding)


def write_record_file(
    path: str | Path,
    text: str,
    encoding: str = DEFAULT_RECORD_ENCODING,
) -> None:
    """Write a record text file using the requested encoding."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(text, encoding=encoding)


__all__ = [
    "DEFAULT_FALLBACK_ENCODINGS",
    "DEFAULT_RECORD_ENCODING",
    "read_record_file",
    "write_record_file",
]
