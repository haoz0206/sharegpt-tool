"""Path helpers for deterministic row slicing of JSONL-style datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SlicedJsonlPath:
    """A dataset path plus an optional Python-style row slice."""

    path: Path
    row_slice: slice | None = None


def parse_sliced_jsonl_path(spec: str | Path) -> SlicedJsonlPath:
    """Parse ``path@[start:stop:step]`` without evaluating arbitrary Python.

    The slice suffix is intentionally limited to Python's colon-based slice
    syntax. Examples: ``@[0:512]``, ``@[:512]``, ``@[-100:]``, ``@[::2]``.
    Plain paths are returned unchanged.
    """
    text = str(spec)
    marker = "@["
    if marker not in text:
        return SlicedJsonlPath(Path(text))
    if not text.endswith("]"):
        raise ValueError(f"Invalid sliced JSONL path {text!r}: missing closing ']'")

    path_text, slice_text = text.rsplit(marker, 1)
    slice_text = slice_text[:-1]
    if not path_text:
        raise ValueError(f"Invalid sliced JSONL path {text!r}: empty path")
    return SlicedJsonlPath(Path(path_text), _parse_slice(slice_text, text))


def _parse_slice(expr: str, original: str) -> slice:
    parts = expr.split(":")
    if len(parts) not in (2, 3) or all(part == "" for part in parts):
        raise ValueError(
            f"Invalid sliced JSONL path {original!r}: expected [start:stop] or [start:stop:step]"
        )
    return slice(*(_parse_slice_int(part, original) for part in parts))


def _parse_slice_int(text: str, original: str) -> int | None:
    if text == "":
        return None
    try:
        return int(text, 10)
    except ValueError as exc:
        raise ValueError(
            f"Invalid sliced JSONL path {original!r}: slice bounds must be integers"
        ) from exc
