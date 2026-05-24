"""ShareGPT row loading with dataset path slice support."""

from __future__ import annotations

import logging
from pathlib import Path
from .io import iter_sharegpt_jsonl_rows, load_sharegpt_rows

try:
    from ..sliced_jsonl import parse_sliced_jsonl_path
except ImportError:  # pragma: no cover - top-level use inside the standalone repo
    from sliced_jsonl import parse_sliced_jsonl_path

logger = logging.getLogger(__name__)


def load_sharegpt_rows_from_specs(file_specs: str | list[str], source_file_key: str) -> list[dict]:
    """Load rows from one or more ShareGPT file specs and stamp source paths."""
    if isinstance(file_specs, str):
        file_specs = [file_specs]

    all_rows: list[dict] = []
    for file_spec in file_specs:
        rows, path = _load_rows_from_spec(file_spec)
        for row in rows:
            row.setdefault(source_file_key, str(path))
        all_rows.extend(rows)
    return all_rows


def _load_rows_from_spec(file_spec: str) -> tuple[list[dict], Path]:
    sliced_path = parse_sliced_jsonl_path(file_spec)
    path = sliced_path.path
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if sliced_path.row_slice is None:
        return load_sharegpt_rows(path), path

    if _can_stream_bounded_jsonl_slice(path, sliced_path.row_slice):
        rows, total_scanned = _load_bounded_jsonl_slice(path, sliced_path.row_slice)
        _warn_if_slice_truncated(path, sliced_path.row_slice, total_scanned, len(rows))
        return rows, path

    rows = load_sharegpt_rows(path)
    original_len = len(rows)
    rows = rows[sliced_path.row_slice]
    _warn_if_slice_truncated(path, sliced_path.row_slice, original_len, len(rows))
    return rows, path


def _can_stream_bounded_jsonl_slice(path: Path, row_slice: slice) -> bool:
    step = 1 if row_slice.step is None else row_slice.step
    start = 0 if row_slice.start is None else row_slice.start
    return path.suffix.lower() == ".jsonl" and step == 1 and start >= 0 and row_slice.stop is not None and row_slice.stop >= 0


def _load_bounded_jsonl_slice(path: Path, row_slice: slice) -> tuple[list[dict], int]:
    start = 0 if row_slice.start is None else row_slice.start
    stop = row_slice.stop
    assert stop is not None

    rows: list[dict] = []
    scanned = 0
    for index, row in enumerate(iter_sharegpt_jsonl_rows(path, stop=stop)):
        scanned = index + 1
        if index >= start:
            rows.append(row)
    return rows, scanned


def _warn_if_slice_truncated(
    path: Path,
    row_slice: slice,
    total_rows: int,
    selected_rows: int,
) -> None:
    requested_rows = _requested_positive_slice_len(row_slice, total_rows)
    if requested_rows is None or selected_rows >= requested_rows:
        return
    logger.warning(
        "truncated sliced dataset path=%s slice=%s total=%d requested=%d selected=%d",
        path,
        _format_slice(row_slice),
        total_rows,
        requested_rows,
        selected_rows,
    )


def _requested_positive_slice_len(row_slice: slice, total_rows: int) -> int | None:
    """Return requested length for simple forward slices, or None when ambiguous."""
    step = 1 if row_slice.step is None else row_slice.step
    if step <= 0:
        return None
    start = 0 if row_slice.start is None else row_slice.start
    stop = row_slice.stop
    if start < 0 or stop is None or stop < 0:
        return None
    if start >= total_rows:
        return max(0, _ceil_div(stop - start, step))
    if stop <= total_rows:
        return None
    return max(0, _ceil_div(stop - start, step))


def _ceil_div(numerator: int, denominator: int) -> int:
    if numerator <= 0:
        return 0
    return (numerator + denominator - 1) // denominator


def _format_slice(row_slice: slice) -> str:
    parts = [
        "" if row_slice.start is None else str(row_slice.start),
        "" if row_slice.stop is None else str(row_slice.stop),
    ]
    if row_slice.step is not None:
        parts.append(str(row_slice.step))
    return f"[{':'.join(parts)}]"
