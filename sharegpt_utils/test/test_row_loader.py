import json
from pathlib import Path

import pytest

from sharegpt_utils.row_loader import load_sharegpt_rows_from_specs


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_bounded_jsonl_slice_streams_only_requested_rows(tmp_path: Path) -> None:
    data_path = tmp_path / "dataset.jsonl"
    _write_jsonl(
        data_path,
        [
            {"id": "sample-0"},
            {"id": "sample-1"},
        ],
    )
    with data_path.open("a", encoding="utf-8") as f:
        f.write("{not valid json}\n")

    rows = load_sharegpt_rows_from_specs(f"{data_path}@[0:2]", source_file_key="source")

    assert [row["id"] for row in rows] == ["sample-0", "sample-1"]
    assert [row["source"] for row in rows] == [str(data_path), str(data_path)]


def test_slice_truncation_warning_for_quota_style_slice(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    data_path = tmp_path / "dataset.jsonl"
    _write_jsonl(data_path, [{"id": "sample-0"}, {"id": "sample-1"}])

    rows = load_sharegpt_rows_from_specs(f"{data_path}@[0:5]", source_file_key="source")

    assert len(rows) == 2
    assert "truncated sliced dataset" in caplog.text
    assert "requested=5" in caplog.text
    assert "selected=2" in caplog.text


def test_in_range_slice_does_not_warn(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    data_path = tmp_path / "dataset.jsonl"
    _write_jsonl(data_path, [{"id": f"sample-{idx}"} for idx in range(5)])

    rows = load_sharegpt_rows_from_specs(f"{data_path}@[1:4]", source_file_key="source")

    assert [row["id"] for row in rows] == ["sample-1", "sample-2", "sample-3"]
    assert "truncated sliced dataset" not in caplog.text


def test_complex_slice_keeps_python_slice_semantics(tmp_path: Path) -> None:
    data_path = tmp_path / "dataset.jsonl"
    _write_jsonl(data_path, [{"id": f"sample-{idx}"} for idx in range(5)])

    rows = load_sharegpt_rows_from_specs(f"{data_path}@[-3::2]", source_file_key="source")

    assert [row["id"] for row in rows] == ["sample-2", "sample-4"]
    assert all(row["source"] == str(data_path) for row in rows)
