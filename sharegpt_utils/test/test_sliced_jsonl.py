import json
from pathlib import Path

import pytest

from sharegpt_utils import ShareGPTMessageDataset
from sliced_jsonl import parse_sliced_jsonl_path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_parse_sliced_jsonl_path_supports_python_style_slice_suffix() -> None:
    spec = parse_sliced_jsonl_path("/tmp/data.jsonl@[1:10:2]")

    assert spec.path == Path("/tmp/data.jsonl")
    assert spec.row_slice == slice(1, 10, 2)


@pytest.mark.parametrize("suffix", ["@[0]", "@[foo:10]", "@[1:2:3:4]", "@[]"])
def test_parse_sliced_jsonl_path_rejects_non_slice_expressions(suffix: str) -> None:
    with pytest.raises(ValueError):
        parse_sliced_jsonl_path(f"/tmp/data.jsonl{suffix}")


def test_sharegpt_message_dataset_applies_jsonl_slice_suffix(tmp_path: Path) -> None:
    data_path = tmp_path / "dataset.jsonl"
    rows = [
        {
            "id": f"sample-{idx}",
            "data_source": "demo",
            "ground_truth": {"label": idx},
            "conversations": [{"from": "human", "value": f"hello {idx}"}],
        }
        for idx in range(5)
    ]
    _write_jsonl(data_path, rows)

    dataset = ShareGPTMessageDataset(f"{data_path}@[1:5:2]")

    assert len(dataset) == 2
    assert dataset[0].id == "sample-1"
    assert dataset[1].id == "sample-3"
    assert dataset.samples[0]["__sharegpt_source_file"] == str(data_path)


def test_sharegpt_message_dataset_accepts_open_ended_slice(tmp_path: Path) -> None:
    data_path = tmp_path / "dataset.jsonl"
    rows = [
        {
            "id": f"sample-{idx}",
            "data_source": "demo",
            "conversations": [{"from": "human", "value": f"hello {idx}"}],
        }
        for idx in range(4)
    ]
    _write_jsonl(data_path, rows)

    dataset = ShareGPTMessageDataset(f"{data_path}@[-2:]")

    assert [dataset[idx].id for idx in range(len(dataset))] == ["sample-2", "sample-3"]


def test_sharegpt_message_dataset_warns_when_slice_stop_exceeds_file_length(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    data_path = tmp_path / "dataset.jsonl"
    rows = [
        {
            "id": f"sample-{idx}",
            "data_source": "demo",
            "conversations": [{"from": "human", "value": f"hello {idx}"}],
        }
        for idx in range(3)
    ]
    _write_jsonl(data_path, rows)

    dataset = ShareGPTMessageDataset(f"{data_path}@[0:10]")

    assert len(dataset) == 3
    assert "truncated sliced dataset" in caplog.text
    assert "requested=10" in caplog.text
    assert "selected=3" in caplog.text


def test_sharegpt_message_dataset_does_not_warn_for_in_range_slice(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    data_path = tmp_path / "dataset.jsonl"
    rows = [
        {
            "id": f"sample-{idx}",
            "data_source": "demo",
            "conversations": [{"from": "human", "value": f"hello {idx}"}],
        }
        for idx in range(5)
    ]
    _write_jsonl(data_path, rows)

    dataset = ShareGPTMessageDataset(f"{data_path}@[1:4]")

    assert len(dataset) == 3
    assert "truncated sliced dataset" not in caplog.text
