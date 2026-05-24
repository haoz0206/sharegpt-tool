import importlib
import json
from pathlib import Path
from typing import Any

try:
    import orjson

    _loads = orjson.loads
    print("[sharegpt_utils.io] Using orjson for JSON parsing.")
except ModuleNotFoundError:
    _loads = json.loads
    print("[sharegpt_utils.io] orjson not found, falling back to stdlib json.")


def load_sharegpt_json(path: Path) -> list[dict[str, Any]]:
    """Load ShareGPT rows from a JSON file with a dict or list-of-dicts root."""
    with path.open("r", encoding="utf-8") as f:
        data = _loads(f.read())

    if isinstance(data, list):
        rows: list[dict[str, Any]] = []
        for index, item in enumerate(data):
            if not isinstance(item, dict):
                raise TypeError(f"Item {index} in {path} is not a JSON object of type dict, got {type(item).__name__}")
            rows.append(item)
        return rows

    if isinstance(data, dict):
        return [data]

    raise TypeError(f"Unsupported JSON root type: {type(data).__name__}")


def load_sharegpt_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load ShareGPT rows from a JSONL file containing one object per line."""
    return list(iter_sharegpt_jsonl_rows(path))


def iter_sharegpt_jsonl_rows(path: Path, stop: int | None = None):
    """Yield JSON object rows from a JSONL file."""
    with path.open("r", encoding="utf-8") as f:
        row_index = 0
        for line_num, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            if stop is not None and row_index >= stop:
                break
            obj = _loads(text)
            if not isinstance(obj, dict):
                raise TypeError(f"Line {line_num} in {path} is not a JSON object of type dict, got {type(obj).__name__}")
            row_index += 1
            yield obj


def load_sharegpt_parquet(path: Path) -> list[dict[str, Any]]:
    """Load ShareGPT rows from a parquet file through Hugging Face Datasets."""
    datasets_module = importlib.import_module("datasets")
    hf_ds = datasets_module.load_dataset("parquet", data_files=str(path))["train"]
    return [hf_ds[i] for i in range(len(hf_ds))]


def load_sharegpt_rows(path: Path) -> list[dict[str, Any]]:
    """Dispatch row loading by file extension."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        return load_sharegpt_json(path)
    if suffix == ".jsonl":
        return load_sharegpt_jsonl(path)
    if suffix == ".parquet":
        return load_sharegpt_parquet(path)
    raise ValueError(f"Unsupported file extension: {path.suffix}")
