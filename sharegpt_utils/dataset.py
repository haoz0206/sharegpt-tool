from pathlib import Path

from .config import ShareGPTDatasetConfig
from .io import load_sharegpt_rows
from .parser import ShareGPTParser


class ShareGPTMessageDataset:
    """Framework-agnostic ShareGPT dataset that returns normalized messages."""

    def __init__(self, data_files: str | list[str], config: ShareGPTDatasetConfig | None = None):
        """Load one or more ShareGPT files and prepare a parser for row access."""
        if isinstance(data_files, str):
            data_files = [data_files]

        self.data_files = [str(Path(path)) for path in data_files]
        self.config = config or ShareGPTDatasetConfig()
        self.parser = ShareGPTParser(self.config)
        self.samples = self._load_samples(self.data_files)

    def _load_samples(self, files: list[str]) -> list[dict]:
        """Read raw rows and stamp each one with its source file path."""
        all_rows: list[dict] = []
        for file_path in files:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"Data file not found: {path}")

            rows = load_sharegpt_rows(path)
            for row in rows:
                row.setdefault(self.config.source_file_key, str(path))
            all_rows.extend(rows)
        return all_rows

    def __len__(self) -> int:
        """Return the number of raw ShareGPT rows loaded into memory."""
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        """Parse one raw row into the normalized intermediate sample schema."""
        return self.parser.parse_sample(self.samples[index], index)
