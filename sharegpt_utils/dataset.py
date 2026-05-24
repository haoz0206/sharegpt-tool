from .config import ShareGPTDatasetConfig
from .parser import ShareGPTParser
from .row_loader import load_sharegpt_rows_from_specs


class ShareGPTMessageDataset:
    """Framework-agnostic ShareGPT dataset that returns normalized messages."""

    def __init__(self, data_files: str | list[str], config: ShareGPTDatasetConfig | None = None):
        """Load one or more ShareGPT files and prepare a parser for row access."""
        if isinstance(data_files, str):
            data_files = [data_files]

        self.data_files = [str(path) for path in data_files]
        self.config = config or ShareGPTDatasetConfig()
        self.parser = ShareGPTParser(self.config)
        self.samples = self._load_samples(self.data_files)

    def _load_samples(self, files: list[str]) -> list[dict]:
        """Read raw rows and stamp each one with its source file path."""
        return load_sharegpt_rows_from_specs(files, source_file_key=self.config.source_file_key)

    def __len__(self) -> int:
        """Return the number of raw ShareGPT rows loaded into memory."""
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        """Parse one raw row into the normalized intermediate sample schema."""
        return self.parser.parse_sample(self.samples[index], index)
