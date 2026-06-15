# ShareGPT Dataset Utils

This submodule is a self-contained ShareGPT dataset utility package. It owns
row loading, ShareGPT message normalization, media/reference handling, and
explicit adapters for training frameworks.

## Public Modules

- `sharegpt_utils`: framework-agnostic parser, config, and dataset classes.
- `sliced_jsonl.py`: parser for `path@[start:stop:step]` dataset slice specs.
- `agent_dataset.py` and `rlvr_dataset.py`: explicit verl adapters.

The root package exports only framework-agnostic utilities. Runtime adapters
should be imported through their explicit module paths.

## Development

Run submodule tests from this directory:

```bash
rtk /workspace/PointCycle/.venv/bin/python -m pytest -q sharegpt_utils/test
rtk /workspace/PointCycle/.venv/bin/python -m compileall -q .
rtk git diff --check
```

## Environment Variables

- `SHAREGPT_LOAD_TRUNCATED_IMAGES` (default OFF): when set to `1`/`true`/`yes`,
  `mm_utils` enables `PIL.ImageFile.LOAD_TRUNCATED_IMAGES`, tolerating web-scale
  JPEG/PNG files with a few truncated trailing bytes instead of raising. Read
  ONCE at import (set it before importing the package), and it flips a
  PROCESS-GLOBAL PIL flag affecting all decoding in the host process. Whether
  truncation tolerance should become the default is still pending.

## Docs

- [PRD: ShareGPT Dataset Utils](prd-sharegpt-dataset-utils.md)
- [Issue Tracker](issues/README.md)
