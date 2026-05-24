# 002 Stream Sliced JSONL Loader

Status: done

## Problem

`path@[0:512]` currently slices after the entire JSONL file is loaded. This is
correct but wastes IO and memory for large training files.

## Desired Outcome

Simple bounded forward JSONL slices stream only the requested rows. Complex
slice forms keep Python slice semantics by falling back to full-file loading.

## Acceptance

- `file.jsonl@[0:512]` returns the first 512 rows without loading the whole file.
- Truncated quota-style slices warn once through module logging.
- Source file metadata uses the real file path, not the slice suffix.
