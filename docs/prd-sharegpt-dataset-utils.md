# PRD: ShareGPT Dataset Utils

## Goal

Maintain this submodule as a self-contained ShareGPT dataset utility package
with a small framework-agnostic core and explicit runtime adapters.

## Current Problems

- Root package boundaries, bounded JSONL slice loading, and import-time logging
  side effects have been cleaned up.
- `ShareGPTParser` owns several responsibilities and should eventually be
  split, but that refactor is intentionally out of scope for the current pass.

## Success Criteria

- Core imports do not require verl, torch, ray, or transformers.
- Runtime adapters remain available through explicit module imports.
- Bounded forward JSONL slices can be read without loading the full file.
- Importing the package does not print to stdout/stderr.
- Documentation and issues in `docs/` describe this submodule only.

## Non-goals

- Do not redesign the ShareGPT sample schema.
- Do not split `ShareGPTParser` in this cleanup pass.
- Do not move PointCycle training policy or launch documentation into this
  submodule.
