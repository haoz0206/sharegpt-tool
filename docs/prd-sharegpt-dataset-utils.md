# PRD: ShareGPT Dataset Utils

## Goal

Maintain this submodule as a self-contained ShareGPT dataset utility package
with a small framework-agnostic core and explicit runtime adapters.

## Current Problems

- Root package exports optional verl adapters through a broad exception guard,
  which can hide real adapter bugs as missing optional dependencies.
- JSONL slice specs such as `file.jsonl@[0:512]` are parsed correctly, but the
  current loader reads the entire file before slicing.
- JSON backend selection prints during import, which is noisy in distributed
  training workers.
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
